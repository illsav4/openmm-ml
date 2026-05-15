"""
ubiomolfmpotential.py: Implements the UBio-MolFM (E2Former) potential function.

UBio-MolFM is a foundation MLIP whose backbone is E2Former-V2, trained on
UBio-Mol26 + OMol25 (https://github.com/IQuestLab/UBio-MolFM). It exposes an
ASE calculator (``E2FormerCalculator``) that loads a Hydra YAML config and a
PyTorch checkpoint. This module wraps that calculator inside an OpenMM
``PythonForce`` so it can be used like any other MLPotential backend:

    >>> potential = MLPotential('ubio-molfm')                    # auto-download
    >>> potential = MLPotential('ubio-molfm',
    ...                         checkpoint_path='ckpt.pt',
    ...                         config_path='./configs',
    ...                         config_name='config_molfm.yaml',
    ...                         head_name='omol25')              # manual

Energy is returned in kJ/mol and forces in kJ/mol/nm (OpenMM conventions),
converted from ASE's eV / eV/Å.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
import openmm
from openmm import unit
from typing import Any, Dict, Iterable, Optional, Tuple
from functools import partial
import os
import threading
import numpy as np


# Default HuggingFace repo holding the pretrained checkpoint + config.
DEFAULT_HF_REPO_ID = "IQuestLab/IQuest-UBio-MolFM-V1"
# Default to None so the upstream resolver auto-picks the single yaml file
# in the config directory. The HF snapshot ships ``config.yaml`` while the
# upstream README example uses ``config_molfm.yaml``; ``None`` works for both.
DEFAULT_CONFIG_NAME = None
DEFAULT_HEAD_NAME = "omol25"


# Process-wide calculator cache. ``E2FormerModelInterface`` does a full
# ``torch.load(checkpoint_path)`` + ``.to(device)`` on every construction, so
# multiple ``createSystem`` / ``createMixedSystem`` calls (one per component
# in a typical complex/protein/peptide scoring run) otherwise reload the
# 277 MB checkpoint each time. Keying on the *resolved* constructor args
# means callers that don't pass an explicit checkpoint_path still hit the
# cache after the first HuggingFace download.
#
# Thread-safety: the lock protects build-or-fetch only. The underlying
# calculator is not safe for concurrent ``calculate()`` calls (it mutates
# ``self.atoms`` between super().calculate and predict()) — peptide scoring
# uses process-based parallelism (MPS / spawn) by default, so each worker
# has its own cache and there is no contention. Callers using thread-based
# parallelism must serialize predict() externally.
_CALCULATOR_CACHE: Dict[Tuple[Any, ...], Any] = {}
_CALCULATOR_CACHE_LOCK = threading.Lock()

# Second-tier cache for the HuggingFace path resolution. ``snapshot_download``
# is fast on a cache hit (filesystem walk + etag check) but not free. Keying
# on (hf_repo_id, hf_revision) lets repeat ``addForces`` calls skip the call
# entirely once paths have been resolved once in this process.
_HF_RESOLUTION_CACHE: Dict[Tuple[str, Optional[str]], Tuple[str, str]] = {}
_HF_RESOLUTION_LOCK = threading.Lock()


def clear_ubiomolfm_calculator_cache() -> None:
    """Drop all cached E2FormerCalculator instances and HF path resolutions.

    Mostly for tests / interactive use. After this returns, the next
    ``addForces`` call will re-run ``snapshot_download`` (network-free on
    a HF cache hit) and re-load the checkpoint from disk to the device.
    """
    with _CALCULATOR_CACHE_LOCK:
        _CALCULATOR_CACHE.clear()
    with _HF_RESOLUTION_LOCK:
        _HF_RESOLUTION_CACHE.clear()


def _resolve_hf_paths(
    hf_repo_id: str, hf_revision: Optional[str]
) -> Tuple[str, str]:
    """Return (checkpoint_path, snapshot_dir) for ``hf_repo_id`` at
    ``hf_revision``. Uses an in-process cache so repeat calls skip the
    ``snapshot_download`` etag walk.

    The lock is held across the whole resolution so concurrent callers
    don't both call ``snapshot_download`` on the first miss — the second
    thread waits for the first to populate the cache, then returns
    immediately. ``snapshot_download`` is fast on a HF cache hit but not
    free (filesystem walk + etag check), and running it once per thread
    is a noticeable footgun under thread parallelism.
    """
    key = (hf_repo_id, hf_revision)
    with _HF_RESOLUTION_LOCK:
        cached = _HF_RESOLUTION_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                f"checkpoint_path / config_path were not given and "
                f"huggingface_hub is not installed ({e}). "
                "Either install huggingface_hub (`pip install huggingface_hub`) "
                "or pass explicit paths."
            )
        snapshot_dir = snapshot_download(
            repo_id=hf_repo_id,
            revision=hf_revision,
            allow_patterns=["*.pt", "*.yaml", "*.yml", "*.json"],
        )
        # Prefer a *.pt at the top level; fall back to any *.pt.
        pt_files = sorted(
            os.path.join(snapshot_dir, f)
            for f in os.listdir(snapshot_dir)
            if f.endswith(".pt")
        )
        if not pt_files:
            for root, _dirs, files in os.walk(snapshot_dir):
                for f in files:
                    if f.endswith(".pt"):
                        pt_files.append(os.path.join(root, f))
        if not pt_files:
            raise FileNotFoundError(
                f"No .pt checkpoint found in HuggingFace snapshot at {snapshot_dir}"
            )
        resolved = (pt_files[0], snapshot_dir)
        _HF_RESOLUTION_CACHE[key] = resolved
        return resolved


def _get_or_build_calculator(
    *,
    checkpoint_path: str,
    config_path: Optional[str],
    config_name: Optional[str],
    head_name: str,
    device: str,
    use_tf32: bool,
    use_compile: bool,
    use_faiss: bool,
    force_use_aperiodic: bool,
    add_ref_energy: bool,
):
    """Return a cached E2FormerCalculator for these args, or build & cache one."""
    from molfm.interface.ase.calculator.e2former_calculator import E2FormerCalculator

    key = (
        os.path.abspath(checkpoint_path) if checkpoint_path else None,
        os.path.abspath(config_path) if config_path else None,
        config_name,
        head_name,
        str(device),
        bool(use_tf32),
        bool(use_compile),
        bool(use_faiss),
        bool(force_use_aperiodic),
        bool(add_ref_energy),
    )
    with _CALCULATOR_CACHE_LOCK:
        cached = _CALCULATOR_CACHE.get(key)
        if cached is not None:
            return cached
        calc = E2FormerCalculator(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            config_name=config_name,
            head_name=head_name,
            device=device,
            use_tf32=use_tf32,
            use_compile=use_compile,
            use_faiss=use_faiss,
            force_use_aperiodic=force_use_aperiodic,
            add_ref_energy=add_ref_energy,
            auto_setup=True,
        )
        # Per-calculator lock for the predict() path. ASE's Calculator base
        # mutates self.atoms / self.results between super().calculate() and
        # predict(), so concurrent calculate() calls on the same instance
        # corrupt each other's state. _computeUBioMolFM acquires this lock
        # around the get_potential_energy / get_forces block. Single-threaded
        # callers (process/MPS) acquire and release with no contention.
        calc._ubiomolfm_predict_lock = threading.Lock()
        _CALCULATOR_CACHE[key] = calc
        return calc


class UBioMolFMPotentialImplFactory(MLPotentialImplFactory):
    """Factory that creates UBioMolFMPotentialImpl objects."""

    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return UBioMolFMPotentialImpl(name)


class UBioMolFMPotentialImpl(MLPotentialImpl):
    """MLPotentialImpl that wraps UBio-MolFM's ``E2FormerCalculator``.

    The implementation builds an ASE calculator under the hood and attaches it
    to an OpenMM ``PythonForce`` (same mechanism as :mod:`asepotential`).

    Construct via :class:`~openmmml.mlpotential.MLPotential`:

    >>> potential = MLPotential('ubio-molfm')

    Recognized kwargs (passed to ``createSystem`` / ``createMixedSystem``):

    * ``checkpoint_path`` (str, default ``None``) — path to the ``.pt`` weights.
      If ``None``, weights are auto-downloaded from HuggingFace.
    * ``config_path`` (str, default ``None``) — directory holding the Hydra
      config YAML. If ``None``, the auto-downloaded snapshot dir is used.
    * ``config_name`` (str, default ``None``) — config filename. When ``None``,
      the upstream resolver auto-picks the single ``*.yaml`` in ``config_path``
      (works for both the HF ``config.yaml`` and a local ``config_molfm.yaml``).
    * ``head_name`` (str, default ``'omol25'``) — task head identifier used to
      pick the reference-energy table.
    * ``device`` (str, default ``'cuda:0'`` if available else ``'cpu'``).
    * ``use_tf32`` (bool, default ``True``).
    * ``use_compile`` (bool, default ``False``).
    * ``use_faiss`` (bool, default ``True``).
    * ``force_use_aperiodic`` (bool, default ``False``) — force non-PBC mode.
    * ``add_ref_energy`` (bool, default ``True``) — add per-atom reference energies.
    * ``hf_repo_id`` (str, default ``'IQuestLab/IQuest-UBio-MolFM-V1'``).
    * ``hf_revision`` (str, default ``None``) — pin a HuggingFace revision.
    * ``charge`` (int, default ``0``) — total charge, forwarded to the calculator
      via ``aseAtoms.set_initial_charges``.
    """

    def __init__(self, name):
        self.name = name

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  checkpoint_path: Optional[str] = None,
                  config_path: Optional[str] = None,
                  config_name: Optional[str] = DEFAULT_CONFIG_NAME,
                  head_name: str = DEFAULT_HEAD_NAME,
                  device: Optional[str] = None,
                  use_tf32: bool = True,
                  use_compile: bool = False,
                  use_faiss: bool = True,
                  force_use_aperiodic: bool = False,
                  add_ref_energy: bool = True,
                  hf_repo_id: str = DEFAULT_HF_REPO_ID,
                  hf_revision: Optional[str] = None,
                  charge: int = 0,
                  **args):
        try:
            import ase
        except ImportError as e:
            raise ImportError(
                f"Failed to import ASE with error: {e}. "
                "Install it as described at https://ase-lib.org/install.html."
            )
        try:
            from molfm.interface.ase.calculator.e2former_calculator import E2FormerCalculator
        except ImportError as e:
            raise ImportError(
                f"Failed to import UBio-MolFM (molfm package) with error: {e}. "
                "Install from https://github.com/IQuestLab/UBio-MolFM."
            )

        # Resolve the device (default to CUDA when available).
        if device is None:
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # Auto-download checkpoint + config from HuggingFace if either was
        # not supplied. The snapshot is cached by huggingface_hub on disk;
        # we additionally cache the resolved (.pt, snapshot_dir) tuple in
        # this process so repeat calls skip the snapshot_download call too.
        if checkpoint_path is None or config_path is None:
            resolved_pt, resolved_dir = _resolve_hf_paths(hf_repo_id, hf_revision)
            if checkpoint_path is None:
                checkpoint_path = resolved_pt
            if config_path is None:
                config_path = resolved_dir

        # Build (or fetch from the process-wide cache) the ASE calculator.
        # The first call for a given (checkpoint, config, head, device, …)
        # tuple does the torch.load + .to(device); subsequent calls reuse
        # the same in-memory model.
        calculator = _get_or_build_calculator(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            config_name=config_name,
            head_name=head_name,
            device=device,
            use_tf32=use_tf32,
            use_compile=use_compile,
            use_faiss=use_faiss,
            force_use_aperiodic=force_use_aperiodic,
            add_ref_energy=add_ref_energy,
        )

        if any(atom.element is None for atom in topology.atoms()):
            raise ValueError("All atoms in the Topology must have elements defined.")
        includedAtoms = list(topology.atoms())
        if atoms is None:
            indices = None
        else:
            includedAtoms = [includedAtoms[i] for i in atoms]
            indices = np.array(atoms)

        numbers = [atom.element.atomic_number for atom in includedAtoms]
        periodic = topology.getPeriodicBoxVectors() is not None
        aseAtoms = ase.Atoms(numbers=numbers, pbc=periodic, calculator=calculator)

        # Convey total charge to the calculator via per-atom initial charges
        # (E2FormerCalculator's ``ase_atom_to_dict`` does ``sum(get_initial_charges())``).
        if charge != 0 and len(aseAtoms) > 0:
            n = len(aseAtoms)
            aseAtoms.set_initial_charges([charge / n] * n)

        # Create the PythonForce and add it to the System.
        compute = partial(_computeUBioMolFM, atoms=aseAtoms, indices=indices)
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)


def _computeUBioMolFM(state, atoms, indices):
    """PythonForce callback. Returns (energy_kJ_per_mol, forces_kJ_per_mol_per_nm).

    Acquires the calculator's predict lock so concurrent thread-mode callers
    serialize the ASE ``calculate()`` step (which mutates the shared
    calculator's ``self.atoms`` / ``self.results``). Single-threaded
    callers — process / MPS backends — acquire and release with no contention.
    """
    import ase.units
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    numAtoms = positions.shape[0]
    if indices is not None:
        positions = positions[indices]
    # Fall back to a no-op context if the lock attr isn't there (e.g. a
    # user-constructed calculator passed by hand) so we never crash.
    lock = getattr(atoms.calc, "_ubiomolfm_predict_lock", None)
    if lock is None:
        import contextlib
        lock = contextlib.nullcontext()
    with lock:
        atoms.set_positions(positions)
        if any(atoms.get_pbc()):
            atoms.set_cell(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom))
        energy = atoms.get_potential_energy(apply_constraint=False)
        forces = atoms.get_forces(apply_constraint=False)
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=np.float32)
        f[indices] = forces
        forces = f
    # eV → kJ/mol; eV/Å → kJ/mol/nm (×10 for Å→nm).
    return (energy / (ase.units.kJ / ase.units.mol),
            forces * 10 / (ase.units.kJ / ase.units.mol))
