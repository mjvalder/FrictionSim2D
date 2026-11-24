"""Structure library for heterostructures: matched and twisted supercells, stack assembly.

Defines no builder class. It generates geometry for the hetero builders
(``afm/hetero.py``, ``sheetonsheet/hetero.py``, ``sheetonsheet/flake_slide.py``).

The match is computed on the cells that are tiled
----------------------------------------------------
``make_supercell`` tiles a cell as ``new = matrix @ old``, so a coincidence match
is only meaningful for the exact cell it was derived from. Applying a match found
on a CIF primitive ``P`` to ``build_monolayer``'s orthogonalized ``T @ P`` gives
``matrix @ T @ P``, which is not commensurate for any anisotropic or sheared
match. :func:`build_matched_supercells` therefore reads each monolayer's own cell
from disk and matches on that.

Symmetric mean-cell strain
--------------------------
Both supercells are strained onto ``mean_cell = (sc_a + sc_b) / 2``; neither
material is an unstrained reference. Each strain must be ``<= strain_tol``, else
the build fails loud.

The strain is applied in the co-oriented frame
----------------------------------------------
Writing each supercell on its own re-canonicalizes it (a1 along x) with a
different rotation per material, so a per-file ``change_box`` remap (which cannot
rotate) would read that rotation as spurious shear. Both supercells are instead
strained with an affine ``set_cell(scale_atoms=True)`` while still co-oriented,
then written. They then share one in-plane cell and one canonicalizing rotation,
so their on-disk boxes coincide and the relative A/B registry survives.
Per-atom metadata (charges, masses, atom_style) round-trips through
:func:`~frictionsim2d.interfaces.lammps_monolayer.write_supercell`.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...core.config import (
    FlakeConfig, GlobalSettings, SheetConfig, SheetOnSheetSimulationConfig)
from ...lattice.matching import (
    MatchResult,
    max_strain,
    relative_rotation_strain,
    find_coincidence_lattice,
    find_commensurate_twist,
    scan_registry_index,
)
from ...core.mlip import is_mlip, validate_mlip_cell
from ...potentials.manager import PotentialManager
from ...interfaces.jinja import render_template
from .components import (
    apply_sheet_edge_types, build_monolayer, calculate_layer_shifts, carve_flake)
from ..utils.interlayer import interlayer_style, set_interlayer_from_config
from ...interfaces.ase import read_lammps_data
from ...interfaces.lammps_monolayer import (
    detect_atom_style, monolayer_to_atoms, read_monolayer, write_supercell,
)

if TYPE_CHECKING:                      # ase.io costs ~270 ms; annotations only
    from ase import Atoms

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)

@dataclass
class MatchedStack:
    """Two per-material coincidence supercells sharing one in-plane mean cell.

    Both materials are strained symmetrically onto the shared
    ``mean_cell = (sc_a + sc_b) / 2``; there is no unstrained reference material.

    Attributes:
        supercell_a: Path to material A's output supercell data file.
        supercell_b: Path to material B's output supercell data file.
        strain_a: Max absolute principal strain applied to material A to reach
            the shared mean cell (``<= strain_tol``).
        strain_b: Max absolute principal strain applied to material B to reach
            the shared mean cell (``<= strain_tol``).
        mean_cell: The shared 2x2 in-plane box (Å) both supercells were strained
            onto, in LAMMPS-canonical form (first vector along +x).
        match: The coincidence-lattice match found (internally) between the two
            monolayers' actual in-plane cells.
    """
    supercell_a: Path
    supercell_b: Path
    strain_a: float
    strain_b: float
    mean_cell: np.ndarray
    match: MatchResult = field(default=None)  # type: ignore[assignment]
    # Twist path only (None otherwise): achieved relative twist (deg) and moiré
    # periodicity (Å). Set by :func:`build_twisted_supercells`.
    twist_angle_achieved: Optional[float] = None
    moire_period: Optional[float] = None


@dataclass
class HeteroPotentials:
    """Registered potential spine for a stacked heterostructure.

    Every material is its own :class:`PotentialManager` component with a
    disjoint, contiguous block of global LAMMPS atom-type IDs, its own
    many-body potential inside a single ``pair_style hybrid``, and UFF-mixed
    ``lj/cut`` cross-terms for every A-B element pair at the interface -- all
    emitted into ``system.in.settings``.

    ``type_ids`` / ``type_offsets`` / ``component_names`` always describe the
    REAL material types only. When the manager was built with
    ``reserve_virtual_type=True`` (for the ``virtual_atom`` drive), an extra
    non-interacting virtual type is present in ``pm.types`` but is NOT a
    material, so it is excluded from those fields -- i.e.
    ``len(pm.types) == (sum of len(ids) for ids in type_ids) + 1`` in that case,
    and ``== that sum`` otherwise.

    Attributes:
        pm: The configured :class:`PotentialManager` (type registry +
            interactions already written). ``len(pm.types)`` includes the
            reserved virtual type when one was requested.
        settings_path: Path to the written ``system.in.settings``.
        component_names: Component name per REAL material, in stacking order
            (``['sheet_1', 'sheet_2', ...]``); never includes the virtual type.
        type_ids: Per-material list of that material's global atom-type IDs, in
            stacking order. Read straight from the PM type registry
            (``pm.types.ids_by_component``); never hand-assigned. The blocks are
            disjoint and together cover ``1..len(pm.types)`` with no gaps.
        type_offsets: Per-material starting type offset (0-based) = the number
            of types registered before that material = ``type_ids[i][0] - 1``.
            Fed to ``read_data ... add append offset <k>`` so the
            assembled data file's atom types line up with this PM exactly.
    """
    pm: PotentialManager
    settings_path: Path
    component_names: List[str]
    type_ids: List[List[int]]
    type_offsets: List[int]


def register_sheet_block(
    pm: PotentialManager,
    sheets: List[SheetConfig],
    regions: Optional[List[Optional[str]]] = None,
) -> List[str]:
    """Register the per-material sheet stack onto ``pm`` as ``sheet_1``/``sheet_2``/...

    Appends one component ``sheet_{i+1}`` per material (in order) onto whatever
    ``pm`` it is given, returning the component names. This is the single source
    of truth for hetero sheet numbering: it is called by
    :func:`register_hetero_potentials` (which numbers the assembled data file)
    AND by the unified AFM PotentialManager, so the sim-PM sheet block and the
    data-file sheet block cannot diverge (with or without ``regions``). The
    ``regions`` argument (default None; the finite path passes ``[None, 'edge']``)
    is applied identically on both sides through this one call.

    Args:
        pm: The PotentialManager to append onto (at its current type offset).
        sheets: Per-material sheet configs.
        regions: Optional per-type region split forwarded to
            :meth:`PotentialManager.register_component`.

    Returns:
        The registered component names (``['sheet_1', 'sheet_2', ...]``).
    """
    component_names: List[str] = []
    for i, sheet in enumerate(sheets):
        name = f"sheet_{i + 1}"
        pm.register_component(name, sheet, n_layers=1, regions=regions)
        component_names.append(name)
    return component_names


def register_hetero_potentials(
    sheets: List[SheetConfig],
    settings: GlobalSettings,
    workdir: PathLike,
    reserve_virtual_type: bool = False,
    regions: Optional[List[Optional[str]]] = None,
    potentials_prefix: Optional[str] = None,
    interlayer_general: Any = None,
    interlayer_terms: bool = True,
) -> HeteroPotentials:
    """Register the stacked materials as PotentialManager components.

    This is the component/atom-type spine of the heterostructure: it is
    geometry-independent (it needs only the per-material configs, not any
    assembled data file), so it runs *before* stack assembly -- assembly then
    uses the returned potential to minimize the interface spacing, and reads
    ``type_offsets`` to keep the assembled data file's atom types aligned with
    this PM.

    Each material ``sheets[i]`` is registered as a single component
    ``sheet_{i+1}`` (one component sharing types across all its layers -- for
    SW materials the short many-body cutoff prevents spurious interlayer terms,
    so per-layer type expansion is neither needed nor wanted). Every component
    gets its own many-body ``add_self_interaction`` inside a ``pair_style
    hybrid``, and every distinct material pair gets a UFF-mixed
    ``add_cross_interaction`` (``lj/cut`` for each A-B element pair). The
    settings are written to ``<workdir>/lammps/system.in.settings`` and the
    potential files are staged under ``<workdir>/provenance/potentials``.

    Args:
        sheets: Per-material sheet configs (>= 2 materials).
        settings: Global simulation settings.
        workdir: Directory to write ``system.in.settings`` and the staged
            provenance potentials into (created if missing).
        reserve_virtual_type: Register one extra non-interacting ``virtual`` type
            (for the ``virtual_atom`` drive) after the material components but
            BEFORE the many-body self-interactions, the same order the
            homogeneous path uses, so every ``pair_coeff * * sw ...`` list is
            sized to the full box type count the first time. The virtual type is
            not a material: ``type_ids`` / ``type_offsets`` / ``component_names``
            exclude it; only ``len(hp.pm.types)`` counts it.
        regions: Optional per-type region split forwarded through
            :func:`register_sheet_block`. The finite path passes
            ``[None, 'edge']``, doubling every type into an interleaved core
            (odd) / edge (even) pair, the layout
            :func:`~frictionsim2d.builders.construction.components.apply_sheet_edge_types`
            writes. Default ``None``.
        potentials_prefix: Prefix the ``pair_coeff`` lines use to reach the
            staged potential files (always written to
            ``<workdir>/provenance/potentials``); relative to the directory
            LAMMPS runs from. Default ``None`` is the run-dir-relative
            ``provenance/potentials``.
        interlayer_general: The run's ``[general]`` config; its ``interlayer_style``
            acts between every 2D layer. ``None`` or ``lj`` is the plain spine.
            Any other style is set before registration, so under ilp/tmd each
            material's chalcogen sub-layers get their own types, the split
            :func:`build_monolayer` applies to its data file.
        interlayer_terms: Emit the style's interlayer term. ``False`` keeps the
            non-``lj`` type layout but the ``lj`` interactions, for the assembly
            runs (``lmp_serial`` may lack the INTERLAYER package and no
            molecule IDs are set). Under ``ilp`` / ``kc`` with terms the
            cross-material ``lj/cut`` is dropped: one ILP/KC ``pair_coeff``
            spans every material.

    Returns:
        :class:`HeteroPotentials` with the configured manager, the written
        settings path and the per-material type blocks / offsets, read from the
        PM type registry.

    Raises:
        ValueError: If fewer than two materials are supplied (a heterostructure
            needs at least two).
    """
    if len(sheets) < 2:
        raise ValueError(
            f"register_hetero_potentials needs >= 2 materials to build a "
            f"heterostructure interface, got {len(sheets)}."
        )

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # deepmd rejects NULL type maps (pair_deepmd.cpp:816), so a partial-cell MLIP
    # is rejected. A hetero cell is multi-component: one instance per component
    # is a NULL-map topology whichever framework it names, so full-cell MLIP is
    # rejected as well.
    validate_mlip_cell(
        sheets[0],
        components=[(f"sheet_{i + 1}", sheet)
                    for i, sheet in enumerate(sheets[1:], start=1)],
        constraint_mode=settings.simulation.constraint_mode,
    )

    # Mirror the AFM / sheet-on-sheet potential path exactly: stage the real
    # potential files under the run dir's provenance folder and reference them
    # by the caller's prefix in the pair_coeff lines.
    pm = PotentialManager(
        settings,
        potentials_dir=workdir / "provenance" / "potentials",
        potentials_prefix=(
            potentials_prefix if potentials_prefix is not None
            else str(Path("provenance") / "potentials")
        ),
    )

    # 1. One component per material -> disjoint, contiguous global type blocks
    #    (the PM's TypeRegistry assigns the IDs; we never hand-roll them).
    #    Single-sourced through register_sheet_block so the unified AFM PM
    #    numbers sheet_1/sheet_2 identically to this data-file spine. When
    #    ``regions`` is given (finite/edge path, [None, 'edge']), every material's
    #    types double into an interleaved core/edge pair through that one call.
    style = interlayer_style(interlayer_general)
    if style != 'lj':
        # Before registration: ilp/tmd's chalcogen split is decided there.
        set_interlayer_from_config(
            pm, interlayer_general,
            {f"sheet_{i + 1}": sheet for i, sheet in enumerate(sheets)})
    component_names = register_sheet_block(pm, sheets, regions=regions)

    # 1b. Optionally reserve the dedicated non-interacting virtual type for the
    #     virtual_atom drive -- BEFORE the self-interactions, mirroring the
    #     homogeneous drive-atom ordering. Registering it here (not after) means
    #     each many-body null-map below is built over the full box type count, so
    #     ``pair_coeff * * sw ...`` carries the trailing NULL for the virtual type
    #     the first time (no clearing/rebuilding). It stays out of type_ids /
    #     component_names because it is not a material.
    if reserve_virtual_type:
        pm.register_virtual_atom()

    # 2. Each material's own many-body potential (hybrid entry per component).
    for name in component_names:
        pm.add_self_interaction(name)

    # 3. UFF-mixed lj/cut cross-terms for every material pair at the interface
    #    -- replaced by the many-body term under ilp/kc.
    emit_interlayer = style != 'lj' and interlayer_terms
    if not (emit_interlayer and style in ('ilp', 'kc')):
        for i, name_i in enumerate(component_names):
            for j in range(i + 1, len(component_names)):
                pm.add_cross_interaction(name_i, component_names[j])
    if emit_interlayer:
        pm.add_interlayer(component_names)
    if sheets[0].dispersion == 'd3':
        pm.enable_d3()

    settings_path = workdir / "lammps" / "system.in.settings"
    pm.write_file(settings_path)

    # 4. Read the per-material type blocks back from the registry (source of
    #    truth); derive read_data offsets from where each block starts.
    type_ids = [pm.types.ids_by_component(name) for name in component_names]
    type_offsets = [ids[0] - 1 for ids in type_ids]

    return HeteroPotentials(
        pm=pm,
        settings_path=settings_path,
        component_names=component_names,
        type_ids=type_ids,
        type_offsets=type_offsets,
    )


def _embed_3x3(matrix_2x2: np.ndarray) -> np.ndarray:
    """Embed a 2x2 integer in-plane supercell matrix into a 3x3 matrix.

    ``ase.build.make_supercell`` requires a full 3x3 integer matrix; the
    z-row/column is left as identity so the out-of-plane lattice vector
    (and periodicity) is left untouched.
    """
    p = np.eye(3, dtype=int)
    p[:2, :2] = np.asarray(matrix_2x2, dtype=int)
    return p


#: Per-atom tolerance (A) for accepting a translation as a lattice repeat. The
#: residual of a true repeat is ~1e-7 A (data-file write precision), and the
#: closest same-species in-plane separation is >= 1.8 A. Too tight only yields a
#: superlattice of the true repeat; too loose would over-divide the energy.
_REPEAT_TOL = 0.05
#: Shortest in-plane repeat (A) proposed; no 2D lattice has a shorter period.
_MIN_REPEAT = 1.0
#: Cap on repeats searched per axis; exceeding it yields a superlattice, never
#: a too-small cell.
_MAX_REPEATS = 32


def _is_inplane_translation(
    pos: np.ndarray,
    species: np.ndarray,
    lx: float,
    ly: float,
    axis: int,
    delta: float,
    tol: float,
) -> bool:
    """Does translating every atom by ``delta`` along ``axis`` map the structure onto itself?

    Species must match and the out-of-plane coordinate is compared as-is (z is
    not periodic), so a shift can never map one layer onto another. x/y use the
    minimum image convention, so no pre-wrapping is needed.
    """
    shifted = pos.copy()
    shifted[:, axis] += delta
    for s in np.unique(species):
        mask = species == s
        a, b = shifted[mask], pos[mask]
        for start in range(0, len(a), 256):          # bound peak memory
            d = a[start:start + 256, None, :] - b[None, :, :]
            d[:, :, 0] -= lx * np.round(d[:, :, 0] / lx)
            d[:, :, 1] -= ly * np.round(d[:, :, 1] / ly)
            nearest = np.sqrt((d * d).sum(axis=2)).min(axis=1)
            if nearest.size and nearest.max() > tol:
                return False
    return True


def detect_inplane_repeat(
    data_path: PathLike,
) -> Tuple[Tuple[float, float], Tuple[int, int]]:
    """Find an assembled stack's MINIMAL axis-aligned in-plane repeat unit.

    A coincidence match is not in general a minimal cell: ``build_monolayer``
    tiles each material to the configured x/y footprint before matching, so a
    near-lattice-matched pair yields a box holding n x m identical interface
    repeat units. The PES scan period must be the repeat unit, and the per-cell
    energy normalisation must divide by the number of repeat units, or the
    corrugation would scale with the footprint.

    The repeat is measured off the assembled structure: per axis, the largest
    ``n`` such that translating by ``L/n`` maps every atom onto a same-species
    atom at the same height. This holds however the match was built.

    Only axis-aligned candidates are tested: the scan grid steps along +x/+y and
    the homogeneous PES path normalises per orthogonalised unit cell, so a
    centred-rectangular glide is intentionally not counted. Assumes an
    orthogonal box (both PES callers refuse a triclinic cell); xy tilt is
    ignored.

    Args:
        data_path: Assembled LAMMPS data file (e.g. ``hetero.data``).

    Returns:
        ``((period_x, period_y), (nx, ny))`` with ``nx * period_x == lx`` and
        ``ny * period_y == ly``. Falls back to ``((lx, ly), (1, 1))`` when no
        sub-box translation exists (an incommensurate or twisted stack).

    Raises:
        ValueError: If the in-plane box is not positive.
    """
    atoms = read_lammps_data(data_path)
    cell = atoms.cell
    lx, ly = float(cell[0, 0]), float(cell[1, 1])
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError(
            f"Non-positive in-plane box in {data_path}: {lx} x {ly} A."
        )

    pos = np.asarray(atoms.positions, dtype=float)
    species = np.asarray(atoms.numbers)

    def _axis_repeat_count(axis: int, length: float) -> int:
        n_cap = min(_MAX_REPEATS, max(1, int(length // _MIN_REPEAT)))
        for n in range(n_cap, 1, -1):
            if _is_inplane_translation(
                    pos, species, lx, ly, axis, length / n, _REPEAT_TOL):
                return n
        return 1

    nx = _axis_repeat_count(0, lx)
    ny = _axis_repeat_count(1, ly)
    return (lx / nx, ly / ny), (nx, ny)


def _strain_onto_and_write(
    atoms: Atoms,
    target_2x2: np.ndarray,
    out_path: Path,
    mono: Dict[str, object],
) -> np.ndarray:
    """Affinely set the in-plane cell of ``atoms`` to ``target_2x2`` and write it.

    ``set_cell(scale_atoms=True)`` preserves fractional coordinates; the cell's
    z extent is kept. Metadata (atom_style, type count, masses) comes from
    ``mono``. Returns the written in-plane cell, read back from disk.
    """
    lz = np.array(atoms.cell)[2, 2]
    atoms.set_cell(
        np.array([
            [target_2x2[0, 0], target_2x2[0, 1], 0.0],
            [target_2x2[1, 0], target_2x2[1, 1], 0.0],
            [0.0, 0.0, lz],
        ]),
        scale_atoms=True,
    )
    return write_supercell(
        atoms, out_path,
        atom_style=str(mono["atom_style"]),
        n_types=int(mono["n_types"]),
        masses=mono["masses"],  # type: ignore[arg-type]
    )


def build_matched_supercells(
    mono_a_path: PathLike,
    mono_b_path: PathLike,
    workdir: PathLike,
    strain_tol: float = 0.02,
    max_supercell: int = 10,
    area_tol: float = 0.10,
) -> MatchedStack:
    """Build A's and B's coincidence supercells sharing one in-plane mean cell.

    Reads each monolayer's *actual* in-plane cell from disk, finds their
    Zur-McGill coincidence lattice on those exact cells (so the integer match
    matrices are consistent with the cells that get tiled), tiles each material
    with its match matrix via :func:`ase.build.make_supercell`, and strains
    **both** supercells symmetrically onto the shared
    ``mean_cell = (sc_a + sc_b) / 2`` **in the co-oriented frame**
    (an affine ``set_cell(scale_atoms=True)`` remap that preserves fractional
    coordinates and per-atom metadata), so both output files share exactly one
    LAMMPS-canonical in-plane box with each material carrying at most
    ``strain_tol`` of distortion.

    Args:
        mono_a_path: LAMMPS data file for material A's monolayer.
        mono_b_path: LAMMPS data file for material B's monolayer.
        workdir: Directory to write the two output supercell data files into
            (created if missing).
        strain_tol: Max per-material strain (vs the symmetric mean cell) allowed
            both when searching for a coincidence match (forwarded to
            :func:`~frictionsim2d.lattice.matching.find_coincidence_lattice`) and as
            the fail-loud bound on each material's applied mean-cell strain.
        max_supercell: Max supercell repeat along each axis for the match search.
        area_tol: Relative supercell-area agreement tolerance for the search.

    Returns:
        :class:`MatchedStack` with both output supercell paths (ordered A then
        B), the per-material mean-cell strains ``strain_a`` / ``strain_b``, the
        shared ``mean_cell`` box, and the match found.

    Raises:
        ValueError: If no coincidence lattice is found within the given
            tolerances, or if either material's geometric strain onto the shared
            mean cell exceeds the bound (i.e. the pair is not actually
            commensurate at this match -- fail loud rather than silently emit an
            incommensurate "matched" stack).
    """
    # Lazy: ase.build costs ~290-300 ms (measured with python -X importtime);
    # deferred so this module stays cheap to import.
    from ase.build import make_supercell  # pylint: disable=import-outside-toplevel

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mono_a_path = Path(mono_a_path)
    mono_b_path = Path(mono_b_path)

    # 1. Read each monolayer (metadata-preserving) and its ACTUAL in-plane cell.
    mono = {"a": read_monolayer(mono_a_path), "b": read_monolayer(mono_b_path)}
    cell_mono = {label: np.asarray(m["cell"])[:2, :2] for label, m in mono.items()}

    # 2. Find the coincidence match on the very cells that will be tiled.
    match = find_coincidence_lattice(
        cell_mono["a"], cell_mono["b"],
        strain_tol=strain_tol, max_supercell=max_supercell, area_tol=area_tol,
    )
    if match is None:
        raise ValueError(
            f"No coincidence lattice found between '{mono_a_path.name}' and "
            f"'{mono_b_path.name}' within strain_tol={strain_tol:.4%}, "
            f"max_supercell={max_supercell}, area_tol={area_tol:.2%}. The two "
            f"monolayers are not commensurate under these constraints."
        )

    matrices = {"a": match.matrix_a, "b": match.matrix_b}

    # 3. Tile each monolayer into its coincidence supercell (co-oriented frame).
    atoms = {label: monolayer_to_atoms(mono[label]) for label in ("a", "b")}
    sc_atoms = {
        label: make_supercell(atoms[label], _embed_3x3(matrices[label]))
        for label in ("a", "b")
    }
    sc_cell = {label: np.array(sc_atoms[label].cell)[:2, :2] for label in ("a", "b")}

    # 4. Shared symmetric mean cell in the co-oriented (make_supercell) frame.
    #    Both materials strain onto this ONE box -- no unstrained reference.
    mean_co = (sc_cell["a"] + sc_cell["b"]) / 2.0

    # 5. Measure each material's strain onto the mean cell IN THE CO-ORIENTED
    #    FRAME (rotation-free), and fail loud if either exceeds the bound (no
    #    silent incommensurate fallback).
    strains = {label: max_strain(mean_co, sc_cell[label]) for label in ("a", "b")}
    if strains["a"] > strain_tol or strains["b"] > strain_tol:
        raise ValueError(
            f"Cannot build a commensurate heterostructure stack between "
            f"'{mono_a_path.name}' and '{mono_b_path.name}': symmetric mean-cell "
            f"strain exceeds tol -- strain_a={strains['a']:.4%}, "
            f"strain_b={strains['b']:.4%} > {strain_tol:.4%}."
        )

    # 6. Strain BOTH supercells onto the mean cell IN THE CO-ORIENTED FRAME
    #    (set_cell with scale_atoms is a pure affine remap that preserves
    #    fractional coordinates -- like change_box remap -- but keeps A and B in
    #    one shared frame so no per-material canonicalization rotation is baked
    #    in as spurious shear). Since both now carry the identical in-plane cell,
    #    ASE canonicalizes them by the same rotation on write, so their on-disk
    #    boxes coincide exactly; write_supercell preserves per-atom metadata.
    outputs: Dict[str, Path] = {}
    mean_cell: Optional[np.ndarray] = None
    for label in ("a", "b"):
        at = sc_atoms[label]
        out_path = workdir / f"{label}_supercell.lmp"
        written_cell = _strain_onto_and_write(at, mean_co, out_path, mono[label])
        # Both materials write the identical in-plane cell -> identical canonical
        # on-disk box; capture it once as the shared mean cell.
        mean_cell = written_cell if mean_cell is None else mean_cell
        outputs[label] = out_path

    return MatchedStack(
        supercell_a=outputs["a"],
        supercell_b=outputs["b"],
        strain_a=strains["a"],
        strain_b=strains["b"],
        mean_cell=mean_cell,  # type: ignore[arg-type]
        match=match,
    )


def build_twisted_supercells(
    mono_a_path: PathLike,
    mono_b_path: PathLike,
    workdir: PathLike,
    twist_angle: float,
    angle_tol: float = 2.0,
    strain_tol: float = 0.02,
    max_supercell: int = 12,
) -> MatchedStack:
    """Build two layers' commensurate twisted-moiré supercells on one ortho box.

    The twist analogue of :func:`build_matched_supercells`. Where the plain
    path finds the *aligned* coincidence lattice, this finds the commensurate
    moiré **approximant nearest a user-selected twist angle**
    (:func:`~frictionsim2d.lattice.matching.find_commensurate_twist`) and emits both
    layers tiled into the shared **orthorhombic** moiré box (xy tilt ≈ 0 --
    downstream requires an orthogonal box; the search wraps the rhombic moiré in
    its minimal rectangular super-supercell).

    Mechanism (why the twist survives an orthogonal LAMMPS-canonical box). The
    search returns an integer supercell matrix for each layer that tiles that
    layer's monolayer directly into the moiré box; layer B's matrix selects the
    θ-rotated sublattice. Tiling A with ``supercell_a`` and B with
    ``supercell_b`` (via ``make_supercell``) gives two boxes of the same shape
    but rotated by the achieved angle relative to each other; mapping BOTH onto
    the single canonical orthorhombic ``moire_cell`` with an affine
    ``set_cell(scale_atoms=True)`` (which also closes the residual strain)
    realises the relative twist in the atomic positions while emitting one
    shared orthogonal box. The twist is carried entirely by the commensurate
    tiling (``supercell_b``); layer B's monolayer is tiled as read.

    Args:
        mono_a_path: LAMMPS data file for layer A's monolayer.
        mono_b_path: LAMMPS data file for layer B's monolayer (equals A's for a
            same-material moiré).
        workdir: Directory to write the two output supercell data files into.
        twist_angle: Target relative twist (deg).
        angle_tol: Max ``|achieved - target|`` accepted by the search (deg).
        strain_tol: Max residual per-material strain to close the approximant.
        max_supercell: Max moiré supercell repeat along each axis.

    Returns:
        :class:`MatchedStack` with both output
        supercell paths sharing one orthorhombic box, the per-material residual
        strains, ``mean_cell`` = the emitted orthorhombic box, and the twist
        fields ``twist_angle_achieved`` / ``moire_period`` set.

    Raises:
        ValueError: If no commensurate moiré lands within the tolerances /
            budget (:func:`find_commensurate_twist` fails loud, naming the
            nearest achievable angle), or if a per-material strain onto the
            emitted box exceeds ``strain_tol``.
    """
    # Lazy: ase.build costs ~290-300 ms (measured with python -X importtime);
    # deferred so this module stays cheap to import.
    from ase.build import make_supercell  # pylint: disable=import-outside-toplevel

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mono_a_path = Path(mono_a_path)
    mono_b_path = Path(mono_b_path)

    mono_a = read_monolayer(mono_a_path)
    mono_b = read_monolayer(mono_b_path)
    cell_a = np.asarray(mono_a["cell"])[:2, :2]
    cell_b = np.asarray(mono_b["cell"])[:2, :2]

    # 1. Nearest commensurate twisted moiré -> achieved angle + integer supercell
    #    matrices tiling each monolayer into the shared orthorhombic box.
    twist = find_commensurate_twist(
        cell_a, cell_b, target_angle=twist_angle, angle_tol=angle_tol,
        strain_tol=strain_tol, max_supercell=max_supercell,
    )
    achieved = twist.achieved_angle_deg
    ortho = np.asarray(twist.moire_cell, dtype=float)   # canonical, xy tilt 0

    # 2. Tile each monolayer into its moiré supercell with the integer matrices.
    atoms = {
        "a": monolayer_to_atoms(mono_a),
        "b": monolayer_to_atoms(mono_b),
    }
    matrices = {"a": twist.supercell_a, "b": twist.supercell_b}
    sc_atoms = {
        lbl: make_supercell(atoms[lbl], _embed_3x3(matrices[lbl])) for lbl in ("a", "b")
    }

    # 3. Map BOTH supercells onto the single canonical orthorhombic moiré box
    #    (affine remap: closes residual strain AND, because the two tiled boxes
    #    are rotated by the achieved angle relative to each other, realises the
    #    relative twist in the atoms). Both end with the identical in-plane cell,
    #    so ASE canonicalizes them identically -> on-disk boxes coincide exactly.
    strains: Dict[str, float] = {}
    outputs: Dict[str, Path] = {}
    for lbl, mono in (("a", mono_a), ("b", mono_b)):
        at = sc_atoms[lbl]
        sc_cell = np.array(at.cell)[:2, :2]
        # Residual strain to close this layer onto the ortho box (rotation-
        # invariant: the tiled box and the ortho box differ by a rigid rotation
        # plus this strain).
        strains[lbl] = relative_rotation_strain(ortho, sc_cell)[1]
        if strains[lbl] > strain_tol + 1e-9:
            raise ValueError(
                f"build_twisted_supercells: layer '{lbl}' residual strain "
                f"{strains[lbl]:.4%} onto the moiré box exceeds strain_tol="
                f"{strain_tol:.4%} (achieved twist {achieved:.3f} deg). Widen "
                f"twist_strain_tol or pick a cleaner angle."
            )
        out_path = workdir / f"{lbl}_twist_supercell.lmp"
        _strain_onto_and_write(at, ortho, out_path, mono)
        outputs[lbl] = out_path

    return MatchedStack(
        supercell_a=outputs["a"],
        supercell_b=outputs["b"],
        strain_a=strains["a"],
        strain_b=strains["b"],
        mean_cell=ortho,
        match=None,  # type: ignore[arg-type]
        twist_angle_achieved=float(achieved),
        moire_period=float(twist.moire_period),
    )


# =============================================================================
# Stack assembly: two matched supercells -> one periodic hetero cell
# =============================================================================


@dataclass
class HeteroStackLayer:
    """One placed layer in the assembled heterostructure.

    Attributes:
        material_index: 0-based material index (0 = supercell_a, 1 = supercell_b).
        layer_index: 0-based layer within that material.
        z: z-shift (Å) applied to this layer's supercell when read into the box.
        xy_offset: In-plane displacement (Å) applied to this layer = the
            material's cross-material RI anchor + this layer's intra-material
            AA/AB shift. The reference material's anchor is (0, 0); the
            non-reference material's anchor is the surviving RI offset.
        group: LAMMPS group name assigned to this layer's atoms.
        type_offset: read_data atom-type offset used for this layer (from the
            PotentialManager, so assembled global types match the spine).
        source: Path to the per-material supercell data file this layer was
            read from.
    """
    material_index: int
    layer_index: int
    z: float
    xy_offset: Tuple[float, float]
    group: str
    type_offset: int
    source: Path


@dataclass
class HeteroStack:
    """Result of assembling the matched supercells into one periodic cell.

    Attributes:
        data_path: Path to the single written ``hetero.data`` (all layers of
            both materials in one shared in-plane periodic box).
        settings_path: The real potential settings (carried through from
            :class:`HeteroPotentials`) that describe the written types/masses.
        layers: Per-layer bookkeeping (material/layer index, z, xy offset,
            group, type offset, source) for a later sim builder.
        ri_offset: The cross-material Registry-Index offset (Å) actually applied
            to the displaced material's layers (one-sided; NOT
            centroid-cancelled). Non-zero for AB / off-registry stackings.
        interface_spacing: Cross-material interlayer COM z-gap (Å) found by real
            minimization with the registered potential.
        total_types: Number of global atom types in the assembled cell
            (== len(pm.types)).
    """
    data_path: Path
    settings_path: Path
    layers: List[HeteroStackLayer]
    ri_offset: np.ndarray
    interface_spacing: float
    total_types: int
    # Twist path only (None otherwise): achieved relative twist (deg) + moiré
    # periodicity (Å), logged into the run manifest by the slide builder.
    twist_angle_achieved: Optional[float] = None
    moire_period: Optional[float] = None
    #: The manager behind the last ``settings_path`` write (set by
    #: :func:`build_hetero_structure`): its ``needs_molecule_ids`` and
    #: ``interlayer_substyle`` drive the run script's interlayer keys.
    pm: Optional[PotentialManager] = None


def run_lammps(workdir: Path, script_path: Path) -> str:
    """Run ``lmp_serial`` on a script with ``cwd = workdir`` and return stdout.

    Running from ``workdir`` is mandatory: the settings file's ``pair_coeff``
    lines reference the staged potentials by the run-dir-relative prefix
    ``provenance/potentials/...``, so those paths only resolve when LAMMPS'
    working directory is the run dir.

    Raises:
        RuntimeError: If LAMMPS exits non-zero (with a tail of stdout/stderr).
    """
    result = subprocess.run(
        ["lmp_serial", "-in", str(script_path), "-log", "none"],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,  # returncode is inspected explicitly below
    )
    if result.returncode != 0:
        tail_out = "\n".join(result.stdout.splitlines()[-30:])
        tail_err = "\n".join(result.stderr.splitlines()[-30:])
        raise RuntimeError(
            f"LAMMPS run failed (exit {result.returncode}) for "
            f"'{script_path.name}' in {workdir}.\n"
            f"--- stdout (tail) ---\n{tail_out}\n"
            f"--- stderr (tail) ---\n{tail_err}"
        )
    return result.stdout


def box_region_cmd(lx: float, ly: float, xy: float, zlo: float, zhi: float) -> str:
    """LAMMPS ``region box ...`` command reproducing the shared in-plane box.

    Uses a ``prism`` region when the shared box is sheared (non-zero xy tilt),
    else a plain ``block`` region.
    """
    if abs(xy) > 1e-9:
        return (
            f"region box prism 0.0 {lx:.6f} 0.0 {ly:.6f} "
            f"{zlo:.6f} {zhi:.6f} {xy:.6f} 0.0 0.0"
        )
    return f"region box block 0.0 {lx:.6f} 0.0 {ly:.6f} {zlo:.6f} {zhi:.6f}"


def _minimize_interface_gap(
    *,
    workdir: Path,
    settings: GlobalSettings,
    settings_path: Path,
    atom_style: str,
    total_types: int,
    masses_block: str,
    lx: float,
    ly: float,
    xy: float,
    bot_path: Path,
    bot_off: int,
    bot_zext: Tuple[float, float],
    top_path: Path,
    top_off: int,
    top_zext: Tuple[float, float],
    top_shift_xy: np.ndarray,
    initial_guess: float,
) -> float:
    """Find one interface's interlayer COM z-gap by REAL minimization.

    Builds a 2-layer trial (bottom layer at z=0, top layer at ``initial_guess``
    with the interface's relative in-plane offset applied), includes the real
    registered potential, minimizes at fixed box, and reads the COM z-gap
    ``xcm(g2,z) - xcm(g1,z)`` -- mirroring the ``stack_multilayer_sheet`` idiom.
    """
    zlo = min(bot_zext[0], top_zext[0]) - 10.0
    zhi = max(bot_zext[1], top_zext[1] + initial_guess) + 10.0
    box_region = box_region_cmd(lx, ly, xy, zlo, zhi)

    dx, dy = float(top_shift_xy[0]), float(top_shift_xy[1])
    disp = ""
    if abs(dx) > 1e-12 or abs(dy) > 1e-12:
        disp = f"displace_atoms  g2 move {dx:.6f} {dy:.6f} 0.0 units box\n"

    script = f"""clear
units           metal
atom_style      {atom_style}
boundary        p p p

{box_region}
create_box      {total_types} box

read_data       {bot_path} add append shift 0.0 0.0 0.0 group g1 offset {bot_off} 0 0 0 0 nocoeff
read_data       {top_path} add append shift 0.0 0.0 {initial_guess:.6f} group g2 offset {top_off} 0 0 0 0 nocoeff
{disp}{masses_block}
include         {settings_path}
run             0

min_style       {settings.simulation.min_style}
{settings.simulation.minimization_command}
variable        z1 equal xcm(g1,z)
variable        z2 equal xcm(g2,z)
variable        lat_c equal v_z2-v_z1
print           "RESULT_LAT_C ${{lat_c}}"
"""
    script_path = workdir / f"min_interface_{bot_off}_{top_off}.lmp"
    script_path.write_text(script, encoding="utf-8")
    out = run_lammps(workdir, script_path)

    value: Optional[float] = None
    for line in out.splitlines():
        if line.startswith("RESULT_LAT_C"):
            value = float(line.split()[1])
    if value is None:
        raise RuntimeError(
            f"Could not parse interface spacing (RESULT_LAT_C) from minimize "
            f"trial '{script_path.name}'."
        )
    return abs(value)


def assemble_hetero_stack(
    matched: MatchedStack,
    sheets: List[SheetConfig],
    hetero_potentials: HeteroPotentials,
    hetero_stacking: str,
    settings: GlobalSettings,
    workdir: PathLike,
) -> HeteroStack:
    """Stack two matched supercells into ONE genuinely-periodic LAMMPS cell.

    Consumes the two per-material coincidence supercells (which already share
    one in-plane box) and the registered potential spine, and
    writes a single ``hetero.data`` with:

    * **Cross-material registry (RI offset).** :func:`scan_registry_index` on the
      two interface layers gives the AA (max-RI) / AB (min-RI) lateral offset;
      the choice follows the non-reference (top) material's ``stack_type``. The
      offset is applied as a **one-sided relative in-plane displacement of the
      non-reference material's layers only** -- the reference stays at the
      origin and the two materials are NEVER re-centered to a common centroid
      (the old branch's bug, which cancelled the offset).
    * **Per-interface z-spacing by real minimization.** For the cross-material
      interface a 2-layer trial is minimized with the real registered potential
      (:func:`_minimize_interface_gap`); the resulting COM z-gap is the interface
      spacing (no fixed nominal gap).
    * **Layer ordering.** ``'grouped'`` = all of material 0's layers then all of
      material 1's; ``'alternating'`` interleaves. Each material's own
      :func:`calculate_layer_shifts` gives its intra-material AA/AB shifts.
    * **Exact type alignment.** Each layer is read with its
      ``hetero_potentials.type_offsets`` so assembled global types match the PM.

    Every LAMMPS invocation runs with ``cwd = workdir`` so the settings file's
    run-dir-relative ``provenance/potentials/...`` references resolve.

    Args:
        matched: The two matched per-material supercells.
        sheets: Per-material sheet configs (exactly two; index 0 = supercell_a).
        hetero_potentials: Registered potential spine supplying the
            settings path and per-material type offsets.
        hetero_stacking: ``'grouped'`` or ``'alternating'``.
        settings: Global simulation settings.
        workdir: Run directory (also where ``hetero.data`` is written).

    Returns:
        :class:`HeteroStack` with the written data path, the carried settings
        path, per-layer bookkeeping, and the surviving RI offset.

    Raises:
        ValueError: On an unknown stacking, a material count other than two, an
            empty layers list, or mixed atom styles.
        RuntimeError: If a LAMMPS run fails or does not produce ``hetero.data``.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    stacking = hetero_stacking.lower()
    if stacking not in ("grouped", "alternating"):
        raise ValueError(
            f"hetero_stacking must be 'grouped' or 'alternating', got "
            f"{hetero_stacking!r}."
        )

    n_mats = len(sheets)
    if n_mats != 2:
        raise ValueError(
            f"assemble_hetero_stack builds one interface between exactly two "
            f"matched materials (MatchedStack holds a pair); got {n_mats} sheets."
        )

    mat_paths = [Path(matched.supercell_a), Path(matched.supercell_b)]
    type_offsets = list(hetero_potentials.type_offsets)

    n_layers: List[int] = []
    for sheet in sheets:
        if not sheet.layers:
            raise ValueError(f"SheetConfig for {sheet.mat!r} has an empty layers list.")
        n_layers.append(int(sheet.layers[0]))

    # Symmetric strain leaves no unstrained "reference" material. The cross-
    # material RI offset is a RELATIVE in-plane displacement between the two
    # interface layers, so anchor material 0 (supercell_a) unshifted at (0, 0)
    # and displace material 1 (supercell_b) by the offset -- the anchor choice
    # does not change the relative registry.
    ref_index = 0
    nonref_index = 1

    styles = {detect_atom_style(p) for p in mat_paths}
    if len(styles) > 1:
        raise ValueError(
            f"Cannot assemble materials with differing atom_style {sorted(styles)} "
            f"into one data file."
        )
    atom_style = styles.pop()

    # In-plane geometry (shared box) + per-material atom xy / z-extent.
    carts: Dict[int, np.ndarray] = {}
    z_ext: Dict[int, Tuple[float, float]] = {}
    cell2: Optional[np.ndarray] = None
    for m, path in enumerate(mat_paths):
        layer = read_lammps_data(path)
        this_cell2 = layer.cell[:2, :2]
        carts[m] = layer.positions[:, :2]
        z_ext[m] = (float(layer.positions[:, 2].min()), float(layer.positions[:, 2].max()))
        if cell2 is None:
            cell2 = this_cell2
        elif not np.allclose(this_cell2, cell2, atol=1e-4):
            # The matched supercells MUST share one in-plane box (build_matched_
            # supercells' guarantee). If they don't, stacking them into one cell
            # would create a periodic seam -- fail loud rather than emit a
            # silently mismatched structure.
            raise ValueError(
                "Matched supercells do not share one in-plane box "
                f"(material {m} cell {this_cell2.tolist()} != {cell2.tolist()}); "
                "cannot assemble a seamless periodic stack."
            )
    assert cell2 is not None
    lx, ly, xy = float(cell2[0, 0]), float(cell2[1, 1]), float(cell2[1, 0])

    # Cross-material RI anchor: scan shifts its SECOND argument, so pass the
    # reference material first and the non-reference second -> the returned
    # offset displaces the non-reference material's layers (one-sided).
    ri_result = scan_registry_index(carts[ref_index], carts[nonref_index], cell2)
    top_stack_type = sheets[nonref_index].stack_type.upper()
    ri_offset = np.array(
        ri_result.aa_shift if top_stack_type == "AA" else ri_result.ab_shift,
        dtype=float,
    )
    material_anchor: Dict[int, np.ndarray] = {
        ref_index: np.zeros(2, dtype=float),
        nonref_index: ri_offset,
    }

    # Intra-material AA/AB shifts (independent of the cross-material RI offset).
    dims = {"xlo": 0.0, "xhi": lx, "ylo": 0.0, "yhi": ly, "zlo": 0.0, "zhi": 1.0}
    intra_shifts: Dict[int, List[Tuple[float, float]]] = {}
    for m in range(n_mats):
        intra_shifts[m] = calculate_layer_shifts(
            sheets[m].mat, dims, n_layers=n_layers[m], stacking_type=sheets[m].stack_type
        )

    # Layer ordering.
    if stacking == "grouped":
        seq = [(m, k) for m in range(n_mats) for k in range(n_layers[m])]
    else:  # alternating
        seq = []
        for k in range(max(n_layers)):
            for m in range(n_mats):
                if k < n_layers[m]:
                    seq.append((m, k))

    total_types = len(hetero_potentials.pm.types)
    masses_block = hetero_potentials.pm.get_masses_string()
    settings_path = Path(hetero_potentials.settings_path)
    initial_guess = settings.geometry.lat_c_default

    # Per-interface spacing by real minimization, cached per distinct pair.
    gap_cache: Dict[Tuple[int, int], float] = {}

    def _gap(bot_m: int, top_m: int) -> float:
        key = (bot_m, top_m)
        if key in gap_cache:
            return gap_cache[key]
        if any(is_mlip(sheets[m].pot_type) for m in (bot_m, top_m)):
            # An MLIP interface cannot be minimized here: the generation machine
            # is not assumed to have an MLIP-enabled LAMMPS, and the
            # zero surrogate that makes the *assembly* run possible would
            # "minimize" a flat landscape. lat_c is mandatory for MLIP sheets
            # precisely so it can stand in. The LARGER of the pair is used: a
            # too-small starting gap under an attractive foundation model is the
            # documented "dome plow" failure mode, whereas a slightly too-large
            # one simply relaxes inward during the real run.
            spacings = [sheets[m].lat_c for m in (bot_m, top_m)
                        if sheets[m].lat_c is not None]
            if not spacings:
                raise ValueError(
                    f"An MLIP hetero interface between {sheets[bot_m].mat!r} and "
                    f"{sheets[top_m].mat!r} needs an explicit [2D] lat_c on at "
                    f"least one material: the interface spacing cannot be found "
                    f"by minimization without an MLIP-enabled LAMMPS."
                )
            gap_cache[key] = max(spacings)
            return gap_cache[key]
        if bot_m == top_m:
            if n_layers[bot_m] > 1:
                rel = np.array(intra_shifts[bot_m][1], dtype=float) - np.array(
                    intra_shifts[bot_m][0], dtype=float
                )
            else:
                rel = np.zeros(2, dtype=float)
        else:
            rel = material_anchor[top_m] - material_anchor[bot_m]
        gap = _minimize_interface_gap(
            workdir=workdir,
            settings=settings,
            settings_path=settings_path,
            atom_style=atom_style,
            total_types=total_types,
            masses_block=masses_block,
            lx=lx,
            ly=ly,
            xy=xy,
            bot_path=mat_paths[bot_m],
            bot_off=type_offsets[bot_m],
            bot_zext=z_ext[bot_m],
            top_path=mat_paths[top_m],
            top_off=type_offsets[top_m],
            top_zext=z_ext[top_m],
            top_shift_xy=rel,
            initial_guess=initial_guess,
        )
        gap_cache[key] = gap
        return gap

    # Place layers in z, accumulating per-interface gaps.
    layers_spec: List[HeteroStackLayer] = []
    z_cursor = 0.0
    prev_m: Optional[int] = None
    interface_spacing: Optional[float] = None
    for idx, (m, k) in enumerate(seq):
        if idx == 0:
            z_layer = 0.0
        else:
            assert prev_m is not None
            gap = _gap(prev_m, m)
            z_cursor += gap
            z_layer = z_cursor
            if prev_m != m and interface_spacing is None:
                interface_spacing = gap
        anchor = material_anchor[m]
        intra = intra_shifts[m][k] if k < len(intra_shifts[m]) else (0.0, 0.0)
        layers_spec.append(
            HeteroStackLayer(
                material_index=m,
                layer_index=k,
                z=z_layer,
                xy_offset=(float(anchor[0] + intra[0]), float(anchor[1] + intra[1])),
                group=f"_m{m}l{k}",
                type_offset=int(type_offsets[m]),
                source=mat_paths[m],
            )
        )
        prev_m = m

    # Box z-window from the actual (shifted) atom extents so nothing wraps.
    zmins = [z_ext[s.material_index][0] + s.z for s in layers_spec]
    zmaxs = [z_ext[s.material_index][1] + s.z for s in layers_spec]
    box_region = box_region_cmd(lx, ly, xy, min(zmins) - 10.0, max(zmaxs) + 10.0)

    # Render + run the assembly. The assembly is a pure placement (`run 0` then
    # write_data), so for an MLIP cell the zero surrogate stands in for the real
    # settings — otherwise generating inputs would need an MLIP-enabled LAMMPS
    # The REAL settings path is still what the run scripts include.
    # ``pair zero`` serialises a ``Pair Coeffs`` block that the run decks cannot
    # read back before their pair_style, so an MLIP cell writes ``nocoeff``.
    mlip_cell = hetero_potentials.pm.has_mlip()
    if mlip_cell:
        include_path = hetero_potentials.pm.write_surrogate_file(
            workdir / "stack_hetero.in.settings")
    else:
        include_path = settings_path

    output_path = workdir / "hetero.data"
    layers_ctx = [
        {
            "material_index": s.material_index,
            "layer_index": s.layer_index,
            "z": s.z,
            "path": str(s.source),
            "group": s.group,
            "type_offset": s.type_offset,
            "shift_x": s.xy_offset[0],
            "shift_y": s.xy_offset[1],
        }
        for s in layers_spec
    ]
    context = {
        "atom_style": atom_style,
        "box_region": box_region,
        "total_types": total_types,
        "layers": layers_ctx,
        "masses_block": masses_block,
        "settings_path": str(include_path),
        "output_path": str(output_path),
        "nocoeff": mlip_cell,
    }
    script_path = workdir / "stack_hetero.lmp"
    script_path.write_text(render_template("hetero/stack_hetero.lmp", context), encoding="utf-8")
    run_lammps(workdir, script_path)

    if not output_path.exists():
        raise RuntimeError(
            f"assemble_hetero_stack: LAMMPS did not produce {output_path}."
        )

    return HeteroStack(
        data_path=output_path,
        settings_path=settings_path,
        layers=layers_spec,
        ri_offset=ri_offset,
        interface_spacing=float(interface_spacing) if interface_spacing is not None else 0.0,
        total_types=total_types,
    )


def build_hetero_flake_component(
    flake_sheets: List[SheetConfig],
    settings: GlobalSettings,
    flake_cfg: FlakeConfig,
    workdir: Path,
    hetero_stacking: str = "grouped",
    interlayer_general: Any = None,
) -> Tuple[Path, dict]:
    """Build a shaped flake carved from a matched hetero bilayer.

    The bilayer is assembled as :func:`build_hetero_structure` does, then
    outlined by :func:`carve_flake`. The flake carries both materials' types,
    both layers sharing the same xy corners; under ilp/tmd the chalcogen
    sub-layer split follows ``interlayer_general``, while the assembly keeps
    the ``lj`` interactions.

    Returns:
        Tuple of (flake_path, flake_box_dims).

    Raises:
        ValueError: If ``flake_sheets`` does not hold exactly two materials.
    """
    if len(flake_sheets) != 2:
        raise ValueError(
            f"build_hetero_flake_component builds a flake from exactly two "
            f"materials (a bilayer); got {len(flake_sheets)}."
        )

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Monolayer per flake material (build_monolayer returns the path first).
    style = interlayer_style(interlayer_general)
    mono_paths = [Path(build_monolayer(sheet, interlayer=style)[0])
                  for sheet in flake_sheets]

    # 2. Match -> two supercells sharing one in-plane box.
    matched = build_matched_supercells(mono_paths[0], mono_paths[1], workdir)

    # 3. Register the flake materials' potential/type spine (geometry-independent).
    hetero_potentials = register_hetero_potentials(
        flake_sheets, settings, workdir,
        interlayer_general=interlayer_general, interlayer_terms=False)

    # 4. Assemble the two matched supercells into one stacked bilayer hetero.data
    #    (interface z-spacing by real minimization; types aligned to the spine).
    stack = assemble_hetero_stack(
        matched, flake_sheets, hetero_potentials, hetero_stacking, settings, workdir,
    )

    # 5. Carve the requested shape from the assembled bilayer. build_flake keeps
    #    every type and both layers (they share xy), removing only out-of-shape
    #    atoms via the boundary mask.
    # 6. Corner-type ALL bilayer types (core 2t-1 / corner 2t, across both
    #    materials) -> matches register_component(..., regions=[None, 'corner']).
    flake_path = workdir / f"hetero_flake_{flake_cfg.shape}.data"
    flake_dims = carve_flake(stack.data_path, flake_path, flake_cfg)
    logger.info("Built hetero flake component: %s", flake_path.name)
    return flake_path, flake_dims


# =============================================================================
# Top-level entrypoint: config -> ready heterostructure (Tasks 3-5 tied together)
# =============================================================================


def build_hetero_structure(
    config: SheetOnSheetSimulationConfig,
    settings: GlobalSettings,
    workdir: PathLike,
    potentials_prefix: Optional[str] = None,
) -> HeteroStack:
    """Build a ready heterostructure stack from a two-material config.

    This is the single entrypoint a later sim builder calls to obtain a
    self-consistent hetero stack (``hetero.data`` + ``system.in.settings``). It
    runs the steps in the order they require:

    1. **Monolayers.** Build each material's orthogonalized monolayer with the
       existing :func:`~frictionsim2d.builders.construction.components.build_monolayer` (first
       return
       value is the data-file path).
    2. **Match.** :func:`build_matched_supercells` finds the two
       monolayers' coincidence lattice on their actual in-plane cells, tiles
       each, and strains both onto their mean cell so both output supercells
       share exactly one in-plane periodic box.
    3. **Register potentials.** :func:`register_hetero_potentials` is
       geometry-independent, so it runs before assembly: it registers each
       material as its own :class:`PotentialManager` component (disjoint,
       contiguous atom-type blocks), writes ``system.in.settings``, and exposes
       the per-material ``type_offsets`` assembly needs.
    4. **Assemble.** :func:`assemble_hetero_stack` stacks the two matched
       supercells into one genuinely-periodic ``hetero.data`` -- applying the
       cross-material RI offset one-sided, finding the interface z-spacing by
       real minimization, and aligning atom types to the registered spine.

    Scope: the **two-material** case (one interface) -- what the coincidence
    match, the registry scan, and the single-interface assembler all handle. A
    general N-material generalization is intentionally out of scope here.

    Args:
        config: Two-material sheet-on-sheet config. ``config.sheets`` supplies
            the per-material :class:`SheetConfig` list (from the ``[2D-1]`` /
            ``[2D-2]`` sections) and ``config.general.hetero_stacking`` the layer
            ordering (``'grouped'`` | ``'alternating'``).
        settings: Global simulation settings, threaded to every sub-step (in
            practice the same object as ``config.settings``; the passed
            ``settings`` is authoritative).
        workdir: Run directory. All artefacts (per-material supercells,
            ``lammps/system.in.settings``, staged ``provenance/potentials``, and
            the final ``hetero.data``) are written under it.
        potentials_prefix: How the written ``pair_coeff`` lines name the staged
            potential files -- forwarded verbatim to
            :func:`register_hetero_potentials`, whose docstring states the
            contract. Default ``None`` keeps the run-dir-relative prefix.

    Returns:
        The assembled :class:`HeteroStack`; ``.data_path`` and ``.settings_path``
        are the ready heterostructure structure + potentials.

    Raises:
        ValueError: If fewer than two materials are configured (a
            heterostructure needs an interface between two materials).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    twist_angle = getattr(config.general, "twist_angle", None)

    sheets: List[SheetConfig] = list(config.sheets)
    if twist_angle is not None and len(sheets) == 1:
        # Same-material twist: a single [2D] material twisted against itself is
        # a genuine moiré -> synthesise two identical sheet entries so the
        # (two-material) matcher/registrar/assembler run. (Same-material stacks
        # otherwise skip matching entirely; a twist re-enables it.)
        sheets = [sheets[0], sheets[0]]
    if len(sheets) < 2:
        raise ValueError(
            f"build_hetero_structure needs >= 2 materials (a heterostructure "
            f"interface), or a single material with general.twist_angle set (a "
            f"same-material moiré); config.sheets has {len(config.sheets)} and "
            f"twist_angle={twist_angle}."
        )

    # 1. Monolayer per material (build_monolayer returns the path as element 0).
    #    Under ilp/tmd each one carries split chalcogen sub-layer types, which
    #    matching, tiling and assembly all keep.
    general = config.general
    style = interlayer_style(general)
    mono_paths = [Path(build_monolayer(sheet, interlayer=style)[0]) for sheet in sheets]

    # 2. Match -> two supercells sharing one shared box.
    if twist_angle is not None:
        # Twist path: the nearest commensurate moiré approximant to the
        # user-selected angle, emitted on an orthorhombic box.
        matched = build_twisted_supercells(
            mono_paths[0], mono_paths[1], workdir,
            twist_angle=twist_angle,
            angle_tol=general.twist_angle_tol,
            strain_tol=general.twist_strain_tol,
            max_supercell=general.twist_max_supercell,
        )
    else:
        # Plain (aligned) coincidence path. Thread the optional [lattice_match]
        # config through; when unset, build_matched_supercells' defaults apply.
        lm = config.lattice_match
        match_kwargs = (
            {
                "strain_tol": lm.strain_tol,
                "max_supercell": lm.max_supercell,
                "area_tol": lm.area_tol,
            }
            if lm is not None
            else {}
        )
        matched = build_matched_supercells(mono_paths[0], mono_paths[1], workdir, **match_kwargs)

    # 3. Register the materials' potential/type spine (geometry-independent).
    #    The finite/edge path needs the SINGLE-type spine here (assembly aligns
    #    the data file to it); the doubled core/edge spine is registered AFTER
    #    the patch is edge-typed (below), so the two stay consistent.
    #
    #    Always the DEFAULT run-dir-relative prefix, whatever the caller asked
    #    for: assembly below runs LAMMPS with ``cwd = workdir``, so that is the
    #    only prefix its own minimization runs can open. A caller-supplied
    #    prefix is for the RUN scripts, which may start elsewhere, and is
    #    applied by re-registering once the assembly is done (step 6).
    #
    #    A non-lj interlayer style gets its type layout here (the ilp/tmd
    #    chalcogen split, matching the monolayers) but not its interlayer term:
    #    the assembly runs need neither molecule IDs nor the INTERLAYER
    #    package, and the slide relaxes the stack under the real style before
    #    it drives anything. Step 6 writes the term.
    hetero_potentials = register_hetero_potentials(
        sheets, settings, workdir, interlayer_general=general, interlayer_terms=False)

    # 4. Assemble into one periodic hetero.data aligned to that spine.
    stack = assemble_hetero_stack(
        matched,
        sheets,
        hetero_potentials,
        config.general.hetero_stacking,
        settings,
        workdir,
    )
    # Carry the achieved twist + moiré period onto the stack (None for non-twist).
    stack.twist_angle_achieved = matched.twist_angle_achieved
    stack.moire_period = matched.moire_period
    stack.pm = hetero_potentials.pm

    # 5. Finite/edge path (opt-in). Edge typing runs AFTER matching + stacking,
    #    on the ASSEMBLED patch, so the edge band is the patch's TRUE perimeter
    #    (apply_sheet_edge_types keys off bound(all,*), the atom bounding box).
    #    Doing it per-monolayer before tiling would bury the band in the
    #    coincidence supercell's interior (the cell is generally not 1x1).
    edged = False
    if bool(config.general.finite_sheet) and len(sheets) >= 2:
        edged = _apply_finite_hetero_edges(
            stack, sheets, settings, workdir,
            potentials_prefix=potentials_prefix,
            interlayer_general=general,
        )

    # 6. Re-write the settings so the ``pair_coeff`` lines name the staged
    #    potentials the way the RUN scripts must name them. Assembly needed the
    #    run-dir-relative prefix (it runs LAMMPS from ``workdir``); a builder
    #    whose job scripts start from the simulation root needs its own. The
    #    registration is geometry-independent and deterministic, so replaying it
    #    changes the path strings and nothing else.
    #
    #    Skipped only when the edge path ABOVE actually re-registered: that
    #    write carries the doubled core/edge types and the appended edge groups,
    #    which a plain re-registration here would silently drop. It returns False
    #    when it no-ops (edge_width <= 0), and then this must still run.
    #
    #    A non-lj style always re-registers, prefix or not: the assembly spine
    #    above left its interlayer term out.
    if (potentials_prefix is not None or style != 'lj') and not edged:
        retargeted = register_hetero_potentials(
            sheets, settings, workdir, potentials_prefix=potentials_prefix,
            interlayer_general=general,
        )
        stack.settings_path = Path(retargeted.settings_path)
        stack.pm = retargeted.pm

    return stack


def _apply_finite_hetero_edges(
    stack: HeteroStack,
    sheets: List[SheetConfig],
    settings: GlobalSettings,
    workdir: Path,
    potentials_prefix: Optional[str] = None,
    interlayer_general: Any = None,
) -> bool:
    """Edge-type an assembled hetero patch in place and re-register its spine.

    Mirrors the homogeneous finite path's replicate-then-edge-type ordering
    (:func:`~frictionsim2d.builders.construction.components.build_monolayer`): the assembled
    ``hetero.data`` is retyped so every original type ``t`` splits into an
    interleaved core (``2t-1``) / edge (``2t``) pair via
    :func:`~frictionsim2d.builders.construction.components.apply_sheet_edge_types` (which keys off
    the
    atom bounding box, so the edge band is the patch's true perimeter), then the
    potential spine is re-registered with ``regions=[None, 'edge']`` so
    ``system.in.settings`` describes the doubled types. ``stack`` is updated in
    place (``settings_path`` / ``total_types``).

    This re-registration is also where a caller-supplied ``potentials_prefix``
    lands on the finite path: it is the last write of the settings file, so it
    is the one the run scripts read. See
    :func:`register_hetero_potentials` for what the prefix means, and for
    ``interlayer_general``, which this write carries for the same reason.

    Returns:
        Whether the spine was re-registered. False means this was a no-op
        (``finite_sheet_edge_width <= 0``) and the settings file still holds
        whatever the assembly wrote -- which is why
        :func:`build_hetero_structure` reads this rather than assuming the
        finite branch always had the last word on the prefix.

    Raises:
        ValueError: If ``finite_sheet_edge_width`` is not smaller than half the
            assembled patch's smaller in-plane span (the edge band would swallow
            the whole patch, leaving no core).
    """
    edge_width = float(settings.geometry.finite_sheet_edge_width)
    if edge_width <= 0:
        return False

    # Guard: the edge band must not swallow the patch. Span from the assembled
    # patch's atom bounding box (the same extent apply_sheet_edge_types uses).
    at = read_lammps_data(stack.data_path)
    xy = at.positions[:, :2]
    span_x = float(xy[:, 0].max() - xy[:, 0].min())
    span_y = float(xy[:, 1].max() - xy[:, 1].min())
    smaller = min(span_x, span_y)
    if edge_width >= 0.5 * smaller:
        raise ValueError(
            f"finite_sheet_edge_width={edge_width} A is not smaller than half the "
            f"assembled patch's smaller in-plane span ({smaller:.2f} A) -- the "
            f"edge band would swallow the whole patch, leaving no core. Reduce "
            f"finite_sheet_edge_width or enlarge the sheet."
        )

    # Retype the assembled patch: per original type t -> core 2t-1 / edge 2t.
    apply_sheet_edge_types(Path(stack.data_path), edge_width=edge_width)

    # Re-register the spine with the doubled core/edge types so
    # system.in.settings matches the retyped data file (same shared
    # register_sheet_block routine, now with regions).
    hp_edge = register_hetero_potentials(
        sheets, settings, workdir, regions=[None, "edge"],
        potentials_prefix=potentials_prefix,
        interlayer_general=interlayer_general,
    )
    stack.settings_path = Path(hp_edge.settings_path)
    stack.total_types = len(hp_edge.pm.types)
    stack.pm = hp_edge.pm

    # Emit the edge groups explicitly. get_component_groups_string special-cases
    # only the literal name 'sheet', so sheet_1/sheet_2 fall to the generic
    # {name}_all branch with NO 2D_edge split -- so the per-material edge groups,
    # the combined 2D_edge union, and the 2D union (sheet_1_all + sheet_2_all,
    # both already written by write_file) are appended here so the finite run
    # templates' `group sheet_edge intersect 2D 2D_edge` fires unchanged.
    component_names = hp_edge.component_names
    group_lines = ["", "# Finite-sheet edge groups (per-material + combined 2D)"]
    for name in component_names:
        edge_ids = hp_edge.pm.types.ids_by_component_region(name, "edge")
        group_lines.append(
            f"group {name}_edge type {' '.join(str(tid) for tid in edge_ids)}"
        )
    group_lines.append(
        f"group 2D_edge union {' '.join(f'{name}_edge' for name in component_names)}"
    )
    group_lines.append(
        f"group 2D union {' '.join(f'{name}_all' for name in component_names)}"
    )
    with open(stack.settings_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(group_lines) + "\n")

    return True
