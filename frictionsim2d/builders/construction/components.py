"""Component builders for FrictionSim2D.

This module contains the logic to construct the physical components of the simulation:
the AFM tip, the substrate, and the 2D material sheet. It handles the calculation
of duplication factors based on requested dimensions and unit cell properties.

Sections: flake carving, data-file helpers and orthogonalization, layer
stacking, then the monolayer, sheet, tip and substrate builders.
"""

import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast
import re
import math
import numpy as np


from ...core.config import (
    GlobalSettings, SheetConfig, SubstrateConfig, TipConfig, FlakeConfig
)
from ...core.mlip import is_mlip
from ...potentials.manager import PotentialManager
from ..utils.lammps_data import (
    atoms_section_start, atom_count_and_z_extent, atomic2charge, charge2atom, get_model_dimensions,
    get_num_atom_types, parse_cell_atoms_masses, parse_lmp_atoms,
    shift_atoms_to_z_zero, snap_small_tilts, write_typed_atomic,
)
from ...core.utils.paths import get_material_path
from ...core.utils.structures import cifread
from ...potentials.utils import (
    check_potential_cif_compatibility, get_potential_element_order
)
from ...interfaces.lammps_monolayer import restore_masses_section
from ...interfaces.jinja import render_lammps_commands
from ...interfaces.lammps import (
    create_lammps_instance, replicate_data_file, run_lammps_commands, run_on_data_file)
# No cycle: builders/utils/interlayer imports only frictionsim2d.potentials.interlayer.
from ..utils.interlayer import interlayer_style, set_interlayer_from_config
from ...potentials.interlayer import CHALCOGENS, chalcogen_split_elements
# Looked up here by _create_base_slab, so `components.make_amorphous` stays the patch point.
from .amorphous import make_amorphous

logger = logging.getLogger(__name__)


def _tmp_path(stem: str, tag: str = "", ext: str = ".lmp") -> Path:
    """A unique scratch path ``<tmp>/<stem>_<hex>[_<tag>]<ext>``."""
    suffix = f"_{tag}" if tag else ""
    return Path(tempfile.gettempdir()) / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}{ext}"


# =============================================================================
# Flake Geometry and Building
# =============================================================================


def _require_box_dims(
    dims: Dict[str, Optional[float]],
    *,
    context: str,
) -> Dict[str, float]:
    """Validate and narrow parsed box dimensions to floats.

    Args:
        dims: Mapping returned by get_model_dimensions.
        context: Text included in ValueError when a key is missing.

    Returns:
        Dictionary with fully-typed float values for x/y/z bounds.
    """
    keys = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")
    missing = [k for k in keys if dims.get(k) is None]
    if missing:
        raise ValueError(f"Could not parse box dimensions ({missing}) from {context}")
    return {key: float(cast(float, dims[key])) for key in keys}


def _to_shape_frame(
    x: float,
    y: float,
    center: Tuple[float, float],
    rotation_deg: float,
) -> Tuple[float, float]:
    """Point (x, y) relative to ``center``, rotated by ``-rotation_deg``.

    Testing the un-rotated shape against this point is the same as testing the
    shape rotated CCW by ``rotation_deg`` about ``center`` against (x, y).
    """
    x_rel = x - center[0]
    y_rel = y - center[1]
    if rotation_deg == 0.0:
        return x_rel, y_rel
    angle_rad = math.radians(-rotation_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return cos_a * x_rel - sin_a * y_rel, sin_a * x_rel + cos_a * y_rel


def _in_equilateral_triangle(
    x: float,
    y: float,
    edge_length: float,
    center: Tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
) -> bool:
    """Check if point (x, y) is inside a (optionally rotated) equilateral triangle.

    ``edge_length`` is the side ``a``. The triangle is centred on its CENTROID
    at ``center``: at ``rotation_deg = 0`` the apex points up (+y) at
    ``2h/3`` and the base lies along ``y = -h/3`` (``h = sqrt(3)/2 a``);
    ``rotation_deg`` rotates it CCW about the centroid.
    """
    x_rot, y_rot = _to_shape_frame(x, y, center, rotation_deg)

    height = edge_length * math.sqrt(3) / 2.0
    y_apex = 2.0 * height / 3.0
    y_base = -height / 3.0

    if y_rot < y_base or y_rot > y_apex:
        return False

    # Half-width shrinks linearly from a/2 at the base to 0 at the apex.
    half_width = 0.5 * edge_length * (y_apex - y_rot) / height
    return -half_width <= x_rot <= half_width


def _in_square(
    x: float,
    y: float,
    edge_length: float,
    center: Tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
) -> bool:
    """Check if point (x, y) is inside a (optionally rotated) square."""
    x_rot, y_rot = _to_shape_frame(x, y, center, rotation_deg)
    half = edge_length / 2.0
    return -half <= x_rot <= half and -half <= y_rot <= half


def _in_hexagon(
    x: float,
    y: float,
    edge_length: float,
    center: Tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
) -> bool:
    """Check if point (x, y) is inside a (optionally rotated) regular hexagon.

    ``edge_length`` is the distance from the center to a vertex (circumradius
    ``R``, which for a regular hexagon also equals the side). At
    ``rotation_deg = 0`` two vertices sit on the x axis at ``(+-R, 0)`` and
    two edges are horizontal at ``y = +-sqrt(3)/2 R``; ``rotation_deg``
    rotates it CCW about ``center``. Area ``3*sqrt(3)/2 * R**2``.
    """
    x_rot, y_rot = _to_shape_frame(x, y, center, rotation_deg)
    apothem = edge_length * math.sqrt(3) / 2.0
    # The three pairs of parallel edges have outward normals at 90, 30 and
    # 150 deg, each at distance `apothem` from the centre.
    half_sqrt3 = math.sqrt(3) / 2.0
    return (
        abs(y_rot) <= apothem
        and abs(half_sqrt3 * x_rot + 0.5 * y_rot) <= apothem
        and abs(half_sqrt3 * x_rot - 0.5 * y_rot) <= apothem
    )


def build_flake(
    base_lmp_path: Path,
    output_path: Path,
    shape: str = "triangle",
    edge_length: float = 40.0,
    rotation_deg: float = 0.0,
    center_xy: Optional[Tuple[float, float]] = None,
) -> Path:
    """Extract a natural-shaped flake from a supercell LAMMPS file.

    Creates a finite patch (triangle/square/hexagon) by selecting atoms within
    the shape boundary and writing them to a new LAMMPS data file. The flake is
    centered at the box center (or ``center_xy``) and can be rotated.

    Args:
        base_lmp_path: Path to source LAMMPS data file (single layer).
        output_path: Path for output flake file.
        shape: Flake shape ('triangle', 'square', 'hexagon').
        edge_length: Characteristic edge length (Å). For triangle/square: side;
            for hexagon: circumradius (vertex distance from center).
        rotation_deg: Rotation angle (degrees, counter-clockwise).
        center_xy: Override flake center (default: box center).

    Returns:
        Path to the output flake file.

    Raises:
        ValueError: If shape is not recognized or if base file has no atoms.
    """
    shape_lower = shape.lower()
    if shape_lower not in ("triangle", "square", "hexagon"):
        raise ValueError(
            f"Unknown flake shape: {shape}. Choose from: triangle, square, hexagon."
        )

    dims = _require_box_dims(get_model_dimensions(base_lmp_path), context=str(base_lmp_path))
    xlo, xhi = dims["xlo"], dims["xhi"]
    ylo, yhi = dims["ylo"], dims["yhi"]

    if center_xy is None:
        cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    else:
        cx, cy = center_xy

    parsed = parse_lmp_atoms(base_lmp_path)
    if parsed is None or len(parsed[0]) == 0:
        raise ValueError(f"No atoms found in {base_lmp_path}")

    types, xs, ys, zs = parsed
    natoms = len(types)

    boundary_check = {
        "triangle": lambda x, y: _in_equilateral_triangle(
            x, y, edge_length, (cx, cy), rotation_deg),
        "square": lambda x, y: _in_square(x, y, edge_length, (cx, cy), rotation_deg),
        "hexagon": lambda x, y: _in_hexagon(x, y, edge_length, (cx, cy), rotation_deg),
    }[shape_lower]

    mask = np.array([boundary_check(xs[i], ys[i]) for i in range(natoms)], dtype=bool)
    if not mask.any():
        raise ValueError(
            f"No atoms fell within the {shape_lower} flake boundary "
            f"(edge_length={edge_length:.1f} Å). Is the supercell large enough?"
        )

    selected_indices = np.where(mask)[0]
    lines = base_lmp_path.read_text(encoding="utf-8").splitlines()

    section = atoms_section_start(lines)
    if section is None:
        raise ValueError(f"Could not find Atoms section in {base_lmp_path}")
    atoms_start, atom_style = section

    atom_lines: List[str] = []
    new_id = 1
    for idx in selected_indices:
        raw_line = lines[atoms_start + idx]
        parts = raw_line.strip().split()
        if not parts or not parts[0].isdigit():
            raise ValueError(
                f"Malformed Atoms entry while building flake at line "
                f"{atoms_start + idx + 1}: '{raw_line}'"
            )
        parts[0] = str(new_id)
        atom_lines.append("  ".join(parts))
        new_id += 1

    selected_z = zs[selected_indices]
    z_min, z_max = float(np.min(selected_z)), float(np.max(selected_z))
    num_types = int(np.max(types[selected_indices]))

    output_lines: List[str] = []
    output_lines.append(f"Flake ({shape_lower}, edge={edge_length:.1f} A)")
    output_lines.append("")
    output_lines.append(f"{len(selected_indices)} atoms")
    output_lines.append("0 bonds")
    output_lines.append("0 angles")
    output_lines.append("0 dihedrals")
    output_lines.append("")
    output_lines.append(f"{num_types} atom types")
    output_lines.append("")
    output_lines.append(f"{xlo:.6f} {xhi:.6f} xlo xhi")
    output_lines.append(f"{ylo:.6f} {yhi:.6f} ylo yhi")
    output_lines.append(f"{z_min - 1.0:.6f} {z_max + 1.0:.6f} zlo zhi")
    output_lines.append("")

    # Copy the Masses section verbatim from the source file.
    masses_section: List[str] = []
    in_masses = False
    seen_mass_values = False
    for line in lines:
        if line.strip() == "Masses":
            in_masses = True
            masses_section.append(line)
            continue
        if in_masses:
            if not line.strip() and not seen_mass_values:
                masses_section.append(line)
                continue
            if not line.strip() and seen_mass_values:
                masses_section.append(line)
                break
            if line.strip() and line.strip()[0].isalpha():
                break
            seen_mass_values = True
            masses_section.append(line)

    if masses_section:
        output_lines.extend(masses_section)
        output_lines.append("")

    output_lines.append(f"Atoms # {atom_style}")
    output_lines.append("")
    output_lines.extend(atom_lines)
    output_lines.append("")

    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    logger.info(
        "[flake] Created %s flake at (%.1f, %.1f) with %d atoms",
        shape_lower, cx, cy, len(selected_indices),
    )

    return output_path


_SHAPE_N_VERTICES = {"triangle": 3, "square": 4, "hexagon": 6}


def _detect_corner_atom_indices(
    xs: np.ndarray,
    ys: np.ndarray,
    n_vertices: int,
    corner_radius: float,
) -> np.ndarray:
    """Identify the atom indices that sit at the polygon's corners.

    Robust to shape orientation/rotation: the flake centroid is computed, atoms
    are sorted into ``n_vertices`` angular sectors aligned with the farthest
    atom, the farthest atom in each sector is taken as the vertex, and every
    atom within ``corner_radius`` of any vertex is flagged as a corner atom.

    Args:
        xs, ys: Atom coordinates.
        n_vertices: Number of polygon corners (3, 4 or 6).
        corner_radius: Capture radius (Å) around each vertex.

    Returns:
        Sorted numpy array of corner atom indices.
    """
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    dx = xs - cx
    dy = ys - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    # Align sector boundaries so one sector is centered on the farthest atom.
    theta0 = float(theta[int(np.argmax(r))])
    sector = 2.0 * math.pi / n_vertices

    corner_mask = np.zeros(len(xs), dtype=bool)
    for k in range(n_vertices):
        center_angle = theta0 + k * sector
        # Wrap angular difference into [-pi, pi].
        d_angle = np.angle(np.exp(1j * (theta - center_angle)))
        in_sector = np.abs(d_angle) <= sector / 2.0
        if not in_sector.any():
            continue
        sector_idx = np.where(in_sector)[0]
        vertex_idx = sector_idx[int(np.argmax(r[sector_idx]))]
        vx, vy = xs[vertex_idx], ys[vertex_idx]
        near_vertex = np.hypot(xs - vx, ys - vy) <= corner_radius
        corner_mask |= near_vertex

    return np.where(corner_mask)[0]


def apply_flake_corner_types(
    flake_path: Path,
    shape: str,
    corner_radius: float,
) -> Path:
    """Re-type the flake's corner atoms into distinct LAMMPS atom types.

    Mirrors :func:`apply_sheet_edge_types`: every original type ``t`` is split
    into an interleaved core type ``2t-1`` and corner type ``2t`` so the result
    matches ``PotentialManager.register_component(..., regions=[None, 'corner'])``.
    The corner atoms can then be selected in LAMMPS via ``group ... type``.

    The flake data file is modified in-place. Returns the same path.

    Args:
        flake_path: LAMMPS data file produced by :func:`build_flake`.
        shape: Flake shape ('triangle', 'square', 'hexagon') — sets vertex count.
        corner_radius: Capture radius (Å) around each detected vertex.

    Raises:
        ValueError: If the shape is unknown or the file has no atoms.
    """
    shape_lower = shape.lower()
    if shape_lower not in _SHAPE_N_VERTICES:
        raise ValueError(
            f"Unknown flake shape: {shape}. Choose from: triangle, square, hexagon."
        )
    n_vertices = _SHAPE_N_VERTICES[shape_lower]

    parsed = parse_lmp_atoms(flake_path)
    if parsed is None or len(parsed[0]) == 0:
        raise ValueError(f"No atoms found in {flake_path}")
    types, xs, ys, _zs = parsed

    corner_indices = set(
        int(i) for i in _detect_corner_atom_indices(xs, ys, n_vertices, corner_radius)
    )
    logger.info(
        "[flake] Marked %d corner atoms (of %d) for %s flake",
        len(corner_indices), len(types), shape_lower,
    )

    lines = flake_path.read_text(encoding="utf-8").splitlines()

    section = atoms_section_start(lines)
    if section is None:
        raise ValueError(f"Could not find Atoms section in {flake_path}")
    atoms_start = section[0]
    masses_start: Optional[int] = None
    for i in range(atoms_start - 2):
        if lines[i].strip() == "Masses":
            masses_start = i

    num_types = int(np.max(types))

    # Rewrite the atom type column: core -> 2t-1, corner -> 2t.
    atom_seq = 0
    for li in range(atoms_start, len(lines)):
        stripped = lines[li].strip()
        if not stripped or stripped.startswith("#"):
            if atom_seq > 0:
                break
            continue
        parts = stripped.split()
        if not parts[0].isdigit():
            break
        old_type = int(parts[1])
        is_corner = atom_seq in corner_indices
        parts[1] = str(2 * old_type if is_corner else 2 * old_type - 1)
        lines[li] = "  ".join(parts)
        atom_seq += 1

    # Update the atom-types header.
    for i, line in enumerate(lines):
        if re.match(r"^\s*\d+\s+atom types\s*$", line.strip()):
            lines[i] = f"{2 * num_types} atom types"
            break

    # Expand the Masses section: each original type -> core + corner with same mass.
    if masses_start is not None:
        new_mass_lines: List[str] = []
        mi = masses_start + 1
        # Skip the blank line after 'Masses'.
        while mi < len(lines) and not lines[mi].strip():
            mi += 1
        mass_end = mi
        old_masses: Dict[int, str] = {}
        while mass_end < len(lines):
            s = lines[mass_end].strip()
            if not s:
                break
            if s[0].isalpha():
                break
            mparts = s.split()
            old_masses[int(mparts[0])] = mparts[1]
            mass_end += 1
        for t in range(1, num_types + 1):
            mass_val = old_masses.get(t, "1.0")
            new_mass_lines.append(f"{2 * t - 1} {mass_val}")
            new_mass_lines.append(f"{2 * t} {mass_val}")
        lines[mi:mass_end] = new_mass_lines

    flake_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return flake_path


def flake_atom_span(flake_path: Path) -> Tuple[float, float]:
    """The flake's own ``(span_x, span_y)`` in A, measured from its atoms.

    Not the box: :func:`build_flake` writes the carving supercell's x/y bounds
    into the flake file unchanged, so the box says nothing about how wide the
    carved shape is.
    """
    parsed = parse_lmp_atoms(flake_path)
    if parsed is None or len(parsed[0]) == 0:
        raise ValueError(f"No atoms found in {flake_path}")
    _types, xs, ys, _zs = parsed
    return float(xs.max() - xs.min()), float(ys.max() - ys.min())


def check_flake_pbc_clearance(
    flake_span: Tuple[float, float],
    cell: Tuple[float, float],
    lj_cutoff: float,
) -> None:
    """Refuse a flake that interacts with its own periodic image.

    The sheet is periodic; the flake is not. What separates a flake from its
    image is ``cell - span``, and while that gap is within the pair cutoff the
    flake feels a copy of itself and the recorded friction is not the friction
    of one flake. This is the ``[flake]`` counterpart of
    :meth:`~frictionsim2d.core.config.AFMSimulationConfig.validate_tip_sheet_footprint`,
    and it uses the same criterion as the finite-sheet substrate margin: the
    gap must exceed ``settings.potential.lj_cutoff``.

    Checked here rather than in the config models because the span is a
    property of the carved structure (shape, rotation and lattice registry all
    move it), and for a hetero stack the cell is not known until the
    coincidence cell has been matched and replicated.

    Raises:
        ValueError: naming the offending axis, its gap and the cutoff.
    """
    span_x, span_y = float(flake_span[0]), float(flake_span[1])
    cell_x, cell_y = float(cell[0]), float(cell[1])
    gaps = {'x': cell_x - span_x, 'y': cell_y - span_y}
    tight = {axis: gap for axis, gap in gaps.items() if gap <= float(lj_cutoff)}
    if not tight:
        return
    detail = ", ".join(f"{axis}: {gap:.2f} A" for axis, gap in sorted(tight.items()))
    raise ValueError(
        f"Flake ({span_x:.2f} x {span_y:.2f} A) is too close to its own "
        f"periodic image in a {cell_x:.2f} x {cell_y:.2f} A cell ({detail}; "
        f"pair cutoff {float(lj_cutoff)}). The flake would feel a copy of "
        "itself and the recorded forces would not be those of one flake. "
        "Shrink [Flake] edge_length, or enlarge the sheet ([2D] x/y, or a "
        "larger [tip] r, which replicates the hetero cell further)."
    )


def carve_flake(
    src_path: Path, flake_path: Path, cfg: FlakeConfig,
) -> Dict[str, float]:
    """Carve ``cfg``'s shape out of ``src_path`` into ``flake_path``, split core/corner types.

    Returns:
        The flake's box dimensions.
    """
    center_xy: Optional[Tuple[float, float]] = None
    if cfg.center_x is not None and cfg.center_y is not None:
        center_xy = (float(cfg.center_x), float(cfg.center_y))

    build_flake(
        src_path,
        flake_path,
        shape=cfg.shape,
        edge_length=float(cfg.edge_length),
        rotation_deg=float(cfg.rotation_deg),
        center_xy=center_xy,
    )
    apply_flake_corner_types(
        flake_path,
        shape=cfg.shape,
        corner_radius=float(cfg.corner_radius),
    )
    return _require_box_dims(get_model_dimensions(flake_path), context=str(flake_path))


def build_flake_component(
    config: "FlakeConfig",
    build_dir: Path,
    interlayer: Optional[str] = None,
) -> Tuple[Path, dict]:
    """Build a shaped flake (monolayer supercell → cut shape → mark corners).

    Reuses :func:`build_monolayer` to make a supercell of the flake's own
    material, :func:`build_flake` to carve the requested outline, and
    :func:`apply_flake_corner_types` to split corner atoms into their own
    interleaved types (matching ``regions=[None, 'corner']``).

    Args:
        config: Flake configuration (``mat``/``cif``/``pot`` + ``shape``,
            ``edge_length``, ``rotation_deg``, ``corner_radius``, supercell
            ``x``/``y``).
        build_dir: Directory to write the final ``flake_*.data`` file into.
        interlayer: The run's interlayer style, for :func:`build_monolayer`'s
            ilp/tmd chalcogen split.

    Returns:
        Tuple of (flake_path, flake_box_dims).
    """
    base_path, _dims, _pot_counts, _total_types, _supercell = build_monolayer(
        config, edge_mode='none', edge_width=0.0, interlayer=interlayer,
    )

    flake_path = build_dir / f"flake_{config.mat}_{config.shape}.data"
    flake_dims = carve_flake(base_path, flake_path, config)
    base_path.unlink(missing_ok=True)
    logger.info("Built flake component: %s", flake_path.name)
    return flake_path, flake_dims


def _write_cif_as_lammps_atomic(
    cif_path: Path,
    output_path: Path,
    specorder: Optional[List[str]] = None
) -> None:
    """Write a CIF structure as LAMMPS data in atomic style.

    ASE's LAMMPS writer omits the Masses section, which LAMMPS requires and
    which downstream readers use for element identification.  This function
    injects the missing Masses block after the atom-type count header.
    Type order is set by ``specorder`` rather than by post-hoc renumbering.

    Args:
        cif_path: Input CIF path.
        output_path: Output LAMMPS data path.
        specorder: Optional element order for deterministic type assignment.
    """
    # Lazy: ase.io costs ~270 ms; deferred so this module stays cheap to import.
    from ase import io as ase_io  # pylint: disable=import-outside-toplevel
    from ase.data import atomic_masses, atomic_numbers  # pylint: disable=import-outside-toplevel

    atoms = ase_io.read(str(cif_path))
    if isinstance(atoms, list):
        atoms = atoms[0]

    if output_path.exists():
        output_path.unlink()

    write_kwargs = {
        'filename': str(output_path),
        'images': atoms,
        'format': 'lammps-data',
        'atom_style': 'atomic'
    }
    if specorder:
        write_kwargs['specorder'] = specorder

    ase_io.write(**write_kwargs)

    # Inject Masses section (ASE omits it); use specorder when available.
    # Insert just before the Atoms section so box-bounds lines aren't mistaken for masses.
    elements = specorder if specorder else sorted(set(atoms.get_chemical_symbols()))
    masses_block = "Masses\n\n"
    for idx, element in enumerate(elements, start=1):
        mass = atomic_masses[atomic_numbers[element]]
        masses_block += f"{idx} {mass:.6f}  # {element}\n"
    masses_block += "\n"

    content = output_path.read_text(encoding='utf-8')
    # Insert the Masses block immediately before the "Atoms" section header
    content = re.sub(
        r'(Atoms\s)',
        masses_block + r'\1',
        content,
        count=1
    )
    output_path.write_text(content, encoding='utf-8')


def _orthogonalize_lammps_data(input_path: Path, output_path: Path) -> None:
    """Orthogonalize a LAMMPS data file.

    For orthogonal (no-tilt) boxes, uses LAMMPS ``change_box all ortho``.
    For triclinic (tilted) boxes, uses ASE ``make_supercell`` to build the
    minimum orthogonal supercell, then writes the result as LAMMPS data and
    re-injects the Masses section preserved from the original file.

    Args:
        input_path: Input LAMMPS data path.
        output_path: Orthogonalized output path.
    """
    # Lazy: ase.io costs ~270 ms; deferred so this module stays cheap to import.
    from ase import io as ase_io  # pylint: disable=import-outside-toplevel
    from ase.build import make_supercell as _make_supercell  # pylint: disable=import-outside-toplevel

    if output_path.exists():
        output_path.unlink()

    # Check whether the box is already orthogonal (no xy xz yz line with non-zero values)
    content = input_path.read_text(encoding='utf-8')
    tilt_match = re.search(
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+'
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+'
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+xy xz yz',
        content
    )
    has_tilt = tilt_match is not None and any(
        abs(float(tilt_match.group(i))) > 1e-10 for i in (1, 2, 3)
    )

    if not has_tilt:
        # Orthogonal box: LAMMPS change_box approach
        run_on_data_file(input_path, "atomic", ["change_box all ortho"], output_path)
        return

    # Triclinic box: tile a minimum orthogonal supercell with ASE. Each LAMMPS
    # type travels as a distinct placeholder element, so sublattice types that
    # share a chemical element (Se1/Se2) are not collapsed; the real masses are
    # restored afterwards. ase_io.write keeps the box in LAMMPS
    # restricted-triclinic orientation.
    from ase import Atoms as _Atoms  # pylint: disable=import-outside-toplevel
    from ase.data import chemical_symbols as _csym  # pylint: disable=import-outside-toplevel

    cell, atom_rows, masses = parse_cell_atoms_masses(content)
    types = [t for (t, *_r) in atom_rows]
    positions = [(x, y, z) for (_t, x, y, z) in atom_rows]
    n_types = max(masses) if masses else (max(types) if types else 1)
    placeholders = [_csym[i] for i in range(1, n_types + 1)]  # H, He, Li, ... per type
    symbols = [placeholders[t - 1] for t in types]

    atoms = _Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    transform_matrix = _find_ortho_transform(atoms.get_cell())
    ortho_atoms = _make_supercell(atoms, transform_matrix)
    ortho_atoms.wrap()

    ase_io.write(str(output_path), ortho_atoms, format='lammps-data',
                 atom_style='atomic', specorder=placeholders)

    # Restore the real Masses section (ASE writes placeholder masses / omits it).
    real_masses = {t: masses.get(t, (1.0, 'X')) for t in range(1, n_types + 1)}
    restore_masses_section(output_path, real_masses)
    snap_small_tilts(output_path)


def _find_ortho_transform(cell) -> "np.ndarray":
    """Find a transformation matrix P such that P @ cell is (nearly) orthogonal.

    For hexagonal cells, the canonical 2× transformation [[1,0,0],[1,2,0],[0,0,1]]
    produces an orthorhombic cell.  For cells already orthogonal, returns the
    identity matrix.

    Args:
        cell: ASE Cell or 3×3 array of cell vectors (rows).

    Returns:
        3×3 integer numpy array P.
    """
    _np = np

    c = _np.array(cell)
    a1, a2 = c[0], c[1]

    # If already orthogonal, return identity
    if abs(_np.dot(a1, a2)) < 1e-6 * _np.linalg.norm(a1) * _np.linalg.norm(a2):
        return _np.eye(3, dtype=int)

    # Search for n1, n2 integers (smallest cell) such that n1*a1 + n2*a2 is ~along y.
    # A relative x-tolerance accepts slightly-distorted (near-hexagonal) cells whose
    # a/b ratio is not exactly commensurate (e.g. GaO's a2/a1 = -0.50002); the tiny
    # residual tilt is snapped to zero after the supercell is built.
    best_transform = None
    best_key = None
    x_tol = max(1e-6, 5e-3 * float(_np.linalg.norm(a1)))
    search_range = range(-12, 13)
    for n1 in search_range:
        for n2 in search_range:
            if n1 == 0 and n2 == 0:
                continue
            trial = n1 * a1 + n2 * a2
            if abs(trial[0]) < x_tol and abs(trial[1]) > 1e-3:
                key = (abs(n2), abs(n1), abs(trial[0]))
                if best_key is None or key < best_key:
                    best_key = key
                    best_transform = _np.array([[1, 0, 0], [n1, n2, 0], [0, 0, 1]], dtype=int)

    if best_transform is not None:
        return best_transform

    # Fallback: identity (leave as-is)
    return _np.eye(3, dtype=int)


def _assign_pot_types_by_order(path: Path, order: List[str],
                               counts: Dict[str, int]) -> None:
    """Assign potential sublattice types to a primitive cell, in-place.

    Used when the number of atoms equals the number of potential types
    (``type_multiplier == 1`` with sublattice types, e.g. t-PtSe2's Pt/Se1/Se2).
    Each element's atoms are assigned that element's block of sublattice type ids
    in file order (cycling if there are more atoms than sublattice types), giving
    exactly ``sum(counts)`` types. The subsequent orthogonalization tiles these
    types across the supercell.

    Args:
        path: Primitive LAMMPS data file (modified in-place).
        order: Element order from the potential (e.g. ['Pt', 'Se']).
        counts: Sublattice-type count per element (e.g. {'Pt': 1, 'Se': 2}).
    """
    # Lazy: ase.data costs ~70-90 ms (measured with python -X importtime);
    # deferred so this module stays cheap to import.
    from ase.data import atomic_masses as _am, atomic_numbers as _an  # pylint: disable=import-outside-toplevel

    cell, atoms, masses = parse_cell_atoms_masses(path.read_text(encoding='utf-8'))

    # Map each current type-id to an element. Prefer the Masses ``# element``
    # comment, but fall back to the nearest atomic mass among the potential's
    # elements when it is missing: LAMMPS ``replicate``/``write_data`` (used by
    # replicate_data_file just before this call) strips those comments. Without
    # the fallback every atom is left unmatched, its type is never reassigned, and
    # the potential's sublattice types collapse to one type per element.
    order_masses = {el: float(_am[_an[el]]) for el in order}

    def _element_for(cur_type: int) -> str:
        mass, elem = masses.get(cur_type, (None, 'X'))
        if elem in order_masses:
            return elem
        if mass is not None and order_masses:
            return min(order_masses, key=lambda el: abs(order_masses[el] - mass))
        return order[0]

    cur_type_elem = {t: _element_for(t) for t in masses}

    # Contiguous type-id block per element, in potential order.
    elem_type_ids: Dict[str, List[int]] = {}
    next_id = 1
    for el in order:
        n = int(counts[el])
        elem_type_ids[el] = list(range(next_id, next_id + n))
        next_id += n

    counters = {el: 0 for el in order}
    new_types = []
    for (cur_type, *_xyz) in atoms:
        el = cur_type_elem.get(cur_type, order[0])
        ids = elem_type_ids.get(el)
        if not ids:  # element not in potential order: leave as-is
            new_types.append(cur_type)
            continue
        new_types.append(ids[counters[el] % len(ids)])
        counters[el] += 1

    new_masses = {
        tid: (float(_am[_an[el]]), el)
        for el, ids in elem_type_ids.items() for tid in ids
    }
    positions = [(x, y, z) for (_t, x, y, z) in atoms]
    write_typed_atomic(path, cell, positions, new_types, new_masses)


def split_chalcogen_sublayers(path: Path, elements: List[str],
                              split: Set[str]) -> Dict[int, Tuple[int, int]]:
    """Give each split chalcogen's two sub-layers their own type, in place.

    For ilp/tmd (see :func:`frictionsim2d.potentials.interlayer.chalcogen_split_elements`).
    ``path`` is a MONOLAYER atomic data file whose type ids are contiguous per
    element. Every type of an element in ``split`` becomes two consecutive ids:
    atoms above the layer's metal plane (mean z of the non-chalcogen atoms)
    take the first, those below the second -- decided by position, never by
    file order -- and every later type shifts up to make room. That is the
    element-first order :meth:`PotentialManager._assign_types` registers
    (``S`` -> ``S`` top, ``S`` bottom), so the stacking loop and the
    potential manager agree on every id.

    Only the type column, the ``Masses`` section and the ``atom types`` count
    change; geometry and every other line are left byte-for-byte alone.

    Returns:
        ``{old type: (top type, bottom type)}`` for the split types.

    Raises:
        ValueError: If a split type has no atom on one side of the metal plane.
    """
    # Lazy: ase.data costs ~70-90 ms; see _assign_pot_types_by_order.
    from ase.data import atomic_masses as _am, atomic_numbers as _an  # pylint: disable=import-outside-toplevel

    text = path.read_text(encoding='utf-8')
    _cell, atoms, masses = parse_cell_atoms_masses(text)
    el_mass = {el: float(_am[_an[el]]) for el in elements}

    def _element(t: int) -> str:
        mass, label = masses.get(t, (None, 'X'))
        if label in el_mass:
            return label
        return min(el_mass, key=lambda el: abs(el_mass[el] - mass)) if mass else 'X'

    type_el = {t: _element(t) for t in sorted(masses)}
    metal_z = [z for (t, _x, _y, z) in atoms if type_el.get(t) not in CHALCOGENS]
    if not metal_z:
        raise ValueError(f"{path}: no metal atoms to split the chalcogen sub-layers around.")
    plane = sum(metal_z) / len(metal_z)

    new_ids: Dict[int, Tuple[int, int]] = {}
    split_ids: Dict[int, Tuple[int, int]] = {}
    nxt = 1
    for t in sorted(type_el):
        if type_el[t] in split:
            new_ids[t] = (nxt, nxt + 1)
            split_ids[t] = new_ids[t]
            nxt += 2
        else:
            new_ids[t] = (nxt, nxt)
            nxt += 1
    for t, _x, _y, z in atoms:
        if t in split_ids and abs(z - plane) < 1e-6:
            raise ValueError(f"{path}: chalcogen atom on the metal plane (z={z}).")
    for t in split_ids:
        zs = [z for (tt, _x, _y, z) in atoms if tt == t]
        if not (any(z > plane for z in zs) and any(z < plane for z in zs)):
            raise ValueError(
                f"{path}: type {t} ({type_el[t]}) does not lie on both sides of "
                f"the metal plane (z = {plane:.4f}); cannot split its sub-layers.")

    out: List[str] = []
    section = None
    for raw in text.splitlines(keepends=True):
        line = raw.strip()
        body = line.split('#', 1)[0].strip()
        if body.endswith('atom types'):
            out.append(f"{nxt - 1} atom types\n")
            continue
        if line == 'Masses' or line.startswith('Atoms'):
            section = 'Masses' if line == 'Masses' else 'Atoms'
            out.append(raw)
            continue
        if body and body[0].isalpha():
            section = None
        parts = body.split()
        if section == 'Masses' and parts and parts[0].isdigit():
            old = int(parts[0])
            for new in sorted(set(new_ids[old])):
                comment = f"  # {type_el[old]}" if type_el[old] != 'X' else ''
                out.append(f"{new} {parts[1]}{comment}\n")
            continue
        if section == 'Atoms' and len(parts) >= 5 and parts[0].isdigit():
            old = int(parts[1])
            top, bottom = new_ids.get(old, (old, old))
            new = top if float(parts[4]) > plane else bottom
            parts[1] = str(new)
            tail = raw[len(raw.rstrip('\n')):]
            comment = ' #' + line.split('#', 1)[1] if '#' in line else ''
            out.append(' '.join(parts) + comment + tail)
            continue
        out.append(raw)
    path.write_text(''.join(out), encoding='utf-8')
    return split_ids


def _replicate_or_copy(src: Path, dst: Path, nx: int, ny: int, nz: int) -> None:
    """Replicate ``src`` ``nx x ny x nz`` into ``dst``; a 1x1x1 request is a plain copy."""
    if nx == 1 and ny == 1 and nz == 1:
        shutil.copy(src, dst)
    else:
        replicate_data_file(src, dst, nx, ny, nz)


def calculate_layer_shifts(
    mat_name: str,
    dims: Dict[str, float],
    n_layers: int = 1,
    use_pair_bonding: bool = False,
    stacking_type: str = 'AB'
) -> List[Tuple[float, float]]:
    """Calculate in-plane stacking shifts for each layer.

    Args:
        mat_name: Material name.
        dims: Box dimensions.
        n_layers: Number of layers.
        use_pair_bonding: If True, first and last pairs are bonded with no relative shift.
        stacking_type: Stacking type ('AA' or 'AB'). AA has no shifts, AB has shifts.

    Returns:
        List of (shift_x, shift_y) tuples per layer.
    """
    if stacking_type.upper() == 'AA':
        return [(0.0, 0.0) for _ in range(n_layers)]

    if mat_name.startswith('p-') or mat_name == 'black_phosphorus':
        base_shift_x = (dims['xhi'] - dims['xlo']) / 2
        base_shift_y = (dims['yhi'] - dims['ylo']) / 2
    else:
        base_shift_x = 0.0
        base_shift_y = (dims['yhi'] - dims['ylo']) / 3

    layer_shifts = []

    if use_pair_bonding and n_layers >= 3:
        for i in range(n_layers):
            if n_layers == 3:
                if i == 0:
                    layer_shifts.append((0.0, 0.0))
                else:
                    layer_shifts.append((base_shift_x, base_shift_y))
            else:
                if i <= 1:
                    layer_shifts.append((0.0, 0.0))
                elif i >= n_layers - 2:
                    num_shifts = n_layers - 3
                    layer_shifts.append((base_shift_x * num_shifts, base_shift_y * num_shifts))
                else:
                    num_shifts = i - 1
                    layer_shifts.append((base_shift_x * num_shifts, base_shift_y * num_shifts))
    else:
        for i in range(n_layers):
            layer_shifts.append((base_shift_x * i, base_shift_y * i))

    return layer_shifts


def create_orthogonal_slab(
    cif_path: Path,
    output_path: Path,
    target_x: float,
    target_y: float,
    target_z: Optional[float] = None,
) -> Tuple[Path, Dict[str, float], List[int]]:
    """Create orthogonalized slab from CIF file sized to target dimensions.

    Args:
        cif_path: Input CIF file path.
        output_path: Output LAMMPS data file path.
        target_x: Target X dimension (Angstroms).
        target_y: Target Y dimension (Angstroms).
        target_z: Target Z dimension (Angstroms), or None for single layer.

    Returns:
        Tuple of (output_path, box_dimensions, duplication_factors).
    """
    cif_path = Path(cif_path).absolute()

    unit_cell = _tmp_path(cif_path.stem, "unit")
    predup_cell = _tmp_path(cif_path.stem, "predup")
    ortho_cell = _tmp_path(cif_path.stem, "ortho")

    cif_data = cifread(cif_path)
    _write_cif_as_lammps_atomic(cif_path, unit_cell, specorder=cif_data['elements'])
    replicate_data_file(unit_cell, predup_cell, 2, 2, 1)
    _orthogonalize_lammps_data(predup_cell, ortho_cell)
    unit_cell.unlink()
    predup_cell.unlink()

    dims = get_model_dimensions(ortho_cell)
    assert all(dims[k] is not None for k in ['xhi', 'xlo', 'yhi', 'ylo', 'zhi', 'zlo'])

    unit_x = (cast(float, dims['xhi']) - cast(float, dims['xlo'])) / 2.0
    unit_y = (cast(float, dims['yhi']) - cast(float, dims['ylo'])) / 2.0
    unit_z = cast(float, dims['zhi']) - cast(float, dims['zlo'])

    dup_x = max(1, round(target_x / unit_x / 2.0))
    dup_y = max(1, round(target_y / unit_y / 2.0))
    dup_z = 1 if target_z is None else max(1, round(target_z / unit_z))

    duplication = [dup_x, dup_y, dup_z]

    _replicate_or_copy(ortho_cell, output_path, dup_x, dup_y, dup_z)

    ortho_cell.unlink()

    final_dims = get_model_dimensions(output_path)
    assert all(final_dims[k] is not None for k in ['xhi', 'xlo', 'yhi', 'ylo', 'zhi', 'zlo'])
    final_dims_typed: Dict[str, float] = {k: cast(float, v) for k, v in final_dims.items()}
    return output_path, final_dims_typed, duplication


def _orthogonal_primitive_cell(cif_path: Path, out_path: Path) -> Dict[str, float]:
    """Write the orthogonalized **primitive** unit cell of a CIF (no supercelling).

    Unlike :func:`create_orthogonal_slab` (which duplicates 2×2 in-plane before
    orthogonalizing, so its smallest output is a 2× supercell), this orthogonalizes
    the raw unit cell directly, yielding the primitive in-plane repeat. For an
    already-orthogonal cell (e.g. cubic Si, 5.44 Å) it is a no-op; for a
    non-orthogonal cell the coincidence match downstream yields a sheared supercell
    and the caller falls back to amorphous.

    Using the *primitive* cell (vs the 2× supercell) is what lets the substrate
    commensurate onto a wider set of hetero boxes at low residual strain — the 2×
    cell needlessly doubles the tiling granularity and forces large-strain matches.

    Args:
        cif_path: Source CIF.
        out_path: Output LAMMPS data path for the orthogonal primitive cell.

    Returns:
        The cell's box dimensions (``get_model_dimensions`` keys, floats).
    """
    cif_path = Path(cif_path).absolute()
    unit = _tmp_path(cif_path.stem, "unit")
    cif_data = cifread(cif_path)
    _write_cif_as_lammps_atomic(cif_path, unit, specorder=cif_data['elements'])
    _orthogonalize_lammps_data(unit, out_path)
    unit.unlink(missing_ok=True)
    dims = get_model_dimensions(out_path)
    return {k: cast(float, v) for k, v in dims.items()}


def extract_substrate_surface_cell(
    sub_config: SubstrateConfig,
    build_dir: Path,
) -> Tuple[np.ndarray, Path]:
    """Return the in-plane **primitive** surface cell (2x2) + a one-layer atom file.

    Produces the coincidence-search input for a crystalline substrate match: the
    orthogonalized in-plane *primitive* surface cell (a 2x2 row-vector lattice
    matrix) and a path to a data file holding just the topmost atomic layer. This
    is deliberately distinct from the bulk-thickness :func:`create_orthogonal_slab`
    output — it is the thin, finest-granularity surface repeat the Zur-McGill
    search operates on. Using the primitive cell (not the 2× supercell
    ``create_orthogonal_slab`` emits) minimises the residual match strain, which is
    what keeps a strained crystalline substrate mechanically stable end to end.

    Args:
        sub_config: Substrate configuration (sources the crystalline cell from
            ``sub_config.cif_path``).
        build_dir: Directory for the intermediate + one-layer data files.

    Returns:
        ``(surface_cell_2x2, mono_path)`` — the in-plane primitive surface cell (Å,
        row vectors, LAMMPS-canonical orthogonal) and the one-layer data-file path.
    """
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    cif_path = get_material_path(sub_config.cif_path)

    prim_cell = build_dir / f"{sub_config.mat}_surface_prim.lmp"
    dims = _orthogonal_primitive_cell(cif_path, prim_cell)
    lx = dims['xhi'] - dims['xlo']
    ly = dims['yhi'] - dims['ylo']
    surface_cell = np.array([[lx, 0.0], [0.0, ly]])

    # Slice the topmost atomic layer -> a one-layer surface data file. The
    # coincidence search only needs the surface termination (single layer), not
    # the full z-repeat, as its representative atom set.
    # Lazy: ase.io costs ~270 ms; deferred so this module stays cheap to import.
    from ase import io as ase_io  # pylint: disable=import-outside-toplevel

    atoms = ase_io.read(str(prim_cell), format="lammps-data")
    z = atoms.get_positions()[:, 2]
    top_mask = z >= (z.max() - 0.5)          # topmost atomic plane (~0 A ptp)
    mono = atoms[top_mask]

    mono_path = build_dir / f"{sub_config.mat}_surface_mono.lmp"
    ase_io.write(str(mono_path), mono, format="lammps-data")
    prim_cell.unlink(missing_ok=True)

    return surface_cell, mono_path


def build_matched_crystalline_substrate(
    hetero_box: np.ndarray,
    sub_config: SubstrateConfig,
    thickness: float,
    build_dir: Path,
) -> Path:
    """Build a crystalline substrate strained exactly onto the fixed hetero box.

    Fixed-reference coincidence build: the hetero box is pinned and the crystalline
    substrate is strained onto it so the substrate tiles the same periodic cell with
    no seam. Pipeline: extract the substrate surface cell ->
    :func:`~frictionsim2d.lattice.matching.match_to_fixed_box` (fail-loud if the residual
    strain exceeds ``2 * match_strain_tol``) -> apply the supercell matrix in-plane
    -> strain onto the hetero box -> a proper full-thickness crystalline block.

    **Thickness note:** replicating the single surface layer in z is not viable: a
    lone atomic layer stacked at the intra-cell spacing puts atoms ~1.4 Å apart (Si
    bond ~2.35 Å) and blows up under Stillinger-Weber. The thickness is built by
    replicating the true z-periodic primitive cell instead.

    Args:
        hetero_box: 2×2 fixed in-plane hetero box (row vectors, Å, LAMMPS-canonical
            — first vector along x). Pinned; never strained.
        sub_config: Crystalline substrate configuration (``cif_path``, ``mat``,
            optional ``match_strain_tol``).
        thickness: Target substrate thickness (Å).
        build_dir: Output directory for ``sub.lmp`` and intermediates.

    Returns:
        Path to ``sub.lmp`` — the crystalline substrate on the hetero box exactly.

    Raises:
        ValueError: If no substrate supercell matches the box within
            ``2 * match_strain_tol`` (propagated from ``match_to_fixed_box``), or if
            the least-strain supercell is sheared (non-diagonal) — an orthogonal box
            cannot be tiled by a sheared substrate cell.
    """
    # Lazy: deliberate, breaks the builders<->lattice import cycle.
    from ...lattice.matching import match_to_fixed_box  # pylint: disable=import-outside-toplevel
    from ...lattice.apply import strain_monolayer_to_cell  # pylint: disable=import-outside-toplevel

    hetero_box = np.asarray(hetero_box, dtype=float)
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # 1) Surface cell for the coincidence search (the mono file is the surface
    #    representation; only the in-plane cell metric drives the match).
    surface_cell, _ = extract_substrate_surface_cell(sub_config, build_dir)

    # 2) Fixed-reference match: pin the hetero box, strain the substrate onto it.
    strain_tol = sub_config.match_strain_tol
    match = match_to_fixed_box(hetero_box, surface_cell, strain_tol=strain_tol)
    matrix = match.supercell_matrix
    if int(matrix[0, 1]) != 0 or int(matrix[1, 0]) != 0:
        raise ValueError(
            f"Crystalline substrate match yielded a sheared supercell {matrix.tolist()}; "
            "an orthogonal hetero box cannot be tiled by a sheared substrate cell. "
            "Use an amorphous substrate ([sub] amorph = a) or a commensurate "
            "orthogonal substrate/hetero pair."
        )
    nx, ny = int(matrix[0, 0]), int(matrix[1, 1])

    # 3) Build the crystalline block from the SAME primitive cell the match used
    #    (so `matrix` tiles it consistently); dup_z reaches `thickness`.
    cif_path = get_material_path(sub_config.cif_path)
    prim = build_dir / f"{sub_config.mat}_prim.lmp"
    pdims = _orthogonal_primitive_cell(cif_path, prim)
    unit_z = pdims['zhi'] - pdims['zlo']
    dup_z = max(1, round(thickness / unit_z))

    # 4) Apply the supercell matrix in-plane + thicken in z (single replicate).
    rep = build_dir / f"{sub_config.mat}_crystal_rep.lmp"
    _replicate_or_copy(prim, rep, nx, ny, dup_z)

    # 5) Strain the tiled block exactly onto the fixed hetero box (no seam). The
    #    out-of-plane vector and z coordinates are left untouched.
    sub_lmp = build_dir / "sub.lmp"
    strain_monolayer_to_cell(str(rep), str(sub_lmp), hetero_box)

    prim.unlink(missing_ok=True)
    rep.unlink(missing_ok=True)
    logger.info(
        "Built matched crystalline substrate %s: %dx%d surface supercell strained "
        "onto %.2f x %.2f A hetero box (residual strain %.4f), thickness ~%.1f A.",
        sub_config.mat, nx, ny, float(hetero_box[0, 0]), float(hetero_box[1, 1]),
        match.residual_strain, thickness,
    )
    return sub_lmp


def _create_base_slab(
    config,
    cif_path: Path,
    target_x: float,
    target_y: float,
    target_z: float,
    output_path: Path,
    settings: GlobalSettings,
) -> None:
    """Create base slab (amorphous or crystalline).

    Args:
        config: TipConfig or SubstrateConfig.
        cif_path: CIF file path.
        target_x: Target X dimension (Angstroms).
        target_y: Target Y dimension (Angstroms).
        target_z: Target Z dimension (Angstroms).
        output_path: Output slab file path.
        settings: Global simulation settings.

    Raises:
        AmorphousMissing: an amorphous component has no matching library block
            (:func:`~frictionsim2d.builders.construction.amorphous.make_amorphous`
            never melt-quenches).
    """
    if config.amorph == 'a':
        pot_path = get_material_path(config.pot_path)
        amor_path = make_amorphous(
            config.mat, cif_path,
            target_x=target_x, target_y=target_y, target_z=target_z,
            pot_path=pot_path, pot_type=config.pot_type, settings=settings,
        )
        shutil.copy(amor_path, output_path)
    else:
        create_orthogonal_slab(
            cif_path, output_path,
            target_x=target_x, target_y=target_y, target_z=target_z
        )


def build_stacking_potential_file(
    config: SheetConfig,
    settings: GlobalSettings,
    layers_for_setup: int,
    pot_file: Path,
    edge_mode: str = 'none',
    interlayer_general: Any = None,
    potentials_dir: Optional[Path] = None,
) -> PotentialManager:
    """Write the potential file that the stacking LAMMPS run ``include``s.

    Classical sheets get their real potential (self-interactions plus the
    interlayer term of the run's ``interlayer_style`` where the sheet needs
    one) -- byte-identical to the inline block this replaced.

    MLIP sheets get a **``pair_style zero`` surrogate** instead (spec §3.3):
    stacking merely places layers at a known ``lat_c`` and renumbers types, and
    requiring an MLIP-enabled python-LAMMPS just to *generate* inputs would make
    the generator undeployable on ordinary machines and in CI. ``pair_style
    zero`` is stock LAMMPS, is defined for every element (unlike an SW
    surrogate, which presumes a parametrization exists for the material — false
    in general for a screening tool) and exerts exactly zero force, so it cannot
    influence the stacked geometry. The masses still have to be there: LAMMPS
    refuses to run without a mass for every type. No interlayer style applies
    (config validation refuses a non-lj ``interlayer_style`` under an MLIP), so
    an MLIP stack never needs molecule IDs.

    Args:
        config: the sheet configuration.
        settings: global settings (LJ cutoff, thermostat, ...).
        layers_for_setup: how many layers this stacking pass builds.
        pot_file: destination path.
        edge_mode: finite-sheet edge mode; ``!= 'none'`` splits edge types out.
        interlayer_general: the run's ``[general]`` config whose
            ``interlayer_style`` couples the layers (``None`` -> plain LJ).
        potentials_dir: private directory for generated interlayer parameter
            files (LJ-G's KC/z pair); ``None`` for plain LJ.

    Returns:
        The :class:`PotentialManager` used, whose type registry the caller
        needs for ``create_box`` and the renumbering loop, and whose
        ``needs_molecule_ids`` tells the template whether to set molecule IDs.
    """
    pm = PotentialManager(settings, potentials_dir=potentials_dir)
    regions = [None, 'edge'] if edge_mode != 'none' else None

    if is_mlip(config.pot_type):
        pm.register_component("sheet", config, n_layers=layers_for_setup,
                              regions=regions)
        pm.write_surrogate_file(pot_file)
        return pm

    # Before registration: ilp/tmd's chalcogen split is decided there.
    if interlayer_general is not None and pm.is_sheet_lj(config.pot_type):
        set_interlayer_from_config(pm, interlayer_general, {'sheet': config})
    pm.register_component("sheet", config, n_layers=layers_for_setup,
                          regions=regions)
    pm.add_self_interaction("sheet")

    if pm.is_sheet_lj(config.pot_type):
        pm.add_interlayer(["sheet"])

    pm.write_file(pot_file)
    return pm


def stack_multilayer_sheet(
    base_layer_path: Path,
    config: SheetConfig,
    output_path: Path,
    box_dims: Dict[str, float],
    n_layers: int,
    types_per_layer: int,
    pot_counts: Dict[str, int],
    supercell_dims: Dict[str, float],
    lat_c: Optional[float] = None,
    settings: Optional[GlobalSettings] = None,
    use_pair_bonding: bool = False,
    stacking_type: str = 'AB',
    edge_mode: str = 'none',
    info_out: Optional[Dict[str, Any]] = None,
    _lat_c_source: Optional[str] = None,
    interlayer_general: Any = None,
) -> float:
    """Stack multiple 2D material layers with proper atom type renumbering.

    Args:
        base_layer_path: Single-layer LAMMPS data file path.
        config: Sheet configuration.
        output_path: Stacked output file path.
        box_dims: Box dimensions from base layer.
        n_layers: Number of layers to stack.
        types_per_layer: Atom types per layer before renumbering.
        pot_counts: Element to type count mapping.
        supercell_dims: Original supercell dimensions for calculating shifts.
        lat_c: Interlayer distance (calculated if None).
        settings: Global settings (required if lat_c is None).
        use_pair_bonding: If True, layers are bonded in pairs (L1-L2, L3-L4)
            with no relative shift within pairs. Used for sheetonsheet simulations.
        stacking_type: Stacking type ('AA' or 'AB'). AA has no shifts, AB has shifts.
        info_out: Optional mutable dict; ``lat_c_source`` is set to ``'config'``
            (lat_c passed in), ``'minimised'`` (extracted from LAMMPS) or
            ``'default'`` (extraction failed, ``lat_c_default`` used).
        _lat_c_source: Internal override used by the recursive call, which
            receives a non-None ``lat_c`` that was minimised one level up.
        interlayer_general: The run's ``[general]`` config. Its
            ``interlayer_style`` is used between the layers here too, so a
            minimised ``lat_c`` is the chosen style's spacing; ``None`` (or
            ``lj``) is the plain interlayer LJ, as before.
    Returns:
        float: Interlayer distance (lat_c).
    """
    if n_layers < 2:
        raise ValueError("stack_multilayer_sheet requires at least 2 layers")
    assert settings is not None

    calculate_lat_c = lat_c is None
    if info_out is not None and not calculate_lat_c:
        info_out['lat_c_source'] = _lat_c_source or 'config'

    if calculate_lat_c:
        if is_mlip(config.pot_type):
            # Belt and braces: SheetConfig makes lat_c mandatory for MLIP
            # sheets precisely so this branch is unreachable. Reaching it would
            # mean minimizing under the pair_style zero surrogate (which exerts
            # no force) and returning a meaningless interlayer distance.
            raise ValueError(
                f"An MLIP sheet (pot_type = '{config.pot_type}') cannot have its "
                f"lat_c found by minimization: stacking runs under a "
                f"'pair_style zero' surrogate because no MLIP-enabled "
                f"python-LAMMPS is assumed on the generation machine. Set an "
                f"explicit [2D] lat_c (see scripts/mlip_bilayer_benchmark.py)."
            )
        logger.info("Running minimization to find optimal interlayer distance")
        initial_guess = settings.geometry.lat_c_default
    else:
        initial_guess = lat_c

    layers_for_setup = 2 if calculate_lat_c else n_layers
    pot_file = (
        _tmp_path(f"sheet_{layers_for_setup}", ext=".in.settings")
    )

    style = interlayer_style(interlayer_general)
    # A non-lj style may write generated parameter files (LJ-G's KC/z pair);
    # give them a private directory so concurrent builds cannot share one.
    work_dir = (Path(tempfile.mkdtemp(prefix='stack_interlayer_'))
                if style != 'lj' else None)
    pm = build_stacking_potential_file(
        config, settings, layers_for_setup=layers_for_setup,
        pot_file=pot_file, edge_mode=edge_mode,
        interlayer_general=interlayer_general, potentials_dir=work_dir,
    )
    needs_molecule_ids = pm.needs_molecule_ids

    total_types = len(pm.types)
    max_z = initial_guess * (layers_for_setup - 1) + 20.0

    layer_shifts = [initial_guess * l for l in range(layers_for_setup)]

    per_layer_shifts = calculate_layer_shifts(
        config.mat, supercell_dims, n_layers=layers_for_setup,
        use_pair_bonding=use_pair_bonding, stacking_type=stacking_type
    )
    layer_shifts_x = [shift[0] for shift in per_layer_shifts]
    layer_shifts_y = [shift[1] for shift in per_layer_shifts]

    atom_style = 'charge' if config.pot_type in ('reaxff', 'reax/c') else 'atomic'

    context = {
        'box_xlo': box_dims['xlo'],
        'box_xhi': box_dims['xhi'],
        'box_ylo': box_dims['ylo'],
        'box_yhi': box_dims['yhi'],
        'box_zhi': max_z,
        'total_types': total_types,
        'atom_style': atom_style,
        'base_layer_path': base_layer_path,
        'n_layers': layers_for_setup,
        'layer_shifts': layer_shifts,
        'layer_shifts_x': layer_shifts_x,
        'layer_shifts_y': layer_shifts_y,
        'types_per_layer': types_per_layer,
        'pot_counts': pot_counts,
        'pot_file': pot_file,
        'pot_type': config.pot_type.lower(),
        'calculate_lat_c': calculate_lat_c,
        'min_style': settings.simulation.min_style if calculate_lat_c else None,
        'minimization_command': (
            settings.simulation.minimization_command if calculate_lat_c else None
        ),
        'timestep': settings.simulation.timestep if calculate_lat_c else None,
        'output_path': output_path,
        'needs_molecule_ids': needs_molecule_ids,
        'molecule_groups': [f'layer_{i}' for i in range(1, layers_for_setup + 1)],
    }

    commands = render_lammps_commands('common/stack_layers.lmp', context)

    lmp = create_lammps_instance()
    try:
        for cmd in commands:
            lmp.command(cmd)

        if calculate_lat_c:
            extracted = lmp.extract_variable("lat_c", None, 0)
            lat_c = float(extracted) if extracted is not None else initial_guess
            lat_c_source = 'minimised' if extracted is not None else 'default'
            if info_out is not None:
                info_out['lat_c_source'] = lat_c_source
            logger.info("Found optimal lat_c: %.4f", lat_c)

            if n_layers > 2:
                lmp.close()
                return stack_multilayer_sheet(
                    base_layer_path=base_layer_path,
                    config=config,
                    output_path=output_path,
                    box_dims=box_dims,
                    n_layers=n_layers,
                    types_per_layer=types_per_layer,
                    pot_counts=pot_counts,
                    supercell_dims=supercell_dims,
                    lat_c=lat_c,
                    settings=settings,
                    use_pair_bonding=use_pair_bonding,
                    stacking_type=stacking_type,
                    edge_mode=edge_mode,
                    info_out=info_out,
                    _lat_c_source=lat_c_source,
                    interlayer_general=interlayer_general,
                )

        lat_c = lat_c or initial_guess

    except Exception as e:
        logger.error("Failed to stack layers: %s", e)
        raise
    finally:
        lmp.close()
        if pot_file.exists():
            pot_file.unlink()
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

    # Re-anchor the stacked sheet so its lowest atom is at z=0.
    # Without this, some stacking/minimization outcomes can leave atoms below
    # zero, which reduces the intended sheet-substrate separation in AFM setup.
    if output_path.exists():
        shift_atoms_to_z_zero(output_path)

    return lat_c


def build_tip(
    config: TipConfig,
    build_dir: Path,
    settings: GlobalSettings,
) -> Tuple[Path, float]:
    """Build AFM tip structure (spherical geometry).

    Args:
        config: Tip configuration.
        build_dir: Output directory.
        settings: Global simulation settings.

    Returns:
        Tuple of (tip_path, actual_radius).
    """
    cif_path = get_material_path(config.cif_path)

    base_lmp = _tmp_path(config.mat, "base")
    final_lmp = build_dir / "tip.lmp"
    radius = config.r
    box_size = radius * 2.0

    _create_base_slab(
        config, cif_path,
        target_x=box_size, target_y=box_size, target_z=box_size,
        output_path=base_lmp,
        settings=settings,
    )

    if config.amorph != 'a':
        dim = get_model_dimensions(base_lmp)
        assert all(dim[k] is not None for k in ['xhi', 'xlo', 'yhi', 'ylo'])
        radius = min(cast(float, dim['xhi']) - cast(float, dim['xlo']),
                    cast(float, dim['yhi']) - cast(float, dim['ylo'])) / 2.0

    h = radius / settings.geometry.tip_reduction_factor

    context = {
        'base_lmp': base_lmp,
        'radius': radius,
        'tip_height': h,
        'output_path': final_lmp
    }

    commands = render_lammps_commands('common/build_tip.lmp', context)

    run_lammps_commands(commands)
    base_lmp.unlink()
    return final_lmp, radius


def build_substrate(
    config: SubstrateConfig,
    build_dir: Path,
    box_dims: dict,
    settings: Optional[GlobalSettings] = None,
    target_x: Optional[float] = None,
    target_y: Optional[float] = None,
    ) -> Path:
    """Build substrate slab.

    Args:
        config: Substrate configuration.
        build_dir: Output directory.
        box_dims: Reference box dimensions from sheet.
        settings: Global settings (required for amorphous).
        target_x: Optional explicit substrate size in x.
        target_y: Optional explicit substrate size in y.

    Returns:
        Path to substrate file.
    """

    cif_path = get_material_path(config.cif_path)
    base_lmp = _tmp_path(config.mat, "base")
    final_lmp = build_dir / "sub.lmp"
    target_x = float(target_x) if target_x is not None else box_dims['xhi'] - box_dims['xlo']
    target_y = float(target_y) if target_y is not None else box_dims['yhi'] - box_dims['ylo']
    target_z = config.thickness

    _create_base_slab(
        config, cif_path,
        target_x=target_x, target_y=target_y, target_z=target_z,
        output_path=base_lmp,
        settings=settings or GlobalSettings(),
    )

    # Preserve periodic tiling for crystalline substrates by using the
    # commensurate replicated slab span in x/y instead of forcing a non-
    # commensurate crop to the requested target.
    if config.amorph == 'a':
        box_x_span = target_x
        box_y_span = target_y
    else:
        base_dims = get_model_dimensions(base_lmp)
        assert all(base_dims[k] is not None for k in ['xlo', 'xhi', 'ylo', 'yhi'])
        box_x_span = cast(float, base_dims['xhi']) - cast(float, base_dims['xlo'])
        box_y_span = cast(float, base_dims['yhi']) - cast(float, base_dims['ylo'])

    context = {
        'base_lmp': base_lmp,
        'box_xlo': box_dims['xlo'],
        'box_xhi': box_dims['xlo'] + box_x_span,
        'box_ylo': box_dims['ylo'],
        'box_yhi': box_dims['ylo'] + box_y_span,
        'box_zlo': box_dims['zlo'],
        'target_z': target_z,
        'output_path': final_lmp
    }

    commands = render_lammps_commands('common/build_substrate.lmp', context)

    run_lammps_commands(commands)
    base_lmp.unlink()
    return final_lmp


def apply_langevin_regions(
    component_name: str,
    component_path: Path,
    settings: GlobalSettings,
    component_height: float,
    substrate_thickness: Optional[float] = None
) -> Path:
    """Apply Langevin thermostat region splitting to a component.

    Splits a component (tip or substrate) into three spatial regions:
    - fix: Bottom region (fixed atoms)
    - thermo: Middle region (thermalized atoms)
    - normal: Top region (mobile atoms)

    Each original atom type gets expanded to 3 types (normal, fix, thermo).
    Generates appropriate mass and potential commands.

    Args:
        component_name: Name of component ('tip' or 'sub').
        component_path: Path to the component LAMMPS data file.
        settings: Global simulation settings with langevin_boundaries.
        component_height: Height of component (tip_height or substrate thickness).
        substrate_thickness: Required if component_name='sub' to calculate boundaries.

    Returns:
        Path to the modified component file with 3-region types.
    """
    num_types = get_num_atom_types(component_path)

    boundaries = {}
    if component_name == 'tip':
        h = component_height
        bounds_config = settings.thermostat.langevin_boundaries['tip']
        boundaries['f_zlo'] = h - bounds_config['fix'][0]
        boundaries['f_zhi'] = h - bounds_config['fix'][1]
        boundaries['t_zlo'] = h - bounds_config['thermo'][0]
        boundaries['t_zhi'] = h - bounds_config['thermo'][1]
        boundaries['n_zhi'] = h
    elif component_name == 'sub':
        if substrate_thickness is None:
            raise ValueError("substrate_thickness required for substrate component")
        bounds_config = settings.thermostat.langevin_boundaries['sub']
        boundaries['f_zlo'] = bounds_config['fix'][0] * substrate_thickness
        boundaries['f_zhi'] = bounds_config['fix'][1] * substrate_thickness
        boundaries['t_zlo'] = bounds_config['thermo'][0] * substrate_thickness
        boundaries['t_zhi'] = bounds_config['thermo'][1] * substrate_thickness
        boundaries['n_zhi'] = substrate_thickness
    else:
        raise ValueError(f"Unknown component name: {component_name}")

    # No potential is needed: the template only relabels types.

    context = {
        'component_name': component_name,
        'input_path': component_path,
        'num_types': num_types,
        'boundaries': boundaries,
    }

    commands = render_lammps_commands('common/apply_langevin_regions.lmp', context)

    run_lammps_commands(commands)

    return component_path


def apply_sheet_edge_types(component_path: Path, edge_width: float) -> Path:
    """Split a built sheet into core and edge atom types."""
    if edge_width <= 0:
        return component_path

    num_types = get_num_atom_types(component_path)
    if num_types < 1:
        return component_path

    context = {
        'input_path': component_path,
        'num_types': num_types,
        'edge_width': edge_width,
    }

    commands = render_lammps_commands('common/apply_sheet_edge_types.lmp', context)

    run_lammps_commands(commands)

    return component_path


def build_monolayer(
    config: SheetConfig,
    edge_mode: str = 'none',
    edge_width: float = 0.0,
    interlayer: Optional[str] = None,
) -> Tuple[Path, dict, dict, int, Dict[str, float]]:
    """Build single-layer 2D material sheet.

    Workflow:
    1. Read CIF structure and validate against potential.
    2. Write as LAMMPS data with types matching CIF element order.
    3. Renumber types to match potential file expectations (if multi-type).
    4. Duplicate and orthogonalize as needed.

    Args:
        config: Sheet configuration.
        interlayer: The run's interlayer style. Under ``ilp`` resolving to
            ilp/tmd, a chalcogen the potential types once is split into a
            top and a bottom sub-layer type (:func:`split_chalcogen_sublayers`);
            ``pot_counts`` then reports 2 for it. ``None``/any other style:
            unchanged.

    Returns:
        Tuple of (path, box_dims, pot_counts, total_types, supercell_dims).

    Raises:
        ValueError: If CIF elements do not match potential file expectations.
    """
    cif_path = get_material_path(config.cif_path)
    pot_path = get_material_path(config.pot_path)
    base_path = _tmp_path(f"{config.mat}_1")

    cif_data = cifread(cif_path)
    cif_elements = cif_data['elements']

    pot_element_order, pot_counts = get_potential_element_order(
        pot_path, config.pot_type, cif_elements
    )
    total_pot_types = sum(pot_counts.values())

    logger.debug(
        "Potential mapping: CIF elements %s -> %d types per element %s",
        cif_elements, total_pot_types, pot_counts
    )

    type_multiplier = 1 if config.pot_type in ['rebo', 'rebomos', 'airebo', 'meam', 'reaxff'] else \
        check_potential_cif_compatibility(cif_path, pot_path)

    temp_unit_cell = _tmp_path(config.mat, "unit")
    _write_cif_as_lammps_atomic(cif_path, temp_unit_cell, specorder=cif_elements)

    has_sublattice_types = any(v != 1 for v in pot_counts.values())

    # When the primitive already holds exactly `total_pot_types` atoms
    # (type_multiplier == 1, e.g. t-PtSe2's Pt/Se1/Se2), assign the potential
    # sublattice types on the PRIMITIVE, then let the (type-preserving)
    # orthogonalization tile them across the supercell. Assigning them *after*
    # orthogonalization instead produced one unique type per atom, because
    # orthogonalization multiplies the atom count (the pre-fix failure mode).
    assign_on_primitive = has_sublattice_types and type_multiplier == 1
    if assign_on_primitive:
        _assign_pot_types_by_order(temp_unit_cell, pot_element_order, pot_counts)

    # Orthogonalize. _orthogonalize_lammps_data preserves each atom's explicit
    # LAMMPS type, so sublattice types assigned above survive the tiling.
    ortho_cell = _tmp_path(config.mat, "ortho")
    _orthogonalize_lammps_data(temp_unit_cell, ortho_cell)
    temp_unit_cell.unlink()

    # Types are assigned either on the primitive (type_multiplier == 1, above) or
    # on the duplicated potential cell (type_multiplier > 1, below), so the
    # orthogonalized element-typed cell needs no post-ortho renumber here.

    if type_multiplier != 1:
        # The potential defines `total_pot_types` types for a k× supercell of the
        # primitive (e.g. p-SnS: 8 types on a 2× cell; h-MoS2: 12 on a 2×2 cell).
        # Duplicate the orthogonal cell to exactly `total_pot_types` atoms, then
        # assign the sublattice types by element order (each element's atoms take
        # that element's block of type ids). Counting atoms via a manual parse
        # avoids ase.io.read's StopIteration on LAMMPS write_data output
        # (Velocities section + image flags).
        num_atoms = len(parse_cell_atoms_masses(ortho_cell.read_text(encoding='utf-8'))[1])
        if num_atoms and total_pot_types % num_atoms == 0:
            factor = total_pot_types // num_atoms
            dup_factor_x, dup_factor_y = 1, 1
            for factor_candidate in range(int(np.sqrt(factor)) + 1, 0, -1):
                if factor % factor_candidate == 0:
                    dup_factor_x = factor_candidate
                    dup_factor_y = factor // factor_candidate
                    break
            dup_ortho = _tmp_path(config.mat, "dup")
            replicate_data_file(ortho_cell, dup_ortho, dup_factor_x, dup_factor_y, 1)
            ortho_cell.unlink()
            ortho_cell = dup_ortho
            _assign_pot_types_by_order(ortho_cell, pot_element_order, pot_counts)

    supercell_dims: Dict[str, float] = {
        k: cast(float, v) for k, v in get_model_dimensions(ortho_cell).items()}

    unit_x = supercell_dims['xhi'] - supercell_dims['xlo']
    unit_y = supercell_dims['yhi'] - supercell_dims['ylo']

    target_x = config.x[0] if isinstance(config.x, list) else config.x
    target_y = config.y[0] if isinstance(config.y, list) else config.y
    dup_x = max(1, round(target_x / unit_x))
    dup_y = max(1, round(target_y / unit_y))

    _replicate_or_copy(ortho_cell, base_path, dup_x, dup_y, 1)
    ortho_cell.unlink()

    shift_atoms_to_z_zero(base_path)
    dims = get_model_dimensions(base_path)

    split = chalcogen_split_elements(interlayer or 'lj', cif_elements, pot_counts)
    if split:
        split_chalcogen_sublayers(base_path, cif_elements, split)
        pot_counts = {el: (2 if el in split else n) for el, n in pot_counts.items()}
        total_pot_types = sum(pot_counts.values())
        logger.info("ilp/tmd: split %s into top/bottom sub-layer types", sorted(split))

    edge_enabled = edge_mode != 'none'
    if edge_enabled:
        apply_sheet_edge_types(base_path, edge_width=edge_width)

    if config.pot_type in ['tersoff', 'sw', 'rebo', 'airebo']:
        charge2atom(base_path)
    elif config.pot_type in ['reaxff', 'reax/c']:
        atomic2charge(base_path)

    if edge_enabled:
        total_pot_types *= 2

    observed_types = get_num_atom_types(base_path)
    if observed_types != total_pot_types:
        raise ValueError(
            f"Type count mismatch after build_monolayer: expected {total_pot_types} "
            f"(from potential) but observed {observed_types} in final structure. "
            f"Potential mapping: {pot_counts}"
        )

    logger.info(
        "Built %s monolayer: %d atom types, mapping %s",
        config.mat, total_pot_types, pot_counts
    )

    return base_path, dims, pot_counts, total_pot_types, supercell_dims


def build_sheet(
    config: SheetConfig,
    build_dir: Path,
    stack_if_multi: bool = False,
    settings: Optional[GlobalSettings] = None,
    n_layers_override: Optional[int] = None,
    use_pair_bonding: bool = False,
    stacking_type: str = 'AB',
    edge_mode: str = 'none',
    edge_width: float = 0.0,
    unit_cell_out: Optional[Dict[str, float]] = None,
    lat_c_info_out: Optional[Dict[str, Any]] = None,
    interlayer_general: Any = None,
) -> Tuple[Path, dict, Optional[float]]:
    """Build 2D material sheet (single or multi-layer).

    Args:
        config: Sheet configuration.
        build_dir: Output directory.
        stack_if_multi: If True, stack multiple layers.
        settings: Global settings (required for multi-layer).
        n_layers_override: Override layer count.
        use_pair_bonding: If True, use pair bonding stacking (for sheetonsheet).
        stacking_type: Stacking type ('AA' or 'AB'). AA has no shifts, AB has shifts.
        unit_cell_out: Optional mutable dict populated in-place with the
            orthogonalized single-unit-cell bounds (xlo/xhi/ylo/yhi/...). Used by
            PES scans to size the lateral grid to one surface lattice period.
        lat_c_info_out: Optional mutable dict filled with ``lat_c_source``,
            ``atoms_per_layer`` and ``layer_thickness`` (the last two measured
            on the monolayer before stacking).
        interlayer_general: The run's ``[general]`` config, forwarded to
            :func:`stack_multilayer_sheet` for its interlayer style.

    Returns:
        Tuple of (path, box_dims, lat_c).
    """
    base_path, dims, pot_counts, total_pot_types, supercell_dims = build_monolayer(
        config,
        edge_mode=edge_mode,
        edge_width=edge_width,
        interlayer=interlayer_style(interlayer_general),
    )
    if unit_cell_out is not None:
        unit_cell_out.update({k: float(v) for k, v in supercell_dims.items()})

    n_layers = n_layers_override or (max(config.layers) if config.layers else 1)
    stacked_path = build_dir / f"{config.mat}_{n_layers}.lmp"
    lat_c = config.lat_c
    if lat_c_info_out is not None:
        n_atoms, thickness = atom_count_and_z_extent(base_path)
        lat_c_info_out['atoms_per_layer'] = n_atoms
        lat_c_info_out['layer_thickness'] = thickness
        lat_c_info_out['lat_c_source'] = (
            'config' if config.lat_c is not None else 'not_applicable')
    if not stack_if_multi or n_layers <= 1:
        shutil.copy(base_path, stacked_path)
        return stacked_path, dims, lat_c

    lat_c = stack_multilayer_sheet(
        base_layer_path=base_path,
        config=config,
        output_path=stacked_path,
        box_dims=dims,
        n_layers=n_layers,
        types_per_layer=total_pot_types,
        pot_counts=pot_counts,
        supercell_dims=supercell_dims,
        lat_c=lat_c,
        settings=settings,
        use_pair_bonding=use_pair_bonding,
        stacking_type=stacking_type,
        edge_mode=edge_mode,
        info_out=lat_c_info_out,
        interlayer_general=interlayer_general,
    )
    return stacked_path, dims, lat_c
