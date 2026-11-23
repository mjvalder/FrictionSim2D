"""Tests for the shaped-flake AFM use case.

Covers the geometry predicates, flake cutting (build_flake), corner detection +
retyping (apply_flake_corner_types), the FlakeConfig model, and the LAMMPS
template wiring (corner restraint present during indentation, absent in slide).
"""

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from frictionsim2d.builders import components
from frictionsim2d.builders.afm import AFMSimulation
from frictionsim2d.builders.utils.lammps_data import parse_lmp_atoms
from frictionsim2d.core.config import AFMSimulationConfig, FlakeConfig, load_settings


# ---------------------------------------------------------------------------
# Geometry predicates
# ---------------------------------------------------------------------------

def test_geometry_predicates_membership():
    # Center is always inside; far-away points are outside.
    assert components._in_square(0.0, 0.0, 10.0)
    assert not components._in_square(100.0, 0.0, 10.0)
    assert components._in_equilateral_triangle(0.0, 0.0, 10.0)
    assert not components._in_equilateral_triangle(0.0, 100.0, 10.0)
    assert components._in_hexagon(0.0, 0.0, 10.0)
    assert not components._in_hexagon(0.0, 100.0, 10.0)


def _outline_vertices(inside, r_max: float, n_theta: int = 7200):
    """Vertices of the convex region ``inside`` carves, recovered numerically.

    Bisects the boundary radius r(theta) along ``n_theta`` rays from the
    origin; the polygon's vertices are the strict local maxima of r(theta).
    Independent of how the predicate is written: an octagon shows 8 maxima.
    """
    thetas = [2.0 * math.pi * k / n_theta for k in range(n_theta)]
    radii = []
    for th in thetas:
        c, s = math.cos(th), math.sin(th)
        lo, hi = 0.0, r_max
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if inside(mid * c, mid * s):
                lo = mid
            else:
                hi = mid
        radii.append(lo)
    verts = []
    for k in range(n_theta):
        r_prev, r, r_next = radii[k - 1], radii[k], radii[(k + 1) % n_theta]
        if r > r_prev and r >= r_next:
            verts.append((r * math.cos(thetas[k]), r * math.sin(thetas[k])))
    return verts


def _polygon_edges_and_angles(verts):
    """Edge lengths and interior angles (deg) of a polygon in CCW order."""
    n = len(verts)
    edges = [math.dist(verts[i], verts[(i + 1) % n]) for i in range(n)]
    angles = []
    for i in range(n):
        ax, ay = verts[i - 1][0] - verts[i][0], verts[i - 1][1] - verts[i][1]
        bx, by = verts[(i + 1) % n][0] - verts[i][0], verts[(i + 1) % n][1] - verts[i][1]
        cos_t = (ax * bx + ay * by) / (math.hypot(ax, ay) * math.hypot(bx, by))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cos_t)))))
    return edges, angles


def _shoelace(verts):
    n = len(verts)
    return 0.5 * abs(sum(verts[i][0] * verts[(i + 1) % n][1]
                         - verts[(i + 1) % n][0] * verts[i][1] for i in range(n)))


@pytest.mark.parametrize("rotation_deg", [0.0, 17.0, 45.0])
def test_hexagon_carves_a_regular_hexagon(rotation_deg):
    """`_in_hexagon` is the regular hexagon of circumradius R its docstring
    promises -- 6 equal edges (= R), 120 deg angles, area 3 sqrt(3)/2 R^2 --
    vertex-on-x at rotation 0, rotated rigidly by rotation_deg. It used to be
    an irregular octagon (oblique strip normals at +-60 deg, not +-30)."""
    R = 10.0
    verts = _outline_vertices(
        lambda x, y: components._in_hexagon(x, y, R, (0.0, 0.0), rotation_deg), 2 * R)
    assert len(verts) == 6
    edges, angles = _polygon_edges_and_angles(verts)
    assert edges == pytest.approx([R] * 6, rel=2e-3)
    assert angles == pytest.approx([120.0] * 6, abs=0.3)
    assert _shoelace(verts) == pytest.approx(3 * math.sqrt(3) / 2 * R * R, rel=2e-3)
    # Vertex-on-x convention, rotated CCW by rotation_deg.
    angles_deg = sorted((math.degrees(math.atan2(y, x)) - rotation_deg) % 60.0
                        for x, y in verts)
    for a in angles_deg:
        assert min(a, 60.0 - a) < 0.2, angles_deg


@pytest.mark.parametrize("rotation_deg", [0.0, 17.0, 45.0, 90.0])
def test_triangle_is_centred_equilateral_and_honours_rotation(rotation_deg):
    """`_in_equilateral_triangle` is centred on its centroid (the origin), has
    side a and area sqrt(3)/4 a^2, and rotation_deg rotates it (apex at
    90 + rotation_deg). It used to sit with its centroid at -h/3 and ignore
    rotation entirely."""
    a = 10.0
    verts = _outline_vertices(
        lambda x, y: components._in_equilateral_triangle(x, y, a, (0.0, 0.0), rotation_deg),
        2 * a)
    assert len(verts) == 3
    edges, angles = _polygon_edges_and_angles(verts)
    assert edges == pytest.approx([a] * 3, rel=2e-3)
    assert angles == pytest.approx([60.0] * 3, abs=0.3)
    assert _shoelace(verts) == pytest.approx(math.sqrt(3) / 4 * a * a, rel=2e-3)
    cx = sum(v[0] for v in verts) / 3
    cy = sum(v[1] for v in verts) / 3
    assert abs(cx) < 1e-2 and abs(cy) < 1e-2
    apex = 90.0 + rotation_deg
    got = [math.degrees(math.atan2(y, x)) for x, y in verts]
    assert min(abs((g - apex + 180.0) % 360.0 - 180.0) for g in got) < 0.2, got


def test_triangle_rotation_moves_a_vertex():
    a = 10.0
    h = a * math.sqrt(3) / 2
    apex = (0.0, 2 * h / 3 * 0.99)
    assert components._in_equilateral_triangle(*apex, a)
    assert not components._in_equilateral_triangle(*apex, a, (0.0, 0.0), 60.0)


def test_build_flake_triangle_honours_rotation_deg(tmp_path: Path):
    """build_flake passes rotation_deg to the triangle (it was silently dropped)."""
    src = tmp_path / "super.lmp"
    _write_supercell(src, span=60.0, spacing=1.0)
    carved = {}
    for rot in (0.0, 60.0):
        out = tmp_path / f"tri_{rot}.data"
        components.build_flake(src, out, shape="triangle", edge_length=30.0,
                               rotation_deg=rot)
        _t, xs, ys, _z = parse_lmp_atoms(out)
        carved[rot] = (float(ys.max() - 30.0), float(ys.min() - 30.0))
    h = 30.0 * math.sqrt(3) / 2
    # Rotation 0: apex up at +2h/3, base at -h/3; 60 deg flips it.
    assert carved[0.0][0] == pytest.approx(2 * h / 3, abs=1.0)
    assert carved[0.0][1] == pytest.approx(-h / 3, abs=1.0)
    assert carved[60.0][0] == pytest.approx(h / 3, abs=1.0)
    assert carved[60.0][1] == pytest.approx(-2 * h / 3, abs=1.0)


def test_square_rotation_is_symmetric():
    # A point on the +x axis just inside an axis-aligned square edge stays inside
    # after a 90° rotation (square maps onto itself).
    half = 5.0
    assert components._in_square(half - 0.1, 0.0, 2 * half, (0.0, 0.0), 0.0)
    assert components._in_square(half - 0.1, 0.0, 2 * half, (0.0, 0.0), 90.0)


# ---------------------------------------------------------------------------
# Synthetic supercell helper
# ---------------------------------------------------------------------------

def _write_supercell(path: Path, span: float = 60.0, spacing: float = 2.0) -> int:
    """Write a single-type square-lattice atomic LAMMPS file; return atom count."""
    coords = []
    n = int(span / spacing) + 1
    for i in range(n):
        for j in range(n):
            coords.append((i * spacing, j * spacing))
    header = [
        "Test supercell", "",
        f"{len(coords)} atoms", "0 bonds", "0 angles", "0 dihedrals", "",
        "1 atom types", "",
        f"0.0 {span} xlo xhi", f"0.0 {span} ylo yhi", "-1.0 1.0 zlo zhi", "",
        "Masses", "", "1 95.94  # Mo", "",
        "Atoms # atomic", "",
    ]
    body = [f"{k+1} 1 {x:.4f} {y:.4f} 0.0" for k, (x, y) in enumerate(coords)]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return len(coords)


@pytest.mark.parametrize("shape,expected_vertices", [
    ("triangle", 3), ("square", 4), ("hexagon", 6),
])
def test_build_flake_cuts_shape(tmp_path: Path, shape: str, expected_vertices: int):
    src = tmp_path / "super.lmp"
    n_full = _write_supercell(src)

    out = tmp_path / f"flake_{shape}.data"
    components.build_flake(src, out, shape=shape, edge_length=20.0)

    types, xs, ys, _zs = parse_lmp_atoms(out)
    # The cut keeps a strict subset of atoms and at least a handful.
    assert 0 < len(types) < n_full
    assert int(types.max()) == 1


@pytest.mark.parametrize("shape", ["triangle", "square", "hexagon"])
def test_apply_flake_corner_types_interleaves(tmp_path: Path, shape: str):
    src = tmp_path / "super.lmp"
    _write_supercell(src)
    out = tmp_path / f"flake_{shape}.data"
    components.build_flake(src, out, shape=shape, edge_length=24.0)

    components.apply_flake_corner_types(out, shape=shape, corner_radius=5.0)
    types, _xs, _ys, _zs = parse_lmp_atoms(out)

    # One base type -> interleaved core(1)/corner(2); corner atoms are the even type.
    assert int(types.max()) == 2
    n_corner = int((types % 2 == 0).sum())
    n_core = int((types % 2 == 1).sum())
    assert n_corner > 0, "expected at least some corner atoms"
    assert n_core > 0, "expected at least some core atoms"
    # Corners should be the minority for a reasonably sized flake.
    assert n_corner < n_core


def test_detect_corner_atoms_free_in_z_unchanged(tmp_path: Path):
    """Corner detection / retyping must not move atoms (z stays free for indent)."""
    src = tmp_path / "super.lmp"
    _write_supercell(src)
    out = tmp_path / "flake.data"
    components.build_flake(src, out, shape="square", edge_length=24.0)
    before = parse_lmp_atoms(out)
    components.apply_flake_corner_types(out, shape="square", corner_radius=5.0)
    after = parse_lmp_atoms(out)
    # Same atom count and identical coordinates (only the type column changed).
    assert len(before[0]) == len(after[0])
    assert (before[1] == after[1]).all()  # x
    assert (before[3] == after[3]).all()  # z


# ---------------------------------------------------------------------------
# FlakeConfig
# ---------------------------------------------------------------------------

@pytest.fixture
def real_files(tmp_path: Path):
    pot = tmp_path / "dummy.sw"
    cif = tmp_path / "dummy.cif"
    pot.write_text("# pot\n", encoding="utf-8")
    cif.write_text("# cif\n", encoding="utf-8")
    return str(pot), str(cif)


def test_flake_config_autosizes_supercell(real_files):
    pot, cif = real_files
    fc = FlakeConfig(mat="MoS2", pot_type="sw", pot_path=pot, cif_path=cif,
                     edge_length=30.0, corner_radius=5.0)
    assert fc.shape == "hexagon"
    # span = 2*edge_length + max(2*corner_radius, 10)
    assert fc.x == fc.y == 2 * 30.0 + 10.0


def test_afm_config_without_flake_is_none(real_files):
    pot, cif = real_files
    data = {
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif, 'r': 10.0},
        'sub': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif,
                'thickness': 10.0, 'amorph': 'a'},
        '2D': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif,
               'x': 50.0, 'y': 50.0, 'layers': [1]},
        'settings': load_settings().model_dump(),
    }
    cfg = AFMSimulationConfig(**data)
    assert cfg.flake is None


def test_afm_config_parses_flake_section(real_files):
    pot, cif = real_files
    data = {
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif, 'r': 10.0},
        'sub': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif,
                'thickness': 10.0, 'amorph': 'a'},
        '2D': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif,
               'x': 50.0, 'y': 50.0, 'layers': [1]},
        'flake': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif,
                  'shape': 'square', 'edge_length': 25.0, 'corner_spring_k': 12.0},
        'settings': load_settings().model_dump(),
    }
    cfg = AFMSimulationConfig(**data)
    assert cfg.flake is not None
    assert cfg.flake.shape == 'square'
    assert cfg.flake.edge_length == 25.0
    assert cfg.flake.corner_spring_k == 12.0


# ---------------------------------------------------------------------------
# Template wiring (corner restraint present in indentation, absent in slide)
# ---------------------------------------------------------------------------

def _make_flake_builder(tmp_path: Path) -> AFMSimulation:
    pot = tmp_path / "dummy.sw"
    cif = tmp_path / "dummy.cif"
    pot.write_text("# pot\n", encoding="utf-8")
    cif.write_text("# cif\n", encoding="utf-8")
    data = {
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(pot), 'cif_path': str(cif), 'r': 10.0},
        'sub': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(pot), 'cif_path': str(cif),
                'thickness': 10.0, 'amorph': 'a'},
        '2D': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': str(pot), 'cif_path': str(cif),
               'x': 50.0, 'y': 50.0, 'layers': [1]},
        'flake': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': str(pot), 'cif_path': str(cif),
                  'shape': 'hexagon', 'edge_length': 20.0, 'corner_spring_k': 9.0},
        'settings': load_settings().model_dump(),
    }
    config = AFMSimulationConfig(**data)
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))

    # Hand-populate the state write_inputs() consumes (bypassing real builds).
    builder.flake_path = tmp_path / "flake_h-MoS2_hexagon.data"
    builder.flake_dims = {'xlo': 0.0, 'xhi': 50.0, 'ylo': 0.0, 'yhi': 50.0, 'zlo': -1.0, 'zhi': 4.0}
    builder.sheet_dims = {'xlo': 0.0, 'xhi': 50.0, 'ylo': 0.0, 'yhi': 50.0, 'zlo': 0.0, 'zhi': 3.0}
    builder.box_dims = {'xlo': 0.0, 'xhi': 70.0, 'ylo': 0.0, 'yhi': 70.0, 'zlo': -5.0, 'zhi': 0.0}
    builder.sheet_offset = {'x': 0.0, 'y': 0.0}
    builder.lat_c = 6.0
    builder.sheet_paths[1] = tmp_path / "sheet.lmp"
    builder.tip_path = tmp_path / "tip.lmp"
    builder.sub_path = tmp_path / "sub.lmp"
    # One atom each: build_render_context sizes the z box from them.
    for path in (builder.flake_path, builder.sheet_paths[1], builder.tip_path,
                 builder.sub_path):
        path.write_text("# dummy\n\nAtoms # atomic\n\n1 1 0.0 0.0 0.0\n", encoding="utf-8")
    builder.output_dir_layer[1] = builder.output_dir / "L1"
    builder.relative_run_dir_layer[1] = builder.relative_run_dir / "L1"
    (builder.output_dir_layer[1] / "lammps").mkdir(parents=True, exist_ok=True)

    builder.z_positions[1] = {'sub': 0.0, 'sheet': 12.0, 'flake': 18.0, 'tip': 40.0, 'tip_contact_gap': 3.5}
    builder.groups[1] = {
        'sub_types': '1 2',
        'tip_types': '3',
        'sheet_types': '4 5',
        'flake_types': '6 7 8 9',
        'flake_corner_types': '7 9',
    }
    fake_pm = MagicMock()
    fake_pm.types.__len__.return_value = 9
    fake_pm.max_pair_cutoff.return_value = 11.0
    builder.pm[1] = fake_pm
    return builder


def test_system_init_has_corner_restraint(tmp_path: Path):
    builder = _make_flake_builder(tmp_path)
    builder.write_inputs(1)

    system_in = (builder.output_dir_layer[1] / "lammps" / "system.in").read_text()
    # Flake is read into the indentation system.
    assert "group flake offset" in system_in
    assert "flake_h-MoS2_hexagon.data" in system_in
    # Corner atoms tethered in xy (free in z) with the configured stiffness.
    assert "group           flake_corner type 7 9" in system_in
    assert "fix             corner_hold flake_corner spring/self 9.0 xy" in system_in


def test_slide_has_no_corner_restraint(tmp_path: Path):
    builder = _make_flake_builder(tmp_path)
    builder.write_inputs(1)

    slide_in = (builder.output_dir_layer[1] / "lammps" / "slide.in").read_text()
    # The flake rides into the load data file; sliding must not re-tether corners.
    assert "corner_hold" not in slide_in
    assert "flake_corner" not in slide_in


# ---------------------------------------------------------------------------
# PBC self-interaction guard
# ---------------------------------------------------------------------------

def test_flake_atom_span_measures_the_atoms_not_the_box(tmp_path: Path):
    """The carved file keeps the SUPERCELL box in x/y -- the flake's own span
    is only in its atom coordinates, and it is the span that has to clear the
    periodic image."""
    src = tmp_path / "super.lmp"
    _write_supercell(src, span=60.0, spacing=1.0)
    out = tmp_path / "hex.data"
    components.build_flake(src, out, shape="hexagon", edge_length=10.0)

    span_x, span_y = components.flake_atom_span(out)
    # hexagon circumradius 10 -> 20 A vertex-to-vertex, sqrt(3)*10 across flats
    assert span_x == pytest.approx(20.0, abs=1.5)
    assert span_y == pytest.approx(10.0 * math.sqrt(3), abs=1.5)
    assert span_x < 60.0 and span_y < 60.0        # not the box


def test_a_flake_clear_of_its_image_passes():
    # 19.1 A flake in a 50.96 x 44.13 cell: the r=10 row of the measured table
    components.check_flake_pbc_clearance(
        (19.10, 16.54), (50.96, 44.13), lj_cutoff=11.0)


def test_a_flake_inside_the_cutoff_of_its_image_is_refused():
    # the same flake in the r=5 cell: 6.38 x 5.52 A to its own image
    with pytest.raises(ValueError, match=r"periodic image"):
        components.check_flake_pbc_clearance(
            (19.10, 16.54), (25.48, 22.07), lj_cutoff=11.0)


def test_the_refusal_names_the_axis_and_the_numbers():
    with pytest.raises(ValueError) as excinfo:
        components.check_flake_pbc_clearance(
            (19.10, 5.0), (25.48, 44.13), lj_cutoff=11.0)
    message = str(excinfo.value)
    assert "x" in message and "6.38" in message and "11.0" in message


def test_a_flake_wider_than_the_cell_is_refused():
    with pytest.raises(ValueError, match=r"periodic image"):
        components.check_flake_pbc_clearance(
            (60.0, 10.0), (25.48, 44.13), lj_cutoff=11.0)


def test_the_afm_builder_checks_its_flake_against_the_sheet_cell(tmp_path: Path):
    """The guard is wired into the builder, not just available as a function:
    a real carved flake against a too-small sheet is refused."""
    src = tmp_path / "super.lmp"
    _write_supercell(src, span=60.0, spacing=1.0)
    flake = tmp_path / "flake.data"
    components.build_flake(src, flake, shape="hexagon", edge_length=10.0)

    sim = MagicMock()
    sim.flake_path = flake
    sim.sheet_dims = {'xlo': 0.0, 'xhi': 25.0, 'ylo': 0.0, 'yhi': 25.0}
    sim.config.settings.potential.lj_cutoff = 11.0
    # the cell comes from the real accessor (the hetero builder overrides it)
    sim._flake_periodic_cell = lambda: AFMSimulation._flake_periodic_cell(sim)

    with pytest.raises(ValueError, match=r"periodic image"):
        AFMSimulation._check_flake_fits_the_cell(sim)

    sim.sheet_dims = {'xlo': 0.0, 'xhi': 60.0, 'ylo': 0.0, 'yhi': 60.0}
    AFMSimulation._check_flake_fits_the_cell(sim)      # clears at 60 A
