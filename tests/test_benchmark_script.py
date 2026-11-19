"""Tests for ``scripts/mlip_bilayer_benchmark.py`` — pure functions only.

The benchmark's job is to drive a real MLIP-enabled LAMMPS binary on a GPU, so
nothing here executes LAMMPS or touches ``nvidia-smi``. Everything that *can*
be wrong without a GPU — the deck text, the log parsing, the parabola fit and
the JSON assembly — is a pure function, and this file pins all four.

The script lives in ``scripts/`` (not a package), so it is loaded by path.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "mlip_bilayer_benchmark.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load(SCRIPT, "mlip_bilayer_benchmark")


# ---------------------------------------------------------------------------
# (a) deck text generation
# ---------------------------------------------------------------------------

def test_deck_deepmd_golden():
    """DeePMD carries the model in the pair_style line (registry contract).

    NOTE the ``pair_coeff`` line has NO ``deepmd`` token. This deck is
    non-hybrid, and outside ``pair_style hybrid`` that leading token is not a
    sub-style selector -- pair_deepmd reads it as an element symbol and aborts
    with "element deepmd not found in the model" (``pair_deepmd.cpp:860``).
    This golden previously pinned the hybrid form and so pinned a deck no
    LAMMPS build accepts; it was caught by actually running the benchmark.
    """
    deck = bench.render_deck(
        framework="deepmd",
        data_file="bilayer_sep6.150.data",
        model_path="model.pth",
    )
    assert deck == (
        "# FrictionSim2D bilayer MoS2 binding benchmark -- rigid single point\n"
        "# framework: deepmd\n"
        "units metal\n"
        "boundary p p p\n"
        "atom_style atomic\n"
        "read_data bilayer_sep6.150.data\n"
        "pair_style deepmd model.pth\n"
        "pair_coeff * * Mo S Mo S\n"
        "neighbor 2.0 bin\n"
        "neigh_modify delay 0 every 1 check yes\n"
        "thermo_style custom step pe\n"
        "thermo 1\n"
        "run 0\n"
    )


def test_deck_mace_golden():
    """symmetrix/MACE: model in pair_coeff, style token dropped when non-hybrid.

    Both forms measured on the real binary 2026-08-24: keeping the
    ``symmetrix/mace`` token outside ``pair_style hybrid`` makes LAMMPS read it
    as the model filename and reject the pair_style.
    """
    deck = bench.render_deck(
        framework="mace",
        data_file="bilayer_sep6.150.data",
        model_path="mace.json",
    )
    assert "pair_style symmetrix/mace\n" in deck
    assert "pair_coeff * * mace.json Mo S Mo S\n" in deck
    # hard requirement of pair_symmetrix_mace.cpp:172
    assert "atom_modify     map yes\n" in deck
    assert "newton          on\n" in deck


def test_deck_d3_overlay_golden():
    """--d3 switches to hybrid/overlay and adds a real element for every type."""
    deck = bench.render_deck(
        framework="deepmd",
        data_file="b.data",
        model_path="model.pth",
        d3=bench.D3Settings(),
    )
    assert (
        "pair_style hybrid/overlay deepmd model.pth "
        "dispersion/d3 bj pbe 16.0 8.0\n"
    ) in deck
    assert "pair_coeff * * deepmd Mo S Mo S\n" in deck
    assert "pair_coeff * * dispersion/d3 Mo S Mo S\n" in deck
    assert "NULL" not in deck


def test_deck_sw_reference_golden():
    """SW reference: one SW instance per layer + interlayer lj/cut."""
    deck = bench.render_deck(
        framework="sw",
        data_file="b.data",
        sw_path="MoS.sw",
    )
    assert "pair_style hybrid sw sw lj/cut 11.0\n" in deck
    assert "pair_coeff * * sw 1 MoS.sw Mo S NULL NULL\n" in deck
    assert "pair_coeff * * sw 2 MoS.sw NULL NULL Mo S\n" in deck
    assert "pair_coeff 1*2 3*4 lj/cut 0.0224 3.13\n" in deck


def test_deck_is_a_rigid_single_point():
    """The binding curve must not relax away the separation being scanned."""
    for deck in (
        bench.render_deck(framework="deepmd", data_file="b.data",
                          model_path="m.pth"),
        bench.render_deck(framework="mace", data_file="b.data", model_path="m.pt"),
        bench.render_deck(framework="sw", data_file="b.data", sw_path="m.sw"),
    ):
        assert deck.rstrip().endswith("run 0")
        assert "minimize" not in deck
        assert "fix" not in deck
        assert "velocity" not in deck


def test_deck_rejects_missing_model_and_d3_on_sw():
    with pytest.raises(ValueError, match="model"):
        bench.render_deck(framework="deepmd", data_file="b.data")
    with pytest.raises(ValueError, match="[Ss][Ww]"):
        bench.render_deck(framework="sw", data_file="b.data")
    with pytest.raises(ValueError, match="full-cell"):
        bench.render_deck(framework="sw", data_file="b.data", sw_path="m.sw",
                          d3=bench.D3Settings())


# ---------------------------------------------------------------------------
# (b) log energy parsing
# ---------------------------------------------------------------------------

REALISTIC_LOG = """LAMMPS (29 Aug 2024 - Update 3)
OMP_NUM_THREADS environment is not set. Defaulting to 1 thread. (src/comm.cpp:98)
  using 1 OpenMP thread(s) per MPI task
Reading data file ...
  orthogonal box = (0 0 0) to (6.36 5.50792157 33.34)
  1 by 1 by 1 MPI processor grid
  reading atoms ...
  24 atoms
  read_data CPU = 0.002 seconds
Loaded model from: model.pth
Neighbor list info ...
  update: every = 1 steps, delay = 0 steps, check = yes
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 8
  ghost atom cutoff = 8
Setting up Verlet run ...
  Unit style    : metal
  Current step  : 0
  Time step     : 0.001
Per MPI rank memory allocation (min/avg/max) = 4.271 | 4.271 | 4.271 Mbytes
   Step         PotEng
         0  -223.15974
Loop time of 1.4e-06 on 1 procs for 0 steps with 24 atoms

99.9% CPU use with 1 MPI tasks x 1 OpenMP threads
Total wall time: 0:00:03
"""


def test_parse_potential_energy_realistic_log():
    assert bench.parse_potential_energy(REALISTIC_LOG) == pytest.approx(-223.15974)


def test_parse_potential_energy_takes_the_last_thermo_row():
    log = REALISTIC_LOG + """Per MPI rank memory allocation (min/avg/max) = 4.2 | 4.2 | 4.2 Mbytes
   Step         PotEng
         0  -223.15974
        20  -223.98765
Loop time of 2.0 on 1 procs for 20 steps with 24 atoms
"""
    assert bench.parse_potential_energy(log) == pytest.approx(-223.98765)


def test_parse_potential_energy_handles_extra_columns():
    log = """Per MPI rank memory allocation (min/avg/max) = 4.2 Mbytes
   Step          Temp          E_pair         PotEng         TotEng
         0      0            -223.2         -223.15974     -223.15974
Loop time of 1.0 on 1 procs for 0 steps with 24 atoms
"""
    assert bench.parse_potential_energy(log) == pytest.approx(-223.15974)


def test_parse_potential_energy_raises_without_thermo_header():
    log = "LAMMPS (29 Aug 2024)\nERROR: Unknown pair style deepmd (src/force.cpp:266)\n"
    with pytest.raises(bench.LogParseError, match="PotEng"):
        bench.parse_potential_energy(log)


def test_parse_potential_energy_raises_when_header_has_no_rows():
    log = """Per MPI rank memory allocation (min/avg/max) = 4.2 Mbytes
   Step         PotEng
ERROR on proc 0: Domain too large for neighbor bins (src/nbin.cpp:97)
"""
    with pytest.raises(bench.LogParseError, match="no numeric"):
        bench.parse_potential_energy(log)


# ---------------------------------------------------------------------------
# (c) quadratic minimum fit
# ---------------------------------------------------------------------------

def test_quadratic_minimum_recovers_analytic_vertex():
    """y = 3(x - 2.5)^2 + 1  ->  vertex (2.5, 1)."""
    xs = [2.0, 2.5, 3.0]
    ys = [3.0 * (x - 2.5) ** 2 + 1.0 for x in xs]
    x_min, y_min = bench.quadratic_minimum(xs, ys)
    assert x_min == pytest.approx(2.5)
    assert y_min == pytest.approx(1.0)


def test_quadratic_minimum_with_asymmetric_samples():
    """The vertex must not be assumed to sit on the sampled grid."""
    xs = [6.0, 6.1, 6.4]
    ys = [0.5 * (x - 6.234) ** 2 - 7.0 for x in xs]
    x_min, y_min = bench.quadratic_minimum(xs, ys)
    assert x_min == pytest.approx(6.234)
    assert y_min == pytest.approx(-7.0)


def test_quadratic_minimum_rejects_non_convex_samples():
    xs = [1.0, 2.0, 3.0]
    ys = [-(x ** 2) for x in xs]
    with pytest.raises(ValueError, match="minimum"):
        bench.quadratic_minimum(xs, ys)


def test_quadratic_minimum_needs_three_points():
    with pytest.raises(ValueError, match="three"):
        bench.quadratic_minimum([1.0, 2.0], [1.0, 2.0])


def test_equilibrium_from_curve_interpolates_between_grid_points():
    """The binding minimum is refined off-grid from its bracketing triple."""
    curve = {round(5.5 + 0.1 * i, 2): 0.4 * (5.5 + 0.1 * i - 6.27) ** 2 - 1.0
             for i in range(36)}
    sep, energy = bench.equilibrium_from_curve(curve)
    assert sep == pytest.approx(6.27, abs=1e-6)
    assert energy == pytest.approx(-1.0, abs=1e-6)


def test_equilibrium_from_curve_falls_back_to_endpoint():
    """A monotonic curve has no interior bracket; report the sampled endpoint."""
    curve = {5.5: -1.0, 5.6: -0.9, 5.7: -0.8}
    sep, energy = bench.equilibrium_from_curve(curve)
    assert sep == pytest.approx(5.5)
    assert energy == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# separations grid + binding energy
# ---------------------------------------------------------------------------

def test_separation_grid_is_5p5_to_9p0_in_0p1_steps():
    seps = bench.separation_grid()
    assert seps[0] == pytest.approx(5.5)
    assert seps[-1] == pytest.approx(9.0)
    assert len(seps) == 36
    assert all(s == round(s, 2) for s in seps)  # no float dust in the keys


def test_binding_energy_per_area():
    # 2 eV of binding spread over 20 A^2 -> 0.1 eV/A^2
    result = bench.binding_energy_per_area(e_min=-10.0, e_reference=-8.0, area=20.0)
    assert result["eV_per_A2"] == pytest.approx(0.1)
    assert result["J_per_m2"] == pytest.approx(0.1 * bench.EV_PER_A2_TO_J_PER_M2)
    assert result["meV_per_atom"] is None


# ---------------------------------------------------------------------------
# (d) JSON assembly
# ---------------------------------------------------------------------------

def test_assemble_results_is_json_serialisable_and_complete():
    curve = {round(5.5 + 0.1 * i, 2): 0.4 * (5.5 + 0.1 * i - 6.27) ** 2 - 1.0
             for i in range(36)}
    results = bench.assemble_results(
        framework="deepmd",
        model="model.pth",
        lmp="/opt/lammps-deepmd/bin/lmp",
        d3=False,
        n_atoms=24,
        area=35.03,
        curve=curve,
        in_plane_curve={3.116: -0.9, 3.18: -1.0, 3.244: -0.95},
        gpu_memory={5.5: 812, 5.6: 830},
    )
    text = json.dumps(results)  # must not raise
    assert json.loads(text)["framework"] == "deepmd"

    assert results["n_atoms"] == 24
    assert results["dispersion"] == "none"
    # {separation: energy} map, string keys so JSON round-trips losslessly
    assert set(results["binding_curve_eV"]) == {f"{s:.2f}" for s in curve}
    assert results["equilibrium_separation_A"] == pytest.approx(6.27, abs=1e-6)
    assert results["binding_energy_per_area"]["eV_per_A2"] > 0
    # in-plane lattice constant from the 3-point scaled-cell fit
    assert 3.1 < results["lat_c_in_plane_A"] < 3.25
    assert results["vram"]["peak_MiB"] == 830
    assert results["vram"]["n_atoms"] == 24


def test_in_plane_fit_rejects_a_vertex_outside_the_sampled_window():
    """Three monotonic points fit a parabola whose vertex is pure extrapolation."""
    curve = {3.1164: 45.80, 3.1800: 36.70, 3.2436: 28.70}
    value, note = bench.fit_in_plane_constant(curve)
    assert value is None
    assert "outside the sampled window" in note


def test_in_plane_fit_accepts_a_bracketed_minimum():
    curve = {3.116: -0.9, 3.18: -1.0, 3.244: -0.95}
    value, note = bench.fit_in_plane_constant(curve)
    assert 3.116 < value < 3.244
    assert note is None


def test_assemble_results_reports_an_unresolved_in_plane_fit():
    results = bench.assemble_results(
        framework="sw", model=None, lmp="lmp", d3=False, n_atoms=24, area=35.03,
        curve={6.0: -1.0, 6.1: -1.2, 6.2: -1.0},
        in_plane_curve={3.1164: 45.80, 3.1800: 36.70, 3.2436: 28.70},
        gpu_memory=None,
    )
    assert results["lat_c_in_plane_A"] is None
    assert "outside the sampled window" in results["lat_c_in_plane_note"]


def test_assemble_results_without_gpu_or_in_plane_scan():
    curve = {6.0: -1.0, 6.1: -1.2, 6.2: -1.0}
    results = bench.assemble_results(
        framework="sw", model=None, lmp="lmp", d3=False, n_atoms=24,
        area=35.03, curve=curve, in_plane_curve=None, gpu_memory=None,
    )
    json.dumps(results)
    assert results["lat_c_in_plane_A"] is None
    assert results["vram"]["available"] is False
    assert results["vram"]["peak_MiB"] is None


def test_assemble_results_marks_d3():
    curve = {6.0: -1.0, 6.1: -1.2, 6.2: -1.0}
    results = bench.assemble_results(
        framework="mace", model="m.pt", lmp="lmp", d3=True, n_atoms=24,
        area=35.03, curve=curve, in_plane_curve=None, gpu_memory=None,
    )
    assert results["dispersion"] == "d3"


# ---------------------------------------------------------------------------
# geometry — consistency with scripts/milestone0/make_geometries.py
# ---------------------------------------------------------------------------

def test_bilayer_geometry_matches_milestone0_at_default_spacing():
    """Same MoS2 recipe as the Milestone-0 decks (orthogonal cell, per-layer types)."""
    pytest.importorskip("ase")
    import numpy as np

    mg = _load(REPO_ROOT / "scripts" / "milestone0" / "make_geometries.py",
               "m0_make_geometries")
    reference = mg._assign_layer_types(mg._stack(2), 2)
    atoms = bench.build_bilayer(separation=mg.INTERLAYER, a=mg.LATTICE_A)

    assert len(atoms) == len(reference) == 24
    assert atoms.get_chemical_symbols() == reference.get_chemical_symbols()
    assert np.allclose(np.array(atoms.cell), np.array(reference.cell))
    assert np.allclose(atoms.positions, reference.positions, atol=1e-9)
    assert list(atoms.arrays["type"]) == list(reference.arrays["type"])
    assert sorted(set(atoms.arrays["type"])) == [1, 2, 3, 4]


def test_build_bilayer_honours_separation_and_in_plane_scale():
    pytest.importorskip("ase")
    import numpy as np

    atoms = bench.build_bilayer(separation=7.25, a=3.18)
    types = np.asarray(atoms.arrays["type"])
    z_mo_lower = atoms.positions[types == 1][:, 2].mean()
    z_mo_upper = atoms.positions[types == 3][:, 2].mean()
    assert z_mo_upper - z_mo_lower == pytest.approx(7.25)

    scaled = bench.build_bilayer(separation=7.25, a=3.18 * 1.02)
    assert scaled.cell[0][0] == pytest.approx(atoms.cell[0][0] * 1.02)
    assert scaled.cell[1][1] == pytest.approx(atoms.cell[1][1] * 1.02)
    # scaling the cell must not change the interlayer separation being scanned
    z_lo = scaled.positions[np.asarray(scaled.arrays["type"]) == 1][:, 2].mean()
    z_hi = scaled.positions[np.asarray(scaled.arrays["type"]) == 3][:, 2].mean()
    assert z_hi - z_lo == pytest.approx(7.25)


def test_stage_potential_returns_a_whitespace_free_basename(tmp_path):
    """LAMMPS splits deck lines on whitespace, so pair_coeff needs a basename."""
    source_dir = tmp_path / "ML Analysis" / "potentials"
    source_dir.mkdir(parents=True)
    pot = source_dir / "MoS2_wen.sw"
    pot.write_text("sw params\n", encoding="utf-8")
    workdir = tmp_path / "bench"
    workdir.mkdir()

    name = bench.stage_potential(str(pot), workdir)
    assert name == "MoS2_wen.sw"
    assert " " not in name
    staged = workdir / name
    assert staged.exists() and staged.read_text() == "sw params\n"

    # idempotent: re-staging the same file must not fail
    assert bench.stage_potential(str(pot), workdir) == "MoS2_wen.sw"


def test_stage_potential_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        bench.stage_potential(str(tmp_path / "absent.pth"), tmp_path)


def test_in_plane_area_is_the_cell_cross_section():
    pytest.importorskip("ase")
    atoms = bench.build_bilayer(separation=6.15, a=3.18)
    assert bench.in_plane_area(atoms) == pytest.approx(
        atoms.cell[0][0] * atoms.cell[1][1]
    )
    assert math.isclose(bench.in_plane_area(atoms), 6.36 * 5.50792157, rel_tol=1e-6)
