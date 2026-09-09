"""Tests for the hetero (two-material) sheet-on-sheet static PES scan.

Config fixtures mirror ``tests/builders/test_hetero_slide.py``'s ``_hetero_2p2_config``
(two ``[2D-N]`` sections, ``sw`` potentials) but with one layer per material
(the 1+1 PES bilayer: frozen bottom + laterally scanned top) plus a ``[pes]``
section as in ``tests/builders/test_pes_scan_builder.py``.

Covers the MRO/``SPLIT_OUTER_LOOPS`` resolution, routing, the relaxed layer
gate (and that the *dynamic* hetero slide keeps its own 4-layer gate), and a
real-geometry build smoke asserting that the emitted scan script uses the
hetero data file, z-band layer groups, and a scan period equal to the matched
interface's **minimal in-plane repeat** -- not the assembled box, which is a
4x2 superlattice of it here, and whose use also mis-normalized the energy.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from src.builders import components
from src.core.config import SheetOnSheetSimulationConfig, load_settings

MAT, POT = "examples/materials", "examples/potentials/sw"
LMP = shutil.which("lmp_serial")


def _hetero_pes_raw(layers=(1, 1), grid_n=4):
    return {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
                 "pot_path": f"{POT}/MoS2_wen.sw", "pot_type": "sw",
                 "x": 12.0, "y": 12.0, "layers": [layers[0]]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif",
                 "pot_path": f"{POT}/sw_lammps/t-WS2.sw", "pot_type": "sw",
                 "x": 12.0, "y": 12.0, "layers": [layers[1]]},
        "pes": {"grid_n": grid_n},
    }


def _settings():
    """Settings a full ``build()`` can complete on.

    ``hpc.modules`` gates job-script generation, which ``build()`` runs and
    does not swallow the failure of -- see ``test_hetero_slide``.
    """
    settings = load_settings()
    settings.hpc.modules = ["lammps"]
    return settings


def _hetero_pes_config(layers=(1, 1), grid_n=4):
    return SheetOnSheetSimulationConfig(**_hetero_pes_raw(layers, grid_n),
                                        settings=_settings())


def _single_material_pes_config():
    raw = {
        "general": {"temp": 300, "scan_speed": 1},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
               "pot_path": f"{POT}/MoS2_wen.sw", "pot_type": "sw",
               "x": 12.0, "y": 12.0, "layers": [2]},
        "pes": {"grid_n": 4},
    }
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


def _twist_pes_config():
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "twist_angle": 21.8},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
               "pot_path": f"{POT}/MoS2_wen.sw", "pot_type": "sw",
               "x": 4.0, "y": 4.0, "layers": [2]},
        "pes": {"grid_n": 4},
    }
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


# --- MRO / SPLIT_OUTER_LOOPS resolution --------------------------------------

def test_hetero_pes_sheet_mro_and_split_outer_loops():
    """Pin the MRO/``SPLIT_OUTER_LOOPS`` resolution.

    ``PESGridMixin`` must resolve before ``HeteroSheetOnSheetSimulation`` so the
    run-phase identity is the static scan's, and ``SPLIT_OUTER_LOOPS`` must stay
    ``frozenset()``: the scan replaces the run phase outright and never reads
    ``outer_loop``, so a non-empty value would make the GUI's job-count
    prediction (``src.run.expand._run_scripts_per_build``) promise scripts the
    build never writes.
    """
    from src.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation
    from src.builders.sheetonsheet.pes_sheet import (
        HeteroPESSheetSimulation,
        PESGridMixin,
        PESSheetSimulation,
    )

    mro = [c.__name__ for c in HeteroPESSheetSimulation.__mro__]
    print("HeteroPESSheetSimulation MRO:", mro)
    print("HeteroPESSheetSimulation.SPLIT_OUTER_LOOPS:",
          HeteroPESSheetSimulation.SPLIT_OUTER_LOOPS)
    assert mro.index("PESGridMixin") < mro.index("HeteroSheetOnSheetSimulation")
    assert HeteroPESSheetSimulation.SPLIT_OUTER_LOOPS == frozenset()
    assert PESSheetSimulation.SPLIT_OUTER_LOOPS == frozenset()
    assert PESGridMixin.SPLIT_OUTER_LOOPS == frozenset()

    # The homogeneous class keeps its own composition too.
    homo_mro = [c.__name__ for c in PESSheetSimulation.__mro__]
    assert homo_mro.index("PESGridMixin") < homo_mro.index("SheetOnSheetSimulation")


def test_hetero_pes_sheet_run_script_count_is_one_per_build():
    """``outer_loop`` must not multiply the run scripts for the hetero PES scan,
    while ``pes.grid_n_refine`` must still add exactly one refinement script."""
    from src.run.expand import _run_scripts_per_build

    raw = _hetero_pes_raw()
    raw["general"]["outer_loop"] = "pressure"
    raw["general"]["pressure"] = [1.0, 2.0, 3.0]
    scripts, notes = _run_scripts_per_build(raw, "pes_sheet")
    assert scripts == 1, notes

    raw["pes"]["grid_n_refine"] = 20
    scripts, notes = _run_scripts_per_build(raw, "pes_sheet")
    assert scripts == 2, notes


# --- routing -----------------------------------------------------------------

def test_routing_selects_hetero_pes_sheet_for_two_materials():
    from src.builders.sheetonsheet.pes_sheet import HeteroPESSheetSimulation
    from src.run.simulate import _select_sheet_builder_cls
    cfg = _hetero_pes_config()
    assert _select_sheet_builder_cls("pes_sheet", cfg) is HeteroPESSheetSimulation


def test_routing_selects_hetero_pes_sheet_for_single_material_twist():
    from src.builders.sheetonsheet.pes_sheet import HeteroPESSheetSimulation
    from src.run.simulate import _select_sheet_builder_cls
    cfg = _twist_pes_config()
    assert len(cfg.sheets) == 1 and cfg.general.twist_angle is not None
    assert _select_sheet_builder_cls("pes_sheet", cfg) is HeteroPESSheetSimulation


def test_routing_keeps_homogeneous_pes_sheet_for_single_material():
    from src.builders.sheetonsheet.pes_sheet import PESSheetSimulation
    from src.run.simulate import _select_sheet_builder_cls
    cfg = _single_material_pes_config()
    assert _select_sheet_builder_cls("pes_sheet", cfg) is PESSheetSimulation


def test_routing_hetero_slide_unchanged_for_sheetonsheet():
    """The dynamic model must keep routing to the plain hetero slide builder."""
    from src.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation
    from src.run.simulate import _select_sheet_builder_cls
    cfg = _hetero_pes_config(layers=(2, 2))
    assert _select_sheet_builder_cls("sheetonsheet", cfg) is HeteroSheetOnSheetSimulation


# --- layer gate ---------------------------------------------------------------

def test_hetero_pes_sheet_min_layers_relaxed_to_two():
    from src.builders.sheetonsheet.pes_sheet import HeteroPESSheetSimulation
    assert HeteroPESSheetSimulation.MIN_LAYERS == 2


def test_dynamic_hetero_slide_keeps_four_layer_gate(tmp_path):
    """Relaxing the gate for the PES variant must NOT relax it for the dynamic
    slide: a 1+1 dynamic hetero slide is physically meaningless (no fixed
    bottom + thermostatted middle + driven top) and must still be refused with
    the existing message."""
    from src.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation
    assert HeteroSheetOnSheetSimulation.MIN_LAYERS == 4
    sim = HeteroSheetOnSheetSimulation(_hetero_pes_config(), str(tmp_path))
    with pytest.raises(ValueError, match=r"[Aa]t least 4"):
        sim.build()


def test_hetero_pes_sheet_rejects_non_bilayer(tmp_path):
    """More than one layer per material is not a PES bilayer."""
    from src.builders.sheetonsheet.pes_sheet import HeteroPESSheetSimulation
    sim = HeteroPESSheetSimulation(_hetero_pes_config(layers=(2, 2)), str(tmp_path))
    with pytest.raises(ValueError, match="exactly 2 layers"):
        sim.build()


# --- job name ------------------------------------------------------------------

def test_hetero_pes_sheet_job_name_carries_pair_token():
    """Two pairs sharing material 1 must not collide on the HPC job name."""
    from src.builders.sheetonsheet.pes_sheet import HeteroPESSheetSimulation, PESSheetSimulation

    raw_b = _hetero_pes_raw()
    raw_b["2D-2"]["mat"] = "h-WSe2"
    cfg_b = SheetOnSheetSimulationConfig(**raw_b, settings=load_settings())

    sim_a = HeteroPESSheetSimulation(_hetero_pes_config(), "out_a")
    sim_b = HeteroPESSheetSimulation(cfg_b, "out_b")
    assert sim_a._get_hpc_job_name() == "pes_sheet_h-MoS2+h-WS2"
    assert sim_b._get_hpc_job_name() == "pes_sheet_h-MoS2+h-WSe2"

    # The homogeneous job name is unchanged by the extraction.
    homo = PESSheetSimulation(_single_material_pes_config(), "out_h")
    assert homo._get_hpc_job_name() == "pes_sheet_h-MoS2"


# --- real-geometry build smoke -------------------------------------------------

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_pes_sheet_build_emits_scan_over_interface_repeat(tmp_path):
    """Build a real 1+1 MoS2/WS2 hetero bilayer and check the emitted scan.

    Asserts the four seams this variant exists for:

    1. the run phase is the static scan (grid loop over ``layer_2``), not the
       dynamic slide (no drive/thermostat);
    2. the data file is the hetero stack's RUN-DIR-RELATIVE ``hetero.data``,
       not the homogeneous sim-root-relative ``.../build/<sheet>.lmp``;
    3. layer groups come from z-band regions (hetero layers share atom types);
    4. the scan period is the matched interface's COINCIDENCE cell.
    """
    from src.builders.sheetonsheet.pes_sheet import HeteroPESSheetSimulation

    cfg = _hetero_pes_config()
    sim = HeteroPESSheetSimulation(cfg, str(tmp_path))
    sim.build()

    slide = tmp_path / "lammps" / "slide.in"
    assert slide.exists()
    text = slide.read_text(encoding="utf-8")

    # (1) static scan, not the dynamic slide
    assert "displace_atoms  layer_2 move" in text
    assert "aveforce" not in text and "langevin" not in text
    assert "create_atoms" not in text          # no virtual drive atom

    # (2) the assembled hetero stack, named relative to `relative_run_dir`
    #     (here the absolute output_dir, since no simulation root was set) --
    #     NOT the homogeneous builder's build/<sheet>.lmp.
    assert f"read_data       {tmp_path}/hetero.data" in text
    assert "build/" not in text
    assert (tmp_path / "hetero.data").exists()
    assert f"include         {tmp_path}/lammps/system.in.settings" in text
    assert f"{tmp_path}/results/pes_scan.csv" in text

    # (3) z-band region groups (hetero layers share atom types)
    assert "group           layer_1 region reg_layer_1" in text
    assert "group           layer_2 region reg_layer_2" in text

    # (4) scan period == coincidence cell of the matched interface
    def _extract(var):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"variable        {var}"):
                return float(stripped.split("equal", 1)[1].strip())
        raise AssertionError(f"variable {var} not found in slide.in")

    period_x, period_y = _extract("Lx"), _extract("Ly")

    # Independent expectation for THIS EXACT fixture (h-MoS2 + h-WS2, both
    # x=y=12 A, `sw`): the interface's MINIMAL in-plane repeat, hardcoded here
    # on purpose -- NOT read back from sim._hetero_coincidence_cell or
    # sim.sheet_unit_cell. Comparing the emitted period against the attribute it
    # was rendered from proves nothing: the renderer faithfully emits whatever
    # that attribute holds, so a bug that stored the wrong cell would still pass.
    #
    # This is NOT the coincidence cell (12.740025 x 11.033185, which an earlier
    # revision of this test pinned). build_monolayer tiles each material to the
    # configured footprint BEFORE the match runs, so this near-lattice-matched
    # pair matches with (near-)identity matrices on already-tiled cells and the
    # box comes out a 4x2 superlattice of the true repeat. Same value pinned
    # independently in tests/builders/test_hetero_pes_tip.py for the same pair.
    EXPECTED_LX, EXPECTED_LY = 3.18500625, 5.516592573245964
    assert period_x == pytest.approx(EXPECTED_LX, rel=1e-6)
    assert period_y == pytest.approx(EXPECTED_LY, rel=1e-6)

    # Both constituents' strained cells ARE a periodicity of this interface --
    # find_coincidence_lattice strains both onto a shared MEAN cell, so the
    # repeat is that mean. Derived here from the constituents, independently of
    # any builder state, as a second route to the same number.
    own_cells = []
    for sheet_cfg in cfg.sheets:
        own_dir = tmp_path / f"own_{sheet_cfg.mat}"
        own_dir.mkdir(exist_ok=True)
        unit_cell_out = {}
        components.build_sheet(
            sheet_cfg, own_dir, settings=cfg.settings,
            unit_cell_out=unit_cell_out,
        )
        own_cells.append((unit_cell_out["xhi"] - unit_cell_out["xlo"],
                          unit_cell_out["yhi"] - unit_cell_out["ylo"]))
    assert period_x == pytest.approx(
        sum(c[0] for c in own_cells) / len(own_cells), rel=1e-6)
    assert period_y == pytest.approx(
        sum(c[1] for c in own_cells) / len(own_cells), rel=1e-6)

    # The energy is normalized per repeat unit, not per box: 4x2 = 8 of them
    # tile this box. The old hardcoded 1 reported a per-BOX energy that scaled
    # with the configured footprint -- 8x too large here, 12x at footprint 14 --
    # so the same physical pair produced different barriers.
    assert _extract("ncell") == pytest.approx(8.0)

    # The box is an exact integer multiple of the emitted period, so displacing
    # the top layer by one period is exactly one lattice translation.
    at = read(str(tmp_path / "hetero.data"), format="lammps-data")
    cell = np.array(at.cell)
    assert float(cell[0, 0]) / period_x == pytest.approx(4.0, rel=1e-9)
    assert float(cell[1, 1]) / period_y == pytest.approx(2.0, rel=1e-9)

    assert sim._get_hpc_job_name() == "pes_sheet_h-MoS2+h-WS2"


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_pes_sheet_scan_runs_in_lammps(tmp_path):
    """Capstone smoke: the emitted scan is valid LAMMPS and actually runs.

    Every path in the script -- ``hetero.data``, ``lammps/system.in.settings``,
    the settings file's staged ``provenance/potentials/...``, the results CSV --
    is written relative to ``relative_run_dir``, which for a builder used
    without ``set_base_output_dir`` is the absolute ``output_dir``. So the
    script resolves from any ``cwd``; the sim-root-relative form the HPC job
    scripts get is covered in ``test_hetero_slide``. Checks a real corrugation
    was written to ``results/``.
    """
    import subprocess

    from src.builders.sheetonsheet.pes_sheet import HeteroPESSheetSimulation

    raw = _hetero_pes_raw(grid_n=3)
    # The matched supercell must exceed the hetero cross-LJ cutoff
    # (``pair_style hybrid ... lj/cut 11.0``) or ``comm_style tiled`` aborts with
    # "Communication cutoff ... cannot exceed periodic box length" -- same
    # constraint as the dynamic hetero slide e2e (tests/builders/test_hetero_slide.py):
    # a 12 A target gives an ~11.03 A edge (too tight), 14 A gives ~12.7 A.
    for key in ("2D-1", "2D-2"):
        raw[key]["x"] = raw[key]["y"] = 14.0
    cfg = SheetOnSheetSimulationConfig(**raw, settings=_settings())

    sim = HeteroPESSheetSimulation(cfg, str(tmp_path))
    sim.build()

    proc = subprocess.run(
        ["lmp_serial", "-in", "lammps/slide.in", "-log", "none"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=600,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"lmp_serial exited {proc.returncode}:\n{out[-3000:]}"
    assert "Lost atoms" not in out, out[-3000:]

    rows = (tmp_path / "results" / "pes_scan.csv").read_text().strip().splitlines()
    assert rows[0] == "dx,dy,energy_eV"
    assert len(rows) == 1 + 3 * 3                      # grid_n x grid_n points
    energies = [float(r.split(",")[2]) for r in rows[1:]]
    # A real, non-degenerate corrugation was mapped (not all grid points equal).
    assert max(energies) - min(energies) > 1e-6


# --- _guard_scannable_cell (builders-sheetonsheet.md finding 7) ---------------
#
# All three raises were uncovered. The triclinic one is the one that matters:
# it is the only thing stopping an axis-aligned (dx, dy) grid from being stepped
# over a tilted cell, where the grid points are not lattice translations and the
# resulting PES is silently meaningless. No build and no LAMMPS is needed --
# the two attributes the guard reads are plain ones set in __init__.

def _guarded_builder(tmp_path, *, cell, tilt=0.0):
    from src.builders.sheetonsheet.pes_sheet import HeteroPESSheetSimulation

    sim = HeteroPESSheetSimulation(_hetero_pes_config(), str(tmp_path))
    sim._hetero_coincidence_cell = cell
    sim._hetero_cell_xy_tilt = tilt
    return sim


def test_guard_scannable_cell_accepts_an_orthogonal_positive_cell(tmp_path):
    """The guard must not over-fire on the shape every shipped fixture has."""
    _guarded_builder(tmp_path, cell=(12.74, 11.03)).\
        _guard_scannable_cell()          # no raise


def test_guard_scannable_cell_refuses_a_triclinic_cell(tmp_path):
    """A tilted cell cannot be stepped by an axis-aligned grid.

    The message must name the measured tilt, since the whole point is telling
    the user which pair they picked is unscannable.
    """
    sim = _guarded_builder(tmp_path, cell=(12.74, 11.03), tilt=6.37)
    with pytest.raises(ValueError, match="triclinic") as exc:
        sim._guard_scannable_cell()
    assert "6.37" in str(exc.value)


def test_guard_scannable_cell_refuses_a_non_positive_cell(tmp_path):
    sim = _guarded_builder(tmp_path, cell=(0.0, 11.03))
    with pytest.raises(ValueError, match="Non-positive"):
        sim._guard_scannable_cell()


def test_guard_scannable_cell_refuses_a_write_before_build(tmp_path):
    """`_hetero_coincidence_cell` is populated by the stack assembly in build(),
    so calling the run-phase seam first must say so rather than raising a bare
    TypeError while unpacking None."""
    sim = _guarded_builder(tmp_path, cell=None)
    with pytest.raises(ValueError, match="cannot be called before it"):
        sim._guard_scannable_cell()
