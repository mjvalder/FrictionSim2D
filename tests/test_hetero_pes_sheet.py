"""Tests for the hetero (two-material) sheet-on-sheet static PES scan.

Config fixtures mirror ``tests/test_hetero_slide.py``'s ``_hetero_2p2_config``
(two ``[2D-N]`` sections, ``sw`` potentials) but with one layer per material
(the 1+1 PES bilayer: frozen bottom + laterally scanned top) plus a ``[pes]``
section as in ``tests/test_pes_scan_builder.py``.

Covers the MRO/``SPLIT_OUTER_LOOPS`` resolution, routing, the relaxed layer
gate (and that the *dynamic* hetero slide keeps its own 4-layer gate), and a
real-geometry build smoke asserting that the emitted scan script uses the
hetero data file, z-band layer groups, and a scan period equal to the
**coincidence cell** of the matched interface -- not either constituent's own
unit cell.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

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


def _hetero_pes_config(layers=(1, 1), grid_n=4):
    return SheetOnSheetSimulationConfig(**_hetero_pes_raw(layers, grid_n),
                                        settings=load_settings())


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
    prediction (``src.core.run._run_scripts_per_build``) promise scripts the
    build never writes.
    """
    from src.builders.hetero_slide import HeteroSheetOnSheetSimulation
    from src.builders.pes_scan import (
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
    from src.core.run import _run_scripts_per_build

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
    from src.builders.pes_scan import HeteroPESSheetSimulation
    from src.core.run import _select_sheet_builder_cls
    cfg = _hetero_pes_config()
    assert _select_sheet_builder_cls("pes_sheet", cfg) is HeteroPESSheetSimulation


def test_routing_selects_hetero_pes_sheet_for_single_material_twist():
    from src.builders.pes_scan import HeteroPESSheetSimulation
    from src.core.run import _select_sheet_builder_cls
    cfg = _twist_pes_config()
    assert len(cfg.sheets) == 1 and cfg.general.twist_angle is not None
    assert _select_sheet_builder_cls("pes_sheet", cfg) is HeteroPESSheetSimulation


def test_routing_keeps_homogeneous_pes_sheet_for_single_material():
    from src.builders.pes_scan import PESSheetSimulation
    from src.core.run import _select_sheet_builder_cls
    cfg = _single_material_pes_config()
    assert _select_sheet_builder_cls("pes_sheet", cfg) is PESSheetSimulation


def test_routing_hetero_slide_unchanged_for_sheetonsheet():
    """The dynamic model must keep routing to the plain hetero slide builder."""
    from src.builders.hetero_slide import HeteroSheetOnSheetSimulation
    from src.core.run import _select_sheet_builder_cls
    cfg = _hetero_pes_config(layers=(2, 2))
    assert _select_sheet_builder_cls("sheetonsheet", cfg) is HeteroSheetOnSheetSimulation


# --- layer gate ---------------------------------------------------------------

def test_hetero_pes_sheet_min_layers_relaxed_to_two():
    from src.builders.pes_scan import HeteroPESSheetSimulation
    assert HeteroPESSheetSimulation.MIN_LAYERS == 2


def test_dynamic_hetero_slide_keeps_four_layer_gate(tmp_path):
    """Relaxing the gate for the PES variant must NOT relax it for the dynamic
    slide: a 1+1 dynamic hetero slide is physically meaningless (no fixed
    bottom + thermostatted middle + driven top) and must still be refused with
    the existing message."""
    from src.builders.hetero_slide import HeteroSheetOnSheetSimulation
    assert HeteroSheetOnSheetSimulation.MIN_LAYERS == 4
    sim = HeteroSheetOnSheetSimulation(_hetero_pes_config(), str(tmp_path))
    with pytest.raises(ValueError, match=r"[Aa]t least 4"):
        sim.build()


def test_hetero_pes_sheet_rejects_non_bilayer(tmp_path):
    """More than one layer per material is not a PES bilayer."""
    from src.builders.pes_scan import HeteroPESSheetSimulation
    sim = HeteroPESSheetSimulation(_hetero_pes_config(layers=(2, 2)), str(tmp_path))
    with pytest.raises(ValueError, match="exactly 2 layers"):
        sim.build()


# --- job name ------------------------------------------------------------------

def test_hetero_pes_sheet_job_name_carries_pair_token():
    """Two pairs sharing material 1 must not collide on the HPC job name."""
    from src.builders.pes_scan import HeteroPESSheetSimulation, PESSheetSimulation

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
def test_hetero_pes_sheet_build_emits_scan_over_coincidence_cell(tmp_path):
    """Build a real 1+1 MoS2/WS2 hetero bilayer and check the emitted scan.

    Asserts the four seams this variant exists for:

    1. the run phase is the static scan (grid loop over ``layer_2``), not the
       dynamic slide (no drive/thermostat);
    2. the data file is the hetero stack's RUN-DIR-RELATIVE ``hetero.data``,
       not the homogeneous sim-root-relative ``.../build/<sheet>.lmp``;
    3. layer groups come from z-band regions (hetero layers share atom types);
    4. the scan period is the matched interface's COINCIDENCE cell.
    """
    from src.builders.pes_scan import HeteroPESSheetSimulation

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

    # (2) run-dir-relative hetero data file (resolves with cwd = run dir)
    assert "read_data       hetero.data" in text
    assert "build/" not in text
    assert (tmp_path / "hetero.data").exists()
    assert "include         lammps/system.in.settings" in text
    assert "results/pes_scan.csv" in text

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
    # x=y=12 A, `sw`): the mean in-plane cell the two monolayers are strained
    # onto, measured once by running build_monolayer -> build_matched_supercells
    # directly (h-MoS2 cell 12.736266 x 11.029930, h-WS2 cell 12.743784 x
    # 11.036441, mean 12.740025 x 11.033185) and hardcoded here on purpose --
    # NOT read back from sim._hetero_coincidence_cell. Comparing the emitted
    # period against the attribute it was rendered from proves nothing: the
    # renderer faithfully emits whatever that attribute holds, so a bug that
    # stored the wrong cell there (a constituent's own unit cell, or a
    # replicated box) would still pass. Same value pinned independently in
    # tests/test_hetero_pes_tip.py for the tip scan over the same pair.
    EXPECTED_LX, EXPECTED_LY = 12.740024828691185, 11.033185146491927
    assert period_x == pytest.approx(EXPECTED_LX, rel=1e-6)
    assert period_y == pytest.approx(EXPECTED_LY, rel=1e-6)

    # ... and it is neither constituent's own primitive unit cell (~3.19 x 5.52).
    assert period_x > 10.0 and period_y > 10.0

    # The scanned box IS one coincidence cell, so the per-cell energy
    # normalization divides by exactly one cell.
    assert _extract("ncell") == pytest.approx(1.0)

    # The emitted period matches the assembled stack's own periodic box (the
    # hetero sheet-on-sheet stack is NOT replicated in-plane), so displacing the
    # top layer by one period is exactly one lattice translation.
    at = read(str(tmp_path / "hetero.data"), format="lammps-data")
    cell = np.array(at.cell)
    assert period_x == pytest.approx(float(cell[0, 0]), rel=1e-9)
    assert period_y == pytest.approx(float(cell[1, 1]), rel=1e-9)

    assert sim._get_hpc_job_name() == "pes_sheet_h-MoS2+h-WS2"


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_pes_sheet_scan_runs_in_lammps(tmp_path):
    """Capstone smoke: the emitted scan is valid LAMMPS and actually runs.

    ``cwd`` MUST be the run dir: every path in the script is run-dir-relative
    (``hetero.data``, ``lammps/system.in.settings``, and the settings file's
    staged ``provenance/potentials/...`` prefix), so this also proves the path
    convention. Checks a real corrugation was written to ``results/``.
    """
    import subprocess

    from src.builders.pes_scan import HeteroPESSheetSimulation

    raw = _hetero_pes_raw(grid_n=3)
    # The matched supercell must exceed the hetero cross-LJ cutoff
    # (``pair_style hybrid ... lj/cut 11.0``) or ``comm_style tiled`` aborts with
    # "Communication cutoff ... cannot exceed periodic box length" -- same
    # constraint as the dynamic hetero slide e2e (tests/test_hetero_slide.py):
    # a 12 A target gives an ~11.03 A edge (too tight), 14 A gives ~12.7 A.
    for key in ("2D-1", "2D-2"):
        raw[key]["x"] = raw[key]["y"] = 14.0
    cfg = SheetOnSheetSimulationConfig(**raw, settings=load_settings())

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
