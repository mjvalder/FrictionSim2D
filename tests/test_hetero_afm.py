"""Tests for D2 periodic hetero AFM tip-slide assembly.

Covers config folding ([2D-1]/[2D-2] -> sheets), the in-plane replication
helpers, the HeteroAFMSimulation structure build + orthogonality guard, the
sheet_1/sheet_2 potential split + 2D union groups, render-context wiring,
routing, and a short real-LAMMPS end-to-end smoke.
"""
import math
import shutil
from pathlib import Path

import pytest

from src.core.config import AFMSimulationConfig, load_settings

MAT, POT = "examples/materials", "examples/potentials/sw"
LMP = shutil.which("lmp_serial")


def _afm_hetero_raw():
    return {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "tip": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "r": 10.0},
        "sub": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "amorph": "a", "thickness": 12.0},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif", "pot_path": f"{POT}/sw_lammps/t-WS2.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
    }


def _afm_homogeneous_raw():
    raw = _afm_hetero_raw()
    del raw["2D-2"]
    raw["2D"] = raw.pop("2D-1")
    return raw


# --- Task 1: config folding ---

def test_afm_config_folds_two_sheets():
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    assert len(cfg.sheets) == 2
    assert cfg.sheets[0].mat == "h-MoS2" and cfg.sheets[1].mat == "h-WS2"
    # single `sheet` still set (homogeneous path unchanged), = first sheet
    assert cfg.sheet.mat == "h-MoS2"


def test_afm_config_single_sheet_is_homogeneous():
    cfg = AFMSimulationConfig(**_afm_homogeneous_raw(), settings=load_settings())
    assert len(cfg.sheets) <= 1
    assert cfg.sheet.mat == "h-MoS2"


# --- Task 2: in-plane replication helpers ---

def test_replication_factor_meets_two_diameter_rule():
    from src.builders.hetero_afm import compute_replication_factors
    # footprint must be >= 4*r in each direction, min 1
    assert compute_replication_factors(lx=19.0, ly=22.0, tip_r=10.0) == (
        max(1, math.ceil(40.0 / 19.0)), max(1, math.ceil(40.0 / 22.0)))   # (3, 2)
    # already large enough -> no replication below the raw cell
    assert compute_replication_factors(lx=100.0, ly=100.0, tip_r=10.0) == (1, 1)
    # never below 1
    assert compute_replication_factors(lx=5.0, ly=5.0, tip_r=0.1) == (1, 1)


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_replicated_data_has_no_pair_coeffs(tmp_path):
    from src.builders.hetero import build_hetero_structure
    from src.builders.hetero_afm import replicate_hetero_in_plane
    from src.core.config import SheetOnSheetSimulationConfig
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif", "pot_path": f"{POT}/sw_lammps/t-WS2.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
    }
    cfg = SheetOnSheetSimulationConfig(**raw, settings=load_settings())
    stack = build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)
    out = tmp_path / "hetero_rep.data"
    replicate_hetero_in_plane(stack.data_path, out, 2, 2, cfg.settings)
    text = out.read_text()
    assert "Pair Coeffs" not in text          # written nocoeff
    assert "Atoms" in text


# --- Task 3: structure build + orthogonality guard ---

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_afm_build_replicates_and_orthogonal(tmp_path):
    from src.builders.hetero_afm import HeteroAFMSimulation
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    sim._build_hetero_sheet(build_dir=Path(tmp_path) / "build")   # structure-only entry
    assert sim._hetero_rep_path.exists()
    lx, ly = sim._hetero_rep_box[:2]
    # footprint >= 4*tip.r each way (tip r=10 -> >= 40 A)
    assert lx >= 40.0 - 1e-6 and ly >= 40.0 - 1e-6


def test_hetero_afm_rejects_triclinic_cell(tmp_path, monkeypatch):
    from src.builders import hetero_afm as ha
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    sim = ha.HeteroAFMSimulation(cfg, str(tmp_path))

    import numpy as np

    class _FakeAtoms:
        def get_cell(self):
            c = np.eye(3) * 20.0
            c[1, 0] = 5.0     # nonzero xy tilt -> triclinic
            return c

        def get_positions(self):
            return np.zeros((1, 3))

    class _FakeStack:  # only .data_path is read before the guard
        data_path = tmp_path / "hetero.data"

    (tmp_path / "hetero.data").write_text("")
    monkeypatch.setattr(ha, "build_hetero_structure", lambda *a, **k: _FakeStack())
    monkeypatch.setattr(ha, "read", lambda *a, **k: _FakeAtoms())
    with pytest.raises(ValueError, match="triclinic"):
        sim._build_hetero_sheet(build_dir=tmp_path / "build")


# --- Task 4: potential split (sheet_1/sheet_2 + 2D union groups) ---

def _prep_layer_dirs(sim, n):
    """Mirror the per-layer dir setup AFMSimulation.build() does before
    _generate_potentials, so the potential method can be exercised in isolation."""
    sim.output_dir_layer[n] = sim.output_dir / f"L{n}"
    sim.relative_run_dir_layer[n] = sim.relative_run_dir / f"L{n}"
    sim._create_directories(sim.output_dir_layer[n])


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_afm_potentials_layout_and_2D_group(tmp_path):
    from src.builders.hetero_afm import HeteroAFMSimulation
    cfg = AFMSimulationConfig(**_afm_hetero_raw(), settings=load_settings())
    sim = HeteroAFMSimulation(cfg, str(tmp_path))
    build_dir = Path(tmp_path) / "build"
    sim._init_provenance()
    sim._build_hetero_sheet(build_dir=build_dir)
    sim.tip_path, _tip_r, sim.sub_path = sim._build_components(build_dir)
    _prep_layer_dirs(sim, 1)
    pm = sim._generate_potentials(1)
    # contiguous, disjoint blocks: sub+tip then sheet_1+sheet_2 spans all types
    n_sub = len(pm.types.ids_by_component('sub'))
    n_tip = len(pm.types.ids_by_component('tip'))
    n_s1 = len(pm.types.ids_by_component('sheet_1'))
    n_s2 = len(pm.types.ids_by_component('sheet_2'))
    assert n_s1 > 0 and n_s2 > 0
    # contiguous blocks: sub+tip, then sheet_1+sheet_2, then any drive virtual type
    n_virtual = 1 if pm.virtual_atom_type is not None else 0
    assert (n_sub + n_tip) + (n_s1 + n_s2) + n_virtual == len(pm.types)
    # the sheet block sits immediately after sub+tip
    assert min(pm.types.ids_by_component('sheet_1')) == n_sub + n_tip + 1
    # 2D_all union emitted INTO the written settings file (both scripts include it)
    settings_text = (sim.output_dir_layer[1] / "lammps" / "system.in.settings").read_text()
    assert "group 2D_all union sheet_1_all sheet_2_all" in settings_text
    # sheet_types context var = sheet_1 ∪ sheet_2 union (slide.lmp `group 2D type {{sheet_types}}`)
    assert sim.groups[1]["sheet_types"].split() == [str(i) for i in
        pm.types.ids_by_component('sheet_1') + pm.types.ids_by_component('sheet_2')]
