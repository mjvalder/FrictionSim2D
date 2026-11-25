"""Tests for hetero (two-material) AFM indentation (hold + retract, no slide).

Config fixture modelled on tests/builders/test_hetero_afm.py's `_afm_hetero_raw()`
(tip + sub + [2D-1]/[2D-2]) — NOT test_hetero_slide.py, which has no tip or
substrate. Covers the MRO/SPLIT_OUTER_LOOPS resolution the class docstring
records, routing (including the finite_sheet guard), and a real-LAMMPS build
smoke that checks both the indent run scripts and the hetero data file land,
and that the pair token (not sheet 1's material alone) drives the job name,
the derive_seed input, and indent_meta.json.
"""
import json
import shutil
from pathlib import Path

import pytest

from frictionsim2d.core.config import AFMSimulationConfig, load_settings

MAT, POT = "examples/materials", "examples/potentials/sw"
LMP = shutil.which("lmp_serial")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _indent_hetero_raw():
    return {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped",
                    "force": [10]},
        "tip": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "r": 10.0},
        "sub": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "amorph": "a", "thickness": 12.0},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif", "pot_path": f"{POT}/sw_lammps/t-WS2.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
    }


# --- job name ---------------------------------------------------------------

def test_hetero_indent_job_name_carries_the_pair_token():
    """Two hetero indent configs sharing material 1 must not share a job name
    (the pair token, not sheet 1's material alone, names the job); a single
    material indent keeps its plain name."""
    from frictionsim2d.builders.afm.hetero import HeteroIndentSimulation
    from frictionsim2d.builders.afm.indent import IndentSimulation

    raw = _indent_hetero_raw()
    pair = HeteroIndentSimulation(AFMSimulationConfig(**raw, settings=load_settings()), "out")
    assert pair._get_hpc_job_name() == "indent_h-MoS2+h-WS2"

    raw["2D-2"]["mat"] = "h-WSe2"
    other = HeteroIndentSimulation(AFMSimulationConfig(**raw, settings=load_settings()), "out")
    assert other._get_hpc_job_name() == "indent_h-MoS2+h-WSe2"

    del raw["2D-2"]
    raw["2D"] = raw.pop("2D-1")
    raw["2D"]["x"] = raw["2D"]["y"] = 50.0
    single = IndentSimulation(AFMSimulationConfig(**raw, settings=load_settings()), "out")
    assert single._get_hpc_job_name() == "indent_h-MoS2"


# --- routing ----------------------------------------------------------------

def test_routing_selects_hetero_indent_for_two_materials():
    from frictionsim2d.builders.afm.hetero import HeteroIndentSimulation
    from frictionsim2d.run.simulate import _select_afm_builder_cls
    cfg = AFMSimulationConfig(**_indent_hetero_raw(), settings=load_settings())
    assert _select_afm_builder_cls("indent", cfg) is HeteroIndentSimulation


def test_routing_keeps_homogeneous_indent_for_single_material():
    from frictionsim2d.builders.afm.indent import IndentSimulation
    from frictionsim2d.run.simulate import _select_afm_builder_cls
    raw = _indent_hetero_raw()
    del raw["2D-2"]
    raw["2D"] = raw.pop("2D-1")
    # Single-sheet drops the hetero (>=2-sheet) exemption from
    # validate_tip_sheet_footprint, so widen past the tip.r=10.0 floor.
    raw["2D"]["x"] = raw["2D"]["y"] = 50.0
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    assert _select_afm_builder_cls("indent", cfg) is IndentSimulation


def test_routing_hetero_indent_rejects_finite_sheet():
    """Only plain 'afm' has a finite hetero variant (FiniteHeteroAFMSimulation);
    a finite hetero indent config must raise the dedicated NoHeteroBuilderError
    (a ValueError sibling to FreestandingConfigError, same hard-abort
    treatment -- see test_finite_hetero_indent_aborts_run_simulations below)
    rather than silently building a periodic patch under a name that promises
    edges."""
    from frictionsim2d.run.simulate import NoHeteroBuilderError, _select_afm_builder_cls
    raw = _indent_hetero_raw()
    raw["general"]["finite_sheet"] = True
    raw["sub"]["x"] = 80.0        # finite_sheet AFM requires substrate x/y set
    raw["sub"]["y"] = 80.0
    cfg = AFMSimulationConfig(**raw, settings=load_settings())
    with pytest.raises(NoHeteroBuilderError, match="finite"):
        _select_afm_builder_cls("indent", cfg)


def test_finite_hetero_indent_aborts_run_simulations_not_skips(tmp_path, monkeypatch):
    """finite_sheet is a config-level [general] setting, not a per-sweep-member
    one -- every member of a sweep would fail identically. run_simulations
    must therefore propagate NoHeteroBuilderError and abort (no output
    directory created), not swallow it via its per-config broad-except and
    report a misleading "created 0 simulation(s)". Mirrors how
    FreestandingConfigError is proven to hard-abort in test_run.py."""
    from frictionsim2d.core.config import write_config
    from frictionsim2d.run.simulate import NoHeteroBuilderError, run_simulations

    monkeypatch.chdir(REPO_ROOT)
    raw = _indent_hetero_raw()
    raw["general"]["finite_sheet"] = True
    raw["sub"]["x"] = 80.0
    raw["sub"]["y"] = 80.0
    config_path = write_config(raw, tmp_path / "finite_hetero_indent.ini")

    # The call must raise -- not return normally with an empty/partial
    # created_simulations list (that would be the "catch-and-skip" behaviour
    # the coordinator ruled out: a misleading "created 0 simulation(s)" with
    # no actionable reason).
    with pytest.raises(NoHeteroBuilderError, match="finite"):
        run_simulations(config_file=str(config_path), model="indent",
                        settings_file=None, output_root=tmp_path / "out",
                        generate_hpc=False)

    # No build ever ran (builder selection failed before build()): the
    # per-config scaffolding (provenance dir) may exist, but no LAMMPS inputs
    # were written for this config.
    assert not list((tmp_path / "out").rglob("lammps"))


# --- real-LAMMPS build smoke -------------------------------------------------

@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_indent_build_produces_scripts_data_and_pair_token(tmp_path):
    """builder.build() into tmp_path must produce: the indent hold/retract
    script (system.in + slide.in, mirroring the homogeneous indent build's
    layout) AND the replicated hetero data file (build/hetero_rep.data,
    emitted by HeteroAFMSimulation._build_hetero_sheet) -- proving the pure
    MRO composition actually builds, not just imports. The job name,
    derive_seed input and indent_meta.json must all carry the pair token
    'h-MoS2+h-WS2', not just sheet 1's material."""
    from frictionsim2d.builders.afm.hetero import HeteroIndentSimulation
    from frictionsim2d.builders.afm.indent import derive_seed

    cfg = AFMSimulationConfig(**_indent_hetero_raw(), settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]   # HPC script gen needs a non-empty module list
    sim = HeteroIndentSimulation(cfg, str(tmp_path))
    sim.build()

    pair = "h-MoS2+h-WS2"

    layer_dir = sim.output_dir_layer[1]
    lammps_dir = layer_dir / "lammps"
    assert (lammps_dir / "system.in").exists()
    assert (lammps_dir / "slide.in").exists()          # single load -> unsplit hold+retract script
    assert (sim.output_dir / "build" / "hetero_rep.data").exists()

    assert sim._get_hpc_job_name() == f"indent_{pair}"

    meta_path = sim.output_dir / "provenance" / "indent_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["material"] == pair
    seed = meta["layers"]["1"]
    assert seed == derive_seed(sim.base_seed, pair, 1)
    # Two pairs sharing material 1 must not collide: the seed is derived from
    # the FULL pair token, not sheet 1's material alone.
    assert seed != derive_seed(sim.base_seed, "h-MoS2", 1)
