"""End-to-end LAMMPS smoke suite for the instrumentation (spec §10, Task 14).

Builds a REAL homogeneous h-MoS2 AFM run (the empirically tuned minimal
config from tests/test_hetero_afm.py: r=5 tip, 1 K, single 10 nN load) and
executes system.in + the slide script under lmp_serial, then asserts the
spec §10 smoke rows: all three result tiers parse with registry names, the
pe/atom partition reconciles with global pe, RAMP_STATUS and the hold file
exist, and the final-state retention landed.

Marked `lammps`; skipped when lmp_serial is absent. Runtime ~2 min.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from src.core.config import AFMSimulationConfig, load_settings

pytestmark = pytest.mark.lammps

LMP = shutil.which("lmp_serial")
MAT, POT = "examples/materials", "examples/potentials/sw"


def _afm_smoke_raw():
    return {
        "general": {"temp": 1.0, "scan_speed": 1, "force": 10.0},
        "tip": {"mat": "Si", "cif_path": f"{MAT}/Si.cif",
                "pot_path": f"{POT}/Si.sw", "pot_type": "sw", "r": 5.0},
        "sub": {"mat": "Si", "cif_path": f"{MAT}/Si.cif",
                "pot_path": f"{POT}/Si.sw", "pot_type": "sw",
                "amorph": "a", "thickness": 12.0},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif",
               "pot_path": f"{POT}/MoS2_wen.sw", "pot_type": "sw",
               "x": 25.0, "y": 25.0, "layers": [1]},
    }


@pytest.fixture(scope="module")
def built_run(tmp_path_factory):
    from src.builders.afm import AFMSimulation
    out = tmp_path_factory.mktemp("smoke")
    cfg = AFMSimulationConfig(**_afm_smoke_raw(), settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]
    cfg.settings.geometry.tip_reduction_factor = 0.7
    cfg.settings.simulation.slide_run_steps = 2000
    cfg.general.campaign_seed = 20260819

    AFMSimulation(cfg, str(out)).build()

    # system.in MUST run before the slide (it writes load_*.data); cwd is the
    # run dir - every baked path is relative to it.
    scripts = [out / "L1" / "lammps" / "system.in"]
    scripts += sorted((out / "L1" / "lammps").glob("slide*.in"))
    logs = []
    for script in scripts:
        proc = subprocess.run(
            ["lmp_serial", "-in", str(script)], cwd=str(out),
            capture_output=True, text=True, timeout=1800,
        )
        logs.append(proc.stdout + proc.stderr)
        assert proc.returncode == 0, (
            f"{script.name} exited {proc.returncode}:\n"
            f"{(proc.stdout + proc.stderr)[-3000:]}")
    return out, "\n".join(logs)


@pytest.mark.skipif(not LMP, reason="lmp_serial not on PATH")
def test_all_three_result_tiers_parse(built_run):
    from src.postprocessing.read_data import DataReader
    out, _ = built_run
    r = DataReader.__new__(DataReader)
    for prefix, must_have in (("friction_", "nf"), ("raw_", "raw_lfx"),
                              ("peratom_", "E_int_tip_sheet")):
        files = list(out.rglob(f"results/{prefix}*"))
        assert files, f"no {prefix} output"
        df = r._read_dataframe(files[0], "tip")
        assert len(df) > 0, prefix
        assert must_have in df.columns, (prefix, list(df.columns))


@pytest.mark.skipif(not LMP, reason="lmp_serial not on PATH")
def test_pe_partition_reconciles(built_run):
    from src.postprocessing.read_data import DataReader
    out, _ = built_run
    r = DataReader.__new__(DataReader)
    df = r._read_dataframe(next(out.rglob("results/peratom_*")), "tip")
    win = r._read_dataframe(next(out.rglob("results/friction_*")), "tip")
    total = (df["pe_sum_tip_all"] + df["pe_sum_sub_all"]
             + df["pe_sum_2D_all"])
    pe = win["pe"].iloc[-len(df):].to_numpy()
    # group pe/atom sums partition global pe (virtual atom contributes ~0);
    # tolerance covers the different sampling instants of the two fixes
    rel = abs(total.to_numpy() - pe) / abs(pe)
    assert rel.max() < 1e-3, rel.max()


@pytest.mark.skipif(not LMP, reason="lmp_serial not on PATH")
def test_ramp_status_and_hold_emitted(built_run):
    from src.data.provenance import parse_log_extras
    out, log_text = built_run
    assert "RAMP_STATUS" in log_text
    holds = list(out.rglob("data/hold_*N.txt"))
    assert holds, "hold file missing"
    log = out / "log.lammps"
    if log.is_file():
        up = parse_log_extras(log)
        assert "ramp_converged" in up


@pytest.mark.skipif(not LMP, reason="lmp_serial not on PATH")
def test_final_state_retained_and_verifiable(built_run):
    from src.postprocessing.retention import (register_artefacts,
                                              verify_retention)
    out, _ = built_run
    finals = list(out.rglob("data/final_*.data"))
    assert finals, "final-state .data missing"
    run_dir = out / "L1"
    entries = register_artefacts(run_dir, ["data/*.data", "results/*"])
    assert entries
    assert verify_retention(run_dir, {"retained_artefacts": entries}) == []


@pytest.mark.skipif(not LMP, reason="lmp_serial not on PATH")
def test_provenance_written_at_build(built_run):
    import json
    out, _ = built_run
    prov = out / "L1" / "provenance.json"
    assert prov.is_file()
    data = json.loads(prov.read_text())
    assert data["seed_thermal_tip"] is not None   # campaign_seed was set
    assert data["cutoff_source"].startswith("uff_sigma")
