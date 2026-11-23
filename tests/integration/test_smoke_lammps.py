"""End-to-end LAMMPS smoke suite for the instrumentation.

Builds a REAL homogeneous h-MoS2 AFM run (the empirically tuned minimal
config from tests/builders/test_hetero_afm.py: r=5 tip, 1 K, single 10 nN load) and
executes system.in + the slide script under lmp_serial, then asserts: all
three result tiers parse with registry names, the pe/atom partition
reconciles with global pe, RAMP_STATUS and the hold file exist, and the
final-state .data was written.

Marked `lammps`; skipped when lmp_serial is absent. Runtime ~2 min.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from frictionsim2d.core.config import AFMSimulationConfig, load_settings

pytestmark = [pytest.mark.lammps, pytest.mark.integration]

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
    from frictionsim2d.builders.afm import AFMSimulation
    out = tmp_path_factory.mktemp("smoke")
    cfg = AFMSimulationConfig(**_afm_smoke_raw(), settings=load_settings())
    cfg.settings.hpc.modules = ["lammps"]
    cfg.settings.geometry.tip_reduction_factor = 0.7
    cfg.settings.simulation.slide_run_steps = 2000

    AFMSimulation(cfg, str(out)).build()

    # system.in MUST run before the slide (it writes load_*.data); cwd is the
    # run dir - every baked path is relative to it.
    scripts = [out / "L1" / "lammps" / "system.in"]
    scripts += sorted((out / "L1" / "lammps").glob("slide*.in"))
    logs = []
    for script in scripts:
        # one log per phase: a shared log.lammps would let the slide phase
        # overwrite system.in's RAMP_STATUS/UNLOADED_REF lines
        log_file = out / f"log_{script.stem}.lammps"
        proc = subprocess.run(
            ["lmp_serial", "-in", str(script), "-log", str(log_file)],
            cwd=str(out), capture_output=True, text=True, timeout=1800,
        )
        logs.append(proc.stdout + proc.stderr)
        assert proc.returncode == 0, (
            f"{script.name} exited {proc.returncode}:\n"
            f"{(proc.stdout + proc.stderr)[-3000:]}")
    return out, "\n".join(logs)


@pytest.mark.skipif(not LMP, reason="lmp_serial not on PATH")
def test_all_three_result_tiers_parse(built_run):
    from frictionsim2d.postprocessing.read_data import DataReader
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
    from frictionsim2d.postprocessing.read_data import DataReader
    out, _ = built_run
    r = DataReader.__new__(DataReader)
    df = r._read_dataframe(next(out.rglob("results/peratom_*")), "tip")
    win = r._read_dataframe(next(out.rglob("results/friction_*")), "tip")
    df["pe_total"] = (df["pe_sum_tip_all"] + df["pe_sum_sub_all"]
                      + df["pe_sum_2D_all"])
    # align by timestep: fc_pa's aligned immediate sample gives it an extra
    # row relative to fc_win, so positional slicing mismatches
    merged = df.merge(win[["time", "pe"]], on="time")
    assert len(merged) > 0, (list(df["time"]), list(win["time"]))
    # group pe/atom sums partition global pe (virtual atom contributes ~0).
    # NB fc_win's pe is a 1000-step BLOCK MEAN while fc_pa's sums are
    # instantaneous; at 1 K the fluctuation is tiny, hence the loose band.
    rel = (merged["pe_total"] - merged["pe"]).abs() / merged["pe"].abs()
    assert rel.max() < 1e-3, rel.max()


@pytest.mark.skipif(not LMP, reason="lmp_serial not on PATH")
def test_ramp_status_and_hold_emitted(built_run):
    from frictionsim2d.data.provenance import parse_log_extras
    out, log_text = built_run
    assert "RAMP_STATUS" in log_text
    holds = list(out.rglob("data/hold_*N.txt"))
    assert holds, "hold file missing"
    sys_log = out / "log_system.lammps"
    assert sys_log.is_file()
    up = parse_log_extras(sys_log)
    assert "ramp_converged" in up
    assert "unloaded_reference_z" in up


@pytest.mark.skipif(not LMP, reason="lmp_serial not on PATH")
def test_final_state_written(built_run):
    out, _ = built_run
    finals = list(out.rglob("data/final_*.data"))
    assert finals, "final-state .data missing"
    assert finals[0].stat().st_size > 0


@pytest.mark.skipif(not LMP, reason="lmp_serial not on PATH")
def test_provenance_written_at_build(built_run):
    import json
    out, _ = built_run
    prov = out / "L1" / "provenance.json"
    assert prov.is_file()
    data = json.loads(prov.read_text())
    # seeds are drawn in the builder, so provenance records them (a
    # template-side random would leave these null)
    assert isinstance(data["seed_thermal_tip"], int)
    assert data["seed_thermal_tip"] != data["seed_thermal_sub"]
    assert data["cutoff_source"].startswith("uff_sigma")
