"""`hpc generate`: artifact tests over a real simulation tree.

`hpc generate` writes files rather than calling one obvious expensive
boundary, so these tests assert on the files it produces and on
`result.output` -- never on a mock (tests/README.md). The same holds for
`hpc enrich-provenance` (bottom of this file), whose whole job is to rewrite
a provenance.json on disk.
"""
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src.cli import cli
from src.core.config import GlobalSettings
from src.run.hpc_dispatch import collect_hpc_simulation_paths


@pytest.fixture
def runner():
    return CliRunner()


def _sim_tree(tmp_path: Path, script: str = "slide.in", n_sims: int = 1) -> Path:
    """A simulation tree shaped like a real one: <root>/afm/<mat>/L1/lammps/<script>."""
    root = tmp_path / "sim_root"
    for i in range(n_sims):
        lammps = root / "afm" / f"mat{i}" / "L1" / "lammps"
        lammps.mkdir(parents=True)
        (lammps / script).write_text("# lammps\n", encoding="utf-8")
    return root


def _settings_file_with_modules(tmp_path: Path) -> Path:
    """`hpc generate` calls HPCScriptGenerator._validate_config, which raises
    unless settings.hpc.modules is set -- the default GlobalSettings() leaves
    it None."""
    settings = GlobalSettings()
    settings.hpc.modules = ["lammps"]
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings.model_dump(mode="json")), encoding="utf-8")
    return path


def test_scheduler_flag_picks_the_pbs_script(runner, tmp_path):
    sim_root = _sim_tree(tmp_path)
    settings_file = _settings_file_with_modules(tmp_path)

    result = runner.invoke(cli, ["hpc", "generate", str(sim_root),
                                 "--settings-file", str(settings_file),
                                 "--scheduler", "pbs"])

    assert result.exit_code == 0, result.output
    assert (sim_root / "hpc" / "run.pbs").exists()
    assert not (sim_root / "hpc" / "run.sh").exists()


def test_scheduler_flag_picks_the_slurm_script(runner, tmp_path):
    sim_root = _sim_tree(tmp_path)
    settings_file = _settings_file_with_modules(tmp_path)

    result = runner.invoke(cli, ["hpc", "generate", str(sim_root),
                                 "--settings-file", str(settings_file),
                                 "--scheduler", "slurm"])

    assert result.exit_code == 0, result.output
    assert (sim_root / "hpc" / "run.sh").exists()
    assert not (sim_root / "hpc" / "run.pbs").exists()


def test_a_tree_with_no_lammps_directory_exits_non_zero(runner, tmp_path):
    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()
    (empty_root / "notes.txt").write_text("nothing here\n", encoding="utf-8")

    result = runner.invoke(cli, ["hpc", "generate", str(empty_root)])

    assert result.exit_code != 0
    assert "No simulation directories found" in result.output
    # hpc/ is created up front (before discovery), so it exists-and-is-empty
    # today; the right fix is moving that mkdir after the refusal checks, at
    # which point hpc/ would never be created here at all. Tolerate both so
    # this test doesn't block that fix.
    hpc_dir = empty_root / "hpc"
    assert not hpc_dir.exists() or list(hpc_dir.iterdir()) == []


def test_output_dir_option_is_honoured(runner, tmp_path):
    sim_root = _sim_tree(tmp_path)
    settings_file = _settings_file_with_modules(tmp_path)
    out_dir = tmp_path / "somewhere_else"

    result = runner.invoke(cli, ["hpc", "generate", str(sim_root),
                                 "--settings-file", str(settings_file),
                                 "--output-dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert (out_dir / "run.pbs").exists()
    assert not (sim_root / "hpc").exists()


def test_discovery_agrees_with_collect_hpc_simulation_paths_for_suffixed_scripts(
        runner, tmp_path):
    """C2-1 equivalence check.

    ``hpc_generate`` used to re-implement simulation discovery inline with a
    private ``script_exists`` that special-cased only ``slide.in`` (matching
    ``slide_*.in``), never ``system.in`` -- so a directory containing only a
    suffixed system script (e.g. ``system_30nN.in``, the naming a
    per-condition system-phase sweep would use, mirroring slide's existing
    per-load suffixes) was invisible to the CLI even though
    ``run.hpc_dispatch.collect_hpc_simulation_paths`` (used by every other
    caller: ``run --hpc-scripts``, ``aiida rebuild --hpc-scripts``) already
    finds it via ``*system*.in``. This must find both directories, matching
    ``collect_hpc_simulation_paths`` exactly -- it fails against the
    pre-refactor CLI, which only ever found the slide one.
    """
    sim_root = tmp_path / "sim_root"

    system_dir = sim_root / "afm" / "mat_system" / "L1" / "lammps"
    system_dir.mkdir(parents=True)
    (system_dir / "system_30nN.in").write_text("# lammps\n", encoding="utf-8")

    slide_dir = sim_root / "afm" / "mat_slide" / "L1" / "lammps"
    slide_dir.mkdir(parents=True)
    (slide_dir / "slide_p1gpa.in").write_text("# lammps\n", encoding="utf-8")

    settings_file = _settings_file_with_modules(tmp_path)

    expected = collect_hpc_simulation_paths(sim_root)
    assert len(expected) == 2, expected  # sanity: both dirs are real candidates

    result = runner.invoke(cli, ["hpc", "generate", str(sim_root),
                                 "--settings-file", str(settings_file)])

    assert result.exit_code == 0, result.output
    assert f"Found {len(expected)} simulations" in result.output


def test_missing_modules_is_blocked_before_anything_is_written(runner, tmp_path):
    """cli.md finding 2 (fix, user-approved).

    On the common AFM path (1:1 system.in/slide.in per simulation directory)
    ``dispatch_hpc_generation`` calls ``generate_combined_scripts``, which
    never reaches ``HPCScriptGenerator._validate_config`` -- so with no HPC
    modules configured, ``hpc generate`` used to exit 0 and write a
    ``run.pbs`` containing ``module purge`` and no ``module load`` line: a
    job script that only fails once submitted to the cluster.
    ``hpc_generation_blocker`` (the same precondition
    ``generate_hpc_scripts_for_root`` already enforces for the GUI/
    ``run --hpc-scripts`` path) must gate this before any script or manifest
    file is written.
    """
    sim_root = tmp_path / "sim_root"
    lammps = sim_root / "afm" / "mat0" / "L1" / "lammps"
    lammps.mkdir(parents=True)
    (lammps / "system.in").write_text("# lammps\n", encoding="utf-8")
    (lammps / "slide.in").write_text("# lammps\n", encoding="utf-8")

    # No --settings-file: default GlobalSettings() leaves hpc.modules unset.
    result = runner.invoke(cli, ["hpc", "generate", str(sim_root)])

    assert result.exit_code != 0
    assert "'Modules'" in result.output
    assert not (sim_root / "hpc" / "run.pbs").exists()


# ---------------------------------------------------------------------------
# `hpc enrich-provenance` -- the shell entry point the generated job scripts
# call once LAMMPS has finished. Asserts on the JSON on disk, not on a mock.
# ---------------------------------------------------------------------------

def _run_dir_with_provenance(tmp_path: Path) -> Path:
    """A minimal but real simulation dir: provenance.json + lammps/."""
    from src.data.provenance import ProvenanceRecord, write_provenance

    run_dir = tmp_path / "afm" / "h_MoS2" / "L1"
    (run_dir / "lammps").mkdir(parents=True)
    write_provenance(run_dir, ProvenanceRecord(
        seed_thermal_tip=1, seed_thermal_sub=2, seed_velocity=3,
        substrate_realization_id=None, damping_ratio=0.1, damp_ev=None,
        git_commit="abc123", git_dirty=False, config_hash="deadbeef",
        layer_type_map={"layer_1": [1, 2]}, contact_cutoff=5.0,
        cutoff_source="uff_sigma_k1.5", thermostat_scheme="nvt_system",
        load_requested=10.0,
    ))
    return run_dir


def test_enrich_provenance_writes_the_settle_loop_fields_into_the_json(runner, tmp_path):
    run_dir = _run_dir_with_provenance(tmp_path)
    log_path = run_dir / "lammps" / "run.log"
    log_path.write_text(
        "LAMMPS (2 Aug 2023 - Update 3)\n"
        "# UNLOADED_REF layer=1 z=15.5\n"
        "# RAMP_STATUS converged=1 r=0.042 iterations=17\n",
        encoding="utf-8")

    result = runner.invoke(cli, ["hpc", "enrich-provenance",
                                 str(run_dir), str(log_path)])

    assert result.exit_code == 0, result.output
    data = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert data["ramp_converged"] is True
    assert data["ramp_r"] == 0.042
    assert data["ramp_iterations"] == 17
    assert data["unloaded_reference_z"] == {"layer_1": 15.5}
    assert data["lammps_version"] == "2 Aug 2023 - Update 3"
    assert "enriched" in result.output


def test_enrich_provenance_reports_a_non_converged_settle_loop(runner, tmp_path):
    """The whole point of the command: converged=0 must survive into the
    record, where a later analysis can see it."""
    run_dir = _run_dir_with_provenance(tmp_path)
    log_path = run_dir / "lammps" / "run.log"
    log_path.write_text("# RAMP_STATUS converged=0 r=1.62 iterations=100\n",
                        encoding="utf-8")

    result = runner.invoke(cli, ["hpc", "enrich-provenance",
                                 str(run_dir), str(log_path)])

    assert result.exit_code == 0, result.output
    data = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert data["ramp_converged"] is False
    assert data["ramp_iterations"] == 100


def test_enrich_provenance_exits_zero_when_the_log_is_missing(runner, tmp_path):
    """A completed simulation must never be reported as failed because the
    provenance nicety had nothing to read -- the job script calls this
    unconditionally after every run, including runs with no settle loop."""
    run_dir = _run_dir_with_provenance(tmp_path)
    before = (run_dir / "provenance.json").read_text(encoding="utf-8")

    result = runner.invoke(cli, ["hpc", "enrich-provenance",
                                 str(run_dir), str(run_dir / "lammps" / "run.log")])

    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "Traceback" not in result.output
    assert "not enriched" in result.output
    assert (run_dir / "provenance.json").read_text(encoding="utf-8") == before


def test_enrich_provenance_exits_zero_when_the_run_dir_does_not_exist(runner, tmp_path):
    result = runner.invoke(cli, ["hpc", "enrich-provenance",
                                 str(tmp_path / "nope"),
                                 str(tmp_path / "nope" / "run.log")])

    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "not enriched" in result.output


def test_enrich_provenance_says_so_when_the_log_holds_no_settle_lines(runner, tmp_path):
    run_dir = _run_dir_with_provenance(tmp_path)
    log_path = run_dir / "lammps" / "run.log"
    log_path.write_text("Total wall time: 0:00:01\n", encoding="utf-8")
    before = (run_dir / "provenance.json").read_text(encoding="utf-8")

    result = runner.invoke(cli, ["hpc", "enrich-provenance",
                                 str(run_dir), str(log_path)])

    assert result.exit_code == 0, result.output
    assert "unchanged" in result.output
    assert (run_dir / "provenance.json").read_text(encoding="utf-8") == before


def test_a_slide_phase_log_does_not_undo_an_already_enriched_record(runner, tmp_path):
    """The two-phase array workflow calls this once per task, each with its own
    log: the system task's log carries RAMP_STATUS and does the real merge, and
    the slide task -- which runs afterwards, on the same simulation directory --
    is handed a slide log with no settle-loop lines. That second call must be a
    no-op on the record, not a reset of the fields the first one wrote."""
    run_dir = _run_dir_with_provenance(tmp_path)
    system_log = run_dir / "lammps" / "run_system.log"
    system_log.write_text("# RAMP_STATUS converged=0 r=1.62 iterations=100\n",
                          encoding="utf-8")
    slide_log = run_dir / "lammps" / "run_slide.log"
    slide_log.write_text("Step Temp E_pair\n     0   300   -1.5\n"
                         "Total wall time: 0:12:34\n", encoding="utf-8")

    first = runner.invoke(cli, ["hpc", "enrich-provenance",
                                str(run_dir), str(system_log)])
    assert first.exit_code == 0, first.output
    enriched = (run_dir / "provenance.json").read_text(encoding="utf-8")
    assert json.loads(enriched)["ramp_converged"] is False

    second = runner.invoke(cli, ["hpc", "enrich-provenance",
                                 str(run_dir), str(slide_log)])

    assert second.exit_code == 0, second.output
    assert "unchanged" in second.output
    assert (run_dir / "provenance.json").read_text(encoding="utf-8") == enriched
