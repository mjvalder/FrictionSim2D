"""`hpc generate`: artifact tests over a real simulation tree.

`hpc generate` writes files rather than calling one obvious expensive
boundary, so these tests assert on the files it produces and on
`result.output` -- never on a mock (tests/README.md). The same holds for
`hpc enrich-provenance` (bottom of this file), whose whole job is to rewrite
a provenance.json on disk.
"""
import json
from pathlib import Path
from typing import Optional

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


def test_preflight_flag_writes_the_preflight_pair(runner, tmp_path):
    sim_root = _sim_tree(tmp_path)
    settings_file = _settings_file_with_modules(tmp_path)

    plain = runner.invoke(cli, ["hpc", "generate", str(sim_root),
                                "--settings-file", str(settings_file)])
    assert plain.exit_code == 0, plain.output
    assert not (sim_root / "hpc" / "preflight.sh").exists()

    result = runner.invoke(cli, ["hpc", "generate", str(sim_root),
                                 "--settings-file", str(settings_file), "--preflight"])

    assert result.exit_code == 0, result.output
    assert (sim_root / "hpc" / "preflight.sh").is_file()
    assert (sim_root / "hpc" / "preflight.pbs").is_file()
    assert "qsub hpc/preflight.pbs" in result.output


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

#: Every real LAMMPS run opens with a `LAMMPS (<version>)` banner, so a
#: fixture without one cannot occur in production -- and leaving it out is what
#: made a slide-phase log look inert when it is not: the banner alone is enough
#: for parse_log_extras to return an update and rewrite the record.
_CONVERGED_LOG = ("LAMMPS (2 Aug 2023 - Update 3)\n"
                  "# UNLOADED_REF layer=1 z=15.5\n"
                  "# RAMP_STATUS converged=1 r=0.042 iterations=17\n")
_SLIDE_LOG = ("LAMMPS (2 Aug 2023 - Update 3)\n"
              "Step Temp E_pair\n     0   300   -1.5\n"
              "Total wall time: 0:12:34\n")


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
    assert result.output.startswith("skipped ")
    assert (run_dir / "provenance.json").read_text(encoding="utf-8") == before


def test_enrich_provenance_exits_zero_when_the_run_dir_does_not_exist(runner, tmp_path):
    result = runner.invoke(cli, ["hpc", "enrich-provenance",
                                 str(tmp_path / "nope"),
                                 str(tmp_path / "nope" / "run.log")])

    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert result.output.startswith("skipped ")


def test_re_reading_the_same_log_reports_unchanged(runner, tmp_path):
    """The 'unchanged' outcome is what a repeat pass over an already-swept tree
    produces: the same log yields the same values, so the record's bytes do not
    move. (A log with no recognisable lines at all would also land here, but
    that is not a shape a real run produces -- every run prints its version.)
    """
    run_dir = _run_dir_with_provenance(tmp_path)
    log_path = run_dir / "lammps" / "run_system.log"
    log_path.write_text(_CONVERGED_LOG, encoding="utf-8")

    first = runner.invoke(cli, ["hpc", "enrich-provenance",
                                str(run_dir), str(log_path)])
    assert first.output.startswith("enriched "), first.output
    enriched = (run_dir / "provenance.json").read_text(encoding="utf-8")

    result = runner.invoke(cli, ["hpc", "enrich-provenance",
                                 str(run_dir), str(log_path)])

    assert result.exit_code == 0, result.output
    assert "unchanged" in result.output
    # The reason must not claim the log was empty: this one demonstrably holds
    # a RAMP_STATUS line, and saying otherwise is how a user concludes the
    # settle loop never ran.
    assert "no settle-loop lines" not in result.output
    assert "already up to date" in result.output
    assert (run_dir / "provenance.json").read_text(encoding="utf-8") == enriched


def test_a_slide_phase_log_cannot_disturb_the_system_phase_s_settle_result(
        runner, tmp_path):
    """The two-phase array workflow calls this once per task, each with its own
    log: the system task records the settle result, then the slide task -- same
    simulation directory, later -- is handed a slide log.

    That second call is *not* a no-op. Every real LAMMPS run opens with a
    `LAMMPS (<version>)` banner, so parse_log_extras returns a
    `lammps_version` update and the record is rewritten. What it cannot do is
    lose anything: the merge is a `model_copy(update=...)`, which only touches
    the keys the log supplied. This uses two different LAMMPS versions so the
    rewrite is visible, and pins the documented consequence -- the later log
    wins `lammps_version` -- alongside the property that matters, that the
    ramp fields survive it untouched.
    """
    run_dir = _run_dir_with_provenance(tmp_path)
    system_log = run_dir / "lammps" / "run_system.log"
    system_log.write_text("LAMMPS (2 Aug 2023 - Update 3)\n"
                          "# UNLOADED_REF layer=1 z=15.5\n"
                          "# RAMP_STATUS converged=0 r=1.62 iterations=100\n",
                          encoding="utf-8")
    slide_log = run_dir / "lammps" / "run_slide.log"
    slide_log.write_text("LAMMPS (21 Nov 2023)\n"
                         "Step Temp E_pair\n     0   300   -1.5\n"
                         "Total wall time: 0:12:34\n", encoding="utf-8")

    first = runner.invoke(cli, ["hpc", "enrich-provenance",
                                str(run_dir), str(system_log)])
    assert first.exit_code == 0, first.output
    settled = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert settled["ramp_converged"] is False

    second = runner.invoke(cli, ["hpc", "enrich-provenance",
                                 str(run_dir), str(slide_log)])

    assert second.exit_code == 0, second.output
    after = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    # The settle result -- the whole reason this command exists -- is intact.
    for field in ("ramp_converged", "ramp_r", "ramp_iterations",
                  "unloaded_reference_z"):
        assert after[field] == settled[field], field
    # ...and the record really was rewritten, rather than left alone.
    assert after["lammps_version"] == "21 Nov 2023"
    assert second.output.startswith("enriched ")


def test_a_fresh_record_is_enriched_even_by_a_slide_only_log(runner, tmp_path):
    """A run whose only log is a slide log still gains its LAMMPS version, and
    still gains no settle fields -- nothing is invented from a log that never
    printed RAMP_STATUS."""
    run_dir = _run_dir_with_provenance(tmp_path)
    slide_log = run_dir / "lammps" / "run_slide.log"
    slide_log.write_text(_SLIDE_LOG, encoding="utf-8")

    result = runner.invoke(cli, ["hpc", "enrich-provenance",
                                 str(run_dir), str(slide_log)])

    assert result.exit_code == 0, result.output
    data = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert data["lammps_version"] == "2 Aug 2023 - Update 3"
    assert data["ramp_converged"] is None
    assert data["ramp_iterations"] is None


# ---------------------------------------------------------------------------
# The local post-transfer sweep. This is the working path today: FrictionSim2D
# is not installed on the cluster, so the job script's guarded call is a no-op
# and the merge happens here, over a whole rsynced-back tree, in one command.
# ---------------------------------------------------------------------------

def _run_dir(root: Path, rel: str, *, provenance: bool = True,
             logs: Optional[dict] = None,
             record_extras: Optional[dict] = None) -> Path:
    """One run directory inside a tree, shaped like a real one.

    `lammps/*.in` is what `collect_hpc_simulation_paths` keys on, so every run
    dir gets one -- the inputs stay on this side of the transfer anyway, since
    the tree is built locally and only results come back.

    `record_extras` seeds the post-run fields (`ramp_*`, `unloaded_reference_z`,
    `lammps_version`) so a run can start out ALREADY enriched, which is what a
    second sweep over a two-phase tree actually meets.
    """
    from src.data.provenance import ProvenanceRecord, write_provenance

    run = root / rel
    (run / "lammps").mkdir(parents=True)
    (run / "lammps" / "system.in").write_text("# lammps\n", encoding="utf-8")
    for name, text in (logs or {}).items():
        (run / "lammps" / name).write_text(text, encoding="utf-8")
    if provenance:
        write_provenance(run, ProvenanceRecord(
            seed_thermal_tip=1, seed_thermal_sub=2, seed_velocity=3,
            substrate_realization_id=None, damping_ratio=0.1, damp_ev=None,
            git_commit="abc123", git_dirty=False, config_hash="deadbeef",
            layer_type_map={"layer_1": [1, 2]}, contact_cutoff=5.0,
            cutoff_source="uff_sigma_k1.5", thermostat_scheme="nvt_system",
            load_requested=10.0, **(record_extras or {}),
        ))
    return run


#: A record that has already been enriched by its own system phase, carrying an
#: older LAMMPS banner so a later slide log's version update is visible.
_SETTLED_FIELDS = {
    "lammps_version": "29 Sep 2021 - Update 2",
    "ramp_converged": True,
    "ramp_r": 0.031,
    "ramp_iterations": 12,
    "unloaded_reference_z": {"layer_1": 14.25},
}


def _mixed_tree(tmp_path: Path) -> tuple:
    """A rsynced-back tree with one of each outcome."""
    root = tmp_path / "simulation_20260908_101500"
    good = _run_dir(root, "afm/h_MoS2/L1", logs={"run_system.log": _CONVERGED_LOG})
    slide_only = _run_dir(root, "afm/h_MoS2/L2", logs={"run_slide.log": _SLIDE_LOG})
    settled = _run_dir(root, "afm/h_WS2/L1", logs={"run_slide.log": _SLIDE_LOG},
                       record_extras=_SETTLED_FIELDS)
    no_record = _run_dir(root, "afm/t_CoTe2/L1", provenance=False,
                         logs={"run_system.log": _CONVERGED_LOG})
    return root, good, slide_only, settled, no_record


def _outcomes(output: str) -> dict:
    """Parse the sweep's per-directory lines into {relative path: outcome}."""
    parsed = {}
    for line in output.splitlines():
        head, sep, _ = line.partition(":")
        outcome, _, path = head.partition(" ")
        if sep and outcome in ("enriched", "unchanged", "skipped") and path:
            parsed[path] = outcome
    return parsed


def test_recursive_sweep_enriches_every_run_it_can_and_reports_the_rest(
        runner, tmp_path):
    """The realistic mix: a run with a system log, a run whose only log is a
    slide log, a run already enriched by its own system phase, and a run with
    no provenance.json at all.

    The slide-only run is 'enriched' rather than 'unchanged' -- its record had
    no lammps_version and the log's banner supplies one. What it must *not*
    gain is a settle result invented out of a log that never printed one.

    That run alone cannot guard the merge, though: it starts from a fresh
    record, so `ramp_* is None` afterwards holds whether the merge keeps the
    fields the log did not mention or rebuilds the record and nulls them. The
    already-settled run is the one that tells those apart -- its settle result
    predates this sweep and must survive a slide log that never mentions it.
    """
    root, good, slide_only, settled, no_record = _mixed_tree(tmp_path)

    result = runner.invoke(cli, ["hpc", "enrich-provenance", "-r", str(root)])

    assert result.exit_code == 0, result.output

    # The run with a system log gets the settle result.
    data = json.loads((good / "provenance.json").read_text(encoding="utf-8"))
    assert data["ramp_converged"] is True
    assert data["ramp_iterations"] == 17
    assert data["unloaded_reference_z"] == {"layer_1": 15.5}

    # The slide-only run gains its version and nothing it cannot know.
    slide_data = json.loads(
        (slide_only / "provenance.json").read_text(encoding="utf-8"))
    assert slide_data["lammps_version"] == "2 Aug 2023 - Update 3"
    assert slide_data["ramp_converged"] is None
    assert slide_data["ramp_iterations"] is None
    assert slide_data["unloaded_reference_z"] is None

    # The already-settled run keeps every post-run field the slide log is
    # silent about, and takes only what that log actually supplies (the newer
    # banner wins lammps_version -- the documented last-log-wins consequence).
    settled_data = json.loads(
        (settled / "provenance.json").read_text(encoding="utf-8"))
    assert settled_data["ramp_converged"] is True
    assert settled_data["ramp_r"] == 0.031
    assert settled_data["ramp_iterations"] == 12
    assert settled_data["unloaded_reference_z"] == {"layer_1": 14.25}
    assert settled_data["lammps_version"] == "2 Aug 2023 - Update 3"

    # The run with no record is skipped, not crashed on.
    assert not (no_record / "provenance.json").exists()

    assert _outcomes(result.output) == {
        "afm/h_MoS2/L1": "enriched",
        "afm/h_MoS2/L2": "enriched",
        "afm/h_WS2/L1": "enriched",
        "afm/t_CoTe2/L1": "skipped",
    }
    assert "3 enriched, 0 unchanged, 1 skipped." in result.output


def test_recursive_sweep_is_idempotent(runner, tmp_path):
    """Re-running over an already-swept tree must not undo anything: the
    second pass re-reads the same logs and writes the same values."""
    root, good, _, _, _ = _mixed_tree(tmp_path)

    runner.invoke(cli, ["hpc", "enrich-provenance", "-r", str(root)])
    after_first = (good / "provenance.json").read_text(encoding="utf-8")
    result = runner.invoke(cli, ["hpc", "enrich-provenance", "-r", str(root)])

    assert result.exit_code == 0, result.output
    assert (good / "provenance.json").read_text(encoding="utf-8") == after_first


def test_recursive_sweep_over_a_tree_with_no_runs_exits_zero(runner, tmp_path):
    """Not an error: pointing this at the wrong directory, or at a tree whose
    jobs have not run yet, must not look like a failure."""
    empty = tmp_path / "nothing_here"
    (empty / "notes").mkdir(parents=True)

    result = runner.invoke(cli, ["hpc", "enrich-provenance", "-r", str(empty)])

    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "nothing to enrich" in result.output


def test_recursive_sweep_rejects_a_log_path(runner, tmp_path):
    """-r resolves each run's own log; a single LOG_PATH cannot mean anything
    across a tree, so it is a usage error rather than a silently ignored arg."""
    root, _, _, _, _ = _mixed_tree(tmp_path)

    result = runner.invoke(cli, ["hpc", "enrich-provenance", "-r", str(root),
                                 str(root / "afm/h_MoS2/L1/lammps/run_system.log")])

    assert result.exit_code != 0
    assert "cannot also take a single LOG_PATH" in result.output


def test_log_path_defaults_to_run_system_log(runner, tmp_path):
    """system*.in is the only phase that prints RAMP_STATUS, so run_system.log
    wins even when other run logs sit beside it."""
    run = _run_dir(tmp_path, "L1", logs={"run_system.log": _CONVERGED_LOG,
                                         "run_slide.log": _SLIDE_LOG})

    result = runner.invoke(cli, ["hpc", "enrich-provenance", str(run)])

    assert result.exit_code == 0, result.output
    assert "run_system.log" in result.output
    data = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    assert data["ramp_iterations"] == 17


def test_log_path_falls_back_to_a_sole_run_log(runner, tmp_path):
    """A directory whose only log is not called run_system.log is unambiguous,
    so use it -- covers a slide-only run and the fixed `run.log` an earlier
    revision of the array templates wrote."""
    run = _run_dir(tmp_path, "L1", logs={"run.log": _CONVERGED_LOG})

    result = runner.invoke(cli, ["hpc", "enrich-provenance", str(run)])

    assert result.exit_code == 0, result.output
    data = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    assert data["ramp_converged"] is True


def test_several_logs_with_no_run_system_log_are_skipped_not_guessed(
        runner, tmp_path):
    run = _run_dir(tmp_path, "L1", logs={"run_slide_f10N.log": _CONVERGED_LOG,
                                         "run_slide_f20N.log": _SLIDE_LOG})
    before = (run / "provenance.json").read_text(encoding="utf-8")

    result = runner.invoke(cli, ["hpc", "enrich-provenance", str(run)])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("skipped ")
    assert "none named run_system.log" in result.output
    assert "run_slide_f10N.log" in result.output and "run_slide_f20N.log" in result.output
    assert (run / "provenance.json").read_text(encoding="utf-8") == before


def test_no_log_at_all_is_skipped_with_a_reason(runner, tmp_path):
    run = _run_dir(tmp_path, "L1")

    result = runner.invoke(cli, ["hpc", "enrich-provenance", str(run)])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("skipped ")
    assert "no run log in" in result.output
