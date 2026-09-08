"""Every hpc/*.j2 must be rendered by some test: they are reached only by
f-string name construction (`scripts.py` builds `f"{scheduler}_array.j2"` /
`f"{scheduler}_combined.j2"`), so a filename grep across src/ finds no static
reference and a typo in one scheduler/shape combination would ship undetected.

Renders through HPCScriptGenerator directly (not the now-deleted
create_hpc_package, which only ever took the plain array path -- see hpc.md
H-7), covering both schedulers and both script shapes so all four templates
are exercised. `generate_two_phase_scripts` reuses the *_array.j2 templates
already covered here, so it needs no separate rendering for template
liveness (its own dependency-submission logic is out of scope per the task's
Phase 2c deferral).
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.hpc.manifest import JobEntry, JobManifest
from src.hpc.scripts import HPCConfig, HPCScriptGenerator

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "src" / "templates" / "hpc"


def _config() -> HPCConfig:
    config = HPCConfig()
    config.hpc_settings.modules = ["lammps"]
    return config


def _combined_manifest(sim_path: str = "afm/mat0/L1") -> JobManifest:
    manifest = JobManifest()
    manifest.add_job(JobEntry(job_id="sys", simulation_path=sim_path,
                               lammps_script="system.in", material="mat0"))
    manifest.add_job(JobEntry(job_id="sld", simulation_path=sim_path,
                               lammps_script="slide.in", material="mat0"))
    return manifest


def test_every_hpc_template_is_rendered_by_the_script_generator(tmp_path):
    texts = {}
    for scheduler in ("pbs", "slurm"):
        script_ext = ".pbs" if scheduler == "pbs" else ".sh"
        generator = HPCScriptGenerator(_config())

        array_dir = tmp_path / scheduler / "array"
        generator.generate_scripts(["mat0/L1"], array_dir, scheduler=scheduler)
        texts[(scheduler, "array")] = (array_dir / f"run{script_ext}").read_text(encoding="utf-8")

        combined_dir = tmp_path / scheduler / "combined"
        generator.generate_combined_scripts(_combined_manifest(), combined_dir, scheduler=scheduler)
        texts[(scheduler, "combined")] = (combined_dir / f"run{script_ext}").read_text(encoding="utf-8")

    on_disk = {p.name for p in TEMPLATE_DIR.glob("*.j2")}
    assert on_disk, f"no templates found under {TEMPLATE_DIR}"

    for template in on_disk:
        scheduler, _, shape = template.replace(".j2", "").partition("_")
        assert (scheduler, shape) in texts, (
            f"{template}: no test renders scheduler={scheduler!r} shape={shape!r}")
        text = texts[(scheduler, shape)]
        if shape == "combined":
            # Combined scripts hardcode the two-phase filenames rather than
            # reading them from a manifest file.
            assert "system.in" in text and "slide.in" in text, template
        else:
            # Array scripts resolve their per-task path from a manifest file.
            assert "sed -n" in text, template


# ---------------------------------------------------------------------------
# Post-run provenance enrichment (review A-1).
#
# `lmp_flags` defaults to "-l none", so LAMMPS writes no log of its own and the
# settle loop's `# RAMP_STATUS ...` lines reach only the scheduler's spool
# file, which the running job cannot read. Every template therefore has to tee
# stdout to a per-run log and hand that log to
# `FrictionSim2D hpc enrich-provenance` -- see src/builders/afm/base.py's
# module-level `enrich_provenance_post_run`.
# ---------------------------------------------------------------------------

#: Per-shape expectations for the run log.
#:
#: The array template renders *both* phases of the two-phase workflow
#: (`generate_two_phase_scripts` runs it once per manifest), and both tasks
#: share one SIM_PATH -- so its log name is derived from the input script at
#: run time (run_system.log / run_slide.log) rather than fixed, and the tee,
#: the scratch copy-back and the enrichment argument all go through the same
#: ${LOG_PATH} variable. The combined template runs both phases in one task and
#: can name its two logs statically.
_TEE_TARGETS = {
    "array": ['| tee "${LOG_PATH}"'],
    "combined": ['| tee ${LAMMPS_SIM_PATH}/lammps/run_system.log',
                 '| tee ${LAMMPS_SIM_PATH}/lammps/run_slide.log'],
}

#: The second argument of the enrichment call, per shape.
_ENRICH_LOG_ARG = {
    # Each task enriches from its own log: the system task's holds RAMP_STATUS
    # and drives the real merge, the slide task's holds none and leaves the
    # record untouched (see test_a_slide_phase_log_does_not_undo... in
    # tests/cli/test_cli_hpc_group.py, which proves the no-op is harmless).
    "array": '"{workdir}/${{LOG_PATH}}"',
    # One task, two logs -- enrich from the system one, the only RAMP_STATUS
    # source.
    "combined": '"{workdir}/${{SIM_PATH}}/lammps/run_system.log"',
}

#: The log-rescue calls each shape must make. `save_run_log` is a shell
#: function the templates define once: a real copy under use_tmpdir, a no-op
#: without it (the log is already in its final place). Calling it on the
#: failure path as well as the success path is what keeps a crashed run's log,
#: and that behaviour is pinned by execution further down, not here.
_RESCUE_CALLS = {
    "array": ['save_run_log "${LOG_PATH}"'],
    "combined": ['save_run_log "${LAMMPS_SIM_PATH}/lammps/run_system.log"',
                 'save_run_log "${LAMMPS_SIM_PATH}/lammps/run_slide.log"'],
}


def _render_all(tmp_path, use_tmpdir=False):
    """Render every scheduler x shape combination; returns {(sch, shape): text}."""
    texts = {}
    for scheduler in ("pbs", "slurm"):
        script_ext = ".pbs" if scheduler == "pbs" else ".sh"
        config = _config()
        config.hpc_settings.use_tmpdir = use_tmpdir
        config.hpc_settings.scratch_dir = "/scratch"
        generator = HPCScriptGenerator(config)

        array_dir = tmp_path / f"{scheduler}_{use_tmpdir}" / "array"
        generator.generate_scripts(["mat0/L1"], array_dir, scheduler=scheduler)
        texts[(scheduler, "array")] = (
            array_dir / f"run{script_ext}").read_text(encoding="utf-8")

        combined_dir = tmp_path / f"{scheduler}_{use_tmpdir}" / "combined"
        generator.generate_combined_scripts(
            _combined_manifest(), combined_dir, scheduler=scheduler)
        texts[(scheduler, "combined")] = (
            combined_dir / f"run{script_ext}").read_text(encoding="utf-8")
    return texts


@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
@pytest.mark.parametrize("shape", ["array", "combined"])
def test_every_template_tees_lammps_stdout_to_a_per_run_log(tmp_path, scheduler, shape):
    text = _render_all(tmp_path)[(scheduler, shape)]

    lammps_lines = [ln for ln in text.splitlines()
                    if " lmp " in ln and "-in " in ln]
    assert lammps_lines, text
    for line in lammps_lines:
        assert "| tee " in line, line
    for target in _TEE_TARGETS[shape]:
        assert target in text, (target, text)
    # Never a fixed run.log: in the two-phase array workflow the slide task
    # would overwrite the system task's log, which is the one with RAMP_STATUS.
    assert "/lammps/run.log" not in text


def test_the_two_phase_array_tasks_derive_two_different_log_names(tmp_path):
    """Run the template's own derivation line under bash, once per phase, to
    prove the two-phase workflow's two array tasks -- which share a SIM_PATH --
    end up with different logs instead of clobbering one run.log."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")

    text = _render_all(tmp_path)[("pbs", "array")]
    derivation = next(ln for ln in text.splitlines() if ln.startswith("LOG_PATH="))

    names = []
    for script in ("afm/mat0/L1/lammps/system.in", "afm/mat0/L1/lammps/slide.in"):
        snippet = (f'SCRIPT_PATH="{script}"\n'
                   'SCRIPT_DIR=$(dirname "$SCRIPT_PATH")\n'
                   f'{derivation}\n'
                   'echo "$LOG_PATH"\n')
        proc = subprocess.run([bash, "-c", snippet], capture_output=True, text=True,
                              check=False)
        assert proc.returncode == 0, proc.stderr
        names.append(proc.stdout.strip())

    assert names == ["afm/mat0/L1/lammps/run_system.log",
                     "afm/mat0/L1/lammps/run_slide.log"]


def test_the_array_template_derives_its_log_name_from_the_input_script(tmp_path):
    """`generate_two_phase_scripts` renders *_array.j2 twice -- once against
    manifest_system.txt, once against manifest_slide.txt -- and both arrays
    resolve to the same SIM_PATH. The log name therefore has to come from
    $SCRIPT_PATH at run time, or the second array clobbers the first's log."""
    for scheduler in ("pbs", "slurm"):
        text = _render_all(tmp_path / scheduler)[(scheduler, "array")]
        assert 'LOG_PATH="${SCRIPT_DIR}/run_$(basename "$SCRIPT_PATH" .in).log"' in text
        # Derived once, then reused everywhere the log is named.
        assert text.count("${LOG_PATH}") >= 2


@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
@pytest.mark.parametrize("shape", ["array", "combined"])
def test_every_template_tests_pipestatus_not_dollar_question(tmp_path, scheduler, shape):
    """With a pipe in front of it, `$?` is tee's status -- always 0 -- so the
    old `if [ $? -ne 0 ]` guards would have reported every LAMMPS crash as a
    successful run."""
    text = _render_all(tmp_path)[(scheduler, shape)]

    assert "if [ $? -ne 0 ]" not in text
    lines = text.splitlines()
    lammps_idx = [i for i, ln in enumerate(lines) if " lmp " in ln and "-in " in ln]
    assert lammps_idx, text
    for i in lammps_idx:
        # The status must be captured on the line immediately after the
        # pipeline: anything in between (even an echo) resets PIPESTATUS.
        assert "${PIPESTATUS[0]}" in lines[i + 1], lines[i:i + 2]
        status_var = lines[i + 1].split("=", 1)[0].strip()
        assert f"if [ ${{{status_var}}} -ne 0 ]; then" in text


@pytest.mark.parametrize("scheduler, workdir", [
    ("pbs", "$PBS_O_WORKDIR"), ("slurm", "$SLURM_SUBMIT_DIR")])
@pytest.mark.parametrize("shape", ["array", "combined"])
def test_every_template_calls_enrich_provenance_against_the_workdir_copy(
        tmp_path, scheduler, workdir, shape):
    """Asserts the properties, not the exact rendered string: the command is
    invoked, its run-directory argument is workdir-rooted (not the scratch
    copy, which is gone by then), its log argument is the derived path, and it
    cannot fail the job. Reformatting the line must not break this."""
    text = _render_all(tmp_path)[(scheduler, shape)]

    call = [ln for ln in text.splitlines()
            if "hpc enrich-provenance" in ln and not ln.lstrip().startswith("#")]
    assert len(call) == 1, call
    line = call[0]

    run_arg = f'"{workdir}/${{SIM_PATH}}"'
    log_arg = _ENRICH_LOG_ARG[shape].format(workdir=workdir)
    assert run_arg in line, line
    assert log_arg in line, line
    assert line.index(run_arg) < line.index(log_arg), line
    # Non-fatal, and skipped entirely where the package is not installed.
    assert line.rstrip().endswith("|| true"), line
    assert "if command -v FrictionSim2D >/dev/null 2>&1; then" in text


@pytest.mark.parametrize("scheduler, workdir", [
    ("pbs", "$PBS_O_WORKDIR"), ("slurm", "$SLURM_SUBMIT_DIR")])
@pytest.mark.parametrize("shape", ["array", "combined"])
def test_the_log_rescue_helper_copies_out_of_scratch_only_under_tmpdir(
        tmp_path, scheduler, workdir, shape):
    """Under use_tmpdir the job runs in scratch, which is deleted when the task
    ends, and the end-of-script copy-back keeps only results/ visuals/ data/
    and load_*.data -- so the log needs its own rescue. Without scratch it is
    already in its final place and the helper is a no-op."""
    plain = _render_all(tmp_path / "plain", use_tmpdir=False)[(scheduler, shape)]
    scratch = _render_all(tmp_path / "scratch", use_tmpdir=True)[(scheduler, shape)]

    assert f'mkdir -p "{workdir}/${{SIM_PATH}}/lammps"' in scratch
    assert f'cp "$1" "{workdir}/${{SIM_PATH}}/lammps/" 2>/dev/null || true' in scratch
    assert "save_run_log() { :; }" in plain
    assert "save_run_log() { :; }" not in scratch

    for call in _RESCUE_CALLS[shape]:
        assert call in plain, call
        assert call in scratch, call


@pytest.mark.parametrize("use_tmpdir", [False, True])
@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
@pytest.mark.parametrize("shape", ["array", "combined"])
def test_rendered_scripts_are_valid_bash(tmp_path, use_tmpdir, scheduler, shape):
    """`${PIPESTATUS[0]}` is a bashism and the templates declare `#!/bin/bash`;
    parse the rendered script to prove the added lines are syntactically sound
    in both the scratch and no-scratch shapes."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")

    text = _render_all(tmp_path, use_tmpdir=use_tmpdir)[(scheduler, shape)]
    script = tmp_path / "rendered.sh"
    script.write_text(text, encoding="utf-8")

    proc = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True,
                          check=False)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Execution tests: render a script, put a stub `lmp` on PATH, run it for real.
#
# Everything above asserts on rendered text, which cannot tell whether the
# script *behaves*. Two properties need execution to pin down:
#   * a failing LAMMPS run must still fail the job (the `${PIPESTATUS[0]}`
#     capture -- a future edit back to a bare `$?` would test tee's status and
#     silently report every crash as a success), and
#   * a failing run must still leave its log in the workdir, which under
#     use_tmpdir means rescuing it before `exit 1` rather than in the
#     end-of-script copy-back that a failure never reaches. A crashed job is
#     the one whose log matters most: a non-converged ramp right before the
#     crash is exactly what you would want to read afterwards.
# ---------------------------------------------------------------------------

_SIM_REL = "afm/mat0/L1"

_STUB_HEAD = ('#!/bin/bash\n'
              'echo "LAMMPS (2 Aug 2023 - Update 3)"\n'
              'echo "# RAMP_STATUS converged=0 r=1.62 iterations=100"\n')
#: A stub `lmp` that dies, like a real crash, after printing its banner.
_STUB_FAILS = _STUB_HEAD + 'exit 1\n'
#: A stub `lmp` that succeeds -- proves the harness is not trivially red.
_STUB_SUCCEEDS = _STUB_HEAD + 'exit 0\n'
#: Succeeds for system.in, dies for slide.in: the combined script's second
#: failure path, which exits before the end-of-script copy-back.
_STUB_FAILS_ON_SLIDE = _STUB_HEAD + 'case "$*" in *slide.in*) exit 1;; esac\nexit 0\n'


def _run_rendered(base: Path, scheduler: str, shape: str, use_tmpdir: bool,
                  stub: str):
    """Render a job script into a fake submit dir and execute it.

    Returns (CompletedProcess, sorted run*.log names left in the workdir).
    `mpi_command` is emptied so only `lmp` needs stubbing, and `module` is
    stubbed to a no-op since no module system exists here.
    """
    workdir = base / "submit_root"
    lammps = workdir / _SIM_REL / "lammps"
    lammps.mkdir(parents=True)
    for script in ("system.in", "slide.in"):
        (lammps / script).write_text("# lammps\n", encoding="utf-8")

    config = _config()
    config.hpc_settings.mpi_command = ""
    config.hpc_settings.use_tmpdir = use_tmpdir
    config.hpc_settings.scratch_dir = str(base / "scratch")
    generator = HPCScriptGenerator(config)
    ext = ".pbs" if scheduler == "pbs" else ".sh"
    if shape == "array":
        # The array manifest holds *script* paths; SIM_PATH is dirname twice.
        generator.generate_scripts([f"{_SIM_REL}/lammps/system.in"],
                                   workdir / "hpc", scheduler=scheduler)
    else:
        generator.generate_combined_scripts(_combined_manifest(_SIM_REL),
                                            workdir / "hpc", scheduler=scheduler)

    stub_bin = base / "bin"
    stub_bin.mkdir()
    (stub_bin / "lmp").write_text(stub, encoding="utf-8")
    (stub_bin / "module").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    for name in ("lmp", "module"):
        (stub_bin / name).chmod(0o755)

    env = dict(os.environ, PATH=f"{stub_bin}{os.pathsep}{os.environ['PATH']}")
    env["PBS_O_WORKDIR" if scheduler == "pbs" else "SLURM_SUBMIT_DIR"] = str(workdir)
    proc = subprocess.run(["bash", str(workdir / "hpc" / f"run{ext}")],
                          env=env, capture_output=True, text=True,
                          cwd=str(base), check=False)
    logs = sorted(p.name for p in (workdir / _SIM_REL / "lammps").glob("run*.log"))
    return proc, logs


def _require_execution_tools():
    for tool in ("bash", "rsync"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not available")


_EXECUTION_MATRIX = [(scheduler, shape, use_tmpdir)
                     for scheduler in ("pbs", "slurm")
                     for shape in ("array", "combined")
                     for use_tmpdir in (False, True)]


@pytest.mark.parametrize("scheduler, shape, use_tmpdir", _EXECUTION_MATRIX)
def test_a_failing_lammps_run_fails_the_job(tmp_path, scheduler, shape, use_tmpdir):
    """The `${PIPESTATUS[0]}` capture, proven by running it: with `| tee` in
    front of the old `$?` check the script would exit 0 on a dead LAMMPS."""
    _require_execution_tools()
    proc, _ = _run_rendered(tmp_path, scheduler, shape, use_tmpdir, _STUB_FAILS)

    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "Error:" in proc.stdout


@pytest.mark.parametrize("scheduler, shape, use_tmpdir", _EXECUTION_MATRIX)
def test_a_failed_run_still_leaves_its_log_in_the_workdir(
        tmp_path, scheduler, shape, use_tmpdir):
    """`exit 1` fires before the end-of-script copy-back, so under use_tmpdir
    the log died with the scratch directory until `save_run_log` was called on
    the failure path too. Without that call this fails only for
    use_tmpdir=True -- which is the shape the cluster actually runs."""
    _require_execution_tools()
    proc, logs = _run_rendered(tmp_path, scheduler, shape, use_tmpdir, _STUB_FAILS)

    assert proc.returncode != 0
    assert logs == ["run_system.log"], logs


@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
@pytest.mark.parametrize("use_tmpdir", [False, True])
def test_a_combined_phase_two_failure_preserves_both_phase_logs(
        tmp_path, scheduler, use_tmpdir):
    """The combined script has two failure paths. A slide-phase crash must not
    take the *system* phase's log with it -- that is the one holding
    RAMP_STATUS, and it is still only in scratch when phase 2 dies."""
    _require_execution_tools()
    proc, logs = _run_rendered(tmp_path, scheduler, "combined", use_tmpdir,
                               _STUB_FAILS_ON_SLIDE)

    assert proc.returncode != 0
    assert logs == ["run_slide.log", "run_system.log"], logs


@pytest.mark.parametrize("scheduler, shape, use_tmpdir", _EXECUTION_MATRIX)
def test_a_successful_run_exits_zero_and_keeps_its_logs(
        tmp_path, scheduler, shape, use_tmpdir):
    """Control for the two tests above: the harness is not trivially red, and
    the added rescue/enrichment lines do not break a green run."""
    _require_execution_tools()
    expected = ["run_system.log"] if shape == "array" else [
        "run_slide.log", "run_system.log"]
    proc, logs = _run_rendered(tmp_path, scheduler, shape, use_tmpdir,
                               _STUB_SUCCEEDS)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert logs == expected, logs
    assert "Simulation completed successfully" in proc.stdout
