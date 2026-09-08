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


def _combined_manifest() -> JobManifest:
    manifest = JobManifest()
    manifest.add_job(JobEntry(job_id="sys", simulation_path="afm/mat0/L1",
                               lammps_script="system.in", material="mat0"))
    manifest.add_job(JobEntry(job_id="sld", simulation_path="afm/mat0/L1",
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

#: The scratch copy-back line(s), per shape.
_COPY_BACK = {
    "array": ['cp "${{LOG_PATH}}" "{workdir}/${{SIM_PATH}}/lammps/" 2>/dev/null || true'],
    "combined": ['cp ${{LAMMPS_SIM_PATH}}/lammps/run_system.log '
                 '"{workdir}/${{SIM_PATH}}/lammps/" 2>/dev/null || true',
                 'cp ${{LAMMPS_SIM_PATH}}/lammps/run_slide.log '
                 '"{workdir}/${{SIM_PATH}}/lammps/" 2>/dev/null || true'],
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
    text = _render_all(tmp_path)[(scheduler, shape)]
    log_arg = _ENRICH_LOG_ARG[shape].format(workdir=workdir)

    expected = (f'FrictionSim2D hpc enrich-provenance "{workdir}/${{SIM_PATH}}" '
                f'{log_arg} || true')
    assert expected in text, text
    # Guarded and non-fatal: a node without the package must still finish green.
    assert "if command -v FrictionSim2D >/dev/null 2>&1; then" in text


@pytest.mark.parametrize("scheduler, workdir", [
    ("pbs", "$PBS_O_WORKDIR"), ("slurm", "$SLURM_SUBMIT_DIR")])
@pytest.mark.parametrize("shape", ["array", "combined"])
def test_tmpdir_variant_copies_the_run_log_back_out_of_scratch(
        tmp_path, scheduler, workdir, shape):
    """Under use_tmpdir the job runs in scratch and copies back only
    results/ visuals/ data/ and load_*.data -- a log left out of that list is
    deleted with the scratch directory, taking the RAMP_STATUS lines with it."""
    plain = _render_all(tmp_path / "plain", use_tmpdir=False)[(scheduler, shape)]
    scratch = _render_all(tmp_path / "scratch", use_tmpdir=True)[(scheduler, shape)]

    assert f'mkdir -p "{workdir}/${{SIM_PATH}}/lammps"' in scratch
    for line in _COPY_BACK[shape]:
        assert line.format(workdir=workdir) in scratch, line
    # The copy-back is inside `{% if use_tmpdir %}`: absent without it.
    assert '/lammps/" 2>/dev/null' not in plain


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
