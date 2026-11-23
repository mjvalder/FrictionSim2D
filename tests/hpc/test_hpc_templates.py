"""Every hpc/*.j2 must be rendered by some test: they are reached only by
f-string name construction (`scripts.py` builds `f"{scheduler}_array.j2"` /
`f"{scheduler}_combined.j2"`), so a filename grep across frictionsim2d/ finds no static
reference and a typo in one scheduler/shape combination would ship undetected.

Renders through HPCScriptGenerator directly (not the now-deleted
create_hpc_package, which only ever took the plain array path -- see hpc.md
H-7), covering both schedulers and both script shapes so all four templates
are exercised. `generate_two_phase_scripts` reuses the *_array.j2 templates
already covered here, so it needs no separate rendering for template
liveness (its own dependency-submission logic is out of scope per the task's
Phase 2c deferral).

Both schedulers render the same minimal shape -- environment, copy to scratch,
`lmp -log`, copy back -- so the execution and preflight sections at the bottom
run over both. The one real difference is that SLURM does not expand shell
variables in `#SBATCH` directives, which is pinned separately.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from frictionsim2d.hpc.manifest import JobEntry, JobManifest
from frictionsim2d.hpc.scripts import HPCConfig, HPCScriptGenerator

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "frictionsim2d" / "templates" / "hpc"


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


#: Which rendered file proves each template was reached, as
#: {template: (scheduler, shape, rendered file name)}. The shapes are rendered
#: below; `preflight_sh.j2` is scheduler-neutral (its dialect comes from
#: `_PREFLIGHT_DIALECT`) and is covered through the PBS render.
_RENDERED_BY = {
    "pbs_array.j2": ("pbs", "array", "run.pbs"),
    "pbs_combined.j2": ("pbs", "combined", "run.pbs"),
    "pbs_preflight.j2": ("pbs", "combined", "preflight.pbs"),
    "slurm_array.j2": ("slurm", "array", "run.sh"),
    "slurm_combined.j2": ("slurm", "combined", "run.sh"),
    "slurm_preflight.j2": ("slurm", "combined", "preflight.slurm"),
    "preflight_sh.j2": ("pbs", "combined", "preflight.sh"),
}


def test_every_hpc_template_is_rendered_by_the_script_generator(tmp_path):
    # Partials (leading underscore) are reached through {% include %}.
    on_disk = {q.name for q in TEMPLATE_DIR.glob("*.j2") if not q.name.startswith("_")}
    assert on_disk, f"no templates found under {TEMPLATE_DIR}"
    assert on_disk == set(_RENDERED_BY), (
        "a template is not rendered by any test, or _RENDERED_BY names one that "
        f"no longer exists: {on_disk ^ set(_RENDERED_BY)}")

    written = {}
    for scheduler in ("pbs", "slurm"):
        for shape in ("array", "combined"):
            generator = HPCScriptGenerator(_config())
            out = tmp_path / scheduler / shape
            if shape == "array":
                generator.generate_scripts(["mat0/L1"], out, scheduler=scheduler)
            else:
                generator.generate_combined_scripts(_combined_manifest(), out,
                                                    scheduler=scheduler)
                generator.generate_preflight(out)
            for path in out.iterdir():
                written[(scheduler, shape, path.name)] = path.read_text(encoding="utf-8")

    for template, key in _RENDERED_BY.items():
        assert key in written, f"{template}: nothing rendered {key}"
        text = written[key]
        assert "{%" not in text and "{{" not in text, template
        if key[1] == "combined" and template.endswith(("array.j2", "combined.j2")):
            # Combined scripts hardcode the two-phase filenames rather than
            # reading them from a manifest file.
            assert "system.in" in text and "slide.in" in text, template
        elif key[1] == "array":
            # Array scripts resolve their per-task path from a manifest file.
            assert "sed -n" in text, template


def _render_all(tmp_path, use_tmpdir=False, **settings):
    """Render every scheduler x shape combination; returns {(sch, shape): text}."""
    texts = {}
    for scheduler in ("pbs", "slurm"):
        script_ext = ".pbs" if scheduler == "pbs" else ".sh"
        config = _config()
        config.hpc_settings.use_tmpdir = use_tmpdir
        config.hpc_settings.scratch_dir = "/scratch"
        for key, value in settings.items():
            setattr(config.hpc_settings, key, value)
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


@pytest.mark.parametrize("use_tmpdir", [False, True])
@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
@pytest.mark.parametrize("shape", ["array", "combined"])
def test_rendered_scripts_are_valid_bash(tmp_path, use_tmpdir, scheduler, shape):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")

    text = _render_all(tmp_path, use_tmpdir=use_tmpdir)[(scheduler, shape)]
    script = tmp_path / "rendered.sh"
    script.write_text(text, encoding="utf-8")

    proc = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True,
                          check=False)
    assert proc.returncode == 0, proc.stderr


_SIM_REL = "afm/mat0/L1"


def _stub_bin(base: Path, lmp: str) -> Path:
    stub_bin = base / "bin"
    stub_bin.mkdir()
    (stub_bin / "lmp").write_text(lmp, encoding="utf-8")
    (stub_bin / "module").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    for name in ("lmp", "module"):
        (stub_bin / name).chmod(0o755)
    return stub_bin


# ---------------------------------------------------------------------------
# Minimal job scripts, PBS and SLURM alike.
#
# Environment (modules and/or a conda prefix), copy to scratch if enabled,
# `lmp -in <deck> -log <deck dir>/run_<deck>.log`, copy results/visuals/data
# and the run logs back -- also after a failure -- and remove scratch. Nothing
# else: no tee, no in-job provenance enrichment (`hpc enrich-provenance -r`
# runs on the workstation over the rsynced logs).
# ---------------------------------------------------------------------------

#: A stub `lmp` that behaves like LAMMPS where these scripts can see it: it
#: needs its deck relative to the cwd (exit 3 otherwise), writes the `-log`
#: file itself, and leaves a results file. `FAIL_ON` makes it die after that.
_PBS_STUB = r'''#!/bin/bash
[ "$1" = -h ] && { printf 'Large-scale Atomic/Molecular Massively Parallel Simulator - 2 Aug 2023\n\n* Pair styles:\n  sw   lj/cut   hybrid/overlay\n'; exit 0; }
while [ $# -gt 0 ]; do
    case $1 in -in) deck=$2; shift ;; -log) log=$2; shift ;; esac
    shift
done
[ -f "$deck" ] || { echo "stub lmp: no deck $deck in $PWD" >&2; exit 3; }
for p in $(awk '$1=="read_data" && $2 !~ /\$/ {print $2}' "$deck"); do
    [ -e "$p" ] || { echo "stub lmp: no $p in $PWD" >&2; exit 4; }
done
echo "LAMMPS (2 Aug 2023)" >"$log"
sim=$(dirname "$(dirname "$deck")")
echo done >"$sim/results/$(basename "$deck" .in).txt"
case "$deck" in *"${FAIL_ON:-never}"*) exit 1 ;; esac
exit 0
'''


#: What running a rendered job script by hand needs, per scheduler: the file
#: name the generator gives it, the environment variable naming the submit
#: directory, the one holding the job id the scratch path is built from, the
#: array index to clear, and the preflight submit wrapper's name.
_DIALECT = {
    "pbs": {"script": "run.pbs", "workdir_var": "PBS_O_WORKDIR",
            "jobid_var": "PBS_JOBID", "index_var": "PBS_ARRAY_INDEX",
            "preflight_job": "preflight.pbs"},
    "slurm": {"script": "run.sh", "workdir_var": "SLURM_SUBMIT_DIR",
              "jobid_var": "SLURM_JOB_ID", "index_var": "SLURM_ARRAY_TASK_ID",
              "preflight_job": "preflight.slurm"},
}


def _job_tree(base: Path, sim_rel: str = _SIM_REL) -> Path:
    workdir = base / "submit_root"
    lammps = workdir / sim_rel / "lammps"
    lammps.mkdir(parents=True)
    for script in ("system.in", "slide.in"):
        (lammps / script).write_text("# lammps\n", encoding="utf-8")
    for name in ("results", "visuals", "data"):   # the builder writes these
        (workdir / sim_rel / name).mkdir()
    return workdir


def _run_job(base: Path, workdir: Path, scheduler: str, shape: str, use_tmpdir: bool,
             sim_rel: str = _SIM_REL, fail_on: str = "", **settings):
    config = _config()
    config.hpc_settings.mpi_command = ""
    config.hpc_settings.use_tmpdir = use_tmpdir
    config.hpc_settings.scratch_dir = str(base / "scratch")
    for key, value in settings.items():
        setattr(config.hpc_settings, key, value)
    generator = HPCScriptGenerator(config)
    if shape == "array":
        generator.generate_scripts([f"{sim_rel}/lammps/system.in"],
                                   workdir / "hpc", scheduler=scheduler)
    else:
        generator.generate_combined_scripts(_combined_manifest(sim_rel),
                                            workdir / "hpc", scheduler=scheduler)
    (base / "scratch").mkdir(exist_ok=True)
    stub_bin = _stub_bin(base, _PBS_STUB)
    dialect = _DIALECT[scheduler]
    env = dict(os.environ, PATH=f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
               FAIL_ON=fail_on)
    env[dialect["workdir_var"]] = str(workdir)
    env[dialect["jobid_var"]] = "42"
    env.pop(dialect["index_var"], None)
    return subprocess.run(["bash", str(workdir / "hpc" / dialect["script"])], env=env,
                          capture_output=True, text=True, cwd=str(base), check=False)


def _require_bash():
    if shutil.which("bash") is None:
        pytest.skip("bash not available")


@pytest.mark.parametrize("use_tmpdir", [False, True])
@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
@pytest.mark.parametrize("shape", ["array", "combined"])
def test_job_scripts_carry_no_troubleshooting_machinery(tmp_path, scheduler, shape, use_tmpdir):
    text = _render_all(tmp_path, use_tmpdir=use_tmpdir)[("pbs", shape)]

    for gone in ("tee", "PIPESTATUS", "save_run_log", "enrich-provenance",
                 "rsync", "-l none", "STATUS", "RUN_ROOT", "while ", "exit"):
        assert gone not in text, (gone, text)
    assert '-log "$SIM/lammps/run_' in text
    assert "CASE=${LINE%%$'\\t'*}" in text
    assert ("cp -a" in text) == use_tmpdir
    assert ('rm -rf "$SCRATCH"' in text) == use_tmpdir


def test_the_environment_block_follows_the_settings(tmp_path):
    modules_only = _render_all(tmp_path / "m", modules=["LAMMPS/2024"])[("pbs", "array")]
    conda_only = _render_all(tmp_path / "c", modules=None,
                             conda_env="$HOME/envs/mlip")[("pbs", "array")]

    assert "module purge" in modules_only
    assert "module load LAMMPS/2024" in modules_only
    assert "CONDA_ENV" not in modules_only

    assert "module" not in conda_only
    assert 'CONDA_ENV="$HOME/envs/mlip"' in conda_only
    assert 'export PATH="$CONDA_ENV/bin:$PATH"' in conda_only
    assert 'export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"' in conda_only


@pytest.mark.parametrize("use_tmpdir", [False, True])
@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
@pytest.mark.parametrize("shape", ["array", "combined"])
def test_a_successful_run_brings_results_and_logs_home(tmp_path, scheduler, shape, use_tmpdir):
    _require_bash()
    workdir = _job_tree(tmp_path)
    proc = _run_job(tmp_path, workdir, scheduler, shape, use_tmpdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    sim = workdir / _SIM_REL
    phases = ["system"] if shape == "array" else ["slide", "system"]
    assert sorted(p.name for p in (sim / "lammps").glob("run_*.log")) == [
        f"run_{p}.log" for p in phases]
    assert sorted(p.name for p in (sim / "results").iterdir()) == [f"{p}.txt" for p in phases]
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize("use_tmpdir", [False, True])
@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
@pytest.mark.parametrize("shape", ["array", "combined"])
def test_a_failed_run_still_brings_its_log_home(tmp_path, scheduler, shape, use_tmpdir):
    _require_bash()
    workdir = _job_tree(tmp_path)
    _run_job(tmp_path, workdir, scheduler, shape, use_tmpdir, fail_on="system.in")

    # The combined job still runs slide.in: it slides the loads system.in wrote
    # before failing and skips the rest (the array job here is the one deck).
    phases = ["system"] if shape == "array" else ["slide", "system"]
    assert sorted(p.name for p in (workdir / _SIM_REL / "lammps").glob("run_*.log")) == [
        f"run_{p}.log" for p in phases]
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
def test_a_slide_failure_keeps_both_phase_logs(tmp_path, scheduler):
    _require_bash()
    workdir = _job_tree(tmp_path)
    _run_job(tmp_path, workdir, scheduler, "combined", True, fail_on="slide.in")

    assert sorted(p.name for p in (workdir / _SIM_REL / "lammps").glob("run_*.log")) == [
        "run_slide.log", "run_system.log"]


@pytest.mark.parametrize("shape", ["array", "combined"])
def test_the_manifest_names_the_case_directory_to_copy(tmp_path, shape):
    """The case is the nearest ancestor holding provenance/ or build/; without
    one it is the simulation directory itself."""
    workdir = _job_tree(tmp_path)
    _job_tree(tmp_path / "other", "afm/mat1/L1")
    (workdir / "afm" / "mat0" / "build").mkdir()
    generator = HPCScriptGenerator(_config())
    if shape == "array":
        generator.generate_scripts([f"{_SIM_REL}/lammps/slide.in"], workdir / "hpc")
        manifest, expected = "manifest_1.txt", f"afm/mat0\t{_SIM_REL}/lammps/slide.in"
    else:
        generator.generate_combined_scripts(_combined_manifest(), workdir / "hpc")
        manifest, expected = "manifest_combined.txt", f"afm/mat0\t{_SIM_REL}"
    assert (workdir / "hpc" / manifest).read_text(encoding="utf-8").strip() == expected

    bare = HPCScriptGenerator(_config())
    bare.generate_combined_scripts(_combined_manifest("afm/mat1/L1"),
                                   tmp_path / "other" / "submit_root" / "hpc")
    assert (tmp_path / "other" / "submit_root" / "hpc" / "manifest_combined.txt").read_text(
        encoding="utf-8").strip() == "afm/mat1/L1\tafm/mat1/L1"


@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
def test_the_scratch_copy_takes_the_nearest_provenance_ancestor(tmp_path, scheduler):
    _require_bash()
    workdir = _job_tree(tmp_path)
    shared = workdir / "afm" / "mat0" / "provenance"
    shared.mkdir()
    (shared / "pot.sw").write_text("x\n", encoding="utf-8")
    # The stub exits 4 unless the shared file made it to scratch.
    deck = workdir / _SIM_REL / "lammps" / "system.in"
    deck.write_text("read_data afm/mat0/provenance/pot.sw\n", encoding="utf-8")
    proc = _run_job(tmp_path, workdir, scheduler, "array", True)

    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _preflight_tree(base: Path, *, potential_present=True) -> Path:
    """A combined AFM tree whose decks read real files, plus a conda prefix
    holding the stub lmp."""
    workdir = _job_tree(base)
    sim = workdir / _SIM_REL
    (sim / "data" / "sheet.data").write_text("x\n", encoding="utf-8")
    pot = sim / "lammps" / "MoS.sw"
    if potential_present:
        pot.write_text("x\n", encoding="utf-8")
    (sim / "lammps" / "system.in.settings").write_text(
        f"pair_style sw\npair_coeff * * {_SIM_REL}/lammps/MoS.sw Mo S\n", encoding="utf-8")
    (sim / "lammps" / "system.in").write_text(
        f"read_data {_SIM_REL}/data/sheet.data\ninclude {_SIM_REL}/lammps/system.in.settings\n",
        encoding="utf-8")
    (sim / "lammps" / "slide.in").write_text(
        f"read_data {_SIM_REL}/data/load_$(v_find)N.data\n"
        f"include {_SIM_REL}/lammps/system.in.settings\n", encoding="utf-8")
    conda = base / "conda"
    (conda / "bin").mkdir(parents=True)
    (conda / "bin" / "lmp").write_text(_PBS_STUB, encoding="utf-8")
    (conda / "bin" / "lmp").chmod(0o755)
    return workdir


def _write_preflight(base: Path, workdir: Path, scheduler: str = "pbs"):
    config = _config()
    config.hpc_settings.modules = None
    config.hpc_settings.conda_env = str(base / "conda")
    config.hpc_settings.mpi_command = ""
    config.hpc_settings.use_tmpdir = True
    config.hpc_settings.scratch_dir = str(base / "scratch")
    (base / "scratch").mkdir(exist_ok=True)
    generator = HPCScriptGenerator(config)
    generator.generate_combined_scripts(_combined_manifest(), workdir / "hpc",
                                        scheduler=scheduler)
    return generator.generate_preflight(workdir / "hpc")


def _run_preflight(workdir: Path):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PBS_", "SLURM_"))}
    return subprocess.run(["bash", str(workdir / "hpc" / "preflight.sh")], env=env,
                          capture_output=True, text=True, cwd="/", check=False)


@pytest.mark.parametrize("scheduler, directives", [
    ("pbs", ["#PBS -l select=1:ncpus=32:mem=62gb:mpiprocs=32",
             "#PBS -N friction2d_preflight"]),
    ("slurm", ["#SBATCH --job-name=friction2d_preflight",
               "#SBATCH --ntasks-per-node=32",
               "#SBATCH --time=00:10:00"]),
])
def test_preflight_writes_a_check_script_and_its_submit_wrapper(
        tmp_path, scheduler, directives):
    workdir = _preflight_tree(tmp_path)
    sh, wrapper = _write_preflight(tmp_path, workdir, scheduler)

    assert sh.name == "preflight.sh" and os.access(sh, os.X_OK)
    assert wrapper.name == _DIALECT[scheduler]["preflight_job"]
    text = wrapper.read_text(encoding="utf-8")
    for directive in directives:
        assert directive in text, text
    assert "bash hpc/preflight.sh" in text
    assert "lmp -in" not in sh.read_text(encoding="utf-8")


@pytest.mark.parametrize("script", ["run.sh", "preflight.slurm"])
def test_no_sbatch_directive_carries_a_shell_variable(tmp_path, script):
    """SLURM does not expand shell variables in #SBATCH directives -- a literal
    $SLURM_SUBMIT_DIR in --output is taken as a directory name and the job is
    rejected before it starts. Relative paths resolve against the submit
    directory, which is what the job wanted anyway. (PBS *does* expand
    $PBS_O_WORKDIR in -o, which is why only SLURM needs this.)"""
    workdir = _preflight_tree(tmp_path)
    _write_preflight(tmp_path, workdir, "slurm")

    directives = [ln for ln in (workdir / "hpc" / script).read_text(
        encoding="utf-8").splitlines() if ln.startswith("#SBATCH")]
    assert directives
    assert not [ln for ln in directives if "$" in ln], directives
    assert any(ln.startswith("#SBATCH --output=logs/") for ln in directives), directives


@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
def test_preflight_passes_on_a_complete_tree(tmp_path, scheduler):
    _require_bash()
    workdir = _preflight_tree(tmp_path)
    _write_preflight(tmp_path, workdir, scheduler)

    proc = _run_preflight(workdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[FAIL]" not in proc.stdout
    assert (f"{_DIALECT[scheduler]['script']}: array 1-1 matches manifest_combined.txt"
            in proc.stdout)
    assert "all 2 decks and the files they read resolve" in proc.stdout
    assert "pair_style sw" in proc.stdout
    assert "LAMMPS version 2 Aug 2023" in proc.stdout


def test_preflight_names_a_missing_potential_file(tmp_path):
    _require_bash()
    workdir = _preflight_tree(tmp_path, potential_present=False)
    _write_preflight(tmp_path, workdir)

    proc = _run_preflight(workdir)

    assert proc.returncode == 1, proc.stdout
    assert any("[FAIL]" in ln and "MoS.sw" in ln for ln in proc.stdout.splitlines()), proc.stdout


@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
def test_preflight_catches_an_array_range_that_disagrees_with_its_manifest(
        tmp_path, scheduler):
    _require_bash()
    workdir = _preflight_tree(tmp_path)
    _write_preflight(tmp_path, workdir, scheduler)
    manifest = workdir / "hpc" / "manifest_combined.txt"
    manifest.write_text(manifest.read_text(encoding="utf-8") + f"{_SIM_REL}\n",
                        encoding="utf-8")

    proc = _run_preflight(workdir)

    assert proc.returncode == 1
    assert "array 1-1 but manifest_combined.txt has 2 entries" in proc.stdout


def test_preflight_fails_a_pair_style_the_build_lacks(tmp_path):
    _require_bash()
    workdir = _preflight_tree(tmp_path)
    settings = workdir / _SIM_REL / "lammps" / "system.in.settings"
    settings.write_text(settings.read_text(encoding="utf-8").replace(
        "pair_style sw", "pair_style rebomos"), encoding="utf-8")
    _write_preflight(tmp_path, workdir)

    proc = _run_preflight(workdir)

    assert proc.returncode == 1
    assert "pair_style rebomos is not in this LAMMPS build" in proc.stdout


@pytest.mark.parametrize("scheduler", ["pbs", "slurm"])
@pytest.mark.parametrize("modules", [["lammps"], None])
def test_rendered_preflight_is_valid_bash(tmp_path, modules, scheduler):
    _require_bash()
    config = _config()
    config.hpc_settings.modules = modules
    config.hpc_settings.conda_env = None if modules else "$HOME/envs/x"
    generator = HPCScriptGenerator(config)
    generator.generate_combined_scripts(_combined_manifest(), tmp_path / "hpc",
                                        scheduler=scheduler)
    for path in generator.generate_preflight(tmp_path / "hpc"):
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True,
                              check=False)
        assert proc.returncode == 0, (path.name, proc.stderr)


def test_preflight_refuses_before_any_job_script_or_across_schedulers(tmp_path):
    generator = HPCScriptGenerator(_config())
    with pytest.raises(ValueError, match="before"):
        generator.generate_preflight(tmp_path)
    generator.generate_combined_scripts(_combined_manifest(), tmp_path / "hpc",
                                        scheduler="slurm")
    generator.generate_scripts(["mat0/L1"], tmp_path / "hpc", scheduler="pbs")
    with pytest.raises(ValueError, match="mix of schedulers"):
        generator.generate_preflight(tmp_path / "hpc")
