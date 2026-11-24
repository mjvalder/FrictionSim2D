"""HPCScriptGenerator: the live path behind `FrictionSim2D hpc generate` and
`dispatch_hpc_generation`. Exercises both schedulers and both script shapes
(plain array, combined system+slide) directly against the generator, since
`create_hpc_package` (the previous indirection this file tested through) is
unreferenced by the CLI and takes only the array path -- see hpc.md H-7.
"""
from pathlib import Path

import pytest

from frictionsim2d.hpc.manifest import JobEntry, JobManifest
from frictionsim2d.hpc.scripts import HPCConfig, HPCScriptGenerator


def _config(**overrides) -> HPCConfig:
    config = HPCConfig()
    config.hpc_settings.modules = ["lammps"]
    for key, value in overrides.items():
        setattr(config.hpc_settings, key, value)
    return config


def _combined_manifest(sim_path: str = "afm/mat0/L1") -> JobManifest:
    manifest = JobManifest()
    manifest.add_job(JobEntry(job_id="sys", simulation_path=sim_path,
                               lammps_script="system.in", material="mat0"))
    manifest.add_job(JobEntry(job_id="sld", simulation_path=sim_path,
                               lammps_script="slide.in", material="mat0"))
    return manifest


@pytest.mark.parametrize("scheduler, script_name, directive", [
    ("pbs", "run.pbs", "#PBS"),
    ("slurm", "run.sh", "#SBATCH"),
])
def test_generate_scripts_writes_the_scheduler_s_own_script(tmp_path, scheduler, script_name, directive):
    generator = HPCScriptGenerator(_config())
    generator.generate_scripts(["mat0/L1"], tmp_path / "hpc", scheduler=scheduler)

    script = tmp_path / "hpc" / script_name
    assert script.exists(), sorted(p.name for p in (tmp_path / "hpc").iterdir())
    assert directive in script.read_text(encoding="utf-8")


def test_generated_script_carries_the_requested_modules(tmp_path):
    generator = HPCScriptGenerator(_config(modules=["lammps/2023", "python/3.11"]))
    generator.generate_scripts(["mat0/L1"], tmp_path / "hpc")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "module load lammps/2023" in text
    assert "module load python/3.11" in text


def test_walltime_and_cpu_resources_reach_the_pbs_script(tmp_path):
    """PBS's `select=` line encodes cpus_per_node/memory_gb and, since the
    node-count fix, `nodes` too (see test_pbs_select_line_honours_node_count
    and test_pbs_select_line_is_unchanged_for_the_single_node_default)."""
    generator = HPCScriptGenerator(_config(walltime_hours=8, num_cpus=16, memory_gb=64))
    generator.generate_scripts(["mat0/L1"], tmp_path / "hpc")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "walltime=8:00:00" in text
    assert "ncpus=16" in text
    assert "mem=64gb" in text


def test_pbs_select_line_honours_node_count(tmp_path):
    """A multi-node PBS config must put the real node count in `select=`,
    not the hardcoded `1` that silently dropped `num_nodes` before this fix."""
    generator = HPCScriptGenerator(_config(num_nodes=4, num_cpus=16, memory_gb=64))
    generator.generate_scripts(["mat0/L1"], tmp_path / "hpc")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "select=4:ncpus=16:mem=64gb:mpiprocs=16" in text


def test_pbs_select_line_is_unchanged_for_the_single_node_default(tmp_path):
    """Compatibility check: with the default num_nodes=1, the `select=` line
    must render byte-identical to what it produced before the fix."""
    generator = HPCScriptGenerator(_config())
    generator.generate_scripts(["mat0/L1"], tmp_path / "hpc")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "select=1:ncpus=32:mem=62gb:mpiprocs=32" in text


def test_node_count_reaches_the_slurm_script(tmp_path):
    generator = HPCScriptGenerator(_config(num_nodes=4, num_cpus=8, memory_gb=32))
    generator.generate_scripts(["mat0/L1"], tmp_path / "hpc", scheduler="slurm")

    text = (tmp_path / "hpc" / "run.sh").read_text(encoding="utf-8")
    assert "--nodes=4" in text
    assert "--ntasks-per-node=8" in text
    assert "--mem=32G" in text


def _submit_lines(submit_all_path: Path) -> list:
    """The lines in submit_all.txt that actually submit a job."""
    return [ln for ln in submit_all_path.read_text(encoding="utf-8").splitlines()
            if ln.startswith("qsub") or ln.startswith("sbatch")]


def test_submit_all_lists_one_entry_per_array_chunk(tmp_path):
    """submit_all.txt has one qsub/sbatch line per *generated script*, not per
    simulation -- max_array_size chunks many simulations into one script."""
    generator = HPCScriptGenerator(_config(max_array_size=1))
    generator.generate_scripts(["a", "b", "c"], tmp_path / "hpc")

    assert len(_submit_lines(tmp_path / "hpc" / "submit_all.txt")) == 3


def test_submit_all_lists_one_entry_per_script_not_per_simulation(tmp_path):
    """With the default max_array_size, several simulations still fit in a
    single array script, so submit_all.txt gets exactly one line."""
    generator = HPCScriptGenerator(_config())
    generator.generate_scripts(["a", "b", "c"], tmp_path / "hpc")

    assert len(_submit_lines(tmp_path / "hpc" / "submit_all.txt")) == 1


def test_no_simulations_produces_an_empty_submit_all_without_raising(tmp_path):
    """Contract finding: an empty simulation list is not rejected -- it
    silently yields zero scripts and a submit_all.txt with no submit lines."""
    generator = HPCScriptGenerator(_config())
    scripts = generator.generate_scripts([], tmp_path / "hpc")

    assert scripts == []
    text = (tmp_path / "hpc" / "submit_all.txt").read_text(encoding="utf-8")
    assert "qsub" not in text


def test_empty_modules_list_is_rejected(tmp_path):
    generator = HPCScriptGenerator(_config(modules=[]))
    with pytest.raises(ValueError, match="modules"):
        generator.generate_scripts(["mat0/L1"], tmp_path / "hpc")


def test_a_conda_env_stands_in_for_modules(tmp_path):
    generator = HPCScriptGenerator(_config(modules=[], conda_env="$HOME/envs/lammps"))
    generator.generate_scripts(["mat0/L1"], tmp_path / "hpc")

    assert 'CONDA_ENV="$HOME/envs/lammps"' in (tmp_path / "hpc" / "run.pbs").read_text(
        encoding="utf-8")


def test_use_tmpdir_without_scratch_dir_is_rejected(tmp_path):
    generator = HPCScriptGenerator(_config(use_tmpdir=True, scratch_dir=None))
    with pytest.raises(ValueError, match="scratch_dir"):
        generator.generate_scripts(["mat0/L1"], tmp_path / "hpc")


def test_generate_scripts_rejects_an_unknown_scheduler(tmp_path):
    generator = HPCScriptGenerator(_config())
    with pytest.raises(ValueError, match="Unknown scheduler"):
        generator.generate_scripts(["mat0/L1"], tmp_path / "hpc", scheduler="lsf")


@pytest.mark.parametrize("scheduler, script_name, directive", [
    ("pbs", "run.pbs", "#PBS"),
    ("slurm", "run.sh", "#SBATCH"),
])
def test_combined_script_runs_system_then_slide_for_both_schedulers(tmp_path, scheduler, script_name, directive):
    generator = HPCScriptGenerator(_config())
    generator.generate_combined_scripts(_combined_manifest(), tmp_path / "hpc", scheduler=scheduler)

    text = (tmp_path / "hpc" / script_name).read_text(encoding="utf-8")
    assert directive in text
    assert "system.in" in text
    assert "slide.in" in text
    assert (tmp_path / "hpc" / "submit_jobs.sh").exists()


@pytest.mark.parametrize("kind", ["array", "combined", "two_phase"])
def test_the_download_hint_brings_the_run_logs_home(tmp_path, kind):
    """The rsync-down line in the submission helper is what the user actually
    copies. It used to pull only results/ and visuals/, which would have left
    every run log -- the sole on-disk home of the settle loop's RAMP_STATUS
    lines -- stranded on the cluster, with nothing for
    `hpc enrich-provenance -r` to read on this side.
    """
    generator = HPCScriptGenerator(_config())
    out_dir = tmp_path / "hpc"
    if kind == "array":
        generator.generate_scripts(["mat0/L1"], out_dir)
        helper = out_dir / "submit_all.txt"
    elif kind == "combined":
        generator.generate_combined_scripts(_combined_manifest(), out_dir)
        helper = out_dir / "submit_jobs.sh"
    else:
        generator.generate_two_phase_scripts(_combined_manifest(), out_dir)
        helper = out_dir / "submit_jobs.sh"

    text = helper.read_text(encoding="utf-8")
    download = [ln for ln in text.splitlines() if "rsync -avzhP" in ln]
    assert download, text
    for line in download:
        assert '--include="run*.log"' in line, line
        # Still ahead of the catch-all exclude, or the include never applies.
        assert line.index('--include="run*.log"') < line.index('--exclude="*"'), line


def test_generate_scripts_rejects_an_unknown_scheduler(tmp_path):
    with pytest.raises(ValueError, match="Unknown scheduler"):
        HPCScriptGenerator(_config()).generate_scripts(
            ["mat0/L1"], tmp_path / "hpc", scheduler="lsf")


@pytest.mark.parametrize("scheduler, script, expected", [
    ("pbs", "run.pbs", "#PBS -o $PBS_O_WORKDIR/logs/friction2d_out.txt"),
    ("slurm", "run.sh", "#SBATCH --output=logs/friction2d_out.txt"),
])
def test_base_dir_defaults_to_the_schedulers_submit_directory(
        tmp_path, scheduler, script, expected):
    HPCScriptGenerator(_config()).generate_scripts(
        ["mat0/L1"], tmp_path / "hpc", scheduler=scheduler)

    assert expected in (tmp_path / "hpc" / script).read_text(encoding="utf-8")


def test_an_explicit_base_dir_roots_the_default_log_directory(tmp_path):
    HPCScriptGenerator(_config()).generate_scripts(
        ["mat0/L1"], tmp_path / "hpc", base_dir="/work/sim")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "#PBS -o /work/sim/logs/friction2d_out.txt" in text


def _rsync_lines(path: Path) -> list:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.startswith("# rsync")]


@pytest.mark.parametrize("host, home, target", [
    (None, None, "<HPC_HOST>:<HPC_HOME>/"),
    ("me@hpc", "/home/me/", "me@hpc:/home/me/"),
])
def test_submit_files_carry_the_rsync_lines_for_the_host_and_home(
        tmp_path, host, home, target):
    out = tmp_path / "hpc"
    generator = HPCScriptGenerator(_config(hpc_host=host, hpc_home=home))
    generator.generate_scripts(["mat0/L1"], out)
    master = _rsync_lines(out / "submit_all.txt")

    manifest = _combined_manifest()
    for scheduler_dir, method in (("comb", generator.generate_combined_scripts),
                                  ("two", generator.generate_two_phase_scripts)):
        method(manifest, tmp_path / scheduler_dir, simulation_root=tmp_path)
    combined = _rsync_lines(tmp_path / "comb" / "submit_jobs.sh")
    two_phase = _rsync_lines(tmp_path / "two" / "submit_jobs.sh")

    for lines in (master, combined, two_phase):
        assert lines[0] == f"# rsync -rvltoD {tmp_path.name} {target}"
        assert lines[1].endswith(f'--exclude="*" {target}{tmp_path.name} ./')
        assert '--include="run*.log"' in lines[1]


def test_hpc_home_gives_one_shared_log_directory(tmp_path):
    HPCScriptGenerator(_config(hpc_home="/home/me/work/")).generate_scripts(
        ["mat0/L1"], tmp_path / "hpc")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "#PBS -o /home/me/work/logs/friction2d_out.txt" in text


def test_an_explicit_log_dir_wins_over_hpc_home(tmp_path):
    HPCScriptGenerator(_config(hpc_home="/home/me/work", log_dir="/scratch/logs")).generate_scripts(
        ["mat0/L1"], tmp_path / "hpc")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "#PBS -o /scratch/logs/friction2d_out.txt" in text


def test_a_relative_log_dir_is_created_beside_the_hpc_directory(tmp_path):
    HPCScriptGenerator(_config(log_dir="joblogs")).generate_scripts(
        ["mat0/L1"], tmp_path / "hpc")

    assert (tmp_path / "joblogs").is_dir()
    assert (tmp_path / "logs").is_dir()


@pytest.mark.parametrize("log_dir", ["$SLURM_SUBMIT_DIR/joblogs", "/abs/joblogs"])
def test_absolute_and_variable_log_dirs_are_not_created_locally(tmp_path, log_dir):
    HPCScriptGenerator(_config(log_dir=log_dir)).generate_scripts(
        ["mat0/L1"], tmp_path / "hpc", scheduler="slurm")

    assert not (tmp_path / "joblogs").exists()
    assert not (tmp_path / "$SLURM_SUBMIT_DIR").exists()


def test_slurm_directive_drops_the_submit_dir_prefix_from_log_dir(tmp_path):
    HPCScriptGenerator(_config(log_dir="$SLURM_SUBMIT_DIR/joblogs")).generate_scripts(
        ["mat0/L1"], tmp_path / "hpc", scheduler="slurm")

    text = (tmp_path / "hpc" / "run.sh").read_text(encoding="utf-8")
    assert "#SBATCH --output=joblogs/friction2d_out.txt" in text


def test_generator_renders_through_the_shared_template_environment():
    from frictionsim2d.interfaces.jinja import PackageLoader
    env = HPCScriptGenerator(_config()).jinja_env
    assert isinstance(env.loader, PackageLoader)
    assert env.trim_blocks and env.lstrip_blocks
    assert not env.keep_trailing_newline
