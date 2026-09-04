"""HPCScriptGenerator: the live path behind `FrictionSim2D hpc generate` and
`dispatch_hpc_generation`. Exercises both schedulers and both script shapes
(plain array, combined system+slide) directly against the generator, since
`create_hpc_package` (the previous indirection this file tested through) is
unreferenced by the CLI and takes only the array path -- see hpc.md H-7.
"""
from pathlib import Path

import pytest

from src.hpc.manifest import JobEntry, JobManifest
from src.hpc.scripts import HPCConfig, HPCScriptGenerator


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
    generator.generate_pbs_scripts(["mat0/L1"], tmp_path / "hpc")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "module load lammps/2023" in text
    assert "module load python/3.11" in text


def test_walltime_and_cpu_resources_reach_the_pbs_script(tmp_path):
    """PBS's `select=` line encodes cpus_per_node/memory_gb and, since the
    node-count fix, `nodes` too (see test_pbs_select_line_honours_node_count
    and test_pbs_select_line_is_unchanged_for_the_single_node_default)."""
    generator = HPCScriptGenerator(_config(walltime_hours=8, num_cpus=16, memory_gb=64))
    generator.generate_pbs_scripts(["mat0/L1"], tmp_path / "hpc")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "walltime=8:00:00" in text
    assert "ncpus=16" in text
    assert "mem=64gb" in text


def test_pbs_select_line_honours_node_count(tmp_path):
    """A multi-node PBS config must put the real node count in `select=`,
    not the hardcoded `1` that silently dropped `num_nodes` before this fix."""
    generator = HPCScriptGenerator(_config(num_nodes=4, num_cpus=16, memory_gb=64))
    generator.generate_pbs_scripts(["mat0/L1"], tmp_path / "hpc")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "select=4:ncpus=16:mem=64gb:mpiprocs=16" in text


def test_pbs_select_line_is_unchanged_for_the_single_node_default(tmp_path):
    """Compatibility check: with the default num_nodes=1, the `select=` line
    must render byte-identical to what it produced before the fix."""
    generator = HPCScriptGenerator(_config())
    generator.generate_pbs_scripts(["mat0/L1"], tmp_path / "hpc")

    text = (tmp_path / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "select=1:ncpus=32:mem=62gb:mpiprocs=32" in text


def test_node_count_reaches_the_slurm_script(tmp_path):
    generator = HPCScriptGenerator(_config(num_nodes=4, num_cpus=8, memory_gb=32))
    generator.generate_slurm_scripts(["mat0/L1"], tmp_path / "hpc")

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
    generator.generate_pbs_scripts(["a", "b", "c"], tmp_path / "hpc")

    assert len(_submit_lines(tmp_path / "hpc" / "submit_all.txt")) == 3


def test_submit_all_lists_one_entry_per_script_not_per_simulation(tmp_path):
    """With the default max_array_size, several simulations still fit in a
    single array script, so submit_all.txt gets exactly one line."""
    generator = HPCScriptGenerator(_config())
    generator.generate_pbs_scripts(["a", "b", "c"], tmp_path / "hpc")

    assert len(_submit_lines(tmp_path / "hpc" / "submit_all.txt")) == 1


def test_no_simulations_produces_an_empty_submit_all_without_raising(tmp_path):
    """Contract finding: an empty simulation list is not rejected -- it
    silently yields zero scripts and a submit_all.txt with no submit lines."""
    generator = HPCScriptGenerator(_config())
    scripts = generator.generate_pbs_scripts([], tmp_path / "hpc")

    assert scripts == []
    text = (tmp_path / "hpc" / "submit_all.txt").read_text(encoding="utf-8")
    assert "qsub" not in text


def test_empty_modules_list_is_rejected(tmp_path):
    generator = HPCScriptGenerator(_config(modules=[]))
    with pytest.raises(ValueError, match="modules"):
        generator.generate_pbs_scripts(["mat0/L1"], tmp_path / "hpc")


def test_use_tmpdir_without_scratch_dir_is_rejected(tmp_path):
    generator = HPCScriptGenerator(_config(use_tmpdir=True, scratch_dir=None))
    with pytest.raises(ValueError, match="scratch_dir"):
        generator.generate_pbs_scripts(["mat0/L1"], tmp_path / "hpc")


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
