"""MLIP HPC script generation: GPU job groups and the per-framework overlay.

Named HPC profiles are gone (merge decision 2); a GPU campaign is one
settings.yaml with ``hpc.gpu`` and ``hpc.frameworks``, merged per job group by
``HPCScriptGenerator._config_for_group`` through
``frictionsim2d.core.config.apply_framework_overlay``. CPU-only byte-identity against
main's templates is pinned in ``test_hpc_templates.py``; these tests pin the
GPU path and the CPU/GPU/framework job split.
"""
import subprocess
from pathlib import Path

import pytest

from frictionsim2d.core.config import GlobalSettings, GPUSettings, FrameworkOverlay
from frictionsim2d.core.mlip import get_mlip
from frictionsim2d.hpc.manifest import JobEntry, JobManifest
from frictionsim2d.hpc.scripts import HPCConfig, HPCScriptGenerator

TEMPLATES = ["pbs_array.j2", "pbs_combined.j2", "slurm_array.j2", "slurm_combined.j2"]


def _gpu_settings(gpu_type="L40S"):
    hpc = GlobalSettings().hpc
    hpc.modules = ["tools/prod", "lammps"]
    hpc.num_cpus = 8
    hpc.memory_gb = 64
    hpc.queue = "gpu72"
    hpc.env_lines = ["export BASE_ENV=1"]
    hpc.gpu = GPUSettings(ngpus=1, gpu_type=gpu_type)
    hpc.frameworks = {
        "deepmd": FrameworkOverlay(binary="$HOME/deepmd-kit/bin/lmp",
                                   env_lines=["export DP_THREADS=1"]),
        "mace": FrameworkOverlay(binary="$HOME/lammps-mace/build/lmp"),
    }
    return hpc


def _group_config(framework, gpu_type="L40S"):
    gen = HPCScriptGenerator(HPCConfig.from_settings(_gpu_settings(gpu_type)))
    return gen._config_for_group((True, framework), split=False)


def _render(template, config, array_size=3):
    gen = HPCScriptGenerator(config)
    ctx = config.to_dict()
    ctx.update({"array_size": array_size, "manifest_file": "m", "manifest_filename": "m.txt",
                "base_dir": "$PBS_O_WORKDIR", "log_dir": "logs", "directive_log_dir": "logs"})
    return gen.jinja_env.get_template(template).render(ctx)


def _manifest(specs):
    jobs = []
    for i, (path, gpu, fw) in enumerate(specs):
        for script in ("system.in", "slide.in"):
            jobs.append(JobEntry(job_id=f"{script[0]}{i}", simulation_path=path,
                                 lammps_script=script, needs_gpu=gpu, framework=fw))
    return JobManifest(jobs=jobs)


def test_cpu_launch_command_is_mains_expression():
    ctx = HPCConfig().to_dict()
    assert ctx["launch_command"] == "mpirun lmp"
    assert ctx["env_lines"] == [] and ctx["gres"] is None and ctx["omp_threads"] == 1
    assert ctx["ntasks_per_node"] == 32 and ctx["cpus_per_task"] == 1


def test_gpu_select_line_exact_string():
    ctx = _group_config("deepmd").to_dict()
    assert ctx["select_single"] == "1:ncpus=8:mem=64gb:mpiprocs=1:ngpus=1:gpu_type=L40S"
    assert ctx["select_multi"] == ctx["select_single"]


def test_gpu_select_line_omits_gpu_type_when_unset():
    assert _group_config("mace", None).to_dict()["select_single"] == \
        "1:ncpus=8:mem=64gb:mpiprocs=1:ngpus=1"


def test_slurm_gpu_ntasks_is_mpi_ranks_not_cpus():
    ctx = _group_config("mace").to_dict()
    assert ctx["ntasks_per_node"] == 1
    assert ctx["cpus_per_task"] == 8
    text = _render("slurm_array.j2", _group_config("mace"))
    assert "#SBATCH --ntasks-per-node=1\n" in text
    assert "#SBATCH --cpus-per-task=8\n" in text
    assert "#SBATCH --gres=gpu:L40S:1\n" in text


def test_launch_command_uses_framework_binary_and_flags():
    assert _group_config("deepmd").to_dict()["launch_command"] == "mpirun $HOME/deepmd-kit/bin/lmp"
    assert _group_config("mace").to_dict()["launch_command"] == (
        "mpirun $HOME/lammps-mace/build/lmp -k on t ${OMP_NUM_THREADS:-8} -sf kk -pk kokkos neigh half")


def test_env_lines_base_then_overlay_then_launcher():
    lines = _group_config("deepmd").to_dict()["env_lines"]
    launcher = list(get_mlip("deepmd").launcher.env_lines)
    assert lines[:2] == ["export BASE_ENV=1", "export DP_THREADS=1"]
    assert lines[2:] == launcher


@pytest.mark.parametrize("template", TEMPLATES)
def test_env_lines_render_after_the_omp_export(template):
    rendered = _render(template, _group_config("deepmd")).splitlines()
    purge = rendered.index("module purge")
    assert rendered[purge + 1:purge + 3] == ["module load tools/prod", "module load lammps"]
    omp = rendered.index("export OMP_NUM_THREADS=1")
    assert rendered[omp + 1:omp + 3] == ["export BASE_ENV=1", "export DP_THREADS=1"]
    launcher = list(get_mlip("deepmd").launcher.env_lines)
    assert rendered[omp + 3:omp + 3 + len(launcher)] == launcher


@pytest.mark.parametrize("template", TEMPLATES)
def test_mace_gets_omp_threads_of_its_rank(template):
    rendered = _render(template, _group_config("mace"))
    assert "export OMP_NUM_THREADS=8\n" in rendered
    assert "mpirun $HOME/lammps-mace/build/lmp -k on t" in rendered
    assert "mpirun lmp " not in rendered
    proc = subprocess.run(["bash", "-n"], input=rendered, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr


def test_conda_overlay_exports_conda_prefix_for_mlip_only():
    hpc = _gpu_settings()
    hpc.frameworks["deepmd"].conda_env = "$HOME/envs/dp"
    gen = HPCScriptGenerator(HPCConfig.from_settings(hpc))
    text = _render("pbs_array.j2", gen._config_for_group((True, "deepmd"), False))
    assert 'CONDA_ENV="$HOME/envs/dp"' in text and 'export CONDA_PREFIX="$CONDA_ENV"' in text
    cpu = GlobalSettings().hpc
    cpu.conda_env = "$HOME/envs/x"
    assert "CONDA_PREFIX" not in _render("pbs_array.j2", HPCConfig.from_settings(cpu))


def test_mixed_cpu_and_deepmd_combined_split(tmp_path):
    out = tmp_path / "root" / "hpc"
    m = _manifest([("afm/cpuA/L1", False, None), ("afm/cpuB/L1", False, None),
                   ("afm/mlA/L1", True, "deepmd"), ("afm/mlB/L1", True, "deepmd")])
    scripts = HPCScriptGenerator(HPCConfig.from_settings(_gpu_settings())).generate_combined_scripts(
        m, out, scheduler="pbs")
    assert sorted(s.name for s in scripts) == ["run_cpu.pbs", "run_gpu_deepmd.pbs"]
    cpu = (out / "run_cpu.pbs").read_text()
    gpu = (out / "run_gpu_deepmd.pbs").read_text()
    assert "#PBS -l select=1:ncpus=8:mem=64gb:mpiprocs=8\n" in cpu and "ngpus" not in cpu
    assert "mpirun lmp -in " in cpu
    assert "#PBS -l select=1:ncpus=8:mem=64gb:mpiprocs=1:ngpus=1:gpu_type=L40S\n" in gpu
    assert "mpirun $HOME/deepmd-kit/bin/lmp -in " in gpu
    assert "export DP_THREADS=1" in gpu
    assert "#PBS -N friction2d_cpu\n" in cpu and "#PBS -N friction2d_gpu_deepmd\n" in gpu
    sub = (out / "submit_jobs.sh").read_text()
    assert "qsub hpc/run_cpu.pbs" in sub and "qsub hpc/run_gpu_deepmd.pbs" in sub


def test_all_cpu_manifest_keeps_plain_names_and_no_gpu_request(tmp_path):
    out = tmp_path / "root" / "hpc"
    m = _manifest([("afm/cpuA/L1", False, None), ("afm/cpuB/L1", False, None)])
    scripts = HPCScriptGenerator(HPCConfig.from_settings(GlobalSettings().hpc)
                                 ).generate_combined_scripts(m, out, scheduler="pbs")
    assert [s.name for s in scripts] == ["run.pbs"]
    text = (out / "run.pbs").read_text()
    assert "ngpus" not in text
    assert "mpirun lmp -in " in text


def test_slurm_cpu_render_has_no_gres_line():
    text = _render("slurm_array.j2", HPCConfig.from_settings(GlobalSettings().hpc))
    assert "--gres" not in text
    assert "CONDA_PREFIX" not in text


def test_mixed_deepmd_and_mace_split(tmp_path):
    out = tmp_path / "root" / "hpc"
    m = _manifest([("afm/dp/L1", True, "deepmd"), ("afm/mc/L1", True, "mace")])
    gen = HPCScriptGenerator(HPCConfig.from_settings(_gpu_settings()))
    scripts = gen.generate_combined_scripts(m, out, scheduler="slurm")
    assert sorted(s.name for s in scripts) == ["run_gpu_deepmd.sh", "run_gpu_mace.sh"]
    sh, wrapper = gen.generate_preflight(out)
    text = sh.read_text()
    assert "command -v $HOME/deepmd-kit/bin/lmp" in text
    assert "command -v $HOME/lammps-mace/build/lmp" in text
    assert 'hdr "jobs (gpu_mace)"' in text
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0
    assert "#SBATCH --gres=gpu:L40S:1" in wrapper.read_text()


def test_single_gpu_group_keeps_plain_names(tmp_path):
    out = tmp_path / "root" / "hpc"
    m = _manifest([("afm/dp1/L1", True, "deepmd"), ("afm/dp2/L1", True, "deepmd")])
    scripts = HPCScriptGenerator(HPCConfig.from_settings(_gpu_settings())).generate_combined_scripts(
        m, out, scheduler="pbs")
    assert [s.name for s in scripts] == ["run.pbs"]
    assert "ngpus=1:gpu_type=L40S" in (out / "run.pbs").read_text()


def test_array_split_by_job_entries_and_by_pot_token(tmp_path):
    out = tmp_path / "a" / "hpc"
    paths = ["afm/cpuA/L1/lammps/slide.in", "afm/mlA/L1/lammps/slide.in"]
    entries = [JobEntry(job_id="a", simulation_path="afm/cpuA/L1"),
               JobEntry(job_id="b", simulation_path="afm/mlA/L1", needs_gpu=True, framework="deepmd")]
    scripts = HPCScriptGenerator(HPCConfig.from_settings(_gpu_settings())).generate_pbs_scripts(
        paths, out, job_entries=entries)
    assert sorted(s.name for s in scripts) == ["run_cpu.pbs", "run_gpu_deepmd.pbs"]
    assert (out / "manifest_cpu_1.txt").read_text().split("\t")[-1] == paths[0]
    # No entries: inferred from the pot_ segment (sheet-on-sheet path).
    out2 = tmp_path / "b" / "hpc"
    p2 = ["sheetonsheet/MoS2/pot_sw_x/K300/lammps/slide.in",
          "sheetonsheet/MoS2/pot_deepmd-d3_dpa3/K300/lammps/slide.in"]
    scripts = HPCScriptGenerator(HPCConfig.from_settings(_gpu_settings())).generate_pbs_scripts(p2, out2)
    assert sorted(s.name for s in scripts) == ["run_cpu.pbs", "run_gpu_deepmd.pbs"]
    assert "LAMMPS_PLUGIN_PATH" in (out2 / "run_gpu_deepmd.pbs").read_text()


def test_two_phase_is_grouped(tmp_path):
    out = tmp_path / "root" / "hpc"
    m = _manifest([("afm/cpuA/L1", False, None), ("afm/mc/L1", True, "mace")])
    m.add_job(JobEntry(job_id="x", simulation_path="afm/mc/L1", lammps_script="slide_b.in",
                       needs_gpu=True, framework="mace"))
    res = HPCScriptGenerator(HPCConfig.from_settings(_gpu_settings())).generate_two_phase_scripts(
        m, out, scheduler="slurm")
    assert sorted(p.name for p in res["system"]) == ["run_system_cpu.sh", "run_system_gpu_mace.sh"]
    assert sorted(p.name for p in res["slide"]) == ["run_slide_cpu.sh", "run_slide_gpu_mace.sh"]
    sub = (out / "submit_jobs.sh").read_text()
    assert sub.count("--dependency=afterany:$SYSTEM_JOB_ID") == 2
    assert subprocess.run(["bash", "-n", str(out / "submit_jobs.sh")]).returncode == 0
    assert "#SBATCH --array=1-2" in (out / "run_slide_gpu_mace.sh").read_text()
