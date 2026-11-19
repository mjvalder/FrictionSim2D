"""HPC script generation: CPU regression goldens + GPU/MLIP job splitting.

The load-bearing property of this module is the **CPU regression**: a config
with no profile must render byte-for-byte what the pre-MLIP code rendered.
Everything MLIP adds (GPU select tokens, per-framework launch commands,
``env_lines``, array splitting) is additive and must be invisible to a campaign
that does not ask for it. The digests below were captured from the templates
*before* the MLIP changes; if one of them moves, an existing campaign's
submission scripts changed and that is a bug until proven otherwise.
"""

import hashlib
import json
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from src.core.config import HPCSettings, resolve_profile
from src.hpc.manifest import JobEntry, JobManifest
from src.hpc.scripts import (
    TEMPLATES_DIR,
    HPCConfig,
    HPCScriptGenerator,
    create_hpc_package,
)


def test_create_hpc_package_writes_hpc_directory(tmp_path: Path) -> None:
    sim_root = tmp_path / "sim_root"
    lammps_dir = sim_root / "afm" / "mat" / "L1" / "lammps"
    lammps_dir.mkdir(parents=True)
    (lammps_dir / "slide.in").write_text("# lammps\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    config = HPCConfig()
    config.hpc_settings.modules = ["lammps"]
    package_dir = create_hpc_package(sim_root, out_dir, scheduler="pbs", config=config)

    assert (package_dir / "hpc" / "run.pbs").exists()
    assert (package_dir / "hpc" / "submit_all.txt").exists()
    assert not (package_dir / "scripts").exists()

    readme = (package_dir / "README.md").read_text(encoding="utf-8")
    assert "cd hpc && bash submit_all.txt" in readme


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _cpu_config() -> HPCConfig:
    """A fully-specified, profile-free config — the regression baseline."""
    config = HPCConfig()
    config.hpc_settings.modules = ["tools/prod", "lammps"]
    config.hpc_settings.queue = "v1_small24"
    config.hpc_settings.account = "acct123"
    config.hpc_settings.partition = "compute"
    return config


def _gpu_settings() -> HPCSettings:
    """Base settings carrying a CX3-like GPU profile with two MLIP builds."""
    return HPCSettings(
        queue="v1_small24",
        account="acct123",
        modules=["tools/prod", "lammps"],
        active_profile="cx3",
        env_lines=["export BASE_ENV=1"],
        profiles={
            "cx3": {
                "queue": "gpu72",
                "gpu": {"ngpus": 1, "gpu_type": "L40S"},
                "num_cpus": 8,
                "memory_gb": 64,
                "frameworks": {
                    "deepmd": {
                        "binary": "$HOME/deepmd-kit/bin/lmp",
                        "env_lines": ["export DP_THREADS=1"],
                    },
                    "mace": {"binary": "$HOME/lammps-mace/build/lmp"},
                },
            }
        },
    )


def _render(template_name: str, config: HPCConfig, array_size: int) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR),
                      trim_blocks=True, lstrip_blocks=True)
    context = config.to_dict()
    context.update({
        "array_size": array_size,
        "manifest_file": "$PBS_O_WORKDIR/hpc/manifest_1.txt",
        "manifest_filename": "manifest_1.txt",
        "base_dir": "$PBS_O_WORKDIR",
        "log_dir": "logs",
    })
    return env.get_template(template_name).render(context)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Task 17: CPU regression — every template, byte-identical to pre-MLIP output
# ---------------------------------------------------------------------------

# Captured from the templates before launch_command/env_lines/GPU support was
# added. Keyed by (template, array_size).
CPU_GOLDEN_DIGESTS = {
    ("pbs_array.j2", 1): "aa0d99bb2f2673e870635cb300aa35dc105591ba17f8e552c6d0f19c21b63d45",
    ("pbs_array.j2", 3): "25ec9ea5a19eda750a83032411e0d8ecd595fd18aa0083f707d9722737e7b62e",
    ("pbs_combined.j2", 1): "96e90308f7ab9dc98b2339a84a204d067d89bf34cb0180182e758e6b5557929d",
    ("pbs_combined.j2", 3): "42654162406626333ce356d6103d8156f931b2941aa800ac5fa3e915154e6739",
    ("slurm_array.j2", 1): "2542d90e6e396a9abde1d08bddd076fffc2d89c0b32785133846c072dfcee5e0",
    ("slurm_array.j2", 3): "2527fc6116ae5bf943d5b9ed5f3d080fb1b3a3d9c29463a0dffc80b422bf52ef",
    ("slurm_combined.j2", 1): "d83f7ca9a371f6eb0ae6841da08952d4e40d96611140813a05fe43c4df79dca2",
    ("slurm_combined.j2", 3): "7ccbf0dfd28b149649fc3cb3ee27136e8557b58b054494c51092e0acc7ffe1cf",
}

# The full pre-MLIP text of the template that carries both the select line and
# the launch line, kept verbatim so a digest mismatch is diagnosable by diff.
PBS_ARRAY_GOLDEN = '''#!/bin/bash
#PBS -l select=1:ncpus=32:mem=62gb:mpiprocs=32
#PBS -l walltime=20:00:00
#PBS -J 1-3
#PBS -o logs/friction2d_^array_index^_out.txt
#PBS -e logs/friction2d_^array_index^_err.txt
#PBS -N friction2d
#PBS -q v1_small24
#PBS -A acct123

module purge
module load tools/prod
module load lammps

TASK_ID=${PBS_ARRAY_INDEX}

SCRIPT_PATH=$(sed -n "${TASK_ID}p" $PBS_O_WORKDIR/hpc/manifest_1.txt)
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
SIM_PATH=$(dirname "$SCRIPT_DIR")

if [ -z "$SCRIPT_PATH" ]; then
    echo "Error: No script path for index ${TASK_ID}"
    exit 1
fi

SIM_ROOT=$(basename "$PBS_O_WORKDIR")
# Per-task scratch root keeps concurrent array tasks isolated from each other.
SCRATCH=$TMPDIR/${SIM_ROOT}/task_${TASK_ID}

# Shared inputs (build/, provenance/potentials/) live at the simulation's
# config level, which may be an ancestor of SIM_PATH (e.g. AFM layer dirs share
# one provenance/ folder). Walk up to find it and copy that whole subtree so
# every relative path the LAMMPS script references resolves under SCRATCH.
COPY_ROOT="$SIM_PATH"
search="$SIM_PATH"
while [ "$search" != "." ] && [ "$search" != "/" ]; do
    if [ -d "$PBS_O_WORKDIR/$search/provenance" ] || [ -d "$PBS_O_WORKDIR/$search/build" ]; then
        COPY_ROOT="$search"
        break
    fi
    search=$(dirname "$search")
done

mkdir -p "$SCRATCH/$COPY_ROOT"
rsync -av --exclude='*.lammpstrj' --exclude='results/' --exclude='visuals/' "$PBS_O_WORKDIR/$COPY_ROOT/" "$SCRATCH/$COPY_ROOT/"
mkdir -p "$SCRATCH/${SIM_PATH}/results" "$SCRATCH/${SIM_PATH}/visuals" "$SCRATCH/${SIM_PATH}/data"
cd $SCRATCH

mpirun lmp -l none -in ${SCRIPT_PATH}
if [ $? -ne 0 ]; then
    echo "Error: LAMMPS failed on ${SCRIPT_PATH}"
    exit 1
fi

mkdir -p "$PBS_O_WORKDIR/${SIM_PATH}"
cp -r ${SIM_PATH}/results ${SIM_PATH}/visuals ${SIM_PATH}/data "$PBS_O_WORKDIR/${SIM_PATH}/"
# Persist indentation load states so a dependent slide phase (AFM) can read them.
cp ${SIM_PATH}/load_*.data "$PBS_O_WORKDIR/${SIM_PATH}/" 2>/dev/null || true

echo "Simulation completed successfully: ${SIM_PATH}"'''


@pytest.mark.parametrize("template,array_size", sorted(CPU_GOLDEN_DIGESTS))
def test_cpu_template_rendering_is_byte_identical_to_pre_mlip(template, array_size):
    rendered = _render(template, _cpu_config(), array_size)

    assert _sha(rendered) == CPU_GOLDEN_DIGESTS[(template, array_size)], (
        f"{template} (array_size={array_size}) changed for a profile-free "
        f"config. MLIP support must be additive.\n--- rendered ---\n{rendered}"
    )


def test_pbs_array_cpu_render_matches_full_golden_text():
    assert _render("pbs_array.j2", _cpu_config(), 3) == PBS_ARRAY_GOLDEN


def test_cpu_launch_command_reproduces_the_old_template_expression():
    """Old templates wrote ``{{ mpi_command }} lmp {{ lmp_flags }}``."""
    config = _cpu_config()
    context = config.to_dict()

    assert context["launch_command"] == "mpirun lmp -l none"
    assert context["env_lines"] == []
    # Spacing must survive an empty flag string too, since the templates append
    # " -in ..." straight after the variable.
    config.lmp_flags = ""
    assert config.to_dict()["launch_command"] == "mpirun lmp "


# ---------------------------------------------------------------------------
# Task 17: GPU select line, launch command, env_lines
# ---------------------------------------------------------------------------

def _gpu_config(framework: str = "deepmd", gpu_type: str = "L40S") -> HPCConfig:
    settings = _gpu_settings()
    if gpu_type is None:
        settings.profiles["cx3"].gpu.gpu_type = None
    merged = resolve_profile(settings, "cx3", framework=framework)
    config = HPCConfig.from_settings(merged)
    config.framework = framework
    return config


def test_gpu_select_line_exact_string():
    """mpiprocs comes from LauncherSpec.mpi_ranks (1), NOT from ncpus (8)."""
    context = _gpu_config("deepmd").to_dict()

    expected = "1:ncpus=8:mem=64gb:mpiprocs=1:ngpus=1:gpu_type=L40S"
    assert context["select_single"] == expected
    assert context["select_multi"] == expected


def test_gpu_select_line_omits_gpu_type_when_unset():
    context = _gpu_config("mace", gpu_type=None).to_dict()

    assert context["select_single"] == "1:ncpus=8:mem=64gb:mpiprocs=1:ngpus=1"


def test_gpu_launch_command_uses_framework_binary_and_merged_flags():
    deepmd = _gpu_config("deepmd").to_dict()
    assert deepmd["launch_command"] == "mpirun $HOME/deepmd-kit/bin/lmp -l none"

    mace = _gpu_config("mace").to_dict()
    # MACE's Kokkos flags merge onto the base -l none.
    assert mace["launch_command"] == (
        "mpirun $HOME/lammps-mace/build/lmp -l none -k on t ${OMP_NUM_THREADS:-8} -sf kk -pk kokkos neigh half"
    )


def test_env_lines_are_base_then_overlay_then_launcher_in_order():
    """Settings lines, then the profile overlay, then the launcher's block.

    The launcher block is deliberately NOT asserted to be one line: DeePMD's
    plugin lives in a different directory under conda than under pip (v4 spec
    §1), so resolving it takes a small shell block rather than a single export.
    """
    from src.core.mlip import get_mlip

    context = _gpu_config("deepmd").to_dict()

    lines = context["env_lines"]
    launcher_lines = list(get_mlip("deepmd").launcher.env_lines)
    assert lines[0] == "export BASE_ENV=1"
    assert lines[1] == "export DP_THREADS=1"
    assert lines[2:] == launcher_lines
    assert any("LAMMPS_PLUGIN_PATH" in line for line in launcher_lines)


@pytest.mark.parametrize(
    "template", ["pbs_array.j2", "pbs_combined.j2", "slurm_array.j2", "slurm_combined.j2"]
)
def test_env_lines_render_in_order_immediately_after_module_loads(template):
    from src.core.mlip import get_mlip

    rendered = _render(template, _gpu_config("deepmd"), 3).splitlines()

    purge = rendered.index("module purge")
    assert rendered[purge + 1] == "module load tools/prod"
    assert rendered[purge + 2] == "module load lammps"
    assert rendered[purge + 3] == "export BASE_ENV=1"
    assert rendered[purge + 4] == "export DP_THREADS=1"
    launcher_lines = list(get_mlip("deepmd").launcher.env_lines)
    assert rendered[purge + 5:purge + 5 + len(launcher_lines)] == launcher_lines


def test_gpu_launch_command_replaces_the_launch_line_in_every_template():
    for template in ("pbs_array.j2", "pbs_combined.j2",
                     "slurm_array.j2", "slurm_combined.j2"):
        rendered = _render(template, _gpu_config("deepmd"), 3)
        assert "mpirun $HOME/deepmd-kit/bin/lmp -l none -in " in rendered
        assert "mpirun lmp " not in rendered


def test_slurm_gpu_render_requests_a_gres():
    rendered = _render("slurm_array.j2", _gpu_config("deepmd"), 3)
    assert "#SBATCH --gres=gpu:L40S:1" in rendered


def test_slurm_cpu_render_has_no_gres_line():
    rendered = _render("slurm_array.j2", _cpu_config(), 3)
    assert "--gres" not in rendered


# ---------------------------------------------------------------------------
# Task 16: array splitting by (needs_gpu, framework)
# ---------------------------------------------------------------------------

def _combined_manifest(specs) -> JobManifest:
    """Build a manifest of system+slide pairs from (path, needs_gpu, framework)."""
    jobs = []
    for i, (path, needs_gpu, framework) in enumerate(specs):
        for script in ("system.in", "slide.in"):
            jobs.append(JobEntry(
                job_id=f"{script[0]}{i}", simulation_path=path,
                lammps_script=script, needs_gpu=needs_gpu, framework=framework,
            ))
    return JobManifest(jobs=jobs)


# Digests of the all-CPU combined-script output captured before MLIP support.
# submit_jobs.sh is deliberately absent: it embeds the absolute path of the
# simulation root, so it is not reproducible across runs and is checked
# structurally instead.
COMBINED_CPU_GOLDEN = {
    "run.pbs": "b63d8953c61f87bb343bcd1211ef49d4338e98ed3587fce7f870cb1886114007",
    "manifest_combined.txt": "a21cc2478eb9826723d750faad614be518c719333cc08bf1d964d7061e57b728",
}


def test_all_cpu_manifest_yields_todays_filenames_and_bytes(tmp_path: Path):
    manifest = _combined_manifest([
        ("afm/matA/L1", False, None),
        ("afm/matB/L1", False, None),
    ])
    out = tmp_path / "root" / "hpc"

    scripts = HPCScriptGenerator(_cpu_config()).generate_combined_scripts(
        manifest, out, scheduler="pbs", base_dir="$PBS_O_WORKDIR")

    assert [s.name for s in scripts] == ["run.pbs"]
    assert sorted(p.name for p in out.iterdir()) == [
        "manifest_combined.txt", "run.pbs", "submit_jobs.sh"
    ]
    for name, digest in COMBINED_CPU_GOLDEN.items():
        assert _sha((out / name).read_text(encoding="utf-8")) == digest, (
            f"{name} is no longer byte-identical for an all-CPU campaign"
        )
    submit = (out / "submit_jobs.sh").read_text(encoding="utf-8")
    assert submit.count("qsub hpc/") == 1
    assert "qsub hpc/run.pbs" in submit


def test_mixed_cpu_and_deepmd_manifest_splits_into_two_scripts(tmp_path: Path):
    manifest = _combined_manifest([
        ("afm/cpuA/L1", False, None),
        ("afm/cpuB/L1", False, None),
        ("afm/mlA/L1", True, "deepmd"),
        ("afm/mlB/L1", True, "deepmd"),
    ])
    out = tmp_path / "root" / "hpc"

    config = HPCConfig.from_settings(resolve_profile(_gpu_settings(), "cx3"))
    scripts = HPCScriptGenerator(config).generate_combined_scripts(
        manifest, out, scheduler="pbs", base_dir="$PBS_O_WORKDIR")

    assert sorted(s.name for s in scripts) == ["run_cpu.pbs", "run_gpu_deepmd.pbs"]

    # Per-group manifests carry exactly their own simulation dirs.
    cpu_manifest = (out / "manifest_combined_cpu.txt").read_text(encoding="utf-8")
    gpu_manifest = (out / "manifest_combined_gpu_deepmd.txt").read_text(encoding="utf-8")
    assert cpu_manifest.split() == ["afm/cpuA/L1", "afm/cpuB/L1"]
    assert gpu_manifest.split() == ["afm/mlA/L1", "afm/mlB/L1"]

    cpu_script = (out / "run_cpu.pbs").read_text(encoding="utf-8")
    gpu_script = (out / "run_gpu_deepmd.pbs").read_text(encoding="utf-8")

    # CPU group keeps the CPU select line even under a GPU profile.
    assert "#PBS -l select=1:ncpus=8:mem=64gb:mpiprocs=8\n" in cpu_script
    assert "ngpus" not in cpu_script
    assert "mpirun lmp -l none -in " in cpu_script

    assert ("#PBS -l select=1:ncpus=8:mem=64gb:mpiprocs=1:ngpus=1:gpu_type=L40S\n"
            in gpu_script)
    assert "mpirun $HOME/deepmd-kit/bin/lmp -l none -in " in gpu_script
    assert "export DP_THREADS=1" in gpu_script

    # Each group reads its own manifest and writes its own logs.
    assert "manifest_combined_cpu.txt" in cpu_script
    assert "manifest_combined_gpu_deepmd.txt" in gpu_script
    assert "#PBS -N friction2d_cpu" in cpu_script
    assert "#PBS -N friction2d_gpu_deepmd" in gpu_script

    # submit_jobs.sh must list both.
    submit = (out / "submit_jobs.sh").read_text(encoding="utf-8")
    assert "qsub hpc/run_cpu.pbs" in submit
    assert "qsub hpc/run_gpu_deepmd.pbs" in submit


def test_mixed_deepmd_and_mace_split_by_framework(tmp_path: Path):
    manifest = _combined_manifest([
        ("afm/dp/L1", True, "deepmd"),
        ("afm/mc/L1", True, "mace"),
    ])
    out = tmp_path / "root" / "hpc"

    config = HPCConfig.from_settings(resolve_profile(_gpu_settings(), "cx3"))
    scripts = HPCScriptGenerator(config).generate_combined_scripts(
        manifest, out, scheduler="pbs", base_dir="$PBS_O_WORKDIR")

    assert sorted(s.name for s in scripts) == ["run_gpu_deepmd.pbs", "run_gpu_mace.pbs"]
    mace_script = (out / "run_gpu_mace.pbs").read_text(encoding="utf-8")
    assert "mpirun $HOME/lammps-mace/build/lmp -l none -k on t ${OMP_NUM_THREADS:-8} -sf kk -pk kokkos neigh half -in " in mace_script


def test_single_gpu_group_keeps_todays_filenames(tmp_path: Path):
    """A campaign that is entirely one framework is still a single script."""
    manifest = _combined_manifest([
        ("afm/dp1/L1", True, "deepmd"),
        ("afm/dp2/L1", True, "deepmd"),
    ])
    out = tmp_path / "root" / "hpc"

    config = HPCConfig.from_settings(resolve_profile(_gpu_settings(), "cx3"))
    scripts = HPCScriptGenerator(config).generate_combined_scripts(
        manifest, out, scheduler="pbs", base_dir="$PBS_O_WORKDIR")

    assert [s.name for s in scripts] == ["run.pbs"]
    assert (out / "manifest_combined.txt").exists()
    script = (out / "run.pbs").read_text(encoding="utf-8")
    assert "ngpus=1:gpu_type=L40S" in script
    assert "mpirun $HOME/deepmd-kit/bin/lmp" in script


def test_array_scripts_split_by_group(tmp_path: Path):
    paths = ["afm/cpuA/L1/lammps/slide.in", "afm/mlA/L1/lammps/slide.in"]
    entries = [
        JobEntry(job_id="a", simulation_path="afm/cpuA/L1", lammps_script="slide.in"),
        JobEntry(job_id="b", simulation_path="afm/mlA/L1", lammps_script="slide.in",
                 needs_gpu=True, framework="deepmd"),
    ]
    out = tmp_path / "root" / "hpc"

    config = HPCConfig.from_settings(resolve_profile(_gpu_settings(), "cx3"))
    scripts = HPCScriptGenerator(config).generate_pbs_scripts(
        paths, out, job_entries=entries)

    assert sorted(s.name for s in scripts) == ["run_cpu.pbs", "run_gpu_deepmd.pbs"]
    assert (out / "manifest_cpu_1.txt").read_text().split() == [paths[0]]
    assert (out / "manifest_gpu_deepmd_1.txt").read_text().split() == [paths[1]]
    submit_all = (out / "submit_all.txt").read_text(encoding="utf-8")
    assert "qsub hpc/run_cpu.pbs" in submit_all
    assert "qsub hpc/run_gpu_deepmd.pbs" in submit_all


def test_array_scripts_without_job_entries_are_unchanged(tmp_path: Path):
    """The plain string API (create_hpc_package, sheet campaigns) is untouched."""
    out = tmp_path / "root" / "hpc"
    scripts = HPCScriptGenerator(_cpu_config()).generate_pbs_scripts(
        ["afm/matA/L1/lammps/slide.in", "afm/matB/L1/lammps/slide.in"], out)

    assert [s.name for s in scripts] == ["run.pbs"]
    assert _sha((out / "run.pbs").read_text(encoding="utf-8")) == (
        "5a0de52ea1ff1fb79d45ca8d4264e865640b87ea46fe3b75b9da09e4e249d65a"
    )
    assert _sha((out / "manifest_1.txt").read_text(encoding="utf-8")) == (
        "4c8238865a260ee79f52d72e394a53634d0e9921ef002266ba095ad0cf6ce0bb"
    )


# ---------------------------------------------------------------------------
# Task 16: JobEntry resource fields
# ---------------------------------------------------------------------------

def test_job_entry_resource_fields_default_to_cpu():
    entry = JobEntry(job_id="j", simulation_path="afm/x/L1")

    assert entry.needs_gpu is False
    assert entry.framework is None
    assert entry.launcher_flags == ""


def test_job_entry_round_trips_new_fields():
    entry = JobEntry(job_id="j", simulation_path="afm/x/L1",
                     needs_gpu=True, framework="mace",
                     launcher_flags="-k on g 1 -sf kk")

    restored = JobEntry.from_dict(json.loads(json.dumps(entry.to_dict())))

    assert restored.needs_gpu is True
    assert restored.framework == "mace"
    assert restored.launcher_flags == "-k on g 1 -sf kk"
