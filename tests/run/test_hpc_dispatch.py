"""The CLI `hpc generate` and `generate_hpc_scripts_for_root` share one generation
helper but keep their own base-dir and job-name rules; these tests pin both."""
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from frictionsim2d.cli import cli
from frictionsim2d.core.config import GlobalSettings
from frictionsim2d.run import hpc_dispatch
from frictionsim2d.run.hpc_dispatch import generate_hpc_scripts_for_root

HPC_HOME = "/home/me/work"


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "sim_root"
    lammps = root / "afm" / "mat0" / "L1" / "lammps"
    lammps.mkdir(parents=True)
    (lammps / "slide.in").write_text("# lammps\n", encoding="utf-8")
    return root


def _settings(scheduler: str = "pbs", hpc_home=None) -> GlobalSettings:
    settings = GlobalSettings()
    settings.hpc.modules = ["lammps"]
    settings.hpc.scheduler_type = scheduler
    settings.hpc.hpc_home = hpc_home
    return settings


@pytest.fixture
def base_dirs(monkeypatch):
    """The ``base_dir`` each generation run hands to ``dispatch_hpc_generation``."""
    seen = []
    real = hpc_dispatch.dispatch_hpc_generation

    def recording(generator, simulation_root, output_dir, scheduler, base_dir,
                  manifest_entries, preflight=False):
        seen.append(base_dir)
        return real(generator, simulation_root, output_dir, scheduler, base_dir,
                    manifest_entries, preflight=preflight)

    monkeypatch.setattr(hpc_dispatch, "dispatch_hpc_generation", recording)
    return seen


def _run_cli(tmp_path: Path, root: Path, settings: GlobalSettings, scheduler: str):
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(yaml.safe_dump(settings.model_dump(mode="json")), encoding="utf-8")
    return CliRunner().invoke(cli, ["hpc", "generate", str(root), "--settings-file",
                                    str(settings_file), "--scheduler", scheduler])


@pytest.mark.parametrize("scheduler, variable", [("pbs", "$PBS_O_WORKDIR"),
                                                 ("slurm", "$SLURM_SUBMIT_DIR")])
def test_cli_base_dir_is_hpc_home_then_the_submit_directory(
        tmp_path, base_dirs, scheduler, variable):
    root = _tree(tmp_path)

    with_home = _run_cli(tmp_path, root, _settings(hpc_home=HPC_HOME), scheduler)
    without = _run_cli(tmp_path, root, _settings(), scheduler)

    assert with_home.exit_code == 0, with_home.output
    assert without.exit_code == 0, without.output
    assert base_dirs == [f"{HPC_HOME}/sim_root", variable]


def test_cli_scripts_point_at_hpc_home(tmp_path):
    root = _tree(tmp_path)

    assert _run_cli(tmp_path, root, _settings(hpc_home=HPC_HOME), "pbs").exit_code == 0

    script = (root / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert f"{HPC_HOME}/logs" in script
    assert "$PBS_O_WORKDIR/logs" not in script


@pytest.mark.parametrize("scheduler, variable", [("pbs", "$PBS_O_WORKDIR"),
                                                 ("slurm", "$SLURM_SUBMIT_DIR")])
def test_root_helper_base_dir_is_hpc_home_then_the_submit_directory(
        tmp_path, base_dirs, scheduler, variable):
    root = _tree(tmp_path)

    assert generate_hpc_scripts_for_root(root, _settings(scheduler, HPC_HOME)) is None
    assert generate_hpc_scripts_for_root(root, _settings(scheduler)) is None

    assert base_dirs == [f"{HPC_HOME}/sim_root", variable]


def test_cli_keeps_the_default_job_name_and_the_helper_names_the_job_after_the_root(tmp_path):
    cli_root = _tree(tmp_path / "a")
    helper_root = _tree(tmp_path / "b")

    assert _run_cli(tmp_path, cli_root, _settings(), "pbs").exit_code == 0
    assert generate_hpc_scripts_for_root(helper_root, _settings()) is None

    cli_script = (cli_root / "hpc" / "run.pbs").read_text(encoding="utf-8")
    helper_script = (helper_root / "hpc" / "run.pbs").read_text(encoding="utf-8")
    assert "#PBS -N friction2d\n" in cli_script
    assert "#PBS -N sim_root\n" in helper_script
    assert (cli_script.replace("friction2d", "sim_root")
            == helper_script.replace("friction2d", "sim_root"))
    assert ((cli_root / "hpc" / "manifest_1.txt").read_text(encoding="utf-8")
            == (helper_root / "hpc" / "manifest_1.txt").read_text(encoding="utf-8"))


def test_preflight_flag_writes_preflight_files_through_the_shared_helper(tmp_path):
    root = _tree(tmp_path)
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(yaml.safe_dump(_settings().model_dump(mode="json")),
                             encoding="utf-8")

    result = CliRunner().invoke(cli, ["hpc", "generate", str(root), "--settings-file",
                                      str(settings_file), "--preflight"])

    assert result.exit_code == 0, result.output
    assert (root / "hpc" / "preflight.sh").exists()
    assert (root / "hpc" / "preflight.pbs").exists()


def _lammps_dir(root: Path, *names: str) -> Path:
    lammps = root / "afm" / "m" / "L1" / "lammps"
    lammps.mkdir(parents=True)
    for name in names:
        (lammps / name).write_text("# deck\n", encoding="utf-8")
    return root


def test_a_glob_entry_in_lammps_scripts_selects_matching_trees(tmp_path):
    root = _lammps_dir(tmp_path, "slide_5ms.in")

    assert hpc_dispatch.collect_hpc_simulation_paths(root, ["slide*.in"]) == ["afm/m/L1"]
    assert hpc_dispatch.collect_hpc_simulation_paths(root, ["quench*.in"]) == []


def test_a_quench_member_is_always_a_job_and_its_deck_is_in_the_manifest(tmp_path):
    """A `run quench` member has only a quench*.in; whatever scripts the settings
    name (slide.in by default), it is still dispatched, and its deck is a row."""
    member = tmp_path / "quench" / "Si" / "ab12cd34" / "lammps"
    member.mkdir(parents=True)
    (member / "quench.in").write_text("# quench\n", encoding="utf-8")

    assert hpc_dispatch.collect_hpc_simulation_paths(
        tmp_path, ["slide.in"]) == ["quench/Si/ab12cd34"]

    assert generate_hpc_scripts_for_root(tmp_path, _settings()) is None
    manifest = (tmp_path / "hpc" / "manifest_1.txt").read_text(encoding="utf-8")
    assert "quench/Si/ab12cd34/lammps/quench.in" in manifest


def test_system_in_is_satisfied_by_any_system_deck(tmp_path):
    root = _lammps_dir(tmp_path, "system_x.in")

    assert hpc_dispatch.collect_hpc_simulation_paths(root, ["system.in"]) == ["afm/m/L1"]
    assert hpc_dispatch.collect_hpc_simulation_paths(root, ["slide.in"]) == []


def test_slide_in_is_satisfied_by_any_slide_deck(tmp_path):
    root = _lammps_dir(tmp_path, "slide_f2N.in")

    assert hpc_dispatch.collect_hpc_simulation_paths(root, ["slide.in"]) == ["afm/m/L1"]


def test_ensure_hpc_settings_fills_only_what_is_missing():
    settings = GlobalSettings().hpc
    settings.scheduler_type, settings.use_tmpdir, settings.scratch_dir = '', True, None
    hpc_dispatch.ensure_hpc_settings(settings)
    assert (settings.scheduler_type, settings.scratch_dir) == ('pbs', '$TMPDIR')
    settings.scheduler_type, settings.scratch_dir = 'slurm', '/scratch'
    hpc_dispatch.ensure_hpc_settings(settings)
    assert (settings.scheduler_type, settings.scratch_dir) == ('slurm', '/scratch')


def test_generation_report_names_the_submit_file_of_each_layout(tmp_path):
    out = tmp_path / 'hpc'
    two_phase = hpc_dispatch.generation_report({'system': [1, 2], 'slide': [3]}, out,
                                               'pbs', preflight=False)
    assert two_phase[0] == f"✅ Generated 3 two-phase script(s) in {out}"
    assert two_phase[-1] == "   3. Follow submit_jobs.sh"
    single = hpc_dispatch.generation_report([1], out, 'slurm', preflight=True)
    assert single[0] == f"✅ Generated 1 script(s) in {out}"
    assert single[-2:] == ["   3. From the simulation root: sbatch hpc/preflight.slurm",
                           "   4. Follow submit_all.txt"]


def test_plan_tree_generation_applies_the_scheduler_and_finds_the_runs(tmp_path):
    root = _tree(tmp_path)
    settings = _settings("pbs")
    settings.hpc.scheduler_type = ""
    hpc_config, paths = hpc_dispatch.plan_tree_generation(root, settings, "slurm")
    assert settings.hpc.scheduler_type == "slurm" == hpc_config.scheduler_type
    assert paths == ["afm/mat0/L1"]
    _, none = hpc_dispatch.plan_tree_generation(tmp_path / "empty", _settings(), "pbs")
    assert none == []
