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
def test_cli_base_dir_is_the_submit_directory_even_with_hpc_home(
        tmp_path, base_dirs, scheduler, variable):
    root = _tree(tmp_path)

    result = _run_cli(tmp_path, root, _settings(hpc_home=HPC_HOME), scheduler)

    assert result.exit_code == 0, result.output
    assert base_dirs == [variable]


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
