"""`hpc generate`: artifact tests over a real simulation tree.

`hpc generate` writes files rather than calling one obvious expensive
boundary, so these tests assert on the files it produces and on
`result.output` -- never on a mock (tests/README.md).
"""
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src.cli import cli
from src.core.config import GlobalSettings


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


def test_a_tree_with_no_lammps_directory_exits_non_zero(runner, tmp_path):
    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()
    (empty_root / "notes.txt").write_text("nothing here\n", encoding="utf-8")

    result = runner.invoke(cli, ["hpc", "generate", str(empty_root)])

    assert result.exit_code != 0
    assert "No simulation directories found" in result.output
    # hpc/ is created up front (before discovery); it must stay empty.
    assert list((empty_root / "hpc").iterdir()) == []


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
