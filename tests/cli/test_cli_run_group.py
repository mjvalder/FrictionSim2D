"""`run` subcommands: the CLI's job is to turn options into run_simulations arguments.

`run_simulations` is imported by name into `src.cli` at module level
(``from .run.simulate import run_simulations``), so the patch target is the
name `src.cli` bound, not `src.run.simulate.run_simulations` -- patching the
latter would leave `src.cli`'s own reference untouched and these tests would
silently exercise the real builder.
"""
from pathlib import Path

import pytest
from click.testing import CliRunner

from src import cli as cli_module


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Replace the builder entry point; keep what the CLI passed it.

    Every current test passes ``--output-dir``, so ``output_root`` is always
    set -- but the fallback must not default to ``.`` (the repo cwd): a
    future test omitting ``--output-dir`` would otherwise mkdir into the
    real working tree. Default to a tmp_path-derived directory instead.
    """
    calls = []

    def fake_run_simulations(config_file, model, **kwargs):
        calls.append({"config_file": config_file, "model": model, **kwargs})
        root = Path(kwargs.get("output_root") or tmp_path) / "simulation_test"
        root.mkdir(parents=True, exist_ok=True)
        return ([], root, [], None)

    monkeypatch.setattr("src.cli.run_simulations", fake_run_simulations)
    return calls


@pytest.mark.parametrize("subcommand, config, model", [
    ("afm", "examples/afm_config.ini", "afm"),
    ("sheetonsheet", "examples/sheet_config.ini", "sheetonsheet"),
    ("pes-sheet", "examples/pes_sheet_config.ini", "pes_sheet"),
    ("pes-tip", "examples/pes_tip_config.ini", "pes_tip"),
    ("indent", "examples/indent_config.ini", "indent"),
])
def test_each_run_subcommand_selects_its_model(runner, captured, tmp_path,
                                                subcommand, config, model):
    result = runner.invoke(cli_module.cli,
                            ["run", subcommand, config,
                             "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0]["model"] == model
    assert captured[0]["config_file"] == config


def test_output_dir_option_reaches_the_runner(runner, captured, tmp_path):
    runner.invoke(cli_module.cli, ["run", "afm", "examples/afm_config.ini",
                                   "--output-dir", str(tmp_path)])
    assert captured[0]["output_root"] == Path(tmp_path)


def test_settings_file_option_reaches_the_runner(runner, captured, tmp_path):
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("output: {}\n", encoding="utf-8")

    runner.invoke(cli_module.cli, ["run", "afm", "examples/afm_config.ini",
                                   "--output-dir", str(tmp_path),
                                   "--settings-file", str(settings_file)])

    assert captured[0]["settings_file"] == settings_file


def test_aiida_flag_reaches_the_runner(runner, captured, tmp_path):
    result = runner.invoke(cli_module.cli, ["run", "afm", "examples/afm_config.ini",
                                             "--output-dir", str(tmp_path), "--aiida"])
    assert result.exit_code == 0, result.output
    assert captured[0]["use_aiida"] is True


def test_without_the_aiida_flag_use_aiida_is_false(runner, captured, tmp_path):
    runner.invoke(cli_module.cli, ["run", "afm", "examples/afm_config.ini",
                                   "--output-dir", str(tmp_path)])
    assert captured[0]["use_aiida"] is False


def test_hpc_scripts_flag_reaches_the_runner(runner, captured, tmp_path):
    result = runner.invoke(cli_module.cli, ["run", "afm", "examples/afm_config.ini",
                                             "--output-dir", str(tmp_path), "--hpc-scripts"])
    assert result.exit_code == 0, result.output
    assert captured[0]["generate_hpc"] is True


def test_without_hpc_scripts_flag_generate_hpc_is_false(runner, captured, tmp_path):
    runner.invoke(cli_module.cli, ["run", "afm", "examples/afm_config.ini",
                                   "--output-dir", str(tmp_path)])
    assert captured[0]["generate_hpc"] is False


def test_a_missing_config_file_is_rejected_before_any_build(runner, captured, tmp_path):
    result = runner.invoke(cli_module.cli, ["run", "afm", str(tmp_path / "nope.ini"),
                                             "--output-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert captured == []
