"""The CLI driven into a real build, with nothing patched.

Spec Phase 3, item 5. ``tests/cli/test_cli_run_group.py`` replaces
``src.cli.run_simulations`` with a stub -- deliberately, since its subject is
option-to-argument translation -- so **no test drives the CLI into a real
builder**. Everything between Click's argument parsing and a rendered script is
untested end to end: an option that reaches ``run_simulations`` correctly and is
then dropped, or a chain step that works alone but not after the previous one,
passes the whole existing suite.

``indent_config.ini`` is the vehicle (1 s to build, and it produces both a
``system.in`` and per-load slide scripts, so the ``hpc generate`` step has a
two-phase tree to walk).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src import cli as cli_module
from tests.helpers.examples import EXAMPLES, REPO_ROOT

pytestmark = pytest.mark.integration

CONFIG = 'indent_config.ini'


@pytest.fixture(name='runner')
def _runner():
    return CliRunner()


@pytest.fixture(name='hpc_settings_file')
def _hpc_settings_file(tmp_path) -> Path:
    """A settings.yaml with ``hpc.modules`` set.

    Without it `hpc generate` correctly refuses -- modules is the one setting
    that gates generation, and it has no default. Passing it as
    ``--settings-file`` exercises that option at the same time.
    """
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump({'hpc': {'modules': ['lammps']}}),
                    encoding='utf-8')
    return path


@pytest.fixture(autouse=True)
def _at_repo_root(monkeypatch):
    """The example spells its paths relative to the repo root, and so does a
    real user typing the command from a checkout."""
    monkeypatch.chdir(REPO_ROOT)


def _simulation_root(output_dir: Path) -> Path:
    roots = [p for p in output_dir.iterdir() if p.is_dir()]
    assert len(roots) == 1, roots
    return roots[0]


def _invoke(runner, *args):
    result = runner.invoke(cli_module.cli, list(args))
    assert result.exit_code == 0, (
        f"{args}\nexit {result.exit_code}\n{result.output}\n{result.exception!r}")
    return result


def test_run_builds_a_real_tree_through_the_cli(runner, tmp_path):
    """No patching: Click -> run_simulations -> builder -> rendered script."""
    out = tmp_path / 'out'
    _invoke(runner, 'run', 'indent', str(EXAMPLES / CONFIG), '-o', str(out))

    root = _simulation_root(out)
    scripts = sorted(root.rglob('lammps/*.in'))
    assert scripts, sorted(p.name for p in root.rglob('*'))
    assert any('system.in' == s.name for s in scripts)
    assert all(s.stat().st_size > 0 for s in scripts)


def test_hpc_generate_runs_on_a_tree_the_run_command_just_wrote(
        runner, tmp_path, hpc_settings_file):
    """The chain the docs tell users to type. `hpc generate` walks a tree it did
    not create, so a layout change in the builder breaks it silently -- the
    command reports success and writes an empty manifest."""
    out = tmp_path / 'out'
    _invoke(runner, 'run', 'indent', str(EXAMPLES / CONFIG), '-o', str(out))
    root = _simulation_root(out)

    result = _invoke(runner, 'hpc', 'generate', str(root),
                     '--settings-file', str(hpc_settings_file))
    assert 'Found' in result.output, result.output

    manifests = sorted((root / 'hpc').glob('manifest*.txt'))
    assert manifests, sorted(p.name for p in (root / 'hpc').iterdir())
    for manifest in manifests:
        entries = [line for line in
                   manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        assert entries, manifest.name
        for entry in entries:
            assert (root / entry).exists(), (manifest.name, entry)


def test_the_scheduler_option_reaches_the_generated_script(
        runner, tmp_path, hpc_settings_file):
    """`--scheduler slurm` is applied after `load_settings`, overriding the
    settings default (`src/cli.py`, hpc_generate). Only the written script shows
    whether that assignment survived."""
    out = tmp_path / 'out'
    _invoke(runner, 'run', 'indent', str(EXAMPLES / CONFIG), '-o', str(out))
    root = _simulation_root(out)

    _invoke(runner, 'hpc', 'generate', str(root), '--scheduler', 'slurm',
            '--settings-file', str(hpc_settings_file))
    written = {p.name for p in (root / 'hpc').iterdir()}
    assert any(name.endswith('.sh') and name.startswith('run') for name in written), written
    assert not any(name.endswith('.pbs') for name in written), written
    text = '\n'.join(p.read_text(encoding='utf-8')
                     for p in (root / 'hpc').glob('run*.sh'))
    assert '#SBATCH' in text
    # the scheduler also picks the submit-dir variable the script resolves every
    # baked path against; getting it wrong makes every task look for its inputs
    # in the wrong place
    assert '$SLURM_SUBMIT_DIR' in text
    assert '$PBS_O_WORKDIR' not in text


def test_run_with_hpc_scripts_does_both_in_one_command(
        runner, tmp_path, hpc_settings_file):
    """`--hpc-scripts` takes the in-process path (`generate_hpc_scripts_for_root`)
    rather than the `hpc generate` command's own walk, so it is a second code
    path to the same artifacts and is checked separately."""
    out = tmp_path / 'out'
    _invoke(runner, 'run', 'indent', str(EXAMPLES / CONFIG), '-o', str(out),
            '--hpc-scripts', '--settings-file', str(hpc_settings_file))
    root = _simulation_root(out)
    assert sorted((root / 'hpc').glob('manifest*.txt')), sorted(
        p.name for p in root.iterdir())


def test_a_rejected_config_is_reported_not_raised(runner, tmp_path):
    """It used to exit on a bare traceback with no CLI output at all, while
    `hpc generate` reported its own refusals properly. The message must name the
    key so the user knows what to change."""
    bad = tmp_path / 'bad.ini'
    text = (EXAMPLES / CONFIG).read_text(encoding='utf-8')
    bad.write_text(
        re.sub(r'^\[general\]$', '[general]\ndriving_spring = [10, 20]',
               text, count=1, flags=re.M), encoding='utf-8')

    result = runner.invoke(cli_module.cli,
                           ['run', 'indent', str(bad), '-o', str(tmp_path / 'out')])
    assert result.exit_code != 0
    assert 'driving_spring' in result.output, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"a rejected config must not surface as {result.exception!r}")


def test_hpc_generate_refuses_a_directory_with_no_simulations(runner, tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    result = runner.invoke(cli_module.cli, ['hpc', 'generate', str(empty)])
    assert result.exit_code != 0
    assert 'No simulation' in result.output, result.output
