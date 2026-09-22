"""run --aiida registers the fresh tree through ingest_root (spec §4.2).

Per R29, no test here runs real generation (atomsk/melt-quench): `run_simulations`
is patched with a fake that writes a minimal, registry-conformant tree via
`tests.helpers.synthetic_results.make_run`, exactly like `tests/cli/test_cli_run_group.py`
patches it. The one test that needs a REAL `ingest_root` against a real AiiDA
store (to prove an all-empty freshly-generated tree still registers set + run
nodes and exits 0, R8) lives in `tests/aiida/test_cli_run_aiida.py` instead,
since only `tests/aiida/conftest.py` registers the `aiida_profile_clean` fixture.
"""
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.cli import cli
from tests.helpers.synthetic_results import make_run

REL = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'


def _fake_run_simulations(config_file, model, **kwargs):
    root = Path(kwargs['output_root']) / 'simulation_test'
    run_dir = make_run(root, REL, model)
    return ([run_dir], root, [config_file], None)


@pytest.fixture
def runner():
    return CliRunner()


def test_run_aiida_calls_ingest_root_with_the_new_root(tmp_path, monkeypatch, runner):
    seen = {}

    class _Report:
        def ok(self):
            return True

        def render(self):
            return 'set X\n  runs created:    1'

    def _fake_ingest(root, label=None, profile=None, dry_run=False):
        seen['root'] = Path(root)
        seen['label'] = label
        return _Report()

    monkeypatch.setattr('src.cli.run_simulations', _fake_run_simulations)
    monkeypatch.setattr('src.aiida.ingest.ingest_root', _fake_ingest)
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)

    result = runner.invoke(cli, [
        'run', 'afm', 'examples/afm_config.ini',
        '-o', str(tmp_path), '--aiida', '--label', '260910-afm'])

    assert result.exit_code == 0, result.output
    assert seen['label'] == '260910-afm'
    assert seen['root'].parent == tmp_path
    assert seen['root'].name.startswith('simulation_')
    assert 'runs created' in result.output


def test_run_without_aiida_never_touches_the_store(tmp_path, monkeypatch, runner):
    def _boom(*args, **kwargs):
        raise AssertionError('ingest_root must not run without --aiida')

    monkeypatch.setattr('src.cli.run_simulations', _fake_run_simulations)
    monkeypatch.setattr('src.aiida.ingest.ingest_root', _boom)

    result = runner.invoke(cli, [
        'run', 'afm', 'examples/afm_config.ini', '-o', str(tmp_path)])

    assert result.exit_code == 0, result.output


def test_a_failing_ingest_aborts_the_command(tmp_path, monkeypatch, runner):
    class _Report:
        def ok(self):
            return False

        def render(self):
            return 'FAILED afm/...: ValueError: boom'

    monkeypatch.setattr('src.cli.run_simulations', _fake_run_simulations)
    monkeypatch.setattr('src.aiida.ingest.ingest_root', lambda *a, **k: _Report())
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)

    result = runner.invoke(cli, [
        'run', 'afm', 'examples/afm_config.ini', '-o', str(tmp_path), '--aiida'])

    assert result.exit_code != 0
    assert 'FAILED' in result.output


def test_register_simulations_aiida_is_gone():
    import src.cli as cli_mod
    assert not hasattr(cli_mod, '_register_simulations_aiida')


def test_an_unloadable_profile_reports_the_generated_root_not_a_traceback(
        tmp_path, monkeypatch, runner):
    """`ingest_root` can raise before it ever gets to build an `IngestReport`
    -- e.g. `load_aiida_profile` failing on a missing/misconfigured profile.
    The tree already exists on disk at that point: the user must be told
    where it landed and how to register it later, not shown a bare traceback.
    """
    def _boom(*args, **kwargs):
        raise RuntimeError('no profile')

    monkeypatch.setattr('src.cli.run_simulations', _fake_run_simulations)
    monkeypatch.setattr('src.aiida.ingest.ingest_root', _boom)
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)

    result = runner.invoke(cli, [
        'run', 'afm', 'examples/afm_config.ini',
        '-o', str(tmp_path), '--aiida', '--label', '260910-afm'])

    assert result.exit_code != 0
    generated_root = tmp_path / 'simulation_test'
    assert str(generated_root.absolute()) in result.output
    assert f'FrictionSim2D aiida import {generated_root} --label 260910-afm' \
        in result.output
    assert 'Traceback' not in result.output
