"""`aiida setup`: SQLite by default, PostgreSQL only when asked."""
from types import SimpleNamespace

import pytest

pytest.importorskip('aiida')

from click.testing import CliRunner  # noqa: E402

import src.aiida.setup as setup_mod  # noqa: E402
from src.cli import cli  # noqa: E402


@pytest.fixture
def presto(monkeypatch):
    """Record every `verdi presto` command; touch no real AiiDA config.
    ``presto.postgres_fails`` makes a `--use-postgres` call fail."""
    import aiida.manage as manage
    import aiida.manage.configuration as configuration

    state = SimpleNamespace(calls=[], postgres_fails=False)
    monkeypatch.setattr(configuration, 'get_config', lambda: SimpleNamespace(profiles=[]))
    monkeypatch.setattr(configuration, 'load_profile', lambda name: None)
    monkeypatch.setattr(configuration, 'reset_config', lambda: None)
    monkeypatch.setattr(manage, 'get_manager',
                        lambda: SimpleNamespace(reset_profile=lambda: None))

    def run(cmd, **_kwargs):
        state.calls.append(cmd)
        failed = '--use-postgres' in cmd and state.postgres_fails
        return SimpleNamespace(returncode=1 if failed else 0, stdout='',
                               stderr='no server' if failed else '')
    monkeypatch.setattr(setup_mod.subprocess, 'run', run)
    return state


SQLITE = ['verdi', 'presto', '--profile-name', 'p1']


def test_setup_makes_an_sqlite_profile_by_default(presto):
    assert setup_mod.setup_profile('p1') is True
    assert presto.calls == [SQLITE]


def test_postgres_is_opt_in(presto):
    assert setup_mod.setup_profile('p1', use_postgres=True) is True
    assert presto.calls == [SQLITE + ['--use-postgres']]


def test_a_failed_postgres_profile_falls_back_to_sqlite(presto):
    presto.postgres_fails = True
    assert setup_mod.setup_profile('p1', use_postgres=True) is True
    assert presto.calls == [SQLITE + ['--use-postgres'], SQLITE]


def test_cli_setup_passes_postgres_only_with_the_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(setup_mod, 'setup_profile',
                        lambda name, use_postgres: seen.append((name, use_postgres)) or True)
    runner = CliRunner()
    assert runner.invoke(cli, ['aiida', 'setup']).exit_code == 0
    assert runner.invoke(cli, ['aiida', 'setup', '-p', 'x', '--postgres']).exit_code == 0
    assert seen == [('friction2d', False), ('x', True)]
