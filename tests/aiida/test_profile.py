"""The AiiDA profile (`frictionsim2d/aiida/profile.py`): `aiida setup` makes SQLite by
default and PostgreSQL only when asked; `aiida preflight` (spec §8, Task 15)
reports what it found and never raises."""
from types import SimpleNamespace

import pytest

pytest.importorskip('aiida')

from click.testing import CliRunner  # noqa: E402

import frictionsim2d.aiida.profile as profile_mod  # noqa: E402
from frictionsim2d.cli import cli  # noqa: E402


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
    monkeypatch.setattr(profile_mod.subprocess, 'run', run)
    return state


def test_setup_makes_an_sqlite_profile_by_default(presto):
    assert profile_mod.setup_profile('p1') is True
    assert len(presto.calls) == 1
    assert 'p1' in presto.calls[0] and '--use-postgres' not in presto.calls[0]


def test_postgres_is_opt_in(presto):
    assert profile_mod.setup_profile('p1', use_postgres=True) is True
    assert len(presto.calls) == 1
    assert 'p1' in presto.calls[0] and '--use-postgres' in presto.calls[0]


def test_a_failed_postgres_profile_falls_back_to_sqlite(presto):
    presto.postgres_fails = True
    assert profile_mod.setup_profile('p1', use_postgres=True) is True
    assert len(presto.calls) == 2
    assert '--use-postgres' in presto.calls[0]
    assert '--use-postgres' not in presto.calls[1] and 'p1' in presto.calls[1]


def test_cli_setup_passes_postgres_only_with_the_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(profile_mod, 'setup_profile',
                        lambda name, use_postgres: seen.append((name, use_postgres)) or True)
    runner = CliRunner()
    assert runner.invoke(cli, ['aiida', 'setup']).exit_code == 0
    assert runner.invoke(cli, ['aiida', 'setup', '-p', 'x', '--postgres']).exit_code == 0
    assert seen == [('friction2d', False), ('x', True)]


# --- preflight ------------------------------------------------------------------

def test_preflight_distinguishes_missing_from_broken(monkeypatch):
    from frictionsim2d.aiida import profile as pf

    class _EmptyConfig:
        profiles = []

    monkeypatch.setattr('aiida.manage.configuration.get_config',
                        lambda: _EmptyConfig())
    rep = pf.preflight()
    assert rep.profile_found is False
    assert rep.profile_ok is False
    assert any('no AiiDA profile initialised' in m for m in rep.messages)


class _EP:
    def __init__(self, name):
        self.name = name


def _fake_entry_points(table):
    return lambda group: [_EP(n) for n in table.get(group, ())]


def test_entry_points_need_the_calcjob_and_the_parser(monkeypatch):
    from frictionsim2d.aiida.profile import _check_entry_points
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points(
        {'aiida.calculations': ['friction2d.lammps']}))
    messages = []
    assert not _check_entry_points(messages)
    assert any('aiida.parsers:friction2d.lammps NOT registered' in m for m in messages)


def test_a_friction2d_data_entry_point_fails_the_check(monkeypatch):
    from frictionsim2d.aiida.profile import _check_entry_points
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points({
        'aiida.calculations': ['friction2d.lammps'], 'aiida.parsers': ['friction2d.lammps'],
        'aiida.data': ['core.dict', 'friction2d.results']}))
    messages = []
    assert not _check_entry_points(messages)
    assert any("stale aiida.data entry points ['friction2d.results']" in m for m in messages)


def test_the_current_entry_points_pass(monkeypatch):
    from frictionsim2d.aiida.profile import _check_entry_points
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points({
        'aiida.calculations': ['friction2d.lammps'], 'aiida.parsers': ['friction2d.lammps'],
        'aiida.data': ['core.dict']}))
    assert _check_entry_points([])


def test_full_setup_sets_up_the_profile_only(monkeypatch):
    seen = []
    monkeypatch.setattr(profile_mod, 'setup_profile',
                        lambda name, use_postgres: seen.append(name) or True)
    assert profile_mod.full_setup('p1') == {'profile': True}
    assert seen == ['p1']


def test_aiida_setup_has_no_remote_options():
    out = CliRunner().invoke(cli, ['aiida', 'setup', '--help']).output

    assert '--use-remote' not in out
    assert '--hpc-config' not in out
    assert '--lammps-path' not in out
    # `aiida setup` CREATES a profile; every other command LOADS one.
    assert '--create-profile' in out
    assert '--profile ' not in out


def test_aiida_setting_reads_falls_back_and_defaults(monkeypatch):
    from types import SimpleNamespace

    import frictionsim2d.core.config as config
    from frictionsim2d import aiida as aiida_pkg
    settings = SimpleNamespace(aiida=SimpleNamespace(machine='m1', profile=None))
    monkeypatch.setattr(config, 'load_settings', lambda settings_file=None: settings)
    assert aiida_pkg.aiida_setting('machine', 'x') == 'm1'
    assert aiida_pkg.aiida_setting('profile', 'dflt') == 'dflt'          # None -> default
    assert aiida_pkg.aiida_setting('missing', 7) == 7                    # no such field

    def unreadable(settings_file=None):
        raise OSError('gone')
    monkeypatch.setattr(config, 'load_settings', unreadable)
    assert aiida_pkg.aiida_setting('machine', 'x') == 'x'
