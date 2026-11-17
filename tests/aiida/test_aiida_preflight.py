"""AiiDA preflight tests (spec §8, Task 15)."""
import pytest

aiida = pytest.importorskip('aiida')


def test_preflight_never_raises_and_reports():
    from src.aiida.preflight import preflight
    rep = preflight()
    assert isinstance(rep.messages, list) and rep.messages
    assert isinstance(rep.profile_found, bool)
    # ok is a strict conjunction — never truthy with a missing profile
    if not rep.profile_found:
        assert not rep.ok


def test_preflight_distinguishes_missing_from_broken(monkeypatch):
    from src.aiida import preflight as pf

    class _EmptyConfig:
        profiles = []

    monkeypatch.setattr('aiida.manage.configuration.get_config',
                        lambda: _EmptyConfig())
    rep = pf.preflight()
    assert rep.profile_found is False
    assert rep.profile_ok is False
    assert any('no AiiDA profile initialised' in m for m in rep.messages)


def test_entry_points_check_runs():
    from src.aiida.preflight import preflight
    rep = preflight()
    assert any('entry point' in m.lower() for m in rep.messages)


class _EP:
    def __init__(self, name):
        self.name = name


def _fake_entry_points(table):
    return lambda group: [_EP(n) for n in table.get(group, ())]


def test_entry_points_need_the_calcjob_and_the_parser(monkeypatch):
    from src.aiida.preflight import _check_entry_points
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points(
        {'aiida.calculations': ['friction2d.lammps']}))
    messages = []
    assert not _check_entry_points(messages)
    assert any('aiida.parsers:friction2d.lammps NOT registered' in m for m in messages)


def test_a_friction2d_data_entry_point_fails_the_check(monkeypatch):
    from src.aiida.preflight import _check_entry_points
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points({
        'aiida.calculations': ['friction2d.lammps'], 'aiida.parsers': ['friction2d.lammps'],
        'aiida.data': ['core.dict', 'friction2d.results']}))
    messages = []
    assert not _check_entry_points(messages)
    assert any("stale aiida.data entry points ['friction2d.results']" in m for m in messages)


def test_the_current_entry_points_pass(monkeypatch):
    from src.aiida.preflight import _check_entry_points
    monkeypatch.setattr('importlib.metadata.entry_points', _fake_entry_points({
        'aiida.calculations': ['friction2d.lammps'], 'aiida.parsers': ['friction2d.lammps'],
        'aiida.data': ['core.dict']}))
    assert _check_entry_points([])
