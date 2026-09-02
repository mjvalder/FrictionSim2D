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
