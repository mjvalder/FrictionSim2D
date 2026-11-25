"""The scenario tier's budget accounting (`conftest.TierClock`).

pytester is not enabled in this suite (`pytest_plugins` may not live in a
non-root conftest), so the accounting the session hooks drive is tested here
directly.
"""
from tests.scenarios.conftest import BUDGET_S, TierClock


def _clock(*nodeids):
    return TierClock(expected=set(nodeids))


def _run(clock, nodeid, setup=0.0, call=0.0, teardown=0.0):
    for when, seconds in (('setup', setup), ('call', call), ('teardown', teardown)):
        clock.add(nodeid, when, seconds)


def test_every_phase_of_every_scenario_item_is_summed():
    clock = _clock('a', 'b')
    _run(clock, 'a', 1.0, 2.0, 0.5)
    _run(clock, 'b', 0.5, 1.0, 0.0)
    assert clock.total == 5.0 and clock.complete


def test_a_complete_tier_over_budget_fails_with_its_time():
    clock = _clock('a')
    _run(clock, 'a', call=BUDGET_S + 1)
    assert 'over its 60 s budget' in clock.verdict()


def test_a_complete_tier_within_budget_passes():
    clock = _clock('a')
    _run(clock, 'a', call=BUDGET_S - 1)
    assert clock.verdict() is None


def test_an_interrupted_tier_is_never_judged():
    clock = _clock('a', 'b')
    _run(clock, 'a', call=BUDGET_S + 1)
    assert not clock.complete and clock.verdict() is None


def test_a_non_scenario_item_does_not_count():
    clock = _clock('a')
    _run(clock, 'other', call=100.0)
    _run(clock, 'a', call=1.0)
    assert clock.total == 1.0


def test_no_scenario_item_selected_is_not_complete():
    assert not TierClock().complete
