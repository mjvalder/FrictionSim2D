"""Fixtures and the wall-time budget of the scenario tier.

The budget is a session check, not a test: the setup + call + teardown
durations of every selected `scenario` item (after `-m`/`-k` deselection) are
summed, and when every one of them finished, a sum over `BUDGET_S` fails the
session (exit status 1 and a terminal summary line). An interrupted run (`-x`,
Ctrl-C) is reported but not judged, and an xdist worker never judges its share.

`aiida_profile_clean` is loaded the way `tests/aiida/conftest.py` loads it: a
guarded star-import, never `pytest_plugins` in a non-root conftest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import pytest

try:
    from aiida.tools.pytest_fixtures import *  # noqa: F401,F403
except ImportError:
    pass

#: The tier's budget, seconds of summed test durations.
BUDGET_S = 60.0


@dataclass
class TierClock:
    """Summed durations of the scenario items, and whether all of them ran."""
    expected: Set[str] = field(default_factory=set)
    durations: Dict[str, float] = field(default_factory=dict)
    ran: Set[str] = field(default_factory=set)

    def add(self, nodeid: str, when: str, duration: float) -> None:
        """One phase report of item ``nodeid`` (ignored unless it is a scenario item)."""
        if nodeid not in self.expected:
            return
        self.durations[nodeid] = self.durations.get(nodeid, 0.0) + duration
        if when == 'teardown':
            self.ran.add(nodeid)

    @property
    def total(self) -> float:
        """Seconds summed over every phase reported so far."""
        return sum(self.durations.values())

    @property
    def complete(self) -> bool:
        """Every collected scenario item reported."""
        return bool(self.expected) and self.expected <= self.ran

    def verdict(self, budget: float = BUDGET_S) -> Optional[str]:
        """The failure line when the whole tier ran over ``budget``, else None."""
        if self.complete and self.total > budget:
            return (f'scenario tier took {self.total:.1f} s, over its {budget:.0f} s budget '
                    f'({len(self.expected)} tests)')
        return None


_CLOCK = TierClock()


def _is_worker(config) -> bool:
    return hasattr(config, 'workerinput')


def pytest_sessionstart(session):  # pylint: disable=unused-argument
    """A fresh clock per session."""
    _CLOCK.expected.clear()
    _CLOCK.durations.clear()
    _CLOCK.ran.clear()


def pytest_collection_finish(session):
    """The scenario items left after deselection: what a complete tier is."""
    _CLOCK.expected.update(item.nodeid for item in session.items
                           if item.get_closest_marker('scenario'))


def pytest_runtest_logreport(report):
    """Add each phase's duration."""
    _CLOCK.add(report.nodeid, report.when, report.duration)


def pytest_sessionfinish(session, exitstatus):  # pylint: disable=unused-argument
    """Fail the session when the complete tier ran over budget."""
    if not _is_worker(session.config) and _CLOCK.verdict() and session.exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # pylint: disable=unused-argument
    """One line: the tier's time, and the verdict when it was complete."""
    if _is_worker(config) or not _CLOCK.expected:
        return
    verdict = _CLOCK.verdict()
    if verdict:
        terminalreporter.write_line(f'FAILED {verdict}', red=True)
    elif _CLOCK.complete:
        terminalreporter.write_line(
            f'scenario tier: {_CLOCK.total:.1f} s of its {BUDGET_S:.0f} s budget')
    else:
        terminalreporter.write_line(
            f'scenario tier: partial run ({len(_CLOCK.ran)}/{len(_CLOCK.expected)}), '
            f'{_CLOCK.total:.1f} s, budget not judged')


@pytest.fixture
def scenario_store(aiida_profile_clean, monkeypatch, tmp_path):
    """A clean profile no command can leave, and no per-user settings."""
    import frictionsim2d.aiida as aiida_pkg  # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg'))
    return aiida_profile_clean
