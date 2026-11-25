"""Fixtures and the wall-time clock of the scenario tier.

`aiida_profile_clean` is loaded the way `tests/aiida/conftest.py` loads it: a
guarded star-import, never `pytest_plugins` in a non-root conftest.
"""
from __future__ import annotations

import time
from typing import Optional

import pytest

try:
    from aiida.tools.pytest_fixtures import *  # noqa: F401,F403
except ImportError:
    pass

_START = pytest.StashKey[float]()


def pytest_runtest_setup(item):
    """Start the tier clock at the first scenario test's setup."""
    if item.get_closest_marker('scenario') and _START not in item.config.stash:
        item.config.stash[_START] = time.monotonic()


def tier_elapsed(config) -> Optional[float]:
    """Seconds since the first scenario test started, or None if none has."""
    start = config.stash.get(_START, None)
    return None if start is None else time.monotonic() - start


@pytest.fixture
def scenario_store(aiida_profile_clean, monkeypatch, tmp_path):
    """A clean profile no command can leave, and no per-user settings."""
    import frictionsim2d.aiida as aiida_pkg  # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg'))
    return aiida_profile_clean
