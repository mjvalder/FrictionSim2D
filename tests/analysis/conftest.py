"""AiiDA fixtures for the analysis tests, loaded the way `tests/aiida/conftest.py`
loads them (a guarded star-import, never `pytest_plugins` in a non-root conftest).

`aiida_profile_clean` is a temporary broker-less SQLite profile; nothing here
touches a live profile.
"""
try:
    from aiida.tools.pytest_fixtures import *  # noqa: F401,F403
except ImportError:
    pass


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _real_config_hash(monkeypatch):
    """The helper trees' records carry the real ``config_hash`` (the importer
    checks it against ``config.json``), as in `tests/aiida/conftest.py`."""
    from tests.helpers import synthetic_results
    monkeypatch.setattr(synthetic_results, 'REAL_CONFIG_HASH', True)
