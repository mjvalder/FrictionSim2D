"""AiiDA fixtures, scoped to this directory only.

`aiida_profile_clean` is a temporary broker-less SQLite profile: nothing here
touches the live `friction2d` profile and nothing needs the daemon (RabbitMQ is
down on this machine).

Not `pytest_plugins = [...]` -- in a non-root conftest that makes
`pytest tests --ignore=tests/gui` error at collection. The star-import is
guarded so this conftest does not crash when aiida-core is absent; individual
tests still skip themselves via `pytest.importorskip('aiida')`.
"""
try:
    from aiida.tools.pytest_fixtures import *  # noqa: F401,F403
except ImportError:
    pass


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _real_config_hash(monkeypatch):
    """The helper trees' records carry the real ``config_hash``: the importer
    checks it against ``config.json`` (``calcjob.run_config_file``)."""
    from tests.helpers import synthetic_results
    monkeypatch.setattr(synthetic_results, 'REAL_CONFIG_HASH', True)
