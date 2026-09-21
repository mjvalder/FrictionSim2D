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
