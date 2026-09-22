"""`db init --drop` deletes every central row, so it must ask first (I4)."""
from click.testing import CliRunner

from src.cli import cli


class _DB:
    def __init__(self):
        self.calls = []

    def init_schema(self, drop=False):
        self.calls.append(drop)


def _patch(monkeypatch):
    db = _DB()
    monkeypatch.setattr('src.data.database.central_db', lambda *a, **k: db)
    return db


def test_drop_asks_and_a_no_aborts(monkeypatch):
    db = _patch(monkeypatch)

    result = CliRunner().invoke(cli, ['db', 'init', '--drop'], input='n\n')

    assert result.exit_code != 0
    assert 'DROP the simulations table' in result.output
    assert db.calls == []


def test_drop_proceeds_on_yes_answer(monkeypatch):
    db = _patch(monkeypatch)

    result = CliRunner().invoke(cli, ['db', 'init', '--drop'], input='y\n')

    assert result.exit_code == 0, result.output
    assert db.calls == [True]


def test_yes_flag_skips_the_prompt(monkeypatch):
    db = _patch(monkeypatch)

    result = CliRunner().invoke(cli, ['db', 'init', '--drop', '--yes'])

    assert result.exit_code == 0, result.output
    assert 'Continue?' not in result.output
    assert db.calls == [True]


def test_plain_init_never_prompts(monkeypatch):
    db = _patch(monkeypatch)

    result = CliRunner().invoke(cli, ['db', 'init'])

    assert result.exit_code == 0, result.output
    assert 'Continue?' not in result.output
    assert db.calls == [False]


def test_a_central_db_failure_is_surfaced_not_retried_on_localhost(monkeypatch):
    """m7: `_make_db` swallowed `central_db()`'s error and silently retried
    with FrictionDB's localhost defaults, hiding the real cause."""
    built = []

    def _bad():
        raise RuntimeError('password authentication failed for user "m"')

    monkeypatch.setattr('src.data.database.central_db', _bad)
    monkeypatch.setattr('src.data.database.FrictionDB',
                        lambda *a, **k: built.append(k))

    result = CliRunner().invoke(cli, ['db', 'init'])

    assert result.exit_code != 0
    assert 'password authentication failed' in result.output
    assert built == []
