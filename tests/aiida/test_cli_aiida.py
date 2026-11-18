"""The `aiida` commands against a real store (plan Task 11).

Here rather than `tests/cli/`: `aiida_profile_clean` is registered only by
`tests/aiida/conftest.py`. `load_aiida_profile` is pinned to a no-op so no
command can switch to the profile a user's settings name.
"""
import pytest

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from src.aiida.ingest import import_root  # noqa: E402
from src.aiida.store import BLOCK_HASH_EXTRA  # noqa: E402
from src.cli import cli  # noqa: E402
from tests.aiida.test_steps_integrity import _pid, _tree, _write_ruling  # noqa: E402


@pytest.fixture
def store(aiida_profile_clean, monkeypatch, tmp_path):
    """A clean profile no command can leave, and no per-user settings."""
    import src.aiida as aiida_pkg
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg'))
    return aiida_profile_clean


def _run(*args):
    return CliRunner().invoke(cli, ['aiida', *args])


def _imported(root, label='camp', **kwargs):
    _tree(root, **kwargs)
    report = import_root(root, label=label)
    assert report.ok(), report.render()
    return report


# --- verify / reindex -----------------------------------------------------------

def test_verify_on_a_clean_store_exits_0(store, tmp_path):
    _imported(tmp_path / 'sim')
    result = _run('verify')
    assert result.exit_code == 0, result.output
    assert '0 duplicate hash(es), 0 mismatch(es)' in result.output


def test_verify_exits_1_and_names_a_block_whose_hash_extra_was_edited(store, tmp_path):
    _imported(tmp_path / 'sim')
    block = orm.QueryBuilder().append(
        orm.Dict, filters={'extras': {'has_key': BLOCK_HASH_EXTRA}}).first(flat=True)
    block.base.extras.set(BLOCK_HASH_EXTRA, '0' * 64)

    result = _run('verify')

    assert result.exit_code == 1
    assert block.uuid in result.output

    fixed = _run('reindex')
    assert fixed.exit_code == 0, fixed.output
    assert 'reindexed 1 node(s)' in fixed.output
    assert _run('verify').exit_code == 0


# --- rule -----------------------------------------------------------------------

def test_rule_without_rulings_applies_and_skips_nothing(store, tmp_path):
    _imported(tmp_path / 'sim')
    result = _run('rule', 'camp')
    assert result.exit_code == 0, result.output
    assert 'applied 0, replaced 0, skipped 0' in result.output


def test_rule_prints_the_overrides_and_the_withdrawn(store, tmp_path):
    root = tmp_path / 'sim'
    _imported(root)
    pid = _pid(root, 10.0)
    rulings = tmp_path / 'review' / 'integrity_rulings.yaml'
    rulings.parent.mkdir()
    _write_ruling(rulings.parent, pid, 'fail', override=True)

    applied = _run('rule', 'camp', '--rulings', str(rulings))
    assert applied.exit_code == 0, applied.output
    assert 'applied 1, replaced 0, skipped 0' in applied.output
    assert f'override {pid}: complete -> ' in applied.output
    assert 'skipped 1' in _run('rule', 'camp', '--rulings', str(rulings)).output

    rulings.write_text('', encoding='utf-8')     # the file withdraws its ruling
    gone = _run('rule', 'camp', '--rulings', str(rulings))
    assert gone.exit_code == 0, gone.output
    assert 'withdrawn 1' in gone.output and f'withdrawn: {pid}' in gone.output
    assert f'ruling retracted by its file: {pid}' in gone.output
    assert 'override ' not in gone.output
    again = _run('rule', 'camp', '--rulings', str(rulings), '--withdraw')
    assert again.exit_code == 0 and 'withdrawn 0' in again.output   # a no-op flag


def test_rule_on_an_unknown_set_fails_cleanly(store):
    result = _run('rule', 'nope')
    assert result.exit_code != 0
    assert 'nope' in result.output


# --- postprocess / dataset / query ---------------------------------------------

def test_postprocess_then_dataset(store, tmp_path):
    _imported(tmp_path / 'sim', quarantine=(30.0,))

    made = _run('postprocess', 'camp')
    assert made.exit_code == 0, made.output
    assert '2 new summar' in made.output
    assert '0 new summar' in _run('postprocess', 'camp').output

    built = _run('dataset', 'd1', 'camp')
    assert built.exit_code == 0, built.output
    assert "dataset 'd1'" in built.output and '2 row(s)' in built.output
    again = _run('dataset', 'd1', 'camp')
    assert again.exit_code != 0 and 'exists' in again.output


def test_postprocess_and_dataset_read_the_given_settings_file(store, tmp_path):
    _imported(tmp_path / 'sim')
    settings = tmp_path / 'settings.yaml'
    settings.write_text('aiida:\n  skip_fraction: 0.3\n', encoding='utf-8')

    assert _run('postprocess', 'camp', '--settings-file', str(settings)).exit_code == 0
    built = _run('dataset', 'd', 'camp', '--settings-file', str(settings))
    assert built.exit_code == 0, built.output
    table = orm.QueryBuilder().append(
        orm.Dict, filters={'extras.friction2d_dataset': 'd'}).one()[0]
    assert table['policy']['summary_options'] == {'skip_fraction': 0.3}
    assert len(table['rows']) == 3


def test_query_prints_the_verdict_columns(store, tmp_path):
    _imported(tmp_path / 'sim', quarantine=(30.0,))
    result = _run('query', '--set-label', 'camp', '--format', 'csv')
    assert result.exit_code == 0, result.output
    header = next(line for line in result.output.splitlines() if 'status' in line)
    columns = header.split(',')
    for column in ('status', 'reasons', 'verdict_kind', 'provisional',
                   'thresholds_version'):
        assert column in columns, column
    assert columns.index('status') < columns.index('point_uuid')
    assert 'quarantined' in result.output and 'complete' in result.output


# --- import ---------------------------------------------------------------------

def _computers():
    return {c.computer.label for c in orm.QueryBuilder().append(orm.CalcJobNode).all(flat=True)}


def test_import_reads_the_machine_from_the_settings_file(store, tmp_path):
    root = tmp_path / 'sim'
    _tree(root)
    settings = tmp_path / 'settings.yaml'
    settings.write_text('aiida:\n  machine: from-file\n', encoding='utf-8')

    result = _run('import', str(root), '--label', 'camp', '--settings-file', str(settings))

    assert result.exit_code == 0, result.output
    assert 'no LAMMPS version:' in result.output and 'already imported:' in result.output
    assert _computers() == {'from-file'}


def test_import_machine_beats_the_settings_file(store, tmp_path):
    root = tmp_path / 'sim'
    _tree(root)
    settings = tmp_path / 'settings.yaml'
    settings.write_text('aiida:\n  machine: from-file\n', encoding='utf-8')

    result = _run('import', str(root), '--machine', 'archer2',
                  '--settings-file', str(settings))

    assert result.exit_code == 0, result.output
    assert _computers() == {'archer2'}


# --- recheck --------------------------------------------------------------------

def test_recheck_reports_flipped_points_and_is_idempotent(store, tmp_path):
    root = tmp_path / 'sim'
    _imported(root, eject=(30.0,))
    settings = tmp_path / 'settings.yaml'
    settings.write_text('integrity:\n  ejected_atoms: 3\n  isolated_atoms: 3\n'
                        "  version: '1.1.0'\n", encoding='utf-8')

    result = _run('recheck', 'camp', '--settings-file', str(settings))

    assert result.exit_code == 0, result.output
    assert 'thresholds v1.1.0' in result.output
    assert f're-checked {_pid(root, 30.0)}: failed_integrity -> complete' in result.output
    again = _run('recheck', 'camp', '--settings-file', str(settings))
    assert again.exit_code == 0, again.output
    assert "set 'camp': 0 point(s) checked" in again.output
    assert 're-checked ' not in again.output


def test_rule_withdraw_pending_drops_a_lost_files_ruling(store, tmp_path):
    root = tmp_path / 'sim'
    _imported(root, eject=(30.0,))
    pid = _pid(root, 30.0)
    review = tmp_path / 'x.yaml'
    review.write_text(f'- point: "{pid}"\n  ruling: pass\n  note: fine\n'
                      f'  override: true\n', encoding='utf-8')
    assert _run('rule', 'camp', '--rulings', str(review)).exit_code == 0
    review.unlink()
    settings = tmp_path / 'settings.yaml'
    settings.write_text("integrity:\n  version: '1.1.0'\n", encoding='utf-8')

    recheck = _run('recheck', 'camp', '--settings-file', str(settings))
    assert f'ruling pending: {review.resolve()} missing: {pid}' in recheck.output

    result = _run('rule', 'camp', '--withdraw-pending', str(review))
    assert result.exit_code == 0, result.output
    assert f'pending ruling withdrawn: {pid} ({review.resolve()})' in result.output
    assert 'ruling pending:' not in _run('rule', 'camp').output
