"""db upload reads the local store, never a results directory (spec §5).

`_make_db` is patched at its own boundary (`src.data.database.central_db`,
an allowed prefix in tests/repo/test_mock_targets.py) rather than at
`src.cli._make_db`, which is not an allowed mock target.
"""
import pytest

pytest.importorskip('aiida')

from click.testing import CliRunner  # noqa: E402

from src.cli import cli  # noqa: E402


class _Record:
    def __init__(self, uuid, uploader=None, contact=None):
        self.run_uuid = uuid
        self.uploader = uploader
        self.contact = contact

    def to_upload_dict(self):
        return {'material': 'h-MoS2', 'run_uuid': self.run_uuid,
                'uploader': self.uploader, 'contact': self.contact}


class _Run:
    """A stand-in run node: real AiiDA run nodes always carry `.uuid`, and
    `_upload_query_results` assumes that invariant rather than tolerating a
    uuid-less run (Task 32 review: the earlier `hasattr`/`getattr` fallback
    added no real protection -- `to_record` would still crash on a `None`
    run_node, it only hid the crash from this test).
    """
    def __init__(self, uuid='run-0'):
        self.uuid = uuid


class _Result:
    def __init__(self, uuid, run_uuid='run-0'):
        self._uuid = uuid
        self.run_uuid = run_uuid

    def to_record(self, run_node, set_node, uploader=None, contact=None):
        return _Record(self._uuid, uploader=uploader, contact=contact)


class _Query:
    def __init__(self, n):
        self.runs = [_Run()]
        self.results = [_Result(f'r{i}') for i in range(n)]


def _patch_store(monkeypatch, n=2):
    loaded = []
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr('src.aiida.load_aiida_profile',
                        lambda profile=None: loaded.append(profile))
    monkeypatch.setattr('src.aiida.query.Friction2DDB',
                        lambda: type('DB', (), {
                            'query': lambda self, **kw: _Query(n)})())
    monkeypatch.setattr('src.aiida.export.resolve_set', lambda label: object())
    return loaded


def _patch_db(monkeypatch, calls_attr='upload_record'):
    """Patch `_make_db`'s own boundary so a fake DB is returned, never a
    real connection. `calls` records whatever `upload_record` receives.
    """
    calls = []

    class _DB:
        def upload_record(self, rec, uploader=None):
            calls.append(rec)
            return len(calls)

        def validate_staged(self, row_id):  # noqa: ARG002
            return type('V', (), {'is_valid': True})()

        def publish(self, row_id):  # noqa: ARG002
            return True

    monkeypatch.setattr('src.data.database.central_db', lambda *a, **k: _DB())
    return calls


def test_upload_uses_the_set_label(monkeypatch):
    loaded = _patch_store(monkeypatch)
    uploaded = _patch_db(monkeypatch)

    result = CliRunner().invoke(cli, [
        'db', 'upload', '260910-afm', '--uploader', 'matteo',
        '--contact', 'matteo@example.org', '--aiida-profile', 'friction2d'])

    assert result.exit_code == 0, result.output
    assert loaded == ['friction2d']
    assert [r.run_uuid for r in uploaded] == ['r0', 'r1']
    # U4: the pointer is worthless without an owner. Asserting only the uuids
    # let a build that dropped `contact` entirely pass.
    assert {r.contact for r in uploaded} == {'matteo@example.org'}
    assert {r.uploader for r in uploaded} == {'matteo'}
    assert 'published=2' in result.output


def test_stage_also_carries_the_contact(monkeypatch):
    loaded = _patch_store(monkeypatch, n=1)
    staged = _patch_db(monkeypatch)

    result = CliRunner().invoke(cli, [
        'db', 'stage', '260910-afm', '--uploader', 'matteo',
        '--contact', 'matteo@example.org'])

    assert result.exit_code == 0, result.output
    assert loaded == [None]
    assert staged[0].contact == 'matteo@example.org'


def test_dry_run_uploads_nothing(monkeypatch):
    _patch_store(monkeypatch)
    monkeypatch.setattr(
        'src.data.database.central_db',
        lambda *a, **k: pytest.fail('must not connect'))

    result = CliRunner().invoke(cli, ['db', 'upload', '260910-afm', '--dry-run'])

    assert result.exit_code == 0, result.output
    assert '2 row(s)' in result.output


def test_results_dir_argument_is_gone():
    out = CliRunner().invoke(cli, ['db', 'upload', '--help']).output

    assert 'RESULTS_DIR' not in out
    assert 'SET_LABEL' in out
    assert '--contact' in out
    assert '--aiida-profile' in out


def test_reader_row_iterator_is_gone():
    import src.cli as cli_mod

    assert not hasattr(cli_mod, '_iter_reader_rows')


def test_duplicate_upload_is_skipped_not_a_crash(monkeypatch):
    """A duplicate `time_series_hash` must be a counted, clean outcome."""
    _patch_store(monkeypatch, n=1)

    class _DB:
        def upload_record(self, rec, uploader=None):  # noqa: ARG002
            return None  # FrictionDB.upload_record: duplicate hash, no row inserted

        def validate_staged(self, row_id):
            raise AssertionError('must not validate a row that was never inserted')

        def publish(self, row_id):
            raise AssertionError('must not publish a row that was never inserted')

    monkeypatch.setattr('src.data.database.central_db', lambda *a, **k: _DB())

    result = CliRunner().invoke(cli, ['db', 'upload', '260910-afm'])

    assert result.exit_code == 0, result.output
    assert 'skipped=1' in result.output
    assert 'published=0' in result.output
    assert 'rejected=0' in result.output


def test_result_with_missing_run_is_rejected_not_a_crash(monkeypatch):
    """U4: a central row is worthless without its run pointer. A result
    naming a run the query didn't return must be rejected cleanly -- never
    uploaded, and never a crash that stops the rest of the batch.
    """
    loaded = []
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr('src.aiida.load_aiida_profile',
                        lambda profile=None: loaded.append(profile))

    class _QueryWithAnOrphan:
        def __init__(self):
            self.runs = [_Run('run-0')]
            self.results = [
                _Result('r-orphan', run_uuid='run-does-not-exist'),
                _Result('r-ok', run_uuid='run-0'),
            ]

    monkeypatch.setattr('src.aiida.query.Friction2DDB',
                        lambda: type('DB', (), {
                            'query': lambda self, **kw: _QueryWithAnOrphan()})())
    monkeypatch.setattr('src.aiida.export.resolve_set', lambda label: object())
    uploaded = _patch_db(monkeypatch)

    result = CliRunner().invoke(cli, ['db', 'upload', '260910-afm'])

    assert result.exit_code == 0, result.output
    assert [r.run_uuid for r in uploaded] == ['r-ok']
    assert 'published=1' in result.output
    assert 'rejected=1' in result.output
