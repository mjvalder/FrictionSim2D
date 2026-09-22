"""The database worker's dispatch, driven against fake backends.

The worker is the only place that touches aiida or httpx, and it runs out of
process precisely because `import aiida` is slow and process-global. These
tests call `dispatch` directly with the backend factories monkeypatched, so
they need neither aiida nor Qt. `tests/aiida/test_db_worker_local.py` covers
the same path against a real temporary profile.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from src.aiida.query import Friction2DDB
from src.gui import db_worker


class FakeQueryResult:
    def __init__(self, rows):
        self._rows = rows
        self.total_count = len(rows)
        self.runs = []          # QueryResult carries this; unused by dispatch

    def to_dataframe(self):
        pandas = pytest.importorskip('pandas')
        return pandas.DataFrame(self._rows)


class FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query(self, **filters):
        self.calls.append(filters)
        return FakeQueryResult(self.rows)

    def get_available_materials(self):
        return ['h-MoS2', 'h-WS2']

    def get_available_conditions(self):
        return {'families': ['afm'], 'models': ['afm'], 'potentials': ['sw'],
                'layers': [1, 2], 'temperatures': [300.0], 'set_labels': ['s1']}

    def list_sets(self):
        return [{'label': 's1', 'n_runs': 4}]


ROWS = [{'rel_path': 'afm/h-MoS2/100x_100y/K300/L2', 'family': 'afm', 'layers': 2,
         'load_kind': 'force', 'load': 30.0, 'mean_cof': 0.0731,
         'run_uuid': '1a2b', 'result_uuid': '9f8e', 'set_label': 's1'}]


@pytest.fixture(name='fake_db')
def _fake_db(monkeypatch):
    db = FakeDB(ROWS)
    monkeypatch.setattr(db_worker, '_local_db', lambda request: db)
    monkeypatch.setattr(db_worker, '_load_profile', lambda request: 'fake-profile')
    return db


def test_list_returns_records_facets_and_counts(fake_db):
    result = db_worker.dispatch({'kind': 'list', 'source': 'local', 'profile': None,
                                 'filters': {'family': 'afm', 'layers': 2}})

    assert result['ok'] is True
    assert result['kind'] == 'list'
    assert result['records'][0]['rel_path'] == ROWS[0]['rel_path']
    assert result['total_count'] == 1
    assert result['returned'] == 1
    assert result['facets']['materials'] == ['h-MoS2', 'h-WS2']
    assert result['facets']['layers'] == [1, 2]
    assert isinstance(result['elapsed_s'], float)
    assert result['set_sizes'] == {'s1': 4}
    assert result['warnings'] == [], 'every facet came from the store'


def test_a_backfilled_facet_is_reported_as_a_warning(fake_db):
    """A choice list derived from one page of rows is a weaker promise than
    one derived from the store, and the pane says so."""
    fake_db.get_available_conditions = lambda: {'families': ['afm']}

    result = db_worker.dispatch({'kind': 'list', 'source': 'local', 'filters': {}})

    assert result['facets']['set_labels'] == ['s1']
    assert result['warnings'] and 'rows fetched' in result['warnings'][0]


def test_list_passes_only_real_filters_to_the_query(fake_db):
    db_worker.dispatch({'kind': 'list', 'source': 'local',
                        'filters': {'family': 'afm', 'model': None,
                                    'nonsense': 'ignored', 'limit': 500}})

    assert fake_db.calls == [{'family': 'afm', 'limit': 500}]


def test_list_records_are_json_serialisable(fake_db):
    """`DataFrame.to_dict` hands back numpy scalars, which `json.dumps`
    refuses -- the result file would never be written."""
    result = db_worker.dispatch({'kind': 'list', 'source': 'local', 'filters': {}})

    json.dumps(result)  # must not raise


def test_an_unknown_kind_is_the_error_shape():
    result = db_worker.dispatch({'kind': 'wat', 'source': 'local'})

    assert result['ok'] is False
    assert result['kind'] == 'wat'
    assert 'wat' in result['error']
    assert result['traceback'] == ''


def test_dispatch_turns_a_raising_handler_into_the_error_shape(monkeypatch):
    def _boom(request):
        raise RuntimeError('no profile named friction2d')

    monkeypatch.setitem(db_worker.HANDLERS, 'list', _boom)

    result = db_worker.dispatch({'kind': 'list', 'source': 'local', 'filters': {}})

    assert result['ok'] is False
    assert result['kind'] == 'list'
    assert result['error'] == 'RuntimeError: no profile named friction2d'
    assert 'Traceback' in result['traceback']


def test_main_writes_the_error_shape_when_a_handler_raises(tmp_path, monkeypatch):
    def _boom(request):
        raise RuntimeError('no profile named friction2d')

    monkeypatch.setitem(db_worker.HANDLERS, 'list', _boom)
    result_file = tmp_path / 'result.json'
    request_file = tmp_path / 'request.json'
    request_file.write_text(json.dumps(
        {'kind': 'list', 'source': 'local', 'filters': {},
         'result_file': str(result_file)}), encoding='utf-8')

    code = db_worker.main([str(request_file)])

    payload = json.loads(result_file.read_text(encoding='utf-8'))
    assert code == 1
    assert payload['ok'] is False
    assert payload['kind'] == 'list'
    assert payload['error'] == 'RuntimeError: no profile named friction2d'
    assert 'Traceback' in payload['traceback']


def test_filter_keys_are_all_real_friction2d_db_query_kwargs():
    """A key here that isn't a real kwarg would be silently dropped by
    `Friction2DDB.query(**filters)` -- or worse, raise -- and `Friction2DDB`
    has no `is_complete` parameter, so the pane has no Completeness filter."""
    accepted = set(inspect.signature(Friction2DDB.query).parameters)

    assert set(db_worker.FILTER_KEYS) <= accepted
    assert 'is_complete' not in db_worker.FILTER_KEYS


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []
        self.closed = False

    def query(self, **filters):
        # The real client returns a DataFrame, not a list of dicts.
        pandas = pytest.importorskip('pandas')
        self.calls.append(filters)
        return pandas.DataFrame(self.rows)

    def get_result(self, result_id):
        return dict(self.rows[0], id=result_id)

    def list_materials(self):
        return ['h-MoS2']

    def close(self):
        self.closed = True


CENTRAL_ROWS = [{'id': 41, 'material': 'h_MoS2', 'family': 'afm', 'mean_cof': 0.07,
                 'run_uuid': '1a2b', 'result_uuid': '9f8e', 'set_label': 's1',
                 'contact': 'someone@example.org', 'source_host': 'cluster-a',
                 'uploaded_at': '2026-09-01T10:00:00'}]


@pytest.fixture(name='fake_client')
def _fake_client(monkeypatch):
    client = FakeClient(CENTRAL_ROWS)
    monkeypatch.setattr(db_worker, '_central_client', lambda request: client)
    return client


def test_central_list_maps_filter_names_to_the_client(fake_client):
    result = db_worker.dispatch({
        'kind': 'list', 'source': 'central', 'api_url': 'http://x',
        'filters': {'materials': ['h-MoS2'], 'potential': 'sw', 'family': 'afm',
                    'limit': 200}})

    assert result['ok'] is True
    assert result['records'] == CENTRAL_ROWS
    assert result['total_count'] == 1
    assert result['limit'] == 200, \
        'not truncated (1 row < limit 200) -- total_count is exact, but the ' \
        'limit is still reported'
    assert fake_client.calls[0]['material'] == 'h-MoS2'
    assert fake_client.calls[0]['potential_type'] == 'sw'
    assert fake_client.calls[0]['family'] == 'afm'
    assert fake_client.calls[0]['limit'] == 200
    assert 'potential' not in fake_client.calls[0]
    assert 'materials' not in fake_client.calls[0]


def test_central_total_count_is_unknown_when_the_limit_is_reached(fake_client):
    """The REST client has no way to ask the store for a true total -- it only
    ever returns what it fetched, up to `limit`. When the returned row count
    reaches that cap, `total_count` must not claim that count is the store's
    total; `limit` is reported instead so the pane can say
    "N results (limit L)" using `returned`."""
    result = db_worker.dispatch({
        'kind': 'list', 'source': 'central',
        'filters': {'limit': len(CENTRAL_ROWS)}})

    assert result['returned'] == len(CENTRAL_ROWS)
    assert result['total_count'] is None
    assert result['limit'] == len(CENTRAL_ROWS)


def test_central_facets_are_read_off_the_rows(fake_client):
    """The REST /conditions endpoint answers with ranges and count maps and
    has no potential or set axis, so the fetched rows are the choices."""
    result = db_worker.dispatch({'kind': 'list', 'source': 'central', 'filters': {}})

    assert result['facets']['materials'] == ['h-MoS2']   # from list_materials()
    assert result['facets']['families'] == ['afm']
    assert result['facets']['set_labels'] == ['s1']
    assert result['facets']['models'] == []


def test_central_detail_wraps_the_record_in_the_shared_shape(fake_client):
    result = db_worker.dispatch({'kind': 'detail', 'source': 'central',
                                 'result_id': 41})

    assert result['ok'] is True
    assert result['kind'] == 'detail'
    assert result['result']['attributes']['id'] == 41
    assert result['result']['uuid'] == '9f8e'
    assert result['run']['attributes'] == {}
    assert result['files'] == []
    assert result['total_bytes'] == 0


def test_central_detail_carries_the_owner_contact(fake_client):
    """The files of a central row live on someone else's machine; the
    contact is how a reader asks for them (D6)."""
    result = db_worker.dispatch({'kind': 'detail', 'source': 'central',
                                 'result_id': 41})

    assert result['result']['attributes']['contact'] == 'someone@example.org'
    assert result['result']['attributes']['source_host'] == 'cluster-a'


def test_summary_csv_writes_the_visible_rows_in_column_order(tmp_path):
    result = db_worker.dispatch({
        'kind': 'export', 'source': 'local', 'what': 'summary_csv',
        'dest': str(tmp_path / 'out'),
        'records': [{'family': 'afm', 'materials': ['h-MoS2'], 'mean_cof': 0.07,
                     'extras': {'gap_min': 3.1}, 'layers': None}],
        'columns': ['family', 'materials', 'mean_cof', 'extras', 'layers']})

    assert result['ok'] is True
    assert result['what'] == 'summary_csv'
    written = Path(result['files'][0]).read_text(encoding='utf-8').splitlines()
    assert written[0] == 'family,materials,mean_cof,extras,layers'
    assert written[1] == 'afm,h-MoS2,0.07,"{""gap_min"": 3.1}",'
    assert result['bytes'] > 0


def test_summary_csv_is_the_only_export_the_central_source_allows(tmp_path):
    result = db_worker.dispatch({
        'kind': 'export', 'source': 'central', 'what': 'results',
        'dest': str(tmp_path / 'out'), 'selection': []})

    assert result['ok'] is False
    assert "'results'" in result['error']


def test_an_unknown_export_kind_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(db_worker, '_load_profile', lambda request: None)

    result = db_worker.dispatch({'kind': 'export', 'source': 'local',
                                 'what': 'everything', 'dest': str(tmp_path)})

    assert result['ok'] is False
    assert 'everything' in result['error']


def test_import_is_refused_against_the_central_database(tmp_path):
    result = db_worker.dispatch({'kind': 'import', 'source': 'central',
                                 'root': str(tmp_path), 'label': 'x'})

    assert result['ok'] is False
    assert 'read-only' in result['error']


def test_import_reports_a_failing_ingest_without_failing_the_request(tmp_path,
                                                                    monkeypatch):
    class FakeReport:
        # `incomplete_results` is the real `IngestReport` field name -- using
        # anything else here would let `_import` read the wrong attribute
        # and still pass, since a fake object with no such attribute would
        # raise AttributeError and get caught, but a fake with the WRONG
        # name attached instead would silently return the wrong value.
        label = 'x'
        set_uuid = '77aa'
        created_runs = ['afm/a']
        created_results = []
        skipped_results = []
        incomplete_results = ['afm/a#force30_a0_s2']
        empty_runs = ['afm/c']
        unparseable = ['afm/d']
        failures = [('afm/b', 'no results file')]
        dry_run = False

        def ok(self):
            return False

        def render(self):
            return 'ingested 1 run, 0 results (1 failure)'

    monkeypatch.setattr(db_worker, '_load_profile', lambda request: None)
    monkeypatch.setattr(db_worker, '_ingest_root',
                        lambda *args, **kwargs: FakeReport())

    result = db_worker.dispatch({'kind': 'import', 'source': 'local',
                                 'root': str(tmp_path), 'label': 'x',
                                 'dry_run': False})

    assert result['ok'] is True, 'the request ran; the ingest is what failed'
    assert result['ingest_ok'] is False
    assert result['failures'] == [['afm/b', 'no results file']]
    assert result['incomplete'] == ['afm/a#force30_a0_s2']
    assert result['empty_runs'] == ['afm/c']
    assert result['unparseable'] == ['afm/d']
    assert result['report'] == 'ingested 1 run, 0 results (1 failure)'
    assert result['set_uuid'] == '77aa'


def test_a_local_check_reports_the_profile_and_the_set_count(monkeypatch):
    monkeypatch.setattr(db_worker, '_load_profile', lambda request: 'friction2d')
    monkeypatch.setattr(db_worker, '_local_db',
                        lambda request: type('DB', (), {'list_sets': lambda self: [1, 2]})())

    result = db_worker.dispatch({'kind': 'check', 'source': 'local',
                                 'profile': 'friction2d'})

    assert result['ok'] is True
    assert result['ready'] is True
    names = [c['name'] for c in result['checks']]
    assert 'profile loaded' in names
    assert [c for c in result['checks'] if c['name'] == 'sets in the store'][0]['detail'] == '2'


def test_a_local_check_turns_a_broken_profile_into_a_failed_check(monkeypatch):
    def _boom(request):
        raise RuntimeError('no profile named friction2d')

    monkeypatch.setattr(db_worker, '_load_profile', _boom)

    result = db_worker.dispatch({'kind': 'check', 'source': 'local'})

    assert result['ok'] is True, 'the check ran -- it is the profile that is broken'
    assert result['ready'] is False
    failed = [c for c in result['checks'] if not c['ok']]
    assert 'no profile named friction2d' in failed[0]['detail']
    assert result['notes'], 'a failed check must say what to do about it'


def test_a_central_check_calls_the_statistics_endpoint(fake_client):
    fake_client.get_statistics = lambda: {'total_rows': 91}

    result = db_worker.dispatch({'kind': 'check', 'source': 'central',
                                 'api_url': 'http://localhost:8000'})

    assert result['ready'] is True
    assert [c for c in result['checks']
            if c['name'] == 'GET /statistics'][0]['detail'] == '91 results'


def test_a_central_check_without_a_url_fails_before_connecting():
    result = db_worker.dispatch({'kind': 'check', 'source': 'central',
                                 'api_url': ''})

    assert result['ready'] is False
    assert any('not configured' in c['detail'] for c in result['checks'])
    assert any('Local store' in note for note in result['notes'])


def test_an_unconfigured_central_check_never_connects(monkeypatch):
    """`configured` False means the URL is only the built-in default: there
    is no central database there, and a connection attempt would end as a
    ConnectionError string the user cannot act on."""
    def _never(request):
        raise AssertionError('the check must not build a client')

    monkeypatch.setattr(db_worker, '_central_client', _never)

    result = db_worker.dispatch({'kind': 'check', 'source': 'central',
                                 'api_url': 'http://localhost:8000',
                                 'configured': False})

    assert result['ready'] is False
