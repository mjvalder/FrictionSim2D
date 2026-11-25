"""The database worker's dispatch, driven against fake backends.

The worker is the only place that touches aiida or httpx, and it runs out of
process precisely because `import aiida` is slow and process-global. These
tests call `dispatch` directly with the backend factories monkeypatched, so
they need neither aiida nor Qt. `tests/aiida/test_store_consumers.py` covers
the same path against a real temporary profile.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from frictionsim2d.aiida.query import Friction2DDB
from frictionsim2d.gui import db_worker


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
         'status': 'complete', 'reasons': '', 'run_uuid': '1a2b',
         'node_uuid': '9f8e', 'point_uuid': '5c5c', 'set_label': 's1'}]


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
    assert {'status', 'reasons', 'point_uuid'} <= set(result['columns'])


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
    # The error shape itself is pinned by the dispatch test above.
    assert code == 1
    assert payload['ok'] is False


def test_filter_keys_are_all_real_friction2d_db_query_kwargs():
    """A key here that isn't a real kwarg would be silently dropped by
    `Friction2DDB.query(**filters)` -- or worse, raise -- and `Friction2DDB`
    has no `is_complete` parameter, so the pane has no Completeness filter."""
    accepted = set(inspect.signature(Friction2DDB.query).parameters)

    assert set(db_worker.FILTER_KEYS) <= accepted
    assert 'is_complete' not in db_worker.FILTER_KEYS


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


def test_an_unknown_export_kind_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(db_worker, '_load_profile', lambda request: None)

    result = db_worker.dispatch({'kind': 'export', 'source': 'local',
                                 'what': 'everything', 'dest': str(tmp_path)})

    assert result['ok'] is False
    assert 'everything' in result['error']


def test_import_reports_a_failing_ingest_without_failing_the_request(tmp_path,
                                                                    monkeypatch):
    class FakeReport:
        # The real `ImportReport` field names: a fake with the WRONG name
        # attached would let `_import` read the wrong attribute and pass.
        label = 'x'
        group_uuid = '77aa'
        created_decks = ['afm/a:slide']
        skipped_decks = []
        superseded_decks = []
        already_imported = []
        points = ['p1', 'p2']
        verdict_counts = {'complete': 2}
        no_version = ['afm/a:system']
        log_missing = ['afm/a:slide']
        unparseable = ['afm/d']
        failures = [('afm/b', 'no results file')]
        warnings = ['afm/a: settings fallback']
        dry_run = False

        def ok(self):
            return False

        def clean(self):
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
    assert result['log_missing'] == ['afm/a:slide']
    assert result['no_version'] == ['afm/a:system']
    assert result['created_decks'] == ['afm/a:slide']
    assert result['unparseable'] == ['afm/d']
    assert result['report'] == 'ingested 1 run, 0 results (1 failure)'
    assert result['group_uuid'] == '77aa'
    assert result['n_points'] == 2
    assert result['verdict_counts'] == {'complete': 2}
    assert result['warnings'] == ['afm/a: settings fallback']
    # The placeholders are gone, not left empty.
    assert not {'set_uuid', 'empty_runs', 'incomplete'} & set(result)
    json.dumps(result)


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
