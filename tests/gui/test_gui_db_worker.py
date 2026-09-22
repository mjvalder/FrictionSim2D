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
