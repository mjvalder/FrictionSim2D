"""The API speaks schema 3 (spec §5)."""
import pytest

pytest.importorskip('fastapi')

from fastapi.testclient import TestClient  # noqa: E402

from src.api.server import ResultResponse, app  # noqa: E402
from src.data.models import ResultRecord  # noqa: E402


def test_response_model_mirrors_the_record():
    assert set(ResultResponse.model_fields) == set(ResultRecord.model_fields)


def test_query_forwards_the_new_filters(monkeypatch):
    import pandas as pd

    from src.api import auth

    seen = {}

    class _DB:
        def query(self, **kwargs):
            seen.update(kwargs)
            return pd.DataFrame(columns=list(ResultRecord.model_fields))

    auth.set_db(_DB())
    client = TestClient(app)

    resp = client.get('/results', params={
        'family': 'hetero', 'model': 'sheetonsheet', 'set_label': '260910',
        'potential_type': 'sw', 'layers': 2, 'pressure_min': 0.5,
        'pressure_max': 2.0, 'limit': 5})

    assert resp.status_code == 200
    assert seen['family'] == 'hetero'
    assert seen['model'] == 'sheetonsheet'
    assert seen['set_label'] == '260910'
    assert seen['potential_type'] == 'sw'
    assert seen['layers'] == 2
    assert seen['pressure_range'] == (0.5, 2.0)
    assert seen['limit'] == 5


def test_single_result_carries_the_pointer_fields(monkeypatch):
    import pandas as pd

    from src.api import auth

    row = {field: None for field in ResultRecord.model_fields}
    row.update(id=1, material='h-MoS2', run_uuid='r-1', result_uuid='x-1',
               set_label='260910-afm', source_host='tuxedo',
               contact='m@example.org', status='published', is_complete=True,
               materials=['h-MoS2'], extras={})

    class _DB:
        def query(self, **kwargs):
            return pd.DataFrame([row])

    auth.set_db(_DB())

    body = TestClient(app).get('/results/1').json()

    assert body['run_uuid'] == 'r-1'
    assert body['result_uuid'] == 'x-1'
    assert body['set_label'] == '260910-afm'
    assert body['source_host'] == 'tuxedo'
    assert body['contact'] == 'm@example.org'


def test_simulation_type_is_no_longer_accepted():
    from src.api.client import FrictionHTTPClient

    import inspect
    params = inspect.signature(FrictionHTTPClient.query).parameters
    assert 'simulation_type' not in params
    assert {'family', 'model', 'set_label', 'pressure_range'} <= set(params)


def test_duplicate_upload_returns_409(monkeypatch):
    from src.api import auth

    class _DB:
        def upload_result(self, **kwargs):
            return None

        def verify_api_key(self, key):
            return 'alice'

    auth.set_db(_DB())
    client = TestClient(app)

    resp = client.post('/results', json={'material': 'h-MoS2'},
                        headers={'X-API-Key': 'valid-key'})

    assert resp.status_code == 409
