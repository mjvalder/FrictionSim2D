"""Materials Project / 2DMatPedia client, over a fake rester and fake HTTP.

Nothing here touches the network: the client takes its rester and its HTTP
GET as injected callables.
"""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip('pymatgen.core')

# pylint: disable=wrong-import-position
from src.agent.materials_db import (API, CACHE_SECONDS, NO_KEY, MPClient, MaterialsDBError,
                                    MaterialsDBUnavailable, default_materials_db,
                                    mp_api_key)
from src.agent.settings import AgentSettings

KEY = 'k' * 32


# -- fakes -------------------------------------------------------------------

class FakeResponse:
    """What the client reads off a reply: a status and a JSON body."""

    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class FakeHTTP:
    """Records each GET; answers from ``replies`` (a list, or a callable)."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def __call__(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if callable(self.replies):
            return self.replies(url, params)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class FakeSummary:
    def __init__(self, docs, error=None):
        self.docs, self.error, self.calls = docs, error, []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.docs


class FakeRester:
    """Stands in for mp_api.client.MPRester: a context manager with
    ``materials.summary.search``."""

    def __init__(self, summary):
        self.materials = SimpleNamespace(summary=summary)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _contribution(n, formula='MoS2', mp_id='mp-2815'):
    return {'identifier': mp_id, 'formula': formula,
            'data': {'details': f'http://www.2dmatpedia.org/2dmaterials/doc/2dm-{n}',
                     'process': 'top-down exfoliation',
                     'E': {'value': -7.1}, 'Ed': {'value': 0.07}, 'Eˣ': {'value': 0.02},
                     'ΔE': {'value': 0.01}, 'Evdw': {'value': -0.3}, 'mu': {'value': 0.0}},
            'structures': [{'id': f'sid{n}'}]}


def _page(rows, has_more):
    return FakeResponse(200, {'data': rows, 'has_more': has_more})


def _client(tmp_path, http=None, rester=None, now=1000.0):
    clock = SimpleNamespace(t=now)
    client = MPClient(KEY, rester_factory=rester, http_get=http,
                      cache_dir=tmp_path / 'cache', clock=lambda: clock.t)
    return client, clock


# -- key lookup ----------------------------------------------------------------

def test_key_from_environment_first(tmp_path):
    (tmp_path / '.mp_api_key').write_text('filekey\n')
    assert mp_api_key(AgentSettings(), env={'MP_API_KEY': ' envkey '},
                      home=tmp_path) == 'envkey'


def test_key_from_file_then_pmgrc(tmp_path):
    (tmp_path / '.config').mkdir()
    (tmp_path / '.config' / '.pmgrc.yaml').write_text('PMG_MAPI_KEY: pmgkey\n')
    assert mp_api_key(AgentSettings(), env={}, home=tmp_path) == 'pmgkey'
    (tmp_path / '.mp_api_key').write_text('filekey\n')
    assert mp_api_key(AgentSettings(), env={}, home=tmp_path) == 'filekey'


def test_no_key_anywhere(tmp_path):
    (tmp_path / '.config').mkdir()
    (tmp_path / '.config' / '.pmgrc.yaml').write_text('OTHER: 1\n')
    assert mp_api_key(AgentSettings(), env={'MP_API_KEY': ''}, home=tmp_path) is None


def test_disabled_setting_gives_no_key(tmp_path):
    settings = AgentSettings()
    settings.materials_project.enabled = False
    assert mp_api_key(settings, env={'MP_API_KEY': 'x'}, home=tmp_path) is None


def test_default_factory_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv('MP_API_KEY', raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    with pytest.raises(MaterialsDBError, match=NO_KEY):
        default_materials_db()


def test_default_factory_honours_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv('MP_API_KEY', 'x')
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    (tmp_path / 'FrictionSim2D').mkdir()
    (tmp_path / 'FrictionSim2D' / 'agent.yaml').write_text(
        'materials_project:\n  enabled: false\n')
    with pytest.raises(MaterialsDBError, match='turned off'):
        default_materials_db()


def test_default_factory_builds_a_client(tmp_path, monkeypatch):
    monkeypatch.setenv('MP_API_KEY', KEY)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    client = default_materials_db()
    assert isinstance(client, MPClient) and KEY not in repr(client)


# -- Materials Project summary -------------------------------------------------

def _doc(mid, ehull, bulk=None, symbol='P6_3/mmc'):
    return SimpleNamespace(material_id=mid, formula_pretty='MoS2', energy_above_hull=ehull,
                           formation_energy_per_atom=-0.9, band_gap=1.2, is_metal=False,
                           density=5.0, bulk_modulus=bulk, shear_modulus={'voigt': 30.0},
                           symmetry=SimpleNamespace(symbol=symbol))


def test_summary_reduces_formula_sorts_and_reads_moduli(tmp_path):
    summary = FakeSummary([_doc('mp-b', 0.05), _doc('mp-a', 0.0, bulk={'vrh': 50.0,
                                                                        'voigt': 60.0})])
    seen = []

    def factory(key):
        seen.append(key)
        return FakeRester(summary)
    client, _ = _client(tmp_path, rester=factory)
    rows = client.summary('Mo2S4')
    assert seen == [KEY]
    call = summary.calls[0]
    assert call['formula'] == ['MoS2']
    assert {'material_id', 'energy_above_hull', 'bulk_modulus', 'symmetry'} <= set(call['fields'])
    assert [r['material_id'] for r in rows] == ['mp-a', 'mp-b']
    assert rows[0]['bulk_modulus'] == 50.0 and rows[0]['shear_modulus'] == 30.0
    assert rows[1]['bulk_modulus'] is None and rows[0]['spacegroup'] == 'P6_3/mmc'


def test_summary_without_mp_api(tmp_path):
    def factory(_key):
        raise MaterialsDBUnavailable('pip install mp-api')
    client, _ = _client(tmp_path, rester=factory)
    with pytest.raises(MaterialsDBUnavailable, match='mp-api'):
        client.summary('MoS2')


@pytest.mark.parametrize('error, message', [
    (RuntimeError('REST query returned with error status code 401 on URL'), 'key rejected'),
    (RuntimeError('REST query returned with error status code 429'), 'rate-limited'),
    (ConnectionError('Max retries exceeded'), 'unreachable'),
])
def test_summary_errors_are_mapped(tmp_path, error, message):
    client, _ = _client(tmp_path, rester=lambda key: FakeRester(FakeSummary([], error)))
    with pytest.raises(MaterialsDBError, match=message):
        client.summary('MoS2')


def test_not_a_formula(tmp_path):
    client, _ = _client(tmp_path, rester=lambda key: FakeRester(FakeSummary([])))
    with pytest.raises(MaterialsDBError, match='not a chemical formula'):
        client.summary('graphene!!')


# -- 2DMatPedia listing ----------------------------------------------------------

def test_listing_pages_with_skip_and_flattens(tmp_path):
    http = FakeHTTP([_page([_contribution(1), _contribution(2, 'WS2', 'mp-224')], True),
                     _page([_contribution(3, 'Mo2S4')], False)])
    client, _ = _client(tmp_path, http=http)
    rows = client.twodmatpedia('MoS2')
    assert [c[1]['_skip'] for c in http.calls] == [0, 500]
    url, params, headers = http.calls[0]
    assert url == f'{API}/contributions/'
    assert params['project'] == '2dmatpedia' and params['_limit'] == 500
    assert params['_fields'] == 'identifier,formula,data,structures'
    assert '_page' not in params and headers == {'X-API-KEY': KEY}
    assert [r['dm_id'] for r in rows] == ['2dm-1', '2dm-3']
    first = rows[0]
    assert (first['mp_id'], first['structure_id'], first['process']) == \
        ('mp-2815', 'sid1', 'top-down exfoliation')
    assert (first['E'], first['Ed'], first['Ex'], first['dE'], first['Evdw'], first['mu']) == \
        (-7.1, 0.07, 0.02, 0.01, -0.3, 0.0)


def test_listing_is_cached_for_a_week(tmp_path):
    http = FakeHTTP(lambda url, params: _page([_contribution(1)], False))
    client, clock = _client(tmp_path, http=http)
    client.twodmatpedia('MoS2')
    cache = tmp_path / 'cache' / '2dmatpedia.json'
    assert cache.is_file() and KEY not in cache.read_text(encoding='utf-8')
    clock.t += CACHE_SECONDS - 10
    again, _ = _client(tmp_path, http=http, now=clock.t)
    assert again.twodmatpedia('MoS2')[0]['dm_id'] == '2dm-1'
    assert len(http.calls) == 1  # cache hit
    clock.t += 20
    expired, _ = _client(tmp_path, http=http, now=clock.t)
    expired.twodmatpedia('MoS2')
    assert len(http.calls) == 2  # expired: fetched again


def test_corrupt_cache_is_refetched(tmp_path):
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache' / '2dmatpedia.json').write_text('{not json')
    http = FakeHTTP([_page([_contribution(1)], False)])
    client, _ = _client(tmp_path, http=http)
    assert client.twodmatpedia('MoS2')[0]['dm_id'] == '2dm-1'


def test_entry_by_dm_id(tmp_path):
    http = FakeHTTP([_page([_contribution(1), _contribution(2, 'WS2')], False)])
    client, _ = _client(tmp_path, http=http)
    assert client.twodmatpedia_entry('2dm-2')['formula'] == 'WS2'
    assert client.twodmatpedia_entry('2dm-9') is None


@pytest.mark.parametrize('reply, message', [
    (FakeResponse(401), 'key rejected'), (FakeResponse(403), 'key rejected'),
    (FakeResponse(429), 'rate-limited, retry later'), (FakeResponse(500), 'HTTP 500'),
    (ConnectionError('refused'), 'unreachable'), (TimeoutError('slow'), 'unreachable'),
])
def test_http_errors_are_mapped(tmp_path, reply, message):
    client, _ = _client(tmp_path, http=FakeHTTP([reply]))
    with pytest.raises(MaterialsDBError, match=message):
        client.twodmatpedia('MoS2')


def test_errors_never_carry_the_key(tmp_path):
    client, _ = _client(tmp_path, http=FakeHTTP([ConnectionError(f'bad header {KEY}')]))
    with pytest.raises(MaterialsDBError) as info:
        client.twodmatpedia('MoS2')
    assert KEY not in str(info.value)


def test_endless_has_more_is_stopped(tmp_path):
    http = FakeHTTP(lambda url, params: _page([_contribution(params['_skip'])], True))
    client, _ = _client(tmp_path, http=http)
    with pytest.raises(MaterialsDBError, match='pages'):
        client.twodmatpedia('MoS2')


# -- structure → CIF -------------------------------------------------------------

STRUCTURE = {'lattice': {'matrix': [[3.19, 0, 0], [-1.595, 2.7626, 0], [0, 0, 20.0]]},
             'sites': [{'species': [{'element': 'Mo', 'occu': 1}], 'abc': [0, 0, 0.5]},
                       {'species': [{'element': 'S', 'occu': 1}],
                        'abc': [1 / 3, 2 / 3, 0.578]},
                       {'species': [{'element': 'S', 'occu': 1}],
                        'abc': [1 / 3, 2 / 3, 0.422]}]}


def test_structure_cif_from_payload(tmp_path):
    http = FakeHTTP([FakeResponse(200, json.loads(json.dumps(STRUCTURE)))])
    client, _ = _client(tmp_path, http=http)
    text = client.structure_cif('abc123')
    url, params, _ = http.calls[0]
    assert url == f'{API}/structures/abc123/' and params == {'_fields': '_all'}
    assert 'data_' in text and '_cell_length_a' in text
    assert text.count(' S ') + text.count(' S1 ') >= 1 and 'Mo' in text


def test_structure_bad_payload(tmp_path):
    client, _ = _client(tmp_path, http=FakeHTTP([FakeResponse(200, {'sites': []})]))
    with pytest.raises(MaterialsDBError, match='structure'):
        client.structure_cif('abc')


def test_structure_id_must_be_plain(tmp_path):
    client, _ = _client(tmp_path, http=FakeHTTP([]))
    with pytest.raises(MaterialsDBError, match='structure id'):
        client.structure_cif('../contributions')
