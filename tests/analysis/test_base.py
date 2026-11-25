"""The analysis registry and the run source (plan Task 1), without a store."""
from importlib import import_module
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import BaseModel

from frictionsim2d.analysis import base
from frictionsim2d.analysis.base import Analysis, get_analysis, register
from frictionsim2d.analysis.source import RunSource


@pytest.fixture
def registry(monkeypatch):
    """A private copy of the registry: test analyses never leak out."""
    # Register the catalogue first: copied before its lazy import, the copy would
    # receive the registrations and later tests would see an empty registry.
    import_module('frictionsim2d.analysis.catalogue')
    monkeypatch.setattr(base, 'ANALYSES', dict(base.ANALYSES))
    return base.ANALYSES


def _analysis(name):
    class _Probe(Analysis):
        version = '1'
        level = 'run'
        families = frozenset({'afm'})

        class Params(BaseModel):
            """No parameters."""

        def compute(self, source, params):
            return pd.DataFrame()

    _Probe.name = name
    return _Probe


def test_registry_refuses_duplicate_names(registry):
    first = register(_analysis('probe'))

    assert get_analysis('probe') is first
    with pytest.raises(ValueError, match='probe'):
        register(_analysis('probe'))
    assert registry['probe'] is first


def test_unknown_analysis_is_a_key_error(registry):
    with pytest.raises(KeyError, match='nope'):
        get_analysis('nope')


def _rows(*specs):
    return pd.DataFrame([{'point_uuid': uuid, 'calc_uuid': f'c-{uuid}',
                          'node_uuid': f'n-{uuid}', 'family': family,
                          'verdict_status': status, 'load': 10.0}
                         for uuid, family, status in specs])


def test_source_drops_non_pass_points():
    seen = []
    fetch = SimpleNamespace(frames=lambda row: seen.append(row.node_uuid) or {'window': 1})
    source = RunSource(_rows(('p1', 'afm', 'complete'), ('p2', 'afm', 'failed_integrity'),
                             ('p3', 'afm', 'complete')), None, fetch)

    assert source.family == 'afm'
    assert list(source.rows.point_uuid) == ['p1', 'p3']
    assert source.dropped == [('p2', 'failed_integrity')]
    assert source.frames(next(source.rows.itertuples())) == {'window': 1}
    assert seen == ['n-p1']


def test_mixed_family_dataset_is_refused():
    with pytest.raises(ValueError, match='afm.*sheetonsheet|sheetonsheet.*afm'):
        RunSource(_rows(('p1', 'afm', 'complete'), ('p2', 'sheetonsheet', 'complete')),
                  None, SimpleNamespace())


def test_given_family_must_match_the_rows():
    with pytest.raises(ValueError, match='afm'):
        RunSource(_rows(('p1', 'afm', 'complete')), 'sheetonsheet', SimpleNamespace())


def test_parse_params_validates_names_and_values(registry):
    class _Typed(_analysis('typed')):
        class Params(BaseModel):
            """One integer."""
            n: int = 1
    _Typed.name = 'typed'
    register(_Typed)
    assert base.parse_params('typed', ['n=3']) == {'n': 3}
    assert base.parse_params('typed', []) == {}
    for pairs, match in ((['n'], 'expected NAME=VALUE'), (['m=1'], 'no such parameter'),
                         (['n=x'], '--param n:')):
        with pytest.raises(ValueError, match=match):
            base.parse_params('typed', pairs)
    with pytest.raises(ValueError, match='no analysis named'):
        base.parse_params('nope', [])
    assert any(line.startswith('typed\t1\trun\tafm') for line in base.catalogue_lines())


def test_stored_result_says_what_to_run(monkeypatch):
    pytest.importorskip('aiida')
    from frictionsim2d.analysis import store
    monkeypatch.setattr(store, 'select_result_node', lambda *a: None)
    with pytest.raises(LookupError, match='run: FrictionSim2D analyse run probe paper'):
        store.stored_result('probe', 'paper')

    def _refuse(*_args):
        raise ValueError('probe has no parameter q')
    monkeypatch.setattr(store, 'select_result_node', _refuse)
    with pytest.raises(LookupError, match='no parameter q'):
        store.stored_result('probe', 'paper', {'q': 1})
