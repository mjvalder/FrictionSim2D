"""The `sheet_geometry` result a period analysis reads is part of its identity:
linked as an input and keyed in its reuse, whether it is the analysed dataset's
own or the one `geometry_dataset` names. Against the store of
`tests/analysis/test_store.py` (dataset `paper`: 30 and 50 nN pass)."""
import pandas as pd
import pytest

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

from frictionsim2d.aiida.steps import build_dataset  # noqa: E402
from frictionsim2d.analysis.base import get_analysis  # noqa: E402
from frictionsim2d.analysis.store import find_result_node, params_hash, store_result  # noqa: E402
from tests.analysis.helpers import run_analysis  # noqa: E402
from tests.analysis.test_store import no_profile_switch, store  # noqa: E402,F401

PERIOD_ANALYSES = ('period_summary', 'cof_ranking', 'contact', 'stick_slip', 'sliding_physics')


def _geometry(dataset, period):
    """A stored `sheet_geometry` result of ``dataset`` giving h_MoS2 ``period``."""
    table = pd.DataFrame([{'material': 'h_MoS2', 'layer': 2, 'period_x_top': period}])
    return store_result(get_analysis('sheet_geometry'), dataset, {}, table, dropped=[])


def _node(name, dataset, params=None):
    cls = get_analysis(name)
    digest = params_hash(cls.Params(**(params or {})).model_dump(mode='json'))
    return find_result_node(name, dataset, digest)


def test_own_geometry_is_linked_and_a_new_one_recomputes(store):  # noqa: F811
    first_geometry = _geometry('paper', 3.16)
    first = run_analysis('period_summary', 'paper')
    node = _node('period_summary', 'paper')

    assert set(first['period_x_A']) == {3.16}
    assert node.creator.inputs.sheet_geometry.uuid == first_geometry
    run_analysis('period_summary', 'paper')
    assert _node('period_summary', 'paper').uuid == node.uuid       # reused

    _geometry('paper', 3.2)                                          # a newer result
    again = run_analysis('period_summary', 'paper')
    assert _node('period_summary', 'paper').uuid != node.uuid
    assert set(again['period_x_A']) == {3.2}


def test_a_new_geometry_version_recomputes(store, monkeypatch):  # noqa: F811
    _geometry('paper', 3.16)
    run_analysis('period_summary', 'paper')
    node = _node('period_summary', 'paper')

    monkeypatch.setattr(get_analysis('sheet_geometry'), 'version', '99')
    newer = _geometry('paper', 3.16)
    run_analysis('period_summary', 'paper')
    again = _node('period_summary', 'paper')
    assert again.uuid != node.uuid
    assert again.creator.inputs.sheet_geometry.uuid == newer


def test_geometry_dataset_result_is_linked_and_keyed(store):  # noqa: F811
    build_dataset('afm', ['camp'])
    other = _geometry('afm', 3.0)
    params = {'geometry_dataset': 'afm'}
    table = run_analysis('period_summary', 'paper', params)
    node = _node('period_summary', 'paper', params)

    assert set(table['period_x_A']) == {3.0}
    assert node.creator.inputs.sheet_geometry.uuid == other
    _geometry('afm', 3.1)
    run_analysis('period_summary', 'paper', params)
    assert _node('period_summary', 'paper', params).uuid != node.uuid


@pytest.mark.parametrize('name', PERIOD_ANALYSES)
def test_period_analyses_depend_on_the_geometry_they_read(store, name):  # noqa: F811
    own = _geometry('paper', 3.16)
    cls = get_analysis(name)

    assert cls.depends_on('paper', cls.Params())['sheet_geometry'].uuid == own
    assert 'sheet_geometry' not in cls.depends_on('paper', cls.Params(period_x={'h_MoS2': 3.0}))
