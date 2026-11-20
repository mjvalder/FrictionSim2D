"""`sliding_physics` against a real store (plan Task 10): the shear strength is
joined from the dataset's stored `contact` result, or left out with a stored note."""
import pandas as pd
import pytest

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from src.analysis.base import get_analysis  # noqa: E402
from src.analysis.store import find_result_node, run_analysis, store_result  # noqa: E402
from tests.analysis.test_store import no_profile_switch, store  # noqa: E402,F401

PERIOD = {'period_x': {'h_MoS2': 3.16}}


def test_sliding_physics_without_contact_leaves_tau_out_with_a_note(store):  # noqa: F811
    table = run_analysis('sliding_physics', 'paper', PERIOD)
    assert sorted(table['load']) == [30.0, 50.0]          # 10 N fails its integrity check
    assert 'shear_strength_GPa' not in table.columns
    node = find_result_node('sliding_physics', 'paper')
    notes = node.base.attributes.get('notes')
    assert len(notes) == 1 and 'contact' in notes[0]


def test_sliding_physics_joins_the_stored_contact(store):  # noqa: F811
    first = run_analysis('sliding_physics', 'paper', PERIOD)
    contact = pd.DataFrame([{'run_uuid': r, 'load': f, 'angle': 0.0, 'state': s,
                             'contact_area': area}
                            for r, f in zip(first['run_uuid'], first['load'])
                            for s, area in (('load', 40.0 + f), ('final', 1.0))])
    store_result(get_analysis('contact'), 'paper', {}, contact, dropped=[])
    table = run_analysis('sliding_physics', 'paper', PERIOD, force=True)
    by_load = table.set_index('load')
    for load in (30.0, 50.0):
        assert by_load.loc[load, 'contact_area_A2'] == 40.0 + load
        assert by_load.loc[load, 'shear_strength_GPa'] == pytest.approx(
            by_load.loc[load, 'lf_mean_nN'] / (40.0 + load) * 100.0)
    node = find_result_node('sliding_physics', 'paper')
    assert node.base.attributes.get('notes') == []
    assert isinstance(orm.load_node(node.uuid), orm.SinglefileData)
