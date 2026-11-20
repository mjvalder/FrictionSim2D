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


def _contact_table(first):
    return pd.DataFrame([{'run_uuid': r, 'load': f, 'angle': 0.0, 'state': s,
                          'contact_area': area}
                         for r, f in zip(first['run_uuid'], first['load'])
                         for s, area in (('load', 40.0 + f), ('final', 1.0))])


def test_sliding_physics_recomputes_when_a_contact_result_appears(store):  # noqa: F811
    first = run_analysis('sliding_physics', 'paper', PERIOD)
    assert 'shear_strength_GPa' not in first.columns
    contact_uuid = store_result(get_analysis('contact'), 'paper', {}, _contact_table(first),
                                dropped=[])
    table = run_analysis('sliding_physics', 'paper', PERIOD)      # no force
    by_load = table.set_index('load')
    for load in (30.0, 50.0):
        assert by_load.loc[load, 'contact_area_A2'] == 40.0 + load
        assert by_load.loc[load, 'shear_strength_GPa'] == pytest.approx(
            by_load.loc[load, 'lf_mean_nN'] / (40.0 + load) * 100.0)
    node = find_result_node('sliding_physics', 'paper')
    assert node.base.attributes.get('notes') == []
    contact = orm.load_node(contact_uuid)
    assert node.base.attributes.get('depends') == {'contact': {
        'uuid': contact_uuid, 'version': contact.base.extras.get('version'),
        'params_hash': contact.base.extras.get('params_hash')}}
    # the contact result is an input of the calcfunction that recorded this one
    creator = node.creator
    assert creator.inputs.contact.uuid == contact_uuid
    # unchanged inputs: reused, not recomputed
    again = run_analysis('sliding_physics', 'paper', PERIOD)
    assert find_result_node('sliding_physics', 'paper').uuid == node.uuid
    pd.testing.assert_frame_equal(again, table)
    # a newer contact result: recomputed again
    store_result(get_analysis('contact'), 'paper', {}, _contact_table(first), dropped=[])
    run_analysis('sliding_physics', 'paper', PERIOD)
    assert find_result_node('sliding_physics', 'paper').uuid != node.uuid
