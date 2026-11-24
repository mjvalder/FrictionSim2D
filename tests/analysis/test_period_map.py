"""`period_map`: the layer a material's ``a_x`` is read from."""
from types import SimpleNamespace

import pandas as pd

from frictionsim2d.analysis import periods
from frictionsim2d.analysis.store import _to_csv


def _source(table):
    node = SimpleNamespace(get_content=lambda mode='rb': _to_csv(table))
    return SimpleNamespace(depends={periods.GEOMETRY: node})


def test_layer_two_wins_then_layer_one_then_the_highest():
    table = pd.DataFrame({
        'material': ['a', 'a', 'a', 'b', 'b', 'c', 'c'],
        'layer': [1, 2, 3, 1, 3, 3, 4],
        'period_x_top': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})

    out = periods.period_map(_source(table), periods.PeriodParams())

    assert out == {'a': 2.0, 'b': 4.0, 'c': 7.0}


def test_period_x_given_overrides_the_lattice_and_reads_no_geometry():
    out = periods.period_map(SimpleNamespace(), periods.PeriodParams(
        period_x={'a': 3.0}, lattice={'a': 1.0, 'b': 2.0}))

    assert out == {'a': 3.0, 'b': 2.0}
