"""The summary rows' identity spelling: the plot stack's store rows and the
analysis tables' `summary_identity` rows say it the same way."""
from types import SimpleNamespace

from frictionsim2d.analysis.periods import summary_identity
from frictionsim2d.postprocessing.plot_common import PlotCommonMixin

_RECORDS = [
    {'materials': ['h_MoS2'], 'size_x': 100.0, 'size_y': 100, 'load_kind': 'force',
     'load': 5, 'layers': 2, 'angle': 0, 'speed': 2, 'pot_token': 'sw'},
    {'materials': ['h_MoS2', 't-WS2'], 'flake_material': 'h-C', 'size_x': 12.5,
     'size_y': 30.0, 'load_kind': 'pressure', 'load': 0.5, 'layers': 1},
    {'materials': ['h_MoS2'], 'size_x': None, 'size_y': None, 'load_kind': 'force',
     'load': 1.5},
]


def test_dataset_rows_are_spelled_as_before():
    out = PlotCommonMixin._dataset_to_summary_rows('lab', _RECORDS)  # pylint: disable=protected-access

    first, second, third = out
    assert (first['id'], first['material'], first['materials']) == ('h_MoS2',) * 3
    assert first['size'] == first['file_key'] == '100x100y'
    assert first['force'] == 5 and str(first['force']) == '5'
    assert first['pressure'] != first['pressure']
    assert first['layer'] == 2 and first['potential'] == 'sw'
    assert second['id'] == 'h_MoS2+t_WS2+flake_h_C'
    assert second['materials'] == 'h_MoS2+t-WS2'
    assert second['size'] == '12.5x30y'
    assert second['pressure'] == 0.5 and second['force'] != second['force']
    assert third['size'] == third['file_key'] == 'unknownx'
    assert third['status'] == 'complete' and third['dataset_label'] == 'lab'


def test_summary_identity_spells_the_same_columns():
    row = SimpleNamespace(materials=['h_MoS2', 't-WS2'], flake_material='h-C',
                          size_x=12.5, size_y=30.0, load_kind='pressure', load=0.5,
                          layers=1, angle=0, speed=2, point_uuid='p', run_uuid='r')

    got = summary_identity(row)

    assert list(got) == ['run_uuid', 'point_uuid', 'id', 'material', 'size', 'layer',
                         'load', 'load_kind', 'force', 'pressure', 'angle', 'speed']
    assert got['id'] == got['material'] == 'h_MoS2+t_WS2+flake_h_C'
    assert got['size'] == '12.5x30y' and got['layer'] == 1
    assert got['pressure'] == 0.5 and got['force'] != got['force']
