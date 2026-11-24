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


def test_summary_csv_rows_are_unchanged():
    import pandas as pd  # pylint: disable=import-outside-toplevel
    out = PlotCommonMixin._dataset_to_summary_rows('lab', _RECORDS)  # pylint: disable=protected-access

    assert pd.DataFrame(out).to_csv(index=False) == (
        'materials,size_x,size_y,load_kind,load,layers,angle,speed,pot_token,dataset_label,'
        'file_key,size,id,material,potential,layer,status,force,pressure,flake_material\n'
        'h_MoS2,100.0,100.0,force,5.0,2.0,0.0,2.0,sw,lab,100x100y,100x100y,h_MoS2,h_MoS2,sw,'
        '2.0,complete,5.0,,\n'
        'h_MoS2+t-WS2,12.5,30.0,pressure,0.5,1.0,,,,lab,12.5x30y,12.5x30y,'
        'h_MoS2+t_WS2+flake_h_C,h_MoS2+t_WS2+flake_h_C,,1.0,complete,,0.5,h-C\n'
        'h_MoS2,,,force,1.5,,,,,lab,unknownx,unknownx,h_MoS2,h_MoS2,,,complete,1.5,,\n')
