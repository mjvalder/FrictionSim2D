"""tools/publication/compare_legacy.py: store export vs legacy output_full, point by point."""

from tools.publication.compare_legacy import compare, legacy_points, schema3_points


def _legacy(rows):
    return {'results': {'h_MoS2': {'K300': {'sheet': {'N': {'A': {'l2': {'s20': {
        'p1.0': {'a90': {'columns': ['lfx', 'v_sx', 'nf'], 'data': rows}}}}}}}}}}}


def _schema3(rows):
    return {'schema': 3, 'runs': [{
        'load_kind': 'pressure', 'load': 1.0, 'angle': 90.0,
        'tiers': {'window': {'columns': ['time', 'lfx', 'react_x', 'nf'],
                             'data': [[i] + r for i, r in enumerate(rows)]}}}]}


def test_points_are_keyed_by_load_and_angle():
    assert [k for k, _ in legacy_points(_legacy([[1, 2, 3]]))] == [('p', 1.0, 90.0)]
    assert [k for k, _ in schema3_points(_schema3([[1, 2, 3]]))] == [('p', 1.0, 90.0)]


def test_identical_data_has_zero_difference():
    rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    report = compare(_legacy(rows), _schema3(rows))
    assert report.max_rel == 0.0
    assert report.missing == [] and report.extra == []
    assert {c.column for c in report.columns} == {'lfx', 'v_sx', 'nf'}


def test_renamed_column_difference_is_reported():
    report = compare(_legacy([[1.0, 2.0, 3.0]]), _schema3([[1.0, 2.5, 3.0]]))
    worst = max(report.columns, key=lambda c: c.max_rel)
    assert worst.column == 'v_sx' and worst.max_abs == 0.5


def test_a_point_missing_from_the_store_is_listed():
    legacy = _legacy([[1.0, 2.0, 3.0]])
    report = compare(legacy, {'schema': 3, 'runs': []})
    assert report.missing == [('p', 1.0, 90.0)]


def test_top_com_maps_to_the_drive_channel():
    """The registry renames v_comx_top to drivex (observables.py, drive{x,y,z})."""
    legacy = {'results': {'s20': {'p1.0': {'a90': {
        'columns': ['v_comx_top'], 'data': [[5.0]]}}}}}
    export = {'schema': 3, 'runs': [{
        'load_kind': 'pressure', 'load': 1.0, 'angle': 90.0,
        'tiers': {'window': {'columns': ['drivex'], 'data': [[5.0]]}}}]}
    report = compare(legacy, export)
    assert report.notes == [] and report.columns[0].max_abs == 0.0
