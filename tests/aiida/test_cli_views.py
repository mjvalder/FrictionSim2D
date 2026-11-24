"""The CLI's views of the store (`aiida query`, `aiida list-sets`, `compare`), no profile."""
import pandas as pd
import pytest

from frictionsim2d.aiida.compare import sidecar_dir
from frictionsim2d.aiida.query import query_filters, query_lines, render_sets, verdict_first


def test_query_filters_pairs_each_axis_into_a_range():
    filters = query_filters(materials=('a', 'b'), limit=3, order_by=None, family='afm',
                            model=None, set_label=None, potential=None, layers=2,
                            force_min=1.0, force_max=None, pressure_min=None,
                            pressure_max=None, angle_min=None, angle_max=None,
                            speed_min=None, speed_max=None, temperature_min=None,
                            temperature_max=None)
    assert list(filters) == ['materials', 'family', 'model', 'set_label', 'potential',
                             'layers', 'force_range', 'pressure_range', 'angle_range',
                             'speed_range', 'temperature_range', 'limit', 'order_by']
    assert filters['materials'] == ['a', 'b'] and filters['force_range'] == (1.0, None)
    assert filters['angle_range'] is None and filters['limit'] == 3


def test_verdict_first_and_the_query_lines():
    class _Result:
        total_count, rows = 2, [{}]

        @staticmethod
        def to_dataframe():
            return pd.DataFrame([{'mean_cof': 0.1, 'status': 'complete', 'rel_path': 'x'}])

    assert list(verdict_first(_Result.to_dataframe()).columns) == ['rel_path', 'status',
                                                                   'mean_cof']
    lines = list(query_lines(_Result(), 'csv', None))
    assert lines[:2] == ["📊 Found 2 results", "Showing 1 of 2"]
    assert lines[2].splitlines()[0] == 'rel_path,status,mean_cof'


def test_render_sets_table_and_csv():
    rows = [{'label': 'camp', 'model': 'afm', 'n_runs': 2, 'uuid': 'u'}]
    table = render_sets(rows, 'table').splitlines()
    assert table[0].split() == ['Label', 'Model', 'Runs', 'Results', 'Materials', 'Host', 'UUID']
    assert table[2].split() == ['camp', 'afm', '2', 'u'] and table[-1] == 'Total sets: 1'
    assert render_sets(rows, 'csv').splitlines()[1] == 'camp,afm,,2,,,,,u,'


def test_sidecar_dir_finds_one_params_child_and_refuses_two(tmp_path):
    assert sidecar_dir(tmp_path / 'missing') is None and sidecar_dir(tmp_path) is None
    one = tmp_path / 'a.params'
    one.mkdir()
    (one / 'schema-1.json').write_text('{}')
    assert sidecar_dir(tmp_path) == one and sidecar_dir(one) == one
    two = tmp_path / 'b.params'
    two.mkdir()
    (two / 'schema-1.json').write_text('{}')
    with pytest.raises(ValueError, match='several sidecars'):
        sidecar_dir(tmp_path)
