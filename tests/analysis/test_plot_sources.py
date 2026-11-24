"""Stored analyses as plot sources (plan Task 5), against a real store.

The store is `tests/analysis/test_store.py`'s: one AFM run at 10/30/50 nN
imported as set `camp` (10 nN fails its integrity check), dataset `paper`.
"""
import pandas as pd
import pytest

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

import matplotlib  # noqa: E402

matplotlib.use('Agg')

import frictionsim2d.analysis.catalogue  # noqa: E402,F401  (registers the analyses)
from frictionsim2d.analysis.store import store_result  # noqa: E402
from tests.analysis.helpers import run_analysis  # noqa: E402
from frictionsim2d.postprocessing.plot_config import TableSource  # noqa: E402
from frictionsim2d.postprocessing.plot_data import PlotDataError, Plotter  # noqa: E402
from frictionsim2d.postprocessing.plot_tables import load_table  # noqa: E402
from tests.analysis.test_store import no_profile_switch, probe, store  # noqa: E402,F401


def test_table_source_resolves_an_analysis(store, probe, tmp_path):  # noqa: F811
    table = pd.DataFrame({'material': ['A', 'B'], 'eligible': [True, False],
                          'value': [0.1 + 0.2, 1.0 / 3.0]})
    store_result(probe, 'paper', {}, table, dropped=[])

    got = load_table(TableSource(analysis='probe', dataset='paper'),
                     required=['material', 'value'])
    pd.testing.assert_frame_equal(got, table)

    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path / 'plots'))
    eligible = plotter._eligible_materials(  # pylint: disable=protected-access
        TableSource(analysis='probe', dataset='paper').model_dump())
    assert list(eligible.index) == ['A']
    with pytest.raises(PlotDataError, match='analyse run probe paper'):
        load_table(TableSource(analysis='probe', dataset='paper', version='9'))


def test_missing_analysis_names_the_command(store, tmp_path):  # noqa: F811
    with pytest.raises(PlotDataError,
                       match='no stored cof_ranking over paper; run: '
                             'FrictionSim2D analyse run cof_ranking paper'):
        load_table(TableSource(analysis='cof_ranking', dataset='paper'))


def test_analysis_data_source_feeds_summary_rows(store, tmp_path):  # noqa: F811
    stored = run_analysis('period_summary', 'paper', {'period_x': {'h_MoS2': 3.16}})
    assert sorted(stored['force']) == [30.0, 50.0]
    assert 'dataset_label' not in stored

    plotter = Plotter(data_dirs=['aiida://analysis/period_summary/paper'
                                 '?period_x={"h_MoS2":3.16}'],
                      labels=['periods'], output_dir=str(tmp_path / 'plots'))
    frame = plotter._get_summary_data_df()  # pylint: disable=protected-access

    assert sorted(frame['force']) == [30.0, 50.0]
    assert set(frame['dataset_label']) == {'periods'}
    assert set(frame['id']) == {'h_MoS2'} and set(frame['size']) == {'100x100y'}
    assert list(frame.sort_values('force')['cof']) == list(
        stored.sort_values('force')['cof_periods'])
    plotter.generate_plot({'plot_type': 'summary', 'filename': 'cof.png',
                           'x_axis': 'force', 'y_axis': 'cof'})
    assert (tmp_path / 'plots' / 'cof.png').is_file()


def _probe_table(value):
    return pd.DataFrame({'material': ['A'], 'eligible': [True], 'value': [value]})


def test_table_source_selects_by_params(store, probe):  # noqa: F811
    store_result(probe, 'paper', {}, _probe_table(1.0), dropped=[])
    store_result(probe, 'paper', {'scale': 2.0}, _probe_table(2.0), dropped=[])   # newest

    default = load_table(TableSource(analysis='probe', dataset='paper'))
    scaled = load_table({'analysis': 'probe', 'dataset': 'paper', 'params': {'scale': 2}})
    assert list(default['value']) == [1.0]
    assert list(scaled['value']) == [2.0]


def test_table_source_refuses_when_only_other_params_are_stored(store, probe, tmp_path):  # noqa: F811,E501
    store_result(probe, 'paper', {'scale': 2.0}, _probe_table(2.0), dropped=[])
    with pytest.raises(PlotDataError, match=r'scale.*2\.0'):
        load_table(TableSource(analysis='probe', dataset='paper'))


def test_analysis_uri_carries_params(store, tmp_path):  # noqa: F811
    run_analysis('period_summary', 'paper', {'period_x': {'h_MoS2': 3.16}})

    with pytest.raises(PlotDataError, match='period_x'):     # the rows load at construction
        Plotter(data_dirs=['aiida://analysis/period_summary/paper'],
                labels=['periods'], output_dir=str(tmp_path / 'plots'))

    uri = 'aiida://analysis/period_summary/paper?period_x={"h_MoS2": 3.16}'
    plotter = Plotter(data_dirs=[uri], labels=['periods'], output_dir=str(tmp_path / 'plots'))
    frame = plotter._get_summary_data_df()  # pylint: disable=protected-access
    assert sorted(frame['force']) == [30.0, 50.0]

    with pytest.raises(PlotDataError, match='bogus'):
        Plotter(data_dirs=['aiida://analysis/period_summary/paper?bogus=1'],
                labels=['periods'], output_dir=str(tmp_path / 'plots'))
