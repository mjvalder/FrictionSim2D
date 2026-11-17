"""The plot stack reads the v2 export and refuses v1 (Q1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.postprocessing.plot_data import PlotDataError, Plotter
from src.postprocessing.read_data import DataReader
from tests.helpers.results_tree import make_family_run


def _dataset(tmp_path: Path, family: str = 'afm', **kw) -> Path:
    """A workspace directory with outputs/output_full_*.json + summary csv."""
    root = tmp_path / f'ds_{family}'
    make_family_run(root, family, **kw)
    reader = DataReader(results_dir=str(root))
    reader.export_full_data_to_json()
    reader.export_summary_csv()
    return root


def test_extract_all_runs_yields_one_dict_per_run(tmp_path):
    root = _dataset(tmp_path, 'afm', loads=(10.0, 30.0), angles=(0.0, 90.0),
                    pot='sw_a')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    runs = list(plotter._extract_all_runs('ds', '100x100y'))  # pylint: disable=protected-access

    assert len(runs) == 4
    one = runs[0]
    assert one['id'] == 'h_MoS2'
    assert one['family'] == 'afm'
    assert one['model'] == 'afm'
    assert one['potential'] == 'pot_sw_a'
    assert one['temperature'] == 300.0
    assert one['load_kind'] == 'force'
    assert one['force'] == one['load']
    assert one['pressure'] is None
    assert one['layer'] == 2
    assert one['speed'] == 2.0
    assert 'lf' in one['df'].columns
    assert 'time' in one['df'].columns


def test_a_sweep_arm_is_not_lost(tmp_path):
    """Two runs differing only in the potential collapsed to one leaf under
    v1; the v2 runs list cannot collapse."""
    root = tmp_path / 'ds'
    make_family_run(root, 'afm', pot='sw_a')
    make_family_run(root, 'afm', pot='sw_b')
    reader = DataReader(results_dir=str(root))
    reader.export_full_data_to_json()
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    runs = list(plotter._extract_all_runs('ds', '100x100y'))

    assert sorted(r['potential'] for r in runs) == ['pot_sw_a', 'pot_sw_b']


def test_a_pressure_run_reports_pressure_not_force(tmp_path):
    root = _dataset(tmp_path, 'sheetonsheet')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    run, = plotter._extract_all_runs('ds', '100x100y')

    assert run['load_kind'] == 'pressure'
    assert run['pressure'] == run['load']
    assert run['force'] is None


def test_a_dataset_directory_that_is_not_there_is_refused(tmp_path):
    """`_discover_data_files` used to log an error and carry on, so a typo in
    a data_dir produced an empty summary frame, an early return in
    `_generate_summary_plot` and a silent no-op -- a plot that never appeared
    and never said why. This is also what makes the worker's and the GUI's
    `failed` path reachable at all."""
    with pytest.raises(PlotDataError, match='no output_full'):
        Plotter(data_dirs=[str(tmp_path / 'typo')], labels=['ds'],
                output_dir=str(tmp_path / 'out'))


def test_an_empty_dataset_list_is_still_fine(tmp_path):
    """`Plotter(data_dirs=[], labels=[])` is how several unit tests reach the
    mixins' helpers; refusing an ABSENT directory must not refuse NO
    directories."""
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))

    assert plotter.full_data_files == {}


def test_a_db_uri_is_refused_naming_the_aiida_source(tmp_path):
    """`db://central` is gone with the central database; the error names the
    `aiida://dataset/<name>` source that replaces it (Task 12)."""
    with pytest.raises(PlotDataError, match='aiida://dataset/'):
        Plotter(data_dirs=['db://central?family=afm'], labels=['db'],
                output_dir=str(tmp_path / 'out'))


def test_a_v1_document_is_refused(tmp_path):
    outputs = tmp_path / 'legacy' / 'outputs'
    outputs.mkdir(parents=True)
    (outputs / 'output_full_100x100y.json').write_text(json.dumps({
        'metadata': {}, 'results': {'MoS2': {'f10.0': {'a0': {
            'columns': ['lfx', 'lfy', 'nf'], 'data': [[1.0, 0.0, 10.0]]}}}},
    }), encoding='utf-8')

    with pytest.raises(PlotDataError, match='schema'):
        Plotter(data_dirs=[str(tmp_path / 'legacy')], labels=['old'],
                output_dir=str(tmp_path / 'out'))


def test_the_refusal_says_how_to_fix_it(tmp_path):
    outputs = tmp_path / 'legacy' / 'outputs'
    outputs.mkdir(parents=True)
    (outputs / 'output_full_100x100y.json').write_text(
        json.dumps({'metadata': {}, 'results': {}}), encoding='utf-8')

    with pytest.raises(PlotDataError, match=r'postprocess read <root> --export'):
        Plotter(data_dirs=[str(tmp_path / 'legacy')], labels=['old'],
                output_dir=str(tmp_path / 'out'))


def test_metadata_merges_across_datasets(tmp_path):
    a = _dataset(tmp_path, 'afm', material='h-MoS2')
    b = tmp_path / 'ds_b'
    make_family_run(b, 'afm', material='h-WS2')
    DataReader(results_dir=str(b)).export_full_data_to_json()

    plotter = Plotter(data_dirs=[str(a), str(b)], labels=['a', 'b'],
                      output_dir=str(tmp_path / 'out'))

    assert sorted(plotter.metadata['materials']) == ['h_MoS2', 'h_WS2']
    assert plotter.material_type_map['h_MoS2'] == 'h_type'


def test_the_tier_field_selects_the_frame(tmp_path):
    root = _dataset(tmp_path, 'afm', tiers=('window', 'raw'))
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    window, = plotter._extract_all_runs('ds', '100x100y')
    raw, = plotter._extract_all_runs('ds', '100x100y', tier='raw')

    assert 'nf' in window['df'].columns
    assert 'raw_nf' in raw['df'].columns


def test_written_starts_empty_and_records_what_save_plot_writes(tmp_path):
    """Asserted through `_save_plot` directly, NOT through a whole summary
    plot: at this point in the plan the summary path still runs the old
    `df.mean()` `_calculate_summary_statistics`, and a test that quietly
    depended on that tolerating the new run records would be pinning the wrong
    thing. Task 19 rewrites that path and re-asserts `written` end to end."""
    import matplotlib.pyplot as plt

    root = _dataset(tmp_path, 'afm', loads=(10.0, 20.0, 30.0))
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))
    assert plotter.written == []

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plotter._save_plot(fig, 'lf_vs_load.png')  # pylint: disable=protected-access

    assert plotter.written == [out / 'lf_vs_load.png']


def test_summary_df_comes_from_the_summary_csv(tmp_path):
    """Three COF definitions used to reach one plot: df.mean() here,
    element-wise compute_derived_columns in the reader, ratio-of-means in the
    DB. There is one now, and it is summarise_run's."""
    import pandas as pd

    root = _dataset(tmp_path, 'afm', loads=(10.0, 30.0))
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    summary = plotter._get_summary_data_df()  # pylint: disable=protected-access

    csv = pd.read_csv(root / 'outputs' / 'summary_100x100y.csv')
    assert len(summary) == len(csv) == 2
    assert sorted(summary['cof']) == pytest.approx(sorted(csv['mean_cof']))
    assert sorted(summary['force']) == [10.0, 30.0]
    assert set(summary['dataset_label']) == {'ds'}
    assert set(summary['file_key']) == {'100x100y'}


def test_summary_rows_carry_the_identity_and_the_spread(tmp_path):
    import numpy as np

    root = _dataset(tmp_path, 'hetero', material='h-MoS2',
                    second_material='h-WS2', pot='sw_a')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    row = plotter._get_summary_data_df().iloc[0]

    assert row['id'] == 'h_MoS2+h_WS2'
    assert row['family'] == 'hetero'
    assert row['potential'] == 'pot_sw_a'
    assert row['size'] == '100x100y'
    assert np.isnan(row['force'])
    assert row['pressure'] == row['load']
    assert 'std_cof' in row.index
    assert row['nf'] == row['mean_nf']


def test_lf_as_was_is_absent_rather_than_aliased(tmp_path):
    """A DB or summary row has no v_xfrict-as-written column. Inventing one
    by aliasing mean_lf made `lf_source: as_was` silently plot `lf`."""
    root = _dataset(tmp_path, 'sheetonsheet')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    summary = plotter._get_summary_data_df()

    assert 'lf_as_was' not in summary.columns
    assert 'lf_sx_sy' in summary.columns


def test_a_window_argument_still_averages_the_frames(tmp_path):
    root = _dataset(tmp_path, 'afm', loads=(10.0, 30.0), n=20)
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    windowed = plotter._get_summary_data_df(first_n=5)

    assert len(windowed) == 2
    run = next(plotter._extract_all_runs('ds', '100x100y'))
    head = run['df'].iloc[:5]
    expected = float(head['lf'].mean()) / float(head['nf'].mean())
    row = windowed[windowed['force'] == run['force']].iloc[0]
    assert row['cof'] == pytest.approx(expected)


def test_written_records_every_saved_file_end_to_end(tmp_path):
    """Now that the summary path reads the CSV, a whole summary plot can be
    driven and `written` compared against what actually reached disk."""
    root = _dataset(tmp_path, 'afm', loads=(10.0, 20.0, 30.0))
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))

    plotter.generate_plot({'plot_type': 'summary', 'x_axis': 'force',
                           'y_axis': 'lf', 'filename': 'lf_vs_load.png'})

    assert plotter.written == [out / 'lf_vs_load.png']
    assert (out / 'lf_vs_load.png').is_file()


def test_a_missing_summary_csv_is_a_plot_data_error(tmp_path):
    root = tmp_path / 'ds'
    make_family_run(root, 'afm')
    DataReader(results_dir=str(root)).export_full_data_to_json()   # no csv

    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    with pytest.raises(PlotDataError, match='summary_100x100y.csv'):
        plotter._get_summary_data_df()


def test_timeseries_uses_each_run_own_time_column(tmp_path):
    """The metadata `time_series` was one global list: two runs of different
    lengths were plotted against the same x, and the shorter one was either
    truncated or raised."""
    root = tmp_path / 'ds'
    make_family_run(root, 'afm', material='h-MoS2', n=20)
    make_family_run(root, 'afm', material='h-WS2', n=18)
    reader = DataReader(results_dir=str(root))
    reader.export_full_data_to_json()
    reader.export_summary_csv()
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))

    plotter.generate_plot({'plot_type': 'timeseries', 'y_axis': 'lf',
                           'filter_size': '100x100y', 'datasets': ['ds'],
                           'filename': 'ts.png'})

    assert (out / 'ts.png').exists()
    lengths = {len(r['df']) for r in plotter._extract_all_runs('ds', '100x100y')}
    assert lengths == {20, 18}


def test_timeseries_can_plot_the_raw_tier(tmp_path):
    root = _dataset(tmp_path, 'afm', tiers=('window', 'raw'))
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))

    plotter.generate_plot({'plot_type': 'timeseries', 'tier': 'raw',
                           'y_axis': 'raw_nf', 'filter_size': '100x100y',
                           'datasets': ['ds'], 'filename': 'raw.png'})

    assert (out / 'raw.png').exists()


def test_a_per_layer_channel_is_plottable_by_name(tmp_path):
    """Q7: per-layer, energy, temperature and contact channels have no
    special plot type -- they are columns, and now they are named columns."""
    root = tmp_path / 'ds'
    run_dir = make_family_run(root, 'sheetonsheet')
    natives = ['v_fz', 'v_xfrict', 'v_yfrict', 'v_comz_L2', 'v_etot']
    path = next((run_dir / 'results').iterdir())
    path.write_text(
        '# schema=2 lf_units=nN family=sheetonsheet\n'
        '# TimeStep ' + ' '.join(natives) + '\n'
        # Every force channel varies: a constant one is `frozen` to stage 3.
        + '\n'.join(f'{(i + 1) * 1000} {1.0 + i * 0.01:g} {2.0 + i * 0.01:g} '
                    f'{i * 0.01:g} {6.5 + i * 0.1:g} -12.0'
                    for i in range(10)) + '\n', encoding='utf-8')
    reader = DataReader(results_dir=str(root))
    reader.export_full_data_to_json()
    reader.export_summary_csv()
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))

    plotter.generate_plot({'plot_type': 'timeseries', 'y_axis': 'comz_L2',
                           'filter_size': '100x100y', 'datasets': ['ds'],
                           'filename': 'comz.png'})

    assert (out / 'comz.png').exists()
    run = next(plotter._extract_all_runs('ds', '100x100y'))
    assert 'comz_L2' in run['df'].columns
    assert 'etotal' in run['df'].columns


def test_stick_slip_reads_the_run_time_column(tmp_path):
    root = _dataset(tmp_path, 'afm', n=64)
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    loaded = plotter._load_stick_slip_run(  # pylint: disable=protected-access
        {'plot_type': 'stick_slip_analysis', 'filter_size': '100x100y',
         'datasets': ['ds'], 'filter_layer': 2})

    assert loaded is not None
    assert len(loaded['t']) == len(loaded['run']['df'])
    assert set(loaded) >= {'run', 't', 'x', 'label', 'file_key',
                           'displacement_col', 'time_scale',
                           'displacement_scale', 'time_unit', 'disp_unit'}


def test_axis_labels_default_to_the_registry(tmp_path):
    import matplotlib.pyplot as plt

    root = _dataset(tmp_path, 'afm', loads=(10.0, 30.0))
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))
    fig, ax = plotter.new_figure()
    try:
        plotter._finalize_plot(ax, {'plot_type': 'summary'}, 'force', 'lf')  # pylint: disable=protected-access

        assert ax.get_xlabel() == 'Normal Load (nN)'
        assert ax.get_ylabel() == 'Lateral Force (nN)'
    finally:
        plt.close(fig)


def test_an_explicit_label_still_wins(tmp_path):
    import matplotlib.pyplot as plt

    root = _dataset(tmp_path, 'afm')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))
    fig, ax = plotter.new_figure()
    try:
        plotter._finalize_plot(ax, {'plot_type': 'summary',
                                    'x_label': 'Load, F$_N$ (nN)'},
                               'force', 'lf')

        assert ax.get_xlabel() == 'Load, F$_N$ (nN)'
    finally:
        plt.close(fig)


def test_the_family_selects_the_label(tmp_path):
    """`lfx` is a different registry entry for AFM and the sheet family.
    Both are nN; the lookup must use the run's own family so a future
    divergence shows up on the axis instead of being wrong in silence."""
    import matplotlib.pyplot as plt

    root = _dataset(tmp_path, 'flake')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))
    list(plotter._extract_all_runs('ds', '100x100y'))

    assert plotter.plot_family == 'flake'
    fig, ax = plotter.new_figure()
    try:
        plotter._finalize_plot(ax, {'plot_type': 'summary'}, 'nf', 'lfx')
        assert ax.get_ylabel() == 'Lateral Force x (nN)'
    finally:
        plt.close(fig)


def test_the_histogram_says_slope_not_cof(tmp_path):
    """A bar chart of `fit['slope']` labelled 'COF' put the regression slope
    and mean(LF)/mean(NF) on the same axis under one name."""
    from src.postprocessing.labels import label_for

    root = _dataset(tmp_path, 'afm', loads=(10.0, 20.0, 30.0))
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))

    plotter.generate_plot({'plot_type': 'cof_histogram', 'x_axis': 'force',
                           'y_axis': 'lf', 'filter_size': '100x100y',
                           'datasets': ['ds'], 'filename': 'hist.png'})

    assert (out / 'hist.png').exists()
    assert label_for('cof_slope') == 'COF (slope of LF vs NF)'


def test_dataset_rows_carry_the_v3_identity_and_no_lf_aliases():
    """`dataset_rows` (a `Friction2DDB.query`-shaped row) to a summary row:
    the CSV path's names, no lateral-force variant aliased from `mean_lf`."""
    from src.postprocessing.plot_common import PlotCommonMixin

    record = {
        'materials': ['h-MoS2'], 'flake_material': None, 'family': 'afm',
        'model': 'afm', 'set': '260910-afm', 'pot_token': 'pot_sw_h-MoS2',
        'layers': 2, 'size_x': 100.0, 'size_y': 100.0, 'temperature': 300.0,
        'tip_material': 'Si', 'tip_radius': 25.0, 'load_kind': 'force',
        'load': 30.0, 'angle': 0.0, 'speed': 2.0, 'load_unit': 'nN',
        'mean_cof': 0.1, 'std_cof': 0.01, 'mean_lf': 3.0, 'std_lf': 0.2,
        'mean_nf': 30.0, 'std_nf': 0.5, 'mean_lfx': 3.0, 'std_lfx': 0.2,
        'mean_lfy': 0.0, 'std_lfy': 0.1, 'ntimesteps': 500,
    }

    row, = PlotCommonMixin._dataset_to_summary_rows('ds', [record])  # pylint: disable=protected-access

    assert row['dataset_label'] == 'ds'
    assert row['id'] == row['material'] == 'h_MoS2'
    assert row['family'] == 'afm'
    assert row['potential'] == 'pot_sw_h-MoS2'
    assert row['temperature'] == 300.0
    assert row['file_key'] == row['size'] == '100x100y'
    assert row['force'] == 30.0
    assert row['pressure'] != row['pressure'], 'NaN: a force run has no pressure'
    assert row['layer'] == 2
    assert row['cof'] == 0.1
    assert row['lf'] == 3.0
    assert row['load_unit'] == 'nN'
    for absent in ('lf_as_was', 'lf_sx_sy', 'lf_fx_fy'):
        assert absent not in row, f'{absent} must not be aliased from mean_lf'


def test_lf_source_on_a_db_row_falls_back_loudly(tmp_path, caplog):
    import pandas as pd

    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    df = pd.DataFrame([{'lf': 3.0, 'nf': 30.0}])

    with caplog.at_level('WARNING'):
        col = plotter._resolve_lf_axis(df, 'lf', {'lf_source': 'sx_sy'})  # pylint: disable=protected-access

    assert col == 'lf'
    assert 'lf_sx_sy' in caplog.text


def test_both_ranking_modes_write_the_same_shape(tmp_path):
    """`aggregate_over_force_range` wrote {ranking_mode, rank_order,
    average_f10_to_f50}; the per-force mode wrote a bare {f30.0: [...]}.
    `correlation` read every top-level key as a force and crashed on
    float('anking_mode')."""
    from src.postprocessing.plot_ranking import read_ranking

    root = _dataset(tmp_path, 'afm', loads=(10.0, 30.0, 50.0))
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))

    plotter.generate_plot({'plot_type': 'rank_friction', 'rank_by': 'cof',
                           'filter_size': '100x100y',
                           'filename': 'friction_ranking_per_force.json'})
    plotter.generate_plot({'plot_type': 'rank_friction', 'rank_by': 'cof',
                           'filter_size': '100x100y',
                           'aggregate_over_force_range': True,
                           'force_range': [10.0, 50.0],
                           'filename': 'friction_ranking_avg.json'})

    per_force = read_ranking(out / 'friction_ranking_per_force.json')
    aggregated = read_ranking(out / 'friction_ranking_avg.json')

    assert set(per_force) == set(aggregated)
    assert per_force['schema'] == aggregated['schema'] == 1
    assert per_force['ranking_mode'] == 'per_force'
    assert aggregated['ranking_mode'] == 'average_over_force_range'
    assert per_force['metric'] == aggregated['metric'] == 'cof'
    assert len(aggregated['groups']) == 1
    assert aggregated['groups'][0]['key'] == 'average_f10_to_f50'
    assert aggregated['groups'][0]['force'] is None
    assert {g['force'] for g in per_force['groups']} == {10.0, 30.0, 50.0}


def test_every_material_entry_has_every_key(tmp_path):
    from src.postprocessing.plot_ranking import read_ranking

    root = _dataset(tmp_path, 'afm', loads=(10.0, 30.0))
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))
    plotter.generate_plot({'plot_type': 'rank_friction', 'rank_by': 'cof',
                           'filter_size': '100x100y',
                           'filename': 'r.json'})

    entry = read_ranking(out / 'r.json')['groups'][0]['materials'][0]

    assert set(entry) == {'material', 'rank', 'metric', 'mean_value',
                          'n_points', 'min_force', 'max_force',
                          'fit_slope_stderr', 'fit_r_squared', 'fit_rmse'}


def test_correlation_reads_an_aggregated_ranking(tmp_path):
    root = _dataset(tmp_path, 'afm', loads=(10.0, 30.0, 50.0))
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))
    plotter.generate_plot({'plot_type': 'rank_friction', 'rank_by': 'cof',
                           'filter_size': '100x100y',
                           'aggregate_over_force_range': True,
                           'filename': 'friction_ranking_100x100y.json'})

    plotter.generate_plot({'plot_type': 'correlation', 'correlate_by': 'force',
                           'filename': 'corr.png'})

    assert not any('anking_mode' in str(w) for w in plotter.written)


def test_correlation_ignores_a_file_that_is_not_a_ranking(tmp_path):
    out = tmp_path / 'out'
    out.mkdir(parents=True)
    (out / 'friction_ranking_bogus.json').write_text('{"nonsense": 1}',
                                                     encoding='utf-8')
    root = _dataset(tmp_path, 'afm')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))

    plotter.generate_plot({'plot_type': 'correlation', 'correlate_by': 'force',
                           'filename': 'corr.png'})

    assert not (out / 'corr.png').exists()   # nothing to correlate, no crash


def test_generate_plot_accepts_a_model(tmp_path):
    from src.postprocessing.plot_config import SummaryPlot

    root = _dataset(tmp_path, 'afm', loads=(10.0, 30.0))
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))

    plotter.generate_plot(SummaryPlot(x_axis='force', y_axis='lf',
                                      filename='model.png',
                                      filter_size='100x100y'))

    assert (out / 'model.png').exists()


def test_only_the_keys_that_were_set_reach_the_mixin(tmp_path, monkeypatch):
    """`spec.model_dump(exclude_unset=True)`: the mixins' own
    `.get(key, default)` defaults stay authoritative, and the Pydantic
    defaults document them. Dumping defaults would change every plot."""
    from src.postprocessing.plot_config import SummaryPlot

    root = _dataset(tmp_path, 'afm')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))
    seen = {}
    monkeypatch.setattr(type(plotter), '_generate_summary_plot',
                        lambda self, cfg: seen.update(cfg))

    plotter.generate_plot(SummaryPlot(filename='x.png', x_axis='force'))

    assert seen == {'plot_type': 'summary', 'filename': 'x.png',
                    'x_axis': 'force'}
    assert 'legend_outside' not in seen
    assert 'tier' not in seen


def test_a_dict_still_works(tmp_path, monkeypatch):
    root = _dataset(tmp_path, 'afm')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))
    seen = {}
    monkeypatch.setattr(type(plotter), '_generate_summary_plot',
                        lambda self, cfg: seen.update(cfg))

    plotter.generate_plot({'plot_type': 'summary', 'filename': 'x.png'})

    assert seen == {'plot_type': 'summary', 'filename': 'x.png'}


def test_an_unknown_plot_type_raises(tmp_path):
    root = _dataset(tmp_path, 'afm')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    with pytest.raises(PlotDataError, match='sunburst'):
        plotter.generate_plot({'plot_type': 'sunburst', 'filename': 'x.png'})


def test_the_taxonomy_types_are_gone(tmp_path):
    """The ML anomaly-mode plots left with the taxonomy: they are unknown types."""
    root = _dataset(tmp_path, 'afm')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    with pytest.raises(PlotDataError, match='taxonomy_prevalence'):
        plotter.generate_plot({'plot_type': 'taxonomy_prevalence', 'filename': 'x.png'})


def test_plot_stack_reads_schema_3_and_still_reads_schema_2(tmp_path):
    root = _dataset(tmp_path, 'afm')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))
    assert len(list(plotter._extract_all_runs('ds', '100x100y'))) == 1  # pylint: disable=protected-access

    golden = (Path(__file__).resolve().parents[1] / 'fixtures' / 'golden'
              / 'output_full_v2_afm.json')
    assert json.loads(golden.read_text(encoding='utf-8'))['schema'] == 2
    old = tmp_path / 'ds_old'
    (old / 'outputs').mkdir(parents=True)
    (old / 'outputs' / 'output_full_100x100y.json').write_text(
        golden.read_text(encoding='utf-8'), encoding='utf-8')
    (old / 'outputs' / 'summary_100x100y.csv').write_text(
        (root / 'outputs' / 'summary_100x100y.csv').read_text(
            encoding='utf-8'), encoding='utf-8')
    plotter = Plotter(data_dirs=[str(old)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    assert list(plotter._extract_all_runs('ds', '100x100y'))  # pylint: disable=protected-access


# --- status filter ---------------------------------------------------------

def _cut_short_dataset(tmp_path: Path, name: str, **reader_kw) -> Path:
    from tests.postprocessing.test_run_status import _cut_short_tree
    root = tmp_path / name
    _cut_short_tree(root)
    reader = DataReader(results_dir=str(root), **reader_kw)
    reader.export_full_data_to_json()
    reader.export_summary_csv()
    return root


def test_truncated_runs_are_hidden_by_default(tmp_path):
    root = _cut_short_dataset(tmp_path, 'ds')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    runs = list(plotter._extract_all_runs('ds', '100x100y'))  # pylint: disable=protected-access

    assert sorted(r['load'] for r in runs) == [10.0, 20.0]


def test_no_output_entries_do_not_break_any_plot_reader(tmp_path):
    results = {}
    for name, kw in (('full', {}), ('only', {'complete_only': True})):
        root = _cut_short_dataset(tmp_path, name, **kw)
        out = tmp_path / f'out_{name}'
        plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                          output_dir=str(out))
        window = list(plotter._extract_all_runs('ds', '100x100y'))  # pylint: disable=protected-access
        raw = list(plotter._extract_all_runs('ds', '100x100y', 'raw'))  # pylint: disable=protected-access
        summary = plotter._calculate_summary_statistics()  # pylint: disable=protected-access
        windowed = plotter._calculate_summary_statistics(first_n=10)  # pylint: disable=protected-access
        plotter.generate_plot({'plot_type': 'timeseries', 'y_axis': 'lf',
                               'filter_size': '100x100y', 'datasets': ['ds'],
                               'filename': 'ts.png'})
        plotter.generate_plot({'plot_type': 'stick_slip_analysis',
                               'filter_size': '100x100y', 'datasets': ['ds'],
                               'filter_layer': 2, 'filename': 'ss.png'})
        assert (out / 'ts.png').exists()
        results[name] = (
            [(r['rel_path'], r['load']) for r in window], len(raw),
            summary.sort_values('force')[['force', 'lf']].to_dict('records'),
            windowed.sort_values('force')[['force', 'lf']].to_dict('records'))

    assert results['full'] == results['only']
    assert [load for _, load in results['full'][0]] == [10.0, 20.0]


def test_a_hand_built_no_output_record_yields_nothing(tmp_path):
    root = _dataset(tmp_path, 'afm', loads=(10.0,))
    path = root / 'outputs' / 'output_full_100x100y.json'
    doc = json.loads(path.read_text())
    doc['runs'].append({**doc['runs'][0], 'status': 'no_output', 'tiers': {}})
    path.write_text(json.dumps(doc))
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    runs = list(plotter._extract_all_runs('ds', '100x100y'))  # pylint: disable=protected-access

    assert len(runs) == 1
