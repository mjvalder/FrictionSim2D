"""The four analysis-figure plot types: table sources, histogram, scatter, profile, panel."""

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.postprocessing.plot_config import (ContactCofScatterPlot,  # noqa: E402
                                            StickslipPeriodHistogramPlot, TableSource)
from src.postprocessing.plot_data import Plotter, PlotDataError  # noqa: E402
from src.postprocessing.plot_tables import load_table  # noqa: E402


def _ranking(tmp_path):
    path = tmp_path / 'ranking.csv'
    pd.DataFrame({'material': ['A', 'B'], 'eligible': [True, False],
                  'cof_10nN': [0.02, 0.5], 'cof_20nN': [0.03, 0.6]}).to_csv(path, index=False)
    return path


def _stickslip(tmp_path):
    path = tmp_path / 'ss.csv'
    pd.DataFrame({'material': ['A', 'A', 'B'], 'load_nN': [10, 20, 10],
                  'lf_period_A': [3.0, 3.3, 3.0], 'a_x_A': [3.0, 3.0, 3.0],
                  'lf_period_clear': [True, True, True]}).to_csv(path, index=False)
    return path


def test_table_source_accepts_a_bare_path():
    assert TableSource.model_validate('x.csv').file.name == 'x.csv'
    assert StickslipPeriodHistogramPlot(
        stickslip_file='a.csv', ranking_file='b.csv').stickslip_file.file.name == 'a.csv'


def test_missing_column_names_it(tmp_path):
    path = tmp_path / 't.csv'
    pd.DataFrame({'material': ['A']}).to_csv(path, index=False)
    with pytest.raises(PlotDataError, match=r"t\.csv: missing column\(s\) \['load_nN'\]"):
        load_table(TableSource(file=path), tmp_path, required=['material', 'load_nN'])


def test_eligible_filter_uses_ranking(tmp_path):
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    eligible = plotter._eligible_materials(TableSource(file=_ranking(tmp_path)))
    assert list(eligible.index) == ['A']


def test_period_histogram_renders(tmp_path, monkeypatch):
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    counts = []
    import matplotlib.axes  # pylint: disable=import-outside-toplevel
    real_hist = matplotlib.axes.Axes.hist

    def spy(ax, x, *args, **kwargs):
        counts.append(list(x))
        return real_hist(ax, x, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'hist', spy)
    plotter.generate_plot(StickslipPeriodHistogramPlot(
        filename='p.png', stickslip_file=str(_stickslip(tmp_path)),
        ranking_file=str(_ranking(tmp_path)), loads=[10, 20]))
    assert (tmp_path / 'p.png').is_file()
    assert counts == [[1.0], [pytest.approx(1.1)]]   # B is not eligible


def test_contact_scatter_takes_cof_at_each_load(tmp_path, monkeypatch):
    contact = tmp_path / 'c.csv'
    pd.DataFrame({'material': ['A', 'A', 'A'], 'load_nN': [10, 20, 10],
                  'state': ['load', 'load', 'final'],
                  'contact_area': [100.0, 150.0, 999.0]}).to_csv(contact, index=False)
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    points = []
    import matplotlib.axes  # pylint: disable=import-outside-toplevel
    real_scatter = matplotlib.axes.Axes.scatter

    def spy(ax, x, y, *args, **kwargs):
        points.append((list(x), list(y)))
        return real_scatter(ax, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'scatter', spy)
    plotter.generate_plot(ContactCofScatterPlot(
        filename='c.png', contact_file=str(contact), ranking_file=str(_ranking(tmp_path)),
        x_column='contact_area', loads=[10, 20]))
    assert points == [([100.0], [0.02]), ([150.0], [0.03])]


def test_period_histogram_reads_an_analysis_table_by_force_and_layer(tmp_path, monkeypatch):
    """`stick_slip` names the load `force` and holds every layer (Task 9)."""
    ss = tmp_path / 'ss_analysis.csv'
    pd.DataFrame({'material': ['A', 'A', 'A'], 'force': [10.0, 20.0, 10.0],
                  'layer': [2, 2, 1], 'lf_period_A': [3.0, 3.3, 6.0],
                  'a_x_A': [3.0, 3.0, 3.0],
                  'lf_period_clear': [True, True, True]}).to_csv(ss, index=False)
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    counts = []
    import matplotlib.axes  # pylint: disable=import-outside-toplevel
    real_hist = matplotlib.axes.Axes.hist

    def spy(ax, x, *args, **kwargs):
        counts.append(list(x))
        return real_hist(ax, x, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'hist', spy)
    plotter.generate_plot(StickslipPeriodHistogramPlot(
        filename='p.png', stickslip_file=str(ss), ranking_file=str(_ranking(tmp_path)),
        loads=[10, 20], filter_layer=2))
    assert counts == [[1.0], [pytest.approx(1.1)]]   # the layer-1 row is filtered out


def test_contact_scatter_reads_an_analysis_table_by_force_and_layer(tmp_path, monkeypatch):
    contact = tmp_path / 'c_analysis.csv'
    pd.DataFrame({'material': ['A', 'A', 'A'], 'force': [10.0, 20.0, 10.0],
                  'layer': [2, 2, 1], 'state': ['load'] * 3,
                  'contact_area': [100.0, 150.0, 999.0]}).to_csv(contact, index=False)
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    points = []
    import matplotlib.axes  # pylint: disable=import-outside-toplevel
    real_scatter = matplotlib.axes.Axes.scatter

    def spy(ax, x, y, *args, **kwargs):
        points.append((list(x), list(y)))
        return real_scatter(ax, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'scatter', spy)
    plotter.generate_plot(ContactCofScatterPlot(
        filename='c.png', contact_file=str(contact), ranking_file=str(_ranking(tmp_path)),
        x_column='contact_area', loads=[10, 20], filter_layer=2))
    assert points == [([100.0], [0.02]), ([150.0], [0.03])]


def test_a_table_without_load_or_layer_names_them(tmp_path):
    bare = tmp_path / 'bare.csv'
    pd.DataFrame({'material': ['A'], 'lf_period_A': [3.0], 'a_x_A': [3.0],
                  'lf_period_clear': [True]}).to_csv(bare, index=False)
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    with pytest.raises(PlotDataError, match=r"bare\.csv: missing column\(s\) \['load_nN'\] \(or 'force'\)"):
        plotter.generate_plot(StickslipPeriodHistogramPlot(
            filename='p.png', stickslip_file=str(bare), ranking_file=str(_ranking(tmp_path))))
    with pytest.raises(PlotDataError, match=r"ss\.csv: filter_layer needs a 'layer' column"):
        plotter.generate_plot(StickslipPeriodHistogramPlot(
            filename='p.png', stickslip_file=str(_stickslip(tmp_path)),
            ranking_file=str(_ranking(tmp_path)), filter_layer=2))


def _write_data(path, types, xyz, box=40.0):
    lines = ['LAMMPS data', '', f'{len(types)} atoms', '',
             f'0 {box} xlo xhi', f'0 {box} ylo yhi', '-20 60 zlo zhi', '', 'Atoms # atomic', '']
    lines += [f'{i + 1} {t} {x} {y} {z}' for i, (t, (x, y, z)) in enumerate(zip(types, xyz))]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _run_dir(tmp_path):
    run = tmp_path / 'run'
    (run / 'L2' / 'lammps').mkdir(parents=True)
    (run / 'L2' / 'data').mkdir()
    (run / 'L2' / 'lammps' / 'system.in.settings').write_text(
        'group layer_1 type 1\ngroup layer_2 type 2\ngroup tip_all type 4\n', encoding='utf-8')
    g = np.arange(0, 40, 1.0)
    xx, yy = (a.ravel() for a in np.meshgrid(g, g))
    sheet = [(1, (x, y, 0.0)) for x, y in zip(xx, yy)] + [(2, (x, y, 3.0)) for x, y in zip(xx, yy)]
    tip = [(4, (20.0 + dx, 20.0 + dy, 6.0 + dz)) for dx in (-1, 0, 1)
           for dy in (-1, 0, 1) for dz in (0, 1)]
    types, xyz = zip(*(sheet + tip))
    for name in ('load_10N.data', 'final_f10N_a0.data'):
        _write_data(run / 'L2' / 'data' / name, types, xyz)
    return run


def test_sheet_profile_renders(tmp_path):
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    plotter.generate_plot({'plot_type': 'sheet_profile', 'filename': 's.png',
                           'run_dir': str(_run_dir(tmp_path)), 'loads': [10],
                           'tip_types': [4], 'far_field_radius': 10})
    assert (tmp_path / 's.png').is_file()


def test_integrity_panel_renders(tmp_path):
    traces = tmp_path / 'traces.csv'
    pd.DataFrame({'run_id': ['M/l2/f10/a0'] * 3, 'step': [0, 1, 2],
                  'temp_sheet_3d': [300.0, 301.0, 299.0], 'comz_L1': [0.0] * 3,
                  'comz_L2': [3.0, 3.1, 3.0]}).to_csv(traces, index=False)
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    plotter.generate_plot({'plot_type': 'integrity_panel', 'filename': 'i.png',
                           'material': 'M', 'run_dir': str(_run_dir(tmp_path)),
                           'loads': [10, 20], 'traces_file': str(traces)})
    assert (tmp_path / 'i.png').is_file()


def test_a_plot_without_its_table_names_the_field(tmp_path):
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    with pytest.raises(PlotDataError, match='contact_cof_scatter needs ranking_file'):
        plotter.generate_plot({'plot_type': 'contact_cof_scatter', 'filename': 'c.png',
                               'contact_file': 'c.csv', 'x_column': 'contact_area'})


# --- analysis results as sources (analysis-layer Task 5) -----------------------

@pytest.fixture
def no_profile(monkeypatch):
    import src.aiida as aiida_pkg  # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)


@pytest.fixture
def stored(monkeypatch, no_profile):
    """`find_result` over a dict of ``(name, dataset) -> table``."""
    from src.analysis import store  # pylint: disable=import-outside-toplevel
    tables, calls = {}, []

    def find(name, dataset, params_hash=None, version=None):
        calls.append((name, dataset, version))
        table = tables.get((name, dataset))
        return None if table is None else table.copy()

    monkeypatch.setattr(store, 'find_result', find)
    tables['calls'] = calls
    return tables


def test_table_source_takes_a_file_or_an_analysis():
    source = TableSource(analysis='cof_ranking', dataset='paper')
    assert source.file is None and source.analysis == 'cof_ranking'
    assert source.model_dump() == {'file': None, 'analysis': 'cof_ranking',
                                   'dataset': 'paper', 'version': None}
    assert TableSource.model_validate(source.model_dump()) == source
    assert TableSource(file='x.csv').model_dump() == 'x.csv', 'a file source stays a bare path'
    with pytest.raises(ValueError, match='exactly one of file or analysis'):
        TableSource(file='x.csv', analysis='cof_ranking', dataset='paper')
    with pytest.raises(ValueError, match='exactly one of file or analysis'):
        TableSource()
    with pytest.raises(ValueError, match='needs dataset'):
        TableSource(analysis='cof_ranking')
    with pytest.raises(ValueError, match='only with an analysis'):
        TableSource(file='x.csv', dataset='paper')


def test_load_table_reads_a_stored_analysis(tmp_path, stored):
    stored[('cof_ranking', 'paper')] = pd.DataFrame(
        {'material': ['A', 'B'], 'eligible': [True, False]})
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    eligible = plotter._eligible_materials(
        {'analysis': 'cof_ranking', 'dataset': 'paper', 'version': '3'})
    assert list(eligible.index) == ['A']
    assert stored['calls'] == [('cof_ranking', 'paper', '3')]
    with pytest.raises(PlotDataError, match=r"cof_ranking over paper: missing column\(s\) \['x'\]"):
        load_table(TableSource(analysis='cof_ranking', dataset='paper'), tmp_path, ['x'])


def test_missing_analysis_names_the_command_fast(tmp_path, stored):
    with pytest.raises(PlotDataError,
                       match='FrictionSim2D analyse run cof_ranking paper'):
        load_table(TableSource(analysis='cof_ranking', dataset='paper'), tmp_path)


def _period_table():
    """`period_summary` rows as the analysis stores them (no dataset_label)."""
    return pd.DataFrame({
        'run_uuid': ['r1', 'r1', 'r2'], 'point_uuid': ['p1', 'p2', 'p3'],
        'id': ['h_MoS2'] * 3, 'material': ['h_MoS2'] * 3, 'size': ['100x100y'] * 3,
        'layer': [2, 2, 2], 'load': [10.0, 30.0, 1.0],
        'load_kind': ['force', 'force', 'pressure'],
        'force': [10.0, 30.0, float('nan')], 'pressure': [float('nan'), float('nan'), 1.0],
        'angle': [0.0] * 3, 'speed': [2.0] * 3,
        'cof_periods': [0.1, 0.2, 0.3], 'cof': [0.1, 0.2, 0.3],
        'lf': [1.0, 6.0, 0.3], 'nf': [10.0, 30.0, 1.0], 'lf_sx_sy': [1.1, 6.1, 0.4]})


def test_analysis_data_source_feeds_summary_rows_fast(tmp_path, stored):
    stored[('period_summary', 'sheets')] = _period_table()
    plotter = Plotter(data_dirs=['aiida://analysis/period_summary/sheets'],
                      labels=['periods'], output_dir=str(tmp_path))

    frame = plotter._get_summary_data_df()

    assert set(frame['dataset_label']) == {'periods'}
    assert set(frame['file_key']) == {'100x100y'}
    assert list(frame['cof']) == [0.1, 0.2, 0.3]
    assert list(frame['force'][:2]) == [10.0, 30.0]
    assert list(frame['pressure'][2:]) == [1.0]
    assert (frame['status'] == 'complete').all()
    assert plotter._resolve_lf_axis(frame, 'lf', {'lf_source': 'sx_sy'}) == 'lf_sx_sy'
    assert plotter.full_data_files['periods'] == {}, 'no file discovery for the URI'


def test_missing_analysis_data_source_names_the_command(tmp_path, stored):
    with pytest.raises(PlotDataError,
                       match='FrictionSim2D analyse run period_summary sheets'):
        Plotter(data_dirs=['aiida://analysis/period_summary/sheets'], labels=['p'],
                output_dir=str(tmp_path))
    with pytest.raises(PlotDataError, match='aiida://analysis/<name>/<dataset>'):
        Plotter(data_dirs=['aiida://analysis/period_summary'], labels=['p'],
                output_dir=str(tmp_path))


class _DirFetch:
    """A fetcher reading a `_run_dir` tree, the way `StoreFetcher` reads the store."""

    def __init__(self, run):
        self.run = run
        self.asked = []

    def snapshot(self, row, kind):
        self.asked.append((kind, row.load))
        name = (f"load_{row.load:g}N.data" if kind == 'load'
                else f"final_f{row.load:g}N_a0.data")
        path = self.run / 'L2' / 'data' / name
        return path if path.exists() else None

    def settings_text(self, row):  # pylint: disable=unused-argument
        return (self.run / 'L2' / 'lammps' / 'system.in.settings').read_text(encoding='utf-8')


@pytest.fixture
def dataset_source(tmp_path, monkeypatch, no_profile):
    """`RunSource.from_dataset('paper')` over the `_run_dir` tree: M at L2,
    10 nN passing and 20 nN failed; another material and layer as decoys."""
    from src.analysis.source import RunSource  # pylint: disable=import-outside-toplevel
    fetch = _DirFetch(_run_dir(tmp_path))
    rows = pd.DataFrame({
        'family': ['afm'] * 4, 'run_uuid': ['r', 'r', 'other', 'r1'],
        'point_uuid': ['p10', 'p20', 'q', 'l1'], 'calc_uuid': ['c'] * 4,
        'materials': [['M'], ['M'], ['N'], ['M']], 'layers': [2, 2, 2, 1],
        'load': [10.0, 20.0, 10.0, 10.0], 'angle': [0.0] * 4,
        'verdict_status': ['complete', 'failed_integrity', 'complete', 'complete']})
    opened = []

    def from_dataset(name):
        opened.append(name)
        return RunSource(rows, None, fetch)

    monkeypatch.setattr(RunSource, 'from_dataset', staticmethod(from_dataset))
    fetch.opened = opened
    return fetch


def test_sheet_profile_reads_a_dataset(tmp_path, dataset_source):
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    plotter.generate_plot({'plot_type': 'sheet_profile', 'filename': 's.png',
                           'dataset': 'paper', 'material': 'M', 'layer': 2, 'loads': [10],
                           'tip_types': [4], 'far_field_radius': 10})
    assert (tmp_path / 's.png').is_file()
    assert dataset_source.opened == ['paper']
    assert dataset_source.asked == [('load', 10.0)]


def test_integrity_panel_reads_a_dataset_failed_loads_included(tmp_path, dataset_source):
    traces = tmp_path / 'traces.csv'
    pd.DataFrame({'run_id': ['M/l2/f10/a0'] * 3, 'step': [0, 1, 2],
                  'temp_sheet_3d': [300.0, 301.0, 299.0], 'comz_L1': [0.0] * 3,
                  'comz_L2': [3.0, 3.1, 3.0]}).to_csv(traces, index=False)
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    plotter.generate_plot({'plot_type': 'integrity_panel', 'filename': 'i.png',
                           'material': 'M', 'dataset': 'paper', 'layer': 2,
                           'loads': [10, 20, 30], 'traces_file': str(traces)})
    assert (tmp_path / 'i.png').is_file()
    assert dataset_source.asked == [('final', 10.0), ('final', 20.0)], \
        'the failed 20 nN point is drawn; 30 nN has no point'


def test_a_dataset_panel_without_the_material_names_it(tmp_path, dataset_source):
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    with pytest.raises(PlotDataError, match='sheet_profile needs material'):
        plotter.generate_plot({'plot_type': 'sheet_profile', 'filename': 's.png',
                               'dataset': 'paper', 'loads': [10]})
    with pytest.raises(PlotDataError, match='needs run_dir or dataset'):
        plotter.generate_plot({'plot_type': 'sheet_profile', 'filename': 's.png',
                               'loads': [10]})
    with pytest.raises(PlotDataError, match="no point of Q at layer 2 in dataset 'paper'"):
        plotter.generate_plot({'plot_type': 'sheet_profile', 'filename': 's.png',
                               'dataset': 'paper', 'material': 'Q', 'loads': [10]})


def _panel_titles(tmp_path, monkeypatch, **extra):
    import matplotlib.axes  # pylint: disable=import-outside-toplevel
    titles = []
    original = matplotlib.axes.Axes.set_title
    monkeypatch.setattr(matplotlib.axes.Axes, 'set_title',
                        lambda self, label, *a, **k: (titles.append(label),
                                                      original(self, label, *a, **k))[1])
    traces = tmp_path / 'traces.csv'
    pd.DataFrame({'run_id': ['M/l2/f10/a0'], 'step': [0], 'temp_sheet_3d': [300.0],
                  'comz_L1': [0.0], 'comz_L2': [3.0]}).to_csv(traces, index=False)
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    plotter.generate_plot({'plot_type': 'integrity_panel', 'filename': 'i.png',
                           'material': 'M', 'dataset': 'paper', 'layer': 2,
                           'loads': [10, 20], 'traces_file': str(traces), **extra})
    return titles


def test_a_dataset_panel_titles_show_completion_by_default(tmp_path, dataset_source,
                                                           monkeypatch):
    assert _panel_titles(tmp_path, monkeypatch)[:2] == ['10 nN', '20 nN']   # failed_integrity finished


def test_a_dataset_panel_can_title_the_integrity_verdict(tmp_path, dataset_source, monkeypatch):
    titles = _panel_titles(tmp_path, monkeypatch, title_status='verdict')
    assert titles[:2] == ['10 nN', '20 nN\nfailed_integrity']


def test_a_pressure_load_has_its_own_status_key(dataset_source):
    from src.postprocessing.plot_tables import _DatasetRunFiles  # pylint: disable=import-outside-toplevel
    files = _DatasetRunFiles('paper', 'M', 2)
    files.rows = files.rows.assign(load=[0.5, 20.0])
    assert files.statuses('verdict') == {0.5: 'complete', 20.0: 'failed_integrity'}
