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
