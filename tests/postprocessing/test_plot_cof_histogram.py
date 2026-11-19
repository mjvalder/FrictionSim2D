"""cof_histogram: the adhesion bar pairs ported from the publication plotter.

With ``show_adhesion_pairs`` each series gets two bars: the mean over loads of
mean(LF)/mean(NF) (with adhesion, error SD/sqrt(n)) and the LF-NF slope (without
adhesion, error the slope's standard error). MLIP_COF_HANDOFF.md defines both.
"""

import json

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.postprocessing.plot_data import Plotter  # noqa: E402


def test_mean_load_ratio_matches_handcalc():
    ratio, err = Plotter._mean_load_ratio([10, 20, 40], [2.0, 3.0, 4.0])
    r = np.array([0.2, 0.15, 0.1])
    assert ratio == pytest.approx(r.mean())
    assert err == pytest.approx(r.std(ddof=1) / np.sqrt(3))


def test_ratio_x_range_drops_low_loads():
    ratio, _ = Plotter._mean_load_ratio([10, 20, 40], [2.0, 3.0, 4.0], x_range=[15, 50])
    assert ratio == pytest.approx((0.15 + 0.1) / 2)


def test_zero_load_has_no_ratio():
    ratio, err = Plotter._mean_load_ratio([0, 10], [1.0, 2.0])
    assert ratio == pytest.approx(0.2) and err == 0.0


def _series_file(tmp_path):
    path = tmp_path / 'ext.json'
    path.write_text(json.dumps({'a': {'nf': [10, 20, 40], 'lf': [2.0, 3.0, 4.0]},
                                'b': {'nf': [10, 20, 40], 'lf': [1.0, 2.0, 3.0]}}),
                    encoding='utf-8')
    return path


def test_paired_bars_draw_two_per_series(tmp_path, monkeypatch):
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    drawn = {}
    import matplotlib.axes  # pylint: disable=import-outside-toplevel
    real_bar = matplotlib.axes.Axes.bar

    def spy(ax, x, height, *args, **kwargs):
        drawn['heights'] = list(height)
        return real_bar(ax, x, height, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'bar', spy)
    ext = str(_series_file(tmp_path))
    plotter.generate_plot({
        'plot_type': 'cof_histogram', 'filename': 'h.png', 'show_adhesion_pairs': True,
        'external_series': [
            {'file': ext, 'path': ['a'], 'x_key': 'nf', 'y_key': 'lf', 'label': 'A',
             'solid': True},
            {'file': ext, 'path': ['b'], 'x_key': 'nf', 'y_key': 'lf', 'label': 'B'}]})
    assert (tmp_path / 'h.png').is_file()
    heights = drawn['heights']
    assert len(heights) == 4
    # order per series: with adhesion (ratio), then without (slope)
    assert heights[0] == pytest.approx(np.mean([0.2, 0.15, 0.1]))
    assert heights[3] == pytest.approx(np.polyfit([10, 20, 40], [1.0, 2.0, 3.0], 1)[0])
