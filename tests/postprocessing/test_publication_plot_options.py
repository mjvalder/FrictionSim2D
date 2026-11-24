"""Real renders of plot options the publication configs name (Q13).

``scatter_comparison`` with ``color_by_material_class`` / ``show_point_labels`` /
``iterative_outlier_removal``, and the summary plot's ``legend_separate_png``
are named in ``documentation/publication/plots_publication.json``; each test
renders to a PNG and asserts the option's effect on what was drawn.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from frictionsim2d.postprocessing.plot_data import Plotter
from frictionsim2d.postprocessing.read_data import DataReader
from tests.helpers.results_tree import make_family_run

MATERIALS = ['h_A', 'h_B', 't_C', 'p_D', 'b_E']
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y = np.array([2.0, 4.0, 40.0, 8.0, 10.0])        # 't_C' is the outlier of y = 2x


@pytest.fixture
def scatter(tmp_path, monkeypatch):
    """Render a scatter_comparison over fixed data; return (figure, png path)."""
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    monkeypatch.setattr(plotter, '_get_summary_data_df', lambda *a, **k: pd.DataFrame())
    seen = []
    real_save = plotter._save_plot  # pylint: disable=protected-access
    monkeypatch.setattr(plotter, '_save_plot', lambda fig, *a, **k: (seen.append(fig), real_save(fig, *a, **k)))

    def render(**options):
        data_reset = iter([(MATERIALS, X, np.zeros(5)), (MATERIALS, Y, np.zeros(5))])
        monkeypatch.setattr(plotter, '_get_aggregated_source_data', lambda *a, **k: next(data_reset))
        plotter.generate_plot({'plot_type': 'scatter_comparison', 'filename': 'sc.png',
                               'x_source': {'dataset': 'x', 'metric': 'm'},
                               'y_source': {'dataset': 'y', 'metric': 'm'}, **options})
        png = tmp_path / 'sc.png'
        assert png.is_file()
        png.unlink()
        fig = seen[-1]
        plt.close('all')
        return fig
    return render


def _points(fig) -> int:
    return sum(len(c.get_offsets()) for c in fig.axes[0].collections)


def test_all_points_are_drawn_by_default(scatter):
    assert _points(scatter()) == 5


def test_iterative_outlier_removal_drops_the_worst_residual(scatter):
    fig = scatter(iterative_outlier_removal=1)
    assert _points(fig) == 4
    xs = np.concatenate([c.get_offsets()[:, 0] for c in fig.axes[0].collections])
    assert 3.0 not in xs                                  # 't_C', the outlier


def test_point_labels_only_with_show_point_labels(scatter):
    assert [t.get_text() for t in scatter().axes[0].texts] == []
    labelled = scatter(show_point_labels=True, point_label_fontsize=7)
    texts = labelled.axes[0].texts
    assert sorted(t.get_text() for t in texts) == sorted(MATERIALS)
    assert all(t.get_fontsize() == 7 for t in texts)


def test_color_by_material_class_gives_one_coloured_group_per_class(scatter):
    plain = scatter(color_by_material_class=False)
    assert len(plain.axes[0].collections) == 1
    classed = scatter(color_by_material_class=True)
    ax = classed.axes[0]
    labels = [c.get_label() for c in ax.collections]
    assert sorted(labels) == ['buckled', 'hexagonal', 'puckered', 'trigonal']
    colours = {tuple(c.get_facecolor()[0]) for c in ax.collections}
    assert len(colours) == 4                              # one colour per class


@pytest.mark.parametrize('separate', [False, True])
def test_legend_separate_png(tmp_path, separate):
    root = tmp_path / 'ds'
    make_family_run(root, 'afm', loads=(10.0, 20.0, 30.0))
    reader = DataReader(results_dir=str(root))
    reader.export_full_data_to_json()
    reader.export_summary_csv()
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))

    plotter.generate_plot({'plot_type': 'summary', 'x_axis': 'force', 'y_axis': 'lf',
                           'filename': 'lf.png', 'legend_separate_png': separate})
    plt.close('all')

    assert (out / 'lf.png').is_file()
    assert (out / 'lf_legend.png').is_file() is separate
