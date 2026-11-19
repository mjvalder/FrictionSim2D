"""Legend placement/order and fixed axes margins, ported from the publication plotter."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from src.postprocessing.plot_config import SummaryPlot  # noqa: E402
from src.postprocessing.plot_data import Plotter  # noqa: E402


def _plotter(tmp_path):
    return Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))


def _axes_with(labels):
    fig, ax = plt.subplots()
    for i, label in enumerate(labels):
        ax.plot([0, 1], [i, i + 1], label=label)
    return fig, ax


def test_legend_outside_accepts_bool_and_side():
    for value in (True, False, 'left', 'top'):
        assert SummaryPlot(legend_outside=value).legend_outside == value
    with pytest.raises(ValueError):
        SummaryPlot(legend_outside='sideways')


def test_legend_order_reorders_handles(tmp_path):
    fig, ax = _axes_with(['a', 'b', 'c'])
    _plotter(tmp_path)._finalize_plot(ax, {'legend_order': ['c', 'a']})
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['c', 'a', 'b']
    plt.close(fig)


def test_legend_outside_top_right_aligned(tmp_path):
    fig, ax = _axes_with(['a', 'b'])
    _plotter(tmp_path)._finalize_plot(
        ax, {'legend_outside': 'top', 'legend_top_align': 'right', 'legend_ncol': 2})
    legend = ax.get_legend()
    assert legend._loc == 4  # 'lower right'  # pylint: disable=protected-access
    assert legend._ncols == 2  # pylint: disable=protected-access
    plt.close(fig)


def test_axes_margins_in_widens_the_saved_image(tmp_path):
    """A 3 in left margin is more than the tight bbox needs, so the PNG gets wider."""
    plotter = _plotter(tmp_path)
    plotter.settings['figure']['dpi'] = 50
    fig, ax = _axes_with(['a'])
    plotter._save_plot(fig, 'tight.png')
    fig, ax = _axes_with(['a'])
    plotter._save_plot(fig, 'wide.png', axes_margins_in={'left': 3.0})
    tight_w = Image.open(tmp_path / 'tight.png').width
    wide_w = Image.open(tmp_path / 'wide.png').width
    assert wide_w - tight_w > 50  # >1 in at 50 dpi
