"""scatter_comparison sources: load_column / load_range / require_all_loads / sem."""

import numpy as np
import pandas as pd
import pytest

from src.postprocessing.plot_data import Plotter


def _df():
    rows = [('ds', 'A', p, c) for p, c in ((5, 0.1), (10, 0.2), (20, 0.3))]
    rows += [('ds', 'B', p, c) for p, c in ((5, 0.4), (10, 0.5))]       # no 20 GPa point
    return pd.DataFrame(rows, columns=['dataset_label', 'id', 'pressure', 'cof'])


def _agg(tmp_path, **src):
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))
    return plotter._get_aggregated_source_data(
        _df(), {'dataset': 'ds', 'metric': 'cof', **src})


def test_require_all_loads_drops_incomplete_material(tmp_path):
    mats, vals, _ = _agg(tmp_path, load_column='pressure', load_range=[5, 20],
                         require_all_loads=True)
    assert mats == ['A'] and vals[0] == pytest.approx(0.2)


def test_without_require_all_loads_both_materials_stay(tmp_path):
    mats, _, _ = _agg(tmp_path, load_column='pressure', load_range=[5, 20])
    assert mats == ['A', 'B']


def test_sem_error_is_sd_over_sqrt_n(tmp_path):
    _, _, errs = _agg(tmp_path, load_column='pressure', load_range=[5, 20],
                      error_metric='sem')
    assert errs[0] == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1) / np.sqrt(3))
