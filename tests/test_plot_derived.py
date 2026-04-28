"""Tests for shared derived-column helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.postprocessing.plot_derived import add_plot_derived_columns


def test_add_plot_derived_columns_produces_finite_core_fields() -> None:
    """Shared derived helper should produce stable lf/cof/tipspeed values."""
    df = pd.DataFrame(
        {
            'lfx': [1.0, 2.0, 3.0],
            'lfy': [0.0, 0.0, 4.0],
            'nf': [2.0, 0.0, -5.0],
            'time': [0.0, 0.0, 1.0],
            'tipx': [0.0, 1.0, 2.0],
            'tipy': [0.0, 0.0, 0.0],
        },
    )

    out = add_plot_derived_columns(df.copy(), time_step_fs=1.0)

    assert 'lf' in out.columns
    assert 'cof' in out.columns
    assert 'tipspeed' in out.columns
    assert np.isfinite(out['lf']).all()
    assert np.isnan(out['cof'].iloc[1])
    assert np.isfinite(out['tipspeed']).all()
