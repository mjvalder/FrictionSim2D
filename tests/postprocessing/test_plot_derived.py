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


def test_lf_sx_sy_and_lf_fx_fy_from_registry_names() -> None:
    """A frame carrying the registry's renamed sheet columns still yields
    lf_sx_sy/lf_fx_fy -- these are the names every retained file now parses
    to, since the registry's rename_map is authoritative for sheet files."""
    df = pd.DataFrame(
        {
            'react_x': [3.0, 0.0],
            'react_y': [4.0, 0.0],
            'drive_fx': [1.0, 6.0],
            'drive_fy': [0.0, 8.0],
        },
    )

    out = add_plot_derived_columns(df.copy(), time_step_fs=1.0)

    assert out['lf_sx_sy'].tolist() == [5.0, 0.0]
    assert out['lf_fx_fy'].tolist() == [1.0, 10.0]


def test_lf_sx_sy_and_lf_fx_fy_from_native_names_still_works() -> None:
    """A frame carrying the pre-registry native names (files parsed before
    the rename was declared, or read through a path that skips it) must
    still produce lf_sx_sy/lf_fx_fy -- the retained-file path is not
    allowed to regress."""
    df = pd.DataFrame(
        {
            'v_sx': [3.0, 0.0],
            'v_sy': [4.0, 0.0],
            'v_fx': [1.0, 6.0],
            'v_fy': [0.0, 8.0],
        },
    )

    out = add_plot_derived_columns(df.copy(), time_step_fs=1.0)

    assert out['lf_sx_sy'].tolist() == [5.0, 0.0]
    assert out['lf_fx_fy'].tolist() == [1.0, 10.0]
