"""Shared derived-column helpers for postprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.models import compute_derived_columns


def add_plot_derived_columns(df: pd.DataFrame, time_step_fs: float) -> pd.DataFrame:
    """Add derived quantities used by plotting workflows.

    This keeps `lf` and `cof` consistent with `read_data.py` by using the
    canonical `compute_derived_columns` helper.
    """
    if 'lfx' in df.columns and 'lfy' in df.columns:
        lateral_force, cof = compute_derived_columns(
            np.asarray(df['lfx'].values),
            np.asarray(df['lfy'].values),
            np.asarray(df['nf'].values) if 'nf' in df.columns else np.ones(len(df)),
        )
        df['lf'] = lateral_force
        if 'nf' in df.columns:
            nf_abs = df['nf'].abs().replace(0, np.nan)
            df['cof'] = pd.Series(lateral_force, index=df.index).abs().div(nf_abs)
        else:
            df['cof'] = cof
    elif 'lf' in df.columns and 'nf' in df.columns:
        nf_abs = df['nf'].abs().replace(0, np.nan)
        df['cof'] = df['lf'].abs().div(nf_abs)

    if 'v_xfrict' in df.columns and 'v_yfrict' in df.columns:
        df['lf_as_was'] = np.sqrt(df['v_xfrict']**2 + df['v_yfrict']**2)

    if 'v_sx' in df.columns and 'v_sy' in df.columns:
        df['lf_sx_sy'] = np.sqrt(df['v_sx']**2 + df['v_sy']**2)

    if 'v_fx' in df.columns and 'v_fy' in df.columns:
        df['lf_fx_fy'] = np.sqrt(df['v_fx']**2 + df['v_fy']**2)

    if 'tipz' in df.columns and 'comz' in df.columns:
        df['tip_sep'] = df['tipz'] - df['comz']

    if 'tipx' in df.columns and len(df) > 0:
        df['tipx'] = df['tipx'] - df['tipx'].iloc[0]
    if 'tipy' in df.columns and len(df) > 0:
        df['tipy'] = df['tipy'] - df['tipy'].iloc[0]

    if 'tipx' in df.columns and 'tipy' in df.columns:
        df['tip_pos'] = np.sqrt(df['tipx']**2 + df['tipy']**2)

    if all(c in df.columns for c in ['tipx', 'tipy', 'time']):
        time_diff_s = (df['time'].diff() * time_step_fs * 1e-15).fillna(0)
        dist_diff_a = np.sqrt(
            df['tipx'].diff().fillna(0)**2
            + df['tipy'].diff().fillna(0)**2
        )
        tipspeed_series = pd.Series(dist_diff_a, index=df.index) * 1e-10 / time_diff_s
        df['tipspeed'] = tipspeed_series.replace([np.inf, -np.inf], 0).fillna(0)

    return df
