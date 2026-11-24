"""Shared derived-column helpers for postprocessing pipelines."""

from __future__ import annotations

from typing import Dict, Optional

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

    # The registry renames the sheet's native v_sx/v_sy and v_fx/v_fy to
    # react_x/react_y and drive_fx/drive_fy for retained files, but older
    # files parsed before that declaration -- or read through a path that
    # skips the rename -- still carry the native names. Both are accepted,
    # mirroring DataReader._branch_column / _lateral_columns.
    if 'react_x' in df.columns and 'react_y' in df.columns:
        df['lf_sx_sy'] = np.sqrt(df['react_x']**2 + df['react_y']**2)
    elif 'v_sx' in df.columns and 'v_sy' in df.columns:
        df['lf_sx_sy'] = np.sqrt(df['v_sx']**2 + df['v_sy']**2)

    if 'drive_fx' in df.columns and 'drive_fy' in df.columns:
        df['lf_fx_fy'] = np.sqrt(df['drive_fx']**2 + df['drive_fy']**2)
    elif 'v_fx' in df.columns and 'v_fy' in df.columns:
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


def linear_fit(x_data, y_data, x_range: Optional[list] = None,
               constraint: str = 'first_point') -> Optional[Dict[str, float]]:
    """Constrained linear regression; ``None`` when there is nothing to fit.

    ``constraint`` is 'first_point' (default, the line passes through the first
    point), 'origin' or 'none'.
    """
    if len(x_data) < 2:
        return None

    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = np.array(x_data)[mask]
    y_clean = np.array(y_data)[mask]

    if x_range is not None and len(x_range) == 2:
        range_mask = (x_clean >= x_range[0]) & (x_clean <= x_range[1])
        x_clean, y_clean = x_clean[range_mask], y_clean[range_mask]

    if len(x_clean) < 2:
        return None

    x_origin = 0.0
    if constraint == 'origin':
        denom = np.sum(x_clean**2)
        if denom == 0:
            return None
        slope = np.sum(x_clean * y_clean) / denom
        intercept = 0.0
        x_shifted = x_clean
    elif constraint == 'none':
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        x_shifted = x_clean - np.mean(x_clean)
    else:
        x_origin, y_origin = x_clean[0], y_clean[0]
        x_shifted = x_clean - x_origin
        y_shifted = y_clean - y_origin

        denom = np.sum(x_shifted**2)
        if denom == 0:
            return None

        slope = np.sum(x_shifted * y_shifted) / denom
        intercept = y_origin - slope * x_origin

    y_pred = slope * x_clean + intercept
    residuals = y_clean - y_pred
    rmse = np.sqrt(np.mean(residuals**2))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    n = len(x_clean)
    if n > 1:
        mse = ss_res / (n - 1)
        slope_stderr = np.sqrt(mse / np.sum(x_shifted**2))
    else:
        slope_stderr = 0

    if constraint == 'first_point':
        se_intercept = slope_stderr * np.abs(x_origin)
    else:
        se_intercept = 0

    return {
        'slope': slope,
        'intercept': intercept,
        'slope_stderr': slope_stderr,
        'intercept_stderr': se_intercept,
        'r_squared': r_squared,
        'rmse': rmse,
    }
