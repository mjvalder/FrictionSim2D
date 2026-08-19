"""Versioned slip-event analyser over the 100 fs raw trace (spec §5 F4).

The slip-event definition is deliberately post-hoc and versioned: an
in-LAMMPS aggregate would freeze it at simulation time, whereas the raw trace
lets it be revised without re-simulating. Statistics are only meaningful on
the long (4 ns) subset — a 500 ps production run holds ~3 sawteeth.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

SLIP_VERSION = '1.0.0'


@dataclass(frozen=True)
class SlipResult:
    n_events: int
    amplitudes: List[float]
    asymmetry: float          # mean rise-time / fall-time ratio; nan if < 2 events
    version: str = SLIP_VERSION


def window_stats(raw_df: pd.DataFrame, window_ps: float = 1.0,
                 dt_ps: float = 0.1) -> pd.DataFrame:
    """Per-window mean/min/max/std for every channel in the raw trace.

    ``dt_ps`` is the raw sampling interval (100 fs default).
    """
    n = max(int(round(window_ps / dt_ps)), 1)
    grouped = raw_df.groupby(raw_df.index // n)
    out = {}
    for col in raw_df.columns:
        if col == 'time':
            out['time'] = grouped['time'].first()
            continue
        agg = grouped[col].agg(['mean', 'min', 'max', 'std'])
        for stat in ('mean', 'min', 'max', 'std'):
            out[f'{col}_{stat}'] = agg[stat]
    return pd.DataFrame(out).reset_index(drop=True)


def detect_slips(raw_df: pd.DataFrame, channel: str = 'raw_lfx',
                 drop_frac: float = 0.5) -> SlipResult:
    """Detect slip events as single-step drops in the lateral force.

    An event = a one-sample decrease exceeding ``drop_frac`` x the rolling
    std of the channel. The sawtooth's slow loading ramp never triggers it;
    the sudden release does.
    """
    f = raw_df[channel].to_numpy(dtype=float)
    if f.size < 3:
        return SlipResult(0, [], float('nan'))
    diffs = np.diff(f)
    scale = float(np.std(f))
    if scale == 0.0:
        return SlipResult(0, [], float('nan'))
    threshold = -drop_frac * scale
    idx = np.where(diffs < threshold)[0]
    amplitudes = [float(-diffs[i]) for i in idx]

    asymmetry = float('nan')
    if len(idx) >= 2:
        # rise time: samples between consecutive events; fall time: 1 sample
        rises = np.diff(idx).astype(float)
        asymmetry = float(np.mean(rises))     # in samples per single-step fall

    return SlipResult(int(len(idx)), amplitudes, asymmetry)
