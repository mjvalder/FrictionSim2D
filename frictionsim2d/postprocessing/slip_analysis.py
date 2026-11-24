"""Versioned slip-event analyser over the 100 fs raw trace.

The slip-event definition is deliberately post-hoc and versioned: an
in-LAMMPS aggregate would freeze it at simulation time, whereas the raw trace
lets it be revised without re-simulating. Statistics are only meaningful on
the long (4 ns) subset — a 500 ps production run holds ~3 sawteeth.
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

SLIP_VERSION = '1.0.0'


@dataclass(frozen=True)
class SlipResult:
    """One slip-event analysis of a raw lateral-force trace.

    ``mean_interval_samples`` is the mean number of raw samples between
    consecutive events -- the sawtooth period in samples.
    """
    n_events: int
    amplitudes: List[float]
    mean_interval_samples: float       # nan if < 2 events
    version: str = SLIP_VERSION

    def summary(self) -> Dict[str, float]:
        """The four scalars ``RunSummary.extras`` carries for this run.

        Every key keeps the ``slip_`` prefix, so it lands in a summary
        beside the friction statistics without colliding with any of them.

        ``slip_mean_interval_samples`` can be nan (fewer than two events);
        callers that hand these to AiiDA pass them through ``_finite`` first,
        as ``summarise_run`` does -- ``clean_value`` raises on nan.
        """
        return {
            'slip_n_events': float(self.n_events),
            'slip_amplitude_mean': (float(np.mean(self.amplitudes))
                                    if self.amplitudes else 0.0),
            'slip_amplitude_std': (float(np.std(self.amplitudes))
                                   if self.amplitudes else 0.0),
            'slip_mean_interval_samples': float(self.mean_interval_samples),
        }


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

    mean_interval_samples = float('nan')
    if len(idx) >= 2:
        # spacing between consecutive events, in raw samples: the sawtooth
        # PERIOD. The fall is one sample by construction, so the old
        # "rise/fall ratio" was only ever the rise.
        mean_interval_samples = float(np.mean(np.diff(idx).astype(float)))

    return SlipResult(int(len(idx)), amplitudes, mean_interval_samples)
