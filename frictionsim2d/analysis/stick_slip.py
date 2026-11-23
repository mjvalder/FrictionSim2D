"""Does the tip stick-slip at the lattice period? One row per AFM run.

Port of ``ML_taxonomy/scripts/s39_stickslip_v2.py`` (``measure`` :146-169)
and of taxlib's slip count (``taxlib/plausibility.py`` ``slip_quantisation``
:51-73), which s39 merges in as ``slips_per_period_taxlib`` (:214-215).

The tip rings on its spring after every slip; the ring period is the
strongest bounded component of the cubic-detrended ``raw_tipx`` spectrum, and
``raw_lfx``, ``raw_tipx`` and ``E_int_tip_sheet`` are averaged over one ring
period before anything else (s39:6-10).

1. Lateral-force period: the smoothed, quadratic-detrended force binned on
   the tip's furthest-reached displacement; the first clear peak of its
   autocorrelation after it first falls below 0 (s39:12-17).
2. Slips: smoothed-force drops larger than :data:`SLIP_K` robust sd during
   which the tip advances >= :data:`MIN_SLIP_A` lattice periods (s39:18-22).
3. Energy oscillation: the peak-to-trough of the smoothed ``E_int_tip_sheet``
   inside each whole lattice period of tip travel, averaged (s39:23-26).

``slips_per_period_taxlib`` counts tip-advance jumps on the window tier
(taxlib's ``tipx``), with taxlib's own lattice ``a`` (param ``taxlib_a``,
the materials database ``lat_a`` of taxlib_adapter.py:46-51; ``a_x`` when not
given). s39's ``pes_*`` and ``cof`` columns (:217-220) are joins with other
tables and are not carried (spec §5.2: no cross-type joins).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pydantic import field_validator
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .base import Analysis, register
from .cof_ranking import WINDOW, _loads
from .periods import (NO_PERIOD, PeriodDepends, PeriodParams, _json_before, drive_step,
                      material_key, period_map, summary_identity)

__all__ = ['BIN_A', 'LOCK', 'MIN_OVERLAP', 'MIN_SLIP_A', 'RING', 'R_CLEAR', 'SLIP_K',
           'StickSlip', 'StickSlipParams', 'TAXLIB_SLIP_K', 'measure', 'on_displacement',
           'period_from_acf', 'ring_period', 'slips', 'slips_per_period_taxlib']

#: Displacement bin of the lateral-force autocorrelation (s39:50, A).
BIN_A = 0.05
#: A of overlapping travel behind every autocorrelation lag (s39:51).
MIN_OVERLAP = 3.0
#: An autocorrelation peak at least this high is a clear period (s39:52).
R_CLEAR = 0.3
#: A clear period within this ratio of a_x is locked to the lattice (s39:53, :221).
LOCK = (0.8, 1.25)
#: Bounds on the tip's ring period (s39:54, raw samples).
RING = (30, 150)
#: A slip drops the smoothed force by this many robust sd (s39:55).
SLIP_K = 1.5
#: ... while the tip advances by at least this many lattice periods (s39:56).
MIN_SLIP_A = 0.3
#: taxlib: a window sample slips when the tip advances by more than the drive
#: step plus this many robust sd of its own jitter (plausibility.py:18).
TAXLIB_SLIP_K = 5.0
#: taxlib: a slip event covers at least this many lattice periods (plausibility.py:19).
TAXLIB_MIN_SLIP_A = 0.3
#: The drop reasons of a run without the tiers s39 reads (s39:212 keeps a
#: run only if it has both).
NO_RAW = 'no_raw_tier'
NO_PERATOM = 'no_peratom_tier'
#: The drop reason of a run whose measures cannot be taken (too short a trace).
UNMEASURABLE = 'stick_slip_unmeasurable'


def detrend(f: np.ndarray, order: int = 2) -> np.ndarray:
    """``f`` less its polynomial fit in sample index (s39:63-65)."""
    i = np.arange(len(f), dtype=float)
    return f - np.polyval(np.polyfit(i, f, order), i)


def ring_period(x: np.ndarray, bounds: Tuple[float, float] = RING) -> float:
    """Dominant oscillation period (samples) of the cubic-detrended tip
    position, within ``bounds`` (s39:68-75)."""
    i = np.arange(len(x), dtype=float)
    y = x - np.polyval(np.polyfit(i, x, 3), i)
    p = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    fr = np.fft.rfftfreq(len(y))
    ok = (fr > 1 / bounds[1]) & (fr < 1 / bounds[0])
    return float(1 / fr[ok][np.argmax(p[ok])])


def on_displacement(x: np.ndarray, f: np.ndarray,
                    bin_a: float = BIN_A) -> Tuple[np.ndarray, np.ndarray]:
    """``f`` binned on the furthest-reached displacement; empty bins (slips)
    interpolated (s39:78-87)."""
    s = np.maximum(np.maximum.accumulate(x - x[0]), 0.0)
    edges = np.arange(0.0, s[-1] + bin_a, bin_a)
    k = np.clip(np.digitize(s, edges) - 1, 0, len(edges) - 1)
    sums = np.bincount(k, f, minlength=len(edges))
    cnt = np.bincount(k, minlength=len(edges))
    grid = edges + 0.5 * bin_a
    ok = cnt > 0
    return grid, np.interp(grid, grid[ok], sums[ok] / cnt[ok])


def lag_acf(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Pearson r(lag) for lag = 0..max_lag samples, each lag on its own
    overlap (s39:90-98)."""
    r = np.full(max_lag + 1, np.nan)
    r[0] = 1.0
    for k in range(1, max_lag + 1):
        a, b = y[:-k], y[k:]
        if a.std() > 0 and b.std() > 0:
            r[k] = np.corrcoef(a, b)[0, 1]
    return r


def period_from_acf(grid: np.ndarray, y: np.ndarray, min_overlap: float = MIN_OVERLAP,
                    r_clear: float = R_CLEAR) -> Tuple[float, float, bool]:
    """``(period A, r at the peak, clear?)`` from the displacement
    autocorrelation (s39:101-117)."""
    step = grid[1] - grid[0] if len(grid) > 1 else BIN_A
    max_lag = int((grid[-1] - grid[0] - min_overlap) / step)
    if max_lag < 4:
        return np.nan, np.nan, False
    r = np.nan_to_num(lag_acf(y - y.mean(), max_lag), nan=-1.0)
    neg = np.flatnonzero(r < 0)
    if neg.size == 0:
        return np.nan, np.nan, False
    pk, _ = find_peaks(r[neg[0]:])
    if pk.size == 0:
        return np.nan, np.nan, False
    pk = neg[0] + pk
    top = pk[np.argmax(r[pk])]
    j = next(p for p in pk if r[p] >= min(0.8 * r[top], r[top]))   # the fundamental, not a multiple
    return float(j * step), float(r[j]), bool(r[j] >= r_clear)


def slips(fs: np.ndarray, xs: np.ndarray, a: float, k: float = SLIP_K,
          min_slip_a: float = MIN_SLIP_A) -> Tuple[np.ndarray, np.ndarray, float]:
    """``(drops, tip advances, threshold)``: smoothed-force drops > ``k`` x
    robust scatter with a tip slip (s39:120-134)."""
    det = detrend(fs)
    thr = k * 1.4826 * np.median(np.abs(det - np.median(det)))
    peaks, _ = find_peaks(fs, prominence=thr)
    drops, adv = [], []
    for p in peaks:
        q = p + 1
        while q < len(fs) and fs[q] <= fs[q - 1]:
            q += 1
        d, dx = fs[p] - fs[q - 1], xs[q - 1] - xs[p]
        if d >= thr and dx >= min_slip_a * a:
            drops.append(d)
            adv.append(dx)
    return np.array(drops), np.array(adv), thr


def energy_amplitude(x_e: np.ndarray, e: np.ndarray, a: float, w: int) -> Tuple[float, int]:
    """Mean peak-to-trough of E inside each whole lattice period of
    furthest-reached travel, and the number of periods (s39:137-143)."""
    s = np.maximum.accumulate(x_e - x_e[0])
    e = uniform_filter1d(e, max(w, 1), mode='nearest')
    inside = [(s >= i * a) & (s < (i + 1) * a) for i in range(int(s[-1] // a))]
    amps = [np.ptp(e[m]) for m in inside if m.sum() >= 5]
    return (float(np.mean(amps)) if amps else np.nan), len(amps)


def measure(raw: pd.DataFrame, pa: pd.DataFrame, a: float) -> Dict[str, Any]:
    """s39's columns of one run (``measure`` :146-169). ``raw``: ``time``,
    ``raw_tipx``, ``raw_lfx``; ``pa``: ``time``, ``E_int_tip_sheet``.

    s39 maps the peratom samples onto the raw ones by ``step * 10`` (:159)
    and smooths E over ``ring / 10`` of its samples (:160): the two tiers'
    sampling ratio. Here both are placed by their MD ``time``, and the ratio
    is the tiers' own (10 for the 260909 campaign: raw every 100 steps,
    peratom every 1000).
    """
    # pylint: disable=too-many-locals
    raw = raw.sort_values('time')
    x = raw['raw_tipx'].to_numpy(float)
    f = raw['raw_lfx'].to_numpy(float)
    t_raw = raw['time'].to_numpy(float)
    ring = ring_period(x)
    w = int(round(ring))
    xs = uniform_filter1d(x, w, mode='nearest')
    fs = uniform_filter1d(f, w, mode='nearest')
    grid, y = on_displacement(xs, detrend(fs))
    per, rpk, clear = period_from_acf(grid, y)
    travel = float(np.maximum.accumulate(xs - xs[0])[-1])
    drops, adv, thr = slips(fs, xs, a)
    pa = pa.sort_values('time')
    t_pa = pa['time'].to_numpy(float)
    ratio = float(np.median(np.diff(t_pa)) / np.median(np.diff(t_raw)))
    xe = np.interp(t_pa, t_raw, xs)
    amp, n_per = energy_amplitude(xe, pa['E_int_tip_sheet'].to_numpy(float), a,
                                  int(round(ring / ratio)))
    k = int(round(a / BIN_A))
    r_at_a = (float(lag_acf(y - y.mean(), k)[k]) if grid[-1] - grid[0] - k * BIN_A >= MIN_OVERLAP
              else np.nan)
    ratio_a = per / a if clear else np.nan
    return {'ring_period_samples': ring, 'travel_A': travel, 'lf_period_A': per,
            'lf_acf_peak': rpk, 'lf_period_clear': clear, 'period_ratio': ratio_a,
            'r_at_a': r_at_a, 'locked': bool(LOCK[0] <= ratio_a <= LOCK[1]),
            'n_slips': len(drops), 'slips_per_period_lf': len(drops) / (travel / a),
            'drop_per_slip_nN': float(drops.mean()) if len(drops) else np.nan,
            'slip_len_a_lf': float(np.median(adv)) / a if len(adv) else np.nan,
            'slip_threshold_nN': thr, 'E_amp_eV': amp, 'E_periods': n_per}


def slips_per_period_taxlib(tipx: np.ndarray, dx: float, a: float,
                            travel: float) -> Tuple[float, float]:
    """``(slip_len_a, slips_per_period)`` of taxlib (plausibility.py:51-73) on
    the window's ``tipx``: a sample slips when the tip advances more than the
    drive step ``dx`` plus :data:`TAXLIB_SLIP_K` robust sd of its jitter;
    consecutive slipping samples make one event, kept when it covers
    >= :data:`TAXLIB_MIN_SLIP_A` ``a``. ``travel`` is taxlib's ``L = dx * N``
    (scales.py:66)."""
    dtx = np.diff(np.asarray(tipx, dtype=float))
    jit = 1.4826 * np.median(np.abs(dtx - np.median(dtx)))      # the run's own thermal jitter
    slipping = dtx > dx + TAXLIB_SLIP_K * jit
    lengths: List[float] = []
    cur = 0.0
    for on, v in zip(slipping, dtx):
        if on:
            cur += v
        elif cur > 0:
            lengths.append(cur)
            cur = 0.0
    if cur > 0:
        lengths.append(cur)
    if np.isfinite(a):
        lengths = [x for x in lengths if x >= TAXLIB_MIN_SLIP_A * a]
    periods = travel / a if np.isfinite(a) and a > 0 else np.nan
    return (float(np.median(lengths)) / a if lengths and np.isfinite(a) else np.nan,
            len(lengths) / periods if np.isfinite(periods) else np.nan)


class StickSlipParams(PeriodParams):
    """The window loads (s39:49) and the angle, where ``a_x`` comes from
    (:class:`~.periods.PeriodParams`), and ``taxlib_a``: material -> the
    lattice ``a`` taxlib counts slips against (its materials database
    ``lat_a``); a material not in it takes ``a_x``."""

    loads: Tuple[float, ...] = WINDOW
    angle: float = 0.0
    taxlib_a: Dict[str, float] = {}

    @field_validator('loads', mode='before')
    @classmethod
    def _parse_loads(cls, value: Any) -> Any:
        return _loads(value)

    @field_validator('taxlib_a', mode='before')
    @classmethod
    def _parse_taxlib(cls, value: Any) -> Any:
        return _json_before(value)


@register
class StickSlip(PeriodDepends, Analysis):  # pylint: disable=too-few-public-methods
    """Ring period, lateral-force period, slips and energy oscillation of
    each run's tip trajectory."""

    name = 'stick_slip'
    version = '2'
    level = 'run'
    families = frozenset({'afm'})
    Params = StickSlipParams

    def compute(self, source, params) -> pd.DataFrame:
        """One row per pass run at the window loads and ``angle``. A run with
        no ``a_x`` (:data:`~.periods.NO_PERIOD`), no raw tier (:data:`NO_RAW`),
        no peratom tier (:data:`NO_PERATOM`), or a trace the measures cannot
        be taken on (:data:`UNMEASURABLE`) is dropped. With no drive step,
        ``slips_per_period_taxlib`` is NaN (s39 merges it ``how='left'``, :215)."""
        loads = tuple(float(f) for f in params.loads)
        rows = source.rows
        rows = rows[rows['load'].astype(float).isin(loads)
                    & (rows['angle'].astype(float) == float(params.angle))]
        period = period_map(source, params)
        out: List[Dict[str, Any]] = []
        for row in rows.itertuples():
            key = material_key(row.materials)
            a_x = float(period.get(key, np.nan))
            if not np.isfinite(a_x):
                source.dropped.append((str(row.point_uuid), NO_PERIOD))
                continue
            frames = source.frames(row)
            missing = next((reason for tier, reason in (('raw', NO_RAW), ('peratom', NO_PERATOM))
                            if tier not in frames), None)
            if missing is not None:
                source.dropped.append((str(row.point_uuid), missing))
                continue
            try:
                measures = measure(frames['raw'], frames['peratom'], a_x)
            except (ValueError, IndexError, ZeroDivisionError):
                source.dropped.append((str(row.point_uuid), UNMEASURABLE))
                continue
            window = frames['window'].sort_values('time')
            dx = drive_step(window, row.speed, source.run_fields(row))[1]
            spp = (slips_per_period_taxlib(window['tipx'].to_numpy(float), dx,
                                           float(params.taxlib_a.get(key, a_x)),
                                           dx * len(window))[1]
                   if np.isfinite(dx) else np.nan)
            out.append({**summary_identity(row), 'a_x_A': a_x, **measures,
                        'slips_per_period_taxlib': spp})
        return pd.DataFrame(out)
