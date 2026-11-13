"""Post-run integrity measurements: a faithful port of ML_taxonomy's rules.

A load point does not describe the physical system when the run blew up,
ejected atoms, split into islands or had a hole punched through it (the
Stillinger-Weber potential cannot break bonds). This module measures those
symptoms; the verdict that compares them with ``IntegritySettings`` sits on top.

Sources of truth (ML_taxonomy, uncommitted at port time, so pinned by sha256):

* ``~/mnt/PhD/Sync/ML_taxonomy/scripts/s48_integrity_rules.py``
  sha256 ``8e488031349a7af8b7d6ea72446be1bfc70c79aead6dc010dc309d1f6f9d3329``
  -- ``read_data`` (:57), ``wrapped`` (:73), ``layer_measures`` (:82),
  ``snapshot`` aggregation (:95-123), ``series`` (:134), AFM heating (:191).
* ``~/mnt/PhD/Sync/ML_taxonomy/scripts/s51_sheet_integrity_v2.py``
  sha256 ``a60b0e3045f6ffa50bfd9902156c697ae7ce9ced4c073e243a650e4d42bf2378``
  -- ``read_molecular`` (:53), ``point`` series measures (:89-102), and the
  VARIANT verdict (:172-175): heating peak after the first samples, gap rule
  = lasting end-of-slide shift of either sign; plus ``gap_min_any`` (collapse).
* ``~/mnt/PhD/Sync/ML_taxonomy/taxlib/gate.py``
  sha256 ``f9b6cb6c4911f462cfeaddbd0c0eb679fe59b4555a077f419e76e242db95e13d``
  -- ``_check`` (:11): core channels, then ``etotal`` for non-finite values.

``scipy`` is imported inside the functions that need it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.core.config import IntegritySettings

AFM_LIKE = ('afm', 'indent', 'pes_tip')
SHEET_LIKE = ('sheetonsheet', 'hetero', 'flake', 'pes_sheet')

_AFM_CORE = ('nf', 'lfx', 'lfy', 'comx', 'comy', 'comz', 'tipx', 'tipy', 'tipz')
_SHEET_CORE = ('lfx', 'lfy', 'nf')

# Atoms-section layouts: (index of the type column, index of the x column).
_STYLE_COLUMNS = {
    'atomic': (1, 2),
    'charge': (1, 3),
    'molecular': (2, 3),
    'bond': (2, 3),
    'angle': (2, 3),
    'full': (2, 4),
}
_BOX_RE = re.compile(r"\s*(\S+)\s+(\S+)\s+(\w)lo\s+\whi")
_ATOMS_RE = re.compile(r"\s*Atoms\b\s*(?:#\s*(\S+))?")

Box = Dict[str, Tuple[float, float]]


def read_snapshot(path: Path) -> Tuple[np.ndarray, np.ndarray, Box]:
    """Types, positions and box of a LAMMPS data file.

    Port of s48 ``read_data`` (atomic) and s51 ``read_molecular``. The style
    comes from the ``Atoms # <style>`` comment; without one, a 5- or 8-column
    row (``id type x y z`` with optional image flags) is read as atomic and
    anything else raises ``ValueError``, as does an unknown style.
    """
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    start = next((k for k, line in enumerate(lines) if _ATOMS_RE.match(line)), None)
    if start is None:
        raise ValueError(f"{path}: no Atoms section")
    box: Box = {}
    for line in lines[:start]:
        m = _BOX_RE.match(line)
        if m:
            box[m[3]] = (float(m[1]), float(m[2]))
    if set(box) != {'x', 'y', 'z'}:
        raise ValueError(f"{path}: box bounds incomplete ({sorted(box)})")
    raw = []
    for line in lines[start + 1:]:
        s = line.split('#', 1)[0].split()
        if not s:
            if raw:
                break
            continue
        raw.append(s)
    if not raw:
        raise ValueError(f"{path}: empty Atoms section")
    style = _ATOMS_RE.match(lines[start])[1]
    if style is None:
        if len(raw[0]) not in (5, 8):
            raise ValueError(f"{path}: Atoms section has no style comment and "
                             f"{len(raw[0])} columns; cannot tell its layout")
        style = 'atomic'
    if style not in _STYLE_COLUMNS:
        raise ValueError(f"{path}: unsupported atom style {style!r}")
    it, ix = _STYLE_COLUMNS[style]
    try:
        types = np.array([int(s[it]) for s in raw], dtype=int)
        xyz = np.array([[float(v) for v in s[ix:ix + 3]] for s in raw], dtype=float)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: malformed Atoms row for style {style!r}") from exc
    if xyz.shape != (len(raw), 3):
        raise ValueError(f"{path}: malformed Atoms row for style {style!r}")
    return types, xyz, box


def _wrapped(xyz: np.ndarray, box: Box) -> Tuple[np.ndarray, np.ndarray]:
    """s48 ``wrapped`` (:73): wrap x, y into the box; z non-periodic (1e6)."""
    lengths = np.array([box['x'][1] - box['x'][0], box['y'][1] - box['y'][0]])
    p = xyz.copy()
    p[:, 0] = (p[:, 0] - box['x'][0]) % lengths[0]
    p[:, 1] = (p[:, 1] - box['y'][0]) % lengths[1]
    p[:, 2] -= p[:, 2].min() - 1.0
    return p, np.array([lengths[0], lengths[1], 1e6])


def layer_measures(final_xyz: np.ndarray, ref_xyz: np.ndarray, final_box: Box,
                   ref_box: Box, settings: IntegritySettings) -> Dict[str, float]:
    """One layer's bond, isolated, ejected, hole and atom count (s48 :82).

    ``bond`` is the median periodic nearest-neighbour distance of the reference;
    ``hole`` the largest distance from an xy grid point (step bond/4) to the
    nearest atom, in bonds. A layer with fewer than two atoms, or a rule whose
    radius setting is ``None``, measures NaN.
    """
    from scipy.spatial import cKDTree  # pylint: disable=import-outside-toplevel

    final_xyz = np.asarray(final_xyz, dtype=float)
    ref_xyz = np.asarray(ref_xyz, dtype=float)
    n = len(final_xyz)
    nan = float('nan')
    if n < 2 or len(ref_xyz) < 2:
        return {'bond': nan, 'isolated': nan, 'ejected': nan, 'hole': nan, 'n': n}
    p0, b0 = _wrapped(ref_xyz, ref_box)
    bond = float(np.median(cKDTree(p0, boxsize=b0).query(p0, k=2)[0][:, 1]))
    p1, b1 = _wrapped(final_xyz, final_box)
    nn = cKDTree(p1, boxsize=b1).query(p1, k=2)[0][:, 1]
    step = bond / 4
    gx, gy = np.meshgrid(np.arange(0, b1[0], step), np.arange(0, b1[1], step))
    hole = cKDTree(p1[:, :2], boxsize=b1[:2]).query(np.c_[gx.ravel(), gy.ravel()], k=1)[0].max()
    dz = np.abs(final_xyz[:, 2] - np.median(final_xyz[:, 2]))
    radius, eject = settings.isolated_radius_bonds, settings.ejected_bonds
    return {
        'bond': bond,
        'isolated': nan if radius is None else int((nn > radius * bond).sum()),
        'ejected': nan if eject is None else int((dz > eject * bond).sum()),
        'hole': float(hole / bond),
        'n': n,
    }


def snapshot_measures(final_path: Path, ref_path: Path, layers: Dict[str, Set[int]],
                      settings: IntegritySettings) -> Dict[str, float]:
    """``layer_measures`` per layer, combined as s48 ``snapshot`` (:95-123).

    ``isolated``, ``ejected`` and ``n`` are summed, ``hole`` is the maximum; a
    NaN layer measure makes the combined one NaN. ``n_by_layer`` maps each
    layer name to its atom count in the final snapshot.
    """
    t1, x1, box1 = read_snapshot(final_path)
    t0, x0, box0 = read_snapshot(ref_path)
    agg: Dict[str, float] = {'isolated': 0, 'ejected': 0, 'hole': 0.0, 'n': 0}
    n_by_layer: Dict[str, int] = {}
    for name, types in layers.items():
        t = sorted(types)
        r = layer_measures(x1[np.isin(t1, t)], x0[np.isin(t0, t)], box1, box0, settings)
        agg['isolated'] += r['isolated']
        agg['ejected'] += r['ejected']
        agg['hole'] = float(np.maximum(agg['hole'], r['hole']))
        agg['n'] += r['n']
        n_by_layer[name] = int(r['n'])
    return {**agg, 'n_by_layer': n_by_layer}


def _present(window: pd.DataFrame, names: Sequence[str]) -> list:
    return [c for c in names if c in window.columns]


def _gap(window: pd.DataFrame, lower: int, upper: int) -> Optional[pd.Series]:
    lo, hi = f'comz_L{lower}', f'comz_L{upper}'
    if lo not in window.columns or hi not in window.columns:
        return None
    return window[hi].astype(float) - window[lo].astype(float)


def _t_sigma(peak: float, t_target: float, n_heat: Optional[int]) -> float:
    """(peak - T) / sigma_T, sigma_T = T sqrt(2 / 3N) (s48 :191, s51 :169)."""
    if n_heat is None or not n_heat or not np.isfinite(peak):
        return float('nan')
    return float((peak - t_target) / (t_target * np.sqrt(2.0 / (3.0 * n_heat))))


def series_measures(window: pd.DataFrame, family: str, n_layers: int, T_target: float,  # pylint: disable=invalid-name
                    n_heat: Optional[int], settings: IntegritySettings) -> Dict[str, float]:
    """Gate, heating, gap and collapse measures of one point's window frame.

    AFM-like families follow s48 ``series`` (:134): peak sheet temperature, gap
    rise = peak of ``comz_L2 - comz_L1`` over its start. Sheet-like families
    follow the s51 variant: temperature peak after ``sheet_heating_skip_samples``,
    gap rise = |end median - start| / start at the shear interface
    ``L{k+1} - L{k}`` (``k = n_layers // 2``), and ``gap_min_any`` over every
    adjacent pair. Only core channels present in ``window`` are checked.
    """
    if family in AFM_LIKE:
        core = _present(window, _AFM_CORE)
        frozen_cols = core
    elif family in SHEET_LIKE:
        layer_cols = [f'comz_L{i}' for i in range(1, n_layers + 1)]
        core = _present(window, (*_SHEET_CORE, *layer_cols))
        # layer 1 is held fixed, so its COM is constant by design (s51 :92-93)
        frozen_cols = [c for c in core if c != 'comz_L1']
    else:
        raise ValueError(f"unknown family {family!r}; known: {AFM_LIKE + SHEET_LIKE}")
    nan = float('nan')

    finite_cols = core + _present(window, ('etotal',))
    non_finite = bool(finite_cols) and bool(
        (~np.isfinite(window[finite_cols].to_numpy(float))).any())

    frozen = False
    if settings.frozen_std is not None and frozen_cols:
        values = window[frozen_cols].astype(float)
        if family in AFM_LIKE:
            std = values.std().to_numpy(float)                   # s48 :142 (pandas, ddof=1)
        else:
            std = np.nanstd(values.to_numpy(float), axis=0)      # s51 :93 (numpy, ddof=0)
        frozen = bool((std < settings.frozen_std).any())

    g0n = settings.gap_start_samples
    if family in AFM_LIKE:
        temp = window['temp_sheet_3d'].astype(float) if 'temp_sheet_3d' in window else None
        t_sigma = nan if temp is None else _t_sigma(temp.max(), T_target, n_heat)
        gap = _gap(window, 1, 2) if n_layers >= 2 else None
        if gap is None:
            gap_rise = nan
        else:
            g0 = gap.head(g0n).median()
            gap_rise = float((gap.max() - g0) / g0)
        gap_min_any = nan
    else:
        temp = window['temp_center_3d'].astype(float) if 'temp_center_3d' in window else None
        t_sigma = nan if temp is None else _t_sigma(
            temp.iloc[settings.sheet_heating_skip_samples:].max(), T_target, n_heat)
        k = n_layers // 2
        gap = _gap(window, k, k + 1) if n_layers >= 3 else None
        if gap is None:
            gap_rise = nan
        else:
            g0 = gap.head(g0n).median()
            gap_rise = float(abs((gap.tail(settings.sheet_gap_end_samples).median() - g0) / g0))
        gaps = [_gap(window, i, i + 1) for i in range(1, n_layers)] if n_layers >= 2 else []
        gaps = [g for g in gaps if g is not None]
        gap_min_any = float(min(g.min() for g in gaps)) if gaps else nan

    return {'non_finite': non_finite, 'frozen': frozen, 'T_sigma': t_sigma,
            'gap_rise': gap_rise, 'gap_min_any': gap_min_any}
