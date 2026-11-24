"""Friction statistics, :class:`RunSummary` and the unit constants.

Every code path that computes COF must use :func:`compute_friction_stats`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..core.observables import PES_FAMILIES, SLIDE_FAMILIES, channel_by_role
# Re-exported: the builders and read_data import it from here.
from ..core.observables import EV_A_TO_NN  # noqa: F401  # pylint: disable=unused-import


# ---------------------------------------------------------------------------
# Physical unit conversion constants (LAMMPS metal units)
# ---------------------------------------------------------------------------

#: eV/Å³ → GigaPascal (pressure conversion).  = EV_A_TO_NN × 100.
# rounded: x100 in floats gives 160.21766340000002, which would change every deck
EV_A3_TO_GPA: float = round(EV_A_TO_NN * 100, 7)

#: Newton/metre → eV/Å² (spring constant conversion).  = EV_A_TO_NN × 10.
NM_TO_EV_A2: float = EV_A_TO_NN * 10


# ---------------------------------------------------------------------------
# Canonical friction statistics
# ---------------------------------------------------------------------------


def compute_friction_stats(
    nf: np.ndarray,
    lfx: np.ndarray,
    lfy: np.ndarray,
    skip_fraction: float = 0.2,
) -> Dict[str, float]:
    """Compute canonical friction statistics from time-series force data.

    Uses **ratio-of-means** (``mean(lf) / mean(nf)``) which is the standard
    convention in tribology for the macroscopic coefficient of friction.

    The first *skip_fraction* of the data is discarded to remove the
    initial transient (acceleration / equilibration phase).

    Args:
        nf: Normal force time-series (1-D array, nN).
        lfx: Lateral force x-component time-series (1-D array, nN).
        lfy: Lateral force y-component time-series (1-D array, nN).
        skip_fraction: Fraction of initial data to skip (default 0.2 = 20 %).

    Returns:
        Dict with keys:
            ``mean_cof``, ``std_cof``,
            ``mean_lf``, ``std_lf``,
            ``mean_nf``, ``std_nf``,
            ``mean_lfx``, ``std_lfx``,
            ``mean_lfy``, ``std_lfy``.

    Raises:
        ValueError: If input arrays are empty or have mismatched lengths.
    """
    nf = np.asarray(nf, dtype=np.float64)
    lfx = np.asarray(lfx, dtype=np.float64)
    lfy = np.asarray(lfy, dtype=np.float64)

    if nf.size == 0 or lfx.size == 0 or lfy.size == 0:
        raise ValueError("Input arrays must not be empty")
    if nf.shape != lfx.shape or nf.shape != lfy.shape:
        raise ValueError(
            f"Array shape mismatch: nf={nf.shape}, lfx={lfx.shape}, lfy={lfy.shape}"
        )
    if skip_fraction < 0.0 or skip_fraction >= 1.0:
        raise ValueError("skip_fraction must be in [0.0, 1.0)")

    # Skip transient
    skip_n = int(len(nf) * skip_fraction)
    nf_s = nf[skip_n:]
    lfx_s = lfx[skip_n:]
    lfy_s = lfy[skip_n:]

    lateral_force = np.sqrt(lfx_s ** 2 + lfy_s ** 2)

    # Ratio-of-means: μ = mean(F_L) / mean(F_N)
    mean_nf = float(np.mean(nf_s))
    mean_lf = float(np.mean(lateral_force))

    if mean_nf > 0:
        mean_cof = mean_lf / mean_nf
    else:
        mean_cof = 0.0

    # Element-wise COF for std calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        cof_elem = np.where(nf_s > 0, lateral_force / nf_s, 0.0)
    std_cof = float(np.std(cof_elem))

    return {
        'mean_cof': mean_cof,
        'std_cof': std_cof,
        'mean_lf': mean_lf,
        'std_lf': float(np.std(lateral_force)),
        'mean_nf': mean_nf,
        'std_nf': float(np.std(nf_s)),
        'mean_lfx': float(np.mean(lfx_s)),
        'std_lfx': float(np.std(lfx_s)),
        'mean_lfy': float(np.mean(lfy_s)),
        'std_lfy': float(np.std(lfy_s)),
    }


def compute_derived_columns(
    lfx: np.ndarray,
    lfy: np.ndarray,
    nf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lateral force magnitude and element-wise COF.

    This is the canonical function for adding derived columns to a
    time-series DataFrame. Used by :class:`DataReader` and the store's
    summaries (:func:`summarise_run`).

    Args:
        lfx: Lateral force x-component.
        lfy: Lateral force y-component.
        nf: Normal force.

    Returns:
        Tuple of ``(lateral_force, cof)`` arrays.
    """
    lfx = np.asarray(lfx, dtype=np.float64)
    lfy = np.asarray(lfy, dtype=np.float64)
    nf = np.asarray(nf, dtype=np.float64)

    lateral_force = np.sqrt(lfx ** 2 + lfy ** 2)
    cof = np.divide(
        lateral_force,
        nf,
        out=np.zeros_like(lateral_force),
        where=nf != 0,
    )
    return lateral_force, cof


def compute_time_series_hash(
    nf: np.ndarray,
    lfx: np.ndarray,
    lfy: np.ndarray,
) -> str:
    """Compute a SHA-256 hash of the raw time-series for deduplication.

    Args:
        nf: Normal force array.
        lfx: Lateral force x array.
        lfy: Lateral force y array.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    h = hashlib.sha256()
    for arr in (nf, lfx, lfy):
        h.update(np.asarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# RunSummary: the one summary function on the store path
# ---------------------------------------------------------------------------

#: The three roles a slide summary needs, in `compute_friction_stats` order.
_SLIDE_ROLES = ('normal_load', 'lateral_force_x', 'lateral_force_y')

#: Where the driven body went. Resolved family-agnostically like the forces --
#: AFM's tipx/tipy/tipz and the sheet family's own channels both carry these roles.
_DRIVEN_ROLES = ('driven_com_x', 'driven_com_y', 'driven_com_z')


def _finite(value):
    """`value` if a finite float, else None: AiiDA rejects nan/inf and None round-trips as null."""
    import math  # pylint: disable=import-outside-toplevel
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else None


@dataclass
class RunSummary:
    """Every number the local AiiDA store keeps for one load point.

    The ten friction statistics are exactly `compute_friction_stats`' output,
    or `None` when this load point has no COF to report -- an empty frame, a
    family whose role columns do not resolve, or a PES grid. `None`, not `0.0`:
    a zero is indistinguishable from real superlubricity, and stored attributes
    are immutable.

    `extras` carries the family-specific scalars (PES descriptors, indent
    retract features, slip counts). Every value in it passes through `_finite`.
    """
    mean_cof: Optional[float] = None
    std_cof: Optional[float] = None
    mean_lf: Optional[float] = None
    std_lf: Optional[float] = None
    mean_nf: Optional[float] = None
    std_nf: Optional[float] = None
    mean_lfx: Optional[float] = None
    std_lfx: Optional[float] = None
    mean_lfy: Optional[float] = None
    std_lfy: Optional[float] = None
    skip_fraction: float = 0.2
    ntimesteps: int = 0
    time_series_hash: str = ''
    is_complete: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """The fourteen scalars with `extras` merged in at the top level."""
        out = {k: v for k, v in self.__dict__.items() if k != 'extras'}
        out.update(self.extras)
        return out


def _role_column(df, family: str, role_key: str) -> Optional[str]:
    """The column serving `role_key` for `family`, if the frame carries it."""
    channel = channel_by_role(family, role_key)
    if channel is not None and channel.name in df.columns:
        return channel.name
    return None


#: summary key -> (x, y) window columns of the sheet family's alternative
#: lateral forces; :func:`summarise_run` records the mean magnitude.
_LF_PAIRS = {'mean_lf_sx_sy': ('react_x', 'react_y'),
            'mean_lf_fx_fy': ('drive_fx', 'drive_fy')}
_LF_PAIR_FAMILIES = frozenset(SLIDE_FAMILIES) - {'afm'}


def summarise_run(frames_by_tier: Dict[str, Any], family: str, *,
                  skip_fraction: float = 0.2,
                  post: Optional[Dict[str, float]] = None) -> RunSummary:
    """Reduce one load point's frames to the store's summary row.

    The single COF computation on the store path: the store's summaries
    (`frictionsim2d.aiida.steps.summarise`) and the postprocessing summary rows read
    this, so a number can never differ between them. `compute_friction_stats` stays the kernel.

    `post` carries the post-stream channels, which are derived from the
    retained final state rather than emitted by LAMMPS and so arrive from
    the caller.
    """
    window = frames_by_tier.get('window')
    if window is None or len(window) == 0:
        # every stat stays None: there is no COF here, and 0.0 would be a lie
        return RunSummary(skip_fraction=skip_fraction, is_complete=False)

    summary = RunSummary(skip_fraction=skip_fraction, ntimesteps=int(len(window)))

    if family in PES_FAMILIES:
        from ..postprocessing.pes_scan import (  # pylint: disable=import-outside-toplevel
            KIND_BY_FAMILY, grid_descriptors)
        # A PES grid has no COF at all -- the ten stats stay None, and
        # is_complete stays True because the grid IS this family's complete
        # result. Consumers filter on `mean_cof is not None`, not is_complete.
        summary.extras.update(
            {k: _finite(v) for k, v in
             grid_descriptors(window, KIND_BY_FAMILY[family]).items()})
        summary.time_series_hash = compute_time_series_hash(
            window['energy_eV'].to_numpy(dtype=float), np.zeros(1), np.zeros(1))
        return summary

    cols = [_role_column(window, family, role) for role in _SLIDE_ROLES]
    if all(cols):
        nf, lfx, lfy = (window[c].to_numpy(dtype=float) for c in cols)
        for key, val in compute_friction_stats(nf, lfx, lfy, skip_fraction).items():
            setattr(summary, key, _finite(val))
        summary.time_series_hash = compute_time_series_hash(nf, lfx, lfy)
    else:
        # the role columns did not resolve: no stats, and say so
        summary.is_complete = False

    # The sheet family's two alternative lateral forces: the magnitude of the
    # bottom layer's reaction (sx_sy) and of the top layer's drive (fx_fy),
    # averaged over the WHOLE window like the postprocessing summary always did.
    if family in _LF_PAIR_FAMILIES:
        for name, (x_col, y_col) in _LF_PAIRS.items():
            if x_col in window.columns and y_col in window.columns:
                summary.extras[name] = _finite(np.mean(np.hypot(
                    window[x_col].to_numpy(dtype=float),
                    window[y_col].to_numpy(dtype=float))))

    dx_col, dy_col, dz_col = (_role_column(window, family, role)
                              for role in _DRIVEN_ROLES)
    if dx_col and dy_col:
        drv_x = window[dx_col].to_numpy(dtype=float)
        drv_y = window[dy_col].to_numpy(dtype=float)
        summary.extras['slide_distance'] = _finite(
            np.sum(np.hypot(np.diff(drv_x), np.diff(drv_y))))
    if dz_col:
        drv_z = window[dz_col].to_numpy(dtype=float)
        summary.extras['driven_com_z_mean'] = _finite(np.mean(drv_z))
        summary.extras['driven_com_z_drift'] = _finite(drv_z[-1] - drv_z[0])

    retract = frames_by_tier.get('retract')
    if family == 'indent' and retract is not None and len(retract):
        from ..postprocessing.indent import retract_features  # pylint: disable=import-outside-toplevel,cyclic-import
        summary.extras.update(
            {k: _finite(v) for k, v in retract_features(retract).items()})

    raw = frames_by_tier.get('raw')
    if raw is not None and 'raw_lfx' in getattr(raw, 'columns', ()):
        from ..postprocessing.slip_analysis import detect_slips  # pylint: disable=import-outside-toplevel
        # Through `_finite`: slip_mean_interval_samples is nan for <2 events,
        # size<3 or a flat trace, and AiiDA's clean_value raises on nan, which
        # would abort Node.store() rather than record a missing number.
        summary.extras.update(
            {k: _finite(v)
             for k, v in detect_slips(raw, channel='raw_lfx').summary().items()})

    if post:
        summary.extras.update({k: float(v) for k, v in post.items()})

    return summary
