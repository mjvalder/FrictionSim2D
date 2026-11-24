"""Whole-period COF of each run: the mean over the whole lattice periods of
sliding that follow the spring-loading transient, counted back from the end.

Port of ``ML_taxonomy/scripts/rank_cof_v2.py`` (``run_cof`` :40-51) with the
transient of ``ML_taxonomy/taxlib/scales.py`` (``settle_skip`` :22-27) for the
AFM family, and of
``publication_export/provenance_scripts/convert_sheet_schema3_to_legacy.py``
(``analyse`` :72-104, the source of ``261107_sheetonsheet/periods``) for the
sheet-on-sheet family:

- ``dx``      = the drive distance per window sample (A);
- ``n_skip``  = ``ceil((|mean lfx over the 2nd half| / K) / dx)``, capped at
  half the window;
- ``period``  = ``a_x / dx`` samples, ``a_x`` the material's ``period_x_top``
  (:mod:`.sheet_geometry`);
- ``k``       = ``floor((N - n_skip) / period)`` whole periods, the last
  ``round(k * period)`` samples; ``k < 1`` falls back to ``[n_skip:]``.

Only 0 degree runs are cut: ``a_x`` is the period along x.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from .base import Analysis, register

logger = logging.getLogger(__name__)

__all__ = ['GEOMETRY', 'NO_DRIVE_STEP', 'NO_PERIOD', 'OFF_AXIS', 'PeriodDepends', 'PeriodParams',
           'PeriodSummary', 'dominant_period', 'geometry_depends', 'period_map', 'run_periods',
           'settle_skip', 'whole_periods']

#: 1 eV/A in nN, per family: taxlib/constants.py:3 (AFM) and the sheet
#: converter's EV_PER_A_TO_NN (:37). They differ in the 8th figure; each is kept.
EV_PER_A_TO_NN = {'afm': 1.602176634, 'sheetonsheet': 1.602176565}
#: 1 N/m in eV/A^2 (runtax/ingest_v2.py:41): a run without a recorded
#: ``drive_k`` takes its ``driving_spring`` through it (ingest_v2.py:102-103).
N_PER_M_TO_EV_PER_A2 = 0.06241509
#: The transient is capped at this fraction of the window (scales.py:22).
SETTLE_CAP = 0.5
#: The drop reason of a run at a nonzero angle.
OFF_AXIS = 'period_x_only_at_0deg'
#: The drop reason of a run whose material has no ``a_x`` (no geometry row,
#: no ``lattice`` fallback): it has no whole periods to average over.
NO_PERIOD = 'no_lattice_period'
#: The drop reason of a run with no drive step (no speed or timestep).
NO_DRIVE_STEP = 'no_drive_step'
#: The layer whose ``period_x_top`` sets ``a_x``: the published tables read
#: the v2 AFM bilayers' (rank_cof_v2.py:72; the converter's GEOMETRY, :36).
#: A material without one takes its monolayer's, then its own stack's.
GEOMETRY_LAYER = 2
#: Fewer raw samples than this give no FFT period (converter :61).
FFT_MIN_SAMPLES = 64


def settle_skip(lfx: np.ndarray, k_spring: float, dx: float, cap: float = SETTLE_CAP) -> int:
    """Samples of the spring-loading transient: the stretch ``|late mean
    lfx| / K`` in drive samples, capped at ``cap * N`` (scales.py:22-27).
    0 when ``K`` or ``dx`` is not a positive number."""
    if not np.isfinite(k_spring) or k_spring <= 0 or not np.isfinite(dx) or dx <= 0:
        return 0
    lfx = np.asarray(lfx, dtype=float)
    late_half = lfx[len(lfx) // 2:]
    if not np.isfinite(late_half).any():
        return 0
    # taxlib's pandas .mean() skips NaN.
    late = float(np.abs(np.nanmean(late_half)))
    return int(min(np.ceil((late / k_spring) / dx), cap * len(lfx)))


def dominant_period(drive: np.ndarray, force: np.ndarray) -> float:
    """Wavelength (in drive distance) of the strongest Fourier component of
    the detrended force (converter ``dominant_period``, :59-69)."""
    drive = np.asarray(drive, dtype=float)
    force = np.asarray(force, dtype=float)
    if len(force) < FFT_MIN_SAMPLES:
        return float('nan')
    detrended = force - np.polyval(np.polyfit(drive, force, 1), drive)
    spec = np.abs(np.fft.rfft(detrended * np.hanning(len(detrended))))
    step = float(np.median(np.diff(drive)))
    freq = np.fft.rfftfreq(len(detrended), d=step)
    spec[:3] = 0.0                       # drop the drift left after detrending
    i = int(np.argmax(spec))
    return 1.0 / freq[i] if freq[i] > 0 else float('nan')


def whole_periods(n: int, n_skip: int, period: float) -> Tuple[int, int]:
    """``(k, start_row)``: the whole periods after the transient, counted back
    from the end (rank_cof_v2.py:48-49, converter :97-98). No usable period,
    or less than one, gives ``(0, n_skip)``."""
    if not np.isfinite(period) or period <= 0:
        return 0, n_skip
    k = int((n - n_skip) // period)
    return (k, n - int(round(k * period))) if k >= 1 else (0, n_skip)


def _forces(family: str, window: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lf, nf, cof)`` per sample. AFM: ``|lfx, lfy|`` over the signed nf,
    0 where nf is 0 (``data.models.compute_derived_columns``, the export's
    ``cof`` that afm2_long carries). Sheet: ``|lfx, lfy|`` over ``|nf|``,
    NaN where it is 0 (converter :74-77)."""
    lfx = window['lfx'].to_numpy(float)
    lfy = window['lfy'].to_numpy(float)
    nf = window['nf'].to_numpy(float)
    lf = np.hypot(lfx, lfy)
    if family == 'sheetonsheet':
        nf = np.abs(nf)
        with np.errstate(divide='ignore', invalid='ignore'):
            return lf, nf, np.where(nf > 0, lf / nf, np.nan)
    from ..data.models import compute_derived_columns  # pylint: disable=import-outside-toplevel
    lf, cof = compute_derived_columns(lfx, lfy, nf)
    return lf, nf, cof


def _mean(values: np.ndarray, family: str) -> float:
    """The sheet converter's ``nanmean`` (:82, :101); a plain mean for AFM
    (rank_cof_v2.py:50)."""
    return float(np.nanmean(values) if family == 'sheetonsheet' else values.mean())


def _number(fields: Dict[str, Any], key: str) -> float:
    """``fields[key]`` as a float; absent or None (a schema default of None)
    is NaN."""
    value = fields.get(key)
    return float('nan') if value is None else float(value)


def _spring(fields: Dict[str, Any], family: str) -> float:
    """K in nN/A: ``derived.drive_k`` (eV/A^2), else ``driving_spring`` (N/m)
    in eV/A^2 (ingest_v2.py:102-103), times the family's eV/A -> nN."""
    drive_k = fields.get('derived.drive_k')
    if drive_k is None:
        spring = fields.get('general.driving_spring')
        if spring is None:
            return float('nan')
        drive_k = float(spring) * N_PER_M_TO_EV_PER_A2
    return float(drive_k) * EV_PER_A_TO_NN[family]


def drive_step(window: pd.DataFrame, speed: float, fields: Dict[str, Any]) -> Tuple[float, float]:
    """``(A per MD step, dx)``: the nominal drive distance per step and per
    window sample (converter :91-92). NaN when the speed or timestep is missing."""
    a_per_step = float(speed) / 100.0 * _number(fields, 'settings.simulation.timestep')
    return a_per_step, float(np.median(np.diff(window['time'].to_numpy(float)))) * a_per_step


def run_periods(family: str, frames: Dict[str, pd.DataFrame], speed: float,
                fields: Dict[str, Any], period_x: float) -> Dict[str, Any]:
    """The whole-period columns of one 0 degree run.

    ``dx`` is the nominal drive distance per window sample, the converter's
    ``median(diff(time)) * speed / 100 * timestep`` (:91-92), which for the
    AFM campaign is taxlib's ``speed * 0.01 * 1 ps`` (taxlib_adapter.py:39-40)
    to the bit. ``fft_period_A`` is the raw tier's dominant ``raw_lfx``
    wavelength against the nominal drive (converter :99, :102-103), NaN
    without a raw tier.
    """
    window = frames['window']
    lf, nf, cof = _forces(family, window)
    n = len(cof)
    a_per_step, dx = drive_step(window, speed, fields)
    k_spring = _spring(fields, family)
    n_skip = settle_skip(window['lfx'].to_numpy(float), k_spring, dx)
    period = period_x / dx
    k, start = whole_periods(n, n_skip, period)
    out = {'n_rows': n, 'dx_A': dx, 'period_x_A': period_x, 'period_samples': period,
           'spring_nN_A': k_spring, 'n_skip': n_skip, 'n_periods': k, 'start_row': start,
           'cof_periods': _mean(cof[start:], family),
           'lf_periods': float(lf[start:].mean()), 'nf_periods': float(nf[start:].mean()),
           'cof_full_run': _mean(cof, family)}
    raw = frames.get('raw')
    fft = float('nan')
    if raw is not None and {'time', 'raw_lfx'} <= set(raw.columns):
        fft = dominant_period(raw['time'].to_numpy(float) * a_per_step,
                              raw['raw_lfx'].to_numpy(float))
    out.update(fft_period_A=fft, fft_ratio=fft / period_x)
    if family == 'sheetonsheet' and {'react_x', 'react_y'} <= set(window.columns):
        # The reaction on the fixed bottom layer, the plotter's lf_source sx_sy
        # (converter :18-20, :81).
        out['lf_sx_sy_periods'] = float(np.hypot(window['react_x'].to_numpy(float),
                                                 window['react_y'].to_numpy(float))[start:].mean())
    return out


def material_key(materials: Any) -> str:
    """The material name `sheet_geometry` keys its rows by (s42:138)."""
    from .sheet_geometry import _material  # pylint: disable=import-outside-toplevel,protected-access
    first = materials if isinstance(materials, str) else list(materials)[0]
    return _material(first)


def _fmt(value: Any) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f'{value:g}'


def summary_identity(row: Any) -> Dict[str, Any]:
    """The identity columns of a summary row, spelled as
    ``PlotCommonMixin._dataset_to_summary_rows`` spells them
    (plot_common.py:313-333): ``id``/``material`` underscored, ``size`` as
    ``100x100y``, ``force`` or ``pressure`` by ``load_kind``, ``layer``."""
    materials = getattr(row, 'materials', None)
    materials = '+'.join([materials] if isinstance(materials, str) else list(materials or ()))
    flake = getattr(row, 'flake_material', None)
    material = f'{materials}+flake-{flake}' if isinstance(flake, str) and flake else materials
    material = material.replace('-', '_')
    sx, sy = getattr(row, 'size_x', None), getattr(row, 'size_y', None)
    size = (f'{_fmt(sx)}x{_fmt(sy)}y' if sx is not None and sy is not None
            and not (pd.isna(sx) or pd.isna(sy)) else 'unknownx')
    kind = getattr(row, 'load_kind', None)
    load = float(row.load)
    layers = getattr(row, 'layers', None)
    return {'run_uuid': getattr(row, 'run_uuid', None), 'point_uuid': row.point_uuid,
            'id': material, 'material': material, 'size': size,
            'layer': None if layers is None or pd.isna(layers) else int(layers),
            'load': load, 'load_kind': kind,
            'force': load if kind == 'force' else float('nan'),
            'pressure': load if kind == 'pressure' else float('nan'),
            'angle': float(row.angle), 'speed': float(row.speed)}


def _json_before(value: Any) -> Any:
    """A JSON text (a CLI ``--param``) as its value."""
    return json.loads(value) if isinstance(value, str) else value


class PeriodParams(BaseModel):
    """Where ``a_x`` comes from.

    ``period_x``: material -> ``period_x_top`` (A), used as given. Without it,
    the stored `sheet_geometry` result of ``geometry_dataset``, else of the
    analysed dataset, supplies it (run, or reused, through
    :func:`~frictionsim2d.analysis.store.result_node`); that result is part of the
    analysis' identity (:func:`geometry_depends`). A source that is not a
    stored dataset (a test's) computes `sheet_geometry` from itself.
    Each material takes its row of layer :data:`GEOMETRY_LAYER`, else layer
    1, else its highest (a sheet stack's). ``lattice``: material -> A, the
    fallback for a material with no geometry row (the database ``lat_a`` of
    rank_cof_v2.py:73, 89). A run whose material has neither is dropped
    (:data:`NO_PERIOD`): it has no whole periods.
    """

    period_x: Optional[Dict[str, float]] = None
    geometry_dataset: Optional[str] = None
    lattice: Dict[str, float] = {}

    @field_validator('period_x', 'lattice', mode='before')
    @classmethod
    def _parse(cls, value: Any) -> Any:
        return _json_before(value)


def period_map(source: Any, params: PeriodParams) -> Dict[str, float]:
    """material -> ``a_x`` (A): the geometry's ``period_x_top`` over the
    ``lattice`` fallback (rank_cof_v2.py:72-73)."""
    period = dict(params.lattice)
    if params.period_x is not None:
        period.update(params.period_x)
        return period
    node = (getattr(source, 'depends', None) or {}).get(GEOMETRY)
    if node is None:
        node = geometry_depends(getattr(source, 'dataset', None), params).get(GEOMETRY)
    if node is not None:
        from .store import table_from_csv  # pylint: disable=import-outside-toplevel
        geometry = table_from_csv(node.get_content(mode='rb'))
    else:
        geometry = _own_geometry(source)
    if not geometry.empty:
        order = {GEOMETRY_LAYER: 0, 1: 1}
        ranked = geometry.assign(_order=[(order.get(int(l), 2), -int(l))
                                         for l in geometry['layer']])
        best = ranked.sort_values('_order', kind='stable').drop_duplicates('material')
        period.update(best.set_index('material')['period_x_top'].astype(float).to_dict())
    return period


#: The name a period analysis keys the `sheet_geometry` result it reads by
#: (:meth:`~frictionsim2d.analysis.base.Analysis.depends_on`).
GEOMETRY = 'sheet_geometry'


def geometry_depends(dataset: Optional[str], params: PeriodParams) -> Dict[str, Any]:
    """``{'sheet_geometry': node}``: the stored `sheet_geometry` result that
    supplies ``a_x`` -- of ``params.geometry_dataset``, else of ``dataset`` --
    run or reused (:func:`~frictionsim2d.analysis.store.result_node`). Empty when
    ``params.period_x`` gives ``a_x`` (nothing is read) or there is no dataset."""
    target = params.geometry_dataset or dataset
    if params.period_x is not None or target is None:
        return {}
    from .store import result_node  # pylint: disable=import-outside-toplevel
    return {GEOMETRY: result_node(GEOMETRY, target)}


class PeriodDepends:  # pylint: disable=too-few-public-methods
    """Mixin of the analyses that read ``a_x`` (:class:`PeriodParams`): the
    `sheet_geometry` result they read is part of their identity."""

    @classmethod
    def depends_on(cls, dataset: str, params: PeriodParams) -> Dict[str, Any]:
        """The `sheet_geometry` result ``a_x`` comes from (:func:`geometry_depends`)."""
        return geometry_depends(dataset, params)


class _GeometryView:  # pylint: disable=too-few-public-methods
    """Every point of the source (`sheet_geometry` reads ``all_rows``), with
    drops kept apart: a material without a structure is handled by
    :func:`period_rows` here, not dropped twice."""

    def __init__(self, source: Any) -> None:
        self.all_rows = source.all_rows
        self.dropped: List[Tuple[str, str]] = []
        self.structure_text = source.structure_text
        self.settings_text = source.settings_text


def _own_geometry(source: Any) -> pd.DataFrame:
    from .sheet_geometry import SheetGeometry  # pylint: disable=import-outside-toplevel
    return SheetGeometry().compute(_GeometryView(source), SheetGeometry.Params())


def period_rows(source: Any, rows: pd.DataFrame, period: Dict[str, float]) -> List[Dict[str, Any]]:
    """One :func:`run_periods` record (with its summary identity) per row of
    ``rows`` (pass rows at 0 degrees). A row with no ``a_x`` (:data:`NO_PERIOD`)
    or no drive step (:data:`NO_DRIVE_STEP`) is dropped, never averaged."""
    out = []
    missing = set()
    for row in rows.itertuples():
        key = material_key(row.materials)
        a_x = float(period.get(key, float('nan')))
        if not np.isfinite(a_x):
            missing.add(key)
            source.dropped.append((str(row.point_uuid), NO_PERIOD))
            continue
        frames = source.frames(row)
        fields = source.run_fields(row)
        if not np.isfinite(drive_step(frames['window'], row.speed, fields)[1]):
            source.dropped.append((str(row.point_uuid), NO_DRIVE_STEP))
            continue
        out.append({**summary_identity(row),
                    **run_periods(source.family, frames, row.speed, fields, a_x)})
    for key in sorted(missing):
        logger.warning('%s: no period_x_top and no lattice fallback; its runs are dropped', key)
    return out


@register
class PeriodSummary(PeriodDepends, Analysis):  # pylint: disable=too-few-public-methods
    """Per run: the COF, LF and NF over its whole lattice periods.

    Summary-shaped: ``id, material, size, layer, force, pressure, angle``
    as a summary row has them, and ``cof``, ``lf``, ``nf`` (plus
    ``lf_sx_sy`` for the sheet family) aliasing the period means, so a plot
    reads it as it reads summaries.
    """

    name = 'period_summary'
    version = '3'
    level = 'run'
    families = frozenset({'afm', 'sheetonsheet'})
    Params = PeriodParams

    def compute(self, source, params) -> pd.DataFrame:
        """The pass rows at 0 degrees; any other angle is dropped
        (:data:`OFF_AXIS`)."""
        rows = source.rows
        on_axis = rows['angle'].astype(float) == 0.0
        source.dropped.extend((str(r.point_uuid), OFF_AXIS) for r in rows[~on_axis].itertuples())
        table = pd.DataFrame(period_rows(source, rows[on_axis], period_map(source, params)))
        if table.empty:
            return table
        table['cof'] = table['cof_periods']
        table['lf'] = table['lf_periods']
        table['nf'] = table['nf_periods']
        if 'lf_sx_sy_periods' in table:
            table['lf_sx_sy'] = table['lf_sx_sy_periods']
        return table
