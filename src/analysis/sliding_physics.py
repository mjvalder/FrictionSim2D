"""Physical scales, an energy-balance residual and sliding descriptors of each
AFM or sheet-on-sheet run. One row per run.

Ports of ML_taxonomy's taxlib, value by value (each function cites its
source), on the window tier (and, for ``e_int_rel``, the peratom tier):

- scales (``taxlib/scales.py``): ``K_fit``, ``K_fit_r2`` (``spring_fit``
  :10-19), ``K_nN_A``/``K_source`` (``_pick_K`` :38-47), ``F_N_nN``/``F_N_source``
  (:63), ``sigma_th_nN`` and ``W_fric_eV`` (:65-70), ``n_skip`` (:22-27,
  :func:`.periods.settle_skip`);
- physics (``taxlib/physics.py``): ``spring_K_ratio`` (:56),
  ``energy_resid_rel`` (``_energy`` :26-42, :59);
- features (``taxlib/features.py``): ``cof_mean`` (:61), ``stick_frac`` (:71),
  ``sheet_carried`` (:70), ``il_slip_frac`` (:86), ``ageing`` (``lat_growth_rel``
  :62), ``e_int_rel`` (:129);
- phenomena (``taxlib/phenomena.py``): ``cof_se`` (``_se`` :18-23),
  ``superlubric_margin`` (:60), ``lattice_periodicity`` (``_periodicity`` :26-38).
  ``superlubric_margin`` is computed as |mean lf| / its block standard error:
  taxlib's |cof_mean| / cof_se with the load cancelled, so it is defined where
  ``F_N_nN`` is NaN (a pressure-loaded sheet run) and cof_mean, cof_se are not.

Units: forces nN, lengths A, energies eV, K in nN/A, T in K, stress GPa.

The shear strength ``shear_strength_GPa`` = ``lf_mean_nN`` / ``contact_area_A2``
is a join with the newest ``contact`` result stored for the same dataset
(its ``load`` state, matched on ``run_uuid``, ``load`` and ``angle``; a run
with no matching row is NaN with a ``nan_reason``). That result is part of
this one's identity (:meth:`SlidingPhysics.depends_on`): it is linked as an
input, and a stored sliding_physics result is reused only while it is the
same one. Without one the two columns are left out and the result carries a
note; contact is never computed here. Contact is AFM-only (tip-sheet), so a
sheet-on-sheet result never has them.

The sheet-on-sheet family has no tip: a value that reads a tip channel
(``tipx``, ``tipz``, ``w_thermostat_tip``/``_sub``, ``E_int_tip_sheet``) is NaN
there, and ``nan_reason`` names it. So is a value that needs a missing
channel, lattice period or load in any family.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import field_validator

from .base import Analysis, register
from .periods import (PeriodParams, _json_before, _spring, drive_step, geometry_depends,
                      material_key, period_map, settle_skip, summary_identity)

__all__ = ['SlidingPhysics', 'SlidingPhysicsParams', 'ageing', 'block_se', 'body_start',
           'e_int_rel', 'energy_residual', 'friction_work', 'il_slip_frac',
           'lattice_periodicity', 'pick_k', 'sheet_carried', 'shear_strength_gpa',
           'sigma_th', 'spring_fit', 'stick_frac', 'superlubric_margin']

#: Boltzmann constant, eV/K (taxlib/constants.py:2).
KB_EV = 8.617333262e-5
#: 1 eV/A = 1.602 nN; 1 eV = 1.602 nN A (taxlib/constants.py:3).
EV_TO_NN_A = 1.602176634
#: Sign of the thermostat work in the energy balance (taxlib/constants.py:11).
THERMO_SIGN = 1.0
#: Below this stretch range (A) the spring fit is unidentifiable (scales.py:7).
MIN_STRETCH_RANGE = 0.5
#: The channels of the energy balance (physics.py:23), beside lfx and tipz.
ENERGY_COLS = ('etotal', 'w_thermostat_tip', 'w_thermostat_sub')
#: Samples after the transient below which the whole run is the body
#: (features.py:54, phenomena.py:46).
MIN_BODY = 100
#: Samples averaged at each end for ``ageing`` (features.py:62).
END_SAMPLES = 100
#: Blocks of the standard error (phenomena.py:18) and the fewest samples per block (:20).
SE_BLOCKS = 10
SE_MIN_PER_BLOCK = 5
#: A sample sticks when the tip advances by less than this many drive steps (features.py:71).
STICK_DX = 0.1
#: The lattice band of ``lattice_periodicity``, in 1/a (phenomena.py:36), and its grid (:31).
BAND = (0.75, 1.25)
PERIODICITY_GRID = 512
#: The channels only a tip run has.
TIP_CHANNELS = frozenset({'tipx', 'tipz', 'w_thermostat_tip', 'w_thermostat_sub',
                          'E_int_tip_sheet'})
#: 1 nN/A^2 = 1e-9 N / 1e-20 m^2 = 1e11 Pa = 100 GPa.
NN_PER_A2_TO_GPA = 100.0
#: The drop reason of a run without a window tier.
NO_WINDOW = 'no_window_tier'
#: The note of a result over a dataset with no stored ``contact`` result.
NO_CONTACT_NOTE = ('shear_strength_GPa, contact_area_A2: left out -- no stored contact '
                   'result for dataset {dataset!r} (run `analyse contact` first)')
#: The note of a sheet-on-sheet result.
SHEET_CONTACT_NOTE = ('shear_strength_GPa, contact_area_A2: left out -- contact is AFM-only '
                      '(tip-sheet contact), so a sheetonsheet dataset has none')

NAN = float('nan')


# --------------------------------------------------------------------------
# scales.py
# --------------------------------------------------------------------------

def spring_fit(lfx: np.ndarray, tipx: np.ndarray, dx: float) -> Tuple[float, float]:
    """``(K nN/A, r^2)`` of ``lfx = K (dx * step - (tipx - tipx0)) + c``
    (scales.py:10-19); NaN when the stretch varies by less than
    :data:`MIN_STRETCH_RANGE` or a value is not finite."""
    lfx = np.asarray(lfx, dtype=float)
    tipx = np.asarray(tipx, dtype=float)
    s = dx * np.arange(len(lfx), dtype=float) - (tipx - tipx[0])
    if np.ptp(s) < MIN_STRETCH_RANGE or not np.isfinite(s).all() or not np.isfinite(lfx).all():
        return NAN, NAN
    k, c = np.polyfit(s, lfx, 1)
    ss = float(((lfx - lfx.mean()) ** 2).sum())
    return float(k), (1.0 - float(((lfx - (k * s + c)) ** 2).sum()) / ss) if ss > 0 else NAN


def pick_k(provenance: float, run_fit: float) -> Tuple[float, str]:
    """``(K, source)``: the recorded spring, else this run's positive fit
    (scales.py:38-47; taxlib's middle choice, the material's median fit, is
    campaign-wide and left out)."""
    if provenance is not None and np.isfinite(provenance):
        return float(provenance), 'provenance'
    if np.isfinite(run_fit) and run_fit > 0:
        return float(run_fit), 'fit'
    return NAN, 'missing'


def sigma_th(k: float, temperature: float) -> float:
    """Thermal force noise sqrt(K kB T), nN (scales.py:65, :69): K nN/A, T K."""
    if not np.isfinite(k):
        return NAN
    return float(np.sqrt(k * KB_EV * temperature * EV_TO_NN_A))


def friction_work(lfx: np.ndarray, length: float) -> float:
    """``mean(lfx) * L`` in eV (scales.py:70): lfx nN, L A."""
    return float(np.mean(lfx)) * length / EV_TO_NN_A


# --------------------------------------------------------------------------
# physics.py
# --------------------------------------------------------------------------

def energy_residual(w: pd.DataFrame, f_n: float, dx: float, w_fric: float,
                    w_floor: float, k: float) -> float:
    """End residual of the energy balance over max(|W_fric|, floor), 1
    (physics.py:26-42): ``dE_tot - W_drive - W_load + THERMO_SIGN dW_thermostat
    + dE_spring``, all eV. NaN without :data:`ENERGY_COLS` or a scale."""
    scale = max(abs(w_fric), w_floor) if np.isfinite(w_floor) else abs(w_fric)
    if not all(c in w.columns for c in ENERGY_COLS) or not np.isfinite(scale) or scale < 1e-9:
        return NAN
    e = w['etotal'].to_numpy(float)
    wth = (w['w_thermostat_tip'] + w['w_thermostat_sub']).to_numpy(float)
    lf = w['lfx'].to_numpy(float)
    w_drive = np.cumsum(lf) * dx / EV_TO_NN_A
    tz = w['tipz'].to_numpy(float)
    w_load = f_n * (tz[0] - tz) / EV_TO_NN_A
    # the drive spring's 1/2 K s^2 = F^2 / (2K) is not in etotal (physics.py:39-40)
    e_spring = (lf ** 2 - lf[0] ** 2) / (2 * k) / EV_TO_NN_A if np.isfinite(k) and k > 0 else 0.0
    r = (e - e[0]) - w_drive - w_load + THERMO_SIGN * (wth - wth[0]) + e_spring
    return float(r[-1] / scale)


# --------------------------------------------------------------------------
# features.py / phenomena.py
# --------------------------------------------------------------------------

def body_start(n: int, n_skip: int) -> int:
    """First sample after the transient, or 0 when fewer than
    :data:`MIN_BODY` would be left (features.py:54, phenomena.py:46)."""
    return int(n_skip) if n - int(n_skip) >= MIN_BODY else 0


def block_se(lf: np.ndarray, f_n: float) -> float:
    """Standard error of the mean friction over :data:`SE_BLOCKS` blocks,
    over the load, 1 (phenomena.py:18-23)."""
    lf = np.asarray(lf, dtype=float)
    n = (len(lf) // SE_BLOCKS) * SE_BLOCKS
    if n < SE_BLOCKS * SE_MIN_PER_BLOCK:
        return NAN
    m = lf[:n].reshape(SE_BLOCKS, -1).mean(1)
    return float(m.std(ddof=1) / np.sqrt(SE_BLOCKS) / f_n)


def superlubric_margin(lf: np.ndarray) -> float:
    """|cof_mean| / cof_se (phenomena.py:60), 1. Both divide by the load, so
    it is |mean lf| / its block standard error: no load is needed."""
    lf = np.asarray(lf, dtype=float)
    se = block_se(lf, 1.0)
    return float(abs(lf.mean()) / se) if np.isfinite(se) and se > 0 else NAN


def stick_frac(tipx: np.ndarray, dx: float) -> float:
    """Share of samples whose tip advance is under ``STICK_DX * dx``, 1 (features.py:71)."""
    return float((np.diff(np.asarray(tipx, dtype=float)) < STICK_DX * dx).mean())


def _advance(x: np.ndarray) -> float:
    return float(x[-1] - x[0])


def sheet_carried(comx: np.ndarray, tipx: np.ndarray, a: float) -> float:
    """Top-layer advance / tip advance, 1 (features.py:70); NaN when the tip
    advanced no more than ``a``."""
    adv = _advance(np.asarray(tipx, dtype=float))
    return _advance(np.asarray(comx, dtype=float)) / adv if abs(adv) > a else NAN


def il_slip_frac(comx_l1: np.ndarray, comx_l2: np.ndarray, tipx: np.ndarray, a: float) -> float:
    """Layer-2-over-layer-1 slip / tip advance, 1 (features.py:83-86)."""
    adv = _advance(np.asarray(tipx, dtype=float))
    slip = np.asarray(comx_l2, dtype=float) - np.asarray(comx_l1, dtype=float)
    return _advance(slip) / adv if abs(adv) > a else NAN


def lattice_periodicity(tipx: np.ndarray, lfx: np.ndarray, a: float) -> float:
    """Share of the detrended lateral-force power (against tip position)
    within :data:`BAND` of 1/a, 1 (phenomena.py:26-38)."""
    x, f = np.asarray(tipx, dtype=float), np.asarray(lfx, dtype=float)
    if not np.isfinite(a) or a <= 0 or (x.max() - x.min()) < 2 * a:
        return NAN
    order = np.argsort(x)
    grid = np.linspace(x.min(), x.max(), PERIODICITY_GRID)
    y = np.interp(grid, x[order], f[order])
    y = y - np.polyval(np.polyfit(grid, y, 1), grid)
    p = np.abs(np.fft.rfft(y)) ** 2
    k = np.fft.rfftfreq(len(grid), d=grid[1] - grid[0])
    band = (k > BAND[0] / a) & (k < BAND[1] / a)
    tot = p[1:].sum()
    return float(p[band].sum() / tot) if tot > 0 else NAN


def ageing(lf: np.ndarray, sigma: float) -> float:
    """(late - early mean friction) / |mean|, 1 (features.py:62, phenomena.py:58);
    NaN when the mean is under ``sigma_th / 10``."""
    lf = np.asarray(lf, dtype=float)
    mean = abs(lf.mean())
    if not mean > sigma / 10:
        return NAN
    return float((lf[-END_SAMPLES:].mean() - lf[:END_SAMPLES].mean()) / mean)


def e_int_rel(e_int: np.ndarray, f_n: float, a: float) -> float:
    """Mean tip-sheet interaction energy / (F_N a), 1 (features.py:129):
    E eV, F_N nN, a A."""
    return float(np.mean(e_int)) * EV_TO_NN_A / (f_n * a)


def shear_strength_gpa(lf: float, area: float) -> float:
    """LF / contact area in GPa: LF nN, area A^2. NaN for no contact."""
    if not np.isfinite(area) or area <= 0:
        return NAN
    return float(lf / area * NN_PER_A2_TO_GPA)


# --------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------

class SlidingPhysicsParams(PeriodParams):
    """The angle read (taxlib reads 0 degree runs, taxlib_adapter.py:35),
    where ``a_x`` comes from (:class:`~.periods.PeriodParams`), and
    ``taxlib_a``: material -> the lattice ``a`` taxlib scales by (its
    materials database ``lat_a``, taxlib_adapter.py:46-51); a material not in
    it takes ``a_x``."""

    angle: float = 0.0
    taxlib_a: Dict[str, float] = {}

    @field_validator('taxlib_a', mode='before')
    @classmethod
    def _parse_taxlib(cls, value: Any) -> Any:
        return _json_before(value)


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def load_scale(row: Any) -> Tuple[float, str]:
    """``(F_N nN, source)``: the achieved load, else the requested force
    (scales.py:63). A pressure run has no force: NaN."""
    achieved = _finite(getattr(row, 'load_achieved', None))
    if achieved is not None:
        return achieved, 'provenance'
    if getattr(row, 'load_kind', None) == 'force':
        return float(row.load), 'requested'
    return NAN, 'missing'


def _top_com(window: pd.DataFrame, layers: Any) -> Optional[str]:
    """The top layer's COM x column: ``comx_L<layers>``, else ``comx``
    (taxlib's ``comx`` is the top layer's, ingest_v2.split_window)."""
    if layers is not None and not pd.isna(layers) and f'comx_L{int(layers)}' in window:
        return f'comx_L{int(layers)}'
    return 'comx' if 'comx' in window else None


class _Row:  # pylint: disable=too-few-public-methods
    """One run's values and the reasons the missing ones are NaN."""

    def __init__(self, family: str, channels: Sequence[str]) -> None:
        self.family = family
        self.channels = set(channels)
        self.values: Dict[str, Any] = {}
        self.reasons: List[str] = []

    def put(self, column: str, needs: Sequence[str], value: Any, extra: Sequence[str] = ()) -> None:
        """``column`` = ``value()`` when the run has every channel of
        ``needs`` and nothing in ``extra`` (a missing scale) is listed; else NaN
        with its reason."""
        missing = [c for c in needs if c not in self.channels]
        if missing:
            tip = self.family == 'sheetonsheet' and set(missing) <= TIP_CHANNELS
            self.reasons.append(f"{column}: no {','.join(missing)}"
                                + (f' ({self.family} has no tip)' if tip else ''))
            self.values[column] = NAN
        elif extra:
            self.reasons.append(f"{column}: no {','.join(extra)}")
            self.values[column] = NAN
        else:
            self.values[column] = value()


def run_values(family: str, frames: Dict[str, pd.DataFrame], row: Any,
               fields: Dict[str, Any], a: float) -> Dict[str, Any]:
    """The columns of one run (see the module docstring)."""
    # pylint: disable=too-many-locals
    w = frames['window'].sort_values('time').reset_index(drop=True)
    peratom = frames.get('peratom')
    n = len(w)
    lfx = w['lfx'].to_numpy(float)
    dx = drive_step(w, row.speed, fields)[1]
    length = dx * n
    f_n, f_src = load_scale(row)
    temperature = _finite(getattr(row, 'temperature', None))
    temperature = NAN if temperature is None else temperature
    out = _Row(family, [*w.columns, *([] if peratom is None else peratom.columns)])
    k_fit, r2 = (spring_fit(lfx, w['tipx'].to_numpy(float), dx)
                 if 'tipx' in w and np.isfinite(dx) else (NAN, NAN))
    out.put('K_fit', ('tipx',), lambda: k_fit)
    out.put('K_fit_r2', ('tipx',), lambda: r2)
    k, k_src = pick_k(_spring(fields, family), k_fit)
    sig = sigma_th(k, temperature)
    w_fric = friction_work(lfx, length)
    n_skip = settle_skip(lfx, k, dx)
    lf = lfx[body_start(n, n_skip):]
    no_a = () if np.isfinite(a) else ('lattice period',)
    no_f = () if np.isfinite(f_n) else ('force load',)
    out.values.update(K_nN_A=k, K_source=k_src, F_N_nN=f_n, F_N_source=f_src, T_K=temperature,
                      dx_A=dx, L_A=length, a_A=a, n_skip=n_skip, sigma_th_nN=sig,
                      W_fric_eV=w_fric, lf_mean_nN=float(lf.mean()))
    out.put('spring_K_ratio', ('tipx',),
            lambda: k_fit / k if np.isfinite(k_fit) and k else NAN)
    out.put('energy_resid_rel', ('tipz', *ENERGY_COLS),
            lambda: energy_residual(w, f_n, dx, w_fric, sig * length / EV_TO_NN_A, k), no_f)
    out.put('cof_mean', (), lambda: float(lf.mean()) / f_n, no_f)
    out.put('cof_se', (), lambda: block_se(lf, f_n), no_f)
    out.values['superlubric_margin'] = superlubric_margin(lf)
    start = body_start(n, n_skip)
    out.put('stick_frac', ('tipx',), lambda: stick_frac(w['tipx'].to_numpy(float)[start:], dx))
    top = _top_com(w, getattr(row, 'layers', None))
    out.put('sheet_carried', ('tipx', top or 'comx'),
            lambda: sheet_carried(w[top].to_numpy(float), w['tipx'].to_numpy(float), a), no_a)
    out.put('il_slip_frac', ('tipx', 'comx_L1', 'comx_L2'),
            lambda: il_slip_frac(w['comx_L1'].to_numpy(float), w['comx_L2'].to_numpy(float),
                                 w['tipx'].to_numpy(float), a), no_a)
    out.put('lattice_periodicity', ('tipx',),
            lambda: lattice_periodicity(w['tipx'].to_numpy(float)[start:], lf, a), no_a)
    out.values['ageing'] = ageing(lf, sig)
    out.put('e_int_rel', ('E_int_tip_sheet',),
            lambda: e_int_rel(peratom['E_int_tip_sheet'].to_numpy(float), f_n, a),
            no_f + no_a)
    out.values['nan_reason'] = '; '.join(out.reasons)
    return out.values


def contact_result(dataset: str) -> Any:
    """The newest stored ``contact`` result node of ``dataset``, or None."""
    from .store import find_result_node  # pylint: disable=import-outside-toplevel
    return find_result_node('contact', dataset)


def _node_table(node: Any) -> pd.DataFrame:
    from .store import _from_csv  # pylint: disable=import-outside-toplevel,protected-access
    return _from_csv(node.get_content(mode='rb'))


CONTACT_KEYS = ('run_uuid', 'load', 'angle')


def join_contact(table: pd.DataFrame, contact: pd.DataFrame) -> pd.DataFrame:
    """``contact_area_A2`` of the matching ``load``-state contact row and
    ``shear_strength_GPa`` = ``lf_mean_nN`` / it; a row with no match (or no
    contact area) gets a ``nan_reason``.

    Raises:
        ValueError: the contact table has two ``load`` rows for one
            ``(run_uuid, load, angle)``.
    """
    keys = list(CONTACT_KEYS)
    loaded = contact[contact['state'] == 'load'][keys + ['contact_area']]
    duplicated = loaded[loaded.duplicated(keys, keep=False)]
    if not duplicated.empty:
        raise ValueError(f'contact result has duplicate load rows for '
                         f'{sorted(set(map(tuple, duplicated[keys].to_numpy().tolist())))}')
    loaded = loaded.rename(columns={'contact_area': 'contact_area_A2'})
    out = table.merge(loaded, on=keys, how='left')
    areas = out['contact_area_A2'].astype(float)
    out['shear_strength_GPa'] = [shear_strength_gpa(lf, area)
                                 for lf, area in zip(out['lf_mean_nN'], areas)]
    reasons = []
    for reason, area in zip(out['nan_reason'].fillna(''), areas):
        extra = ('shear_strength_GPa: no contact row for (run_uuid, load, angle)'
                 if np.isnan(area) else
                 'shear_strength_GPa: no contact area' if area <= 0 else '')
        reasons.append('; '.join(r for r in (reason, extra) if r))
    out['nan_reason'] = reasons
    return out


@register
class SlidingPhysics(Analysis):  # pylint: disable=too-few-public-methods
    """taxlib's per-run scales and sliding descriptors, plus the shear
    strength from a stored contact result."""

    name = 'sliding_physics'
    version = '3'
    level = 'run'
    families = frozenset({'afm', 'sheetonsheet'})
    Params = SlidingPhysicsParams

    @classmethod
    def depends_on(cls, dataset: str, params) -> Dict[str, Any]:
        """The newest stored ``contact`` result of ``dataset`` (None: none)
        and the `sheet_geometry` result ``a_x`` comes from
        (:func:`~.periods.geometry_depends`)."""
        return {**geometry_depends(dataset, params), 'contact': contact_result(dataset)}

    def compute(self, source, params) -> pd.DataFrame:
        """One row per pass run at ``angle``; a run with no window tier is
        dropped (:data:`NO_WINDOW`).

        Raises:
            ValueError: the source is not an AFM or sheet-on-sheet family.
        """
        if source.family not in self.families:
            raise ValueError(f'sliding_physics reads families {sorted(self.families)}, '
                             f'not {source.family!r}')
        rows = source.rows
        rows = rows[rows['angle'].astype(float) == float(params.angle)]
        period = period_map(source, params)
        out: List[Dict[str, Any]] = []
        for row in rows.itertuples():
            frames = source.frames(row)
            if 'window' not in frames:
                source.dropped.append((str(row.point_uuid), NO_WINDOW))
                continue
            key = material_key(row.materials)
            a = float(params.taxlib_a.get(key, period.get(key, NAN)))
            out.append({**summary_identity(row),
                        **run_values(source.family, frames, row, source.run_fields(row), a)})
        table = pd.DataFrame(out)
        return self._join(source, table)

    def _join(self, source, table: pd.DataFrame) -> pd.DataFrame:
        """``table`` with the shear strength of the contact result the
        source was handed (``source.depends``), else looked up for its dataset."""
        if source.family == 'sheetonsheet':
            source.notes.append(SHEET_CONTACT_NOTE)
            return table
        depends = getattr(source, 'depends', None)
        dataset = getattr(source, 'dataset', None)
        if depends is None:
            depends = {'contact': contact_result(dataset)} if dataset is not None else {}
        node = depends.get('contact')
        contact = None if node is None else _node_table(node)
        if contact is None or contact.empty:
            source.notes.append(NO_CONTACT_NOTE.format(dataset=dataset))
        elif not table.empty:
            table = join_contact(table, contact)
        return table
