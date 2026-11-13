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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.core.config import IntegritySettings

from .run_status import UNOBSERVED

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
    A sheet window with fewer than two rows is not evaluated (s51 :89): every
    measure, ``non_finite`` and ``frozen`` included, is NaN.
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
    if family in SHEET_LIKE and len(window) < 2:
        # s51 :89 evaluates a point's series only when it has more than one
        # row; NaN here reads as "not applicable", never as a pass.
        return {'non_finite': nan, 'frozen': nan, 'T_sigma': nan,
                'gap_rise': nan, 'gap_min_any': nan}

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


# --------------------------------------------------------------------------
# Verdict (s48 ``verdicts`` :152, s51 ``verdicts`` :138 + the variant :172)
# --------------------------------------------------------------------------

#: Statuses whose point never produced a window: failed, and nothing observed.
_UNOBSERVED = UNOBSERVED

#: (rule, measure key, IntegritySettings attribute, unit), in s48's order
#: (:159-172) with s51's ``collapse`` (:157) last. ``None`` attribute: the
#: rule is always on. Boolean rules fire on True, ``collapse`` on a negative
#: gap, every other rule on ``value >= threshold``.
_RULES = (
    ('non_finite', 'non_finite', None, ''),
    ('frozen', 'frozen', 'frozen_std', ''),
    ('hole', 'hole', 'hole_bonds', ' bonds'),
    ('isolated_atoms', 'isolated', 'isolated_atoms', ''),
    ('ejected_atoms', 'ejected', 'ejected_atoms', ''),
    ('sheet_heating', 'T_sigma', 'heating_sigma', ' sigma_T'),
    ('gap_rise', 'gap_rise', 'gap_rise', ''),
    ('collapse', 'gap_min_any', 'collapse', ''),
)
_BOOLEAN_RULES = ('non_finite', 'frozen')

MULTI_SPEED_DETAIL = 'snapshot rules skipped: multi-speed run'
NO_SNAPSHOT_DETAIL = 'snapshot rules not applicable: run writes no end snapshot'
#: The descriptive (non-rule-name) entries ``check_files`` puts in ``not_applicable``.
SKIP_DETAILS = (MULTI_SPEED_DETAIL, NO_SNAPSHOT_DETAIL)


@dataclass
class IntegrityVerdict:  # pylint: disable=too-many-instance-attributes
    """Stage-3 outcome of one load point.

    ``status`` is ``complete``, ``failed_integrity`` or ``quarantined`` for a
    complete input and the input status otherwise. ``reasons`` are the fired
    rules (``'hole 3.6 bonds'``), preceded by the gate status for a point that
    did not complete (s48 :158). ``not_applicable`` names the rules whose
    measure is NaN or absent, plus any skip detail.
    """
    status: str
    failed: bool
    observed: bool
    reasons: List[str] = field(default_factory=list)
    quarantine: List[str] = field(default_factory=list)
    not_applicable: List[str] = field(default_factory=list)
    measures: Dict[str, float] = field(default_factory=dict)
    thresholds_version: str = ''

    def as_dict(self) -> dict:
        """JSON-safe form: NaN and infinite measures become ``None``."""
        out = asdict(self)
        out['measures'] = {k: _json_value(v) for k, v in self.measures.items()}
        return out


def _json_value(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if np.isfinite(f) else None


def _missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return True


def _fires(rule: str, value, threshold) -> bool:
    if rule in _BOOLEAN_RULES:
        return bool(value)
    if rule == 'collapse':
        return float(value) < 0
    return float(value) >= threshold


def _reason(rule: str, value, unit: str) -> str:
    if rule in _BOOLEAN_RULES:
        return rule
    return f'{rule} {float(value):.3g}{unit}'


def verdict(status: str, measures: Dict[str, float], settings: IntegritySettings,
            *, quarantine: Sequence[str] = (),
            not_applicable: Sequence[str] = ()) -> IntegrityVerdict:
    """Apply ``settings`` to one point's ``measures`` (pure).

    ``quarantine`` and ``not_applicable`` carry what the caller already knows
    (a missing snapshot, a skipped rule set). A margin ``m`` on a rule (keyed
    by rule name or measure key) quarantines a complete point whose value is
    in ``[m * threshold, threshold)``; a NaN measure is not applicable, never
    a pass.
    """
    reasons: List[str] = [] if status == 'complete' else [status]
    quarantined = list(quarantine)
    skipped: List[str] = []
    for rule, key, attr, unit in _RULES:
        threshold = True if attr is None else getattr(settings, attr)
        if threshold is None or threshold is False:
            continue
        value = measures.get(key)
        if _missing(value):
            skipped.append(rule)
            continue
        if _fires(rule, value, threshold):
            reasons.append(_reason(rule, value, unit))
            continue
        margin = settings.margins.get(rule, settings.margins.get(key))
        if (margin is not None and rule not in _BOOLEAN_RULES and rule != 'collapse'
                and margin * threshold <= float(value) < threshold):
            quarantined.append(f'{_reason(rule, value, unit)} within margin '
                               f'{margin:g} of {threshold:g}')
    skipped += [d for d in not_applicable if d not in skipped]

    if status == 'complete':
        if reasons:
            final = 'failed_integrity'
        elif quarantined:
            final = 'quarantined'
        else:
            final = 'complete'
    else:
        final = status
    return IntegrityVerdict(
        status=final, failed=final not in ('complete', 'quarantined'),
        observed=final not in _UNOBSERVED, reasons=reasons,
        quarantine=quarantined if status == 'complete' else [],
        not_applicable=skipped, measures=dict(measures),
        thresholds_version=settings.version)


# --------------------------------------------------------------------------
# Layer and reference resolution
# --------------------------------------------------------------------------

_GROUP_RE = re.compile(r"group (layer_\d+) type ([\d* ]+)")
_LOAD_RE = re.compile(r'^load_(?P<load>-?\d+\.?\d*(?:[eE][+-]?\d+)?)N\.data$')


def expand(spec: str) -> Set[int]:
    """s48 ``expand`` (:49): ``'1*3 7'`` -> ``{1, 2, 3, 7}``."""
    out: Set[int] = set()
    for tok in spec.split():
        a, _, b = tok.partition('*')
        out |= set(range(int(a), int(b or a) + 1))
    return out


def _layer_index(name: str) -> int:
    m = re.search(r'(\d+)$', str(name))
    return int(m[1]) if m else 0


def _ordered(layers: Dict[str, Set[int]]) -> Dict[str, Set[int]]:
    return dict(sorted(layers.items(), key=lambda kv: _layer_index(kv[0])))


def _record_map(record) -> dict:
    if record is None:
        return {}
    raw = (record.get('layer_type_map') if isinstance(record, dict)
           else getattr(record, 'layer_type_map', None))
    return dict(raw or {})


def _zband_layers(ref_path: Optional[Path], n_layers: int) -> Dict[str, Set[int]]:
    """Split the reference into ``n_layers`` z bands at its largest gaps."""
    if ref_path is None or n_layers < 1:
        return {}
    try:
        types, xyz, _box = read_snapshot(ref_path)
    except (OSError, ValueError):
        return {}
    if len(types) < n_layers:
        return {}
    order = np.argsort(xyz[:, 2], kind='stable')
    z, t = xyz[order, 2], types[order]
    cuts = np.sort(np.argsort(np.diff(z))[::-1][:n_layers - 1]) + 1 if n_layers > 1 else []
    bands = np.split(t, cuts)
    out: Dict[str, Set[int]] = {}
    seen: Set[int] = set()
    for i, band in enumerate(bands, start=1):
        band_types = {int(v) for v in band}
        if not band_types or band_types & seen:
            return {}
        seen |= band_types
        out[f'layer_{i}'] = band_types
    return out


def resolve_layers(locator, record, ref_path: Optional[Path], n_layers: int,
                   *, family: Optional[str] = None) -> Dict[str, Set[int]]:
    """``{layer name: atom types}``, or ``{}`` when the layers cannot be told.

    Tries the provenance ``layer_type_map``, then the ``group layer_N type``
    lines of ``lammps/system.in.settings`` (s48 :110, s51 :110), then -- for
    the sheet-like families only -- z bands of the reference structure;
    overlapping band types give ``{}``. An AFM reference also holds the tip
    and the substrate, so its z bands are not the sheet layers. ``family``
    defaults to the locator's.
    """
    if family is None:
        family = locator.tokens.family
    mapped = _record_map(record)
    if mapped:
        return _ordered({str(k): {int(t) for t in v} for k, v in mapped.items()})
    settings_file = Path(locator.lammps_dir) / 'system.in.settings'
    try:
        text = settings_file.read_text(encoding='utf-8')
    except OSError:
        text = ''
    groups = _GROUP_RE.findall(text)
    if groups:
        return _ordered({name: expand(spec) for name, spec in groups})
    if family not in SHEET_LIKE:
        return {}
    return _zband_layers(ref_path, n_layers)


_END_SNAPSHOT_RE = re.compile(r'^\s*write_data\s+\S*final_', re.MULTILINE)


def writes_end_snapshot(lammps_dir, data_dir=None) -> bool:
    """True when the run writes an end snapshot.

    That is: a ``final_*.data`` is in ``data_dir`` (a tree copied without its
    ``lammps/`` still counts), or any deck in ``lammps_dir`` has a
    ``write_data ... final_`` line. PES and indent decks never do, nor do
    hetero and flake runs built before end snapshots existed; their snapshot
    rules are not applicable.
    """
    if data_dir is not None and Path(data_dir).is_dir() and any(
            Path(data_dir).glob('final_*.data')):
        return True
    lammps_dir = Path(lammps_dir)
    if not lammps_dir.is_dir():
        return False
    for path in sorted(lammps_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        if _END_SNAPSHOT_RE.search(text):
            return True
    return False


def _first_build(*dirs: Path) -> Optional[Path]:
    for d in dirs:
        found = sorted(Path(d).glob('*.lmp')) if Path(d).is_dir() else []
        if found:
            return found[0]
    return None


def bond_reference(locator, family: str, point) -> Optional[Path]:
    """The structure the bond length is measured on, or None.

    AFM-like: ``data/load_<F>N.data`` matched by value (s48 :104), else the
    first ``build/*.lmp`` (run level, then the level above ``L{n}``).
    Sheet-like: the first ``build/*.lmp`` of the run (s51 :108).
    """
    run_dir = Path(locator.run_dir)
    if family in AFM_LIKE:
        data_dir = Path(locator.data_dir)
        if point.load_kind == 'force' and data_dir.is_dir():
            for path in sorted(data_dir.iterdir()):
                m = _LOAD_RE.match(path.name)
                if m and abs(float(m['load']) - float(point.load)) <= 1e-9:
                    return path
        return _first_build(run_dir / 'build', run_dir.parent / 'build')
    return _first_build(run_dir / 'build')


# --------------------------------------------------------------------------
# File-level and point-level checks
# --------------------------------------------------------------------------

def _n_heat(family: str, n_by_layer: Dict[str, int]) -> Optional[int]:
    """AFM: every sheet layer (s48 :190). Sheet: all but the first and the
    last layer, the thermostatted centre (s51 :118)."""
    if not n_by_layer:
        return None
    counts = [n for _, n in sorted(n_by_layer.items(), key=lambda kv: _layer_index(kv[0]))]
    if family in SHEET_LIKE:
        counts = counts[1:-1]
    return int(sum(counts))


def _count_layers(path: Path, layers: Dict[str, Set[int]]) -> Dict[str, int]:
    types, _xyz, _box = read_snapshot(path)
    return {name: int(np.isin(types, sorted(t)).sum()) for name, t in layers.items()}


def check_files(status: str, window: Optional[pd.DataFrame], family: str,  # pylint: disable=too-many-arguments
                n_layers: int, T_target: float, final_path: Optional[Path],  # pylint: disable=invalid-name
                ref_path: Optional[Path], layers: Dict[str, Set[int]],
                settings: IntegritySettings, *, multi_speed: bool = False,
                writes_snapshot: bool = True) -> IntegrityVerdict:
    """Stage 3 on files alone -- no ``RunFrames``, no locator.

    Snapshot rules run only on a complete point (s51 :105); a missing or
    unreadable snapshot or reference, or unresolved layers, quarantines it.
    Without ``writes_snapshot`` (the run's decks write no ``final_*``) or with
    ``multi_speed`` (one final file per load and angle cannot tell the speeds
    apart) the snapshot rules are not applicable instead. Whatever the
    status, the reference gives the atom counts for ``n_heat`` when there is
    no snapshot (s48 :191, s51 :169).
    """
    if family not in AFM_LIKE + SHEET_LIKE:
        raise ValueError(f"unknown family {family!r}; known: {AFM_LIKE + SHEET_LIKE}")
    quarantine: List[str] = []
    skipped: List[str] = []
    measures: Dict[str, float] = {}
    n_by_layer: Dict[str, int] = {}
    if status == 'complete':
        if not writes_snapshot:
            skipped.append(NO_SNAPSHOT_DETAIL)
        elif not layers:
            quarantine.append('no layer types: provenance map, group lines and '
                              'z bands all failed')
        elif multi_speed:
            skipped.append(MULTI_SPEED_DETAIL)
        elif final_path is None:
            quarantine.append('missing final snapshot (data/final_*.data)')
        elif ref_path is None:
            quarantine.append('missing bond reference structure')
        else:
            try:
                snap = snapshot_measures(final_path, ref_path, layers, settings)
            except (OSError, ValueError) as exc:
                quarantine.append(f'unreadable snapshot or reference: {exc}')
            else:
                n_by_layer = snap.pop('n_by_layer')
                measures.update(snap)
    if layers and not n_by_layer and ref_path is not None:
        try:
            n_by_layer = _count_layers(ref_path, layers)
        except (OSError, ValueError):
            n_by_layer = {}
    n_heat = _n_heat(family, n_by_layer)
    measures['n_heat'] = float('nan') if n_heat is None else n_heat
    if window is not None and len(window.columns):
        measures.update(series_measures(window, family, n_layers, T_target,
                                        n_heat, settings))
    return verdict(status, measures, settings, quarantine=quarantine,
                   not_applicable=skipped)


def check_point(point, record, settings: IntegritySettings, *,
                multi_speed: bool, writes_snapshot: bool = True) -> IntegrityVerdict:
    """Stage 3 for one ``RunFrames``: resolve the files and layers, then
    :func:`check_files`. ``T_target`` is the run's temperature token;
    ``point.n_layers`` is the sheet layer count."""
    from .post_channels import final_data_path  # pylint: disable=import-outside-toplevel

    locator, family = point.locator, point.family
    n_layers = point.n_layers
    final = final_data_path(locator.data_dir, point.load_kind, point.load, point.angle)
    ref = bond_reference(locator, family, point)
    layers = resolve_layers(locator, record, ref, n_layers, family=family)
    window = point.window if point.tiers else None
    return check_files(point.status, window, family, n_layers,
                       float(point.tokens.temperature), final, ref, layers,
                       settings, multi_speed=multi_speed,
                       writes_snapshot=writes_snapshot)
