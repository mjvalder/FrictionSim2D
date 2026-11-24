"""Tip-sheet contact at the loaded state (before sliding) and at the end of the slide.

Method as in ML_taxonomy's ``s40_contact_loaded.py``, whose parsing and
contact rule are ``s36_contact_endstates.py``'s. One row per run and state: ``load`` reads
``load_<F>N.data`` (after loading, before sliding), ``final`` the point's end snapshot
(``final_f<F>N_a0.data`` at 0 degrees).

Per snapshot:

1. contact: a top-layer atom within ``2^(1/6) sigma`` of a tip atom (sigma
   of the tip pair with that atom's type); ``contact_area`` is their xy
   convex hull, ``contact_density`` their areal density over that of the top
   sublayer (the top-layer types whose far-field median z, beyond
   :data:`FAR` from the tip axis, is within :data:`SUBLAYER_DZ` of the highest).
2. interlocking: each contact tip atom's height ``h`` over the mean z of the
   top-sublayer atoms within ``a_x`` laterally, as median ``h / sigma`` and
   median ``sink = 2^(1/6) - h / sigma``; ``frac_below_tops`` is the share
   below the highest of those atoms.
3. the unshifted tip x sheet LJ energy within :data:`CUTOFF`, total and per
   contact atom, beside ``E_int_tip_sheet`` at step 0 of the slide.

The tip is the settings' ``tip_all`` group (ML_taxonomy hard-codes types 4-6); the top
layer is its highest ``layer_<i>`` group, the sheet all of them.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError, cKDTree  # pylint: disable=no-name-in-module

from ..postprocessing.snapshot_geometry import expand_lammps_types, parse_type_groups
from .base import Analysis, register
from .periods import (NO_PERIOD, PeriodDepends, WindowParams, material_key, period_map,
                      summary_identity, window_rows)

__all__ = ['CUTOFF', 'FAR', 'RMIN', 'SUBLAYER_DZ', 'Contact', 'ContactParams', 'anion',
           'contact_metrics', 'lj_energy', 'tip_pairs']

#: LJ cutoff of the tip-sheet energy, as in the runs (A).
CUTOFF = 11.0
#: Lateral distance from the tip axis beyond which the sheet is far field (A).
FAR = 25.0
#: Top-layer types within this of the highest far-field level form the top sublayer (A).
SUBLAYER_DZ = 0.5
#: The LJ minimum over sigma.
RMIN = 2 ** (1 / 6)
#: The drop reason of a run whose snapshot of a state is not stored.
NO_SNAPSHOT = 'no_{state}_snapshot'
#: The drop reason of a snapshot s40's ``one`` could not read.
UNREADABLE = 'contact_unreadable_{state}'
#: The drop reason of a state whose contact atoms are collinear (or coincide):
#: their convex hull is flat, and scipy's Qhull raises instead of giving 0.
DEGENERATE = 'contact_degenerate_{state}'

#: element -> (periodic group, Pauling electronegativity), the ``group`` and
#: ``en`` columns of ML_taxonomy ``baseline_inputs/descriptors/elemental_properties.csv``
#: Neither frictionsim2d/potentials/uff.py (LJ sigma/epsilon only)
#: nor frictionsim2d/data/materials (structure files) holds them.
ELEMENTS: Dict[str, Tuple[int, float]] = {
    'B': (13, 2.04), 'C': (14, 2.55), 'O': (16, 3.44), 'Al': (13, 1.61), 'Si': (14, 1.9),
    'P': (15, 2.19), 'S': (16, 2.58), 'Sc': (3, 1.36), 'Ti': (4, 1.54), 'V': (5, 1.63),
    'Cr': (6, 1.66), 'Mn': (7, 1.55), 'Fe': (8, 1.83), 'Co': (9, 1.88), 'Ni': (10, 1.91),
    'Ga': (13, 1.81), 'Ge': (14, 2.01), 'As': (15, 2.18), 'Se': (16, 2.55), 'Zr': (4, 1.33),
    'Nb': (5, 1.6), 'Mo': (6, 2.16), 'Tc': (7, 1.9), 'Rh': (9, 2.28), 'Pd': (10, 2.2),
    'In': (13, 1.78), 'Sn': (14, 1.96), 'Sb': (15, 2.05), 'Te': (16, 2.1), 'Hf': (4, 1.3),
    'Ta': (5, 1.5), 'W': (6, 2.36), 'Re': (7, 1.9), 'Ir': (9, 2.2), 'Pt': (10, 2.28),
    'Bi': (15, 2.02),
}

#: A tip-sheet LJ pair line of ``system.in.settings`` (the tip types left
#: free; the sheet element from the trailing comment).
_PAIR = re.compile(r'^pair_coeff (\S+) ([\d*]+) lj/cut (\S+) (\S+)\s*# \w+\(tip\)-(\w+)\(sheet\)',
                   re.M)
_LAYER = re.compile(r'^layer_(\d+)$')


def tip_pairs(text: str, tip: Set[int]) -> Dict[int, Tuple[str, float, float]]:
    """``{sheet type: (element, eps, sigma)}`` of the tip-sheet LJ pairs whose
    first type range is the tip's."""
    return {t: (el, float(eps), float(sig))
            for first, spec, eps, sig, el in _PAIR.findall(text) if expand_lammps_types(first) & tip
            for t in expand_lammps_types(spec)}


def anion(elements: Set[str]) -> str:
    """The element furthest right in the periodic table, Pauling
    electronegativity breaking ties: electronegativity alone
    picks the metal in W/Mo/Pt/Ir tellurides (Te 2.1 < W 2.36). An element
    missing from :data:`ELEMENTS` ranks last, as s40's ``.get(el, -1)``."""
    return max(elements, key=lambda el: ELEMENTS.get(el, (-1, -1)))


def lj_energy(tip: np.ndarray, sheet: np.ndarray, sheet_t: np.ndarray,
              pairs: Dict[int, Tuple[str, float, float]]) -> float:
    """Unshifted 12-6 LJ summed over tip x sheet pairs within :data:`CUTOFF`."""
    tree = cKDTree(sheet)
    total = 0.0
    for i, nb in enumerate(tree.query_ball_point(tip, CUTOFF)):
        if not nb:
            continue
        r = np.linalg.norm(sheet[nb] - tip[i], axis=1)
        ts = sheet_t[nb]
        eps = np.array([pairs[t][1] for t in ts])
        sig = np.array([pairs[t][2] for t in ts])
        sr6 = (sig / r) ** 6
        total += float(np.sum(4 * eps * (sr6 * sr6 - sr6)))
    return total


def contact_metrics(typ: np.ndarray, xyz: np.ndarray, area_xy: float, tip_types: Set[int],
                    top_types: Set[int], sheet_types: Set[int],
                    pairs: Dict[int, Tuple[str, float, float]], a_x: float) -> Dict[str, Any]:
    """The contact columns of one snapshot. ``area_xy``
    is the box's xy area."""
    # pylint: disable=too-many-locals
    tip = xyz[np.isin(typ, list(tip_types))]
    top_m = np.isin(typ, list(top_types))
    top, top_t = xyz[top_m], typ[top_m]
    sheet_m = np.isin(typ, list(sheet_types))

    # the top sublayer from the far field
    axis = tip[:, :2].mean(axis=0)
    far = np.linalg.norm(top[:, :2] - axis, axis=1) > FAR
    zt = {t: np.median(top[far & (top_t == t), 2]) for t in np.unique(top_t)
          if (far & (top_t == t)).any()}
    zmax = max(zt.values())
    sub_types = [t for t, z in zt.items() if z >= zmax - SUBLAYER_DZ]
    sub_m = np.isin(top_t, sub_types)
    rho_sub = sub_m.sum() / area_xy

    # 1. contact, the s36 rule
    d, _ = cKDTree(tip).query(top, k=1)
    lim = np.array([RMIN * pairs[int(t)][2] if int(t) in pairs else np.nan for t in top_t])
    c = top[d < lim]
    area = ConvexHull(c[:, :2]).volume if len(c) >= 3 else 0.0
    dens = (len(c) / area) / rho_sub if area > 0 else np.nan

    # 2. interlocking: contact tip atoms vs the local top-sublayer surface
    dt, it = cKDTree(top).query(tip, k=1)
    sig_near = np.array([pairs[int(t)][2] for t in top_t[it]])
    ctip = tip[dt < RMIN * sig_near]
    csig = sig_near[dt < RMIN * sig_near]
    subxy = cKDTree(top[sub_m][:, :2])
    subz = top[sub_m][:, 2]
    hs: List[float] = []
    below: List[bool] = []
    for p, sg in zip(ctip, csig):
        nb = subxy.query_ball_point(p[:2], a_x)
        if nb:
            hs.append((p[2] - subz[nb].mean()) / sg)
            below.append(bool(p[2] < subz[nb].max()))
    h = np.array(hs)

    # 3. tip-sheet LJ energy
    e = lj_energy(tip, xyz[sheet_m], typ[sheet_m], pairs)

    els = {el for el, _, _ in pairs.values()}
    an = anion(els)
    sig_of = {el: sg for el, _, sg in pairs.values()}
    top_el = pairs[int(sub_types[0])][0]
    return {'contact_atoms': len(c), 'contact_area': area, 'top_sub_density': rho_sub,
            'contact_density': dens, 'n_contact_tip': len(h),
            'h_over_sigma': float(np.median(h)) if len(h) else np.nan,
            'sink_over_sigma': float(np.median(RMIN - h)) if len(h) else np.nan,
            'frac_below_tops': float(np.mean(below)) if below else np.nan,
            'E_lj_eV': e, 'E_lj_per_contact_eV': e / len(c) if len(c) else np.nan,
            'anion': an, 'tip_x_sigma': sig_of[an], 'top_element': top_el,
            'top_sigma': sig_of[top_el]}


def _layers(groups: Dict[str, Set[int]]) -> Tuple[Set[int], Set[int]]:
    """``(top layer types, sheet types)``: the highest ``layer_<i>`` group and
    the union of them all."""
    layers = {int(m[1]): types for name, types in groups.items() if (m := _LAYER.match(name))}
    if not layers:
        raise KeyError('no layer_<i> group')
    return layers[max(layers)], set().union(*layers.values())


def _e_int_step0(frames: Dict[str, pd.DataFrame]) -> float:
    """``E_int_tip_sheet`` at step 0 of the peratom tier, NaN
    without that tier, column or step."""
    peratom = frames.get('peratom')
    if peratom is None or 'E_int_tip_sheet' not in peratom or 'time' not in peratom:
        return float('nan')
    at0 = peratom.loc[peratom['time'] == 0, 'E_int_tip_sheet']
    return float(at0.iloc[0]) if len(at0) else float('nan')


class ContactParams(WindowParams):
    """The window loads and the angle whose end snapshot is read, plus where
    ``a_x`` comes from (:class:`~.periods.PeriodParams`)."""


@register
class Contact(PeriodDepends, Analysis):
    """Tip-sheet contact, interlocking and LJ energy of each run's load and
    final snapshots."""

    name = 'contact'
    version = '2'
    level = 'run_state'
    families = frozenset({'afm'})
    Params = ContactParams

    def compute(self, source, params) -> pd.DataFrame:
        """One row per pass run at the window loads and ``angle``, and state.
        A state with no stored snapshot (:data:`NO_SNAPSHOT`), one that
        cannot be read (:data:`UNREADABLE`), or one whose contact atoms have a
        flat hull (:data:`DEGENERATE`), is dropped; a run whose material
        has no ``a_x`` is dropped (:data:`~.periods.NO_PERIOD`)."""
        rows = window_rows(source, params)
        period = period_map(source, params)
        out: List[Dict[str, Any]] = []
        for row in rows.itertuples():
            a_x = float(period.get(material_key(row.materials), np.nan))
            if not np.isfinite(a_x):
                source.dropped.append((str(row.point_uuid), NO_PERIOD))
                continue
            states = self._states(source, row, a_x)
            if states:
                e0 = _e_int_step0(source.frames(row))
                out += [{**summary_identity(row), 'state': state, **metrics,
                         'E_int_step0_eV': e0, 'a_x_A': a_x} for state, metrics in states]
        return pd.DataFrame(out)

    @staticmethod
    def _states(source, row, a_x: float) -> List[Tuple[str, Dict[str, Any]]]:
        from ..postprocessing.integrity import read_snapshot  # pylint: disable=import-outside-toplevel
        found: List[Tuple[str, Dict[str, Any]]] = []
        text: Optional[str] = None
        for state in ('load', 'final'):
            path = source.snapshot(row, state)
            if path is None:
                source.dropped.append((str(row.point_uuid), NO_SNAPSHOT.format(state=state)))
                continue
            try:
                if text is None:
                    text = source.settings_text(row)
                groups = parse_type_groups(text)
                tip = groups['tip_all']
                top, sheet = _layers(groups)
                typ, xyz, box = read_snapshot(path)
                area_xy = (box['x'][1] - box['x'][0]) * (box['y'][1] - box['y'][0])
                found.append((state, contact_metrics(typ, xyz, area_xy, tip, top, sheet,
                                                     tip_pairs(text, tip), a_x)))
            except QhullError:
                source.dropped.append((str(row.point_uuid), DEGENERATE.format(state=state)))
            except (OSError, IndexError, TypeError, ValueError, KeyError):
                source.dropped.append((str(row.point_uuid), UNREADABLE.format(state=state)))
        return found
