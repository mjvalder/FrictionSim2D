"""Surface geometry of a built sheet: the lattice period along the scan.

Port of ``ML_taxonomy/scripts/s42_sheet_geometry.py``. One row per material
and layer count, computed once from the run's built (unrelaxed)
``<mat>_<n>.lmp`` -- the same lattice and orientation as the MD runs -- and
the element/type map and layer groups of its ``system.in.settings``.

``period_x_top`` is the shortest x translation that maps the top atomic
plane of the top layer onto itself (the 0 degree scan direction); the other
columns describe that plane's spacing, its hollows and the layer's stacking.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.spatial import Delaunay, cKDTree  # pylint: disable=no-name-in-module

from .base import Analysis, register
from .source import PASS_STATUS

__all__ = ['TOP_TOL', 'SheetGeometry', 'describe', 'read_lmp', 'read_settings']

#: Atoms within this of the highest one form the top plane (s42:42, A).
TOP_TOL = 0.35


def _expand(spec: str) -> List[int]:
    """``"7*8 10"`` -> ``[7, 8, 10]`` (s42:45-50)."""
    out: List[int] = []
    for tok in spec.split():
        first, _, last = tok.partition('*')
        out += list(range(int(first), int(last or first) + 1))
    return out


#: Atom style -> (column of the type, column of x) in an ``Atoms`` line. s42
#: read only ``atomic`` (``id type x y z``); the sheet family's build files are
#: ``molecular`` (``id mol type x y z``).
_ATOM_COLUMNS = {'atomic': (1, 2), 'molecular': (2, 3), 'bond': (2, 3), 'angle': (2, 3),
                 'full': (2, 4), 'charge': (1, 3)}


def read_lmp(text: str) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """``(box, types, xyz)`` of a LAMMPS data file's ``Atoms`` block (s42:53-60),
    the columns chosen by its ``# <style>`` hint (``atomic`` without one).

    Raises:
        ValueError: an atom style this reader does not know.
    """
    lines = text.splitlines()
    box = {k: float(line.split()[1]) - float(line.split()[0]) for line in lines[:30]
           for k in ('x', 'y') if f'{k}lo {k}hi' in line}
    start = next(k for k, line in enumerate(lines) if line.startswith('Atoms'))
    _, _, hint = lines[start].partition('#')
    style = hint.strip() or 'atomic'
    if style not in _ATOM_COLUMNS:
        raise ValueError(f'unsupported atom style {style!r} (known: {sorted(_ATOM_COLUMNS)})')
    col_t, col_x = _ATOM_COLUMNS[style]
    rows = [line.split() for line in lines[start + 2:] if len(line.split()) >= col_x + 3]
    atoms = np.array([[float(r[col_t])] + [float(v) for v in r[col_x:col_x + 3]]
                      for r in rows])
    return box, atoms[:, 0].astype(int), atoms[:, 1:4]


def read_settings(text: str) -> Tuple[Dict[int, str], Dict[str, float], Dict[int, Set[int]]]:
    """``(type -> element, element -> tip LJ sigma, layer index -> types)`` of a
    ``system.in.settings`` (s42:63-73). The sigma map is empty without a tip."""
    elem: Dict[int, str] = {}
    for spec, name in re.findall(r'^mass ([\d*]+) \S+ #(\w+)', text, re.M):
        for t in _expand(spec):
            elem[t] = name
    sigma = {m[1]: float(m[0]) for m in re.findall(
        r'pair_coeff 4\*6 \S+ lj/cut \S+ (\S+) # Si\(tip\)-(\w+)\(sheet\)', text)}
    layers = {int(i): set(_expand(spec))
              for i, spec in re.findall(r'^group layer_(\d+) type ([\d* ]+)', text, re.M)}
    return elem, sigma, layers


def _period_x(wrapped: np.ndarray, tree: cKDTree, cell: List[float]) -> float:
    """Shortest x shift mapping the plane onto itself: a 0.01 A scan from 0.5 A,
    then a 0.002 A refinement over the next 0.08 A (s42:112-120)."""
    for d0 in np.arange(0.5, 12.0, 0.01):
        if np.median(tree.query(np.mod(wrapped + [d0, 0], cell))[0]) < 0.05:
            cand = np.arange(d0, d0 + 0.08, 0.002)
            err = [np.median(tree.query(np.mod(wrapped + [c, 0], cell))[0]) for c in cand]
            return float(cand[int(np.argmin(err))])
    return float('nan')


def _hollow_radius(xy: np.ndarray, box: Dict[str, float]) -> float:
    """Radius of the largest empty circle between plane atoms: the 95th
    percentile of the Delaunay circumradii of the 3x3 periodic tiling,
    over the triangles centred in the cell (s42:122-131)."""
    shifts = [(i * box['x'], j * box['y']) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    tiles = np.vstack([xy + s for s in shifts])
    p = tiles[Delaunay(tiles).simplices]
    a_ = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    b_ = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    c_ = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])  # 2D cross product
    circ = a_ * b_ * c_ / (4 * area + 1e-12)
    centre = p.mean(axis=1)
    inside = (centre >= 0).all(axis=1) & (centre < [box['x'], box['y']]).all(axis=1)
    return float(np.percentile(circ[inside], 95))


def describe(box: Dict[str, float], typ: np.ndarray, xyz: np.ndarray,
             elem: Dict[int, str], sigma: Dict[str, float],
             layers: Dict[int, Set[int]]) -> Dict[str, Any]:
    """The geometry columns of one built sheet (s42:89-145).

    ``typ`` are the structure's types already shifted to the settings'
    numbering. The top layer is the highest ``layers`` index; the layer
    below it (if any) gives ``d_interlayer``, else that is NaN.
    """
    top_idx = max(layers)
    m_top = np.isin(typ, list(layers[top_idx]))
    z_top = xyz[m_top, 2]
    top = m_top & (xyz[:, 2] > z_top.max() - TOP_TOL)
    els = sorted({elem[t] for t in typ[top]})
    xy = xyz[top, :2]
    cell = [box['x'], box['y']]

    wrapped = np.mod(xy, cell)
    tree = cKDTree(wrapped, boxsize=cell)
    dist, nbr = tree.query(wrapped, k=7)
    d_nn = float(np.median(dist[:, 1]))
    dirs = set()
    for i, here in enumerate(xy):
        for dd, jj in zip(dist[i, 1:], nbr[i, 1:]):
            if dd < 1.1 * d_nn:
                vec = xy[jj] - here
                vec -= np.round(vec / cell) * cell
                dirs.add(round(float(np.degrees(np.arctan2(vec[1], vec[0]))) % 180))
    dirs_sorted = sorted(dirs)

    r_hollow = _hollow_radius(xy, box)
    sig = float(np.mean([sigma[e] for e in els])) if all(e in sigma for e in els) else np.nan
    radius = 2 ** (1 / 6) * sig
    levels = np.sort(np.unique(np.round(z_top, 2)))[::-1]
    below = levels[levels < z_top.max() - TOP_TOL]
    area = box['x'] * box['y']
    lower = layers.get(top_idx - 1)
    m_low = np.isin(typ, list(lower)) if lower else None
    return {
        'top_element': '+'.join(els), 'top_sigma': sig, 'd_nn_top': d_nn,
        'period_x_top': _period_x(wrapped, tree, cell),
        'nn_dirs_deg': ' '.join(str(d) for d in dirs_sorted),
        'x_is_nn': any(min(t, 180 - t) <= 2 for t in dirs_sorted),
        'r_hollow': r_hollow,
        'hollow_depth': radius - np.sqrt(max(radius * radius - r_hollow * r_hollow, 0.0)),
        'rho_top': top.sum() / area, 'rho_layer': m_top.sum() / area,
        'thickness': float(z_top.max() - z_top.min()),
        'top_gap': float(z_top.max() - below[0]) if len(below) else np.nan,
        'd_interlayer': (float(z_top.min() - xyz[m_low, 2].max())
                         if m_low is not None and m_low.any() else np.nan),
    }


def _material(name: str) -> str:
    """s42:138: ``h-MoS2`` -> ``h_MoS2``."""
    return name.replace('-', '_', 1) if name[:2] in ('h-', 't-', 'b-', 'p-') else name


@register
class SheetGeometry(Analysis):  # pylint: disable=too-few-public-methods
    """Lattice period and surface geometry of each material's built sheet."""

    name = 'sheet_geometry'
    version = '2'
    level = 'material'
    families = frozenset({'afm', 'sheetonsheet'})

    class Params(BaseModel):
        """No parameters."""

    def compute(self, source, params) -> pd.DataFrame:
        """One row per (material, layers), over every point of the source
        whatever its verdict (``all_rows``): the built structure does not
        depend on how the run went. The first run of each with a stored
        structure is read. A point with no ``layers`` (the sheet family's)
        takes the layer count of its settings' ``layer_<i>`` groups. A pass
        point of a material with no structure is recorded as dropped."""
        groups: Dict[Tuple[str, Optional[int]], List[Any]] = {}
        for row in source.all_rows.itertuples():
            mats = row.materials
            mats = [mats] if isinstance(mats, str) else list(mats)
            layers = None if row.layers is None or pd.isna(row.layers) else int(row.layers)
            groups.setdefault((_material(mats[0]), layers), []).append(row)
        out = []
        for (material, layers), rows in sorted(groups.items(), key=lambda kv: (kv[0][0],
                                                                                kv[0][1] or 0)):
            found = self._first_structure(source, rows)
            if found is None:
                source.dropped.extend((str(r.point_uuid), 'no_structure') for r in rows
                                      if r.verdict_status == PASS_STATUS)
                continue
            text, row = found
            elem, sigma, groups_t = read_settings(source.settings_text(row))
            box, typ, xyz = read_lmp(text)
            # system.in reads the sheet with a type offset (s42:85).
            typ = typ + (min(t for ts in groups_t.values() for t in ts) - typ.min())
            out.append({'material': material,
                        'layer': layers if layers is not None else max(groups_t),
                        **describe(box, typ, xyz, elem, sigma, groups_t)})
        return pd.DataFrame(out)

    @staticmethod
    def _first_structure(source, rows) -> Optional[Tuple[str, Any]]:
        for row in rows:
            text = source.structure_text(row)
            if text is not None:
                return text, row
        return None
