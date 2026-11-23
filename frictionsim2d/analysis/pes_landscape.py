"""The potential-energy landscape of each static PES scan. PES datasets only.

One row per ``pes_tip`` / ``pes_sheet`` run: the descriptors
:func:`frictionsim2d.postprocessing.pes_scan.grid_descriptors` gives its grid (the
point's window tier, the scan's ``results/pes_scan.csv``: ``x, y, energy_eV,
fx, fy, fz`` for the tip, ``afm/pes_scan.lmp:100``; ``dx, dy, energy_eV`` for
the sheet, ``sheetonsheet/pes_scan.lmp:82``), the straight-channel barriers
along x and y, and the Prandtl-Tomlinson parameter of a given spring and
period:

- ``pes_barrier_x`` / ``pes_barrier_y`` (eV): the easiest straight sliding
  channel along x / y (``pes_scan._channel_barrier``), whose ratio is
  ``pes_barrier_aniso``;
- ``pt_eta`` (1) = 2 pi^2 U / (k a^2), ``U`` = ``pes_barrier_U`` (eV, max - min
  of the grid), ``k`` the param ``k`` (eV/A^2), ``a`` the param ``a`` (A).
  Neither has a default: without both, ``pt_eta`` is NaN rather than an
  invented spring or period. (``pes_eta`` is the grid's own: ``k`` the PES
  curvature at its minimum, ``a`` the scanned cell, `pes_scan.py:104-106`.)

Any other family is refused (spec §5.2: PES is not reachable from other
simulation types).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from .base import Analysis, register
from .periods import material_key, summary_identity

__all__ = ['KIND_BY_FAMILY', 'NO_GRID', 'PesLandscape', 'PesLandscapeParams', 'pt_eta']

#: family -> the ``kind`` of ``pes_scan.grid_descriptors`` (``pes_scan.FAMILY_BY_KIND``).
KIND_BY_FAMILY = {'pes_tip': 'tip', 'pes_sheet': 'sheet'}
#: The drop reason of a run with no usable grid (no window tier, no
#: ``energy_eV``, or fewer than 4 points: ``pes_scan.extract_all`` :178).
NO_GRID = 'no_pes_grid'
#: The fewest grid points ``pes_scan.extract_all`` reduces (:178).
MIN_GRID_POINTS = 4


def pt_eta(barrier_u: float, k: Optional[float], a: Optional[float]) -> float:
    """eta = 2 pi^2 U / (k a^2) (1): ``U`` eV, ``k`` eV/A^2, ``a`` A. NaN when
    ``k`` or ``a`` is missing or not positive."""
    if k is None or a is None or not (np.isfinite(k) and np.isfinite(a) and k > 0 and a > 0):
        return float('nan')
    return float(2.0 * np.pi ** 2 * barrier_u / (k * a * a))


Value = Union[float, Dict[str, float], None]


def _value(value: Value, material: str) -> Optional[float]:
    """A param that is one number or ``material -> number``."""
    if isinstance(value, dict):
        found = value.get(material)
        return None if found is None else float(found)
    return None if value is None else float(value)


class PesLandscapeParams(BaseModel):
    """``k``: the spring of the Prandtl-Tomlinson eta (eV/A^2). ``a``: its
    lattice period (A), one number or ``material -> A``. No defaults: either
    missing leaves ``pt_eta`` NaN."""

    k: Value = None
    a: Value = None

    @field_validator('k', 'a', mode='before')
    @classmethod
    def _parse(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value


@register
class PesLandscape(Analysis):  # pylint: disable=too-few-public-methods
    """Barrier, corrugation, curvature and eta of each PES run's grid."""

    name = 'pes_landscape'
    version = '1'
    level = 'run'
    families = frozenset(KIND_BY_FAMILY)
    Params = PesLandscapeParams

    def compute(self, source, params) -> pd.DataFrame:
        """One row per pass PES run; a run with no usable grid is dropped
        (:data:`NO_GRID`).

        Raises:
            ValueError: the source is not a PES family.
        """
        # pylint: disable=import-outside-toplevel,protected-access
        from ..postprocessing import pes_scan
        if source.family not in KIND_BY_FAMILY:
            raise ValueError(f'pes_landscape reads PES datasets only ({sorted(KIND_BY_FAMILY)}); '
                             f'this one is family {source.family!r}')
        kind = KIND_BY_FAMILY[source.family]
        out: List[Dict[str, Any]] = []
        for row in source.rows.itertuples():
            grid = source.frames(row).get('window')
            if grid is None or 'energy_eV' not in grid or len(grid) < MIN_GRID_POINTS:
                source.dropped.append((str(row.point_uuid), NO_GRID))
                continue
            values = pes_scan.grid_descriptors(grid, kind)
            energy = pes_scan._grid_from_frame(grid)[0]
            material = material_key(row.materials)
            k, a = _value(params.k, material), _value(params.a, material)
            out.append({**summary_identity(row), **values,
                        'pes_barrier_x': pes_scan._channel_barrier(energy, axis=1),
                        'pes_barrier_y': pes_scan._channel_barrier(energy, axis=0),
                        'pt_k_eV_A2': np.nan if k is None else k,
                        'pt_a_A': np.nan if a is None else a,
                        'pt_eta': pt_eta(values['pes_barrier_U'], k, a)})
        return pd.DataFrame(out)
