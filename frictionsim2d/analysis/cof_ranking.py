"""The materials ranked by their whole-period COF over a load window.

Method as in ML_taxonomy's ``rank_cof_v2.py``: each
run's COF is its whole-period mean (:mod:`.periods`); a material's ``cof`` is
the mean over the window loads, ``se`` their sample SD / sqrt(n). A material
is ``eligible`` (ranked) only if every window load was run and passes the
integrity check -- simulation failures alone set it aside.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import field_validator

from .base import Analysis, register
from .periods import (WINDOW, PeriodDepends, WindowParams, json_before, material_key, period_map,
                      period_rows)
from .source import PASS_STATUS

__all__ = ['NOT_RUN', 'CofRanking', 'WINDOW']

#: Verdicts of a load point with no data to analyse: rank_cof_v2 keeps only
#: ``~(i0_missing | unobserved)``, i.e. drops ``no_output``,
#: ``not_reached``, ``unreadable`` and the frame-less ``halted``.
NOT_RUN = frozenset({'no_output', 'not_reached', 'unreadable', 'halted'})


def _label(load: float) -> str:
    return f'{load:g}'


class RankingParams(WindowParams):
    """The window, the slice of the dataset and the tribology index.

    ``trib_index`` is rank_cof_v2's JSON (``materials``, ``tribIndex``,
    ``dev``), given as its content so the parameters hash covers it.
    """

    layer: int = 2
    trib_index: Optional[Dict[str, List[Any]]] = None

    @field_validator('trib_index', mode='before')
    @classmethod
    def _parse_trib(cls, value: Any) -> Any:
        return json_before(value)


def _material_row(material: str, loads: Tuple[float, ...], points: pd.DataFrame,
                  runs: Dict[float, Dict[str, Any]], a_x: float) -> Dict[str, Any]:
    """One material's row. ``points``: its rows at the
    window loads, any verdict; ``runs``: load -> period record of its pass
    runs. A non-pass run is never read, so its ``cof_<L>nN`` is NaN."""
    ran = points[~points['verdict_status'].isin(NOT_RUN)]
    ran_loads = sorted({float(v) for v in ran['load']})
    cofs = [runs[f]['cof_periods'] for f in loads if f in runs]
    row: Dict[str, Any] = {
        'material': material, 'a_x_A': a_x, 'n_loads': len(ran_loads),
        'cof': float(np.mean(cofs)) if cofs else np.nan,
        'se': float(np.std(cofs, ddof=1) / np.sqrt(len(cofs))) if len(cofs) > 1 else np.nan,
        'cof_full_run': (float(np.mean([runs[f]['cof_full_run'] for f in loads if f in runs]))
                         if cofs else np.nan),
        'min_periods': min((runs[f]['n_periods'] for f in loads if f in runs), default=np.nan),
        **{f'cof_{_label(f)}nN': runs[f]['cof_periods'] if f in runs else np.nan
           for f in loads}}
    reasons = []
    not_run = [f for f in loads if f not in ran_loads]
    if not_run:
        reasons.append('not run at ' + ','.join(_label(f) for f in not_run) + ' nN')
    for f in ran_loads:
        failed = ran[(ran['load'] == f) & (ran['verdict_status'] != PASS_STATUS)]
        for status in failed['verdict_status']:
            reasons.append(f'integrity check: {status} at {_label(f)} nN')
        if failed.empty and f not in runs:
            # a pass run `period_rows` dropped (no a_x or no drive step)
            reasons.append(f'no lattice period at {_label(f)} nN' if not np.isfinite(a_x)
                           else f'no drive step at {_label(f)} nN')
    row['eligible'] = not reasons
    row['excluded_because'] = '; '.join(reasons)
    return row


@register
class CofRanking(PeriodDepends, Analysis):
    """Each material's whole-period COF over the load window, and its rank."""

    name = 'cof_ranking'
    version = '3'
    level = 'material'
    families = frozenset({'afm'})
    Params = RankingParams

    def compute(self, source, params) -> pd.DataFrame:
        """One row per material of the dataset at ``layer`` and ``angle``:
        the ranked ones by ``cof``, then the excluded ones by name."""
        loads = tuple(float(f) for f in params.loads)
        scope = source.all_rows
        scope = scope[(scope['layers'] == params.layer)
                      & (scope['angle'].astype(float) == float(params.angle))].copy()
        scope['material_key'] = scope['materials'].map(material_key)
        in_window = scope[scope['load'].astype(float).isin(loads)]
        passed = in_window[in_window['verdict_status'] == PASS_STATUS]
        duplicated = passed[passed.duplicated(['material_key', 'load'], keep=False)]
        if not duplicated.empty:
            raise ValueError('more than one pass point per material and load: '
                             f'{sorted(set(zip(duplicated.material_key, duplicated.load)))}')
        period = period_map(source, params)
        keys = dict(zip(passed['point_uuid'], passed['material_key']))
        runs: Dict[str, Dict[float, Dict[str, Any]]] = {}
        for record in period_rows(source, passed, period):
            runs.setdefault(keys[record['point_uuid']], {})[record['load']] = record
        trib = params.trib_index or {}
        zeta = dict(zip(trib.get('materials', []), zip(trib.get('tribIndex', []),
                                                       trib.get('dev', []))))
        out = []
        for material in sorted(scope['material_key'].unique()):
            row = _material_row(material, loads,
                                in_window[in_window['material_key'] == material],
                                runs.get(material, {}), period.get(material, np.nan))
            row['layer'] = params.layer
            row['trib_index'], row['trib_index_dev'] = zeta.get(material, (np.nan, np.nan))
            out.append(row)
        table = pd.DataFrame(out)
        if table.empty:
            return table
        ok = table[table['eligible']].sort_values('cof', kind='stable')
        table['rank'] = table['material'].map(
            dict(zip(ok['material'], range(1, len(ok) + 1)))).astype('Int64')
        table = pd.concat([table[table['eligible']].sort_values('cof', kind='stable'),
                           table[~table['eligible']].sort_values('material')])
        cols = ['rank', 'material', 'layer', 'a_x_A', 'cof', 'se', 'trib_index',
                'trib_index_dev', *[f'cof_{_label(f)}nN' for f in loads], 'cof_full_run',
                'min_periods', 'n_loads', 'eligible', 'excluded_because']
        return table[cols].reset_index(drop=True)
