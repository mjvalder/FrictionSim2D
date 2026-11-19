"""Integrity traces and run status: the two tables that show failures.

Unlike every other analysis, these two keep the points whose integrity
verdict is not ``complete`` -- a trace or a status of a failed point is what
they exist to show. Neither records a non-pass point as dropped.

``integrity_traces`` is the window tier per sample (``run_id, step,
temp_sheet_3d, comz_L1, comz_L2``, as ML_taxonomy's ``integrity_traces_l2.csv``,
plus ``gap = comz_L2 - comz_L1``) with the point's stored s48 measures
(``hole, isolated, ejected, T_sigma, gap_rise``) read from its verdict
(``steps.check_integrity`` keeps them under ``measures``), never recomputed.
``run_status`` is one row per point: ``run_id, material, layer, force, angle,
status`` (``run_status_l2.csv``'s columns).
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .base import Analysis, register
from .periods import summary_identity

__all__ = ['MEASURES', 'NO_WINDOW', 'IntegrityTraces', 'RunStatus', 'run_id']

#: The stored s48 measures carried per run (`integrity._RULES`' measure keys).
MEASURES = ('hole', 'isolated', 'ejected', 'T_sigma', 'gap_rise')
#: The drop reason of a point with no window tier to trace (halted, not reached, ...).
NO_WINDOW = 'no_window_tier'
_TRACE = ('temp_sheet_3d', 'comz_L1', 'comz_L2')


def run_id(row: Any) -> str:
    """``<material>/l<layers>/f<load>/a<angle>`` (``p`` for a pressure load),
    the id of ``integrity_traces_l2.csv``; material spelled with underscores."""
    ident = summary_identity(row)
    kind = 'p' if ident['load_kind'] == 'pressure' else 'f'
    layer = 'l' if ident['layer'] is None else f"l{ident['layer']}"
    return f"{ident['material']}/{layer}/{kind}{ident['load']:g}/a{ident['angle']:g}"


class _NoParams(BaseModel):
    """Neither analysis has parameters."""


@register
class IntegrityTraces(Analysis):  # pylint: disable=too-few-public-methods
    """Window-tier traces of every point that has one, failed points included."""

    name = 'integrity_traces'
    version = '1'
    level = 'run_state'
    families = frozenset({'afm', 'sheetonsheet'})
    Params = _NoParams

    def compute(self, source, params) -> pd.DataFrame:
        """One row per window sample of each point (any verdict). A point
        with no window tier is recorded as dropped (:data:`NO_WINDOW`); the
        non-pass points :class:`RunSource` pre-lists are not dropped."""
        source.dropped.clear()
        frames: List[pd.DataFrame] = []
        for row in source.all_rows.itertuples():
            window = source.frames(row).get('window')
            if window is None or window.empty:
                source.dropped.append((str(row.point_uuid), NO_WINDOW))
                continue
            measures = (source.verdict(row) or {}).get('measures') or {}
            part = pd.DataFrame({'run_id': run_id(row), 'step': np.arange(len(window))})
            for col in _TRACE:
                part[col] = window[col].to_numpy(float) if col in window else np.nan
            part['gap'] = part['comz_L2'] - part['comz_L1']
            part['verdict_status'] = row.verdict_status
            for name in MEASURES:
                value = measures.get(name)
                part[name] = np.nan if value is None else float(value)
            frames.append(part)
        if not frames:
            return pd.DataFrame(columns=['run_id', 'step', *_TRACE, 'gap', 'verdict_status',
                                         *MEASURES])
        return pd.concat(frames, ignore_index=True)


@register
class RunStatus(Analysis):  # pylint: disable=too-few-public-methods
    """The verdict status of every point of the dataset, failed ones included."""

    name = 'run_status'
    version = '1'
    level = 'run_state'
    families = frozenset({'afm', 'sheetonsheet'})
    Params = _NoParams

    def compute(self, source, params) -> pd.DataFrame:
        """One row per point of ``source.all_rows``. Nothing is dropped: the
        non-pass points :class:`RunSource` pre-lists in ``dropped`` are rows
        here, so the list is cleared and the stored result's ``n_dropped`` is 0."""
        source.dropped.clear()
        out = []
        for row in source.all_rows.itertuples():
            ident = summary_identity(row)
            out.append({'run_id': run_id(row), 'material': ident['material'],
                        'layer': ident['layer'], 'force': ident['force'],
                        'angle': ident['angle'], 'status': row.verdict_status})
        return pd.DataFrame(out, columns=['run_id', 'material', 'layer', 'force', 'angle',
                                          'status'])
