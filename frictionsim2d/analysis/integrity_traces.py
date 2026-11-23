"""Integrity traces and run status: the two tables that show failures.

Unlike every other analysis, these two keep the points that are not
integrity-pass (a ``failed_integrity`` point, an unfinished one): a trace or a
status of such a point is what they exist to show. Neither records a
non-pass point as dropped; ``integrity_traces`` drops only a point with no
window tier to trace.

``integrity_traces`` is the window tier per sample (``run_id, step,
temp_sheet_3d, comz_L1, comz_L2``, as ML_taxonomy's ``integrity_traces_l2.csv``,
plus ``gap = comz_L2 - comz_L1``) with the point's stored s48 measures
(``hole, isolated, ejected, T_sigma, gap_rise``) read from its verdict
(``steps.check_integrity`` keeps them under ``measures``), never recomputed.
``run_status`` is one row per point: ``run_id, material, layer, force, angle,
status`` (``run_status_l2.csv``'s columns) plus ``verdict``. ``status`` is the
run's completion status (:func:`completion_status`), ``verdict`` the stored
integrity verdict.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from .base import Analysis, register
from .periods import summary_identity

__all__ = ['MEASURES', 'NO_WINDOW', 'IntegrityTraces', 'RunStatus', 'TraceParams',
           'completion_status', 'run_id']

#: The stored s48 measures carried per run (`integrity._RULES`' measure keys).
MEASURES = ('hole', 'isolated', 'ejected', 'T_sigma', 'gap_rise')
#: The drop reason of a point with no window tier to trace (halted, not reached, ...).
NO_WINDOW = 'no_window_tier'
_TRACE = ('temp_sheet_3d', 'comz_L1', 'comz_L2')


def completion_status(verdict_status: Any) -> Any:
    """The run's completion status from its integrity verdict status: a
    ``failed_integrity`` or ``quarantined`` run finished, so it is
    ``complete``, as in ``run_status_l2.csv``; any other status is unchanged.
    (A ruling only moves a point between these, so ``original_status`` adds
    nothing.)"""
    return 'complete' if verdict_status in ('failed_integrity', 'quarantined') else verdict_status


def run_id(row: Any) -> str:
    """``<material>/l<layers>/f<load>/a<angle>`` (``p`` for a pressure load),
    the id of ``integrity_traces_l2.csv``; material spelled with underscores."""
    ident = summary_identity(row)
    kind = 'p' if ident['load_kind'] == 'pressure' else 'f'
    layer = 'l' if ident['layer'] is None else f"l{ident['layer']}"
    return f"{ident['material']}/{layer}/{kind}{ident['load']:g}/a{ident['angle']:g}"


class _NoParams(BaseModel):
    """`run_status` has no parameters."""


class TraceParams(BaseModel):
    """Which points `integrity_traces` traces: ``materials`` (names in
    either spelling, ``h-MoS2`` or ``h_MoS2``; a list, JSON text or
    comma-separated) and ``layer``. None (the default) keeps every one. A
    point left out is not dropped: it is outside the selection."""

    materials: Optional[List[str]] = None
    layer: Optional[int] = None

    @field_validator('materials', mode='before')
    @classmethod
    def _parse(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            value = json.loads(text) if text.startswith('[') else text.split(',')
        if value is None:
            return None
        return [str(v).strip().replace('-', '_') for v in value]


def _selected(row: Any, params: TraceParams) -> bool:
    ident = summary_identity(row)
    if params.materials is not None and ident['material'] not in params.materials:
        return False
    return params.layer is None or ident['layer'] == params.layer


@register
class IntegrityTraces(Analysis):  # pylint: disable=too-few-public-methods
    """Window-tier traces of every point that has one, failed points included."""

    name = 'integrity_traces'
    version = '2'
    level = 'run_state'
    families = frozenset({'afm', 'sheetonsheet'})
    Params = TraceParams
    #: Its rows are keyed by ``run_id`` (the frozen table's id), not the level's columns.
    identity = ('run_id',)

    def compute(self, source, params) -> pd.DataFrame:
        """One row per window sample of each selected point (any verdict;
        :class:`TraceParams`). A point with no window tier is recorded as
        dropped (:data:`NO_WINDOW`); the non-pass points :class:`RunSource`
        pre-lists are not dropped."""
        source.dropped.clear()
        frames: List[pd.DataFrame] = []
        for row in source.all_rows.itertuples():
            if not _selected(row, params):
                continue
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
    #: Its rows are keyed by ``run_id`` (the frozen table's id), not the level's columns.
    identity = ('run_id',)

    def compute(self, source, params) -> pd.DataFrame:
        """One row per point of ``source.all_rows``. Nothing is dropped: the
        non-pass points :class:`RunSource` pre-lists in ``dropped`` are rows
        here, so the list is cleared and the stored result's ``n_dropped`` is 0."""
        # `store.run_analysis` reads `source.dropped` after `compute`, so clearing
        # it here is what makes the stored `n_dropped` 0.
        source.dropped.clear()
        out = []
        for row in source.all_rows.itertuples():
            ident = summary_identity(row)
            out.append({'run_id': run_id(row), 'material': ident['material'],
                        'layer': ident['layer'], 'force': ident['force'],
                        'angle': ident['angle'],
                        'status': completion_status(row.verdict_status),
                        'verdict': row.verdict_status})
        return pd.DataFrame(out, columns=['run_id', 'material', 'layer', 'force', 'angle',
                                          'status', 'verdict'])
