"""Compare a store export with a set's legacy ``output_full_*.json``, point by point.

The legacy files (pre-schema-3, top-level ``results`` nested down to
``.../p<load>/a<angle>`` or ``.../f<load>/a<angle>``) are what the published
figures were drawn from. The store is the source of truth (spec 7.3); this lists
every difference so a figure that changes can be traced to its data.

    python tools/publication/compare_legacy.py LEGACY.json SCHEMA3.json [--label L]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

#: Legacy column -> schema-3 column, where the name changed.
RENAMED = {'v_sx': 'react_x', 'v_sy': 'react_y', 'v_sz': 'react_z',
           'v_fx': 'drive_fx', 'v_fy': 'drive_fy', 'v_fz': 'drive_fz',
           'v_comx_top': 'drivex', 'v_comy_top': 'drivey', 'v_comz_top': 'drivez'}

Key = Tuple[str, float, float]  # (load kind 'p' | 'f', load, angle)
_LOAD_RE = re.compile(r'^([pf])(-?[\d.]+)$')
_ANGLE_RE = re.compile(r'^a(-?[\d.]+)$')


@dataclass
class ColumnDiff:
    """The worst difference of one column at one point."""
    key: Key
    column: str
    max_abs: float
    max_rel: float


@dataclass
class Report:
    """Everything :func:`compare` found."""
    columns: List[ColumnDiff] = field(default_factory=list)
    missing: List[Key] = field(default_factory=list)   # legacy point not in the store
    extra: List[Key] = field(default_factory=list)     # store point not in legacy
    notes: List[str] = field(default_factory=list)

    @property
    def max_rel(self) -> float:
        """The largest relative difference of any column at any point."""
        return max((c.max_rel for c in self.columns), default=0.0)


def legacy_points(legacy: dict) -> Iterator[Tuple[Key, dict]]:
    """``(key, {'columns', 'data'})`` for every run in a legacy file."""
    def walk(node, parts):
        if isinstance(node, dict) and 'columns' in node and 'data' in node:
            load = _LOAD_RE.match(parts[-2]) if len(parts) >= 2 else None
            angle = _ANGLE_RE.match(parts[-1])
            if load and angle:
                yield (load.group(1), float(load.group(2)), float(angle.group(1))), node
            return
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, parts + [k])
    yield from walk(legacy.get('results', {}), [])


def schema3_points(export: dict, tier: str = 'window') -> Iterator[Tuple[Key, dict]]:
    """``(key, tier)`` for every run of a schema-3 export."""
    for run in export.get('runs', []):
        kind = 'p' if run.get('load_kind') == 'pressure' else 'f'
        frame = run.get('tiers', {}).get(tier)
        if frame:
            yield (kind, float(run['load']), float(run['angle'])), frame


def _column(frame: dict, name: str) -> Optional[np.ndarray]:
    cols = frame['columns']
    if name not in cols:
        return None
    i = cols.index(name)
    return np.array([row[i] for row in frame['data']], dtype=float)


def compare(legacy: dict, export: dict) -> Report:
    """Every legacy column against its schema-3 counterpart, at matching points."""
    report = Report()
    old: Dict[Key, dict] = dict(legacy_points(legacy))
    new: Dict[Key, dict] = dict(schema3_points(export))
    report.missing = sorted(set(old) - set(new))
    report.extra = sorted(set(new) - set(old))
    for key in sorted(set(old) & set(new)):
        for column in old[key]['columns']:
            a = _column(old[key], column)
            b = _column(new[key], RENAMED.get(column, column))
            if b is None:
                report.notes.append(f'{key}: {column} not in the store export')
                continue
            if len(a) != len(b):
                report.notes.append(f'{key}: {column} has {len(a)} legacy rows, {len(b)} stored')
                n = min(len(a), len(b))
                a, b = a[:n], b[:n]
            diff = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
            scale = float(np.nanmax(np.abs(a))) if len(a) else 0.0
            report.columns.append(ColumnDiff(key, column, diff, diff / scale if scale else diff))
    return report


def render(label: str, report: Report, threshold: float = 1e-9) -> str:
    """A markdown section for STORE_VS_LEGACY.md."""
    lines = [f'## {label}', '',
             f'- points compared: {len({c.key for c in report.columns})}; '
             f'max relative difference: {report.max_rel:.3g}']
    if report.missing:
        lines.append(f'- in legacy, not in the store: {report.missing}')
    if report.extra:
        lines.append(f'- in the store, not in legacy: {report.extra}')
    lines += [f'- note: {n}' for n in sorted(set(report.notes))]
    worse = [c for c in report.columns if c.max_rel > threshold]
    if worse:
        lines += ['', '| point | column | max abs | max rel |', '|---|---|---|---|']
        lines += [f'| {c.key} | {c.column} | {c.max_abs:.4g} | {c.max_rel:.3g} |'
                  for c in sorted(worse, key=lambda c: -c.max_rel)]
    return '\n'.join(lines) + '\n'


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: print the markdown section for one set."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    parser.add_argument('legacy', type=Path)
    parser.add_argument('schema3', type=Path)
    parser.add_argument('--label', default=None)
    args = parser.parse_args(argv)
    report = compare(json.loads(args.legacy.read_text(encoding='utf-8')),
                     json.loads(args.schema3.read_text(encoding='utf-8')))
    print(render(args.label or args.legacy.parent.parent.name, report))
    return 0


if __name__ == '__main__':
    sys.exit(main())
