"""Deck discovery: which `lammps/*.in` decks a finished run holds.

Each deck becomes one import-only CalcJob, so this module decides which decks
exist, which load points each covers and which log belongs to each. It does not
import aiida. `LoadPoint` is only named for typing (under `TYPE_CHECKING`) and
`points_for` duck-types on `.load`/`.angle`/`.speed`, so any object carrying
those attributes works.

The point filter inverts `core.utils.tokens.format_numeric_token` exactly as
`hpc/manifest.py` does: `m` is a minus sign, `p` a decimal point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ..run.tree import RunLocator
    from .ingest import LoadPoint

__all__ = ['Deck', 'discover_decks', 'deck_log', 'points_for']

#: What `format_numeric_token` emits for `f"{value:g}"`: optional leading `m`,
#: digits, optional `p<digits>`, optional exponent `em<digits>` / `e<digits>`.
_NUM = r'm?\d+(?:p\d+)?(?:e[m+]?\d+)?'

#: stem pattern -> filter key. The builders emit exactly these:
#: slide_f<load>N (AFM force), slide_p<load>gpa (sheet pressure),
#: slide_a<angle>, slide_<speed>ms.
_STEM_PATTERNS = (
    (re.compile(rf'^slide_f(?P<v>{_NUM})N$'), 'load'),
    (re.compile(rf'^slide_p(?P<v>{_NUM})gpa$'), 'load'),
    (re.compile(rf'^slide_a(?P<v>{_NUM})$'), 'angle'),
    (re.compile(rf'^slide_(?P<v>{_NUM})ms$'), 'speed'),
)

_CONFLICT = '.sync-conflict-'
_TOL = 1e-9


@dataclass(frozen=True)
class Deck:
    """One LAMMPS input deck of a run."""
    name: str
    path: Path
    log_path: Optional[Path]
    point_filter: Dict[str, float] = field(default_factory=dict)
    writes_structures: bool = False


def _token_to_float(token: str) -> float:
    """Inverse of `format_numeric_token`."""
    sign = -1.0 if token.startswith('m') else 1.0
    body = token[1:] if token.startswith('m') else token
    body = body.replace('em', 'e-').replace('p', '.')
    return sign * float(body)


def _parse_filter(stem: str) -> Dict[str, float]:
    for pattern, key in _STEM_PATTERNS:
        match = pattern.match(stem)
        if match:
            return {key: _token_to_float(match.group('v'))}
    return {}


def _is_conflict(path: Path) -> bool:
    return _CONFLICT in path.name


def _stems(locator: 'RunLocator') -> List[str]:
    """Deck stems: `system*` first, then the rest, each sorted."""
    stems = sorted(p.stem for p in locator.lammps_dir.glob('*.in')
                   if not _is_conflict(p))
    return ([s for s in stems if s.startswith('system')]
            + [s for s in stems if not s.startswith('system')])


def deck_log(locator: 'RunLocator', deck_name: str) -> Optional[Path]:
    """The log of one deck, or None.

    Tried in order: `lammps/run_<stem>.log` (array mode); `lammps/run_system.log`
    or `lammps/run_slide.log` (combined mode); the single `log*.lammps` in the
    run root (local run) when the run has exactly one deck.
    """
    lammps_dir = locator.lammps_dir
    combined = 'run_system.log' if deck_name.startswith('system') else 'run_slide.log'
    for name in (f'run_{deck_name}.log', combined):
        candidate = lammps_dir / name
        if candidate.is_file():
            return candidate
    if len(_stems(locator)) == 1:
        local = [p for p in locator.run_dir.glob('log*.lammps')
                 if not _is_conflict(p)]
        if len(local) == 1:
            return local[0]
    return None


def discover_decks(locator: 'RunLocator') -> List[Deck]:
    """Every deck of the run, system decks first, then sorted by name."""
    return [Deck(name=stem,
                 path=locator.lammps_dir / f'{stem}.in',
                 log_path=deck_log(locator, stem),
                 point_filter=_parse_filter(stem),
                 writes_structures=stem.startswith('system'))
            for stem in _stems(locator)]


def points_for(deck: Deck, points: List['LoadPoint']) -> List['LoadPoint']:
    """The load points this deck covers (all of them for an empty filter)."""
    return [p for p in points
            if all(abs(getattr(p, key) - value) <= _TOL
                   for key, value in deck.point_filter.items())]
