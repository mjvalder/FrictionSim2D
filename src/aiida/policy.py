"""The version policy: what "latest" verdict and summary mean (critique I4).

Selection is by DECLARED version, never by creation time: an imported node
keeps its origin ``ctime``, so "latest by ctime" depends on clocks and import
order, and a verdict re-run with older thresholds (to reproduce a paper)
would silently win over a newer one.

* ``'newest'`` -- the highest semver ``thresholds_version`` (verdicts), then
  the highest semver ``code_version``; within one generation a ruled verdict
  outranks a withdrawn one, which outranks the checker's own; then the uuid,
  so the choice is total and reproducible.
* any other string pins that version: an exact ``thresholds_version``
  (verdicts) or ``code_version`` (summaries) match, else nothing.

A dataset stores its :class:`Policy` (``as_dict``), so a figure names the
verdict and summary generation it used. Pure Python: no AiiDA import.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

__all__ = ['NEWEST', 'Policy', 'select', 'verdict_kind']

NEWEST = 'newest'

#: A verdict candidate's kind -> its rank within one version generation.
_RANK = {'verdict': 0, 'withdrawn_verdict': 1, 'ruled_verdict': 2}

_PART = re.compile(r'^(\d*)(.*)$')


@dataclass(frozen=True)
class Policy:
    """Which verdict and which summary generation a selection takes:
    ``'newest'`` or a pinned version string."""
    verdict: str = NEWEST
    summary: str = NEWEST

    def as_dict(self) -> dict:
        """``{'verdict': ..., 'summary': ...}``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'Policy':
        """The policy ``as_dict`` wrote (missing keys take ``'newest'``)."""
        return cls(verdict=str(data.get('verdict', NEWEST)),
                   summary=str(data.get('summary', NEWEST)))


def _semver(version: Any) -> Tuple[Tuple[int, str], ...]:
    """``'1.10.0'`` -> ``((1, ''), (10, ''), (0, ''))``; a part's non-numeric
    tail (``'0rc1'``) sorts after its number. ``None`` sorts first."""
    if version is None:
        return ()
    parts = []
    for part in str(version).split('.'):
        digits, tail = _PART.match(part).groups()
        parts.append((int(digits) if digits else -1, tail))
    return tuple(parts)


def verdict_kind(candidate: Mapping[str, Any]) -> str:
    """``'verdict'``, ``'withdrawn_verdict'`` or ``'ruled_verdict'``: the
    candidate's ``kind`` key (the ``friction2d_kind`` extra), else read from
    its content (``withdrawn_ruling`` / ``ruled_by_user``)."""
    kind = candidate.get('kind')
    if kind in _RANK:
        return kind
    if candidate.get('withdrawn_ruling'):
        return 'withdrawn_verdict'
    if candidate.get('ruled_by_user'):
        return 'ruled_verdict'
    return 'verdict'


def _verdict_key(candidate: Mapping[str, Any]):
    return (_semver(candidate.get('thresholds_version')),
            _semver(candidate.get('code_version')),
            _RANK[verdict_kind(candidate)], str(candidate.get('uuid', '')))


def _summary_key(candidate: Mapping[str, Any]):
    return (_semver(candidate.get('code_version')), str(candidate.get('uuid', '')))


def select(candidates: Sequence[Mapping[str, Any]],
           kind: Literal['verdict', 'summary'],
           policy: Policy) -> Optional[Mapping[str, Any]]:
    """The candidate ``policy`` selects among ``candidates`` (the live
    verdicts -- checker, ruled and withdrawn -- or the live summaries of one
    point), or None. Never by ``ctime``."""
    if kind == 'verdict':
        pinned, field, key = policy.verdict, 'thresholds_version', _verdict_key
    elif kind == 'summary':
        pinned, field, key = policy.summary, 'code_version', _summary_key
    else:
        raise ValueError(f'unknown selection kind {kind!r} (verdict or summary)')
    pool = list(candidates)
    if pinned != NEWEST:
        pool = [c for c in pool if str(c.get(field)) == pinned]
    return max(pool, key=key) if pool else None
