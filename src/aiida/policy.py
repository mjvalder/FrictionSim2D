"""The version policy: what "latest" verdict and summary mean (critique I4).

Selection is by DECLARED version, never by creation time: an imported node
keeps its origin ``ctime``, so "latest by ctime" depends on clocks and import
order, and a verdict re-run with older thresholds (to reproduce a paper)
would silently win over a newer one.

Verdict order (controller ruling P2-R13; Task 8's query layer uses the same):

1. **Generation first**: the highest semver ``thresholds_version``, then the
   highest semver ``code_version``. A verdict of a newer generation wins
   over every verdict of an older one, a ruled one included.
2. **Within one generation**: ruled (``apply_ruling``) > withdrawn
   (``withdraw_ruling``) > the checker's own (``check_integrity``).
3. **Then the uuid**, so the choice is total and reproducible.

A ruling therefore does not carry over to a new generation by itself: the
rulings pass (``steps.apply_rulings``) re-applies it to the new generation's
checker verdict, as it does to any live verdict its point id names.

Summaries: summaries made with other options (``options_hash``, when one is
requested) are filtered out first; then the highest semver ``code_version``;
then the uuid.

Any policy string other than ``'newest'`` pins that version: an exact
``thresholds_version`` (verdicts) or ``code_version`` (summaries) match,
else nothing.

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
           policy: Policy, *,
           options_hash: Optional[str] = None) -> Optional[Mapping[str, Any]]:
    """The candidate ``policy`` selects among ``candidates`` (the live
    verdicts -- checker, ruled and withdrawn -- or the live summaries of one
    point), or None. Never by ``ctime``.

    ``options_hash`` (summaries only): keep only the candidates made with
    those options, before any version is compared."""
    if kind == 'verdict':
        pinned, field, key = policy.verdict, 'thresholds_version', _verdict_key
    elif kind == 'summary':
        pinned, field, key = policy.summary, 'code_version', _summary_key
    else:
        raise ValueError(f'unknown selection kind {kind!r} (verdict or summary)')
    pool = list(candidates)
    if options_hash is not None:
        if kind != 'summary':
            raise ValueError('options_hash selects summaries, not verdicts')
        pool = [c for c in pool if c.get('options_hash') == options_hash]
    if pinned != NEWEST:
        pool = [c for c in pool if str(c.get(field)) == pinned]
    return max(pool, key=key) if pool else None
