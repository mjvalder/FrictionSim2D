"""Which verdict counts: the version policy and the ruling ledger.

Pure Python, no AiiDA import, in two parts.

**The version policy** -- what "latest" verdict and summary mean (critique
I4): :class:`Policy` and :func:`select`.

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
verdict and summary generation it used.

**The ruling ledger** of a set -- every user ruling of every rulings file,
by point. Plain dicts: `src.aiida.steps` stores the ledger on the set Group
(extra ``friction2d_rulings_ledger``) and reconciles the store with it.

The ledger is ``{point id: {source path: entry}}``. The point id is the
stable ``integrity_rulings.format_point_id`` (never a node pk), so it
survives a re-import that supersedes the point's deck. An entry is::

    {ruling, note, date, override,     # the ruling, as its file says it
     own_file, author,                 # where it came from, who made it
     seen_at,                          # first added, or its decision last changed
     status,                           # 'present', 'missing' (its file is gone) or
                                       # 'archived' (from an archive: no file to read)
     in_force}                         # True on the entry the last pass put in force

* :func:`read_source` -- a source's rulings file replaces that source's
  entries; a missing file marks them ``missing`` and keeps them.
* :func:`ruling_order` -- the sources of one point, strongest first: the
  set's own file, else the most recently seen other source. The entry in
  force is the first of them that applies (``steps._reconcile``).
* :func:`conflicts` -- the sources of one point that disagree.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

__all__ = ['NEWEST', 'Policy', 'select', 'verdict_kind',
           'CONTENT', 'DECISION', 'Ledger', 'Clock', 'stamp', 'content_of', 'read_source',
           'drop_source', 'ruling_order', 'conflicts', 'why_in_force']

# ---------------------------------------------------------------------------
# The version policy
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# The ruling ledger
# ---------------------------------------------------------------------------

#: The fields of an entry that are the ruling itself.
CONTENT = ('ruling', 'note', 'date', 'override')
#: The fields whose change makes an entry newly seen (re-ranks it): the
#: decision, not its wording -- fixing a typo in a note never flips a point.
DECISION = ('ruling', 'override')

Ledger = Dict[str, Dict[str, Dict[str, Any]]]

_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


def stamp(when: datetime) -> str:
    """``when`` as a ledger timestamp (UTC, fixed width: orders as text)."""
    return when.astimezone(timezone.utc).strftime(_FORMAT)


class Clock:  # pylint: disable=too-few-public-methods
    """Strictly increasing UTC timestamps (one fixed-width text format, so
    they order as strings) for one pass: ``start`` (default now), then one
    microsecond later on every call. A source read later in a pass is seen
    later, so ``seen_at`` never ties within one."""

    def __init__(self, start: Optional[str] = None) -> None:
        self._next = (datetime.strptime(start, _FORMAT).replace(tzinfo=timezone.utc)
                      if start else datetime.now(timezone.utc))

    def __call__(self) -> str:
        now = stamp(self._next)
        self._next += timedelta(microseconds=1)
        return now


def content_of(ruling) -> Dict[str, Any]:
    """An ``integrity_rulings.Ruling`` as the ledger's :data:`CONTENT`."""
    return {'ruling': ruling.ruling, 'note': ruling.note, 'date': ruling.date,
            'override': bool(ruling.override)}


def read_source(ledger: Ledger, source: str, rulings: Optional[Mapping[str, Any]], *,
                own: bool, now: str, warnings: List[str]) -> List[str]:
    """Update ``ledger`` in place from ``source``'s rulings; return the point
    ids whose entry the file withdrew (no longer names).

    ``rulings`` is ``integrity_rulings.load_rulings`` of the file, or None
    when the file does not exist: its entries are kept, marked ``missing``.
    An unreadable file (``<file>``) changes nothing. A malformed entry is
    skipped and the others count, but then nothing is withdrawn: the
    unreadable entry may be the one that ruled the missing point. Warnings
    go to ``warnings``.
    """
    if rulings is None:
        for by_source in ledger.values():
            if source in by_source:
                by_source[source]['status'] = 'missing'
        return []
    if '<file>' in rulings:
        warnings.append(f'rulings file {source} not read ({rulings["<file>"].error}); '
                        f'its rulings are kept as they were')
        return []
    partial = False
    for key, ruling in rulings.items():
        if ruling.error:
            warnings.append(f'integrity ruling ignored: {source}: {ruling.error}')
            partial = partial or key.startswith('<malformed')
    valid = {pid: r for pid, r in rulings.items() if not r.error}
    retracted: List[str] = []
    for pid in sorted(ledger):
        if source in ledger[pid] and pid not in valid and not partial:
            del ledger[pid][source]
            retracted.append(pid)
            if not ledger[pid]:
                del ledger[pid]
    if partial:
        warnings.append(f'no ruling of {source} withdrawn: it has entries that cannot be '
                        f'read')
    for pid, ruling in valid.items():
        content = content_of(ruling)
        old = ledger.get(pid, {}).get(source)
        same = old is not None and all(old.get(k) == content[k] for k in DECISION)
        entry = {**content, 'own_file': own, 'author': 'user',
                 'seen_at': old['seen_at'] if same else now, 'status': 'present'}
        if old is not None and old.get('in_force'):
            entry['in_force'] = True
        ledger.setdefault(pid, {})[source] = entry
    return retracted


def drop_source(ledger: Ledger, source: str) -> List[str]:
    """Delete every entry of ``source`` from ``ledger``; return their point ids."""
    dropped = []
    for pid in sorted(ledger):
        if ledger[pid].pop(source, None) is not None:
            dropped.append(pid)
            if not ledger[pid]:
                del ledger[pid]
    return dropped


def ruling_order(entries: Mapping[str, Mapping[str, Any]],
                 own_source: Optional[str]) -> List[str]:
    """The sources of one point's ``entries``, the strongest first: the set's
    own file, then the others by ``seen_at``, the latest first (ties by
    path, so the order is total). A ``missing`` source counts like any other."""
    others = sorted((s for s in entries if s != own_source),
                    key=lambda s: (str(entries[s].get('seen_at', '')), s), reverse=True)
    return ([own_source] if own_source in entries else []) + others


def conflicts(entries: Mapping[str, Mapping[str, Any]], own_source: Optional[str],
              in_force: Optional[str]
              ) -> Optional[Tuple[Optional[Tuple[str, str]], str, List[Tuple[str, str]]]]:
    """None when the sources of one point agree (one decision, ``pass`` or
    ``fail``); else ``((source, ruling) in force or None, why, [(source,
    ruling) of every source that disagrees with it])``. ``in_force`` is
    the first source of :func:`ruling_order` that applies, or None."""
    if len({e['ruling'] for e in entries.values()}) < 2:
        return None
    order = ruling_order(entries, own_source)
    why = why_in_force(entries, own_source, in_force)
    if in_force is None:
        return None, why, [(s, entries[s]['ruling']) for s in order]
    decision = entries[in_force]['ruling']
    others = [(s, entries[s]['ruling']) for s in order
              if s != in_force and entries[s]['ruling'] != decision]
    return (in_force, decision), why, others


def why_in_force(entries: Mapping[str, Mapping[str, Any]], own_source: Optional[str],
                 in_force: Optional[str]) -> str:
    """Why ``in_force`` (the first applying source of :func:`ruling_order`,
    or None) is the
    ruling in force among one point's ``entries``."""
    if not entries:
        return 'no file rules it'
    if in_force is None:
        return 'no ruling applies'
    if in_force == own_source:
        return "the set's own integrity_rulings.yaml wins"
    order = ruling_order(entries, own_source)
    why = f"most recently seen ({entries[in_force].get('seen_at', '')})"
    stronger = order[:order.index(in_force)]
    if stronger:
        why += f'; {", ".join(stronger)} does not apply'
    return why
