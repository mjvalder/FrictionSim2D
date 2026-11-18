"""The ruling ledger of a set: every user ruling of every rulings file, by point.

Plain dicts only -- no AiiDA needed; :mod:`src.aiida.steps` stores the ledger
on the set Group (extra ``friction2d_rulings_ledger``) and reconciles the
store with it.

The ledger is ``{point id: {source path: entry}}``. The point id is the
stable ``integrity_rulings.format_point_id`` (never a node pk), so it
survives a re-import that supersedes the point's deck. An entry is::

    {ruling, note, date, override,     # the ruling, as its file says it
     own_file, author,                 # where it came from, who made it
     seen_at,                          # first added, or last changed in content
     status}                           # 'present' or 'missing' (its file is gone)

* :func:`read_source` -- a source's rulings file replaces that source's
  entries; a missing file marks them ``missing`` and keeps them.
* :func:`desired_ruling` -- the source whose entry should be in force: the
  set's own file, else the most recently seen other source.
* :func:`conflicts` -- the sources of one point that disagree.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

#: The fields of an entry that are the ruling itself.
CONTENT = ('ruling', 'note', 'date', 'override')

Ledger = Dict[str, Dict[str, Dict[str, Any]]]

_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


class Clock:  # pylint: disable=too-few-public-methods
    """Strictly increasing UTC timestamps (one fixed-width text format, so
    they order as strings) for one pass: ``start`` (default now), then one
    microsecond later on every call. A source read later in a pass is seen
    later, so ``seen_at`` never ties within one."""

    def __init__(self, start: Optional[str] = None) -> None:
        self._next = (datetime.strptime(start, _FORMAT).replace(tzinfo=timezone.utc)
                      if start else datetime.now(timezone.utc))

    def __call__(self) -> str:
        stamp = self._next.strftime(_FORMAT)
        self._next += timedelta(microseconds=1)
        return stamp


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
        same = old is not None and all(old.get(k) == content[k] for k in CONTENT)
        ledger.setdefault(pid, {})[source] = {
            **content, 'own_file': own, 'author': 'user',
            'seen_at': old['seen_at'] if same else now, 'status': 'present'}
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


def desired_ruling(entries: Mapping[str, Mapping[str, Any]], own_source: Optional[str],
                   applies: Optional[Callable[[str, Mapping[str, Any]], bool]] = None
                   ) -> Optional[str]:
    """The source whose entry should be in force for one point: the first of
    :func:`ruling_order` that ``applies`` (default: every one), or None."""
    for source in ruling_order(entries, own_source):
        if applies is None or applies(source, entries[source]):
            return source
    return None


def conflicts(entries: Mapping[str, Mapping[str, Any]], own_source: Optional[str],
              in_force: Optional[str]
              ) -> Optional[Tuple[Optional[Tuple[str, str]], str, List[Tuple[str, str]]]]:
    """None when the sources of one point agree (one decision, ``pass`` or
    ``fail``); else ``((source, ruling) in force or None, why, [(source,
    ruling) of every source that disagrees with it])``. ``in_force`` is
    :func:`desired_ruling`'s choice."""
    if len({e['ruling'] for e in entries.values()}) < 2:
        return None
    order = ruling_order(entries, own_source)
    if in_force is None:
        return None, 'none of them applies', [(s, entries[s]['ruling']) for s in order]
    decision = entries[in_force]['ruling']
    if in_force == own_source:
        why = "the set's own integrity_rulings.yaml wins"
    else:
        why = f"most recently seen ({entries[in_force].get('seen_at', '')})"
        stronger = order[:order.index(in_force)]
        if stronger:
            why += f'; {", ".join(stronger)} does not apply'
    others = [(s, entries[s]['ruling']) for s in order
              if s != in_force and entries[s]['ruling'] != decision]
    return (in_force, decision), why, others
