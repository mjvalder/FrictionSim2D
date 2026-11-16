"""Field-by-field comparison of two runs' (or campaigns') parameter blocks.

Pure functions over the canonical block contents of
:mod:`src.aiida.blocks` (flat dotted field path -> value), so it runs without
aiida-core; :mod:`src.aiida.compare` loads the blocks out of the store.

A block elides every field equal to its family default, so "absent" and
"default" are the same value: :func:`compare_blocks` refills absent fields from
``defaults`` before diffing. Builder-computed ``derived`` values are not part
of identity and are compared separately, to a relative tolerance.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = ['DEFAULT_ATOL', 'DEFAULT_RTOL', 'VARIES', 'CompareReport', 'compare_blocks',
           'canonical', 'merge_varying', 'real_key', 'refill',
           'same_value']

#: Stands for a field that takes several values within a campaign.
VARIES = '<varies>'

DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9

#: Content sections holding one entry per index (``sheets.0.x``); a default
#: for one cannot be placed without an index. Mirrors ``block_schema.LIST_SECTIONS``.
_LIST_SECTIONS = ('sheets', 'flake_sheets')

_EMPTY: Mapping[str, Any] = MappingProxyType({})

Diff = Tuple[str, Any, Any, str]


@dataclass
class CompareReport:
    """What differs between two parameter sets.

    Attributes:
        identical: Names of the blocks present in both with every field equal.
        differs: Block name -> ``(field path, a, b, unit)`` per differing
            field. An absent field without a default shows as ``None``.
        only_a, only_b: Names of the blocks present on one side only.
        derived_differs: ``(derived key, a, b, unit)`` for derived values
            apart by more than ``rtol``/``atol``.
        unit_differs: Block name (``derived`` for derived values) ->
            ``(field path, unit A, unit B)`` where the two sides store the
            field in different units; such a field is never compared.
    """
    identical: List[str] = field(default_factory=list)
    differs: Dict[str, List[Diff]] = field(default_factory=dict)
    only_a: List[str] = field(default_factory=list)
    only_b: List[str] = field(default_factory=list)
    derived_differs: List[Diff] = field(default_factory=list)
    unit_differs: Dict[str, List[Tuple[str, str, str]]] = field(default_factory=dict)

    @property
    def same(self) -> bool:
        """True when nothing differs at all."""
        return not (self.differs or self.only_a or self.only_b or self.derived_differs
                    or self.unit_differs)

    def render(self) -> str:
        """A plain-text report, with units."""
        lines: List[str] = []

        def row(path: str, a: Any, b: Any, unit: str) -> str:
            suffix = f' {unit}' if unit else ''
            return f'  {path}: {a!r} -> {b!r}{suffix}'

        for name in sorted(self.differs):
            lines.append(f'{name}:')
            lines.extend(row(*d) for d in self.differs[name])
        for label, names in (('only in A', self.only_a), ('only in B', self.only_b)):
            if names:
                lines.append(f'{label}: {", ".join(sorted(names))}')
        if self.derived_differs:
            lines.append('derived (not part of identity):')
            lines.extend(row(*d) for d in self.derived_differs)
        for name in sorted(self.unit_differs):
            lines.append(f'{name} (unit differs, not compared):')
            lines.extend(f'  {path}: unit differs: A={ua or "-"} B={ub or "-"}, not compared'
                         for path, ua, ub in self.unit_differs[name])
        if self.identical:
            lines.append(f'identical blocks: {", ".join(sorted(self.identical))}')
        if self.same:
            lines.append('no differences')
        return '\n'.join(lines)


def _unindexed(path: str) -> str:
    """``sheets.0.x`` -> ``sheets.x``, the schema path of a content path."""
    return '.'.join(part for part in path.split('.') if not part.isdigit())


def _fields(block: Mapping[str, Any]) -> Mapping[str, Any]:
    """The flat fields of a block given as content (``{'kind', 'fields'}``) or bare."""
    inner = block.get('fields')
    return inner if isinstance(inner, Mapping) and 'kind' in block else block


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def real_key(value: float, rtol: float = DEFAULT_RTOL) -> float:
    """``value`` rounded to the significant figures ``rtol`` resolves
    (``1e-6`` -> 6): one canonical number per tolerance class, so
    ``0.1 + 0.2`` and ``0.3`` key alike while ``1.0`` and ``1.001`` do not.

    :func:`src.aiida.query.related` keys reals with this, and
    :func:`compare_blocks` treats two reals with one key as equal, so the two
    never disagree on whether a value changed.
    """
    value = float(value)
    if value == 0.0 or not math.isfinite(value):
        return value
    digits = max(1, math.ceil(-math.log10(rtol)))
    return float(f'{value:.{digits}g}')


def canonical(value: Any, rtol: float = DEFAULT_RTOL) -> Any:
    """``value`` with every real (inside lists and dicts too) replaced by its
    :func:`real_key`: the form two requested values are equal in."""
    if _is_number(value):
        return real_key(value, rtol)
    if isinstance(value, Mapping):
        return {k: canonical(v, rtol) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v, rtol) for v in value]
    return value


def same_value(a: Any, b: Any, rtol: float = DEFAULT_RTOL) -> bool:
    """Equal after :func:`canonical`: the equality of a requested field in
    :func:`compare_blocks`, :func:`merge_varying` and ``related()``."""
    return canonical(a, rtol) == canonical(b, rtol)


def _close(a: Any, b: Any, rtol: float, atol: float = DEFAULT_ATOL) -> bool:
    """Equal, numbers within ``max(atol, rtol * max(|a|, |b|))``, or numbers
    with one :func:`real_key`."""
    if _is_number(a) and _is_number(b):
        return (abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))
                or real_key(a, rtol) == real_key(b, rtol))
    return a == b


def _visible(name: str, fields: Iterable[Mapping[str, Any]],
             defaults: Iterable[Mapping[str, Any]], kinds: Mapping[str, str]) -> List[str]:
    """Paths to compare in block ``name``: every written field, plus any field
    of the block's kind some side carries a default for (so two different
    defaults, each elided, are seen)."""
    paths = {p for f in fields for p in f}
    written = {_unindexed(p) for p in paths}
    kind = name.partition('__')[0]
    for table in defaults:
        for path in table:
            if (kinds.get(path) == kind and path not in written
                    and path.partition('.')[0] not in _LIST_SECTIONS):
                paths.add(path)
    return sorted(paths)


def refill(name: str, fields: Mapping[str, Any], defaults: Mapping[str, Any],
           kinds: Mapping[str, str]) -> Dict[str, Any]:
    """Block ``name``'s ``fields`` with every elided field of its kind written
    back from ``defaults``: the values :func:`compare_blocks` diffs.

    :mod:`src.aiida.query` builds its point table from this, so ``related()``
    and ``compare`` read "absent" as "default" by one rule.
    """
    out = dict(_fields(fields))
    for path in _visible(name, (out,), (defaults,), kinds):
        if path not in out:
            out[path] = defaults.get(_unindexed(path))
    return out


def compare_blocks(a: Mapping[str, dict], b: Mapping[str, dict], *,
                   units: Mapping[str, str],
                   derived_a: Mapping[str, Any] = _EMPTY,
                   derived_b: Mapping[str, Any] = _EMPTY,
                   defaults: Mapping[str, Any] = _EMPTY,
                   rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
                   units_b: Optional[Mapping[str, str]] = None,
                   defaults_b: Optional[Mapping[str, Any]] = None,
                   kinds: Mapping[str, str] = _EMPTY) -> CompareReport:
    """Compare two parameter sets field by field.

    Each side is read against its own schema: an elided field is refilled from
    that side's defaults, and a field stored in different units on the two
    sides is reported in ``unit_differs`` and never compared.

    Args:
        a, b: Block name -> flat fields (or the full block content).
        units, defaults: Side A's schema path (``general.temp``; derived keys
            as ``derived.geometry.lat_c``) -> unit / default. Side B's too,
            unless ``units_b`` / ``defaults_b`` are given.
        derived_a, derived_b: Derived values, flat.
        rtol, atol: Tolerance for derived values (block fields compare by
            :func:`same_value`, to the default rtol):
            ``|a-b| <= max(atol, rtol * max(|a|, |b|))``.
        kinds: Schema path -> block kind, to see fields both sides elide
            at different defaults.
    """
    units_b = units if units_b is None else units_b
    defaults_b = defaults if defaults_b is None else defaults_b
    report = CompareReport()
    report.only_a = sorted(set(a) - set(b))
    report.only_b = sorted(set(b) - set(a))
    for name in sorted(set(a) & set(b)):
        fa, fb = _fields(a[name]), _fields(b[name])
        diffs: List[Diff] = []
        mismatched: List[Tuple[str, str, str]] = []
        for path in _visible(name, (fa, fb), (defaults, defaults_b), kinds):
            sp = _unindexed(path)
            ua, ub = units.get(sp) or '', units_b.get(sp) or ''
            if ua != ub:
                mismatched.append((path, ua, ub))
                continue
            va = fa[path] if path in fa else defaults.get(sp)
            vb = fb[path] if path in fb else defaults_b.get(sp)
            if not same_value(va, vb):
                diffs.append((path, va, vb, ua))
        if diffs:
            report.differs[name] = diffs
        if mismatched:
            report.unit_differs[name] = mismatched
        if not diffs and not mismatched:
            report.identical.append(name)
    for key in sorted(set(derived_a) | set(derived_b)):
        ua = units.get(f'derived.{key}') or units.get(key) or ''
        ub = units_b.get(f'derived.{key}') or units_b.get(key) or ''
        if ua != ub:
            report.unit_differs.setdefault('derived', []).append((key, ua, ub))
            continue
        va, vb = derived_a.get(key), derived_b.get(key)
        if not _close(va, vb, rtol, atol):
            report.derived_differs.append((key, va, vb, ua))
    return report


def merge_varying(members: Iterable[Tuple[Mapping[str, dict], Mapping[str, Any],
                                          Mapping[str, Any]]], *,
                  rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
                  kinds: Mapping[str, str] = _EMPTY,
                  ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """The fields shared by every member; one that varies becomes :data:`VARIES`.

    Args:
        members: ``(blocks, derived, defaults)`` per run. An absent field counts
            as that member's own default, so a member that elided it agrees
            with one that wrote it. Derived values agree within ``rtol``/``atol``,
            exactly as :func:`compare_blocks` judges them.
        kinds: Schema path -> block kind (see :func:`compare_blocks`).
    """
    listed = [({n: dict(_fields(c)) for n, c in blocks.items()}, dict(derived),
               dict(table)) for blocks, derived, table in members]
    if not listed:
        raise ValueError('nothing to merge')
    merged: Dict[str, Dict[str, Any]] = {}
    for name in sorted({n for blocks, _, _ in listed for n in blocks}):
        have = [(blocks[name], table) for blocks, _, table in listed if name in blocks]
        out: Dict[str, Any] = {}
        for path in _visible(name, [f for f, _ in have], [t for _, t in have], kinds):
            sp = _unindexed(path)
            values = [f[path] if path in f else t.get(sp) for f, t in have]
            out[path] = (values[0] if all(same_value(v, values[0]) for v in values)
                         else VARIES)
        merged[name] = out
    derived: Dict[str, Any] = {}
    for key in sorted({k for _, d, _ in listed for k in d}):
        values = [d.get(key) for _, d, _ in listed]
        derived[key] = (values[0] if all(_close(v, values[0], rtol, atol) for v in values)
                        else VARIES)
    return merged, derived
