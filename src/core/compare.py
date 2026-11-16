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

import numbers
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Tuple

__all__ = ['VARIES', 'CompareReport', 'compare_blocks', 'merge_varying']

#: Stands for a field that takes several values within a campaign.
VARIES = '<varies>'

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
            apart by more than ``rtol``.
    """
    identical: List[str] = field(default_factory=list)
    differs: Dict[str, List[Diff]] = field(default_factory=dict)
    only_a: List[str] = field(default_factory=list)
    only_b: List[str] = field(default_factory=list)
    derived_differs: List[Diff] = field(default_factory=list)

    @property
    def same(self) -> bool:
        """True when nothing differs at all."""
        return not (self.differs or self.only_a or self.only_b or self.derived_differs)

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


def _close(a: Any, b: Any, rtol: float) -> bool:
    if _is_number(a) and _is_number(b):
        return abs(a - b) <= rtol * max(abs(a), abs(b))
    return a == b


def _block_diffs(a: Mapping[str, Any], b: Mapping[str, Any],
                 units: Mapping[str, str], defaults: Mapping[str, Any]) -> List[Diff]:
    out: List[Diff] = []
    for path in sorted(set(a) | set(b)):
        schema_path = _unindexed(path)
        fallback = defaults.get(schema_path)
        va = a[path] if path in a else fallback
        vb = b[path] if path in b else fallback
        if va != vb:
            out.append((path, va, vb, units.get(schema_path) or ''))
    return out


def compare_blocks(a: Mapping[str, dict], b: Mapping[str, dict], *,
                   units: Mapping[str, str],
                   derived_a: Mapping[str, Any] = _EMPTY,
                   derived_b: Mapping[str, Any] = _EMPTY,
                   defaults: Mapping[str, Any] = _EMPTY,
                   rtol: float = 1e-6) -> CompareReport:
    """Compare two parameter sets field by field.

    Args:
        a, b: Block name -> flat fields (or the full block content).
        units: Schema path (``general.temp``; derived keys as
            ``derived.geometry.lat_c``) -> unit.
        derived_a, derived_b: Derived values, flat.
        defaults: Schema path -> default, to refill elided fields.
        rtol: Relative tolerance for derived values (blocks compare exactly).
    """
    report = CompareReport()
    report.only_a = sorted(set(a) - set(b))
    report.only_b = sorted(set(b) - set(a))
    for name in sorted(set(a) & set(b)):
        diffs = _block_diffs(_fields(a[name]), _fields(b[name]), units, defaults)
        if diffs:
            report.differs[name] = diffs
        else:
            report.identical.append(name)
    for key in sorted(set(derived_a) | set(derived_b)):
        va, vb = derived_a.get(key), derived_b.get(key)
        if not _close(va, vb, rtol):
            unit = units.get(f'derived.{key}') or units.get(key) or ''
            report.derived_differs.append((key, va, vb, unit))
    return report


def merge_varying(members: Iterable[Tuple[Mapping[str, dict], Mapping[str, Any]]], *,
                  defaults: Mapping[str, Any] = _EMPTY,
                  ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """The fields shared by every member; one that varies becomes :data:`VARIES`.

    Args:
        members: ``(blocks, derived)`` per run. An absent field counts as its
            default, so a member that elided it agrees with one that wrote it.
    """
    listed = [({n: dict(_fields(c)) for n, c in blocks.items()}, dict(derived))
              for blocks, derived in members]
    if not listed:
        raise ValueError('nothing to merge')
    names = sorted({n for blocks, _ in listed for n in blocks})
    merged: Dict[str, Dict[str, Any]] = {}
    for name in names:
        have = [blocks[name] for blocks, _ in listed if name in blocks]
        out: Dict[str, Any] = {}
        for path in sorted({p for fields in have for p in fields}):
            fallback = defaults.get(_unindexed(path))
            values = [f[path] if path in f else fallback for f in have]
            out[path] = values[0] if all(v == values[0] for v in values) else VARIES
        merged[name] = out
    derived: Dict[str, Any] = {}
    for key in sorted({k for _, d in listed for k in d}):
        values = [d.get(key) for _, d in listed]
        derived[key] = (values[0] if all(_close(v, values[0], 1e-9) for v in values)
                        else VARIES)
    return merged, derived
