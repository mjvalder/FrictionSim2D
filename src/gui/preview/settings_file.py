"""Reading component membership and the element legend from ``system.in.settings``.

This file, written beside every generated run script, is the authoritative
source for what an atom type *is* after type offsets have been applied::

    mass 1*6 28.085000 #Si          group sub_all type 1 2 3
    mass 7  95.950000 #Mo           group tip_all type 4 5 6
    mass 9  1.0 #Virtual atom       group 2D_all  type 7 8

The per-component data files cannot serve this purpose: their ``Masses``
sections use component-local type ids that would have to be re-offset, and the
sheet-like models place no ``group`` labels in their scripts at all -- so
without this file, component visibility toggles would be impossible for them.

Standard library only: no Qt, no ovito.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

__all__ = ['Identity', 'parse_identity', 'expand_type_range']

_GROUP = re.compile(r'^\s*group\s+(?P<name>\S+)\s+type\s+(?P<types>.+?)\s*$')
_MASS = re.compile(
    r'^\s*mass\s+(?P<types>\S+)\s+(?P<mass>[\d.eE+-]+)\s*(?:#\s*(?P<label>.*))?$'
)

#: Groups naming a whole component, as opposed to a region within one
#: (``sub_fix``, ``tip_thermo``) or a layer (``layer_1``).
_COMPONENT_SUFFIX = '_all'


def expand_type_range(spec: str) -> List[int]:
    """Expand a LAMMPS type specifier into explicit ids.

    Accepts a single id (``7``) or an inclusive span (``1*6``). LAMMPS has no
    list form for ``mass``, which is why non-contiguous element types have to be
    emitted as several lines.
    """
    spec = spec.strip()
    if '*' in spec:
        lo_text, _, hi_text = spec.partition('*')
        try:
            lo, hi = int(lo_text), int(hi_text)
        except ValueError:
            return []
        return list(range(lo, hi + 1)) if lo <= hi else []
    try:
        return [int(spec)]
    except ValueError:
        return []


@dataclass
class Identity:
    """What each atom type is, and which component it belongs to."""

    #: group name -> the type ids it contains
    groups: Dict[str, List[int]] = field(default_factory=dict)
    #: type id -> element label, after last-wins resolution
    elements: Dict[int, str] = field(default_factory=dict)
    #: type id -> mass
    masses: Dict[int, float] = field(default_factory=dict)
    #: ranges that overlapped, as (earlier_label, later_label, shared ids)
    overlaps: List[Tuple[str, str, List[int]]] = field(default_factory=list)

    @property
    def component_groups(self) -> Dict[str, List[int]]:
        """Whole-component groups only (``sub_all``, ``tip_all``, ``2D_all``...).

        These drive the visibility toggles; region groups such as ``sub_fix``
        are subsets and would double-count.
        """
        return {
            name[: -len(_COMPONENT_SUFFIX)]: types
            for name, types in self.groups.items()
            if name.endswith(_COMPONENT_SUFFIX)
        }

    @property
    def layer_groups(self) -> Dict[str, List[int]]:
        """Per-layer groups, present for multilayer sheets."""
        return {n: t for n, t in self.groups.items() if n.startswith('layer_')}

    def group_of(self, type_id: int) -> Optional[str]:
        """Which component a type belongs to."""
        for name, types in self.component_groups.items():
            if type_id in types:
                return name
        return None

    def legend(self, present_types: Optional[Set[int]] = None) -> Dict[int, str]:
        """Type-to-element mapping for display.

        Args:
            present_types: Restrict to types that actually have atoms. Driving
                and virtual pseudo-atoms are declared here but contribute no
                atoms to the assembly, so showing them would be misleading.
        """
        if present_types is None:
            return dict(self.elements)
        return {t: e for t, e in self.elements.items() if t in present_types}


def parse_identity(settings_file: Path) -> Identity:
    """Parse ``system.in.settings``.

    Mass lines are applied in file order with later lines winning, which is
    LAMMPS' own rule. Spans are disjoint as of the per-contiguous-run fix in
    ``potential_manager.get_masses_string``; any overlap is recorded in
    :attr:`Identity.overlaps` as a regression signal rather than silently
    resolved.
    """
    identity = Identity()
    if not Path(settings_file).exists():
        return identity

    claimed: Dict[int, str] = {}

    for line in Path(settings_file).read_text(encoding='utf-8').splitlines():
        match = _GROUP.match(line)
        if match:
            types = []
            for token in match.group('types').split():
                types.extend(expand_type_range(token))
            identity.groups[match.group('name')] = types
            continue

        match = _MASS.match(line)
        if match:
            label = (match.group('label') or '').strip()
            mass = float(match.group('mass'))
            types = expand_type_range(match.group('types'))

            overlapping = [t for t in types if t in claimed and claimed[t] != label]
            if overlapping:
                identity.overlaps.append((claimed[overlapping[0]], label, overlapping))

            for type_id in types:
                identity.elements[type_id] = label
                identity.masses[type_id] = mass
                claimed[type_id] = label

    return identity
