"""Potential-sweep type resolution (spec §13.3).

Neither the file extension nor the directory determines the pair style,
because LAMMPS variants share both: ``.sw`` serves ``sw`` AND ``sw/mod``;
``.tersoff`` serves ``tersoff``/``tersoff/mod``/...; ``.reax.*`` serves
``reaxff`` and legacy ``reax/c``. Resolution order per entry:

    1. explicit ``pot_type`` on the entry (always wins)
    2. parent directory under the potentials root, if it names a known type
    3. file extension, through EXTENSION_MAP
    4. otherwise FAIL THE BUILD

Variant styles (anything containing ``/``) are NEVER inferred. This matters
because the legacy fallback is not an error: ``is_sheet_lj`` logs "Unknown
potential type, assuming LJ is required" and proceeds — under a sweep that
would silently emit the wrong pair_style for an entire arm.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .potential_manager import (POTENTIALS_REQUIRING_LJ,
                                POTENTIALS_WITH_INTERNAL_LJ)

KNOWN_TYPES = POTENTIALS_REQUIRING_LJ | POTENTIALS_WITH_INTERNAL_LJ

#: Extension -> BASE family only. No entry ever yields a type with '/'.
EXTENSION_MAP = {
    '.sw': 'sw',
    '.tersoff': 'tersoff',
    '.rebo': 'rebo',
    '.rebomos': 'rebomos',
    '.airebo': 'airebo',
    '.meam': 'meam',
    '.extep': 'extep',
    '.vashishta': 'vashishta',
}


class PotentialTypeError(ValueError):
    """A sweep entry's potential type could not be resolved unambiguously."""


@dataclass(frozen=True)
class ResolvedPotential:
    path: str
    pot_type: str
    route: str   # 'explicit' | 'directory' | 'extension'


def resolve_pot_type(entry: Union[str, dict],
                     potentials_root: Path) -> ResolvedPotential:
    if isinstance(entry, dict):
        path = entry['path']
        explicit = entry.get('pot_type')
    else:
        path = entry
        explicit = None

    if explicit:
        pot_type = explicit.strip().lower()
        if pot_type not in KNOWN_TYPES:
            raise PotentialTypeError(
                f"{path}: explicit pot_type '{explicit}' is not a known "
                f"LAMMPS style (known: {sorted(KNOWN_TYPES)})")
        return ResolvedPotential(path, pot_type, 'explicit')

    parts = Path(path).parts
    for part in parts[:-1]:
        candidate = part.strip().lower()
        if candidate in KNOWN_TYPES and '/' not in candidate:
            return ResolvedPotential(path, candidate, 'directory')

    suffix = Path(path).suffix.lower()
    name = Path(path).name.lower()
    if '.reax.' in name or name.startswith('ffield'):
        raise PotentialTypeError(
            f"{path}: ReaxFF files serve both 'reaxff' and legacy 'reax/c' — "
            "state pot_type explicitly")
    if suffix in EXTENSION_MAP:
        return ResolvedPotential(path, EXTENSION_MAP[suffix], 'extension')

    raise PotentialTypeError(
        f"{path}: cannot resolve potential type from directory or extension "
        f"'{suffix}'. State pot_type explicitly.")


def pot_dir_token(pot_type: str, path: Union[str, Path]) -> str:
    """Output-directory segment: pot_<type-slug>_<stem> (spec §13.4).

    '/' becomes '-' (sw/mod -> sw-mod); the stem is sanitised so the token
    is always a single path segment.
    """
    slug = pot_type.strip().lower().replace('/', '-')
    stem = Path(path).name
    for ext in sorted(EXTENSION_MAP, key=len, reverse=True):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    # No known extension (e.g. ffield_ost.reax.MoS): keep the full name —
    # for ReaxFF files the trailing token is identity, not an extension.
    stem = ''.join(c if (c.isalnum() or c in '-_.') else '-' for c in stem)
    return f'pot_{slug}_{stem}'
