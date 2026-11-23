"""Geometry read from LAMMPS snapshots, for the sheet-profile and integrity plots.

Pure functions; the snapshot itself is read with :func:`.integrity.read_snapshot`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np

_GROUP_RE = re.compile(r'group (\S+) type ([\d* ]+)')


def expand_lammps_types(spec: str) -> Set[int]:
    """``'7 8 11*14'`` -> ``{7, 8, 11, 12, 13, 14}`` (a ``group ... type`` list)."""
    out: Set[int] = set()
    for tok in spec.split():
        a, _, b = tok.partition('*')
        out |= set(range(int(a), int(b or a) + 1))
    return out


def parse_type_groups(text: str) -> Dict[str, Set[int]]:
    """``{group: atom types}`` for every ``group <name> type ...`` line; the first wins."""
    groups: Dict[str, Set[int]] = {}
    for m in _GROUP_RE.finditer(text):
        groups.setdefault(m.group(1), expand_lammps_types(m.group(2)))
    return groups


def read_type_groups(settings_path: Path) -> Dict[str, Set[int]]:
    """:func:`parse_type_groups` of a settings file."""
    return parse_type_groups(Path(settings_path).read_text(encoding='utf-8'))


def radial_profile(xy: np.ndarray, z: np.ndarray, axis: np.ndarray, z_far: float,
                   edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Median height minus ``z_far`` in each radial bin around ``axis``.

    Returns ``(bin midpoints, profile)``; bins with fewer than 3 atoms are skipped.
    """
    r = np.linalg.norm(xy - axis, axis=1)
    mid, prof = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() >= 3:
            mid.append(0.5 * (a + b))
            prof.append(np.median(z[m]) - z_far)
    return np.array(mid), np.array(prof)
