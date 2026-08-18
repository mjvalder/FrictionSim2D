"""Deriving a material sweep from a structure and a potential directory.

The material list is the intersection of the two: a material can be run only
if both its structure file and its potential file exist. ``preflight_materials``
in ``src.core.run`` computes exactly this at build time, dropping what it cannot
resolve; doing it here means the answer is visible while the sweep is being
configured rather than after a build has started.

Standard library only: no Qt, no pydantic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

__all__ = [
    'MaterialCandidate', 'scan_material_directories',
    'infer_potential_extension', 'STRUCTURE_EXTENSION',
]

#: Structure files are CIFs throughout the materials library.
STRUCTURE_EXTENSION = '.cif'


@dataclass(frozen=True)
class MaterialCandidate:
    """One name found in a structure directory, a potential directory, or both."""

    name: str
    has_structure: bool
    has_potential: bool

    @property
    def usable(self) -> bool:
        return self.has_structure and self.has_potential

    @property
    def reason(self) -> str:
        """Why this material cannot be run, or '' when it can."""
        if self.usable:
            return ''
        if not self.has_potential:
            return 'no potential file'
        return 'no structure file'


def _stems(directory: Optional[Union[str, Path]], extension: str) -> set:
    if directory is None:
        return set()
    path = Path(directory)
    if not path.is_dir():
        return set()
    return {p.stem for p in path.iterdir()
            if p.is_file() and p.suffix == extension}


def infer_potential_extension(directory: Optional[Union[str, Path]]) -> Optional[str]:
    """The commonest file extension in a potential directory.

    Potentials are named for their material but suffixed by their kind -- .sw,
    .rebomos, and so on -- so the extension is a property of the directory
    rather than something to ask the user for.
    """
    if directory is None:
        return None
    path = Path(directory)
    if not path.is_dir():
        return None
    suffixes = Counter(p.suffix for p in path.iterdir()
                       if p.is_file() and p.suffix)
    if not suffixes:
        return None
    return suffixes.most_common(1)[0][0]


def scan_material_directories(
        structure_dir: Optional[Union[str, Path]],
        potential_dir: Optional[Union[str, Path]],
        potential_extension: Optional[str] = None,
) -> List[MaterialCandidate]:
    """Every name found in either directory, flagged with what it has.

    Names present in only one directory are returned too, marked unusable. They
    are the ones worth showing: a material silently absent from a sweep is the
    question 'why did this not run?' waiting to be asked.
    """
    extension = potential_extension or infer_potential_extension(potential_dir)
    if extension is None:
        extension = ''

    structures = _stems(structure_dir, STRUCTURE_EXTENSION)
    potentials = _stems(potential_dir, extension) if extension else set()

    return [
        MaterialCandidate(name=name,
                          has_structure=name in structures,
                          has_potential=name in potentials)
        for name in sorted(structures | potentials)
    ]
