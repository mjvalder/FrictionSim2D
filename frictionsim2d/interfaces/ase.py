"""Structure-file reading with ASE's import cost deferred into the function bodies.

Never register this module in ``interfaces/__init__.py``:
``tests/repo/test_import_weight.py`` pins that ASE stays out of the hot imports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union


def read_structure(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a structure file and return its cell, formula, and composition.

    Wraps ``ase.io.read``. A file holding several images (ASE returns a list)
    is reduced to its first, which is what every caller here expects from a
    single-structure CIF.

    Args:
        path: Path to a structure file in any format ASE recognises.

    Returns:
        Dictionary with ``cellpar`` (the six lattice parameters), ``symbols``
        (per-atom chemical symbols, in file order) and ``formula`` (Hill
        notation).
    """
    from ase.io import read as ase_read  # pylint: disable=import-outside-toplevel
    # (deferred: ~270 ms)

    atoms = ase_read(str(path))
    if isinstance(atoms, list):
        atoms = atoms[0]
    return {
        'cellpar': [float(v) for v in atoms.cell.cellpar()],
        'symbols': atoms.get_chemical_symbols(),
        'formula': atoms.get_chemical_formula(mode='hill'),
    }


@dataclass(frozen=True)
class LammpsData:
    """The parts of a LAMMPS data file every caller here actually wants.

    Attributes:
        cell: 3x3 cell matrix as a numpy array.
        positions: (N, 3) atom positions as a numpy array.
        numbers: (N,) atomic numbers, mapped from atom types via the Masses section.
    """
    cell: Any        # np.ndarray (3, 3)
    positions: Any   # np.ndarray (N, 3)
    numbers: Any     # np.ndarray (N,)


def read_lammps_data(path: Union[str, Path]) -> LammpsData:
    """Read a LAMMPS data file and return its cell and positions.

    Args:
        path: Path to a LAMMPS data file.

    Returns:
        A :class:`LammpsData` holding the cell matrix and atom positions.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel
    # (grouped with the ase.io import below so this module stays free of
    # third-party imports at module scope -- see module docstring)
    from ase.io import read  # pylint: disable=import-outside-toplevel
    # (deferred: ~355 ms)

    atoms = read(str(path), format="lammps-data")
    return LammpsData(
        cell=np.array(atoms.cell),
        positions=atoms.get_positions(),
        numbers=atoms.get_atomic_numbers(),
    )


def element_mass(symbol: str) -> float:
    """Standard atomic mass for an element symbol, in amu.

    Args:
        symbol: Element symbol, e.g. ``'Mo'``.

    Returns:
        The element's standard atomic mass.

    Raises:
        KeyError: If ``symbol`` is not an element symbol.
    """
    from ase.data import atomic_masses, atomic_numbers  # pylint: disable=import-outside-toplevel
    # (deferred: ~74 ms)

    return float(atomic_masses[atomic_numbers[symbol]])
