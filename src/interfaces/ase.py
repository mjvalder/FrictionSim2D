"""Structure-file reading, with ASE's import cost deferred.

**Scope: this is not a general ASE facade.** It wraps the ASE calls whose
result is immediately reduced to plain data -- reading a structure or a LAMMPS
data file, looking up an element mass. The builders use ASE's object model
directly (``Atoms``, ``make_supercell``, ``io.write``) and should keep doing
so; funnelling that through here would be a pass-through layer that hides
nothing. ``src.interfaces.jinja`` is the same shape: one ``PackageLoader``,
not a Jinja2 facade.

What it buys is import weight. ``ase.io`` costs ~355 ms and ``ase.data``
~74 ms, and the import is deferred into the function body below, so ASE loads
when a structure is actually read rather than when a module that might read
one is imported. ``src.core.utils.structures`` is the only consumer.

Modules that genuinely need ASE's object model -- constructing ``Atoms``,
``make_supercell``, anything feeding ``ase.io.write`` -- import it directly and
defer it into the functions that use it. After this split that is two files:
``src/builders/components.py`` (the geometry engine) and
``src/builders/hetero.py`` (structure building).

.. warning::
   This module is deliberately **absent** from ``src/interfaces/__init__.py``.
   That ``__init__`` imports its adapters eagerly, and ``src.core.config`` --
   which 66 modules import -- imports ``src.interfaces``. Registering this
   module there would push ASE onto essentially the whole package and undo
   the reason it was split out. ``tests/test_import_weight.py`` pins this.

   Import it as ``from ..interfaces.ase import read_structure``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union


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
    from ase.io import read as ase_read  # noqa: PLC0415  (deferred: ~270 ms)

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
        numbers: (N,) atomic numbers as a numpy array. LAMMPS data files carry
            atom *types*, which ASE maps to elements via the Masses section, so
            this is the mapped result rather than the raw type column.
    """
    cell: Any        # np.ndarray (3, 3)
    positions: Any   # np.ndarray (N, 3)
    numbers: Any     # np.ndarray (N,)


def read_lammps_data(path: Union[str, Path]) -> LammpsData:
    """Read a LAMMPS data file and return its cell and positions.

    Every caller of this in the builders read the file, took ``cell`` and
    ``get_positions()``, and discarded the ``Atoms`` object -- nine sites
    spelling the same operation slightly differently. Returning plain arrays
    keeps ASE's object model from leaking into the builders that do not need
    it, so hetero_slide.py and flake_slide.py touch ASE not at all.

    Args:
        path: Path to a LAMMPS data file.

    Returns:
        A :class:`LammpsData` holding the cell matrix and atom positions.
    """
    import numpy as np  # noqa: PLC0415
    from ase.io import read  # noqa: PLC0415  (deferred: ~355 ms)

    atoms = read(str(path), format="lammps-data")
    return LammpsData(
        cell=np.array(atoms.cell),
        positions=atoms.get_positions(),
        numbers=atoms.get_atomic_numbers(),
    )


def element_mass(symbol: str) -> float:
    """Standard atomic mass for an element symbol, in amu.

    Wraps ``ase.data.atomic_masses[ase.data.atomic_numbers[symbol]]``, which
    three call sites spelled out by hand. Raises the same ``KeyError`` that
    indexing ``atomic_numbers`` would, so existing ``except (KeyError,
    IndexError)`` handlers keep working unchanged.

    Args:
        symbol: Element symbol, e.g. ``'Mo'``.

    Returns:
        The element's standard atomic mass.
    """
    from ase.data import atomic_masses, atomic_numbers  # noqa: PLC0415

    return float(atomic_masses[atomic_numbers[symbol]])
