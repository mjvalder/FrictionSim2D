"""Interface wrapper for the ASE (Atomic Simulation Environment) library.

ASE is the heaviest import in the package -- roughly 270 ms, more than the
whole rest of a ``src.core.config`` import combined -- and it is needed by
exactly one operation: reading a structure file. Confining it here means the
cost is paid only by code that actually reads structures.

.. warning::
   This module is deliberately **absent** from ``src/interfaces/__init__.py``.
   That ``__init__`` imports its adapters eagerly, and ``src.core.config`` --
   which 66 modules import -- imports ``src.interfaces``. Registering this
   module there would push ASE onto essentially the whole package and undo
   the reason it was split out. ``tests/test_import_weight.py`` pins this.

   Import it as ``from ..interfaces.ase import read_structure``.
"""

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
    from ase.io import read as ase_read  # noqa: PLC0415  (deferred: ~270 ms)

    atoms = ase_read(str(path))
    if isinstance(atoms, list):
        atoms = atoms[0]
    return {
        'cellpar': [float(v) for v in atoms.cell.cellpar()],
        'symbols': atoms.get_chemical_symbols(),
        'formula': atoms.get_chemical_formula(mode='hill'),
    }
