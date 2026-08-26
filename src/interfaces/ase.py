"""Structure-file reading, with ASE's import cost deferred.

**Scope: this is not a general ASE facade.** It wraps one operation --
reading a structure file into plain Python data -- for callers that want a
cell and a composition and nothing more. The builders use ASE's object model
directly (``Atoms``, ``make_supercell``, ``io.write``) and should keep doing
so; funnelling that through here would be a pass-through layer that hides
nothing. ``src.interfaces.jinja`` is the same shape: one ``PackageLoader``,
not a Jinja2 facade.

What it buys is import weight. ``ase.io`` costs ~355 ms and ``ase.data``
~74 ms, and the import is deferred into the function body below, so ASE loads
when a structure is actually read rather than when a module that might read
one is imported. ``src.core.utils.structures`` is the only consumer.

Modules that genuinely need ASE's object model import it directly and defer it
into the functions that use it -- see ``src/builders/components.py``.

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
