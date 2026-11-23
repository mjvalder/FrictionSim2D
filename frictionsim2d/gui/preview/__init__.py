"""Reading a generated simulation back for display.

The preview shows what LAMMPS will read, and it does so by parsing the build's
own outputs rather than recomputing anything:

* :mod:`.init_script` -- the assembly script gives the simulation cell and where
  each component is placed;
* :mod:`.settings_file` -- ``system.in.settings`` gives component membership and
  the type-to-element legend, post type-offset;
* :mod:`.assemble` -- combines the two into an ovito data collection.

The first two import nothing but the standard library, so they are testable
without Qt, ovito or a display.
"""

from .init_script import Assembly, Placement, parse_assembly, read_data_file_box
from .settings_file import Identity, parse_identity

__all__ = [
    'Assembly', 'Placement', 'parse_assembly', 'read_data_file_box',
    'Identity', 'parse_identity',
]
