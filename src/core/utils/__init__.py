"""Shared helpers, split by job so callers pay only for what they use.

Intentionally **empty** -- no re-exports. That is the whole point of the
package: ``structures`` reaches ASE (~270 ms), while ``paths``, ``tokens`` and
``lammps_data`` are nearly free. An eager re-export here would collapse that
distinction and hand every consumer the ASE cost again, which is exactly the
state this package replaced -- ``src.core.config``, imported by 66 modules,
was paying 270 ms for a CIF reader it never calls.

Import the submodule you need:

    from ..core.utils.paths import get_material_path
    from ..core.utils.tokens import format_dimension_token

``tests/test_import_weight.py`` fails if this file stops being empty in a way
that pulls ASE into the light modules.
"""
