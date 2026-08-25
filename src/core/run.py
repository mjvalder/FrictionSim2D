"""Deprecated location. Canonical home: src.run.*.

Five-way split, so this cannot be a single star-import: each submodule is
re-exported explicitly, private names included. Star-imports skip underscore
names, and four private helpers are read from here by tests and by two
modules under src/, so they are listed out below.

Note that patching a name on THIS module is inert -- each submodule reads its
own globals, not these copies. Patch src.run.<submodule>.<name> instead.
"""
from ..run.expand import *                  # noqa: F401,F403
from ..run.hpc_dispatch import *            # noqa: F401,F403
from ..run.naming import *                  # noqa: F401,F403
from ..run.simulate import *                # noqa: F401,F403

from ..run.expand import (                  # noqa: F401
    _BOOL_ADAPTER,
    _as_float_list,
    _as_list,
    _as_model_bool,
    _builder_cls_for_count,
    _is_ascending,
    _is_subsequence,
    _pair_template_fault,
    _resolve_lj_override_files,
    _routing_view,
    _run_scripts_per_build,
    _section_markers,
    _validate_runtime_sweep_ordering,
)
from ..run.hpc_dispatch import _build_hpc_manifest_entries   # noqa: F401
from ..run.naming import (                  # noqa: F401
    _NUMBERED_SHEET_SECTION,
    _flake_section,
    _numbered_flake_keys,
    _numbered_sheet_keys,
)
from ..run.simulate import (                # noqa: F401
    _select_afm_builder_cls,
    _select_sheet_builder_cls,
)
