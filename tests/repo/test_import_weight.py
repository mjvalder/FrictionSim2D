"""ASE must stay out of the modules that everything imports.

``ase.io`` costs roughly 270 ms -- more than the entire rest of a
``src.core.config`` import. ``core/utils`` is split by job so that only
structure-reading code pays it: ``paths`` and ``tokens`` are cheap, and so is
``builders/lammps_data.py``; ``structures`` is not.

That split is easy to undo by accident. Adding a convenience re-export to
``src/core/utils/__init__.py``, or registering ``src.interfaces.ase`` in
``src/interfaces/__init__.py`` (which imports its adapters eagerly), silently
hands ASE back to all 66 importers of ``src.core.config``. Nothing else in the
suite would fail. These tests would.

Each check runs in a subprocess: by the time pytest gets here the session has
almost certainly imported ASE already, so an in-process ``sys.modules`` check
would pass trivially and prove nothing.
"""
import subprocess
import sys

import pytest

#: Modules that must import without dragging in ASE. Every one of these is on a
#: hot path -- config is imported by 66 modules, and config_validation runs at
#: GUI startup.
ASE_FREE_MODULES = [
    "src.core.config",
    "src.core.utils.paths",
    "src.core.utils.tokens",
    "src.builders.utils.lammps_data",
    "src.core.__init__",
    "src.interfaces",
    "src.interfaces.ase",   # the adapter itself defers the import into the call
    "src.run.expand",
    "src.run.naming",
    "src.run.placeholders",
    "src.gui.config_validation",
    # Deferred at the call site rather than by module split -- these use ASE's
    # object model directly and import it inside the functions that need it.
    "src.potentials.manager",
    "src.builders.construction.components",
    "src.builders.afm",
    "src.builders.sheetonsheet",
    "src.gui.controllers.elements",
    # ASE file reads now go through src.interfaces.ase; only Atoms construction,
    # make_supercell and writes import ASE directly, deferred into their functions.
    "src.builders.construction.hetero_stack",
    "src.builders.afm.hetero",
    "src.builders.sheetonsheet.hetero",
    "src.builders.sheetonsheet.flake_slide",
    "src.run.simulate",
    "src.cli",
]


def _ase_loaded_after_importing(module: str) -> bool:
    """Whether importing ``module`` in a fresh interpreter pulls in ASE."""
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        "print(any(m == 'ase' or m.startswith('ase.') for m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return out.stdout.strip() == "True"


@pytest.mark.parametrize("module", ASE_FREE_MODULES)
def test_module_does_not_import_ase(module):
    assert not _ase_loaded_after_importing(module), (
        f"{module} now pulls in ASE (~270 ms). Something re-exported "
        f"src.core.utils.structures or src.interfaces.ase from a package "
        f"__init__, or imported it at module scope on a hot path. Import the "
        f"specific submodule instead, or defer the import into the function."
    )


def test_structures_module_is_also_import_light():
    """Even the structure reader is cheap to *import* -- ASE is deferred.

    src.interfaces.ase defers ``from ase.io import read`` into the body of
    read_structure, so ASE is paid for when a structure is actually read, not
    when a module that might read one is imported.
    """
    assert not _ase_loaded_after_importing("src.core.utils.structures")


def test_reading_a_structure_does_load_ase(materials_path):
    """...and the deferred import really is wired up, not silently dead.

    Without this, `_ase_loaded_after_importing` returning False everywhere
    would be satisfied by an adapter that never calls ASE at all.
    """
    code = (
        "import sys\n"
        "from src.core.utils.structures import cifread\n"
        f"d = cifread({str(materials_path / 'Si.cif')!r})\n"
        "assert d['elements'], 'cifread returned no elements'\n"
        "assert d['lat_a'] > 0, 'cifread returned no lattice constant'\n"
        "print(any(m == 'ase' or m.startswith('ase.') for m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "True", (
        "Reading a CIF did not load ASE -- the adapter in src/interfaces/ase.py "
        "is not actually calling it."
    )


#: Modules of the GUI assistant that must import without Qt (the agent core is
#: headless and tested without a display). Later tasks append to this list.
NO_QT_MODULES = ['src.agent', 'src.agent.events', 'src.agent.providers',
                  'src.agent.runtime', 'src.agent.tools', 'src.agent.changes', 'src.agent.context',
                  'src.agent.tools.setup', 'src.agent.tools.materials',
                  'src.agent.tools.workspace', 'src.agent.ollama_server']

#: Modules that must not pull pydantic-ai in: GUI startup and the agent core.
#: Only src.agent.runtime (and providers, inside functions) may import it.
NO_PYDANTIC_AI_MODULES = ['src.agent', 'src.agent.events', 'src.agent.providers',
                           'src.agent.tools', 'src.gui.app',
                           'src.agent.changes', 'src.agent.context',
                           'src.agent.tools.setup', 'src.agent.tools.materials',
                           'src.agent.tools.workspace', 'src.agent.ollama_server']


def _loaded_after_importing(module: str, banned: str) -> bool:
    """Whether importing ``module`` in a fresh interpreter loads ``banned``."""
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        f"print({banned!r} in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True)
    return out.stdout.strip().splitlines()[-1] == 'True'


@pytest.mark.parametrize('module', NO_QT_MODULES)
def test_agent_core_does_not_import_qt(module):
    assert not _loaded_after_importing(module, 'PySide6')


@pytest.mark.parametrize('module', NO_PYDANTIC_AI_MODULES)
def test_module_does_not_import_pydantic_ai(module):
    pytest.importorskip('pydantic_ai')
    if module.startswith('src.gui'):
        pytest.importorskip('PySide6')
    assert not _loaded_after_importing(module, 'pydantic_ai')
