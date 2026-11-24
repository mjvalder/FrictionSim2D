"""ASE must stay out of the modules that everything imports.

``ase.io`` costs roughly 270 ms -- more than the entire rest of a
``frictionsim2d.core.config`` import. ``core/utils`` is split by job so that only
structure-reading code pays it: ``paths`` and ``tokens`` are cheap, and so is
``builders/lammps_data.py``; ``structures`` is not.

That split is easy to undo by accident. Adding a convenience re-export to
``frictionsim2d/core/utils/__init__.py``, or registering ``frictionsim2d.interfaces.ase`` in
``frictionsim2d/interfaces/__init__.py``, silently
hands ASE back to all 66 importers of ``frictionsim2d.core.config``. Nothing else in the
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
    "frictionsim2d.core.config",
    "frictionsim2d.core.utils.paths",
    "frictionsim2d.core.utils.tokens",
    "frictionsim2d.builders.utils.lammps_data",
    "frictionsim2d.interfaces.lammps_monolayer",
    "frictionsim2d.lattice.apply",
    "frictionsim2d.lattice.matching",
    "frictionsim2d.core.__init__",
    "frictionsim2d.interfaces",
    "frictionsim2d.interfaces.ase",   # the adapter itself defers the import into the call
    "frictionsim2d.run.expand",
    "frictionsim2d.run.naming",
    "frictionsim2d.run.placeholders",
    "frictionsim2d.gui.config_validation",
    # Deferred at the call site rather than by module split -- these use ASE's
    # object model directly and import it inside the functions that need it.
    "frictionsim2d.potentials.manager",
    "frictionsim2d.builders.construction.components",
    "frictionsim2d.builders.afm",
    "frictionsim2d.builders.sheetonsheet",
    "frictionsim2d.gui.controllers.elements",
    # ASE file reads now go through frictionsim2d.interfaces.ase; only Atoms construction,
    # make_supercell and writes import ASE directly, deferred into their functions.
    "frictionsim2d.builders.construction.hetero_stack",
    "frictionsim2d.builders.afm.hetero",
    "frictionsim2d.builders.sheetonsheet.hetero",
    "frictionsim2d.builders.sheetonsheet.flake_slide",
    "frictionsim2d.run.simulate",
    "frictionsim2d.cli",
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
        f"frictionsim2d.core.utils.structures or frictionsim2d.interfaces.ase from a package "
        f"__init__, or imported it at module scope on a hot path. Import the "
        f"specific submodule instead, or defer the import into the function."
    )


def test_structures_module_is_also_import_light():
    """Even the structure reader is cheap to *import* -- ASE is deferred.

    frictionsim2d.interfaces.ase defers ``from ase.io import read`` into the body of
    read_structure, so ASE is paid for when a structure is actually read, not
    when a module that might read one is imported.
    """
    assert not _ase_loaded_after_importing("frictionsim2d.core.utils.structures")


def test_reading_a_structure_does_load_ase(materials_path):
    """...and the deferred import really is wired up, not silently dead.

    Without this, `_ase_loaded_after_importing` returning False everywhere
    would be satisfied by an adapter that never calls ASE at all.
    """
    code = (
        "import sys\n"
        "from frictionsim2d.core.utils.structures import cifread\n"
        f"d = cifread({str(materials_path / 'Si.cif')!r})\n"
        "assert d['elements'], 'cifread returned no elements'\n"
        "assert d['lat_a'] > 0, 'cifread returned no lattice constant'\n"
        "print(any(m == 'ase' or m.startswith('ase.') for m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "True", (
        "Reading a CIF did not load ASE -- the adapter in frictionsim2d/interfaces/ase.py "
        "is not actually calling it."
    )


#: Modules of the GUI assistant that must import without Qt (the agent core is
#: headless and tested without a display). Later tasks append to this list.
NO_QT_MODULES = ['frictionsim2d.agent', 'frictionsim2d.agent.events', 'frictionsim2d.agent.providers',
                  'frictionsim2d.agent.runtime', 'frictionsim2d.agent.tools', 'frictionsim2d.agent.changes', 'frictionsim2d.agent.context',
                  'frictionsim2d.agent.tools.setup', 'frictionsim2d.agent.tools.materials',
                  'frictionsim2d.agent.tools.workspace', 'frictionsim2d.agent.ollama_server',
                  'frictionsim2d.core.scaling', 'frictionsim2d.agent.tools.analysis', 'frictionsim2d.gui.datasets',
                  'frictionsim2d.agent.materials_db', 'frictionsim2d.agent.tools.materials_db',
                  'frictionsim2d.agent.plot_changes', 'frictionsim2d.agent.tools.plots']

#: Modules that must not pull pydantic-ai in: GUI startup and the agent core.
#: Only frictionsim2d.agent.runtime (and providers, inside functions) may import it.
NO_PYDANTIC_AI_MODULES = ['frictionsim2d.agent', 'frictionsim2d.agent.events', 'frictionsim2d.agent.providers',
                           'frictionsim2d.agent.tools', 'frictionsim2d.gui.app',
                           'frictionsim2d.agent.changes', 'frictionsim2d.agent.context',
                           'frictionsim2d.agent.tools.setup', 'frictionsim2d.agent.tools.materials',
                           'frictionsim2d.agent.tools.workspace', 'frictionsim2d.agent.ollama_server',
                           'frictionsim2d.core.scaling', 'frictionsim2d.agent.tools.analysis',
                           'frictionsim2d.agent.materials_db', 'frictionsim2d.agent.tools.materials_db',
                  'frictionsim2d.agent.plot_changes', 'frictionsim2d.agent.tools.plots']


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
    if module.startswith('frictionsim2d.gui'):
        pytest.importorskip('PySide6')
    assert not _loaded_after_importing(module, 'pydantic_ai')


@pytest.mark.parametrize("module", ["frictionsim2d.core.config", "frictionsim2d.interfaces"])
@pytest.mark.parametrize("banned", ["jinja2", "lammps"])
def test_config_and_interfaces_package_do_not_import_jinja_or_lammps(module, banned):
    """``interfaces/__init__.py`` stays empty: its adapters load on demand."""
    assert not _loaded_after_importing(module, banned)


def test_lattice_imports_nothing_from_builders_at_module_level():
    """lattice depends on core and interfaces only."""
    code = (
        "import sys\n"
        "import frictionsim2d.lattice.apply, frictionsim2d.lattice.matching\n"
        "print(any(m == 'frictionsim2d.builders' or m.startswith('frictionsim2d.builders.')"
        " for m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_importing_data_provenance_does_not_load_numpy():
    """``data/__init__.py`` stays empty: provenance needs no numpy."""
    assert not _loaded_after_importing("frictionsim2d.data.provenance", "numpy")
