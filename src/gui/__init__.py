"""Desktop GUI for FrictionSim2D (Part 1: simulation builder + visualizer).

This root module is deliberately **free of Qt and ovito imports**.  Availability
is probed with :func:`importlib.util.find_spec`, which resolves a module without
executing it, so ``import src.gui`` stays safe on headless compute nodes where
neither PySide6 nor ovito is installed.

The heavy imports live in :mod:`src.gui.app`, reached through :func:`main`.
"""

from importlib import import_module
from importlib.util import find_spec
from typing import List, Tuple

#: Modules that must be importable for the GUI to start.
_REQUIRED = ('PySide6', 'ovito')


def _missing_dependencies() -> List[str]:
    """Names of required GUI modules that are not installed.

    Uses ``find_spec`` rather than ``import`` so that probing never pulls Qt or
    ovito into the process.
    """
    missing = []
    for name in _REQUIRED:
        try:
            if find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


GUI_AVAILABLE: bool = not _missing_dependencies()

#: Backwards-compatible alias matching the ``_XXXX_AVAILABLE`` convention used
#: elsewhere in the package.
_GUI_AVAILABLE = GUI_AVAILABLE

INSTALL_HINT = (
    "GUI dependencies are not installed. Install them with:\n"
    "    pip install -e '.[gui]'\n"
    "Note: ovito supplies its own PySide6 build. Do not also install PySide6 "
    "from conda -- the two Qt builds conflict."
)


def main(argv: Tuple[str, ...] = ()) -> int:
    """Launch the GUI. Imports Qt and ovito only when called."""
    missing = _missing_dependencies()
    if missing:
        raise ImportError(f"Missing GUI dependencies: {', '.join(missing)}.\n{INSTALL_HINT}")
    app_module = import_module(f'{__name__}.app')
    return app_module.main(argv)


__all__ = ['GUI_AVAILABLE', '_GUI_AVAILABLE', 'INSTALL_HINT', 'main']
