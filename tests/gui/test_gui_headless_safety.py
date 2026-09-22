"""The GUI package must stay importable on machines without Qt or ovito.

``src.gui`` is reachable from ``src.cli``; if importing it pulled in PySide6 or
ovito, every CLI invocation on a compute node would break. These tests pin the
lazy-import contract.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(code: str) -> subprocess.CompletedProcess:
    """Run code in a fresh interpreter so module caches don't hide an import."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


LEAKY = ('PySide6', 'ovito', 'aiida', 'psycopg2', 'httpx')


def test_importing_gui_does_not_import_qt_ovito_aiida_or_psycopg2():
    result = _run(
        "import sys\n"
        "import src.gui\n"
        f"leaked = [m for m in {LEAKY!r} if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(leaked))\n"
    )
    assert result.returncode == 0, result.stderr
    assert "LEAKED:" in result.stdout
    leaked = result.stdout.split("LEAKED:")[1].strip()
    assert leaked == "", f"src.gui pulled in {leaked} at import time"


def test_importing_cli_does_not_import_qt_ovito_aiida_or_psycopg2():
    result = _run(
        "import sys\n"
        "import src.cli\n"
        f"leaked = [m for m in {LEAKY!r} if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(leaked))\n"
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.split("LEAKED:")[1].strip()
    assert leaked == "", f"src.cli pulled in {leaked} at import time"


def test_the_db_worker_module_imports_nothing_heavy():
    """Every backend import in the worker is inside a function: the module is
    imported by the controller's test suite and by `python -m` on machines
    that may have only one of the two backends."""
    result = _run(
        "import sys\n"
        "import src.gui.db_worker\n"
        f"leaked = [m for m in {LEAKY!r} if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(leaked))\n"
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.split("LEAKED:")[1].strip()
    assert leaked == "", f"src.gui.db_worker pulled in {leaked} at import time"


def test_gui_available_flag_is_a_bool():
    import src.gui as gui

    assert isinstance(gui.GUI_AVAILABLE, bool)
    assert gui._GUI_AVAILABLE is gui.GUI_AVAILABLE  # pylint: disable=protected-access


def test_main_raises_a_helpful_error_when_dependencies_are_missing(monkeypatch):
    import src.gui as gui

    monkeypatch.setattr(gui, "_missing_dependencies", lambda: ["PySide6"])
    with pytest.raises(ImportError) as excinfo:
        gui.main()
    assert "PySide6" in str(excinfo.value)
    assert "pip install -e '.[gui]'" in str(excinfo.value)


def test_importing_gui_does_not_import_matplotlib():
    """The Plots tab renders through a subprocess (G4). If importing
    `src.gui` pulled matplotlib in, the GUI process would install a second Qt
    backend beside ovito's -- which is the whole reason there is a worker."""
    result = _run(
        "import sys\n"
        "import src.gui\n"
        "print('LEAKED:' + ('matplotlib' if 'matplotlib' in sys.modules else ''))\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("LEAKED:")[1].strip() == ""


def test_the_plots_widgets_pull_qt_but_never_matplotlib():
    """`src.gui.widgets.plots` is a QWidget package, so PySide6 is expected
    and asserting otherwise would be asserting it cannot exist. What must NOT
    arrive is matplotlib: the pane shows a PNG the worker produced, and every
    `postprocessing` import in it is deferred into a method precisely so a
    refactor cannot reach `plot_data` (which does import matplotlib)."""
    result = _run(
        "import sys\n"
        "import src.gui.widgets.plots\n"
        "assert 'PySide6' in sys.modules, 'a QWidget package without Qt?'\n"
        "print('LEAKED:' + ','.join(\n"
        "    m for m in ('matplotlib', 'ovito') if m in sys.modules))\n"
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.split("LEAKED:")[1].strip()
    assert leaked == "", f"src.gui.widgets.plots pulled in {leaked}"
