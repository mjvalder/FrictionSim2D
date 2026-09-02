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


def test_importing_gui_does_not_import_qt_or_ovito():
    result = _run(
        "import sys\n"
        "import src.gui\n"
        "leaked = [m for m in ('PySide6', 'ovito') if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(leaked))\n"
    )
    assert result.returncode == 0, result.stderr
    assert "LEAKED:" in result.stdout
    leaked = result.stdout.split("LEAKED:")[1].strip()
    assert leaked == "", f"src.gui pulled in {leaked} at import time"


def test_importing_cli_does_not_import_qt_or_ovito():
    result = _run(
        "import sys\n"
        "import src.cli\n"
        "leaked = [m for m in ('PySide6', 'ovito') if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(leaked))\n"
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.split("LEAKED:")[1].strip()
    assert leaked == "", f"src.cli pulled in {leaked} at import time"


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
