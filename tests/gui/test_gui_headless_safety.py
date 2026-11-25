"""The GUI package must stay importable on machines without Qt or ovito.

``frictionsim2d.gui`` is reachable from ``frictionsim2d.cli``; if importing it pulled in PySide6 or
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


@pytest.mark.parametrize(
    'module,forbidden',
    [
        ('frictionsim2d.gui', LEAKY),
        ('frictionsim2d.cli', LEAKY),
        # Every backend import in the worker is inside a function: the module is
        # imported by the controller's tests and by `python -m` on machines that
        # may have only one of the two backends.
        ('frictionsim2d.gui.db_worker', LEAKY),
        # The Plots tab renders through a subprocess: a matplotlib import here
        # would install a second Qt backend beside ovito's.
        ('frictionsim2d.gui', ('matplotlib',)),
    ],
)
def test_import_leaks_nothing(module, forbidden):
    result = _run(
        "import sys\n"
        f"import {module}\n"
        f"leaked = [m for m in {forbidden!r} if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(leaked))\n"
    )
    assert result.returncode == 0, result.stderr
    assert "LEAKED:" in result.stdout
    leaked = result.stdout.split("LEAKED:")[1].strip()
    assert leaked == "", f"{module} pulled in {leaked} at import time"


def test_main_raises_a_helpful_error_when_dependencies_are_missing(monkeypatch):
    import frictionsim2d.gui as gui

    monkeypatch.setattr(gui, "_missing_dependencies", lambda: ["PySide6"])
    with pytest.raises(ImportError) as excinfo:
        gui.main()
    assert "PySide6" in str(excinfo.value)
    assert "pip install -e '.[gui]'" in str(excinfo.value)


def test_the_plots_widgets_pull_qt_but_never_matplotlib():
    """`frictionsim2d.gui.widgets.plots` is a QWidget package, so PySide6 is expected
    and asserting otherwise would be asserting it cannot exist. What must NOT
    arrive is matplotlib: the pane shows a PNG the worker produced, and every
    `postprocessing` import in it is deferred into a method precisely so a
    refactor cannot reach `plot_data` (which does import matplotlib)."""
    result = _run(
        "import sys\n"
        "import frictionsim2d.gui.widgets.plots\n"
        "assert 'PySide6' in sys.modules, 'a QWidget package without Qt?'\n"
        "print('LEAKED:' + ','.join(\n"
        "    m for m in ('matplotlib', 'ovito') if m in sys.modules))\n"
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.split("LEAKED:")[1].strip()
    assert leaked == "", f"frictionsim2d.gui.widgets.plots pulled in {leaked}"
