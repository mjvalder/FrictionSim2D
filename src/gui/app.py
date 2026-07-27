"""Qt application bootstrap for the FrictionSim2D GUI.

Imports Qt, so it must only ever be reached through :func:`src.gui.main`.

At this stage the window hosts the environment report only; the config tabs and
viewport are added in later stages.
"""

from __future__ import annotations

import sys
from typing import Sequence

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from .diagnostics import run_checks

APP_NAME = "FrictionSim2D"


class MainWindow(QMainWindow):
    """Top-level window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — Simulation Builder")
        self.resize(1280, 800)

        central = QWidget(self)
        layout = QVBoxLayout(central)

        report = QPlainTextEdit(run_checks().render(), central)
        report.setReadOnly(True)
        layout.addWidget(report)

        self.setCentralWidget(central)


def main(argv: Sequence[str] = ()) -> int:
    """Run the Qt event loop. Returns the process exit code."""
    app = QApplication.instance() or QApplication(list(argv) or sys.argv[:1])
    window = MainWindow()
    window.show()
    return app.exec()
