"""The window is two top-level tabs: the builder, and the database (D1).

A database table does not pair with the 3D viewport -- it needs the full
width -- so the splitter that was the central widget becomes the content of
one tab rather than the window itself.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QProcess  # noqa: E402
from PySide6.QtWidgets import QApplication, QTabWidget  # noqa: E402

from src.gui.workspace import Workspace  # noqa: E402


@pytest.fixture(name='qt_app', scope='module')
def _qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(name='window')
def _window(qt_app, tmp_path):
    del qt_app
    from src.gui.app import MainWindow
    window = MainWindow(Workspace.open(tmp_path / 'ws'))
    yield window
    # closeEvent prompts on a dirty document; a real QMessageBox in teardown
    # would block forever with no one to click it.
    window.document.dirty = False
    window.settings_document.dirty = False
    window.close()


def test_the_window_has_three_top_level_tabs(window):
    assert isinstance(window.top_tabs, QTabWidget)
    labels = [window.top_tabs.tabText(i) for i in range(window.top_tabs.count())]
    assert labels == ['Builder', 'Database', 'Plots']


def test_the_builder_tab_still_holds_the_form_and_the_viewport(window):
    assert window.top_tabs.widget(0) is window.builder_page
    assert window.builder_page.isAncestorOf(window.form)
    assert window.builder_page.isAncestorOf(window.viewport)
    assert window.builder_page.isAncestorOf(window.render_button)


def test_the_database_tab_is_its_own_page(window):
    assert window.top_tabs.widget(1) is window.database_page
    assert not window.database_page.isAncestorOf(window.viewport)


def test_building_the_window_starts_no_subprocess(window):
    """The pane must never query on construction -- every smoke test in this
    directory builds a MainWindow, and a QProcess per construction would make
    the suite spawn dozens of aiida interpreters."""
    assert window.top_tabs.currentIndex() == 0
    assert window.findChildren(QProcess) == []
    assert not window.database.running
