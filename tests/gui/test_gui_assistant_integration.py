"""The Assistant tab inside MainWindow: lazy build, apply, stale and unsaved paths."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')
pytest.importorskip('pydantic_ai')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from src.agent.changes import Op, build_changeset  # noqa: E402
from src.gui.workspace import Workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(name='window')
def _window(qt_app, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    from src.gui.app import MainWindow
    window = MainWindow(Workspace.open(tmp_path / 'ws'))
    yield window
    window.document.dirty = False
    window.settings_document.dirty = False
    window.close()


def _open_assistant(window):
    window.top_tabs.setCurrentWidget(window.assistant_page)
    QApplication.processEvents()
    return window._assistant_pane  # pylint: disable=protected-access


def test_tab_is_built_only_when_shown(window):
    assert window.top_tabs.tabText(window.top_tabs.indexOf(window.assistant_page)) == 'Assistant'
    assert window._assistant_pane is None  # pylint: disable=protected-access
    assert _open_assistant(window) is not None


def test_apply_sets_live_document_and_returns_to_builder(window):
    _open_assistant(window)
    change = build_changeset(window.assistant_deps(), 'loads', [Op('set', 'general', 'force', [5, 10])])
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert window.document.get('general', 'force') == [5, 10]
    assert window.top_tabs.currentIndex() == 0


def test_stale_apply_replaces_card(window, monkeypatch):
    pane = _open_assistant(window)
    change = build_changeset(window.assistant_deps(), 'loads', [Op('set', 'general', 'force', [5])])
    window.document.set('general', 'temp', 123)
    replaced = []
    monkeypatch.setattr(pane, 'replace_change', lambda old, new, note: replaced.append(new))
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert window.document.get('general', 'force') != [5]
    assert len(replaced) == 1 and replaced[0].id != change.id


def test_model_change_with_unsaved_edits_asks_first(window, monkeypatch):
    _open_assistant(window)
    window.document.dirty = True
    asked = []
    monkeypatch.setattr(QMessageBox, 'question',
                        lambda *a, **k: asked.append(1) or QMessageBox.No)
    change = build_changeset(window.assistant_deps(), 'sheet', [Op('set_model', value='sheetonsheet')])
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert asked and window.document.model == 'afm'


def test_model_change_resyncs_selectors(window, monkeypatch):
    _open_assistant(window)
    window.document.dirty = False
    change = build_changeset(window.assistant_deps(), 'sheet', [Op('set_model', value='sheetonsheet')])
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert window.document.model == 'sheetonsheet'
    assert window.family_box.currentData() == 'sheetonsheet'
    assert window.mode_box.currentData() == 'sheetonsheet'


def test_fs_apply_creates_folder(window):
    from src.agent.changes import build_fs_changeset
    _open_assistant(window)
    change = build_fs_changeset(window.assistant_deps(), 'mk', [('mkdir', 'materials')])
    window._apply_assistant_fs_change(change)  # pylint: disable=protected-access
    assert (window.workspace.root / 'materials').is_dir()
