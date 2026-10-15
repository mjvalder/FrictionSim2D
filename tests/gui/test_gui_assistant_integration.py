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
    window.form.rebuild()  # as a real form edit leaves it: editors show the value
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


def test_as_new_replaces_document_and_freezes_original_card(window, monkeypatch):
    pane = _open_assistant(window)
    window.document.dirty = False
    old = window.document
    change = build_changeset(window.assistant_deps(), 'loads', [Op('set', 'general', 'force', [7])])
    marked = []
    real = pane.mark_applied
    monkeypatch.setattr(pane, 'mark_applied', lambda c, m: (marked.append(c.id), real(c, m)))
    window._apply_assistant_change(change, True)  # pylint: disable=protected-access
    assert window.document is not old
    assert window.document.workspace_dir == window.workspace.configs_dir
    assert window.document.get('general', 'force') == 7
    assert marked == [change.id]


def test_as_new_with_unsaved_edits_declined_changes_nothing(window, monkeypatch):
    pane = _open_assistant(window)
    window.document.dirty = True
    old = window.document
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.No)
    change = build_changeset(window.assistant_deps(), 'loads', [Op('set', 'general', 'force', [7])])
    marked = []
    monkeypatch.setattr(pane, 'mark_applied', lambda c, m: marked.append(c.id))
    window._apply_assistant_change(change, True)  # pylint: disable=protected-access
    assert window.document is old and not marked
    assert window.document.get('general', 'force') != 7


def test_settings_change_refreshes_live_settings(window):
    from src.agent.changes import Op as _Op
    _open_assistant(window)
    doc = window.settings_document
    section = next(k for k, v in doc.values.items() if isinstance(v, dict) and v)
    key = next(k for k, v in doc.values[section].items() if isinstance(v, (int, float)) and not isinstance(v, bool))
    new = doc.values[section][key] + 1
    change = build_changeset(window.assistant_deps(), 's', [_Op('settings_set', section, key, new)])
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert getattr(getattr(window.settings, section), key) == new


def test_invalid_settings_apply_anyway_is_refused_cleanly(window):
    pane = _open_assistant(window)
    change = build_changeset(window.assistant_deps(), 's',
                             [Op('settings_set', 'simulation', 'timestep', 'abc')])
    assert not change.ok
    pane._add_card(change)  # pylint: disable=protected-access
    before = window.settings
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert window.settings is before
    assert window.settings_document.get('simulation', 'timestep') != 'abc'
    assert 'settings invalid' in pane.cards[change.id].status.text()


def test_apply_commits_pending_form_edits_first(window, monkeypatch):
    pane = _open_assistant(window)
    change = build_changeset(window.assistant_deps(), 'loads', [Op('set', 'general', 'force', [5])])
    # A value typed into the form but not yet committed when Apply is clicked.
    monkeypatch.setattr(window.form, 'commit',
                        lambda *a, **k: window.document.set('general', 'temp', 123))
    replaced = []
    monkeypatch.setattr(pane, 'replace_change', lambda old, new, note: replaced.append(new))
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert len(replaced) == 1 and window.document.get('general', 'force') != [5]
