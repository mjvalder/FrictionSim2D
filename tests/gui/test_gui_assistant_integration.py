"""The Assistant side pane in MainWindow: lazy dock, toggle, state, apply paths."""
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


@pytest.fixture(name='make_window')
def _make_window(qt_app, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    from src.gui.app import MainWindow
    made = []

    def make(root=tmp_path / 'ws'):
        window = MainWindow(Workspace.open(root))
        made.append(window)
        return window
    yield make
    for window in made:
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()


@pytest.fixture(name='window')
def _window(make_window):
    return make_window()


def _open_assistant(window):
    window.set_assistant_open(True)
    QApplication.processEvents()
    return window._assistant_pane  # pylint: disable=protected-access


def test_dock_is_hidden_and_unbuilt_at_startup(window):
    assert window.top_tabs.count() == 3
    assert [window.top_tabs.tabText(i) for i in range(3)] == ['Builder', 'Database', 'Plots']
    assert not window.assistant_dock.isVisibleTo(window)
    assert not window.assistant_button.isChecked()
    assert window._assistant_pane is None  # pylint: disable=protected-access


def test_dock_sits_beside_the_tabs_not_inside_them(window):
    assert window.main_splitter.indexOf(window.top_tabs) == 0
    assert window.main_splitter.indexOf(window.assistant_dock) == 1
    assert not window.top_tabs.isAncestorOf(window.assistant_dock)


def test_button_builds_and_shows_the_pane(window):
    window.assistant_button.click()
    pane = window._assistant_pane  # pylint: disable=protected-access
    assert pane is not None and window.assistant_dock.isAncestorOf(pane)
    assert window.assistant_dock.isVisibleTo(window)
    window.assistant_button.click()
    assert not window.assistant_dock.isVisibleTo(window)
    assert window._assistant_pane is pane  # pylint: disable=protected-access


def test_shortcut_toggles_the_dock(window):
    assert window.assistant_shortcut.key().toString() == 'Ctrl+Shift+A'
    window.assistant_shortcut.activated.emit()
    assert window.assistant_dock.isVisibleTo(window)
    assert window.assistant_button.isChecked()
    window.assistant_shortcut.activated.emit()
    assert not window.assistant_dock.isVisibleTo(window)
    assert not window.assistant_button.isChecked()


def test_dock_stays_open_across_tab_switches(window):
    pane = _open_assistant(window)
    for index in (1, 2, 0):
        window.top_tabs.setCurrentIndex(index)
        QApplication.processEvents()
        assert window.assistant_dock.isVisibleTo(window)
        assert window._assistant_pane is pane  # pylint: disable=protected-access


def test_context_label_follows_the_tab(window):
    window.top_tabs.setCurrentIndex(1)
    pane = _open_assistant(window)
    assert 'Database' in pane.context_label.text()
    window.top_tabs.setCurrentIndex(2)
    assert 'Plots' in pane.context_label.text()
    window.top_tabs.setCurrentIndex(0)
    assert 'Builder' in pane.context_label.text()


def test_state_is_remembered_and_restored(make_window, tmp_path):
    first = make_window()
    first.show()
    _open_assistant(first)
    total = sum(first.main_splitter.sizes())
    # 300, not wider: the Builder tab's minimum width caps the dock offscreen
    first.main_splitter.setSizes([total - 300, 300])  # as a drag of the handle leaves it
    QApplication.processEvents()
    assert first.main_splitter.sizes()[1] == 300
    first.close()
    state = Workspace.open(tmp_path / 'ws').state['assistant']
    assert state['open'] is True and state['width'] == 300
    second = make_window()
    assert second.assistant_dock.isVisibleTo(second)
    assert second.assistant_button.isChecked()
    assert second._assistant_pane is not None  # pylint: disable=protected-access
    second.show()
    QApplication.processEvents()
    assert second.main_splitter.sizes()[1] == 300


def test_collapse_is_remembered(make_window, tmp_path):
    first = make_window()
    _open_assistant(first)
    first.set_assistant_open(False)
    assert Workspace.open(tmp_path / 'ws').state['assistant']['open'] is False
    second = make_window()
    assert not second.assistant_dock.isVisibleTo(second)
    assert second._assistant_pane is None  # pylint: disable=protected-access


def test_workspace_change_restores_its_state_and_clears_the_chat(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))  # RecentWorkspaces writes there
    pane = _open_assistant(window)
    cleared = []
    monkeypatch.setattr(pane, 'new_chat', lambda: cleared.append(1))
    other = Workspace.open(tmp_path / 'other')
    other.remember('assistant', {'open': False, 'width': 300})
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                        lambda *a, **k: str(other.root))
    window._choose_workspace()  # pylint: disable=protected-access
    assert cleared == [1]
    assert not window.assistant_dock.isVisibleTo(window)
    assert not window.assistant_button.isChecked()


def test_apply_sets_live_document_and_returns_to_builder(window):
    _open_assistant(window)
    change = build_changeset(window.assistant_deps(), 'loads', [Op('set', 'general', 'force', [5, 10])])
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert window.document.get('general', 'force') == [5, 10]
    assert window.top_tabs.currentIndex() == 0
    assert window.assistant_dock.isVisibleTo(window)  # Apply leaves the dock open


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


def test_toggle_button_is_visible_on_every_tab(window):
    window.show()
    QApplication.processEvents()
    assert window.top_tabs.cornerWidget() is window.assistant_button
    for index in range(window.top_tabs.count()):
        window.top_tabs.setCurrentIndex(index)
        QApplication.processEvents()
        assert window.assistant_button.isVisibleTo(window), window.top_tabs.tabText(index)


def test_dock_width_saves_are_debounced(window, monkeypatch):
    from PySide6.QtCore import QEventLoop, QTimer
    window.show()
    _open_assistant(window)
    writes = []
    real = window.workspace.remember
    monkeypatch.setattr(window.workspace, 'remember',
                        lambda key, value: (writes.append(key), real(key, value)))
    for pos in (900, 905, 910, 915, 920):
        window.main_splitter.splitterMoved.emit(pos, 1)
    assert writes == []
    loop = QEventLoop()
    QTimer.singleShot(800, loop.quit)
    loop.exec()
    assert writes == ['assistant']


def _switch_workspace(window, root, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))  # RecentWorkspaces writes there
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *a, **k: str(root))
    window._choose_workspace()  # pylint: disable=protected-access
    QApplication.processEvents()


def _drag_dock_to(window, width):
    total = sum(window.main_splitter.sizes())
    window.main_splitter.setSizes([total - width, width])
    window.main_splitter.splitterMoved.emit(total - width, 1)


def test_switch_uses_the_new_workspaces_width(window, monkeypatch, tmp_path):
    window.show()
    _open_assistant(window)
    _drag_dock_to(window, 300)
    other = Workspace.open(tmp_path / 'b')
    # 290: inside the dock's offscreen limits (pane minimum ~260, Builder cap ~337)
    other.remember('assistant', {'open': False, 'width': 290})
    _switch_workspace(window, other.root, monkeypatch, tmp_path)
    # the pending drag was flushed into the old workspace, not the new one
    assert Workspace.open(tmp_path / 'ws').state['assistant']['width'] == 300
    assert Workspace.open(tmp_path / 'b').state['assistant']['width'] == 290
    window.set_assistant_open(True)
    QApplication.processEvents()
    assert window.main_splitter.sizes()[1] == 290


def test_switch_to_a_fresh_workspace_resets_the_width(window, monkeypatch, tmp_path):
    from src.gui.app import ASSISTANT_DEFAULT_WIDTH
    window.show()
    _open_assistant(window)
    _drag_dock_to(window, 300)
    window.set_assistant_open(False)
    _switch_workspace(window, tmp_path / 'fresh', monkeypatch, tmp_path)
    assert window._assistant_width == ASSISTANT_DEFAULT_WIDTH  # pylint: disable=protected-access


def test_a_clipped_width_does_not_replace_the_remembered_one(make_window, tmp_path):
    Workspace.open(tmp_path / 'ws').remember('assistant', {'open': True, 'width': 5000})
    window = make_window()
    window.show()
    QApplication.processEvents()
    assert window.main_splitter.sizes()[1] < 5000  # the splitter had to clip it
    window.set_assistant_open(False)
    assert Workspace.open(tmp_path / 'ws').state['assistant']['width'] == 5000


def test_reshowing_the_window_keeps_a_recent_drag(window):
    window.show()
    _open_assistant(window)
    QApplication.processEvents()
    _drag_dock_to(window, 300)
    window.hide()
    window.show()
    QApplication.processEvents()
    assert window.main_splitter.sizes()[1] == 300


def test_a_failing_build_shows_an_error_in_the_dock(qt_app, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    from src.gui.app import MainWindow
    Workspace.open(tmp_path / 'ws').remember('assistant', {'open': True, 'width': 300})

    def broken():
        raise ValueError('agent.yaml is malformed')
    window = MainWindow(Workspace.open(tmp_path / 'ws'), agent_settings_loader=broken)
    try:
        pane = window._assistant_pane  # pylint: disable=protected-access
        assert 'agent.yaml is malformed' in pane.text()
        assert window.assistant_dock.isAncestorOf(pane)
        assert window.assistant_dock.isVisibleTo(window)
        assert window.assistant_button.isChecked()
        window.toggle_assistant()
        assert not window.assistant_button.isChecked()
    finally:
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()
