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

    def no_modal(text):
        raise AssertionError(f'unexpected Undo confirmation: {text}')

    def make(root=tmp_path / 'ws'):
        window = MainWindow(Workspace.open(root))
        window._confirm_undo = no_modal  # pylint: disable=protected-access
        window._user_busy = lambda: False  # pylint: disable=protected-access
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


def test_window_hands_the_pane_a_real_ollama_server(window):
    from src.agent.ollama_server import OllamaServer
    window.assistant_button.click()
    pane = window._assistant_pane  # pylint: disable=protected-access
    assert isinstance(pane._server, OllamaServer)  # pylint: disable=protected-access
    assert not pane._server.started_by_us  # Gemini default: nothing was started


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


def _card_mode(pane):
    pane.ask_box.setChecked(True)


def test_card_apply_sets_live_document_and_stays_put(window):
    pane = _open_assistant(window)
    _card_mode(pane)
    window.top_tabs.setCurrentIndex(1)
    change = build_changeset(window.assistant_deps(), 'loads', [Op('set', 'general', 'force', [5, 10])])
    pane.receive_proposal(change)
    pane.cards[change.id].apply_button.click()
    assert window.document.get('general', 'force') == [5, 10]
    assert window.top_tabs.currentIndex() == 1
    assert window.assistant_dock.isVisibleTo(window)  # Apply leaves the dock open
    assert pane.cards == {} and pane.pending_box.isHidden()
    assert window.form.highlights() == {('general', 'force'): 'None → [5, 10]'}
    assert pane.undo_state(window._assistant_undo[-1].id) == 'enabled'  # pylint: disable=protected-access


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


def test_card_model_change_does_not_ask_and_can_be_undone(window, monkeypatch):
    _open_assistant(window)
    window.document.dirty = True
    asked = []
    monkeypatch.setattr(QMessageBox, 'question',
                        lambda *a, **k: asked.append(1) or QMessageBox.No)
    change = build_changeset(window.assistant_deps(), 'sheet', [Op('set_model', value='sheetonsheet')])
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert not asked and window.document.model == 'sheetonsheet'
    window._undo_assistant_change(window._assistant_undo[-1].id)  # pylint: disable=protected-access
    assert window.document.model == 'afm' and window.document.dirty


def test_model_change_resyncs_selectors(window, monkeypatch):
    _open_assistant(window)
    window.document.dirty = False
    change = build_changeset(window.assistant_deps(), 'sheet', [Op('set_model', value='sheetonsheet')])
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert window.document.model == 'sheetonsheet'
    assert window.family_box.currentData() == 'sheetonsheet'
    assert window.mode_box.currentData() == 'sheetonsheet'


def test_fs_card_applies_and_leaves_the_strip(window):
    from src.agent.changes import build_fs_changeset
    pane = _open_assistant(window)
    change = build_fs_changeset(window.assistant_deps(), 'mk', [('mkdir', 'materials')])
    pane.receive_proposal(change)  # file operations ask even in direct mode
    assert not pane.pending_box.isHidden()
    pane.cards[change.id].apply_button.click()
    assert (window.workspace.root / 'materials').is_dir()
    assert pane.cards == {} and pane.pending_box.isHidden()
    assert 'Applied: mk' in pane.chat.toPlainText()


def test_as_new_replaces_document_and_removes_the_card(window, monkeypatch):
    pane = _open_assistant(window)
    _card_mode(pane)
    window.document.dirty = False
    old = window.document
    change = build_changeset(window.assistant_deps(), 'loads', [Op('set', 'general', 'force', [7])])
    pane.receive_proposal(change)
    pane.cards[change.id].apply_new_button.click()
    assert window.document is not old
    assert window.document.workspace_dir == window.workspace.configs_dir
    assert window.document.get('general', 'force') == 7
    assert pane.cards == {}
    window._undo_assistant_change(window._assistant_undo[-1].id)  # pylint: disable=protected-access
    assert window.document.get('general', 'force') == old.get('general', 'force')


def test_as_new_with_unsaved_edits_declined_changes_nothing(window, monkeypatch):
    pane = _open_assistant(window)
    _card_mode(pane)
    window.document.dirty = True
    old = window.document
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.No)
    change = build_changeset(window.assistant_deps(), 'loads', [Op('set', 'general', 'force', [7])])
    pane.receive_proposal(change)
    window._apply_assistant_change(change, True)  # pylint: disable=protected-access
    assert window.document is old and change.id in pane.cards
    assert window.document.get('general', 'force') != 7


def _numeric_setting(doc):
    section = next(k for k, v in doc.values.items() if isinstance(v, dict) and v)
    key = next(k for k, v in doc.values[section].items()
               if isinstance(v, (int, float)) and not isinstance(v, bool))
    return section, key


def test_settings_change_refreshes_live_settings(window):
    pane = _open_assistant(window)
    doc = window.settings_document
    section, key = _numeric_setting(doc)
    new = doc.values[section][key] + 1
    change = build_changeset(window.assistant_deps(), 's', [Op('settings_set', section, key, new)])
    pane.receive_proposal(change)
    assert getattr(getattr(window.settings, section), key) == new
    assert (section, key) in window.settings_form.highlights()


def test_invalid_settings_apply_anyway_is_refused_cleanly(window):
    pane = _open_assistant(window)
    _card_mode(pane)
    change = build_changeset(window.assistant_deps(), 's',
                             [Op('settings_set', 'simulation', 'timestep', 'abc')])
    assert not change.ok
    pane.receive_proposal(change)
    before = window.settings
    window._apply_assistant_change(change, False)  # pylint: disable=protected-access
    assert window.settings is before
    assert window.settings_document.get('simulation', 'timestep') != 'abc'
    assert pane.cards == {} and 'settings invalid' in pane.chat.toPlainText()


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


# ------------------------------------------------------------ direct edits

def _editor(form, section, key):
    return form.sections[section].editors[key]


def test_direct_apply_changes_document_and_highlights_exactly_those_fields(window):
    pane = _open_assistant(window)
    window.top_tabs.setCurrentIndex(2)
    before_temp = window.document.get('general', 'temp')
    change = build_changeset(window.assistant_deps(), 'loads',
                             [Op('set', 'general', 'force', [5, 10]),
                              Op('set', 'general', 'temp', 250)])
    pane.receive_proposal(change)
    assert window.document.get('general', 'force') == [5, 10]
    assert window.document.get('general', 'temp') == 250
    assert window.top_tabs.currentIndex() == 2  # stays where the user is
    marked = [(s, k) for s, f in window.form.sections.items()
              for k, e in f.editors.items() if e.highlight()]
    assert sorted(marked) == [('general', 'force'), ('general', 'temp')]
    tip = _editor(window.form, 'general', 'temp').toolTip()
    assert tip.startswith(f'Set by assistant: {before_temp!r} → 250')
    assert pane.cards == {} and pane.pending_box.isHidden()
    assert 'Set 2 fields: [general] force, [general] temp' in pane.chat.toPlainText()


def test_direct_highlights_survive_rebuild_and_clear_on_user_edit(window):
    pane = _open_assistant(window)
    change = build_changeset(window.assistant_deps(), 'loads',
                             [Op('set', 'general', 'force', [5]), Op('set', 'general', 'temp', 250)])
    pane.receive_proposal(change)
    window.form.rebuild()
    editor = _editor(window.form, 'general', 'temp')
    assert editor.highlight() is not None
    editor.edit.setText('260')
    editor.edit.textEdited.emit('260')
    assert set(window.form.highlights()) == {('general', 'force')}


def test_accept_clears_every_highlight(window):
    pane = _open_assistant(window)
    section, key = _numeric_setting(window.settings_document)
    new = window.settings_document.values[section][key] + 1
    pane.receive_proposal(build_changeset(window.assistant_deps(), 'loads',
                                          [Op('set', 'general', 'force', [5])]))
    pane.receive_proposal(build_changeset(window.assistant_deps(), 's',
                                          [Op('settings_set', section, key, new)]))
    assert pane.accept_button.isEnabled()
    pane.accept_button.click()
    assert window.form.highlights() == {} and window.settings_form.highlights() == {}
    assert not pane.accept_button.isEnabled()


def test_save_clears_highlights_and_ends_undo(window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    pane = _open_assistant(window)
    pane.receive_proposal(build_changeset(window.assistant_deps(), 'loads',
                                          [Op('set', 'general', 'force', [5, 10])]))
    entry = window._assistant_undo[-1].id  # pylint: disable=protected-access
    target = tmp_path / 'saved.ini'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(target), ''))
    window._save_config()  # pylint: disable=protected-access
    assert target.is_file()
    assert window.form.highlights() == {}
    assert pane.undo_state(entry) == 'locked'
    window._undo_assistant_change(entry)  # pylint: disable=protected-access
    assert window.document.get('general', 'force') == [5, 10]


def test_undo_restores_document_and_settings_latest_first(window):
    from PySide6.QtCore import QUrl
    pane = _open_assistant(window)
    force_before = window.document.get('general', 'force')
    section, key = _numeric_setting(window.settings_document)
    old_setting = window.settings_document.values[section][key]
    pane.receive_proposal(build_changeset(window.assistant_deps(), 'loads',
                                          [Op('set', 'general', 'force', [5])]))
    pane.receive_proposal(build_changeset(window.assistant_deps(), 's',
                                          [Op('settings_set', section, key, old_setting + 1)]))
    first, second = (e.id for e in window._assistant_undo)  # pylint: disable=protected-access
    assert pane.undo_state(first) == 'blocked' and pane.undo_state(second) == 'enabled'
    pane.chat.anchorClicked.emit(QUrl(f'undo:{first}'))  # older: refused
    assert window.settings_document.values[section][key] == old_setting + 1
    pane.chat.anchorClicked.emit(QUrl(f'undo:{second}'))
    assert window.settings_document.values[section][key] == old_setting
    assert getattr(getattr(window.settings, section), key) == old_setting
    assert window.settings_form.document is window.settings_document
    assert (section, key) not in window.settings_form.highlights()
    assert pane.undo_state(second) == 'undone' and pane.undo_state(first) == 'enabled'
    pane.chat.anchorClicked.emit(QUrl(f'undo:{first}'))
    assert window.document.get('general', 'force') == force_before
    assert window.form.document is window.document
    assert window.form.highlights() == {}


SW = REPO_ROOT / 'examples' / 'potentials' / 'sw' / 'sw_lammps'
CIF = REPO_ROOT / 'examples' / 'materials'


def _sweep(window):
    return build_changeset(window.assistant_deps(), 'sweep', [
        Op('material_dirs', value=(str(CIF), str(SW), '.sw')),
        Op('materials', value=['h-MoS2', 'h-WS2']),
        Op('set', '2D', 'pot_type', 'sw')])


@pytest.mark.parametrize('existed', [True, False])
def test_undo_restores_the_material_list_file(window, existed):
    pane = _open_assistant(window)
    target = window.workspace.configs_dir / 'materials.txt'
    target.parent.mkdir(parents=True, exist_ok=True)
    if existed:
        target.write_bytes(b'old-list\n')
    pane.receive_proposal(_sweep(window))
    assert 'h-WS2' in target.read_text()
    window._undo_assistant_change(window._assistant_undo[-1].id)  # pylint: disable=protected-access
    if existed:
        assert target.read_bytes() == b'old-list\n'
    else:
        assert not target.exists()


def test_stale_direct_apply_is_rechecked_and_applied(window):
    pane = _open_assistant(window)
    change = build_changeset(window.assistant_deps(), 'loads', [Op('set', 'general', 'force', [5, 10])])
    window.document.set('general', 'temp', 123)
    window.form.rebuild()  # the user edited while the turn ran
    pane.receive_proposal(change)
    assert window.document.get('general', 'force') == [5, 10]
    assert window.document.get('general', 'temp') == 123
    assert 're-checked' in pane.chat.toPlainText()


def test_tool_input_error_is_a_red_line_and_changes_nothing(window):
    from src.agent.changes import ChangeSet
    pane = _open_assistant(window)
    before = dict(window.document.values.get('2D') or {})
    change = ChangeSet(id='bad001', title='bad path', notes=[], base_fingerprint='stale',
                       ops=[Op('set', '2D', 'materials_list', '/etc/passwd')])
    pane.receive_proposal(change)
    assert dict(window.document.values.get('2D') or {}) == before
    assert window._assistant_undo == []  # pylint: disable=protected-access
    assert 'Not applied' in pane.chat.toPlainText() and '#bb0000' in pane.chat.toHtml().lower()


def test_ask_switch_is_persisted(window, tmp_path):
    from src.agent.settings import load_agent_settings
    pane = _open_assistant(window)
    pane.ask_box.setChecked(True)
    assert load_agent_settings().ask_before_apply is True
    assert load_agent_settings(tmp_path / 'cfg' / 'FrictionSim2D' / 'agent.yaml').ask_before_apply


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


# ------------------------------------------------------------- review fixes

def _settings_change(window):
    section, key = _numeric_setting(window.settings_document)
    old = window.settings_document.values[section][key]
    return section, key, old, build_changeset(window.assistant_deps(), 's',
                                              [Op('settings_set', section, key, old + 1)])


@pytest.mark.parametrize('write', ['_apply_settings', '_save_settings'])
def test_writing_settings_to_disk_ends_undo(window, write):
    pane = _open_assistant(window)
    section, key, old, change = _settings_change(window)
    pane.receive_proposal(change)
    entry = window._assistant_undo[-1].id  # pylint: disable=protected-access
    getattr(window, write)()
    assert window.workspace.settings_path.is_file()
    assert pane.undo_state(entry) == 'locked'
    window._undo_assistant_change(entry)  # pylint: disable=protected-access
    assert window.settings_document.values[section][key] == old + 1


def test_undo_of_a_settings_change_leaves_settings_to_be_written(window):
    pane = _open_assistant(window)
    section, key, old, change = _settings_change(window)
    pane.receive_proposal(change)
    window._undo_assistant_change(window._assistant_undo[-1].id)  # pylint: disable=protected-access
    assert window.settings_document.values[section][key] == old
    assert window.settings_document.dirty


def _user_edits_temp(window):
    window.document.set('general', 'temp', 123)
    window.form.rebuild()


def test_undo_over_later_user_edits_asks_and_no_keeps_everything(window):
    pane = _open_assistant(window)
    pane.receive_proposal(build_changeset(window.assistant_deps(), 'loads',
                                          [Op('set', 'general', 'force', [5, 10])]))
    entry = window._assistant_undo[-1].id  # pylint: disable=protected-access
    _user_edits_temp(window)
    asked = []
    window._confirm_undo = lambda text: asked.append(text) or False  # pylint: disable=protected-access
    window._undo_assistant_change(entry)  # pylint: disable=protected-access
    assert asked and '[general] temp' in asked[0]
    assert window.document.get('general', 'force') == [5, 10]
    assert window.document.get('general', 'temp') == 123
    assert pane.undo_state(entry) == 'enabled'


def test_undo_over_later_user_edits_yes_reverts(window):
    pane = _open_assistant(window)
    temp = window.document.get('general', 'temp')
    pane.receive_proposal(build_changeset(window.assistant_deps(), 'loads',
                                          [Op('set', 'general', 'force', [5, 10])]))
    _user_edits_temp(window)
    window._confirm_undo = lambda text: True  # pylint: disable=protected-access
    window._undo_assistant_change(window._assistant_undo[-1].id)  # pylint: disable=protected-access
    assert window.document.get('general', 'force') != [5, 10]
    assert window.document.get('general', 'temp') == temp


def test_direct_change_while_the_user_is_editing_waits_as_a_card(window):
    pane = _open_assistant(window)
    window._user_busy = lambda: True  # pylint: disable=protected-access
    change = build_changeset(window.assistant_deps(), 'loads',
                             [Op('set', 'general', 'force', [5, 10])])
    pane.receive_proposal(change)
    assert window.document.get('general', 'force') != [5, 10]
    assert change.id in pane.cards
    assert 'kept as a card because you were editing' in pane.chat.toPlainText()


def test_user_busy_counts_refusals_and_form_focus(window, monkeypatch):
    from src.gui.app import MainWindow
    busy = MainWindow._default_user_busy.__get__(window)  # the real predicate  # pylint: disable=protected-access
    assert not busy()
    monkeypatch.setattr(window.form, 'refusals', lambda: ['[general] temp: refused'])
    assert busy()
    monkeypatch.setattr(window.form, 'refusals', lambda: [])
    monkeypatch.setattr(QApplication, 'focusWidget',
                        staticmethod(lambda: _editor(window.form, 'general', 'temp').edit))
    assert busy()


def test_partial_direct_apply_offers_undo(window):
    pane = _open_assistant(window)
    before = dict(window.document.values.get('2D') or {})
    change = _sweep(window)
    configs = window.workspace.configs_dir
    configs.mkdir(parents=True, exist_ok=True)
    configs.chmod(0o500)
    try:
        pane.receive_proposal(change)
        assert 'partly changed' in pane.chat.toPlainText()
        assert window._assistant_undo  # pylint: disable=protected-access
        window._undo_assistant_change(window._assistant_undo[-1].id)  # pylint: disable=protected-access
    finally:
        configs.chmod(0o700)
    assert dict(window.document.values.get('2D') or {}) == before


def test_undo_of_a_second_change_restores_the_first_highlight(window):
    pane = _open_assistant(window)
    temp = window.document.get('general', 'temp')
    pane.receive_proposal(build_changeset(window.assistant_deps(), 'a',
                                          [Op('set', 'general', 'temp', 250)]))
    pane.receive_proposal(build_changeset(window.assistant_deps(), 'b',
                                          [Op('set', 'general', 'temp', 260)]))
    editor = _editor(window.form, 'general', 'temp')
    assert editor.highlight() == f'{temp!r} → 260'
    window._undo_assistant_change(window._assistant_undo[-1].id)  # pylint: disable=protected-access
    assert window.form.highlights()[('general', 'temp')] == f'{temp!r} → 250'


# ---------------------------------------------------------------- plots

def _plot_window(window):
    """``window`` with two datasets in its workspace and a recorded render hook."""
    for name in ('campaign', 'other'):
        outputs = window.workspace.root / name / 'outputs'
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / 'output_full_100x100y.json').write_text('{"runs": []}')
    renders = []
    window._render_plot = lambda: renders.append(  # pylint: disable=protected-access
        window.plots_pane.plot_list.currentRow())
    return renders


def _plot(window, datasets=('campaign',), plot_type='summary', fields=None):
    from src.agent.plot_changes import build_plot_change
    return build_plot_change(window.assistant_deps(), 'plot', plot_type,
                             fields or {'y_axis': 'cof'}, list(datasets))


def test_deps_carry_the_plot_document(window):
    _plot_window(window)
    deps = window.assistant_deps()
    assert deps.plot_document['data_dirs'] == [] and deps.plot_document['plots'] == []


def test_direct_plot_apply_adds_dataset_and_plot_and_renders(window):
    renders = _plot_window(window)
    pane = _open_assistant(window)
    pane.receive_proposal(_plot(window))
    document = window.plots_pane.document()
    assert document.data_dirs == [str(window.workspace.root / 'campaign')]
    assert document.labels == ['campaign']
    assert [p.plot_type for p in document.plots] == ['summary']
    assert document.plots[0].datasets == ['campaign']
    assert renders == [0]  # the new row, selected before the render
    assert window.top_tabs.currentIndex() == 0  # the assistant does not switch tabs
    text = pane.chat.toPlainText()
    assert 'Added plot: summary on campaign' in text and 'Show' in text
    assert window._assistant_undo[-1].plots_before is not None  # pylint: disable=protected-access


def test_plot_apply_reuses_a_loaded_dataset_and_selects_the_new_row(window):
    renders = _plot_window(window)
    pane = _open_assistant(window)
    window.top_tabs.setCurrentIndex(2)
    pane.receive_proposal(_plot(window))
    pane.receive_proposal(_plot(window, ('campaign', 'other'), 'timeseries'))
    document = window.plots_pane.document()
    assert document.labels == ['campaign', 'other'] and len(document.plots) == 2
    assert renders == [0, 1] and window.plots_pane.plot_list.currentRow() == 1
    assert 'Show' not in pane.chat.toPlainText()  # the Plots tab is in front


def test_show_link_switches_to_the_plots_tab(window):
    from PySide6.QtCore import QUrl
    _plot_window(window)
    pane = _open_assistant(window)
    pane.receive_proposal(_plot(window))
    pane.chat.anchorClicked.emit(QUrl('show:Plots'))
    assert window.top_tabs.currentWidget() is window.plots_page


def test_plot_undo_restores_the_plots_document(window):
    from PySide6.QtCore import QUrl
    _plot_window(window)
    pane = _open_assistant(window)
    pane.receive_proposal(_plot(window))
    entry = window._assistant_undo[-1].id  # pylint: disable=protected-access
    pane.chat.anchorClicked.emit(QUrl(f'undo:{entry}'))
    document = window.plots_pane.document()
    assert document.data_dirs == [] and document.plots == []
    assert pane.undo_state(entry) == 'undone'


def test_plot_undo_asks_before_reverting_later_plot_edits(window):
    _plot_window(window)
    pane = _open_assistant(window)
    pane.receive_proposal(_plot(window))
    entry = window._assistant_undo[-1].id  # pylint: disable=protected-access
    window.plots_pane.add_plot('rank_friction')  # the user's own edit since
    asked = []
    window._confirm_undo = lambda text: asked.append(text) or False  # pylint: disable=protected-access
    window._undo_assistant_change(entry)  # pylint: disable=protected-access
    assert asked and 'Plots' in asked[0]
    assert len(window.plots_pane.document().plots) == 2
    window._confirm_undo = lambda text: True  # pylint: disable=protected-access
    window._undo_assistant_change(entry)  # pylint: disable=protected-access
    assert window.plots_pane.document().plots == []


def test_plot_card_in_ask_mode_applies(window):
    renders = _plot_window(window)
    pane = _open_assistant(window)
    _card_mode(pane)
    change = _plot(window)
    pane.receive_proposal(change)
    assert change.id in pane.cards and window.plots_pane.document().plots == []
    pane.cards[change.id].apply_button.click()
    assert pane.cards == {} and len(window.plots_pane.document().plots) == 1
    assert renders == [0] and 'Added plot: summary on campaign' in pane.chat.toPlainText()


def test_busy_user_gets_the_plot_as_a_card(window):
    _plot_window(window)
    pane = _open_assistant(window)
    window._user_busy = lambda: True  # pylint: disable=protected-access
    change = _plot(window)
    pane.receive_proposal(change)
    assert change.id in pane.cards and window.plots_pane.document().plots == []


def test_saving_the_config_keeps_the_plot_undo(window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    _plot_window(window)
    pane = _open_assistant(window)
    pane.receive_proposal(build_changeset(window.assistant_deps(), 'loads',
                                          [Op('set', 'general', 'force', [5])]))
    config_entry = window._assistant_undo[-1].id  # pylint: disable=protected-access
    pane.receive_proposal(_plot(window))
    plot_entry = window._assistant_undo[-1].id  # pylint: disable=protected-access
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        lambda *a, **k: (str(tmp_path / 'saved.ini'), ''))
    window._save_config()  # pylint: disable=protected-access
    assert pane.undo_state(config_entry) == 'locked'
    assert pane.undo_state(plot_entry) == 'enabled'
    window._undo_assistant_change(plot_entry)  # pylint: disable=protected-access
    assert window.plots_pane.document().plots == []


def test_an_invalid_plot_being_edited_does_not_break_the_turn_or_the_apply(window):
    from src.gui.controllers.plot_document import COMMON_SECTION
    renders = _plot_window(window)
    pane = _open_assistant(window)
    window.plots_pane.add_plot('summary')
    window.plots_pane._editors[0].values[COMMON_SECTION]['x_limits'] = 'abc'  # pylint: disable=protected-access
    deps = window.assistant_deps()
    assert deps.plot_document is None
    pane.receive_proposal(_plot(window))
    assert renders == [] and 'Not added' in pane.chat.toPlainText()
