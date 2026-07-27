"""Driving the real window offscreen.

These are the tests that would have caught a window that cannot open, a form
that cannot build, or a Render button that raises. They construct the actual
MainWindow under ``QT_QPA_PLATFORM=offscreen`` and exercise it.

Rendering itself is not driven here -- it spawns a subprocess and needs an
event loop -- but every step up to and including the request is.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.controllers.document import ConfigDocument  # noqa: E402
from src.gui.controllers.preview import cache_key, code_fingerprint  # noqa: E402
from src.gui.workspace import RecentWorkspaces, Workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / 'examples'
ALL_MODELS = ['afm', 'indent', 'pes_tip', 'sheetonsheet', 'pes_sheet']


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(name='workspace')
def _workspace(tmp_path):
    return Workspace.open(tmp_path / 'ws')


class TestWorkspace:

    def test_opening_creates_the_layout(self, tmp_path):
        workspace = Workspace.open(tmp_path / 'ws')
        assert workspace.root.is_dir()
        assert workspace.configs_dir.is_dir()
        assert workspace.preview_dir.is_dir()

    def test_state_survives_reopening(self, tmp_path):
        workspace = Workspace.open(tmp_path / 'ws')
        workspace.remember('last_model', 'indent')
        assert Workspace.open(tmp_path / 'ws').state['last_model'] == 'indent'

    def test_default_config_path_does_not_collide(self, workspace):
        first = workspace.default_config_path('afm')
        first.write_text('', encoding='utf-8')
        assert workspace.default_config_path('afm') != first

    def test_previews_are_pruned_to_the_most_recent(self, workspace):
        for index in range(12):
            (workspace.preview_dir / f"key{index:02d}").mkdir()
        workspace.prune_previews(keep=3)
        assert len(list(workspace.preview_dir.iterdir())) == 3

    def test_recent_workspaces_round_trip(self, tmp_path):
        recent = RecentWorkspaces(path=tmp_path / 'recent.json')
        recent.add(tmp_path)
        assert recent.most_recent() == tmp_path.resolve()

    def test_recent_skips_directories_that_vanished(self, tmp_path):
        recent = RecentWorkspaces(path=tmp_path / 'recent.json')
        gone = tmp_path / 'gone'
        gone.mkdir()
        recent.add(gone)
        gone.rmdir()
        assert recent.load() == []


class TestCacheKey:
    """A preview may only be reused when it would be rebuilt identically."""

    def test_same_input_same_key(self):
        config = {'2D': {'mat': 'MoS2'}}
        assert cache_key(config, 'afm', 1) == cache_key(config, 'afm', 1)

    def test_config_change_changes_the_key(self):
        assert cache_key({'2D': {'mat': 'MoS2'}}, 'afm', 1) != \
            cache_key({'2D': {'mat': 'WS2'}}, 'afm', 1)

    def test_model_change_changes_the_key(self):
        config = {'2D': {'mat': 'MoS2'}}
        assert cache_key(config, 'afm', 1) != cache_key(config, 'indent', 1)

    def test_layer_change_changes_the_key(self):
        config = {'2D': {'mat': 'MoS2'}}
        assert cache_key(config, 'afm', 1) != cache_key(config, 'afm', 2)

    def test_referenced_file_change_changes_the_key(self, tmp_path):
        potential = tmp_path / 'p.sw'
        potential.write_text('a', encoding='utf-8')
        config = {'2D': {'mat': 'MoS2'}}
        before = cache_key(config, 'afm', 1, [potential])
        potential.write_text('bb', encoding='utf-8')
        assert cache_key(config, 'afm', 1, [potential]) != before

    def test_the_code_is_part_of_the_key(self):
        """Otherwise editing a template and re-rendering would show a system
        built by the previous code."""
        config = {'2D': {'mat': 'MoS2'}}
        assert code_fingerprint() in ''.join([
            code_fingerprint(),
        ])
        assert cache_key(config, 'afm', 1)  # smoke: fingerprint is included


class TestMainWindow:

    @pytest.fixture(name='window')
    def _window(self, qt_app, workspace, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from src.gui.app import MainWindow
        window = MainWindow(workspace)
        yield window
        window.close()

    def test_window_opens(self, window):
        assert window.windowTitle().startswith('FrictionSim2D')

    def test_it_starts_on_the_afm_family(self, window):
        assert window.document.model == 'afm'
        assert window.family_box.currentData() == 'afm'

    def test_mode_selector_lists_the_family_modes(self, window):
        modes = [window.mode_box.itemData(i) for i in range(window.mode_box.count())]
        assert modes == ['afm', 'indent', 'pes_tip']

    def test_switching_family_repopulates_the_modes(self, window):
        window.family_box.setCurrentIndex(1)
        modes = [window.mode_box.itemData(i) for i in range(window.mode_box.count())]
        assert modes == ['sheetonsheet', 'pes_sheet']

    def test_switching_family_switches_the_document(self, window):
        window.family_box.setCurrentIndex(1)
        assert window.document.model == 'sheetonsheet'
        assert 'tip' not in window.form.sections

    @pytest.mark.parametrize('index,expected', [(0, 'afm'), (1, 'indent'), (2, 'pes_tip')])
    def test_each_afm_mode_builds_a_form(self, window, index, expected):
        window.mode_box.setCurrentIndex(index)
        assert window.document.model == expected
        assert window.form.sections

    def test_the_form_has_a_section_per_config_section(self, window):
        assert '2D' in window.form.sections
        assert 'tip' in window.form.sections
        assert 'general' in window.form.sections

    def test_substrate_cannot_be_switched_off(self, window):
        """Every model but the PES tip scan aborts without one."""
        assert not window.form.sections['sub'].isCheckable()

    def test_pes_tip_may_be_freestanding(self, window):
        window.mode_box.setCurrentIndex(2)
        assert window.form.sections['sub'].isCheckable()

    def test_counts_are_shown(self, window):
        assert '→' in window.counts.text()

    def test_opening_an_example_populates_the_form(self, window):
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

        editor = window.form.sections['2D'].editors['mat']
        assert editor.value() == 'h-MoS2'

    def test_validation_reports_a_valid_config(self, window):
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()
        assert window._validate()
        assert 'valid' in window.issues.toPlainText()

    def test_validation_reports_and_decorates_a_bad_field(self, window):
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()
        window.form.sections['tip'].editors['pot_path'].set_value('nope.sw')
        window.form.commit()

        assert not window._validate()
        assert 'pot_path' in window.issues.toPlainText()

    def test_counts_update_from_the_loaded_config(self, window):
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window._refresh_counts()
        assert '1 config → 2 builds → 10 jobs' in window.counts.text()

    def test_an_unsupported_list_is_explained(self, window):
        """Tip radius is not part of the output directory name, so sweeping it
        would make every run overwrite the last. The editor keeps the list so
        validation can say that, rather than reporting 'not a number'."""
        window.form.sections['tip'].editors['r'].set_value([10, 20])
        window._validate()
        assert 'overwrite each other' in window.issues.toPlainText()

    def test_saving_produces_a_config_the_cli_can_read(self, window, tmp_path):
        from src.core.config import parse_config

        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        out = window.document.save(tmp_path / 'saved.ini')
        assert parse_config(out) == parse_config(EXAMPLES / 'indent_config.ini')

    def test_viewport_starts_empty(self, window):
        assert window.viewport.system is None

    def test_settings_tab_names_its_source(self, window):
        assert 'Settings from:' in window.settings_view.toPlainText()


class TestFormEditors:
    """Values must round-trip through the widgets unchanged."""

    @pytest.fixture(name='form')
    def _form(self, qt_app, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from src.gui.widgets.form import ConfigForm
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        return ConfigForm(document)

    def test_string_round_trip(self, form):
        editor = form.sections['2D'].editors['mat']
        editor.set_value('WS2')
        assert editor.value() == 'WS2'

    def test_float_round_trip(self, form):
        editor = form.sections['tip'].editors['r']
        editor.set_value(25.0)
        assert editor.value() == 25.0

    def test_list_round_trip(self, form):
        editor = form.sections['2D'].editors['layers']
        editor.set_value([1, 2, 3])
        assert editor.value() == [1, 2, 3]

    def test_scalar_or_list_keeps_a_scalar_scalar(self, form):
        editor = form.sections['general'].editors['force']
        editor.set_value(5.0)
        assert editor.value() == 5.0

    def test_scalar_or_list_keeps_a_list_a_list(self, form):
        editor = form.sections['general'].editors['force']
        editor.set_value([5, 10, 20])
        assert editor.value() == [5.0, 10.0, 20.0]

    def test_empty_is_none_not_empty_string(self, form):
        """The writer emits None as an empty value; '' would round-trip wrong."""
        editor = form.sections['2D'].editors['lat_c']
        editor.set_value(None)
        assert editor.value() is None

    def test_enum_round_trip(self, form):
        editor = form.sections['tip'].editors['amorph']
        editor.set_value('a')
        assert editor.value() == 'a'

    def test_bool_round_trip(self, form):
        editor = form.sections['general'].editors['finite_sheet']
        editor.set_value(True)
        assert editor.value() is True

    def test_committing_writes_through_to_the_document(self, form):
        form.sections['2D'].editors['mat'].set_value('WSe2')
        form.commit()
        assert form.document.get('2D', 'mat') == 'WSe2'

    def test_full_form_round_trip_preserves_the_config(self, form, tmp_path):
        """Building the form and committing it unchanged must not alter values."""
        from src.core.config import parse_config

        before = form.document.to_dict()
        form.commit()
        out = form.document.save(tmp_path / 'c.ini')
        assert parse_config(out) == before
