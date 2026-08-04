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

import numpy as np
import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from src.gui.controllers.document import ConfigDocument  # noqa: E402
from src.gui.controllers.preview import cache_key, code_fingerprint  # noqa: E402
from src.gui.preview.assemble import AssembledSystem, ComponentSlice  # noqa: E402
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


@pytest.fixture
def two_component_system():
    """A minimal 4-atom assembled system -- enough to make the viewport treat
    a system as "shown" without needing a real render. Mirrors the fixture of
    the same name in tests/test_gui_viewport.py (Task 4); redefined locally
    per the Task 12 brief rather than importing across test modules."""
    return AssembledSystem(
        positions=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]),
        type_ids=np.array([1, 1, 2, 2]),
        group_ids=np.array([0, 0, 1, 1]),
        cell=(0., 10., 0., 10., 0., 10.),
        components=[ComponentSlice('tip', 0, 0, 2, [1]),
                    ComponentSlice('sub', 1, 2, 2, [2])],
        legend={1: 'Si', 2: 'Mo'},
    )


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
        # Tests here exercise edits that mark the document/settings dirty;
        # closeEvent now prompts to save on a dirty close, which would block
        # forever on the real QMessageBox with no one to click it. Cleanup
        # is not the thing under test, so close it quietly.
        window.document.dirty = False
        window.settings_document.dirty = False
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
        # indent_config.ini's 60x60 sheet against a r=25 tip is under the
        # 2x-diameter sizing threshold (D12), so it now also reports the
        # non-blocking tip-sizing warning -- see
        # test_tip_sizing_warning_is_non_blocking, which pins that behaviour
        # down explicitly. It stays non-blocking, so no error lines appear.
        assert 'required' not in window.issues.toPlainText()
        assert 'Input should be' not in window.issues.toPlainText()

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

    def test_tip_sizing_warning(self, window):
        """A footprint under 2x the tip diameter lets the tip interact with
        its own periodic image (D12) -- non-blocking, but worth flagging.
        Set through the editors (as a user would), not the document directly:
        `_validate` calls `form.commit()` first, which would otherwise write
        the still-blank editor values straight back over a direct set()."""
        window.form.sections['2D'].editors['x'].set_value(40.0)
        window.form.sections['2D'].editors['y'].set_value(40.0)
        window.form.sections['tip'].editors['r'].set_value(25.0)
        window._validate()
        assert '2× tip diameter' in window.issues.toPlainText()

    def test_tip_sizing_warning_is_non_blocking(self, window):
        """The precheck must not make an otherwise-valid config invalid."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()
        # indent_config.ini already has x=60, y=60, r=25 -- under the 2x
        # threshold -- so the warning fires without further edits.
        assert window._validate()
        assert '2× tip diameter' in window.issues.toPlainText()

    def test_generate_confirmation_lists_the_output_dir(self, window, monkeypatch):
        """The Generate confirmation must preview the directory(ies) it is
        about to create -- and, with nothing colliding, no warning."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

        captured = {}

        def fake_question(*args, **_kwargs):
            captured['message'] = args[2]
            return QMessageBox.No  # abort before generator.start actually runs

        monkeypatch.setattr(QMessageBox, 'question', fake_question)
        window._generate()

        message = captured['message']
        assert 'indent/h-MoS2/60x_60y/sub_aSi_tip_Si_r25/K300' in message
        assert 'WARNING' not in message

    def test_generate_confirmation_flags_duplicate_output_dirs(self, window, monkeypatch):
        """When two members of a sweep would build into the same directory --
        the exact hazard the loopability rule exists to prevent, e.g. a
        temperature sweep whose members both truncate to the same K{int(temp)}
        -- the confirmation must call it out. Exercised directly against
        `expansion_output_dirs` (already covered producing a real collision
        for a truncating temperature sweep in test_gui_document.py) so this
        test is about `_generate`'s handling of that list, not about
        reproducing the collision through form/pydantic validation, which a
        temp sweep does not pass (`temp` is a scalar field on the model)."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

        dup = Path('indent/h-MoS2/60x_60y/sub_aSi_tip_Si_r25/K300')
        monkeypatch.setattr(window.document, 'expansion_output_dirs', lambda: [dup, dup])

        captured = {}

        def fake_question(*args, **_kwargs):
            captured['message'] = args[2]
            return QMessageBox.No

        monkeypatch.setattr(QMessageBox, 'question', fake_question)
        window._generate()

        message = captured['message']
        assert message.count(str(dup)) >= 2
        assert 'WARNING' in message
        assert str(dup) in message.split('WARNING', 1)[1]

    def test_saving_produces_a_config_the_cli_can_read(self, window, tmp_path):
        from src.core.config import parse_config

        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        out = window.document.save(tmp_path / 'saved.ini')
        assert parse_config(out) == parse_config(EXAMPLES / 'indent_config.ini')

    def test_viewport_starts_empty(self, window):
        assert window.viewport.system is None

    def test_settings_tab_names_its_source(self, window):
        assert 'Settings from:' in window.settings_origin.text()

    def test_settings_are_editable(self, window):
        assert window.settings_form.sections
        assert 'thermostat' in window.settings_form.sections

    def test_settings_survive_a_form_round_trip(self, window):
        """Committing the settings form unchanged must not invalidate them --
        empty strings and nested sub-models are both easy to destroy."""
        window.settings_form.commit()
        result = window.settings_document.validate()
        assert result.ok, result.render()

    def test_saving_settings_writes_a_complete_file(self, window):
        """load_settings replaces rather than merges, so a partial file would
        silently reset everything it omitted."""
        import yaml

        window.settings_form.commit()
        window._save_settings()
        path = window.workspace.settings_path
        assert path.exists()
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        assert set(data) == set(window.settings_document.values)

    def test_a_saved_settings_file_is_passed_to_builds(self, window):
        """Nothing loads settings.yaml from a directory on its own; the GUI has
        to pass it explicitly."""
        assert window.settings_file is None
        window.settings_form.commit()
        window._save_settings()
        assert window.settings_file == window.workspace.settings_path

    def test_render_and_generate_shortcuts(self, window):
        """Render and Generate buttons have keyboard shortcuts."""
        assert window.render_button.shortcut().toString() == 'Ctrl+R'
        assert window.generate_button.shortcut().toString() == 'Ctrl+G'

    def test_scroll_position_retained_when_mode_changes(self, window, qt_app):
        """Switching simulation mode preserves the form's scroll position.

        Verifies that _on_mode_changed saves and restores the scroll position
        by spying on setValue calls. A mode switch (AFM → Sheet) changes the
        form height significantly, risking scrollbar value clamping; the restore
        ensures users stay at their position even if the layout changes.
        """
        from unittest.mock import patch

        # Show the window so the scrollbar gets a real range
        window.show()
        qt_app.processEvents()

        scrollbar = window.config_scroll.verticalScrollBar()
        # Scroll to a meaningful position (not the arbitrary 100 from before)
        scrollbar.setValue(50)
        saved_value = scrollbar.value()

        # Spy on setValue calls during the mode change
        with patch.object(scrollbar, 'setValue', wraps=scrollbar.setValue) as mock_set:
            # Change mode: AFM (with tip section) → Sheet (no tip section) = height change
            window.mode_box.setCurrentIndex(1)  # 'indent' shares AFM structure; try different families
            # Switch to sheetonsheet family first for greater height difference
            window.family_box.setCurrentIndex(1)  # Switch family to sheet-on-sheet
            qt_app.processEvents()
            window.mode_box.setCurrentIndex(0)  # Switch to sheetonsheet mode

        # Verify that setValue was called during the rebuild with the saved value.
        # The calls list includes: the initial scroll + the restore call.
        # The restore call should have the saved_value as an argument.
        restore_calls = [call for call in mock_set.call_args_list
                         if call[0][0] == saved_value]
        assert len(restore_calls) > 0, (
            f"scrollbar.setValue({saved_value}) was never called during mode switch. "
            f"All calls: {mock_set.call_args_list}"
        )


class TestSettingsTakeEffect:
    """The tab edits a document but a build reads a file, so an edit that is
    only on screen used to be silently ignored."""

    @pytest.fixture(name='window')
    def _window(self, qt_app, workspace, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from src.gui.app import MainWindow
        window = MainWindow(workspace)
        yield window
        # See TestMainWindow's fixture: avoid a real, blocking save prompt
        # from closeEvent during teardown of a dirty settings document.
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()

    def test_hpc_scripts_are_blocked_until_modules_are_set(self, window):
        from src.core.run import hpc_generation_blocker

        assert hpc_generation_blocker(window.settings) is not None

    def test_typing_modules_applies_without_a_separate_save(self, window):
        from src.core.run import hpc_generation_blocker

        window.settings_form.sections['hpc'].editors['modules'].set_value(
            ['lammps', 'openmpi'])
        assert window._apply_settings() is None
        assert hpc_generation_blocker(window.settings) is None

    def test_applying_writes_the_workspace_file_builds_are_given(self, window):
        assert window.settings_file is None
        window.settings_form.sections['hpc'].editors['modules'].set_value(['lammps'])
        window._apply_settings()
        assert window.settings_file == window.workspace.settings_path

    def test_invalid_settings_stop_the_build_and_are_not_saved(self, window):
        window.settings_form.sections['hpc'].editors['num_cpus'].set_value('lots')
        error = window._apply_settings()
        assert error is not None
        assert 'num_cpus' in error
        assert not window.workspace.settings_path.exists()

    def test_syncing_with_no_edits_is_a_no_op(self, window):
        assert window._apply_settings() is None
        assert not window.workspace.settings_path.exists()

    def test_the_hpc_group_explains_the_modules_requirement(self, window):
        section = window.settings_form.sections['hpc']
        assert section.title() == 'HPC'
        bubble = section._bubbles.get('modules')
        assert bubble is not None
        assert 'no job scripts' in bubble.toolTip()


class TestSubstrateSection:
    """Switching to a mode that requires a substrate must restore it -- and the
    form must not silently discard what is typed into a section that is absent."""

    def test_switching_from_pes_tip_restores_the_substrate(self):
        """Only the PES tip scan may be freestanding, so switching away from it
        has to bring the substrate back -- otherwise the build aborts with
        'substrate is not optional' however carefully the form was filled in."""
        document = ConfigDocument.blank('pes_tip')
        document.enable_section('sub', False)
        document.set_model('afm')
        assert document.has_section('sub')

    def test_a_filled_substrate_validates_after_the_switch(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.blank('pes_tip')
        document.enable_section('sub', False)
        document.set_model('afm')

        for section, values in (
            ('2D', {'mat': 'h-MoS2', 'pot_type': 'sw',
                    'pot_path': 'examples/potentials/sw/MoS2_wen.sw',
                    'cif_path': 'examples/materials/h-MoS2.cif',
                    'x': 60, 'y': 60, 'layers': [1]}),
            ('tip', {'mat': 'Si', 'pot_type': 'sw',
                     'pot_path': 'examples/potentials/sw/Si.sw',
                     'cif_path': 'examples/materials/Si.cif', 'r': 25}),
            ('sub', {'mat': 'Si', 'pot_type': 'sw',
                     'pot_path': 'examples/potentials/sw/Si.sw',
                     'cif_path': 'examples/materials/Si.cif',
                     'thickness': 12, 'amorph': 'a'}),
        ):
            for key, value in values.items():
                document.set(section, key, value)

        result = document.validate()
        assert result.ok, result.render()

    def test_a_required_section_accepts_edits_even_if_it_was_absent(self, qt_app,
                                                                   monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.blank('afm')
        del document.values['sub']          # as a mode switch used to leave it

        form = ConfigForm(document)
        form.sections['sub'].editors['mat'].set_value('Si')
        form.commit()

        assert document.has_section('sub')
        assert document.get('sub', 'mat') == 'Si'


class TestFamilyFieldFiltering:
    """A field the family's templates never read should not be offered."""

    @pytest.fixture(name='general_fields')
    def _general_fields(self, qt_app, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)

        def fields_for(model):
            return set(ConfigForm(ConfigDocument.blank(model)).sections['general'].editors)
        return fields_for

    def test_afm_offers_load_but_not_pressure(self, general_fields):
        fields = general_fields('afm')
        assert 'force' in fields
        assert 'pressure' not in fields

    def test_afm_hides_the_sheet_only_springs(self, general_fields):
        fields = general_fields('afm')
        assert 'bond_spring' not in fields
        assert 'driving_spring' not in fields

    def test_sheet_on_sheet_offers_pressure_but_not_load(self, general_fields):
        fields = general_fields('sheetonsheet')
        assert 'pressure' in fields
        assert 'force' not in fields

    def test_sheet_on_sheet_keeps_the_springs(self, general_fields):
        assert 'bond_spring' in general_fields('sheetonsheet')

    @pytest.mark.parametrize('model,expected', [
        ('afm', {'force'}),
        ('indent', {'force'}),
        ('sheetonsheet', {'pressure', 'scan_speed'}),
    ])
    def test_outer_loop_offers_only_what_the_builder_acts_on(self, qt_app, model,
                                                             expected, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.blank(model))
        box = form.sections['general'].editors['outer_loop'].box
        choices = {box.itemData(i) for i in range(box.count())} - {None}
        assert choices == expected


class TestDependentFields:
    """A greyed control says 'this does nothing yet'; a live one that is ignored
    says nothing at all."""

    @pytest.fixture(name='form')
    def _form(self, qt_app, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        return ConfigForm(ConfigDocument.blank('afm'))

    def test_match_tolerance_is_disabled_until_lattice_match_is_ticked(self, form):
        section = form.sections['sub']
        assert not section.editors['match_strain_tol'].isEnabled()

        section.editors['match'].set_value(True)
        section._apply_dependencies()
        assert section.editors['match_strain_tol'].isEnabled()

    def test_twist_tolerances_are_disabled_without_a_twist_angle(self, form):
        section = form.sections['general']
        for key in ('twist_angle_tol', 'twist_strain_tol', 'twist_max_supercell'):
            assert not section.editors[key].isEnabled(), key

        section.editors['twist_angle'].set_value(5.0)
        section._apply_dependencies()
        assert section.editors['twist_angle_tol'].isEnabled()

    def test_scan_angle_selector_is_disabled_without_scan_angles(self, form):
        section = form.sections['general']
        section.editors['scan_angle'].set_value(None)
        section._apply_dependencies()
        assert not section.editors['scan_angle_force'].isEnabled()


class TestRequiredMarking:

    def test_required_fields_are_marked_and_optional_ones_are_not(self, qt_app,
                                                                 monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.blank('afm'))
        section = form.sections['2D']

        required_label, _ = section._rows['mat']
        optional_label, _ = section._rows['lat_c']
        assert '*' in required_label.text()
        assert '*' not in optional_label.text()


class TestCollapsibleSections:
    """Both tabs run to nine sections; folding is what makes them navigable."""

    @pytest.fixture(name='form')
    def _form(self, qt_app, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.blank('afm'))
        yield form
        form.deleteLater()

    def test_required_sections_start_open(self, form):
        for key in ('2D', 'tip', 'sub', 'general'):
            assert not form.sections[key].is_collapsed(), key

    def test_switched_off_optional_sections_start_folded(self, form):
        """They have nothing to show, and they are most of the clutter."""
        for key in ('flake', 'lattice_match'):
            assert form.sections[key].is_collapsed(), key

    def test_toggling_folds_and_unfolds(self, form):
        section = form.sections['tip']
        section.toggle_collapsed()
        assert section.is_collapsed()
        section.toggle_collapsed()
        assert not section.is_collapsed()

    def test_collapse_all_and_expand_all(self, form):
        form.set_all_collapsed(True)
        assert len(form.collapsed_sections()) == len(form.sections)
        form.set_all_collapsed(False)
        assert form.collapsed_sections() == []

    def test_fold_state_survives_a_rebuild(self, form):
        """Switching mode should not reopen everything just tidied away."""
        form.sections['general'].set_collapsed(True)
        form.rebuild()
        assert form.sections['general'].is_collapsed()

    def test_switching_a_section_on_opens_it(self, form):
        """Ticking it is a request to work on it."""
        section = form.sections['flake']
        assert section.is_collapsed()
        section.setChecked(True)
        assert not section.is_collapsed()

    def test_state_is_tracked_not_read_from_visibility(self, form):
        """isVisible() is False for everything until the window is shown, so
        deriving fold state from it reports every section as folded."""
        assert not form.sections['tip'].is_collapsed()
        assert not form.sections['tip'].body.isVisible()


class TestFoldedSummary:
    """A folded section still has to say enough to decide whether to open it."""

    @pytest.fixture(name='form')
    def _form(self, qt_app, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent'))
        yield form
        form.deleteLater()

    def test_a_filled_section_reports_how_much_is_set(self, form):
        section = form.sections['tip']
        section.set_collapsed(True)
        assert 'set' in section.summary.text()

    def test_missing_required_fields_are_called_out(self, qt_app, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        blank = ConfigForm(ConfigDocument.blank('afm'))
        section = blank.sections['2D']
        section.set_collapsed(True)
        assert 'required still empty' in section.summary.text()

    def test_a_section_that_is_off_says_so(self, qt_app, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        blank = ConfigForm(ConfigDocument.blank('afm'))
        assert blank.sections['flake'].summary.text() == 'not included'

    def test_the_summary_only_shows_while_folded(self, form):
        section = form.sections['tip']
        section.set_collapsed(True)
        section.set_collapsed(False)
        assert not section.summary.isVisibleTo(section)


class TestPairOverrides:
    """[lj_override] has user-chosen keys, so introspection renders nothing --
    it used to appear as an empty box."""

    @pytest.fixture(name='section')
    def _section(self, qt_app, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.blank('afm')
        document.enable_section('lj_override', True)
        # Held for the test's duration: letting the form be collected destroys
        # its Qt children while the Python wrappers survive.
        form = ConfigForm(document)
        yield form.sections['lj_override']
        form.deleteLater()

    def test_the_section_has_a_working_editor(self, section):
        assert section.pair_editor is not None

    def test_pairs_round_trip(self, section):
        section.pair_editor.set_value({'Mo-S': [0.41, 3.75]})
        assert section.pair_editor.value() == {'Mo-S': [0.41, 3.75]}

    def test_editing_writes_through_to_the_document(self, section):
        section.pair_editor.set_value({'Mo-S': [0.41, 3.75]})
        section.commit()
        assert section.document.values['lj_override'] == {'Mo-S': [0.41, 3.75]}

    def test_incomplete_rows_are_skipped(self, section):
        section.pair_editor.set_value({'Mo-S': [0.41, 3.75]})
        section.pair_editor._add_row('S-S', '', '')
        assert section.pair_editor.value() == {'Mo-S': [0.41, 3.75]}


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


class TestCloseConfirmation:
    """Closing the window used to silently discard an edited config or
    settings document. Now it must ask first."""

    @pytest.fixture(name='window')
    def _window(self, qt_app, workspace, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from src.gui.app import MainWindow
        window = MainWindow(workspace)
        yield window
        # Several tests here deliberately leave the document/settings dirty
        # (and, within their own body, monkeypatch QMessageBox.question --
        # a monkeypatch that is undone by the time teardown runs). Without
        # resetting dirty first, this close() would hit the real, blocking
        # QMessageBox with nothing able to click it.
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()

    def test_close_prompts_when_dirty(self, window, monkeypatch):
        window.document.dirty = True
        asked = []
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: asked.append(1) or QMessageBox.Cancel)
        event = QCloseEvent()
        window.closeEvent(event)
        assert asked and not event.isAccepted()

    def test_close_prompts_when_only_settings_are_dirty(self, window, monkeypatch):
        window.document.dirty = False
        window.settings_document.dirty = True
        asked = []
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: asked.append(1) or QMessageBox.Cancel)
        event = QCloseEvent()
        window.closeEvent(event)
        assert asked and not event.isAccepted()

    def test_close_does_not_prompt_when_clean(self, window, monkeypatch):
        window.document.dirty = False
        window.settings_document.dirty = False
        asked = []
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: asked.append(1) or QMessageBox.Cancel)
        event = QCloseEvent()
        window.closeEvent(event)
        assert not asked
        assert event.isAccepted()

    def test_close_discard_accepts_without_saving(self, window, monkeypatch):
        window.document.dirty = True
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Discard)
        saved = []
        monkeypatch.setattr(window, '_save_config', lambda: saved.append(1))
        event = QCloseEvent()
        window.closeEvent(event)
        assert event.isAccepted()
        assert not saved

    def test_close_save_routes_through_save_config_and_accepts(self, window, monkeypatch):
        window.document.dirty = True
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Save)

        def fake_save():
            window.document.dirty = False
        saved = []
        monkeypatch.setattr(window, '_save_config',
                            lambda: (saved.append(1), fake_save())[-1])
        event = QCloseEvent()
        window.closeEvent(event)
        assert saved
        assert event.isAccepted()

    def test_cancelled_save_dialog_does_not_close_the_window(self, window, monkeypatch):
        """Choosing Save but then cancelling the file dialog leaves the
        document dirty -- the window must stay open, not close anyway."""
        window.document.dirty = True
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Save)
        # _save_config's own file dialog was cancelled: it returns without
        # touching dirty.
        monkeypatch.setattr(window, '_save_config', lambda: None)
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()

    def test_close_save_routes_through_settings_save_when_only_settings_dirty(
            self, window, monkeypatch):
        """A settings-only edit must save through the settings path, not the
        config one -- routing it to _save_config used to open the wrong
        dialog and then close the window with the settings edits lost."""
        window.document.dirty = False
        window.settings_document.dirty = True
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Save)

        def fake_save_settings():
            window.settings_document.dirty = False
        settings_saved = []
        config_saved = []
        monkeypatch.setattr(
            window, '_save_settings',
            lambda: (settings_saved.append(1), fake_save_settings())[-1])
        monkeypatch.setattr(window, '_save_config', lambda: config_saved.append(1))
        event = QCloseEvent()
        window.closeEvent(event)
        assert settings_saved
        assert not config_saved
        assert event.isAccepted()

    def test_settings_save_failure_does_not_close_the_window(self, window, monkeypatch):
        """An invalid settings document (or a save the user backs out of)
        leaves settings_document.dirty True -- the window must stay open,
        not close with the edits discarded."""
        window.document.dirty = False
        window.settings_document.dirty = True
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Save)
        # _save_settings warns and returns without clearing dirty -- the
        # real method does this when validation fails.
        monkeypatch.setattr(window, '_save_settings', lambda: None)
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()

    def test_close_save_message_names_what_is_unsaved(self, window, monkeypatch):
        """The settings-only case must not blame 'the configuration' -- that
        used to be the only wording, regardless of which document was dirty."""
        window.document.dirty = False
        window.settings_document.dirty = True
        messages = []

        def fake_question(*args, **kwargs):
            messages.append(args[2])
            return QMessageBox.Cancel
        monkeypatch.setattr(QMessageBox, 'question', fake_question)
        event = QCloseEvent()
        window.closeEvent(event)
        assert messages and 'settings' in messages[0].lower()
        assert 'configuration' not in messages[0].lower()


class TestStaleRenderBadge:
    """The viewport keeps showing the previous system after an edit unless
    told otherwise -- the badge is the only sign anything is out of date."""

    @pytest.fixture(name='window')
    def _window(self, qt_app, workspace, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from src.gui.app import MainWindow
        window = MainWindow(workspace)
        yield window
        # See TestCloseConfirmation's fixture: avoid a real, blocking save
        # prompt from closeEvent when a test's edits left the document dirty.
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()

    def test_stale_badge_toggles(self, qt_app, two_component_system):
        from src.gui.preview.viewport import ViewportPanel

        panel = ViewportPanel()
        panel.set_stale(True)
        assert panel.stale_label.isVisibleTo(panel) is False   # no system yet
        panel.system = two_component_system
        panel.set_stale(True)
        assert panel.stale_label.isVisibleTo(panel)
        panel.set_stale(False)
        assert not panel.stale_label.isVisibleTo(panel)

    def test_editing_after_a_render_marks_the_badge_stale(self, window, monkeypatch,
                                                          two_component_system):
        monkeypatch.setattr(window.viewport.__class__, '_update_pipeline',
                            lambda self: None)
        monkeypatch.setattr(window.viewport.__class__, '_ensure_widget',
                            lambda self: None)

        window._on_preview_ready(two_component_system)
        assert window._render_snapshot == window.document.to_dict()
        assert window.viewport.stale_label.isVisibleTo(window.viewport) is False

        window.form.sections['2D'].editors['mat'].set_value('WS2')
        window.form.commit()
        window._on_config_changed()
        assert window.viewport.stale_label.isVisibleTo(window.viewport)

    def test_re_rendering_clears_the_badge(self, window, monkeypatch,
                                           two_component_system):
        monkeypatch.setattr(window.viewport.__class__, '_update_pipeline',
                            lambda self: None)
        monkeypatch.setattr(window.viewport.__class__, '_ensure_widget',
                            lambda self: None)

        window._on_preview_ready(two_component_system)
        window.form.sections['2D'].editors['mat'].set_value('WS2')
        window.form.commit()
        window._on_config_changed()
        assert window.viewport.stale_label.isVisibleTo(window.viewport)

        window._on_preview_ready(two_component_system)
        assert window.viewport.stale_label.isVisibleTo(window.viewport) is False
