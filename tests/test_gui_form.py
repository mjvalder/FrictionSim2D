"""What the form says about itself.

The docs carry a table of which fields may hold a list. It belongs in the form,
generated from the same LOOPABILITY map the expansion uses -- not restated by
hand in the override table, where it drifts.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.controllers.document import ConfigDocument  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(name='form')
def _form(qt_app, monkeypatch):
    from src.gui.widgets.form import ConfigForm

    monkeypatch.chdir(REPO_ROOT)
    form = ConfigForm(ConfigDocument.blank('afm'))
    yield form
    form.deleteLater()


class TestInfoBubbles:
    """Captions ate a caption-worth of vertical space on every row; an ⓘ
    bubble at the end of the row carries the same text as a tooltip instead."""

    def test_bubble_tooltip_carries_help_and_default(self, qt_app):
        from src.gui.widgets.form import SectionForm

        doc = ConfigDocument.blank('sheetonsheet')
        form = SectionForm(doc.section('general'), doc)
        tip = form._bubbles['scan_speed'].toolTip()
        assert '100 m/s' in tip
        assert not hasattr(form, '_captions')

    def test_hint_only_field_still_gets_a_bubble(self, qt_app):
        from src.gui.widgets.form import SectionForm

        doc = ConfigDocument.blank('sheetonsheet')
        form = SectionForm(doc.section('general'), doc)
        assert 'twist_strain_tol' in form._bubbles   # enable_hint only, no help


class TestLoopabilityHints:

    def _tip(self, form, section, key):
        return form.sections[section]._bubbles[key].toolTip()

    def test_a_swept_field_says_a_list_makes_separate_runs(self, form):
        assert 'separate simulation' in self._tip(form, 'general', 'temp')

    def test_a_layer_field_says_a_list_makes_separate_builds(self, form):
        assert 'L{n}' in self._tip(form, '2D', 'layers')

    def test_an_in_file_field_says_a_list_is_looped(self, form):
        assert 'looped inside' in self._tip(form, 'general', 'force')

    def test_a_scalar_field_gets_no_hint(self, form):
        """Bubbling every single-value field with a negative is noise; a list
        that is actually present is already reported by validation."""
        bubble = form.sections['tip']._bubbles.get('r')
        assert bubble is not None
        assert 'list' not in bubble.toolTip()

    def test_the_hint_is_appended_not_substituted(self, qt_app, monkeypatch):
        """No field currently rendered by the form carries both a hand-written
        help text and a non-scalar loopability at once: Task 3 emptied `temp`,
        `layers` and `force` precisely so the hint would be the only text, and
        `[2D] materials_list` -- the one override entry that does have both --
        is never introspected as a real field (it is set programmatically, see
        `tests/test_gui_material_list.py`), so it never reaches `_bubbles` at
        all. A field with only one of the two cannot distinguish append from
        substitute, so this synthesises one: give `[general] temp` (loopability
        'sweep') a temporary hand-written help text and drive it through the
        real `_add_field` path. Substituting instead of appending would drop
        one of the two strings; this fails either way that happens."""
        from src.gui.schema import overrides as overrides_mod
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        temp_fields = overrides_mod._COMMON['general'].fields
        synthetic = replace(temp_fields['temp'], help='Ambient control temperature.')
        monkeypatch.setitem(temp_fields, 'temp', synthetic)

        form = ConfigForm(ConfigDocument.blank('afm'))
        tip = self._tip(form, 'general', 'temp')
        assert 'Ambient control temperature.' in tip      # the declared help
        assert 'separate simulation' in tip               # the generated hint
        form.deleteLater()

    def test_settings_fields_get_no_hint(self, qt_app, monkeypatch):
        """SettingsDocument has no loopability concept at all."""
        from src.gui.controllers.settings_document import SettingsDocument
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(SettingsDocument.defaults())
        bubble = form.sections['hpc']._bubbles.get('num_cpus')
        assert bubble is None or 'list' not in bubble.toolTip()
        form.deleteLater()


class TestDependentFieldsKeepTheirHint:
    """scan_angle_force is the one field with both a dependency and a
    loopability. _apply_dependencies rewrites its tooltip, and rewriting it
    from the override alone silently drops the generated hint."""

    def test_the_hint_returns_when_the_dependency_is_satisfied(self, form):
        section = form.sections['general']
        section.editors['scan_angle'].set_value(5.0)
        section._apply_dependencies()
        tip = section._bubbles['scan_angle_force'].toolTip()
        assert 'Empty: every angle at every load.' in tip
        assert 'looped inside' in tip

    def test_the_enable_hint_still_leads_while_disabled(self, form):
        section = form.sections['general']
        section.editors['scan_angle'].set_value(None)
        section._apply_dependencies()
        assert section._bubbles['scan_angle_force'].toolTip() == \
            'Set one or more scan angles first.'


class TestScanAngleZeroIsSet:
    """scan_angle defaults to 0.0 -- a legitimate straight-line scan, not
    "unset" -- so scan_angle_force's `enable_if` must test presence (`SET`),
    not truthiness (`ENABLED`), or the row is permanently greyed on every
    blank config with no on-screen reason (Task 6 moved that reason into a
    tooltip-only bubble)."""

    def test_scan_angle_force_enabled_on_blank_config(self, form):
        section = form.sections['general']
        assert section.document.get('general', 'scan_angle') == 0.0
        assert section.editors['scan_angle_force'].isEnabled()
        label, _cell = section._rows['scan_angle_force']
        assert label.isEnabled()

    def test_scan_angle_force_disabled_when_scan_angle_cleared(self, form):
        section = form.sections['general']
        section.editors['scan_angle'].set_value(None)
        section._apply_dependencies()
        assert not section.editors['scan_angle_force'].isEnabled()
        label, _cell = section._rows['scan_angle_force']
        assert not label.isEnabled()

    def test_visible_hint_while_disabled_and_placeholder_restored_when_enabled(
            self, form):
        section = form.sections['general']
        editor = section.editors['scan_angle_force']
        normal_placeholder = editor.edit.placeholderText()

        section.editors['scan_angle'].set_value(None)
        section._apply_dependencies()
        assert editor.edit.placeholderText() == 'Set one or more scan angles first.'

        section.editors['scan_angle'].set_value(5.0)
        section._apply_dependencies()
        assert editor.edit.placeholderText() == normal_placeholder


class TestNestedFieldsAreNotHidden:
    """A nested sub-model has no flat control, so it is not rendered. Silently
    omitting it is indistinguishable from the field not existing."""

    def test_a_nested_field_is_shown_as_read_only(self, qt_app, monkeypatch):
        from src.gui.controllers.settings_document import SettingsDocument
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(SettingsDocument.defaults())
        section = form.sections['database']

        for key in ('local', 'central'):
            assert key in section._rows, key
            assert key not in section.editors, key
        form.deleteLater()

    def test_a_nested_value_survives_a_commit(self, qt_app, monkeypatch):
        """`SettingsDocument` has no `to_dict`; `.values` is the same plain
        dict a real save would serialise, so a deep copy of it is the
        equivalent check."""
        import copy

        from src.gui.controllers.settings_document import SettingsDocument
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = SettingsDocument.defaults()
        before = copy.deepcopy(document.values)
        form = ConfigForm(document)
        form.commit()
        assert document.values == before
        form.deleteLater()


class TestErrorClearingPreservesTheTooltip:
    """show_errors() clears every editor's error with set_error(None) before
    reapplying issues -- including on the happy path, where there is nothing
    to reapply. set_error wrote straight to setToolTip, so that alone blanked
    the field's explanatory tooltip every time a config was opened or Render
    was pressed."""

    def test_set_error_none_restores_the_explanatory_tooltip(self, form):
        """stack_type has both a caption/tooltip (help) and a detail."""
        editor = form.sections['2D'].editors['stack_type']
        before = editor.toolTip()
        assert before

        editor.set_error('some validation problem')
        assert editor.toolTip() == 'some validation problem'

        editor.set_error(None)
        assert editor.toolTip() == before

    def test_show_errors_with_no_issues_keeps_the_tooltip(self, form):
        section = form.sections['2D']
        editor = section.editors['stack_type']
        before = editor.toolTip()
        assert before

        section.show_errors([])
        assert editor.toolTip() == before


def _settings_with_drive(method):
    from src.core.config import GlobalSettings

    s = GlobalSettings()
    s.simulation.drive_method = method
    return s


class TestRequiredStars:
    """A field can be obligatory only given a Settings choice: the driving
    spring only matters when the virtual-atom drive method is selected."""

    def test_driving_spring_star_follows_drive_method(self, qt_app):
        from src.gui.controllers.document import ConfigDocument
        from src.gui.widgets.form import SectionForm

        doc = ConfigDocument.blank('sheetonsheet')
        form = SectionForm(doc.section('general'), doc,
                           settings=_settings_with_drive('virtual_atom'))
        assert form.effective_required('driving_spring') is True
        form2 = SectionForm(doc.section('general'), doc,
                            settings=_settings_with_drive('smd'))
        assert form2.effective_required('driving_spring') is False

    def test_temp_is_starred(self, qt_app):
        from src.gui.controllers.document import ConfigDocument
        from src.gui.widgets.form import SectionForm
        from src.gui.style import REQUIRED_MARK

        doc = ConfigDocument.blank('sheetonsheet')
        form = SectionForm(doc.section('general'), doc,
                           settings=_settings_with_drive('smd'))
        label, _ = form._rows['temp']
        assert label.text().endswith(REQUIRED_MARK)

    def test_driving_spring_star_updates_with_set_settings(self, qt_app):
        """set_settings re-derives the star in place, without rebuilding --
        the Settings tab's Apply/Save must move the star on the config form
        that is already on screen."""
        from src.gui.controllers.document import ConfigDocument
        from src.gui.widgets.form import SectionForm
        from src.gui.style import REQUIRED_MARK

        doc = ConfigDocument.blank('sheetonsheet')
        form = SectionForm(doc.section('general'), doc,
                           settings=_settings_with_drive('smd'))
        label, _ = form._rows['driving_spring']
        assert not label.text().endswith(REQUIRED_MARK)

        form.set_settings(_settings_with_drive('virtual_atom'))
        assert label.text().endswith(REQUIRED_MARK)

    def test_set_settings_prefills_a_field_newly_required_by_the_switch(self, qt_app):
        """A field that becomes required only after `set_settings` (not at
        form-build time) must be pre-filled just like `load()`'s own prefill
        does for one that was required from the start -- otherwise it sits
        starred-and-blank, blocking Generate, even though it has a perfectly
        good default."""
        from src.gui.controllers.document import ConfigDocument
        from src.gui.widgets.form import SectionForm

        doc = ConfigDocument.blank('sheetonsheet')
        doc.values['general'].pop('driving_spring', None)
        form = SectionForm(doc.section('general'), doc,
                           settings=_settings_with_drive('smd'))
        assert form.editors['driving_spring'].value() in (None, '', [])

        form.set_settings(_settings_with_drive('virtual_atom'))
        assert form.editors['driving_spring'].value() not in (None, '', [])

    def test_force_is_required_and_blocks(self):
        from src.gui.schema.overrides import overrides_for

        force = overrides_for('afm')['general'].fields['force']
        assert force.required is True
        assert force.required_blocks is True

    def test_pressure_is_required_and_blocks(self):
        """The `[0.0]` fallback only exists in the two outer_loop branches
        (sheetonsheet.py:341,359); the default single-script path passes
        `general.pressure` straight through unguarded, so an absent pressure
        renders `variable pressure equal None` in slide.lmp -- the same
        unparseable-LAMMPS failure class as an empty force list."""
        from src.gui.schema.overrides import overrides_for

        pressure = overrides_for('sheetonsheet')['general'].fields['pressure']
        assert pressure.required is True
        assert pressure.required_blocks is True

    def test_missing_required_reports_force_as_blocking(self, qt_app, monkeypatch):
        from src.gui.controllers.document import ConfigDocument
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.blank('afm'))
        entries = {(sec, key): blocks for sec, key, _label, blocks
                  in form.missing_required()}
        assert entries[('general', 'force')] is True
        form.deleteLater()

    def test_missing_required_reports_pressure_as_blocking(self, qt_app,
                                                            monkeypatch):
        from src.gui.controllers.document import ConfigDocument
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        doc = ConfigDocument.blank('sheetonsheet')
        doc.set('general', 'pressure', None)
        form = ConfigForm(doc)
        entries = {(sec, key): blocks for sec, key, _label, blocks
                  in form.missing_required()}
        assert entries[('general', 'pressure')] is True
        form.deleteLater()


class TestStarredFieldPrefill:
    """A starred field that is empty should load its default and commit it
    explicitly, so the written config is not silently missing a required
    value just because the user never touched the box."""

    def test_starred_field_prefills_and_commits_default(self, qt_app):
        from src.gui.controllers.document import ConfigDocument
        from src.gui.widgets.form import SectionForm

        doc = ConfigDocument.blank('sheetonsheet')
        doc.values['general'].pop('temp', None)
        form = SectionForm(doc.section('general'), doc, settings=None)
        form.load(); form.commit()
        assert doc.get('general', 'temp') == 300.0


class TestTipPositionGate:
    """The four tip-position fields only matter when the tip is deliberately
    placed; the gate keeps them out of sight and out of the document until
    then."""

    def test_gate_unticked_on_blank_ticked_on_explicit(self, qt_app):
        from src.gui.controllers.document import ConfigDocument
        from src.gui.widgets.form import SectionForm

        doc = ConfigDocument.blank('afm')
        form = SectionForm(doc.section('tip'), doc)
        assert not form._gate_boxes['tip_position'].isChecked()
        doc.set('tip', 'tip_x', 12.0)
        form.load()
        assert form._gate_boxes['tip_position'].isChecked()

    def test_unticking_gate_drops_keys(self, qt_app):
        from src.gui.controllers.document import ConfigDocument
        from src.gui.widgets.form import SectionForm

        doc = ConfigDocument.blank('afm')
        doc.set('tip', 'tip_x', 12.0)
        form = SectionForm(doc.section('tip'), doc)
        form._gate_boxes['tip_position'].setChecked(False)
        form.commit()
        assert doc.get('tip', 'tip_x') is None

    def test_opening_the_example_config_stays_clean_after_a_commit(self, qt_app):
        """examples/afm_config.ini ships an explicit `tip_x_offset = 0.0` --
        equal to the field's own spec default. `_load_gates` reads that as
        untouched and ticks no box, so `commit()` pops it right back out;
        that pop is a no-op (the builder already treats an absent offset as
        0.0) and must not be reported as a real edit. Since `_validate()`
        calls `commit()` on every Render, failing this used to mark a config
        the user never touched dirty just from pressing the button."""
        from src.gui.controllers.document import ConfigDocument
        from src.gui.widgets.form import SectionForm

        doc = ConfigDocument.from_file(
            REPO_ROOT / 'examples' / 'afm_config.ini', 'afm')
        assert not doc.dirty
        form = SectionForm(doc.section('tip'), doc)
        form.commit()
        assert not doc.dirty


class TestOuterLoopAngleLabel:
    """The dropdown shows 'angle' for the scan_angle split, but the stored
    value -- and what gets written to the .ini -- must stay 'scan_angle':
    run.py's `_run_scripts_per_build` does a literal `general.get(outer_loop)`
    lookup, and the builders / GeneralConfig.outer_loop's Literal all compare
    against the string 'scan_angle'. A label change that leaked into the
    stored value would silently break every job-count and build that uses it."""

    def test_combo_displays_angle_for_the_scan_angle_choice(self, form):
        box = form.sections['general'].editors['outer_loop'].box
        index = box.findData('scan_angle')
        assert index != -1
        assert box.itemText(index) == 'angle'

    def test_other_choices_keep_their_own_text(self, form):
        box = form.sections['general'].editors['outer_loop'].box
        index = box.findData('force')
        assert index != -1
        assert box.itemText(index) == 'force'

    def test_value_stays_scan_angle_once_angle_is_selected(self, form):
        editor = form.sections['general'].editors['outer_loop']
        editor.box.setCurrentIndex(editor.box.findData('scan_angle'))
        assert editor.value() == 'scan_angle'

    def test_saved_config_writes_scan_angle_not_angle(self, form, tmp_path):
        from src.core.config import parse_config

        editor = form.sections['general'].editors['outer_loop']
        editor.box.setCurrentIndex(editor.box.findData('scan_angle'))
        form.commit()
        out = form.document.save(tmp_path / 'afm.ini')
        assert parse_config(out)['general']['outer_loop'] == 'scan_angle'


class TestEmptySplitControlIsHidden:
    """Both PES modes offer no outer_loop choice at all (their builders
    override write_inputs and never read it), which left the combo rendering
    with nothing in it but '(unset)'. An empty control reads as a bug; the
    row is dropped instead, like any other key the family cannot use."""

    @pytest.mark.parametrize('model', ['pes_tip', 'pes_sheet'])
    def test_no_split_control_where_there_is_nothing_to_split(self, qt_app,
                                                              monkeypatch, model):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.blank(model))
        section = form.sections['general']
        assert 'outer_loop' not in section.editors
        assert 'outer_loop' not in section._rows
        form.deleteLater()

    @pytest.mark.parametrize('model', ['afm', 'indent', 'sheetonsheet'])
    def test_the_control_stays_where_a_split_is_offered(self, qt_app,
                                                        monkeypatch, model):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.blank(model))
        assert 'outer_loop' in form.sections['general'].editors
        form.deleteLater()


class TestCountsRead:

    @pytest.mark.parametrize('n,word,expected', [
        (1, 'config', '1 config'),
        (2, 'config', '2 configs'),
        (1, 'build', '1 build'),
    ])
    def test_plural(self, n, word, expected):
        from src.gui.app import plural

        assert plural(n, word) == expected


class TestSecondMaterialErrors:
    """A validation failure on the second sheet must land on the editor that
    owns it. Pydantic locates it at ('sheets', 1, 'cif_path'), and
    `show_errors` matches on `issue.key` -- so mapping the section alone
    leaves the issue keyed by the list index and decorates nothing at all.
    """

    def test_a_second_sheet_issue_decorates_its_editor(self, qt_app, monkeypatch):
        from src.gui.widgets.form import SectionForm

        monkeypatch.chdir(REPO_ROOT)
        doc = ConfigDocument.blank('sheetonsheet')
        doc.add_second_material()           # required fields empty -> errors
        form = SectionForm(doc.second_material_spec(), doc)

        form.show_errors(doc.validate().issues)

        assert form.editors['cif_path'].error()
        form.deleteLater()

    def test_the_first_sheet_editor_is_still_decorated(self, qt_app, monkeypatch):
        """The pair renames [2D] to [2D-1] on the way out; the form section is
        still '2D', so its issues must be mapped back."""
        from src.gui.widgets.form import SectionForm

        monkeypatch.chdir(REPO_ROOT)
        doc = ConfigDocument.blank('sheetonsheet')
        doc.add_second_material()
        form = SectionForm(doc.section('2D'), doc)

        form.show_errors(doc.validate().issues)

        assert form.editors['cif_path'].error()
        form.deleteLater()


class TestPairConfigSurvivesTheForm:
    """Making a pair config openable (Task 1) also makes it commit-able. The
    `[2D]` section owns the sweep fields, and its commit path rewrites
    mat/cif_path/pot_path whenever it decides a sweep is configured -- so an
    untouched round trip through the form must leave both sheets' {mat1}/
    {mat2} templates and the pair list exactly where they were."""

    def test_an_untouched_commit_changes_nothing(self, qt_app, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        doc = ConfigDocument.from_file(
            REPO_ROOT / 'examples' / 'hetero_pair_sweep.ini', 'sheetonsheet')
        before = doc.to_dict()

        form = ConfigForm(doc)
        form.commit()

        assert doc.to_dict() == before
        form.deleteLater()


class TestSweepDirectoriesFromPairTemplates:
    """`_sweep_directories` reads the path editors, so it has to recognise
    every placeholder spelling: a `{mat1}` path's directory is its parent, not
    the path itself. Testing for the literal `'{mat}'` makes the structure
    picker of an opened pair config point at a file-shaped 'directory' that
    does not exist, so the material listing scans empty and every ticked
    material reads as unusable."""

    def test_a_pair_template_resolves_to_its_directory(self, qt_app, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        doc = ConfigDocument.from_file(
            REPO_ROOT / 'examples' / 'hetero_pair_sweep.ini', 'sheetonsheet')
        form = ConfigForm(doc)

        structures, potentials, extension = form.sections['2D']._sweep_directories()

        base = doc.base_dir
        assert structures == base / 'examples' / 'materials'
        assert potentials == base / 'examples' / 'potentials' / 'sw' / 'sw_lammps'
        assert extension == '.sw'
        form.deleteLater()
