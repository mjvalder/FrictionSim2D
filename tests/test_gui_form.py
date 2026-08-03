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


class TestCountsRead:

    @pytest.mark.parametrize('n,word,expected', [
        (1, 'config', '1 config'),
        (2, 'config', '2 configs'),
        (1, 'build', '1 build'),
    ])
    def test_plural(self, n, word, expected):
        from src.gui.app import plural

        assert plural(n, word) == expected
