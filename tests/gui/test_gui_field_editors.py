"""Editors must return exactly what they were given.

Every widget here replaces a line of raw JSON or a free-text box. The risk they
all share is silent narrowing: dropping a key the widget does not display, or
snapping an unrecognised value to a legal one before validation ever sees it.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.schema.introspect import FieldKind, FieldSpec  # noqa: E402


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


def _spec(key='stack_type', kind=FieldKind.STR, **kwargs):
    return FieldSpec(name=key, key=key, kind=kind, required=False, **kwargs)


class TestStringChoices:
    """stack_type is typed `str` on the model, but the builders only act on AA
    versus not-AA. A combo is honest about that without changing the model."""

    def test_a_string_field_with_choices_becomes_a_combo(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        assert hasattr(editor, 'box')

    def test_the_declared_choices_are_offered(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        offered = {editor.box.itemData(i) for i in range(editor.box.count())}
        assert {'AA', 'AB'} <= offered

    def test_an_unrecognised_value_is_preserved_not_snapped(self, qt_app):
        """The regression: opening a hand-written `stack_type = ABC` and saving
        used to write AA, silently changing the simulation."""
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        editor.set_value('ABC')
        assert editor.value() == 'ABC'

    def test_an_unrecognised_value_is_flagged(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        editor.set_value('ABC')
        assert editor.warning() is not None
        assert 'ABC' in editor.warning()

    def test_a_recognised_value_is_not_flagged(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        editor.set_value('AB')
        assert editor.warning() is None

    def test_the_warning_survives_the_value_being_revisited(self, qt_app):
        """load() reuses long-lived editors, so the same widget sees many
        configs. A stale synthetic item made findData succeed on the second
        visit, so the value was no longer recognised as unrecognised."""
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        editor.set_value('ABC')
        editor.set_value('AB')
        editor.set_value('ABC')
        assert editor.value() == 'ABC'
        assert editor.warning() is not None

    def test_synthetic_entries_do_not_accumulate(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        before = editor.box.count()
        for value in ('ABC', 'ABD', 'ABC', 'AA'):
            editor.set_value(value)
        assert editor.box.count() == before

    def test_picking_a_real_value_retires_the_synthetic_entry(self, qt_app):
        """The user correcting a flagged field is the normal path. Clearing the
        flag without removing the item leaves it for findData to match later,
        which silently un-flags the value on the next load()."""
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        before = editor.box.count()
        editor.set_value('ABC')

        editor.box.setCurrentIndex(editor.box.findData('AB'))   # as the user would
        assert editor.value() == 'AB'
        assert editor.warning() is None
        assert editor.box.count() == before

        editor.set_value('ABC')          # another config, same odd value
        assert editor.value() == 'ABC'
        assert editor.warning() is not None


class TestRegistry:

    def test_an_unknown_slug_falls_back_to_the_kind(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(kind=FieldKind.STR), editor='no_such_slug')
        assert hasattr(editor, 'edit')


class TestStackingOverride:

    def test_stacking_offers_only_aa_and_ab(self):
        from src.gui.schema.overrides import section_override

        assert section_override('2D').fields['stack_type'].choices == ('AA', 'AB')

    def test_stacking_explains_itself(self):
        from src.gui.schema.overrides import section_override

        override = section_override('2D').fields['stack_type']
        assert override.help


DEFAULT_BANDS = {
    'tip': {'fix': [3.0, 0.0], 'thermo': [6.0, 3.0]},
    'sub': {'fix': [0.0, 0.3], 'thermo': [0.3, 0.6]},
}


class TestLangevinBands:
    """One line of JSON, replaced by a table that names its own units."""

    @pytest.fixture(name='editor')
    def _editor(self, qt_app):
        from src.gui.widgets.bands import LangevinBoundaryEditor

        return LangevinBoundaryEditor(_spec(key='langevin_boundaries',
                                            kind=FieldKind.DICT))

    def test_the_default_round_trips(self, editor):
        editor.set_value(DEFAULT_BANDS)
        assert editor.value() == DEFAULT_BANDS

    def test_the_settings_default_round_trips(self, editor):
        """Against the model, not a copy of it."""
        from src.core.config import ThermostatSettings

        bands = ThermostatSettings().langevin_boundaries
        editor.set_value(bands)
        assert editor.value() == bands

    def test_an_undisplayed_component_is_preserved(self, editor):
        """The widget shows tip and sub. A third component must survive it."""
        extra = dict(DEFAULT_BANDS, flake={'fix': [1.0, 0.0]})
        editor.set_value(extra)
        assert editor.value()['flake'] == {'fix': [1.0, 0.0]}

    def test_editing_one_cell_changes_only_that_number(self, editor):
        editor.set_value(DEFAULT_BANDS)
        editor.set_cell('tip', 'fix', 0, 4.5)
        result = editor.value()
        assert result['tip']['fix'] == [4.5, 0.0]
        assert result['sub'] == DEFAULT_BANDS['sub']

    def test_the_two_unit_systems_are_stated(self, editor):
        """Tip bands are absolute Angstrom from the top; substrate bands are a
        fraction of thickness from the bottom. Nothing said so before."""
        text = editor.describe()
        assert 'Å' in text
        assert 'fraction' in text.lower()

    def test_an_emptied_table_is_flagged(self, editor):
        editor.set_value(DEFAULT_BANDS)
        for band in ('fix', 'thermo'):
            for index in (0, 1):
                editor.set_cell('tip', band, index, '')
                editor.set_cell('sub', band, index, '')
        assert editor.value() is None
        assert editor.warning() is not None

    def test_a_half_filled_table_names_what_is_missing(self, editor):
        """The builders index both keys, so one is as fatal as none."""
        editor.set_value({'tip': DEFAULT_BANDS['tip']})
        assert 'sub' in editor.warning()

    def test_a_complete_table_is_not_flagged(self, editor):
        editor.set_value(DEFAULT_BANDS)
        assert editor.warning() is None


class TestKeyValueGrid:

    @pytest.fixture(name='editor')
    def _editor(self, qt_app):
        """output.dump: Dict[str, bool]."""
        from src.gui.widgets.keyvalue import KeyValueGridEditor

        return KeyValueGridEditor(
            _spec(key='dump', kind=FieldKind.DICT, item_kind=FieldKind.BOOL))

    @pytest.fixture(name='int_editor')
    def _int_editor(self, qt_app):
        """output.dump_frequency: Dict[str, int]."""
        from src.gui.widgets.keyvalue import KeyValueGridEditor

        return KeyValueGridEditor(
            _spec(key='dump_frequency', kind=FieldKind.DICT, item_kind=FieldKind.INT))

    def test_bools_round_trip(self, editor):
        value = {'system_init': True, 'slide': False}
        editor.set_value(value)
        assert editor.value() == value

    def test_ints_round_trip(self, int_editor):
        value = {'system_init': 1000, 'slide': 10000}
        int_editor.set_value(value)
        assert int_editor.value() == value

    def test_a_row_can_be_added(self, editor):
        editor.set_value({'slide': True})
        editor.add_row('minimise', True)
        assert editor.value() == {'slide': True, 'minimise': True}

    def test_a_row_can_be_removed(self, editor):
        editor.set_value({'slide': True, 'minimise': False})
        editor.remove_row('minimise')
        assert editor.value() == {'slide': True}

    def test_empty_is_none(self, editor):
        editor.set_value({})
        assert editor.value() is None

    def test_a_dump_phase_key_is_captioned_without_its_underscore(self, editor):
        """`system_init` is a dump-phase key, so it reaches the form verbatim
        and read as an identifier rather than a label. Only the caption
        changes -- value() still returns the stored key."""
        from PySide6.QtWidgets import QLabel

        editor.set_value({'system_init': True})
        row, _control = editor._rows['system_init']
        assert row.findChild(QLabel).text() == 'system init'
        assert editor.value() == {'system_init': True}

    def test_an_added_row_matches_the_declared_item_kind(self, qt_app):
        """add_row used to choose a checkbox from the runtime type of the
        value it was handed, and _on_add_clicked always passes '' -- so a row
        added by hand to a Dict[str, bool] field like output.dump got a text
        box instead of a checkbox."""
        from PySide6.QtWidgets import QCheckBox

        from src.gui.widgets.keyvalue import KeyValueGridEditor

        editor = KeyValueGridEditor(
            _spec(key='dump', kind=FieldKind.DICT, item_kind=FieldKind.BOOL))
        editor.add_row('minimise', '')
        _row, control = editor._rows['minimise']
        assert isinstance(control, QCheckBox)


class TestFixedList:

    @pytest.fixture(name='editor')
    def _editor(self, qt_app):
        from src.gui.widgets.keyvalue import FixedListEditor

        return FixedListEditor(_spec(key='quench_slab_dims', kind=FieldKind.LIST,
                                     item_kind=FieldKind.INT))

    def test_three_dimensions_round_trip(self, editor):
        editor.set_value([200, 200, 50])
        assert editor.value() == [200, 200, 50]

    def test_a_wrong_length_list_is_kept_and_flagged(self, editor):
        """Truncating to fit the cells would silently change the quench box."""
        editor.set_value([200, 200])
        assert editor.value() == [200, 200]
        assert editor.warning() is not None


class TestPathEditorBrowse:
    """_browse's comment: setText() alone does not fire editingFinished, so a
    browse selection has to raise it explicitly, or the [2D] section's
    material-sweep rescan wiring (which listens on editingFinished, not
    `changed` -- see SectionForm._wire_material_sweep) never refreshes the
    listing after a browse pick."""

    def test_a_file_browse_selection_raises_editing_finished(self, qt_app, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        from src.gui.widgets.fields import build_editor

        ed = build_editor(FieldSpec(name='cif_path', key='cif_path',
                                    kind=FieldKind.PATH, required=True))
        monkeypatch.setattr(QFileDialog, 'getOpenFileName',
                            lambda *a, **k: ('/some/chosen/path.cif', ''))

        fired = []
        ed.edit.editingFinished.connect(lambda: fired.append(1))
        ed._browse()

        assert fired
        assert ed.edit.text() == '/some/chosen/path.cif'

    def test_a_directory_browse_selection_raises_editing_finished(self, qt_app,
                                                                   monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        from src.gui.widgets.fields import build_editor

        ed = build_editor(FieldSpec(name='cif_path', key='cif_path',
                                    kind=FieldKind.PATH, required=True))
        ed.set_directory_mode(True)
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                            lambda *a, **k: '/some/chosen/dir')

        fired = []
        ed.edit.editingFinished.connect(lambda: fired.append(1))
        ed._browse()

        assert fired
        assert ed.edit.text() == '/some/chosen/dir'

    def test_a_cancelled_browse_does_not_raise_editing_finished(self, qt_app,
                                                                 monkeypatch):
        """No path chosen -- nothing changed, nothing to rescan."""
        from PySide6.QtWidgets import QFileDialog

        from src.gui.widgets.fields import build_editor

        ed = build_editor(FieldSpec(name='cif_path', key='cif_path',
                                    kind=FieldKind.PATH, required=True))
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: ('', ''))

        fired = []
        ed.edit.editingFinished.connect(lambda: fired.append(1))
        ed._browse()

        assert not fired


class TestSetErrorIdempotence:
    """set_error should not restyle the field when repeating the same state."""

    def test_set_error_is_idempotent(self, qt_app):
        from src.gui.widgets.fields import build_editor

        ed = build_editor(FieldSpec(name='x', key='x', kind=FieldKind.FLOAT, required=False))
        calls = []
        ed._apply_error_style = lambda invalid: calls.append(invalid)
        ed.set_error(None); ed.set_error(None)
        assert calls == []
        ed.set_error('bad'); ed.set_error('bad')
        assert calls == [True]


class TestDefaultGhosting:
    """A default should be visible even when the field is empty, so a blank
    box reads as "unset, will fall back to X" rather than "unset, no idea"."""

    def test_optional_field_ghosts_its_default(self, qt_app):
        from src.gui.schema.introspect import FieldSpec, FieldKind
        from src.gui.widgets.fields import build_editor
        ed = build_editor(FieldSpec(name='x', key='x', kind=FieldKind.FLOAT,
                                    required=False, default=80.0))
        assert ed.edit.placeholderText() == 'default: 80.0'


class TestEmptyListRoundTrip:
    """An empty list must come back as an empty list, not as None.

    `_ListEditor.set_value([])` writes an empty box, and reading that back as
    None made `settings_form.commit()` change `output.optional_channels` from
    [] to None on a form nobody had touched. That dirtied a fresh
    SettingsDocument, so pressing Generate wrote a settings.yaml the user never
    asked for, carrying `optional_channels: null`.
    """

    def _editor(self, qt_app, default):
        del qt_app
        from src.gui.widgets.fields import _ListEditor

        return _ListEditor(_spec(key='optional_channels',
                                 kind=FieldKind.LIST,
                                 item_kind=FieldKind.STR, default=default))

    def test_an_empty_list_survives_set_then_get(self, qt_app):
        editor = self._editor(qt_app, default=[])

        editor.set_value([])

        assert editor.value() == []

    def test_a_cleared_box_is_still_empty_for_an_empty_list_default(
            self, qt_app):
        editor = self._editor(qt_app, default=[])
        editor.set_value(['gap_min'])

        editor.set_value([])

        assert editor.value() == []

    def test_a_non_empty_default_still_reads_an_empty_box_as_unset(
            self, qt_app):
        """`quench_slab_dims` has a real default and `[2D] layers` is
        required. Clearing either genuinely means "not set" -- returning []
        there would invent a zero-layer sheet."""
        editor = self._editor(qt_app, default=[200, 200, 50])

        editor.set_value(None)

        assert editor.value() is None

    def test_a_none_default_still_reads_an_empty_box_as_none(self, qt_app):
        """`hpc.modules` is Optional[List[str]] -- None is a legal value."""
        editor = self._editor(qt_app, default=None)

        editor.set_value(None)

        assert editor.value() is None

    def test_a_populated_list_is_unaffected(self, qt_app):
        editor = self._editor(qt_app, default=[])

        editor.set_value(['gap_min', 'deflect_max'])

        assert editor.value() == ['gap_min', 'deflect_max']
