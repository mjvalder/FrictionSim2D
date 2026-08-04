"""The material sweep: reachable at all, then discoverable.

A sweep config is *supposed* to carry {mat} placeholders -- expansion
substitutes them. preflight_paths says so explicitly ("not user error"), but
the GUI treated every issue as fatal, so Render and Generate refused a
correctly configured sweep.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.controllers.document import ConfigDocument  # noqa: E402
from src.gui.workspace import Workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / 'examples'


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(name='materials_file')
def _materials_file(tmp_path):
    path = tmp_path / 'mats.txt'
    path.write_text('h-MoS2\nt-MoS2\n', encoding='utf-8')
    return path


@pytest.fixture(name='swept')
def _swept(materials_file, monkeypatch):
    """An indent config with a resolving two-material sweep."""
    monkeypatch.chdir(REPO_ROOT)
    document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
    document.set('2D', 'materials_list', str(materials_file))
    document.set('2D', 'mat', '{mat}')
    document.set('2D', 'cif_path', 'examples/materials/{mat}.cif')
    return document


@pytest.fixture(name='dirs')
def _dirs(tmp_path):
    """A structure directory and a potential directory: the intersection is
    three materials, plus Si (structure only) and h-WS2 (potential only) --
    exactly the shape of the real run_afm/ directory. Module-scoped so every
    test class below can share it rather than each re-declaring its own."""
    structures = tmp_path / 'cif'
    potentials = tmp_path / 'sw_lammps'
    structures.mkdir()
    potentials.mkdir()
    for name in ('h-MoS2', 't-MoS2', 'b-AlSb'):
        (structures / f"{name}.cif").write_text('', encoding='utf-8')
        (potentials / f"{name}.sw").write_text('', encoding='utf-8')
    # In cif/ only, exactly as Si is in the real run directory.
    (structures / 'Si.cif').write_text('', encoding='utf-8')
    # In sw_lammps/ only.
    (potentials / 'h-WS2.sw').write_text('', encoding='utf-8')
    return structures, potentials


def test_materials_row_is_first_in_2d_section():
    """The sweep is the first decision a [2D] section should ask -- it
    decides whether Structure/Potential below it are files or folders."""
    from src.gui.schema.overrides import field_override

    assert field_override('afm', '2D', 'materials_list').order == 0


def test_the_sweep_checkbox_carries_no_text_of_its_own(qt_app):
    """The row label ('Sweep over materials', from the override) already
    names the checkbox -- giving the checkbox its own text too ('Sweep over
    many materials') was the duplicated wording the user's screenshot
    flagged."""
    from src.gui.schema.introspect import FieldKind, FieldSpec
    from src.gui.widgets.materials import MaterialListEditor

    spec = FieldSpec(name='materials_list', key='materials_list',
                     kind=FieldKind.STR, required=False)
    editor = MaterialListEditor(spec)
    assert editor.sweep_box.text() == ''


def test_2d_section_row_order_is_checkbox_mat_list_then_paths(qt_app,
                                                              monkeypatch):
    """User's chosen layout: the checkbox governs the cells below it, so it
    stays first; the tick list (and its `.txt` picker) sit after `mat`, ahead
    of Structure/Potential type/Potential file."""
    from src.gui.controllers.document import ConfigDocument
    from src.gui.widgets.form import SectionForm

    monkeypatch.chdir(REPO_ROOT)
    document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
    form = SectionForm(document.section('2D'), document)

    material_editor = form.editors['materials_list']

    def _row(key):
        _label, cell = form._rows[key]
        row, _role = form.form.getWidgetPosition(cell)
        return row

    checkbox_row = _row('materials_list')
    mat_row = _row('mat')
    list_row, _role = form.form.getWidgetPosition(material_editor.list_panel)
    cif_row = _row('cif_path')
    pot_type_row = _row('pot_type')
    pot_path_row = _row('pot_path')

    assert -1 not in (checkbox_row, mat_row, list_row, cif_row, pot_type_row,
                      pot_path_row)
    assert (checkbox_row < mat_row < list_row < cif_row
            < pot_type_row < pot_path_row)


class TestMaterialsPathPicker:
    """The `.txt` picker: a normal editable path row (line edit + browse),
    bound to `materials_list`, letting the user point at an existing curated
    file instead of only ever configuring a sweep through two directories."""

    def test_the_path_row_is_browsable(self, qt_app, monkeypatch):
        """Mirrors TestPathEditorBrowse in test_gui_field_editors.py: a browse
        selection sets the text and raises editingFinished itself, since
        setText() alone would not."""
        from PySide6.QtWidgets import QFileDialog
        from src.gui.schema.introspect import FieldKind, FieldSpec
        from src.gui.widgets.materials import MaterialListEditor

        spec = FieldSpec(name='materials_list', key='materials_list',
                         kind=FieldKind.STR, required=False)
        editor = MaterialListEditor(spec)
        monkeypatch.setattr(QFileDialog, 'getOpenFileName',
                            lambda *a, **k: ('/some/curated_materials.txt', ''))

        fired = []
        editor.path_edit.editingFinished.connect(lambda: fired.append(1))
        editor._browse_list()

        assert fired
        assert editor.path_edit.text() == '/some/curated_materials.txt'

    def test_pointing_the_path_row_at_a_curated_file_loads_its_names(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """The headline behaviour: browsing (or typing and finishing) to an
        existing curated `.txt` loads its names as the selection, driven
        through SectionForm exactly as a real editingFinished would."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        structures, potentials = dirs
        curated = tmp_path / 'curated_materials.txt'
        curated.write_text('h-MoS2\nb-AlSb\n', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        document.apply_material_directories(structures, potentials, '.sw')

        form = ConfigForm(document)
        section = form.sections['2D']
        editor = section.editors['materials_list']
        editor.set_sweeping(True)
        editor.rescan(structures, potentials, '.sw')

        editor.path_edit.setText(str(curated))
        editor.path_edit.editingFinished.emit()

        assert set(editor.selected_names()) == {'h-MoS2', 'b-AlSb'}
        assert document.get('2D', 'materials_list') == str(curated)
        form.deleteLater()


def test_path_editor_directory_mode(qt_app):
    """set_directory_mode is what SectionForm flips as the sweep is ticked or
    unticked -- the widget only remembers the flag and browses accordingly.

    This only pins the primitive's own get/set round trip -- it would still
    pass if SectionForm never called it at all. See
    test_ticking_the_sweep_switches_structure_and_potential_to_folder_pickers
    below for the actual headline behaviour, driven through SectionForm."""
    from src.gui.schema.introspect import FieldKind, FieldSpec
    from src.gui.widgets.fields import build_editor

    ed = build_editor(FieldSpec(name='cif_path', key='cif_path',
                                kind=FieldKind.PATH, required=True))
    assert not ed._directory_mode
    ed.set_directory_mode(True)
    assert ed._directory_mode
    ed.set_directory_mode(False)
    assert not ed._directory_mode


def test_ticking_the_sweep_switches_structure_and_potential_to_folder_pickers(
        qt_app, monkeypatch, tmp_path):
    """Commit 3efbbea's headline behaviour: opening a config that actually
    has a materials_list must switch the [2D] section's cif_path/pot_path
    cells from single-file to directory pickers, and closing the sweep must
    switch them back. Driven through SectionForm.load(), which is what calls
    _apply_sweep_mode() -- exercising set_directory_mode() directly (as
    test_path_editor_directory_mode does) cannot catch _apply_sweep_mode's
    own loop being gutted. (It does *not*, on its own, catch load()'s call to
    _apply_sweep_mode() being skipped: ticking/unticking here both go through
    a real sweep_box.toggled transition, which independently re-applies
    folder mode via _on_sweep_box_toggled -- see
    test_reloading_an_already_ticked_sweep_still_reapplies_folder_mode for
    the isolated case where the checkbox does not change and only load()'s
    own call is left to do it.)"""
    from src.gui.controllers.document import ConfigDocument
    from src.gui.widgets.form import SectionForm

    monkeypatch.chdir(REPO_ROOT)
    listing = tmp_path / 'mats.txt'
    listing.write_text('h-MoS2\n', encoding='utf-8')

    document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
    document.set('2D', 'materials_list', str(listing))

    form = SectionForm(document.section('2D'), document)
    form.load()

    assert form.editors['cif_path']._directory_mode is True
    assert form.editors['pot_path']._directory_mode is True

    document.set('2D', 'materials_list', None)
    form.load()

    assert form.editors['cif_path']._directory_mode is False
    assert form.editors['pot_path']._directory_mode is False


def test_reloading_an_already_ticked_sweep_still_reapplies_folder_mode(
        qt_app, monkeypatch, tmp_path):
    """load() calls _apply_sweep_mode() itself, not only in reaction to the
    sweep checkbox's own toggled signal -- reloading a config whose sweep is
    already ticked (re-opening the same file, or loading a second config that
    is also swept) never re-ticks the checkbox, so `toggled` does not fire a
    second time and _on_sweep_box_toggled's own call to _apply_sweep_mode()
    is never reached. Only load()'s own explicit call can be what puts the
    path editors back into folder mode here."""
    from src.gui.controllers.document import ConfigDocument
    from src.gui.widgets.form import SectionForm

    monkeypatch.chdir(REPO_ROOT)
    listing = tmp_path / 'mats.txt'
    listing.write_text('h-MoS2\n', encoding='utf-8')

    document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
    document.set('2D', 'materials_list', str(listing))

    form = SectionForm(document.section('2D'), document)
    form.load()
    assert form.editors['cif_path']._directory_mode is True   # sanity

    # Simulate folder mode having been lost some other way, with the sweep
    # box left ticked -- so the next load() has no state change to toggle on.
    form.editors['cif_path']._directory_mode = False
    form.editors['pot_path']._directory_mode = False

    form.load()

    assert form.editors['cif_path']._directory_mode is True
    assert form.editors['pot_path']._directory_mode is True


class TestSweepIsReachable:

    def test_the_sweep_expands(self, swept):
        assert swept.expansion().configs == 2

    def test_a_templated_sweep_config_validates(self, swept):
        """The regression: this returned ok=False on the {mat} placeholder
        alone, which blocked both Render and Generate."""
        result = swept.validate()
        assert result.ok, result.render()

    def test_the_pending_substitutions_are_still_reported(self, swept):
        swept.validate()
        keys = {i.key for i in swept.pending_template_issues}
        assert 'cif_path' in keys

    def test_a_placeholder_with_no_list_is_still_an_error(self, swept):
        """Nothing will ever substitute it, so it is a genuine mistake."""
        del swept.values['2D']['materials_list']
        result = swept.validate()
        assert not result.ok
        assert any(i.kind == 'unexpanded_template' for i in result.issues)


class TestWindowIntegration:
    """MainWindow._validate() surfaces pending substitutions without blocking."""

    @pytest.fixture(name='window')
    def _window(self, qt_app, tmp_path, monkeypatch):
        """Create a MainWindow for testing."""
        monkeypatch.chdir(REPO_ROOT)
        workspace = Workspace.open(tmp_path / 'ws')
        from src.gui.app import MainWindow  # noqa: PLC0415
        window = MainWindow(workspace)
        yield window
        # closeEvent (Task 12) prompts to save when the document is dirty --
        # a real, blocking QMessageBox with no one to click it under these
        # tests. Cleanup is not what is under test here, so close quietly.
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()

    def test_valid_sweep_validates_and_reports_pending(self, window, materials_file):
        """A correctly configured sweep: _validate() returns True, issues pane
        mentions the pending substitution."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.document.set('2D', 'materials_list', str(materials_file))
        window.document.set('2D', 'mat', '{mat}')
        window.document.set('2D', 'cif_path', 'examples/materials/{mat}.cif')
        window.form.document = window.document
        window.form.rebuild()

        # _validate() must return True for a valid sweep
        assert window._validate()

        # The issues pane must mention the pending substitution
        issues_text = window.issues.toPlainText()
        assert '{mat}' in issues_text
        assert 'substituted' in issues_text

    def test_unresolvable_materials_list_does_not_raise(self, window, tmp_path, monkeypatch):
        """A materials_list naming materials whose CIF files do not exist:
        expand_config_sweeps will raise, but window._validate() must not crash
        and must still report something useful."""
        monkeypatch.chdir(REPO_ROOT)
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        bad_materials = tmp_path / 'bad_mats.txt'
        bad_materials.write_text('nonexistent1\nnonexistent2\n', encoding='utf-8')

        window.document.set('2D', 'materials_list', str(bad_materials))
        window.document.set('2D', 'mat', '{mat}')
        window.document.set('2D', 'cif_path', 'examples/materials/{mat}.cif')
        window.form.document = window.document
        window.form.rebuild()

        # _validate() must not raise an exception
        try:
            result = window._validate()
        except Exception as exc:  # pylint: disable=broad-except
            pytest.fail(f"_validate() raised {type(exc).__name__}: {exc}")

        # The issues pane should still mention pending substitutions
        issues_text = window.issues.toPlainText()
        assert '{mat}' in issues_text
        # When count fails, it falls back to a generic message
        assert 'substituted' in issues_text

    def test_a_missing_materials_list_is_reported(self, window, tmp_path, monkeypatch):
        """FINDING 2: a materials_list pointing at a file that does not exist
        must be surfaced to the user, not silently swallowed. ExpansionSummary
        already collects this class of message as a note; _validate() must
        show it."""
        monkeypatch.chdir(REPO_ROOT)
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        missing = tmp_path / 'typo_materials.txt'
        window.document.set('2D', 'materials_list', str(missing))
        window.document.set('2D', 'mat', '{mat}')
        window.document.set('2D', 'cif_path', 'examples/materials/{mat}.cif')
        window.form.document = window.document
        window.form.rebuild()

        window._validate()
        issues_text = window.issues.toPlainText()
        assert 'typo_materials.txt' in issues_text


class TestScanningMaterialDirectories:
    """The material list is the intersection of a structure directory and a
    potential directory -- exactly what preflight_materials computes at build
    time, surfaced up front instead."""

    def test_the_usable_materials_are_the_intersection(self, dirs):
        from src.gui.controllers.materials import scan_material_directories

        found = scan_material_directories(*dirs)
        assert [c.name for c in found if c.usable] == ['b-AlSb', 'h-MoS2', 't-MoS2']

    def test_a_structure_with_no_potential_is_reported_not_dropped(self, dirs):
        """Silently omitting it is how 'why is Si missing?' becomes a mystery."""
        from src.gui.controllers.materials import scan_material_directories

        si = next(c for c in scan_material_directories(*dirs) if c.name == 'Si')
        assert not si.usable
        assert 'potential' in si.reason

    def test_a_potential_with_no_structure_is_reported(self, dirs):
        from src.gui.controllers.materials import scan_material_directories

        ws2 = next(c for c in scan_material_directories(*dirs) if c.name == 'h-WS2')
        assert not ws2.usable
        assert 'structure' in ws2.reason

    def test_the_potential_extension_is_inferred(self, dirs):
        from src.gui.controllers.materials import infer_potential_extension

        assert infer_potential_extension(dirs[1]) == '.sw'

    def test_a_mixed_potential_directory_takes_the_commonest_extension(self, tmp_path):
        from src.gui.controllers.materials import infer_potential_extension

        directory = tmp_path / 'pots'
        directory.mkdir()
        for name in ('a.sw', 'b.sw', 'c.rebomos'):
            (directory / name).write_text('', encoding='utf-8')
        assert infer_potential_extension(directory) == '.sw'

    def test_a_missing_directory_scans_empty_rather_than_raising(self, tmp_path):
        """The pickers are half-filled while the user is still choosing."""
        from src.gui.controllers.materials import scan_material_directories

        assert scan_material_directories(tmp_path / 'nope', tmp_path / 'also') == []


class TestApplyingMaterialDirectories:

    @pytest.fixture(name='document')
    def _document(self, monkeypatch, tmp_path):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        return document

    def test_the_templates_are_written(self, document, tmp_path):
        document.apply_material_directories(
            tmp_path / 'cif', tmp_path / 'sw_lammps', '.sw')
        assert document.get('2D', 'mat') == '{mat}'
        assert document.get('2D', 'cif_path') == 'cif/{mat}.cif'
        assert document.get('2D', 'pot_path') == 'sw_lammps/{mat}.sw'

    def test_directories_are_stored_relative_to_the_config(self, document, tmp_path):
        """The existing configs use relative paths and travel with their run
        directory; absolutising them would break that."""
        document.apply_material_directories(
            tmp_path / 'cif', tmp_path / 'sw_lammps', '.sw')
        assert not document.get('2D', 'cif_path').startswith('/')

    def test_a_directory_outside_the_config_tree_stays_absolute(self, document):
        document.apply_material_directories(
            Path('/opt/shared/cif'), Path('/opt/shared/pots'), '.sw')
        assert document.get('2D', 'cif_path') == '/opt/shared/cif/{mat}.cif'

    def test_the_directories_can_be_read_back(self, document, tmp_path):
        """Opening a config has to repopulate the pickers, or every open looks
        like nothing is configured."""
        document.apply_material_directories(
            tmp_path / 'cif', tmp_path / 'sw_lammps', '.sw')
        structures, potentials, extension = document.material_directories()
        assert structures.name == 'cif'
        assert potentials.name == 'sw_lammps'
        assert extension == '.sw'

    def test_reading_back_an_untemplated_config_gives_nothing(self, document):
        assert document.material_directories() == (None, None, None)


class TestMaterialListEditor:

    @pytest.fixture(name='editor')
    def _editor(self, qt_app, dirs):
        from src.gui.schema.introspect import FieldKind, FieldSpec
        from src.gui.widgets.materials import MaterialListEditor

        spec = FieldSpec(name='materials_list', key='materials_list',
                         kind=FieldKind.STR, required=False)
        editor = MaterialListEditor(spec)
        editor.rescan(*dirs, '.sw')
        return editor

    def test_it_starts_not_sweeping(self, qt_app):
        from src.gui.schema.introspect import FieldKind, FieldSpec
        from src.gui.widgets.materials import MaterialListEditor

        spec = FieldSpec(name='materials_list', key='materials_list',
                         kind=FieldKind.STR, required=False)
        assert not MaterialListEditor(spec).is_sweeping()

    def test_setting_directories_lists_the_usable_materials(self, editor):
        assert editor.selected_names() == ['b-AlSb', 'h-MoS2', 't-MoS2']

    def test_an_unusable_material_is_shown_but_not_selected(self, editor):
        """Si is in cif/ with no potential -- the real case in run_afm/."""
        assert 'Si' in editor.listed_names()
        assert 'Si' not in editor.selected_names()

    def test_an_unusable_material_cannot_be_selected(self, editor):
        editor.select_all()
        assert 'Si' not in editor.selected_names()

    def test_deselecting_narrows_the_sweep(self, editor):
        editor.set_selected(['h-MoS2'])
        assert editor.selected_names() == ['h-MoS2']

    def test_the_summary_counts_usable_against_listed(self, editor):
        """3 of the 5 names across the two directories have both files."""
        assert '3 of 5' in editor.summary_text()

    def test_a_path_value_switches_it_into_sweeping(self, editor):
        editor.set_value('material_list.txt')
        assert editor.is_sweeping()

    def test_an_empty_value_leaves_it_single(self, editor):
        editor.set_value(None)
        assert not editor.is_sweeping()

    def test_deselecting_everything_survives_a_rescan(self, editor, dirs):
        """An empty selection is a choice, not an absence. Treating it as
        'select all' on the next scan silently re-widens the sweep."""
        editor.select_none()
        # Force a real rescan of the same directories rather than hitting the
        # unchanged-triple no-op: the guarantee under test is that an actual
        # rebuild of the listing still respects an explicitly emptied
        # selection, not merely that nothing happens when nothing changed
        # (that is test_rescan_is_noop_for_unchanged_triple's job). Poking
        # `_scanned` directly is what stands in for that rebuild: the
        # same-triple rescan this test originally drove through
        # `editor._rescan()` is no longer reachable through any real UI
        # trigger now that rescan() has the no-op guard -- editingFinished,
        # a browse selection, and ticking the box all only ever fire it with
        # a genuinely different triple.
        editor._scanned = None
        editor.rescan(*dirs, '.sw')
        assert editor.selected_names() == []

    def test_set_document_names_is_the_rescan_fallback(self, editor, dirs, tmp_path):
        """set_document_names records the floor rescan() falls back to when
        the widget's own selection has gone empty for a reason other than the
        user actually choosing that -- e.g. an intermediate scan of a
        directory that briefly does not resolve (a moved share, mid-browse).
        Unlike test_deselecting_everything_survives_a_rescan, the emptiness
        here comes from a bad scan, not from the user clicking None, so
        falling back to the document's own recorded list is the right call
        here (and would be the wrong one there)."""
        editor.set_selected(['h-MoS2'])          # touch the selection for real
        editor.set_document_names(['h-MoS2', 't-MoS2'])

        editor.rescan(tmp_path / 'nonexistent', dirs[1], '.sw')
        assert editor.selected_names() == []      # sanity: the bad scan empties it

        editor.rescan(*dirs, '.sw')
        assert editor.selected_names() == ['h-MoS2', 't-MoS2']

    def test_rescan_is_noop_for_unchanged_triple(self, editor, dirs):
        """Re-scanning the same (structures, potentials, extension) triple --
        as happens whenever editingFinished fires without the text actually
        changing -- must not rebuild the listing or re-emit `changed`."""
        listing_before = editor.listed_names()
        calls = []
        editor.changed.connect(lambda: calls.append(1))
        editor.rescan(*dirs, '.sw')
        assert editor.listed_names() == listing_before
        assert calls == []

    def test_reentrant_rescan_during_changed_is_refused(self, editor, dirs, tmp_path):
        """`_rescanning` must still be True for the whole time `changed` is
        out and its listeners are running, not just while the listing is
        being rebuilt -- otherwise a listener that calls back into rescan()
        would sail straight through the guard the moment it is released.

        The reentrant call below targets a directory triple that differs
        from whichever rescan is currently in flight, so the *separate*
        unchanged-triple no-op cannot be what refuses it -- only the
        `_rescanning` guard can.
        """
        structures, potentials = dirs
        other = tmp_path / 'other_structures'
        other.mkdir()
        (other / 'h-MoS2.cif').write_text('', encoding='utf-8')

        calls = []

        def _reenter():
            calls.append(1)
            editor.rescan(structures, potentials, '.sw')

        editor.changed.connect(_reenter)
        editor.rescan(other, potentials, '.sw')

        # The listener ran exactly once: the reentrant call it made was
        # refused outright rather than itself rebuilding the listing and
        # emitting `changed` again (which would have called _reenter a
        # second time).
        assert len(calls) == 1
        # And it left no trace: the in-flight rescan(other, ...) is still
        # what the widget reports, not the reentrant call's triple.
        assert editor.directories() == (other, potentials, '.sw')


class TestSweepRoundTrip:
    """Opening a configured sweep must show what it actually runs."""

    def test_an_existing_list_becomes_the_selection(self, qt_app, dirs, tmp_path,
                                                    monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        structures, potentials = dirs
        listing = tmp_path / 'run_materials.txt'
        listing.write_text('h-MoS2\n', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        document.apply_material_directories(structures, potentials, '.sw')
        document.set('2D', 'materials_list', str(listing))

        form = ConfigForm(document)
        editor = form.sections['2D'].editors['materials_list']
        assert editor.is_sweeping()
        assert editor.selected_names() == ['h-MoS2']
        form.deleteLater()

    def test_committing_writes_the_list_and_the_templates(self, qt_app, dirs,
                                                          tmp_path, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'

        form = ConfigForm(document)
        section = form.sections['2D']
        editor = section.editors['materials_list']
        editor.set_sweeping(True)
        editor.rescan(*dirs, '.sw')
        section.commit()

        assert document.get('2D', 'mat') == '{mat}'
        assert '{mat}' in document.get('2D', 'cif_path')
        assert document.material_names() == ['b-AlSb', 'h-MoS2', 't-MoS2']
        assert document.expansion().configs == 3
        form.deleteLater()

    def test_unticking_removes_the_sweep(self, qt_app, dirs, tmp_path, monkeypatch):
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'

        form = ConfigForm(document)
        section = form.sections['2D']
        editor = section.editors['materials_list']
        editor.set_sweeping(True)
        editor.rescan(*dirs, '.sw')
        section.commit()

        editor.set_sweeping(False)
        section.commit()
        assert not document.has_material_list()
        form.deleteLater()

    def test_an_unscannable_sweep_is_not_destroyed(self, qt_app, tmp_path,
                                                   monkeypatch):
        """A config whose directories do not exist on this machine still has a
        real sweep. Committing must not delete it."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        listing = tmp_path / 'run_materials.txt'
        listing.write_text('h-MoS2\nt-MoS2\n', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        document.apply_material_directories(
            tmp_path / 'gone', tmp_path / 'also_gone', '.sw')
        document.set('2D', 'materials_list', str(listing))

        form = ConfigForm(document)
        form.sections['2D'].commit()
        assert document.has_material_list()
        assert document.material_names() == ['h-MoS2', 't-MoS2']
        form.deleteLater()

    def test_committing_an_unchanged_sweep_rewrites_nothing(self, qt_app, dirs,
                                                            tmp_path, monkeypatch):
        """Opening a config and pressing Render must not retarget the list the
        user maintains by hand."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        listing = tmp_path / 'material_list.txt'
        listing.write_text('h-MoS2\nt-MoS2\nb-AlSb\n', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        document.apply_material_directories(*dirs, '.sw')
        document.set('2D', 'materials_list', str(listing))

        form = ConfigForm(document)
        before = document.get('2D', 'materials_list')
        form.sections['2D'].commit()
        assert document.get('2D', 'materials_list') == before
        form.deleteLater()

    def test_a_half_resolving_scan_does_not_destroy_an_existing_sweep(
            self, qt_app, tmp_path, monkeypatch):
        """FINDING 1: one directory fails to resolve (an unmounted share, a
        moved run directory). scan_material_directories still returns names
        for whichever directory *does* resolve, all marked unusable -- so
        listed_names() is non-empty even though nothing here can actually be
        swept. That must not be read as 'the user emptied the sweep'."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        listing = tmp_path / 'run_materials.txt'
        listing.write_text('h-MoS2\nt-MoS2\n', encoding='utf-8')

        structures = tmp_path / 'cif'
        structures.mkdir()
        (structures / 'h-MoS2.cif').write_text('', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        document.apply_material_directories(
            structures, tmp_path / 'unmounted_potentials', '.sw')
        document.set('2D', 'materials_list', str(listing))

        form = ConfigForm(document)
        form.sections['2D'].commit()

        assert document.has_material_list()
        assert document.material_names() == ['h-MoS2', 't-MoS2']
        form.deleteLater()

    def test_an_unreadable_list_does_not_widen_into_everything(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FINDING 2: materials_list pointing at a missing (or empty) .txt
        must not be read as 'nothing decided yet, select everything usable'.
        A typo in the path must not become a full sweep on the next commit."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        document.apply_material_directories(*dirs, '.sw')
        document.set('2D', 'materials_list', str(tmp_path / 'typo_materials.txt'))

        form = ConfigForm(document)
        form.sections['2D'].commit()

        assert document.material_names() == []
        assert not (tmp_path / 'run_materials.txt').exists()
        form.deleteLater()

    def test_committing_a_sweep_stores_a_relative_materials_list(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FINDING 3: write_material_names() stores a relative path but
        returns the absolute target it wrote to disk. note_written must
        remember what the document actually holds, not that return value, or
        the generic editor loop's write of editor.value() puts the absolute
        path back."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'

        form = ConfigForm(document)
        section = form.sections['2D']
        editor = section.editors['materials_list']
        editor.set_sweeping(True)
        editor.rescan(*dirs, '.sw')
        section.commit()

        stored = document.get('2D', 'materials_list')
        assert stored is not None
        assert not Path(stored).is_absolute()
        form.deleteLater()

    def test_typing_into_mat_survives_commit_when_the_sweep_is_unconfigured(
            self, qt_app, tmp_path, monkeypatch):
        """FINDING 4: ticking the sweep box does not hide mat/cif_path/
        pot_path -- they stay visible and editable until the sweep actually
        has both directories resolved. Committing while the box is ticked but
        the pickers are still empty must not silently revert a same-commit
        edit to 'mat'."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'

        form = ConfigForm(document)
        section = form.sections['2D']
        material_editor = section.editors['materials_list']
        material_editor.set_sweeping(True)   # ticked, but no directories chosen

        mat_editor = section.editors['mat']
        mat_editor.set_value('h_MoS2')
        section.commit()

        assert document.get('2D', 'mat') == 'h_MoS2'
        form.deleteLater()

    def test_unticking_clears_the_mat_placeholders(self, qt_app, dirs, tmp_path,
                                                    monkeypatch):
        """FINDING 5a: unticking must do more than remove materials_list --
        mat/cif_path/pot_path are left holding {mat}, which makes the config
        invalid and forces the user to retype three fields by hand."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'

        form = ConfigForm(document)
        section = form.sections['2D']
        editor = section.editors['materials_list']
        editor.set_sweeping(True)
        editor.rescan(*dirs, '.sw')
        section.commit()
        assert document.get('2D', 'mat') == '{mat}'   # sanity: sweep applied

        editor.set_sweeping(False)
        section.commit()

        assert document.get('2D', 'mat') != '{mat}'
        assert '{mat}' not in (document.get('2D', 'cif_path') or '')
        assert '{mat}' not in (document.get('2D', 'pot_path') or '')
        form.deleteLater()

    def test_opening_a_templated_config_without_a_list_does_not_tick_the_box(
            self, qt_app, tmp_path, monkeypatch):
        """FINDING 5b: a {mat} config with no materials_list is unfinished,
        not swept. rescan() (formerly set_directories) used to force the
        checkbox on whenever the paths happened to be templated, regardless
        of whether a sweep was actually configured -- so opening such a
        config, then committing (e.g. by pressing Render) invented a
        materials_list nobody asked for."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        document.apply_material_directories(
            tmp_path / 'cif', tmp_path / 'sw_lammps', '.sw')
        assert not document.has_material_list()

        form = ConfigForm(document)
        editor = form.sections['2D'].editors['materials_list']
        assert not editor.is_sweeping()

        form.sections['2D'].commit()
        assert not document.has_material_list()
        # Not just "no sweep was invented" -- the placeholders committing put
        # there must survive too. Asserting only has_material_list() walks
        # straight past clear_material_directories() deleting them on every
        # unticked commit rather than only on a genuine untick.
        assert document.get('2D', 'mat') == '{mat}'
        assert document.get('2D', 'cif_path') is not None
        assert document.get('2D', 'pot_path') is not None
        form.deleteLater()

    def test_a_templated_config_without_a_list_keeps_its_placeholders(
            self, qt_app, monkeypatch):
        """sheet_config.ini ships with {mat} in all three fields and no
        materials_list. Clearing placeholders on any unticked commit -- rather
        than on an actual untick -- destroys it on the first keystroke."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(
            EXAMPLES / 'sheet_config.ini', 'sheetonsheet')
        before = {key: document.get('2D', key)
                  for key in ('mat', 'cif_path', 'pot_path')}
        assert not document.has_material_list()

        form = ConfigForm(document)
        form.sections['2D'].commit()
        form.sections['2D'].commit()      # the second one is where it died

        assert {key: document.get('2D', key)
                for key in ('mat', 'cif_path', 'pot_path')} == before
        form.deleteLater()

    def test_editingfinished_scan_then_commit_templates_the_paths(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """The Structure/Potential cells are folder pickers while sweeping:
        typing (or browsing to) a directory and finishing the edit rescans,
        and committing turns what was picked into the {mat} templates -- with
        no directory-row editors of the sweep's own left to drive this from."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        cif_dir, pot_dir = dirs

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'

        config_form = ConfigForm(document)
        form = config_form.sections['2D']
        form.editors['materials_list'].set_sweeping(True)
        form.editors['cif_path'].edit.setText(str(cif_dir))
        form.editors['cif_path'].edit.editingFinished.emit()
        form.editors['pot_path'].edit.setText(str(pot_dir))
        form.editors['pot_path'].edit.editingFinished.emit()
        form.commit()

        assert form.document.get('2D', 'cif_path').endswith('/{mat}.cif')
        assert form.document.has_material_list()
        config_form.deleteLater()

    def test_intermediate_empty_scan_keeps_curated_selection(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FINDING 1/2 combined at the rescan level: a bogus intermediate
        scan (a directory that briefly does not resolve, e.g. mid-re-browse)
        must not read as 'the user emptied the sweep' and overwrite a curated
        .txt with 'everything' or 'nothing' once the real directory is
        scanned. The document's own material list is the floor rescan()
        falls back to.

        Both `editor.rescan()` calls below fire `changed`, which -- wired
        through `_on_field_changed` -- commits the [2D] section after each
        one, so `document.material_names()` is not an independent oracle: a
        selection the bogus scan wiped would itself get committed (as
        `write_material_names([])`, dropping `materials_list` from the
        document entirely) and then read right back as '[] == []' against an
        equally-wiped selection. Pinning both sides to the literal curated
        names, and re-reading the .txt from disk, is what actually catches
        that -- see the regression this guards in the commit history."""
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        structures, potentials = dirs
        listing = tmp_path / 'run_materials.txt'
        listing.write_text('h-MoS2\nt-MoS2\n', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        document.apply_material_directories(structures, potentials, '.sw')
        document.set('2D', 'materials_list', str(listing))

        form = ConfigForm(document)
        editor = form.sections['2D'].editors['materials_list']

        editor.rescan(tmp_path / 'nonexistent', potentials, '.sw')
        editor.rescan(structures, potentials, '.sw')

        assert set(editor.selected_names()) == {'h-MoS2', 't-MoS2'}
        assert document.material_names() == ['h-MoS2', 't-MoS2']
        assert set(listing.read_text().split()) == {'h-MoS2', 't-MoS2'}
        form.deleteLater()
