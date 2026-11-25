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

from frictionsim2d.gui.controllers.document import ConfigDocument  # noqa: E402
from frictionsim2d.gui.workspace import Workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
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


@pytest.fixture(name='editor')
def _editor(qt_app, dirs):
    """A bare MaterialListEditor with `dirs` already scanned.

    Module-level rather than class-level: the pair table below is the same
    editor in another mode, and re-declaring the fixture per class is how the
    two would drift apart.
    """
    from frictionsim2d.gui.schema.introspect import FieldKind, FieldSpec
    from frictionsim2d.gui.widgets.materials import MaterialListEditor

    spec = FieldSpec(name='materials_list', key='materials_list',
                     kind=FieldKind.STR, required=False)
    editor = MaterialListEditor(spec)
    editor.rescan(*dirs, '.sw')
    return editor


def pair_mode(editor, pairing: bool = True) -> None:
    """Put `editor` into (or out of) pair mode the way production reaches it.

    `_pair_mode` is derived, deliberately: `set_pair_capable` re-computes it
    from the second-sheet flag AND the sweep tick, so a commit can never read
    a mode the document disagrees with -- see
    `MaterialListEditor.set_pair_capable` for the signal-ordering bug that
    made an independently-settable mode wrong.

    Calling `set_pair_mode` from a test therefore sets exactly the state
    `SectionForm` can no longer produce, and such a test would keep passing
    if the derivation were reverted -- which is the property it exists to
    protect. The two ingredients go in instead, and the widget derives.
    """
    editor.set_sweeping(True)
    editor.set_pair_capable(pairing)
    assert editor.is_pair_mode() is pairing


def test_materials_row_is_first_in_2d_section():
    """The sweep is the first decision a [2D] section should ask -- it
    decides whether Structure/Potential below it are files or folders."""
    from frictionsim2d.gui.schema.overrides import field_override

    assert field_override('afm', '2D', 'materials_list').order == 0


def test_the_sweep_checkbox_carries_no_text_of_its_own(qt_app):
    """The row label ('Sweep over materials', from the override) already
    names the checkbox -- giving the checkbox its own text too ('Sweep over
    many materials') was the duplicated wording the user's screenshot
    flagged."""
    from frictionsim2d.gui.schema.introspect import FieldKind, FieldSpec
    from frictionsim2d.gui.widgets.materials import MaterialListEditor

    spec = FieldSpec(name='materials_list', key='materials_list',
                     kind=FieldKind.STR, required=False)
    editor = MaterialListEditor(spec)
    assert editor.sweep_box.text() == ''


def test_2d_section_row_order_is_checkbox_mat_list_then_paths(qt_app,
                                                              monkeypatch):
    """User's chosen layout: the checkbox governs the cells below it, so it
    stays first; the tick list (and its `.txt` picker) sit after `mat`, ahead
    of Structure/Potential type/Potential file."""
    from frictionsim2d.gui.controllers.document import ConfigDocument
    from frictionsim2d.gui.widgets.form import SectionForm

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
        from frictionsim2d.gui.schema.introspect import FieldKind, FieldSpec
        from frictionsim2d.gui.widgets.materials import MaterialListEditor

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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

    def test_ticking_after_loading_a_curated_file_rewrites_that_file(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FINDING 1: write_material_names used to always retarget to
        `{stem}_materials.txt`, ignoring what `materials_list` already
        pointed at -- so browsing to a curated file and then changing a
        tick silently abandoned it (stale contents) in favour of a
        freshly created default file. Ticking must rewrite the SAME file
        the picker was pointed at.

        The picker is pointed at the curated file *before* any directory is
        scanned -- the realistic order for "I already have a list, let me
        point the GUI at it" -- so no stem-based default is ever created
        along the way for the fix to coincidentally paper over."""
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        structures, potentials = dirs
        curated = tmp_path / 'my_curated_materials.txt'
        curated.write_text('h-MoS2\nb-AlSb\nt-MoS2\n', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'

        form = ConfigForm(document)
        section = form.sections['2D']
        editor = section.editors['materials_list']
        editor.set_sweeping(True)

        editor.path_edit.setText(str(curated))
        editor.path_edit.editingFinished.emit()
        assert document.get('2D', 'materials_list') == str(curated)

        editor.rescan(structures, potentials, '.sw')
        assert set(editor.selected_names()) == {'h-MoS2', 'b-AlSb', 't-MoS2'}
        # The rescan's own auto-commit must not have retargeted materials_list
        # either: the curated names and the document's were already equal.
        assert document.get('2D', 'materials_list') == str(curated)

        # A real click, not a same-commit no-op: narrow the selection.
        editor.set_selected(['h-MoS2', 'b-AlSb'])
        section.commit()

        assert set(curated.read_text().split()) == {'h-MoS2', 'b-AlSb'}

        stored = document.get('2D', 'materials_list')
        resolved = Path(stored)
        if not resolved.is_absolute():
            resolved = document.base_dir / resolved
        assert resolved.resolve() == curated.resolve()

        default_target = tmp_path / 'run_materials.txt'
        assert not default_target.exists()
        form.deleteLater()

    def test_pointing_at_a_curated_file_before_any_scan_is_legible(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FINDING 2: before any structure/potential directory has been
        scanned, `self.listing` is necessarily empty, so ticking against it
        is a no-op -- but that must read as "loaded, not yet checked", not
        as a bare "0 of 0 usable" that looks like nothing happened. The
        orphan-scan guard must survive: nothing here is usable yet, so a
        commit still cannot destroy anything."""
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        structures, potentials = dirs
        curated = tmp_path / 'curated_materials.txt'
        curated.write_text('h-MoS2\nt-MoS2\n', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'

        form = ConfigForm(document)
        section = form.sections['2D']
        editor = section.editors['materials_list']
        editor.set_sweeping(True)

        editor.path_edit.setText(str(curated))
        editor.path_edit.editingFinished.emit()

        # Nothing has been scanned: the orphan-scan guard must still see
        # this as "nothing usable", not as a real (empty) selection.
        assert not editor.has_usable_names()
        assert '2' in editor.summary.text()
        assert 'choose' in editor.summary.text().lower()
        assert set(editor.listed_names()) == {'h-MoS2', 't-MoS2'}

        # Once the user actually picks both directories, the curated names
        # become the real, usable selection -- the `_document_names`
        # fallback this relies on was primed by load_names() above.
        editor.rescan(structures, potentials, '.sw')

        assert editor.has_usable_names()
        assert set(editor.selected_names()) == {'h-MoS2', 't-MoS2'}
        form.deleteLater()

    def test_the_picker_does_not_overwrite_a_pair_list_on_a_one_sheet_config(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FINAL REVIEW (critical): `load()` refuses to adopt *or* overwrite a
        pair list under a one-sheet config -- the picker path had the same two
        branches and no fallback, so it repointed `materials_list` at the
        curated file and left the tick list exactly as it was. The very next
        commit (any field edit, Render, Generate or Save) then compared a full
        selection against the `[]` a pair file reports through
        `material_names()`, called that a change, and rewrote both ordered
        pairs as two bare single names.

        The picker's own file is what has to survive here -- not the config's,
        which is untouched either way.
        """
        from frictionsim2d.gui.widgets.form import ConfigForm  # noqa: PLC0415

        monkeypatch.chdir(REPO_ROOT)
        structures, potentials = dirs
        curated = tmp_path / 'curated_pairs.txt'
        original = 'h-MoS2 + t-MoS2\nt-MoS2 + b-AlSb\n'
        curated.write_text(original, encoding='utf-8')

        own = tmp_path / 'run_materials.txt'
        own.write_text('h-MoS2\nt-MoS2\n', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        document.set('2D', 'materials_list', own.name)
        document.apply_material_directories(structures, potentials, '.sw')
        assert not document.has_second_material()

        form = ConfigForm(document)
        editor = form.sections['2D'].editors['materials_list']
        assert editor.is_sweeping() and not editor.is_pair_mode()
        assert set(editor.selected_names()) == {'h-MoS2', 't-MoS2'}

        editor.path_edit.setText(str(curated))
        editor.path_edit.editingFinished.emit()

        # Neither adopted (a one-sheet config cannot sweep ordered pairs)...
        assert editor.selected_names() == []
        # ...nor overwritten, on this commit or any later one.
        form.commit()
        assert curated.read_text(encoding='utf-8') == original
        # And the widget says why, rather than looking like nothing happened.
        assert 'pair' in editor.summary.text().lower()

        # One scan later is still too late to lose it: the cleared selection
        # is a deliberate empty one, so a rescan must not read it as "nobody
        # has decided yet" and widen it back to everything usable -- nor
        # restore the pre-picker names through the `_document_names` fallback.
        editor._scanned = None            # force a real rebuild of the listing
        editor.rescan(structures, potentials, '.sw')
        form.commit()

        assert editor.selected_names() == []
        assert curated.read_text(encoding='utf-8') == original
        form.deleteLater()

    def test_a_wrong_kind_list_pointed_at_before_any_scan_stays_unselected(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """The other realistic order -- "I already have a list, let me point
        the GUI at it, then pick the directories" -- reaches the same hazard
        one step later. Before any scan the tick list is empty, so clearing it
        ticks nothing off and the *only* thing standing between the curated
        pair file and the first rescan is the cleared selection being recorded
        as a deliberate one: otherwise the scan defaults to everything usable
        and the commit it emits writes three single names over the pairs."""
        from frictionsim2d.gui.widgets.form import ConfigForm  # noqa: PLC0415

        monkeypatch.chdir(REPO_ROOT)
        structures, potentials = dirs
        curated = tmp_path / 'curated_pairs.txt'
        original = 'h-MoS2 + t-MoS2\nt-MoS2 + b-AlSb\n'
        curated.write_text(original, encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'

        form = ConfigForm(document)
        editor = form.sections['2D'].editors['materials_list']
        editor.set_sweeping(True)

        editor.path_edit.setText(str(curated))
        editor.path_edit.editingFinished.emit()

        # Now the directories arrive, exactly as picking them would deliver
        # them -- and the rescan's own `changed` commits the section.
        editor.rescan(structures, potentials, '.sw')
        form.commit()

        assert editor.selected_names() == []
        assert curated.read_text(encoding='utf-8') == original
        form.deleteLater()

    def test_the_picker_does_not_overwrite_a_single_list_in_pair_mode(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """The mirror image: a two-sheet config in pair mode, pointed at a
        curated list of single names. `material_names()` reads those happily,
        so the picker took the `elif names:` branch and ticked them into the
        (hidden) tick list, leaving the pair table's row untouched -- and the
        next commit went down the pair branch and wrote that one pair over the
        three curated single materials."""
        from frictionsim2d.gui.widgets.form import ConfigForm  # noqa: PLC0415

        monkeypatch.chdir(REPO_ROOT)
        structures, potentials = dirs
        curated = tmp_path / 'curated_singles.txt'
        original = 'h-MoS2\nt-MoS2\nb-AlSb\n'
        curated.write_text(original, encoding='utf-8')

        own = tmp_path / 'run_materials.txt'
        own.write_text('h-MoS2 + t-MoS2\n', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        document.source_path = tmp_path / 'run.ini'
        document.add_second_material()
        document.set('2D', 'materials_list', own.name)
        document.apply_material_directories(structures, potentials, '.sw',
                                            pair_mode=True)

        form = ConfigForm(document)
        editor = form.sections['2D'].editors['materials_list']
        editor.rescan(structures, potentials, '.sw')
        assert editor.is_pair_mode()
        assert editor.selected_pairs() == [('h-MoS2', 't-MoS2')]

        editor.path_edit.setText(str(curated))
        editor.path_edit.editingFinished.emit()

        assert editor.selected_pairs() == []
        form.commit()
        assert curated.read_text(encoding='utf-8') == original
        assert 'pair' in editor.summary.text().lower()
        form.deleteLater()


def test_path_editor_directory_mode(qt_app):
    """set_directory_mode is what SectionForm flips as the sweep is ticked or
    unticked -- the widget only remembers the flag and browses accordingly.

    This only pins the primitive's own get/set round trip -- it would still
    pass if SectionForm never called it at all. See
    test_ticking_the_sweep_switches_structure_and_potential_to_folder_pickers
    below for the actual headline behaviour, driven through SectionForm."""
    from frictionsim2d.gui.schema.introspect import FieldKind, FieldSpec
    from frictionsim2d.gui.widgets.fields import build_editor

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
    from frictionsim2d.gui.controllers.document import ConfigDocument
    from frictionsim2d.gui.widgets.form import SectionForm

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
    from frictionsim2d.gui.controllers.document import ConfigDocument
    from frictionsim2d.gui.widgets.form import SectionForm

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
        from frictionsim2d.gui.app import MainWindow  # noqa: PLC0415
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
        # ...and, when the count fails, say why rather than promising a
        # substitution the refused expansion is never going to perform.
        assert str(bad_materials) in issues_text
        assert 'could be resolved' in issues_text
        assert 'will be substituted' not in issues_text

    def test_opening_a_config_does_not_carry_the_previous_sweep_over(
            self, window, dirs, tmp_path, monkeypatch):
        """FIX ROUND 1 (critical): the ticked-sweep carry-over that survives a
        rebuild must not survive a *document swap*. `_open_config` assigns the
        new document and then rebuilds, so a tick belonging to the config
        being closed was restored onto the config being opened -- and the
        `_validate()` on the very next line committed it, inventing a
        three-material sweep and a `.txt` beside a config the user had only
        just opened. `load()`'s own rule: a {mat} config with no
        materials_list is unfinished, not swept."""
        from PySide6.QtWidgets import QFileDialog  # noqa: PLC0415

        monkeypatch.chdir(REPO_ROOT)
        structures, potentials = dirs
        opened = ConfigDocument.blank('afm')
        opened.source_path = tmp_path / 'opened.ini'
        opened.apply_material_directories(structures, potentials, '.sw')
        opened.save(tmp_path / 'opened.ini')
        assert not opened.has_material_list()          # templated, never swept

        # The config on screen has its sweep ticked but nothing chosen yet --
        # the one state the document cannot express, and the whole reason the
        # tick is carried across a rebuild at all.
        window.form.sections['2D'].editors['materials_list'].set_sweeping(True)

        monkeypatch.setattr(QFileDialog, 'getOpenFileName',
                            lambda *a, **k: (str(tmp_path / 'opened.ini'), ''))
        window._open_config()

        editor = window.form.sections['2D'].editors['materials_list']
        assert not editor.is_sweeping()
        assert not window.document.has_material_list()
        assert not (tmp_path / 'opened_materials.txt').exists()

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

    def test_a_refused_expansion_is_reported_not_swallowed(
            self, window, dirs, tmp_path, monkeypatch):
        """FINAL REVIEW: the journey "open a swept single-material config,
        press Add a material" leaves a list of single names under a two-sheet
        config. `expand_config_sweeps` refuses that in one clear sentence, and
        Generate has reported it since it started calling
        `expansion_output_dirs()` -- but `_validate` caught the same exception
        into a `logger.debug` and said nothing, so Render and the issues panel
        showed a config that reads as fine.

        Worse than silence: the panel went on to promise that the four `{mat}`
        paths "will be substituted across each material in the sweep when you
        generate", which is exactly what the refusal says will not happen.
        """
        monkeypatch.chdir(REPO_ROOT)
        structures, potentials = dirs
        singles = tmp_path / 'c_materials.txt'
        singles.write_text('h-MoS2\nt-MoS2\n', encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini',
                                            'indent')
        document.source_path = tmp_path / 'c.ini'
        document.set('2D', 'materials_list', singles.name)
        document.apply_material_directories(structures, potentials, '.sw')
        document.add_second_material()

        window.document = document
        window.form.document = document
        window.form.rebuild()

        window._validate()
        issues_text = window.issues.toPlainText()

        # The expansion's own sentence, which names the line and what it
        # would take to fix it.
        assert 'h-MoS2' in issues_text
        assert '[2D-2]' in issues_text
        assert 'ordered pairs' in issues_text
        # And no promise that the placeholders are going to be filled in.
        assert 'will be substituted' not in issues_text
        assert document.pending_template_issues                # sanity


class TestScanningMaterialDirectories:
    """The material list is the intersection of a structure directory and a
    potential directory -- exactly what preflight_materials computes at build
    time, surfaced up front instead."""

    def test_the_usable_materials_are_the_intersection(self, dirs):
        from frictionsim2d.gui.controllers.materials import scan_material_directories

        found = scan_material_directories(*dirs)
        assert [c.name for c in found if c.usable] == ['b-AlSb', 'h-MoS2', 't-MoS2']

    def test_a_structure_with_no_potential_is_reported_not_dropped(self, dirs):
        """Silently omitting it is how 'why is Si missing?' becomes a mystery."""
        from frictionsim2d.gui.controllers.materials import scan_material_directories

        si = next(c for c in scan_material_directories(*dirs) if c.name == 'Si')
        assert not si.usable
        assert 'potential' in si.reason

    def test_a_potential_with_no_structure_is_reported(self, dirs):
        from frictionsim2d.gui.controllers.materials import scan_material_directories

        ws2 = next(c for c in scan_material_directories(*dirs) if c.name == 'h-WS2')
        assert not ws2.usable
        assert 'structure' in ws2.reason

    def test_the_potential_extension_is_inferred(self, dirs):
        from frictionsim2d.gui.controllers.materials import infer_potential_extension

        assert infer_potential_extension(dirs[1]) == '.sw'

    def test_a_mixed_potential_directory_takes_the_commonest_extension(self, tmp_path):
        from frictionsim2d.gui.controllers.materials import infer_potential_extension

        directory = tmp_path / 'pots'
        directory.mkdir()
        for name in ('a.sw', 'b.sw', 'c.rebomos'):
            (directory / name).write_text('', encoding='utf-8')
        assert infer_potential_extension(directory) == '.sw'

    def test_a_missing_directory_scans_empty_rather_than_raising(self, tmp_path):
        """The pickers are half-filled while the user is still choosing."""
        from frictionsim2d.gui.controllers.materials import scan_material_directories

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

    def test_it_starts_not_sweeping(self, qt_app):
        from frictionsim2d.gui.schema.introspect import FieldKind, FieldSpec
        from frictionsim2d.gui.widgets.materials import MaterialListEditor

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
        assert '3 of 5' in editor.summary.text()

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


class TestPairTable:
    """A pair sweep names an ORDERED pair per line, so its selection is a
    two-column table, not a tick list: which material is sheet 1 and which is
    sheet 2 is the whole content of a heterostructure, and a set of ticks
    cannot say it."""

    def test_pair_mode_swaps_the_list_for_the_table(self, editor):
        pair_mode(editor)
        assert editor.pair_table.isVisibleTo(editor.list_panel)
        assert not editor.listing.isVisibleTo(editor.list_panel)

        pair_mode(editor, False)
        assert not editor.pair_table.isVisibleTo(editor.list_panel)
        assert editor.listing.isVisibleTo(editor.list_panel)

    def test_generate_pairs_ordered(self, editor):
        """Both orders, because AB and BA are different stacks -- the
        interface is the same but which layer sits on the substrate is not."""
        pair_mode(editor)
        editor.set_selected(['h-MoS2', 't-MoS2'])
        editor.generate_pairs(include_reversed=True, include_same=False)

        assert set(editor.selected_pairs()) == {('h-MoS2', 't-MoS2'),
                                                ('t-MoS2', 'h-MoS2')}
        assert len(editor.selected_pairs()) == 2

    def test_generate_pairs_is_one_way_by_default(self, editor):
        editor.set_selected(['h-MoS2', 't-MoS2'])
        editor.generate_pairs()
        assert editor.selected_pairs() == [('h-MoS2', 't-MoS2')]

    def test_generate_pairs_can_include_same_material_pairs(self, editor):
        """A homostructure bilayer is a legitimate reference run, but it is
        not what 'pair up these materials' means by default."""
        editor.set_selected(['h-MoS2', 't-MoS2'])
        editor.generate_pairs(include_same=True)
        assert editor.selected_pairs() == [('h-MoS2', 'h-MoS2'),
                                           ('h-MoS2', 't-MoS2'),
                                           ('t-MoS2', 't-MoS2')]

    def test_an_unusable_material_is_never_paired(self, editor):
        """Si has a structure file and no potential -- it cannot be either
        member of a pair, exactly as it cannot be a single sweep member."""
        editor.select_all()
        editor.generate_pairs(include_reversed=True)
        assert editor.selected_pairs()
        assert all('Si' not in pair for pair in editor.selected_pairs())

    def test_pairs_round_trip_through_the_table(self, editor):
        pairs = [('h-MoS2', 't-MoS2'), ('t-MoS2', 'b-AlSb')]
        editor.set_selected_pairs(pairs)
        assert editor.selected_pairs() == pairs

    def test_a_pair_member_the_scan_cannot_see_is_kept(self, editor, dirs):
        """A curated list may name a material this machine's directories do
        not currently hold (h-WS2 has a potential and no structure here).
        Rebuilding the combos on a rescan must not quietly drop it -- that is
        the same silent narrowing the tick list's own keep-selection logic
        exists to prevent."""
        editor.set_selected_pairs([('h-MoS2', 'h-WS2')])
        assert editor.selected_pairs() == [('h-MoS2', 'h-WS2')]

        editor._scanned = None            # force a real rebuild of the listing
        editor.rescan(*dirs, '.sw')

        assert editor.selected_pairs() == [('h-MoS2', 'h-WS2')]

    def test_a_rescan_offers_the_new_materials_to_existing_rows(self, qt_app,
                                                                dirs):
        """A row built before any directory resolved can only offer what was
        known then. Scanning is what makes the rest available -- without the
        combos being rebuilt, choosing the second member of a pair the user
        already started would be impossible."""
        from frictionsim2d.gui.schema.introspect import FieldKind, FieldSpec
        from frictionsim2d.gui.widgets.materials import MaterialListEditor

        unscanned = MaterialListEditor(
            FieldSpec(name='materials_list', key='materials_list',
                      kind=FieldKind.STR, required=False))
        unscanned.set_selected_pairs([('h-MoS2', 'h-MoS2')])
        assert unscanned.pair_table.cellWidget(0, 1).count() == 1

        unscanned.rescan(*dirs, '.sw')

        combo = unscanned.pair_table.cellWidget(0, 1)
        offered = {combo.itemText(i) for i in range(combo.count())}
        assert {'b-AlSb', 'h-MoS2', 't-MoS2'} <= offered
        assert unscanned.selected_pairs() == [('h-MoS2', 'h-MoS2')]

    def test_generate_pairs_falls_back_to_everything_usable(self, editor):
        """The tick list is hidden in pair mode, so an empty selection there
        is far more likely to mean 'not used' than 'deliberately none' -- and
        a Generate that produced nothing would just look broken."""
        editor.select_none()
        editor.generate_pairs()
        assert set(editor.selected_pairs()) == {
            ('b-AlSb', 'h-MoS2'), ('b-AlSb', 't-MoS2'), ('h-MoS2', 't-MoS2')}

    def test_the_generate_dialog_passes_its_checkboxes_through(self, editor,
                                                                monkeypatch):
        """The dialog is the only way a user reaches `generate_pairs`, and its
        two checkboxes are the only thing it contributes -- wired to the wrong
        arguments it silently generates the wrong sweep."""
        from PySide6.QtWidgets import QCheckBox, QDialog

        pair_mode(editor)
        editor.set_selected(['h-MoS2', 't-MoS2'])

        def _accept(dialog):
            for box in dialog.findChildren(QCheckBox):
                if 'reversed' in box.text():
                    box.setChecked(True)
            return QDialog.Accepted

        monkeypatch.setattr(QDialog, 'exec', _accept)
        editor._open_generate_dialog()

        assert set(editor.selected_pairs()) == {('h-MoS2', 't-MoS2'),
                                                ('t-MoS2', 'h-MoS2')}

    def test_adding_a_pair_appends_a_row(self, editor):
        editor.set_selected_pairs([('h-MoS2', 't-MoS2')])
        editor.add_pair()
        assert len(editor.selected_pairs()) == 2

    def test_removing_a_pair_drops_the_selected_row(self, editor):
        editor.set_selected_pairs([('h-MoS2', 't-MoS2'), ('t-MoS2', 'b-AlSb')])
        editor.pair_table.setCurrentCell(0, 0)
        editor.remove_pair()
        assert editor.selected_pairs() == [('t-MoS2', 'b-AlSb')]

    def test_each_mode_keeps_its_own_selection(self, editor):
        """Toggling the second material on and off flips the mode; neither
        half of the widget may silently lose what was chosen in it."""
        editor.set_selected(['h-MoS2'])
        editor.set_selected_pairs([('h-MoS2', 't-MoS2')])

        pair_mode(editor)
        pair_mode(editor, False)

        assert editor.selected_names() == ['h-MoS2']
        assert editor.selected_pairs() == [('h-MoS2', 't-MoS2')]

    def test_the_summary_counts_pairs_while_pairing(self, editor):
        pair_mode(editor)
        editor.set_selected_pairs([('h-MoS2', 't-MoS2'), ('t-MoS2', 'b-AlSb')])
        assert '2 pair' in editor.summary.text()


class TestPairSweepCommit:
    """Committing a pair sweep: 'mat1 + mat2' lines, and a placeholder on each
    sheet. Both halves or neither -- a pair list with an untemplated second
    sheet builds the same second material for every run, inside directories
    named after the pairs that were asked for."""

    @staticmethod
    def _pair_form(tmp_path, monkeypatch, dirs=None, pairs=None, listing=None):
        """A [2D]+[2D-2] indent config, its form built and the sweep ticked."""
        from frictionsim2d.gui.widgets.form import ConfigForm  # noqa: PLC0415

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini',
                                            'indent')
        document.source_path = tmp_path / 'c.ini'
        document.add_second_material()
        if listing is not None:
            document.set('2D', 'materials_list', str(listing))

        form = ConfigForm(document)
        editor = form.sections['2D'].editors['materials_list']
        editor.set_sweeping(True)
        if dirs is not None:
            editor.rescan(*dirs, '.sw')
        if pairs is not None:
            editor.set_selected_pairs(pairs)
        return form, editor

    def test_pair_commit_writes_pair_lines_and_both_templates(
            self, qt_app, dirs, tmp_path, monkeypatch):
        form, _editor = self._pair_form(
            tmp_path, monkeypatch, dirs=dirs,
            pairs=[('h-MoS2', 'h-WS2'), ('t-MoS2', 'b-AlSb')])
        form.commit()

        txt = (tmp_path / 'c_materials.txt').read_text(encoding='utf-8')
        assert 'h-MoS2 + h-WS2' in txt
        assert 't-MoS2 + b-AlSb' in txt
        assert form.document.get('2D', 'cif_path').endswith('/{mat1}.cif')
        assert form.document.get('2D-2', 'cif_path').endswith('/{mat2}.cif')
        assert form.document.get('2D-2', 'mat') == '{mat2}'
        assert form.document.material_entries() == [('h-MoS2', 'h-WS2'),
                                                    ('t-MoS2', 'b-AlSb')]
        form.deleteLater()

    def test_the_pair_table_appears_only_with_a_second_material(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """Pair mode is sweeping AND a second sheet -- neither alone."""
        from frictionsim2d.gui.widgets.form import ConfigForm  # noqa: PLC0415

        monkeypatch.chdir(REPO_ROOT)
        listing = tmp_path / 'singles.txt'
        listing.write_text('h-MoS2\n', encoding='utf-8')
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini',
                                            'indent')
        document.source_path = tmp_path / 'c.ini'
        document.set('2D', 'materials_list', str(listing))

        form = ConfigForm(document)
        editor = form.sections['2D'].editors['materials_list']
        assert editor.is_sweeping()
        assert not editor.is_pair_mode()

        document.add_second_material()
        form.rebuild()
        editor = form.sections['2D'].editors['materials_list']
        assert editor.is_sweeping()
        assert editor.is_pair_mode()
        form.deleteLater()

    def test_adding_a_material_while_sweeping_keeps_the_sweep_ticked(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """Ticking the box before choosing anything writes nothing to the
        document, so the rebuild that 'Add a material' triggers had nothing to
        restore the tick from: the sweep silently switched itself off, and the
        two directories the user then picked were saved as literal paths.

        The natural order for building a pair -- sweep, then second material,
        then the directories -- has to survive the rebuild."""
        from frictionsim2d.gui.widgets.form import ConfigForm  # noqa: PLC0415

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini',
                                            'indent')
        document.source_path = tmp_path / 'c.ini'

        form = ConfigForm(document)
        form.sections['2D'].editors['materials_list'].set_sweeping(True)
        form.sections['2D'].add_material_button.click()
        qt_app.processEvents()            # the rebuild is deferred by a timer

        editor = form.sections['2D'].editors['materials_list']
        assert editor.is_sweeping()
        assert editor.is_pair_mode()
        assert form.sections['2D'].editors['cif_path']._directory_mode is True
        form.deleteLater()

    def test_the_second_sheets_paths_follow_the_sweep_into_folder_mode(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """[2D-2]'s Structure/Potential mean a directory while the sweep is
        writing them, exactly as [2D]'s do -- a single-file picker there
        offers to overwrite the sweep with one material's file."""
        form, _editor = self._pair_form(tmp_path, monkeypatch, dirs=dirs)
        second = form.sections['2D-2']
        assert second.editors['cif_path']._directory_mode is True
        assert second.editors['pot_path']._directory_mode is True
        form.deleteLater()

    def test_the_second_sheets_paths_are_greyed_while_the_sweep_writes_them(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FINAL REVIEW: both sheets of a pair sweep share ONE structure and
        ONE potential directory -- `_write_material_templates` builds
        `[2D-2]`'s paths out of the folders `[2D]` holds, on every commit. A
        live folder picker on `[2D-2]` therefore accepts a directory that the
        next commit silently reverts, usually one triggered by an unrelated
        keystroke. Greyed, with the reason in the control, says where the
        choice actually lives.

        `[2D]`'s own pickers must stay live -- that IS the choice -- and
        unticking the sweep hands `[2D-2]`'s back, since without a sweep each
        sheet names its own file again."""
        form, editor = self._pair_form(tmp_path, monkeypatch, dirs=dirs)
        second = form.sections['2D-2']

        assert not second.editors['cif_path'].isEnabled()
        assert not second.editors['pot_path'].isEnabled()
        assert 'first material' in \
            second.editors['cif_path'].edit.placeholderText()
        assert form.sections['2D'].editors['cif_path'].isEnabled()

        editor.sweep_box.setChecked(False)

        assert second.editors['cif_path'].isEnabled()
        assert second.editors['pot_path'].isEnabled()
        assert 'first material' not in \
            second.editors['cif_path'].edit.placeholderText()
        form.deleteLater()

    def test_opening_a_pair_sweep_shows_its_pairs(self, qt_app, dirs, tmp_path,
                                                  monkeypatch):
        listing = tmp_path / 'pairs.txt'
        listing.write_text('h-MoS2 + t-MoS2\nt-MoS2 + b-AlSb\n',
                           encoding='utf-8')
        form, editor = self._pair_form(tmp_path, monkeypatch, listing=listing)

        assert editor.is_pair_mode()
        assert editor.selected_pairs() == [('h-MoS2', 't-MoS2'),
                                           ('t-MoS2', 'b-AlSb')]
        form.deleteLater()

    def test_pointing_the_picker_at_a_curated_pair_list_loads_its_pairs(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """The pair analogue of the curated-`.txt` path row: `material_names`
        reports nothing at all for a pair list, so loading through it would
        leave the table empty and the sweep looking unconfigured."""
        curated = tmp_path / 'curated_pairs.txt'
        curated.write_text('h-MoS2 + h-WS2\nt-MoS2 + b-AlSb\n', encoding='utf-8')
        form, editor = self._pair_form(tmp_path, monkeypatch, dirs=dirs)

        editor.path_edit.setText(str(curated))
        editor.path_edit.editingFinished.emit()

        assert editor.selected_pairs() == [('h-MoS2', 'h-WS2'),
                                           ('t-MoS2', 'b-AlSb')]
        assert form.document.get('2D', 'materials_list') == str(curated)
        form.deleteLater()

    def test_an_empty_pair_table_does_not_delete_the_list(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """Pressing 'Add a material' on a live single-material sweep lands
        here on the very next commit: pair mode is on and no pair has been
        chosen yet. An empty table there means 'not paired up yet', and the
        list the user is in the middle of pairing up must survive it."""
        listing = tmp_path / 'singles.txt'
        listing.write_text('h-MoS2\nt-MoS2\n', encoding='utf-8')
        form, editor = self._pair_form(tmp_path, monkeypatch, dirs=dirs,
                                       listing=listing)
        assert editor.selected_pairs() == []

        form.commit()

        assert form.document.has_material_list()
        assert set(listing.read_text(encoding='utf-8').split()) == {'h-MoS2',
                                                                    't-MoS2'}
        form.deleteLater()

    def test_a_pair_list_on_a_single_sheet_config_is_not_overwritten(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FIX ROUND 1 (important): a pair list configured on a config with NO
        [2D-2] -- hand-edited, or written by a build predating the pair
        support -- is exactly the stale population the collapse feature exists
        for, and opening it must not destroy it.

        `load()` adopting the pairs regardless of the document's shape skipped
        the empty-selection fallback, leaving the tick list at the scan's
        'everything usable' default while the commit ran down the
        single-material branch: the curated file came back as three unrelated
        single materials on the first commit after opening."""
        from frictionsim2d.gui.widgets.form import ConfigForm  # noqa: PLC0415

        monkeypatch.chdir(REPO_ROOT)
        curated = tmp_path / 'curated_pairs.txt'
        original = 'h-MoS2 + t-MoS2\nt-MoS2 + b-AlSb\n'
        curated.write_text(original, encoding='utf-8')

        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini',
                                            'indent')
        document.source_path = tmp_path / 'c.ini'
        document.apply_material_directories(*dirs, '.sw')
        document.set('2D', 'materials_list', str(curated))
        assert not document.has_second_material()

        form = ConfigForm(document)
        form.commit()

        assert curated.read_text(encoding='utf-8') == original
        assert document.material_entries() == [('h-MoS2', 't-MoS2'),
                                               ('t-MoS2', 'b-AlSb')]
        form.deleteLater()

    def test_emptying_a_table_that_held_pairs_clears_the_sweep(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """The other side of the coin: removing every row of a pair sweep is
        the same deliberate act as unticking every material, and clears it."""
        form, editor = self._pair_form(
            tmp_path, monkeypatch, dirs=dirs, pairs=[('h-MoS2', 't-MoS2')])
        form.commit()
        assert form.document.has_material_list()

        editor.set_selected_pairs([])
        form.commit()

        assert not form.document.has_material_list()
        form.deleteLater()

    def test_unticking_and_reticking_a_pair_sweep_commits_again(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FIX ROUND 2: `MaterialListEditor` connects its own `toggled`
        handler -- which emits `changed`, and so commits -- in its
        constructor, before `SectionForm._wire_material_sweep` connects the
        one that re-applies the sweep mode. Qt delivers in connection order,
        so the commit fired by a re-tick ran with `is_pair_mode()` still False
        from the untick while the document still held `[2D-2]`: the commit
        raised `DocumentError` at the signal boundary and aborted, writing
        neither the templates nor the list, and the next unrelated commit
        healed it. An intermittent silent no-op.

        Driven through the checkbox itself, and asserting the commit actually
        happened -- "no exception escaped" is what the old behaviour looked
        like from the outside too."""
        form, editor = self._pair_form(
            tmp_path, monkeypatch, dirs=dirs, pairs=[('h-MoS2', 't-MoS2')])
        form.commit()
        assert form.document.has_material_list()

        editor.sweep_box.setChecked(False)
        assert not form.document.has_material_list()      # the untick cleared it
        editor.sweep_box.setChecked(True)

        assert editor.is_pair_mode()
        assert form.document.get('2D', 'mat') == '{mat1}'
        assert form.document.get('2D-2', 'mat') == '{mat2}'
        assert form.document.material_entries() == [('h-MoS2', 't-MoS2')]
        form.deleteLater()

    def test_a_refused_commit_is_reported_rather_than_thrown(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FIX ROUND 2: a `DocumentError` reaching a Qt slot is invisible --
        `app.py` installs no excepthook, so it prints a traceback to stderr
        and the edit silently does nothing. The document keeps refusing
        loudly; it is the UI that has to say so."""
        form, editor = self._pair_form(
            tmp_path, monkeypatch, dirs=dirs, pairs=[('h-MoS2', 't-MoS2')])
        announced = []
        form.refused.connect(announced.append)

        # Widget and document disagree, which is what `pair_mode` exists to
        # catch -- here it is reached through the slots Qt itself calls.
        del form.document.values['2D-2']

        form.sections['2D']._on_field_changed()
        form.commit()

        # Both slots reported rather than threw. (The orphaned `[2D-2]` form
        # left behind by the teardown above refuses its own commit too --
        # 'Unknown section' -- and that is reported by the same guard.)
        #
        # Note what this does NOT claim: `_on_field_changed` and
        # `_commit_section` are guarded, but `_on_material_button_clicked`
        # calls `self.commit()` outside any try, so a `DocumentError` there
        # still escapes into the clicked() slot. Pre-existing and untouched
        # by this round.
        assert sum('pair_mode' in text for text in announced) >= 2
        # `commit()` is the path Render/Generate/Save take, so what it
        # refused has to still be readable once the signal has been and gone:
        # that is what the window folds into the issues panel.
        assert any('pair_mode' in text for text in form.refusals())
        form.deleteLater()

    @staticmethod
    def _read_only(path):
        """`path`, unwritable, restored however the test leaves."""
        import contextlib  # noqa: PLC0415

        @contextlib.contextmanager
        def guard():
            path.chmod(0o500)
            try:
                yield
            finally:
                path.chmod(0o700)
        return guard()

    def test_a_write_the_filesystem_refuses_is_reported_not_thrown(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """The material list is the one part of an edit that reaches disk while
        it is being made, so it is the one part the filesystem can refuse --
        a read-only config directory, a share that dropped, a full disk.
        `write_material_names` raises OSError, and neither slot that commits
        caught anything but `DocumentError`: the tick printed a traceback to
        stderr from inside a Qt slot and the user saw nothing at all.

        A read-only *directory*, deliberately: the write goes to a sibling
        temp file and is renamed into place (`_write_atomically`), so it needs
        permission to create a file in the directory, where the in-place
        truncation it replaced only needed the list file itself writable.
        """
        if os.geteuid() == 0:
            pytest.skip('root writes through the directory mode this relies on')

        form, _editor = self._pair_form(
            tmp_path, monkeypatch, dirs=dirs, pairs=[('h-MoS2', 't-MoS2')])
        announced = []
        form.refused.connect(announced.append)

        with self._read_only(tmp_path):
            # The per-keystroke slot: this is the one where the refusal was
            # completely invisible, since nothing downstream of it reports.
            form.sections['2D']._on_field_changed()
            assert announced, 'a refused write said nothing at all'

            # And the path Render, Generate and Save take, whose refusals the
            # window folds into the issues panel.
            form.commit()

        assert any('material list' in text for text in form.refusals())
        assert any('directory' in text for text in form.refusals()), \
            'the message does not say which permission is missing'
        assert not (tmp_path / 'c_materials.txt').exists()
        form.deleteLater()

    def test_a_widget_and_document_that_disagree_are_refused(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """FIX ROUND 1: what `pair_mode=` is for. The commit states the mode
        the *widget* is in -- a pair table on screen -- and the document
        refuses it when its own shape says otherwise. Without a real second
        sheet the pairs would be written as 'a + b' lines under a `{mat}`
        single-material template: a config `run.expand` refuses much later, with
        the GUI insisting it is fine."""
        from frictionsim2d.gui.controllers.document import DocumentError  # noqa: PLC0415

        form, editor = self._pair_form(
            tmp_path, monkeypatch, dirs=dirs, pairs=[('h-MoS2', 't-MoS2')])
        assert editor.is_pair_mode()

        # The second sheet goes away without the form being rebuilt -- the
        # shape the widget still believes in is now a lie.
        del form.document.values['2D-2']

        with pytest.raises(DocumentError, match='pair_mode'):
            form.sections['2D'].commit()
        form.deleteLater()

    def test_repointing_the_directories_survives_the_second_sheets_commit(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """[2D]'s commit writes [2D-2]'s placeholders straight into the
        document, bypassing its editors. Committing [2D-2] afterwards without
        resyncing them writes the stale text back over what the sweep just
        decided -- the second sheet silently reverts to the old directory (or,
        on a freshly added sheet, to nothing at all)."""
        form, editor = self._pair_form(
            tmp_path, monkeypatch, dirs=dirs, pairs=[('h-MoS2', 't-MoS2')])
        form.commit()

        moved_structures = tmp_path / 'moved_cif'
        moved_potentials = tmp_path / 'moved_sw'
        moved_structures.mkdir()
        moved_potentials.mkdir()
        for name in ('h-MoS2', 't-MoS2'):
            (moved_structures / f'{name}.cif').write_text('', encoding='utf-8')
            (moved_potentials / f'{name}.sw').write_text('', encoding='utf-8')

        section = form.sections['2D']
        section.editors['cif_path'].edit.setText(str(moved_structures))
        section.editors['cif_path'].edit.editingFinished.emit()
        section.editors['pot_path'].edit.setText(str(moved_potentials))
        section.editors['pot_path'].edit.editingFinished.emit()
        form.commit()

        assert form.document.get('2D-2', 'cif_path') == 'moved_cif/{mat2}.cif'
        assert form.document.get('2D-2', 'pot_path') == 'moved_sw/{mat2}.sw'
        form.deleteLater()

    def test_generating_pairs_commits_them(self, qt_app, dirs, tmp_path,
                                           monkeypatch):
        """Generate is a real edit, like ticking a material: it must reach the
        document rather than sitting in the table until something else
        happens to commit."""
        form, editor = self._pair_form(tmp_path, monkeypatch, dirs=dirs)
        editor.set_selected(['h-MoS2', 't-MoS2'])
        editor.generate_pairs(include_reversed=True)

        assert form.document.material_entries() == [('h-MoS2', 't-MoS2'),
                                                    ('t-MoS2', 'h-MoS2')]
        form.deleteLater()

    def test_a_rescan_after_generating_pairs_does_not_retick_everything(
            self, qt_app, dirs, tmp_path, monkeypatch):
        """Generate is a decision: a later rescan (a material appeared in the
        directory) must keep the pool the pairs came from, not fall back to the
        "nobody has chosen yet" default of ticking everything usable."""
        form, editor = self._pair_form(tmp_path, monkeypatch, dirs=dirs)
        editor.generate_pairs()
        pairs = editor.selected_pairs()
        assert sorted(editor.selected_names()) == ['b-AlSb', 'h-MoS2', 't-MoS2']

        structures, potentials = dirs
        (structures / 'z-New.cif').write_text('', encoding='utf-8')
        (potentials / 'z-New.sw').write_text('', encoding='utf-8')
        editor.rescan(str(structures), str(potentials), '.sw')   # a different triple

        assert 'z-New' in editor.usable_names()
        assert 'z-New' not in editor.selected_names()
        assert editor.selected_pairs() == pairs
        form.deleteLater()

    def test_unticking_a_pair_sweep_clears_both_sheets(self, qt_app, dirs,
                                                       tmp_path, monkeypatch):
        form, editor = self._pair_form(
            tmp_path, monkeypatch, dirs=dirs, pairs=[('h-MoS2', 't-MoS2')])
        form.commit()
        assert form.document.get('2D-2', 'mat') == '{mat2}'

        editor.set_sweeping(False)
        form.commit()

        assert not form.document.has_material_list()
        assert form.document.get('2D-2', 'mat') != '{mat2}'
        assert form.document.get('2D', 'mat') != '{mat1}'
        form.deleteLater()


class TestSweepRoundTrip:
    """Opening a configured sweep must show what it actually runs."""

    def test_an_existing_list_becomes_the_selection(self, qt_app, dirs, tmp_path,
                                                    monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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
        from frictionsim2d.gui.widgets.form import ConfigForm

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


class TestNoCwdPollution:
    """Regression guard for the cwd-pollution bug: write_material_names used
    to fall back to Path.cwd() when a document had never been saved, so
    ticking a sweep on a brand-new config dropped a materials.txt wherever
    the app happened to be launched from -- a real 156-line file landed in
    the repo root exactly this way during a test run. Because
    frictionsim2d.core.utils.search_paths resolves a relative name against the cwd
    before base_dir (deliberately, so that ordering is not touched here),
    the stray file silently shadowed other configs' own relative
    materials_list and broke an unrelated test
    (test_gui_pairs_elements.py::test_scan_elements_across_material_sweep).
    """

    def test_ticking_a_sweep_on_an_unsaved_config_does_not_write_into_the_cwd(
            self, qt_app, dirs, monkeypatch):
        from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415
        from frictionsim2d.gui.widgets.form import ConfigForm  # noqa: PLC0415

        monkeypatch.chdir(REPO_ROOT)
        # A real, blocking QMessageBox with no one to click it -- see the
        # window fixture above for the same reasoning. Declining to write
        # here is expected to warn the user; the warning itself is not what
        # this test is checking.
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: None)
        before = set(REPO_ROOT.iterdir())

        document = ConfigDocument.blank('indent')  # never saved
        form = ConfigForm(document)
        section = form.sections['2D']
        editor = section.editors['materials_list']
        editor.set_sweeping(True)
        editor.rescan(*dirs, '.sw')
        section.commit()

        assert not document.has_material_list()
        assert set(REPO_ROOT.iterdir()) == before
        form.deleteLater()
