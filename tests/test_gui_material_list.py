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
        editor.set_directories(*dirs)
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

    def test_deselecting_everything_survives_a_rescan(self, editor):
        """An empty selection is a choice, not an absence. Treating it as
        'select all' on the next scan silently re-widens the sweep."""
        editor.select_none()
        editor._rescan()
        assert editor.selected_names() == []


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
        editor.set_directories(*dirs)
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
        editor.set_directories(*dirs)
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
