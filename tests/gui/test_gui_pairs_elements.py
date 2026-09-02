"""Scanning simulation elements into the LJ override table.

The pair column ("Mo-S" -> [epsilon, sigma]) is free-typed text easy to get
wrong -- a swapped case, a typo. `scan_config_elements` reads what the
configured simulation actually places atoms of; `PairOverrideEditor` turns
that into dropdowns and a non-blocking note for a pair the scan never saw.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.controllers.document import ConfigDocument  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


# --------------------------------------------------------------------- scan


def test_scan_elements_from_cif():
    from src.gui.controllers.elements import scan_config_elements

    doc = ConfigDocument.blank('sheetonsheet')
    doc.set('2D', 'cif_path', 'examples/materials/h-MoS2.cif')
    doc.source_path = Path('.').resolve() / 'x.ini'   # base_dir = repo root

    assert set(scan_config_elements(doc)) >= {'Mo', 'S'}


def test_scan_elements_across_material_sweep(tmp_path):
    """A `{mat}` sheet with a configured material list scans every selected
    material's CIF, not just the one literal (unresolved) path -- the shared
    overrides file this feature is for is meant to cover a whole sweep."""
    from src.gui.controllers.elements import scan_config_elements

    (tmp_path / 'A.cif').write_text('data_a\n_chemical_formula_sum   "Fe1"\n')
    (tmp_path / 'B.cif').write_text('data_b\n_chemical_formula_sum   "Cu1"\n')
    (tmp_path / 'materials.txt').write_text('A\nB\n')

    doc = ConfigDocument.blank('sheetonsheet')
    doc.source_path = tmp_path / 'x.ini'
    doc.set('2D', 'materials_list', 'materials.txt')
    doc.set('2D', 'mat', '{mat}')
    doc.set('2D', 'cif_path', '{mat}.cif')

    assert set(scan_config_elements(doc)) == {'Fe', 'Cu'}


def test_scan_elements_across_a_pair_sweep(tmp_path):
    """Both sheets of a pair sweep contribute, each taking its own member of
    each entry. Gating the union on the literal '{mat}' left a pair config
    scanning `.../{mat1}.cif` -- a path that cannot exist, with no element
    token in '{mat1}' to fall back on -- so the LJ pair picker came back
    empty for the very configs that need it."""
    from src.gui.controllers.elements import scan_config_elements

    (tmp_path / 'A.cif').write_text('data_a\n_chemical_formula_sum   "Fe1"\n')
    (tmp_path / 'B.cif').write_text('data_b\n_chemical_formula_sum   "Cu1"\n')
    (tmp_path / 'C.cif').write_text('data_c\n_chemical_formula_sum   "Ni1"\n')
    (tmp_path / 'pairs.txt').write_text('A + B\nA + C\n')

    doc = ConfigDocument.blank('sheetonsheet')
    doc.source_path = tmp_path / 'x.ini'
    doc.add_second_material()
    doc.set('2D', 'materials_list', 'pairs.txt')
    doc.set('2D', 'mat', '{mat1}')
    doc.set('2D', 'cif_path', '{mat1}.cif')
    doc.set('2D-2', 'mat', '{mat2}')
    doc.set('2D-2', 'cif_path', '{mat2}.cif')

    assert set(scan_config_elements(doc)) == {'Fe', 'Cu', 'Ni'}


def test_scan_elements_mat_fallback_reachable_without_cif_path():
    """A missing/blank cif_path must not skip the section outright -- the
    third-tier `mat`-token fallback has to run even when there is no CIF to
    even attempt reading."""
    from src.gui.controllers.elements import scan_config_elements

    doc = ConfigDocument.blank('sheetonsheet')
    doc.set('2D', 'mat', 'h_MoS2')
    doc.set('2D', 'cif_path', None)
    doc.source_path = Path('.').resolve() / 'x.ini'

    assert set(scan_config_elements(doc)) == {'Mo', 'S'}


def test_scan_elements_falls_back_to_mat_text_when_cif_is_unreadable():
    from src.gui.controllers.elements import scan_config_elements

    doc = ConfigDocument.blank('sheetonsheet')
    doc.set('2D', 'cif_path', 'no/such/file.cif')
    doc.set('2D', 'mat', 'h_MoS2')
    doc.source_path = Path('.').resolve() / 'x.ini'

    assert set(scan_config_elements(doc)) == {'Mo', 'S'}


# ------------------------------------------------------------- pair editor


class TestPairOverrideEditorElements:

    def test_set_known_elements_populates_pair_combos(self, qt_app):
        from src.gui.widgets.pairs import PairOverrideEditor

        editor = PairOverrideEditor()
        editor.set_known_elements(['Mo', 'S', 'Se'])
        editor._add_row('Mo-S', 0.4124, 3.75114)  # pylint: disable=protected-access

        container = editor.table.cellWidget(0, 0)
        items1 = [container.combo1.itemText(i) for i in range(container.combo1.count())]
        assert items1 == ['Mo', 'S', 'Se']
        assert container.combo1.currentText() == 'Mo'
        assert container.combo2.currentText() == 'S'

        # value() still emits the same 'El1-El2' shape the potential manager
        # parses, regardless of the pair column now being two combo boxes.
        assert editor.value() == {'Mo-S': [0.4124, 3.75114]}

    def test_typing_an_unlisted_element_still_works(self, qt_app):
        from src.gui.widgets.pairs import PairOverrideEditor

        editor = PairOverrideEditor()
        editor.set_known_elements(['Mo', 'S'])
        editor._add_row()  # pylint: disable=protected-access

        container = editor.table.cellWidget(0, 0)
        container.combo1.setEditText('Xx')
        container.combo2.setEditText('Yy')
        editor.table.item(0, 1).setText('0.5')
        editor.table.item(0, 2).setText('3.0')

        assert editor.value() == {'Xx-Yy': [0.5, 3.0]}

    def test_duplicate_normalised_pair_gets_a_note(self, qt_app):
        from src.gui.widgets.pairs import PairOverrideEditor

        editor = PairOverrideEditor()
        editor.set_known_elements(['Mo', 'S'])
        editor._add_row('Mo-S', 0.1, 3.0)     # pylint: disable=protected-access
        editor._add_row('S-Mo', 0.2, 3.1)     # pylint: disable=protected-access

        assert 'duplicate' in editor.table.item(1, 3).text().lower()
        assert editor.table.item(0, 3).text() == ''

    def test_unscanned_element_gets_a_note(self, qt_app):
        from src.gui.widgets.pairs import PairOverrideEditor

        editor = PairOverrideEditor()
        editor.set_known_elements(['Mo', 'S'])
        editor._add_row('Xx-Mo', 0.1, 3.0)    # pylint: disable=protected-access

        assert 'xx' in editor.table.item(0, 3).text().lower()

    def test_set_file_round_trips_through_value_and_set_value(self, qt_app):
        from src.gui.widgets.pairs import PairOverrideEditor

        editor = PairOverrideEditor()
        editor.set_file('shared.txt')
        assert editor.value() == {'file': 'shared.txt'}

        editor.set_value({'file': 'shared2.txt', 'Mo-S': [0.4, 3.7]})
        assert editor.file_edit.text() == 'shared2.txt'
        assert editor.value() == {'file': 'shared2.txt', 'Mo-S': [0.4, 3.7]}

    def test_import_pulls_shared_file_pairs_into_the_table(self, qt_app, tmp_path):
        from src.gui.widgets.pairs import PairOverrideEditor

        lj_file = tmp_path / 'lj.txt'
        lj_file.write_text('Mo-S  0.4124  3.75114\n')

        editor = PairOverrideEditor()
        editor.set_file(str(lj_file))
        editor._import_file()  # pylint: disable=protected-access

        assert editor.value()['Mo-S'] == [0.4124, 3.75114]


# ------------------------------------------------------------ live refresh


def test_known_elements_refresh_is_not_per_keystroke(qt_app, monkeypatch):
    """`_wire_material_sweep`'s docstring (form.py) explains why a
    disk-touching rescan must never be wired through `changed` --
    `_on_field_changed` fires it on every keystroke (textEdited), not just a
    settled edit. The lj_override known-elements refresh reads CIFs off disk
    the same way `scan_config_elements` does, so it has to obey the same
    rule: only `editingFinished` (or a section toggle, or a materials-list
    change) may trigger it, never a keystroke."""
    import src.gui.widgets.form as form_module

    calls = []

    def fake_scan(_document):
        calls.append(1)
        return []

    monkeypatch.setattr(form_module, 'scan_config_elements', fake_scan)

    doc = ConfigDocument.blank('afm')
    cform = form_module.ConfigForm(doc)
    try:
        calls.clear()  # ignore whatever construction itself scanned

        mat_edit = cform.sections['2D'].editors['mat'].edit
        for ch in 'h_MoS2':
            mat_edit.setText(mat_edit.text() + ch)
            mat_edit.textEdited.emit(mat_edit.text())

        assert calls == [], (
            f"expected zero scans from {len('h_MoS2')} keystrokes, got {len(calls)}")

        mat_edit.editingFinished.emit()

        assert len(calls) == 1
    finally:
        cform.deleteLater()
