"""Editor for Lennard-Jones pair overrides.

``[lj_override]`` is a free-form section: its keys are element pairs chosen by
the user, not fields on a model, so schema introspection has nothing to render
and the section appeared empty. Its shape is fixed even though its keys are
not -- ``"Mo-S" -> [epsilon, sigma]`` (see
:meth:`PotentialManager.set_lj_overrides`) -- so a three-column table is both
possible and clearer than free JSON.

The element half of a pair no longer has to be typed correctly by hand: the
pair column is two editable combo boxes, pre-filled with whatever
``set_known_elements`` was told the configured simulation actually places
(typing something not on that list still works -- the scan is advisory, not a
validator). A duplicate normalised pair (``Mo-S`` / ``S-Mo``, or a case
difference) and a pair naming an element the scan never saw both get a
non-blocking note in the fourth column, mirroring
``PotentialManager._pair_key``'s own normalisation so the note means the same
thing the potential manager would do silently.

A shared overrides ``.txt`` (see ``frictionsim2d.run.expand.resolve_lj_override_file``) covers
a whole material sweep from one file -- its keys are element pairs, not
materials, so pairs whose elements are absent from a given run are simply
unused. The file path lives in its own row, independent of the table: it is
resolved at expansion time, not here, but ``Import…`` can still pull its rows
in as ordinary editable ones for a quick look or a one-off tweak.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..style import CAPTION_SMALL, WARNING
from ...core.utils.tokens import normalize_element_symbol
from ...run.expand import resolve_lj_override_file

__all__ = ['PairOverrideEditor']

_COLUMNS = ('Element pair', 'Epsilon (eV)', 'Sigma (Å)', 'Notes')

_PAIR_TOKEN_RE = re.compile(r'[A-Za-z]+')
#: The notes column's warning colour, pulled from the shared WARNING style
#: string rather than restated as a separate constant here.
_NOTE_COLOR = QColor(re.search(r'#[0-9a-fA-F]+', WARNING).group(0))


def _split_pair(pair: str) -> Tuple[str, str]:
    """Split a stored ``'El1-El2'`` (or 'El1_El2', 'El1 El2', ...) key."""
    tokens = _PAIR_TOKEN_RE.findall(pair or '')
    if len(tokens) >= 2:
        return tokens[0], tokens[1]
    if len(tokens) == 1:
        return tokens[0], ''
    return '', ''


class PairOverrideEditor(QWidget):
    """A table of ``pair -> (epsilon, sigma)`` rows, plus an optional shared
    overrides file."""

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        #: Elements the current simulation actually places, as reported by
        #: `scan_config_elements`. Populates the pair combos and, once set,
        #: is what an "unscanned element" note is judged against.
        self._known_elements: List[str] = []
        #: Where a relative "shared file" path resolves against. Set by the
        #: form alongside `set_known_elements`; None (import-as-typed-only,
        #: cwd-relative) until then.
        self._base_dir = None
        self._loading = False
        #: Guards the notes column's own `setText` from re-entering
        #: `_on_item_changed` and emitting a redundant `changed`.
        self._updating_notes = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        caption = QLabel(
            'Override the Lennard-Jones parameters for an element pair, '
            'e.g. Mo-S. Pairs left out here use the built-in UFF values.',
            self)
        caption.setWordWrap(True)
        caption.setStyleSheet(CAPTION_SMALL)
        layout.addWidget(caption)

        #: Shown only under a non-LJ interlayer style: the table then covers
        #: tip and substrate pairs, while the sheet layers use that style.
        self.interlayer_note = QLabel('', self)
        self.interlayer_note.setWordWrap(True)
        self.interlayer_note.setStyleSheet(CAPTION_SMALL)
        self.interlayer_note.setVisible(False)
        layout.addWidget(self.interlayer_note)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel('Shared overrides file:', self))
        self.file_edit = QLineEdit(self)
        self.file_edit.setPlaceholderText(
            'optional .txt of pair overrides, for a whole material sweep')
        self.file_edit.textEdited.connect(self._on_file_edited)
        file_row.addWidget(self.file_edit, 1)
        browse = QPushButton('…', self)
        browse.setFixedWidth(30)
        browse.setToolTip('Browse for a file')
        browse.clicked.connect(self._browse_file)
        file_row.addWidget(browse)
        import_button = QPushButton('Import…', self)
        import_button.setToolTip(
            "Load this file's pairs into the table below as editable rows")
        import_button.clicked.connect(self._import_file)
        file_row.addWidget(import_button)
        layout.addLayout(file_row)

        self.table = QTableWidget(0, len(_COLUMNS), self)
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMaximumHeight(160)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        add = QPushButton('Add pair', self)
        add.clicked.connect(self._add_row)
        buttons.addWidget(add)

        remove = QPushButton('Remove selected', self)
        remove.clicked.connect(self._remove_selected)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ rows

    def _make_pair_cell(self, el1: str, el2: str) -> QWidget:
        container = QWidget(self.table)
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(2, 0, 2, 0)
        row_layout.setSpacing(2)

        combo1 = QComboBox(container)
        combo1.setEditable(True)
        combo2 = QComboBox(container)
        combo2.setEditable(True)
        for combo, text in ((combo1, el1), (combo2, el2)):
            combo.addItems(self._known_elements)
            combo.setCurrentText(text)
            combo.editTextChanged.connect(self._on_pair_edited)
        row_layout.addWidget(combo1, 1)
        row_layout.addWidget(QLabel('-', container))
        row_layout.addWidget(combo2, 1)

        container.combo1 = combo1  # type: ignore[attr-defined]
        container.combo2 = combo2  # type: ignore[attr-defined]
        return container

    def _add_row(self, pair: str = '', epsilon: Any = '', sigma: Any = '') -> None:
        self._loading = True
        try:
            row = self.table.rowCount()
            self.table.insertRow(row)
            el1, el2 = _split_pair(pair)
            self.table.setCellWidget(row, 0, self._make_pair_cell(el1, el2))
            for column, value in ((1, epsilon), (2, sigma)):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
            note_item = QTableWidgetItem('')
            note_item.setFlags(note_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, note_item)
        finally:
            self._loading = False
        self._refresh_notes()
        self.changed.emit()

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()},
                      reverse=True)
        if not rows:
            return
        for row in rows:
            self.table.removeRow(row)
        self._refresh_notes()
        self.changed.emit()

    def _on_item_changed(self, _item) -> None:
        if not self._loading and not self._updating_notes:
            self.changed.emit()

    def _on_pair_edited(self, _text: str) -> None:
        if self._loading:
            return
        self._refresh_notes()
        self.changed.emit()

    def _on_file_edited(self, _text: str) -> None:
        if not self._loading:
            self.changed.emit()

    def _cell(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        return item.text().strip() if item else ''

    def _pair_text(self, row: int) -> str:
        container = self.table.cellWidget(row, 0)
        if container is None:
            return ''
        el1 = container.combo1.currentText().strip()
        el2 = container.combo2.currentText().strip()
        if not el1 or not el2:
            return ''
        return f'{el1}-{el2}'

    # -------------------------------------------------------------- elements

    def set_interlayer_note(self, style_label: Optional[str],
                            partial: bool = False) -> None:
        """Say the table covers tip/substrate pairs only while the layers use
        ``style_label`` (a many-body style); None hides the note. ``partial``:
        a style sweep where only some runs use it."""
        if style_label and partial:
            self.interlayer_note.setText(
                f'Under {style_label}: tip and substrate pairs only — '
                f'layers use {style_label}')
        elif style_label:
            self.interlayer_note.setText(
                f'Tip and substrate pairs only — layers use {style_label}')
        else:
            self.interlayer_note.setText('')
        self.interlayer_note.setVisible(bool(style_label))

    def set_known_elements(self, elements: Optional[List[str]]) -> None:
        """The elements the configured simulation actually places.

        Existing rows keep whatever text they hold -- this only changes what
        the combo boxes offer and what an "unscanned element" note is judged
        against, never the pairs already typed in.
        """
        self._known_elements = sorted(dict.fromkeys(elements or []))
        for row in range(self.table.rowCount()):
            container = self.table.cellWidget(row, 0)
            if container is None:
                continue
            for combo in (container.combo1, container.combo2):
                current = combo.currentText()
                combo.blockSignals(True)
                try:
                    combo.clear()
                    combo.addItems(self._known_elements)
                    combo.setCurrentText(current)
                finally:
                    combo.blockSignals(False)
        self._refresh_notes()

    def set_base_dir(self, base_dir) -> None:
        """Directory a relative shared-file path resolves against."""
        self._base_dir = base_dir

    def _refresh_notes(self) -> None:
        known = set(self._known_elements)
        seen: Dict[Tuple[str, str], int] = {}
        self._updating_notes = True
        try:
            for row in range(self.table.rowCount()):
                container = self.table.cellWidget(row, 0)
                el1 = container.combo1.currentText().strip() if container else ''
                el2 = container.combo2.currentText().strip() if container else ''
                notes: List[str] = []

                if el1 and el2:
                    key = tuple(sorted((normalize_element_symbol(el1),
                                        normalize_element_symbol(el2))))
                    if key in seen:
                        notes.append(f"duplicate of row {seen[key] + 1}")
                    else:
                        seen[key] = row

                if known:
                    unscanned = [el for el in (el1, el2)
                                if el and el not in known]
                    if unscanned:
                        notes.append(
                            f"{', '.join(unscanned)} not in the scanned elements")

                item = self.table.item(row, 3)
                if item is None:
                    item = QTableWidgetItem('')
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, 3, item)
                text = '; '.join(notes)
                item.setText(text)
                item.setToolTip(text)
                item.setForeground(_NOTE_COLOR if text else QColor())
        finally:
            self._updating_notes = False

    # --------------------------------------------------------------- file

    def set_file(self, path: Optional[str]) -> None:
        was_loading = self._loading
        self._loading = True
        try:
            self.file_edit.setText(path or '')
        finally:
            self._loading = was_loading

    def _browse_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, 'Select overrides file')
        if chosen:
            self.set_file(chosen)
            self.changed.emit()

    def _import_file(self) -> None:
        """Pull the shared file's pairs into the table as editable rows.

        Convenience only -- the file is still resolved independently at
        expansion time (`frictionsim2d.run.expand.resolve_lj_override_file`), so importing is
        never required for the file to take effect; it just lets high-
        throughput users see and tweak individual pairs without opening the
        .txt in another editor. Inline rows already present win, matching the
        merge rule expansion itself applies.
        """
        path = self.file_edit.text().strip()
        if not path:
            return

        try:
            merged = resolve_lj_override_file(
                {**self.value(), 'file': path}, self._base_dir)
        except OSError:
            return
        merged.pop('file', None)
        self.set_value(merged)
        self.set_file(path)

    # ----------------------------------------------------------------- value

    def value(self) -> Dict[str, Any]:
        """The overrides, in the shape the potential manager expects.

        Incomplete rows are skipped rather than written as broken entries: a
        half-typed row is a row still being typed. The shared file, if any,
        rides along under the ``'file'`` key -- `resolve_lj_override_file`
        (and `PotentialManager.set_lj_overrides`'s defensive skip) is what
        strips or resolves it downstream.
        """
        result: Dict[str, Any] = {}
        for row in range(self.table.rowCount()):
            pair = self._pair_text(row)
            epsilon, sigma = self._cell(row, 1), self._cell(row, 2)
            if not pair or not epsilon or not sigma:
                continue
            try:
                result[pair] = [float(epsilon), float(sigma)]
            except ValueError:
                # Leave it as text so validation reports it against the model.
                result[pair] = [epsilon, sigma]

        file_path = self.file_edit.text().strip()
        if file_path:
            result['file'] = file_path
        return result

    def set_value(self, values: Optional[Dict[str, Any]]) -> None:
        values = dict(values or {})
        file_path = values.pop('file', None)

        self._loading = True
        try:
            self.table.setRowCount(0)
        finally:
            self._loading = False
        self.set_file(file_path)

        for pair, params in values.items():
            if isinstance(params, (list, tuple)) and len(params) == 2:
                self._add_row(pair, params[0], params[1])
            else:
                self._add_row(pair, params, '')
