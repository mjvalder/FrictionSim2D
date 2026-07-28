"""Editor for Lennard-Jones pair overrides.

``[lj_override]`` is a free-form section: its keys are element pairs chosen by
the user, not fields on a model, so schema introspection has nothing to render
and the section appeared empty. Its shape is fixed even though its keys are
not -- ``"Mo-S" -> [epsilon, sigma]`` (see
:meth:`PotentialManager.set_lj_overrides`) -- so a three-column table is both
possible and clearer than free JSON.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..style import CAPTION_SMALL

__all__ = ['PairOverrideEditor']

_COLUMNS = ('Element pair', 'Epsilon (eV)', 'Sigma (Å)')


class PairOverrideEditor(QWidget):
    """A table of ``pair -> (epsilon, sigma)`` rows."""

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        caption = QLabel(
            'Override the Lennard-Jones parameters for an element pair, '
            'e.g. Mo-S. Pairs left out here use the built-in UFF values.',
            self)
        caption.setWordWrap(True)
        caption.setStyleSheet(CAPTION_SMALL)
        layout.addWidget(caption)

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

        self._loading = False

    # ------------------------------------------------------------------ rows

    def _add_row(self, pair: str = '', epsilon: Any = '', sigma: Any = '') -> None:
        self._loading = True
        try:
            row = self.table.rowCount()
            self.table.insertRow(row)
            for column, value in enumerate((pair, epsilon, sigma)):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        finally:
            self._loading = False
        self.changed.emit()

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()},
                      reverse=True)
        if not rows:
            return
        for row in rows:
            self.table.removeRow(row)
        self.changed.emit()

    def _on_item_changed(self, _item) -> None:
        if not self._loading:
            self.changed.emit()

    def _cell(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        return item.text().strip() if item else ''

    # ----------------------------------------------------------------- value

    def value(self) -> Dict[str, Any]:
        """The overrides, in the shape the potential manager expects.

        Incomplete rows are skipped rather than written as broken entries: a
        half-typed row is a row still being typed.
        """
        result: Dict[str, Any] = {}
        for row in range(self.table.rowCount()):
            pair = self._cell(row, 0)
            epsilon, sigma = self._cell(row, 1), self._cell(row, 2)
            if not pair or not epsilon or not sigma:
                continue
            try:
                result[pair] = [float(epsilon), float(sigma)]
            except ValueError:
                # Leave it as text so validation reports it against the model.
                result[pair] = [epsilon, sigma]
        return result

    def set_value(self, values: Optional[Dict[str, Any]]) -> None:
        self._loading = True
        try:
            self.table.setRowCount(0)
        finally:
            self._loading = False

        for pair, params in (values or {}).items():
            if isinstance(params, (list, tuple)) and len(params) == 2:
                self._add_row(pair, params[0], params[1])
            else:
                self._add_row(pair, params, '')
