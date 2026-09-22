"""The filter controls above the results table.

The field names are the local store's (`materials`, `potential`,
`force_range`); `db_worker._central_filters` translates them for the REST
surface, so one form serves both sources.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)
from PySide6.QtCore import Signal

from ...style import CAPTION

__all__ = ['ANY', 'EMPTY_FILTERS', 'FilterBar', 'HIGH_SENTINEL', 'LOW_SENTINEL']

#: The "no opinion" entry of every combo.
ANY = '(any)'

#: A one-sided range is still a range: the open end gets a bound no stored
#: value can reach, rather than a None the store would have to special-case.
LOW_SENTINEL = -1e30
HIGH_SENTINEL = 1e30

#: The exact `filters` block a `list` request carries. Every key is present
#: always -- a filter that disappears from the dict would change the D13 cache
#: key without changing the query. There is no `is_complete` key: neither the
#: local store (`Friction2DDB.query`) nor the central client accepts one, and
#: spec D3 does not list a Completeness filter.
EMPTY_FILTERS: Dict[str, Any] = {
    'materials': None,
    'family': None,
    'model': None,
    'set_label': None,
    'potential': None,
    'layers': None,
    'force_range': None,
    'pressure_range': None,
    'temperature_range': None,
    'limit': 5000,
    'order_by': None,
}

#: The Load control's unit, and which filter key the range goes to. A tip is
#: pushed with a force (nN) and a sheet is pressed with a pressure (GPa); they
#: are different columns in both stores, and a range sent to the wrong one
#: matches nothing at all rather than matching everything. Sending it to
#: `force_range` unconditionally -- the obvious shortcut -- would silently hide
#: every sheetonsheet, hetero, flake and pes_sheet row the moment a load range
#: was typed, which is the whole pressure half of the campaign.
LOAD_KINDS: Tuple[Tuple[str, str, str], ...] = (
    ('force', 'Force (nN)', 'force_range'),
    ('pressure', 'Pressure (GPa)', 'pressure_range'),
)


def _float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return None


def _range(low: QLineEdit, high: QLineEdit) -> Optional[Tuple[float, float]]:
    """`(min, max)` from two boxes, either of which may be empty."""
    lower = _float(low.text())
    upper = _float(high.text())
    if lower is None and upper is None:
        return None
    return (LOW_SENTINEL if lower is None else lower,
            HIGH_SENTINEL if upper is None else upper)


class FilterBar(QWidget):
    """Family, model, material, potential, layers, load, temperature, set,
    and a free-text box that never reaches the store."""

    changed = Signal()
    submitted = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)

        self.family = self._combo(grid, 0, 0, 'Family')
        self.model = self._combo(grid, 0, 2, 'Model')
        self.material = self._combo(grid, 0, 4, 'Material', editable=True)
        # Editable because the store may not offer these as choices: the
        # potential and set axes are the two `get_available_conditions()` keys
        # the sibling plan adds for this pane, and a pane that cannot filter by
        # set until that lands would be a pane you cannot use on a shared box.
        self.potential = self._combo(grid, 0, 6, 'Potential', editable=True)

        self.layers = self._combo(grid, 1, 0, 'Layers')
        self.set_label = self._combo(grid, 1, 2, 'Set', editable=True)

        grid.addWidget(QLabel('Load', self), 1, 4)
        self.load_min = self._edit(grid, 1, 5, 'min')
        self.load_max = self._edit(grid, 1, 6, 'max')

        self.load_kind = QComboBox(self)
        for key, label, _filter_key in LOAD_KINDS:
            self.load_kind.addItem(label, key)
        self.load_kind.setToolTip(
            'Which load column the range filters. Tip-on-sheet runs carry a '
            'force in nN; sheet-on-sheet, hetero, flake and PES-sheet runs '
            'carry a pressure in GPa. A run has one or the other, never both.')
        self.load_kind.currentIndexChanged.connect(lambda _i: self.changed.emit())
        grid.addWidget(self.load_kind, 1, 7)

        grid.addWidget(QLabel('T (K)', self), 2, 4)
        self.temp_min = self._edit(grid, 2, 5, 'min')
        self.temp_max = self._edit(grid, 2, 6, 'max')

        grid.addWidget(QLabel('Find', self), 2, 0)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText(
            'Narrows the rows already fetched -- does not re-query')
        self.search.returnPressed.connect(self.submitted.emit)
        # Spans through column 6 -- where the Completeness combo (N1: removed,
        # there is no such filter) used to sit at column 7 -- so the Clear
        # button below still lines up with the load/temperature controls'
        # right edge instead of leaving a hole in the grid.
        grid.addWidget(self.search, 2, 1, 1, 6)

        self.clear_button = QPushButton('Clear', self)
        self.clear_button.clicked.connect(self.clear)
        grid.addWidget(self.clear_button, 2, 7)

        note = QLabel('Filters run against the store; Find runs in the table.', self)
        note.setStyleSheet(CAPTION)
        grid.addWidget(note, 3, 0, 1, 8)

    # ---------------------------------------------------------- construction

    def _combo(self, grid: QGridLayout, row: int, column: int, label: str,
               editable: bool = False) -> QComboBox:
        grid.addWidget(QLabel(label, self), row, column)
        box = QComboBox(self)
        box.setEditable(editable)
        box.addItem(ANY)
        box.currentTextChanged.connect(lambda _text: self.changed.emit())
        grid.addWidget(box, row, column + 1)
        return box

    def _edit(self, grid: QGridLayout, row: int, column: int,
              placeholder: str) -> QLineEdit:
        edit = QLineEdit(self)
        edit.setPlaceholderText(placeholder)
        edit.setMaximumWidth(80)
        edit.textChanged.connect(lambda _text: self.changed.emit())
        edit.returnPressed.connect(self.submitted.emit)
        grid.addWidget(edit, row, column)
        return edit

    # -------------------------------------------------------------- choices

    def set_choices(self, facets: Dict[str, List[Any]]) -> None:
        """Repopulate the combos from a query's facets, keeping selections."""
        for box, key in ((self.material, 'materials'), (self.family, 'families'),
                         (self.model, 'models'), (self.potential, 'potentials'),
                         (self.set_label, 'set_labels'), (self.layers, 'layers')):
            self._fill(box, facets.get(key) or [])

    @staticmethod
    def _fill(box: QComboBox, values: List[Any]) -> None:
        current = box.currentText()
        box.blockSignals(True)
        box.clear()
        box.addItem(ANY)
        for value in values:
            box.addItem(str(value))
        index = box.findText(current)
        box.setCurrentIndex(index if index >= 0 else 0)
        box.blockSignals(False)

    # ---------------------------------------------------------------- value

    def value(self) -> Dict[str, Any]:
        """The `filters` block of a `list` request."""
        out = dict(EMPTY_FILTERS)

        material = self.material.currentText().strip()
        if material and material != ANY:
            out['materials'] = [material]

        for key, box in (('family', self.family), ('model', self.model),
                         ('potential', self.potential),
                         ('set_label', self.set_label)):
            text = box.currentText().strip()
            if text and text != ANY:
                out[key] = text

        layers = self.layers.currentText().strip()
        if layers and layers != ANY:
            try:
                out['layers'] = int(layers)
            except ValueError:
                out['layers'] = None

        # One range control, two possible destinations -- see LOAD_KINDS.
        load_range = _range(self.load_min, self.load_max)
        if load_range is not None:
            kind = self.load_kind.currentData()
            key = next(k for name, _label, k in LOAD_KINDS if name == kind)
            out[key] = load_range

        out['temperature_range'] = _range(self.temp_min, self.temp_max)
        return out

    def set_value(self, filters: Dict[str, Any]) -> None:
        """Restore a stored filter block (D10)."""
        materials = filters.get('materials') or []
        self._select(self.material, materials[0] if materials else None)
        self._select(self.family, filters.get('family'))
        self._select(self.model, filters.get('model'))
        self._select(self.potential, filters.get('potential'))
        self._select(self.set_label, filters.get('set_label'))
        self._select(self.layers, filters.get('layers'))
        pressure = filters.get('pressure_range')
        self._set_range(self.load_min, self.load_max,
                        pressure or filters.get('force_range'))
        self.load_kind.blockSignals(True)
        self.load_kind.setCurrentIndex(1 if pressure else 0)
        self.load_kind.blockSignals(False)
        self._set_range(self.temp_min, self.temp_max,
                        filters.get('temperature_range'))

    @staticmethod
    def _select(box: QComboBox, value: Any) -> None:
        """Select a value, adding it if the combo has never heard of it.

        A restored filter (D10) arrives before any query has run, so every
        combo still holds only `(any)`. Falling back to index 0 there would
        silently drop the remembered filter -- which is the whole feature. The
        value is added as a choice instead; the next `set_choices` replaces the
        list with what the store actually holds, and a remembered value the
        store no longer has falls back to `(any)` then, where it is true.
        """
        text = ANY if value is None else str(value)
        box.blockSignals(True)
        index = box.findText(text)
        if index < 0:
            box.addItem(text)
            index = box.findText(text)
        box.setCurrentIndex(index)
        box.blockSignals(False)

    @staticmethod
    def _set_range(low: QLineEdit, high: QLineEdit,
                   bounds: Optional[Any]) -> None:
        low.blockSignals(True)
        high.blockSignals(True)
        if not bounds:
            low.clear()
            high.clear()
        else:
            lower, upper = bounds
            low.setText('' if lower == LOW_SENTINEL else str(lower))
            high.setText('' if upper == HIGH_SENTINEL else str(upper))
        low.blockSignals(False)
        high.blockSignals(False)

    def text(self) -> str:
        """The free-text box. Not a query filter -- see the class docstring."""
        return self.search.text().strip()

    def clear(self) -> None:
        self.set_value(EMPTY_FILTERS)
        self.search.clear()
        self.changed.emit()
