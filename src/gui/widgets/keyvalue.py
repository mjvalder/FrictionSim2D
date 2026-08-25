"""Grids for the dict- and fixed-list-shaped settings.

``output.dump`` and ``output.dump_frequency`` are ``Dict[str, bool]`` and
``Dict[str, int]``; ``quench.quench_slab_dims`` is a three-element list. All
three fall to a single line of JSON otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..schema.introspect import FieldKind, FieldSpec
from .fields import FieldEditor, parse_bool, register_editor

__all__ = ['KeyValueGridEditor', 'FixedListEditor']

#: Display text for dict keys whose stored form reads badly in a form. The
#: stored key is never changed -- only the QLabel beside its control -- so the
#: YAML round-trip is untouched.
_KEY_LABELS = {
    'peratom': 'per atom',
}


def _key_label(key: str) -> str:
    """Form caption for a dict key; the raw key unless one is declared."""
    return _KEY_LABELS.get(key, key)


#: Cell labels for a three-element dimension list.
_XYZ = ('x', 'y', 'z')


class KeyValueGridEditor(FieldEditor):
    """One row per key, with a control chosen from the value's type.

    The keys belong to the user -- ``output.dump`` keys are dump-phase names,
    not a fixed schema -- so rows can be added and removed.
    """

    def __init__(self, spec: FieldSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(spec, parent)
        self._rows: Dict[str, tuple] = {}

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._rows_host = QWidget(self)
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._layout.addWidget(self._rows_host)

        controls = QHBoxLayout()
        self._new_key = QLineEdit(self)
        self._new_key.setPlaceholderText('new key')
        controls.addWidget(self._new_key)

        add = QPushButton('Add', self)
        add.clicked.connect(self._on_add_clicked)
        controls.addWidget(add)
        controls.addStretch(1)
        self._layout.addLayout(controls)

    # ------------------------------------------------------------------ rows

    def add_row(self, key: str, value: Any) -> None:
        """Add one key. A bool gets a checkbox, anything else a text cell.

        Which control to use is decided from ``spec.item_kind`` -- the
        declared value type of the dict, e.g. ``bool`` for ``output.dump`` --
        rather than from ``value``'s runtime type. ``_on_add_clicked`` always
        hands this an empty string, so choosing by runtime type gave every
        hand-added row a text box, even for a bool-valued dict.
        """
        if not key or key in self._rows:
            return

        row = QWidget(self._rows_host)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(_key_label(key), row))

        if self.spec.item_kind is FieldKind.BOOL:
            control = QCheckBox(row)
            control.setChecked(parse_bool(value))
            control.toggled.connect(self.changed)
        else:
            control = QLineEdit(row)
            control.setText('' if value is None else str(value))
            control.setMaximumWidth(90)
            control.textEdited.connect(self.changed)
        layout.addWidget(control)

        remove = QPushButton('−', row)
        remove.setFixedWidth(26)
        remove.clicked.connect(lambda _=False, k=key: self.remove_row(k))
        layout.addWidget(remove)
        layout.addStretch(1)

        self._rows[key] = (row, control)
        self._rows_layout.addWidget(row)
        self.changed.emit()

    def remove_row(self, key: str) -> None:
        entry = self._rows.pop(key, None)
        if entry is None:
            return
        row, _control = entry
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self.changed.emit()

    def _on_add_clicked(self) -> None:
        key = self._new_key.text().strip()
        if key:
            self.add_row(key, '')
            self._new_key.clear()

    # ----------------------------------------------------------------- value

    def value(self) -> Any:
        result: Dict[str, Any] = {}
        for key, (_row, control) in self._rows.items():
            if isinstance(control, QCheckBox):
                result[key] = control.isChecked()
                continue
            text = control.text().strip()
            if not text:
                continue
            try:
                result[key] = int(text)
            except ValueError:
                try:
                    result[key] = float(text)
                except ValueError:
                    result[key] = text
        return result or None

    def set_value(self, value: Any) -> None:
        for key in list(self._rows):
            self.remove_row(key)
        for key, entry in (value or {}).items():
            self.add_row(key, entry)


class FixedListEditor(FieldEditor):
    """A short list with one labelled cell per element.

    A list of the wrong length is kept as it is and flagged rather than padded
    or truncated: silently resizing the quench box is worse than saying so.
    """

    #: Bound by the registry entry; see the register_editor call below.
    LABELS = _XYZ

    def __init__(self, spec: FieldSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(spec, parent)
        self._extra: List[Any] = []
        self._cells: List[QLineEdit] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for label in self.LABELS:
            layout.addWidget(QLabel(label, self))
            cell = QLineEdit(self)
            cell.setMaximumWidth(70)
            cell.textEdited.connect(self.changed)
            self._cells.append(cell)
            layout.addWidget(cell)
        layout.addStretch(1)

    def _coerce(self, text: str) -> Any:
        if self.spec.item_kind is FieldKind.INT:
            try:
                return int(text)
            except ValueError:
                return text
        try:
            return float(text)
        except ValueError:
            return text

    def value(self) -> Any:
        values = [
            self._coerce(cell.text().strip())
            for cell in self._cells if cell.text().strip()
        ]
        values.extend(self._extra)
        return values or None

    def set_value(self, value: Any) -> None:
        items = list(value or [])
        self._extra = items[len(self._cells):]
        for index, cell in enumerate(self._cells):
            cell.setText('' if index >= len(items) else str(items[index]))

    def warning(self) -> Optional[str]:
        filled = [c for c in self._cells if c.text().strip()]
        count = len(filled) + len(self._extra)
        if count and count != len(self._cells):
            return (f"expected {len(self._cells)} values "
                    f"({', '.join(self.LABELS)}), found {count}")
        return None


register_editor('key_value', KeyValueGridEditor)
register_editor('xyz_dims', FixedListEditor)
