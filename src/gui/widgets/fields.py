"""One editor per field kind.

Every editor exposes the same tiny interface -- ``value()``, ``set_value()``,
``changed`` -- so the form builder never needs to know which kind it is holding.

Values round-trip through the same Python types the config mapping uses, so a
blank field is ``None`` (which the writer emits as an empty value, and the
reader reads back as ``None``) rather than an empty string.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from ..schema.introspect import FieldKind, FieldSpec

__all__ = ['FieldEditor', 'build_editor', 'EDITOR_REGISTRY', 'register_editor']


class FieldEditor(QWidget):
    """Base class: a control plus the metadata needed to explain it."""

    changed = Signal()

    def __init__(self, spec: FieldSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._error: Optional[str] = None

    def value(self) -> Any:
        raise NotImplementedError

    def set_value(self, value: Any) -> None:
        raise NotImplementedError

    def set_error(self, message: Optional[str]) -> None:
        """Mark the field as holding an invalid value."""
        self._error = message
        self.setToolTip(message or '')
        self._apply_error_style(bool(message))

    def error(self) -> Optional[str]:
        """The validation message currently shown against this field, if any."""
        return self._error

    def warning(self) -> Optional[str]:
        """A non-fatal note about the current value, or None.

        Distinct from `set_error`, which reports a validation failure the model
        raised. This is the widget noticing something itself.
        """
        return None

    def _apply_error_style(self, invalid: bool) -> None:
        for child in self.findChildren(QLineEdit):
            child.setStyleSheet(
                'border: 1px solid palette(highlight);' if invalid else '')


class _TextEditor(FieldEditor):
    """Free text, and the fallback for anything unusual."""

    def __init__(self, spec: FieldSpec, parent=None) -> None:
        super().__init__(spec, parent)
        self._was_string = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.edit.textEdited.connect(self.changed)
        layout.addWidget(self.edit)

    def value(self) -> Any:
        text = self.edit.text().strip()
        if text:
            return text
        # An empty box means "no value" for an optional field, but an empty
        # *string* for one that held a string: turning '' into None would fail
        # validation on a plain `str` field that simply has no value yet.
        return '' if self._was_string else None

    def set_value(self, value: Any) -> None:
        self._was_string = isinstance(value, str)
        self.edit.setText('' if value is None else str(value))


class _NumberEditor(FieldEditor):
    """A number, kept as text so the field can also be empty.

    A spin box cannot express "unset", and several of these fields are
    genuinely optional -- an empty value means "use the default".
    """

    def __init__(self, spec: FieldSpec, parent=None, integer: bool = False) -> None:
        super().__init__(spec, parent)
        self.integer = integer
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.edit.textEdited.connect(self.changed)
        layout.addWidget(self.edit)

    def _parse(self, text: str) -> Any:
        return int(text) if self.integer else float(text)

    def value(self) -> Any:
        text = self.edit.text().strip()
        if not text:
            return None

        # A list can legitimately be present -- a hand-written config may sweep
        # a field the GUI shows as a number. Keep it a list so validation can
        # say why that field cannot be swept, instead of the value arriving as
        # a string and drawing a generic "not a number".
        if ',' in text or text.startswith('['):
            items = [p.strip() for p in text.strip('[]').split(',') if p.strip()]
            try:
                return [self._parse(item) for item in items]
            except ValueError:
                return text

        try:
            return self._parse(text)
        except ValueError:
            # Hand the raw text on: validation reports it against the model,
            # rather than this widget inventing its own error message.
            return text

    def set_value(self, value: Any) -> None:
        if isinstance(value, (list, tuple)):
            self.edit.setText(', '.join(str(v) for v in value))
        else:
            self.edit.setText('' if value is None else str(value))


#: Strings an .ini may use for false. The reader returns booleans as text, and
#: bool('false') is True, so these have to be recognised explicitly.
_FALSE_TEXT = frozenset({'false', 'no', 'off', '0', 'none', ''})


def parse_bool(value: Any) -> bool:
    """Interpret a config value as a boolean.

    ``read_config`` leaves booleans as the strings the user typed, so a value
    can arrive here as ``'false'`` -- which is truthy in Python. Getting this
    wrong silently inverts a setting on the next save.
    """
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_TEXT
    return bool(value)


class _BoolEditor(FieldEditor):

    def __init__(self, spec: FieldSpec, parent=None) -> None:
        super().__init__(spec, parent)
        self._original: Any = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.box = QCheckBox(self)
        self.box.toggled.connect(self.changed)
        layout.addWidget(self.box)
        layout.addStretch(1)

    def value(self) -> Any:
        checked = self.box.isChecked()
        # Give back exactly what was loaded when the answer has not changed, so
        # opening and saving a config does not rewrite 'false' as 'False'.
        if self._original is not None and parse_bool(self._original) == checked:
            return self._original
        return checked

    def set_value(self, value: Any) -> None:
        self._original = value
        self.box.setChecked(parse_bool(value))


class _EnumEditor(FieldEditor):
    """A fixed set of choices, preserving anything it was not expecting.

    Snapping an unrecognised value to the first choice is destructive: for a
    field the model types as a plain `str` -- `stack_type` -- opening a config
    and saving it would silently rewrite the value. For a real Literal field the
    value is a validation error, and reporting it beats substituting a legal one
    before validation ever sees the original.
    """

    UNKNOWN_SUFFIX = ' (not recognised)'

    def __init__(self, spec: FieldSpec, parent=None, choices=None) -> None:
        super().__init__(spec, parent)
        self._unknown: Optional[Any] = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.box = QComboBox(self)
        if not spec.required:
            self.box.addItem('(unset)', None)
        # Choices may be narrowed below what the model allows: outer_loop
        # accepts pressure/scan_speed/force, but each family's builder only
        # acts on some of them.
        for choice in (choices if choices is not None else spec.choices):
            self.box.addItem(str(choice), choice)
        self.box.currentIndexChanged.connect(self._on_changed)
        layout.addWidget(self.box)
        layout.addStretch(1)

    def _on_changed(self) -> None:
        if self.box.currentData() != self._unknown:
            self._unknown = None
        self.changed.emit()

    def value(self) -> Any:
        return self.box.currentData()

    def _drop_unknown_item(self) -> None:
        """Remove the synthetic entry, if one is present.

        set_value owns its lifetime: leaving it behind makes the next
        findData() succeed against a stale item, so the value is no longer
        recognised as unrecognised.
        """
        if self._unknown is None:
            return
        index = self.box.findData(self._unknown)
        if index >= 0:
            self.box.removeItem(index)
        self._unknown = None

    def set_value(self, value: Any) -> None:
        self.box.blockSignals(True)
        try:
            self._drop_unknown_item()
            index = self.box.findData(value)
            if index < 0 and value not in (None, ''):
                self._unknown = value
                self.box.addItem(f"{value}{self.UNKNOWN_SUFFIX}", value)
                index = self.box.count() - 1
            self.box.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.box.blockSignals(False)

    def warning(self) -> Optional[str]:
        if self._unknown is None or self.box.currentData() != self._unknown:
            return None
        return (f"'{self._unknown}' is not one of the recognised values; "
                f"it is kept as written.")


class _ListEditor(FieldEditor):
    """A comma-separated list, shown the way it is written in the config."""

    def __init__(self, spec: FieldSpec, parent=None) -> None:
        super().__init__(spec, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText('e.g. 1, 2, 3')
        self.edit.textEdited.connect(self.changed)
        layout.addWidget(self.edit)

    def _coerce(self, text: str) -> Any:
        item_kind = self.spec.item_kind
        if item_kind is FieldKind.INT:
            return int(text)
        if item_kind is FieldKind.FLOAT:
            return float(text)
        return text

    def value(self) -> Any:
        text = self.edit.text().strip()
        if not text:
            return None
        items = [part.strip() for part in text.strip('[]').split(',') if part.strip()]
        try:
            return [self._coerce(item) for item in items]
        except ValueError:
            return text

    def set_value(self, value: Any) -> None:
        if value is None:
            self.edit.setText('')
        elif isinstance(value, (list, tuple)):
            self.edit.setText(', '.join(str(v) for v in value))
        else:
            self.edit.setText(str(value))


class _ScalarOrListEditor(_ListEditor):
    """One value, or a list of them.

    The arity matters: a single load is one run, a list is looped. Whichever
    the user types is preserved rather than normalised, because the config
    treats them differently.
    """

    def __init__(self, spec: FieldSpec, parent=None) -> None:
        super().__init__(spec, parent)
        self.edit.setPlaceholderText('one value, or a comma-separated list')

    def value(self) -> Any:
        text = self.edit.text().strip()
        if not text:
            return None
        if ',' not in text and not text.startswith('['):
            try:
                return self._coerce(text)
            except ValueError:
                return text
        return super().value()


class _PathEditor(FieldEditor):
    """A file path, with a browse button."""

    def __init__(self, spec: FieldSpec, parent=None) -> None:
        super().__init__(spec, parent)
        self._was_string = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.edit.textEdited.connect(self.changed)
        layout.addWidget(self.edit, 1)

        browse = QPushButton('…', self)
        browse.setFixedWidth(30)
        browse.setToolTip('Browse for a file')
        browse.clicked.connect(self._browse)
        layout.addWidget(browse)

    def _browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog  # noqa: PLC0415

        chosen, _ = QFileDialog.getOpenFileName(self, f'Select {self.spec.key}')
        if chosen:
            self.edit.setText(chosen)
            self.changed.emit()

    def value(self) -> Any:
        text = self.edit.text().strip()
        if text:
            return text
        return '' if self._was_string else None

    def set_value(self, value: Any) -> None:
        self._was_string = isinstance(value, str)
        self.edit.setText('' if value is None else str(value))


class _DictEditor(FieldEditor):
    """Free-form key/value pairs, entered as JSON."""

    def __init__(self, spec: FieldSpec, parent=None) -> None:
        super().__init__(spec, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText('{"key": "value"}')
        self.edit.textEdited.connect(self.changed)
        layout.addWidget(self.edit)

    def value(self) -> Any:
        text = self.edit.text().strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def set_value(self, value: Any) -> None:
        self.edit.setText('' if not value else json.dumps(value))


_EDITORS = {
    FieldKind.BOOL: _BoolEditor,
    FieldKind.ENUM: _EnumEditor,
    FieldKind.LIST: _ListEditor,
    FieldKind.SCALAR_OR_LIST: _ScalarOrListEditor,
    FieldKind.PATH: _PathEditor,
    FieldKind.DICT: _DictEditor,
}


#: Purpose-built editors, keyed by the slug a FieldOverride requests.
#: Populated by the widget modules themselves so this module does not import
#: them -- see widgets/bands.py and widgets/keyvalue.py.
EDITOR_REGISTRY: Dict[str, type] = {}


def register_editor(slug: str, factory: type) -> None:
    """Make a purpose-built editor requestable by slug."""
    EDITOR_REGISTRY[slug] = factory


def build_editor(spec: FieldSpec, parent: Optional[QWidget] = None,
                 choices=None, editor: Optional[str] = None) -> FieldEditor:
    """The editor for a field.

    Anything without a dedicated editor falls back to text rather than being
    omitted, so a new config field is editable before anyone writes a widget
    for it.

    Args:
        spec: The field to build a control for.
        parent: Qt parent.
        choices: Narrowed enum choices, where the model permits more than this
            simulation family can act on. Declared choices on a plain `str`
            field also select a combo -- `stack_type` is typed `str` but the
            builders only act on AA versus not-AA.
        editor: Registry slug requesting a purpose-built editor. An unknown
            slug falls back to the field kind rather than failing, so an
            override naming a widget that has not been written yet still
            renders something editable.
    """
    if editor and editor in EDITOR_REGISTRY:
        return EDITOR_REGISTRY[editor](spec, parent)

    if spec.kind is FieldKind.INT:
        return _NumberEditor(spec, parent, integer=True)
    if spec.kind is FieldKind.FLOAT:
        return _NumberEditor(spec, parent, integer=False)
    if spec.kind is FieldKind.ENUM or (choices and spec.kind is FieldKind.STR):
        return _EnumEditor(spec, parent, choices=choices)

    factory = _EDITORS.get(spec.kind)
    return factory(spec, parent) if factory else _TextEditor(spec, parent)
