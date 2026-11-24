"""One editor per field kind.

Every editor exposes the same tiny interface -- ``value()``, ``set_value()``,
``changed`` -- so the form builder never needs to know which kind it is holding.

Values round-trip through the same Python types the config mapping uses, so a
blank field is ``None`` (which the writer emits as an empty value, and the
reader reads back as ``None``) rather than an empty string.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from ...potentials.utils import KNOWN_TYPES
from ..schema.introspect import FieldKind, FieldSpec

__all__ = ['FieldEditor', 'build_editor', 'EDITOR_REGISTRY', 'register_editor',
           'HIGHLIGHT_COLOR']

#: Background of a field the assistant just set (see `set_highlight`).
HIGHLIGHT_COLOR = '#fff3b0'


class FieldEditor(QWidget):
    """Base class: a control plus the metadata needed to explain it."""

    changed = Signal()

    def __init__(self, spec: FieldSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._error: Optional[str] = None
        #: The explanatory tooltip set by the form builder (help/detail text),
        #: as opposed to a transient error message. Remembered separately so
        #: clearing an error can restore it instead of leaving the tooltip
        #: blank -- set_error(None) is called on every editor on every
        #: show_errors(), including the common case where there is nothing to
        #: report.
        self._base_tooltip: str = ''
        #: The `enable_hint` currently shown as placeholder text, or None when
        #: the row is enabled. Tracked so `set_disabled_hint` is idempotent
        #: (`_apply_dependencies` calls it on every commit) and so re-enabling
        #: knows to restore rather than merely clear.
        self._disabled_hint: Optional[str] = None
        #: Each QLineEdit's placeholder as it was before a disabled-hint
        #: overwrote it (e.g. "default: 2.0"), so re-enabling gives it back
        #: instead of leaving the placeholder blank.
        self._saved_placeholders: Dict[QLineEdit, str] = {}
        #: "before → after" while the field shows a value the assistant set,
        #: else None. See `set_highlight`.
        self._highlight: Optional[str] = None

    def value(self) -> Any:
        raise NotImplementedError

    def set_value(self, value: Any) -> None:
        raise NotImplementedError

    def set_base_tooltip(self, text: Optional[str]) -> None:
        """The tooltip to show when there is no validation error."""
        self._base_tooltip = text or ''
        if not self._error:
            self.setToolTip(self._tooltip())

    def _tooltip(self) -> str:
        """Error first, then what the assistant changed, then the help text."""
        if self._error:
            return self._error
        if self._highlight:
            text = f'Set by assistant: {self._highlight}'
            return f'{text}\n\n{self._base_tooltip}' if self._base_tooltip else text
        return self._base_tooltip

    def set_error(self, message: Optional[str]) -> None:
        """Mark the field invalid. Idempotent: repeating the same state is a
        no-op, so a full-form show_errors() pass does not repaint every row."""
        if message == self._error:
            return
        self._error = message
        self.setToolTip(self._tooltip())
        self._apply_error_style(bool(message))

    def set_highlight(self, text: Optional[str]) -> None:
        """Mark (``"before → after"``) or unmark a value the assistant set.

        A pale yellow background, on top of -- never instead of -- the error
        border: both are written into one stylesheet by `_apply_style`.
        """
        if text == self._highlight:
            return
        self._highlight = text
        self.setToolTip(self._tooltip())
        self._apply_style(bool(self._error))

    def error(self) -> Optional[str]:
        """The validation message currently shown against this field, if any."""
        return self._error

    def set_disabled_hint(self, hint: Optional[str]) -> None:
        """Show `hint` (an `enable_if` reason) as placeholder text while the
        control is disabled; restore each field's normal placeholder (e.g. a
        "default: X" from `_prefill_required`'s sibling editors) once it is
        enabled again.

        A tooltip-only explanation needs a hover to find; a greyed row with
        nothing on screen reads as a missing feature rather than an inert
        one. Works generically across every editor kind via `findChildren`,
        the same trick `_apply_error_style` uses, rather than each editor
        having to plumb this through by hand.
        """
        if hint == self._disabled_hint:
            return
        self._disabled_hint = hint
        for line_edit in self.findChildren(QLineEdit):
            if hint is not None:
                self._saved_placeholders.setdefault(
                    line_edit, line_edit.placeholderText())
                line_edit.setPlaceholderText(hint)
            else:
                line_edit.setPlaceholderText(
                    self._saved_placeholders.pop(line_edit, ''))

    def warning(self) -> Optional[str]:
        """A non-fatal note about the current value, or None.

        Distinct from `set_error`, which reports a validation failure the model
        raised. This is the widget noticing something itself.
        """
        return None

    def _apply_error_style(self, invalid: bool) -> None:
        self._apply_style(invalid)

    def _apply_style(self, invalid: bool) -> None:
        """Error border and assistant highlight, together.

        Text boxes carry both in their stylesheet (dark text on the yellow, so
        a dark theme stays readable). Every other control -- check boxes,
        combos -- sits on the editor's own background, filled yellow.
        """
        rules = []
        if invalid:
            rules.append('border: 1px solid palette(highlight);')
        if self._highlight:
            rules.append(f'background-color: {HIGHLIGHT_COLOR}; color: black;')
        for child in self.findChildren(QLineEdit):
            child.setStyleSheet(' '.join(rules))
        filled = bool(self._highlight)
        if filled == self.autoFillBackground():
            return  # an error toggled: the fill is already right
        self.setAutoFillBackground(filled)
        if filled:
            palette = self.palette()
            palette.setColor(QPalette.Window, QColor(HIGHLIGHT_COLOR))
            self.setPalette(palette)
        else:
            # A palette with no roles set: everything inherited again.
            self.setPalette(QPalette())


class _TextEditor(FieldEditor):
    """Free text, and the fallback for anything unusual."""

    def __init__(self, spec: FieldSpec, parent=None) -> None:
        super().__init__(spec, parent)
        self._was_string = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.edit.textEdited.connect(self.changed)
        if spec.default not in (None, ''):
            self.edit.setPlaceholderText(f'default: {spec.default}')
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
        if spec.default not in (None, ''):
            self.edit.setPlaceholderText(f'default: {spec.default}')
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

    def __init__(self, spec: FieldSpec, parent=None, choices=None,
                choice_labels=None) -> None:
        super().__init__(spec, parent)
        self._unknown: Optional[Any] = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.box = QComboBox(self)
        if not spec.required:
            unset_text = ('(unset)' if spec.default in (None, '')
                          else f'(default: {spec.default})')
            self.box.addItem(unset_text, None)
        # Choices may be narrowed below what the model allows: outer_loop
        # accepts pressure/scan_speed/force, but each family's builder only
        # acts on some of them.
        labels = choice_labels or {}
        for choice in (choices if choices is not None else spec.choices):
            # itemData keeps the real stored value regardless of the label,
            # so a relabelled choice still round-trips as the original string.
            self.box.addItem(labels.get(choice, str(choice)), choice)
        self.box.currentIndexChanged.connect(self._on_changed)
        layout.addWidget(self.box)
        layout.addStretch(1)

    def _on_changed(self) -> None:
        # Selecting a real value retires the synthetic entry. Clearing the flag
        # without removing the item leaves findData() able to match it later,
        # so the value stops being recognised as unrecognised.
        if self._unknown is not None and self.box.currentData() != self._unknown:
            self.box.blockSignals(True)
            try:
                self._drop_unknown_item()
            finally:
                self.box.blockSignals(False)
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


class _PotTypeEditor(FieldEditor):
    """The known potential types, plus a "Custom…" entry for any other.

    ``pot_type`` accepts any pair style the installed LAMMPS lists, so a closed
    combo would refuse real configs. "Custom…" reveals a text box; a loaded
    value outside :data:`KNOWN_TYPES` selects it with the text filled in, so
    it round-trips exactly as written. The value is always the plain string.
    """

    CUSTOM_TEXT = 'Custom…'
    #: itemData of the Custom entry -- not a string any pot_type could be.
    _CUSTOM = object()

    def __init__(self, spec: FieldSpec, parent=None) -> None:
        super().__init__(spec, parent)
        self._was_string = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.box = QComboBox(self)
        # Always a blank first entry, required or not: without one a blank
        # config would show (and commit) whichever type sorts first.
        self.box.addItem('(unset)', None)
        for name in sorted(KNOWN_TYPES):
            self.box.addItem(name, name)
        self.box.addItem(self.CUSTOM_TEXT, self._CUSTOM)
        self.box.currentIndexChanged.connect(self._on_index_changed)
        layout.addWidget(self.box)
        self.custom_edit = QLineEdit(self)
        self.custom_edit.setPlaceholderText('LAMMPS pair style, e.g. sw/mod')
        self.custom_edit.textEdited.connect(self.changed)
        self.custom_edit.setVisible(False)
        layout.addWidget(self.custom_edit, 1)
        layout.addStretch(0)

    def is_custom(self) -> bool:
        """Whether the Custom entry is selected."""
        return self.box.currentData() is self._CUSTOM

    def _on_index_changed(self) -> None:
        self.custom_edit.setVisible(self.is_custom())
        self.changed.emit()

    def value(self) -> Any:
        if not self.is_custom():
            return self.box.currentData()
        text = self.custom_edit.text().strip()
        if not text:
            return '' if self._was_string else None
        if ',' in text:
            # A hand-written list (a sweep) shown as text; keep it a list.
            return [part.strip() for part in text.strip('[]').split(',')
                    if part.strip()]
        return text

    def set_value(self, value: Any) -> None:
        self._was_string = isinstance(value, str)
        self.box.blockSignals(True)
        try:
            index = (self.box.findData(value) if isinstance(value, str) and value
                     else (0 if value in (None, '') else -1))
            if index >= 0:
                self.box.setCurrentIndex(index)
                self.custom_edit.setText('')
            else:
                self.box.setCurrentIndex(self.box.count() - 1)
                text = (', '.join(str(v) for v in value)
                        if isinstance(value, (list, tuple)) else str(value))
                self.custom_edit.setText(text)
            self.custom_edit.setVisible(self.is_custom())
        finally:
            self.box.blockSignals(False)


class _ListEditor(FieldEditor):
    """A comma-separated list, shown the way it is written in the config."""

    def __init__(self, spec: FieldSpec, parent=None) -> None:
        super().__init__(spec, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText('e.g. 1, 2, 3')
        self.edit.textEdited.connect(self.changed)
        if spec.default not in (None, ''):
            self.edit.setPlaceholderText(f'default: {spec.default}')
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
            # [] round-trips as [] only when the default is [] (else an
            # untouched form dirties the settings); a required or non-empty-
            # default list is genuinely unset when cleared.
            return [] if self.spec.default == [] else None
        items = [part.strip() for part in text.strip('[]').split(',')
                 if part.strip()]
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
        if spec.default in (None, ''):
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

    def set_value(self, value: Any) -> None:
        # A one-item list keeps its brackets: `[7]` and `7` hash differently.
        if isinstance(value, (list, tuple)) and len(value) == 1:
            self.edit.setText(f'[{value[0]}]')
        else:
            super().set_value(value)


class _PathEditor(FieldEditor):
    """A file path, with a browse button.

    Directory mode is a separate axis from the field itself: `cif_path`/
    `pot_path` mean a single file normally, but a directory to scan while a
    material sweep is on. `SectionForm` is what knows which is current --
    this widget only needs to remember the flag and act on it.
    """

    def __init__(self, spec: FieldSpec, parent=None) -> None:
        super().__init__(spec, parent)
        self._was_string = False
        self._directory_mode = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.edit.textEdited.connect(self.changed)
        if spec.default not in (None, ''):
            self.edit.setPlaceholderText(f'default: {spec.default}')
        layout.addWidget(self.edit, 1)

        self._browse_button = QPushButton('…', self)
        self._browse_button.setFixedWidth(30)
        self._browse_button.setToolTip('Browse for a file')
        self._browse_button.clicked.connect(self._browse)
        layout.addWidget(self._browse_button)

    def set_directory_mode(self, directory: bool) -> None:
        """Switch the browse button (and its tooltip) between file and
        directory pickers -- called by SectionForm as the sweep is ticked or
        unticked, never decided by this widget on its own."""
        self._directory_mode = directory
        self._browse_button.setToolTip(
            'Browse for a directory' if directory else 'Browse for a file')

    def _browse(self) -> None:
        if self._directory_mode:
            chosen = QFileDialog.getExistingDirectory(self, f'Select {self.spec.key}')
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, f'Select {self.spec.key}')
        if chosen:
            self.edit.setText(chosen)
            self.changed.emit()
            # A rescan trigger for the 2D section's sweep pickers (wired via
            # editingFinished, not this widget's own `changed`): setText()
            # alone does not fire editingFinished, so a browse selection has
            # to raise it explicitly or the listing would only ever refresh
            # on the next manual edit.
            self.edit.editingFinished.emit()

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
            # A field whose default is a dict (default_factory=dict) rejects
            # None, so "empty" must stay {}; an Optional one is genuinely unset.
            return {} if isinstance(self.spec.default, dict) else None
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


register_editor('pot_type', _PotTypeEditor)
#: A plain ``Optional[str]`` that names a file (``interlayer_file``):
#: introspection sees a string, the override asks for the browse button.
register_editor('path', _PathEditor)


def build_editor(spec: FieldSpec, parent: Optional[QWidget] = None,
                 choices=None, editor: Optional[str] = None,
                 choice_labels=None) -> FieldEditor:
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
        choice_labels: Display text for individual enum choices, keyed by the
            stored value -- e.g. outer_loop's `scan_angle` shown as "angle".
            Cosmetic only: the combo's `itemData` (and so `value()`) still
            holds the original string.
    """
    if editor and editor in EDITOR_REGISTRY:
        return EDITOR_REGISTRY[editor](spec, parent)

    if spec.kind is FieldKind.INT:
        return _NumberEditor(spec, parent, integer=True)
    if spec.kind is FieldKind.FLOAT:
        return _NumberEditor(spec, parent, integer=False)
    if spec.kind is FieldKind.ENUM or (choices and spec.kind is FieldKind.STR):
        return _EnumEditor(spec, parent, choices=choices, choice_labels=choice_labels)

    factory = _EDITORS.get(spec.kind)
    return factory(spec, parent) if factory else _TextEditor(spec, parent)
