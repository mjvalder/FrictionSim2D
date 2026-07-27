"""Assembling editors into a form.

One group box per config section, one row per field, ordered and labelled from
the override table. Optional sections carry a checkbox that adds or removes the
whole section -- which is how the models express "no substrate", "no flake": an
absent section, not a flag.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..controllers.document import ConfigDocument
from ..schema.introspect import FieldSpec, SectionSpec
from ..schema.overrides import SectionOverride, overrides_for
from .fields import FieldEditor, build_editor

__all__ = ['SectionForm', 'ConfigForm']


def _label_for(spec: FieldSpec, override: Optional[SectionOverride]) -> str:
    """Display label for a field: the override's, else a tidied field name."""
    field_override = override.fields.get(spec.key) if override else None
    if field_override and field_override.label:
        label = field_override.label
    else:
        label = spec.key.replace('_', ' ').strip().capitalize()

    if field_override and field_override.unit:
        label = f"{label} ({field_override.unit})"
    return label


def _ordered_fields(section: SectionSpec,
                    override: Optional[SectionOverride]) -> List[FieldSpec]:
    """Fields in display order; unlisted ones keep schema order, at the end."""
    def key(spec: FieldSpec):
        field_override = override.fields.get(spec.key) if override else None
        return (field_override.order if field_override else 100, spec.key)

    return sorted(section.fields, key=key)


class SectionForm(QGroupBox):
    """One config section."""

    changed = Signal()
    section_toggled = Signal(str, bool)

    def __init__(self, section: SectionSpec, document: ConfigDocument,
                 parent: Optional[QWidget] = None) -> None:
        overrides = overrides_for(document.model)
        self.override = overrides.get(section.key)
        title = (self.override.label if self.override and self.override.label
                 else section.key)

        super().__init__(title, parent)
        self.section = section
        self.document = document
        self.editors: Dict[str, FieldEditor] = {}
        self._loading = False

        locked = document.is_section_locked(section.key)
        if section.optional and not locked:
            self.setCheckable(True)
            self.setChecked(document.has_section(section.key))
            self.toggled.connect(self._on_toggled)
        elif locked:
            self.setToolTip(document.section_lock_reason(section.key) or '')

        layout = QVBoxLayout(self)

        help_text = self.override.help if self.override else None
        if help_text:
            caption = QLabel(help_text, self)
            caption.setWordWrap(True)
            caption.setStyleSheet('color: palette(mid);')
            layout.addWidget(caption)

        self.form = QFormLayout()
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addLayout(self.form)

        for spec in _ordered_fields(section, self.override):
            editor = build_editor(spec, self)
            editor.changed.connect(self._on_field_changed)
            self.editors[spec.key] = editor

            label = QLabel(_label_for(spec, self.override), self)
            field_override = self.override.fields.get(spec.key) if self.override else None
            tooltip = (field_override.help if field_override and field_override.help
                       else spec.doc)
            if tooltip:
                label.setToolTip(tooltip)
                editor.setToolTip(tooltip)
            self.form.addRow(label, editor)

        self.load()

    # ---------------------------------------------------------------- state

    def load(self) -> None:
        """Populate every editor from the document."""
        self._loading = True
        try:
            for key, editor in self.editors.items():
                editor.set_value(self.document.get(self.section.key, key))
        finally:
            self._loading = False

    def commit(self) -> None:
        """Write the editors' values back to the document.

        A key absent from the config is left absent unless the user actually
        gave it a value. Writing every editor unconditionally would invent
        entries -- most visibly for booleans, whose editors always report True
        or False and so would turn "not specified" into an explicit False.
        """
        if not self.document.has_section(self.section.key):
            return

        existing = self.document.values.get(self.section.key, {})
        specs = self.section.field_map()

        for key, editor in self.editors.items():
            value = editor.value()
            if key not in existing:
                spec = specs.get(key)
                if value is None or (spec is not None and value == spec.default):
                    continue
            self.document.set(self.section.key, key, value)

    def show_errors(self, issues) -> None:
        """Decorate the fields named by validation issues."""
        for editor in self.editors.values():
            editor.set_error(None)
        for issue in issues:
            if issue.section == self.section.key and issue.key in self.editors:
                self.editors[issue.key].set_error(issue.message)

    # --------------------------------------------------------------- events

    def _on_field_changed(self) -> None:
        if self._loading:
            return
        self.commit()
        self.changed.emit()

    def _on_toggled(self, enabled: bool) -> None:
        if self._loading:
            return
        self.document.enable_section(self.section.key, enabled)
        if enabled:
            self.load()
        self.section_toggled.emit(self.section.key, enabled)
        self.changed.emit()


class ConfigForm(QWidget):
    """Every section of a config."""

    changed = Signal()

    def __init__(self, document: ConfigDocument,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.document = document
        self.sections: Dict[str, SectionForm] = {}

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.rebuild()

    def rebuild(self) -> None:
        """Recreate every section form, e.g. after the model changed."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.sections.clear()

        for section in self.document.sections():
            form = SectionForm(section, self.document, self)
            form.changed.connect(self.changed)
            self.sections[section.key] = form
            self._layout.addWidget(form)

        self._layout.addStretch(1)

    def commit(self) -> None:
        for form in self.sections.values():
            form.commit()

    def load(self) -> None:
        for form in self.sections.values():
            form.load()

    def show_errors(self, issues) -> None:
        """Route validation issues to the sections that own them."""
        for form in self.sections.values():
            form.show_errors(issues)
