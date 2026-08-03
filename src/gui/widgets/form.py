"""Assembling editors into a form.

One group box per config section, one row per field, ordered and labelled from
the override table. Optional sections carry a checkbox that adds or removes the
whole section -- which is how the models express "no substrate", "no flake": an
absent section, not a flag.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ..controllers.document import ConfigDocument
from ..schema.introspect import FieldKind, FieldSpec, SectionSpec
from ..schema.overrides import (
    ENABLED,
    SectionOverride,
    hidden_fields,
    overrides_for,
)
from .fields import FieldEditor, build_editor
from .pairs import PairOverrideEditor
from ..style import CAPTION_SMALL, REQUIRED_MARK, SECTION_HELP

#: Sections that start folded: an optional section that is switched off has
#: nothing to show, and leaving it open is most of the clutter.
#: Names that capitalise() would mangle.
_ACRONYMS = {'hpc': 'HPC', 'aiida': 'AiiDA', 'lj': 'LJ', 'pes': 'PES'}

#: What a list in a field means, keyed by ConfigDocument.loopability_of().
#: Generated rather than restated per field, so this and the LOOPABILITY map
#: in core.run cannot drift apart.
LOOPABILITY_HINTS = {
    'sweep': 'A list runs each value as a separate simulation.',
    'layer': 'A list builds one L{n} tree per value.',
    'in_file': 'A list is looped inside one LAMMPS script.',
}


def loopability_hint(document, section: str, key: str) -> str:
    """What a list in this field would do, or '' when there is nothing to say.

    'scalar' is deliberately silent: a list that is actually present is already
    reported by `unsupported_lists`, and captioning every single-value field
    with a negative is noise. The settings document has no loopability concept,
    hence the getattr.
    """
    resolver = getattr(document, 'loopability_of', None)
    if resolver is None:
        return ''
    return LOOPABILITY_HINTS.get(resolver(section, key), '')


def _compose_tooltip(help_text: Optional[str], detail: Optional[str], doc: str,
                     hint: str, default: object) -> str:
    """The bubble/tooltip body: only the parts that are actually present.

    HTML so the primary explanation (override help, or the model's own
    docstring when there is no override) stands out from the rest; everything
    else is plain rows underneath it.
    """
    primary = help_text or doc
    parts = []
    if primary:
        parts.append(f"<b>{html.escape(primary)}</b>")
    if detail:
        parts.append(html.escape(detail))
    if hint:
        parts.append(f"<i>{html.escape(hint)}</i>")
    if default not in (None, '', [], {}, ()):
        parts.append(f"default: {html.escape(str(default))}")
    return '<br>'.join(parts)


def _prettify(key: str) -> str:
    """A readable title for a section that has no declared label.

    The settings groups are named by their model field -- 'hpc', 'quench' --
    and reading them raw in a heading is needlessly shabby.
    """
    lowered = key.strip().lower()
    if lowered in _ACRONYMS:
        return _ACRONYMS[lowered]
    return key.replace('_', ' ').strip().capitalize()


def _starts_collapsed(section: SectionSpec, document) -> bool:
    if not section.optional or document.is_section_locked(section.key):
        return False
    return not document.has_section(section.key)

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
                    override: Optional[SectionOverride],
                    hidden: Tuple[str, ...] = ()) -> List[FieldSpec]:
    """Fields in display order, minus any this model's templates never read."""
    def key(spec: FieldSpec):
        field_override = override.fields.get(spec.key) if override else None
        return (field_override.order if field_override else 100, spec.key)

    visible = [f for f in section.fields if f.key not in hidden]
    return sorted(visible, key=key)


class SectionForm(QGroupBox):
    """One config section."""

    changed = Signal()
    section_toggled = Signal(str, bool)
    collapsed_changed = Signal(str, bool)

    def __init__(self, section: SectionSpec, document: ConfigDocument,
                 parent: Optional[QWidget] = None) -> None:
        overrides = overrides_for(document.model)
        self.override = overrides.get(section.key)
        title = (self.override.label if self.override and self.override.label
                 else _prettify(section.key))

        super().__init__(title, parent)
        self.section = section
        self.document = document
        self.editors: Dict[str, FieldEditor] = {}
        #: The ⓘ button at the end of a row, keyed by field. Only present when
        #: there is something to say: help, detail, the model's own doc, a
        #: loopability hint, a default, or an `enable_if` reason.
        self._bubbles: Dict[str, QToolButton] = {}
        #: The composed tooltip text as built in `_add_field` -- override help
        #: (or `spec.doc`), detail, and any loopability hint. `_apply_dependencies`
        #: needs this to restore an enabled field's tooltip; recomputing it from
        #: the override alone would drop the hint.
        self._composed_tooltips: Dict[str, str] = {}
        self._rows: Dict[str, tuple] = {}
        self._loading = False
        self._collapsed = False

        locked = document.is_section_locked(section.key)
        if section.optional and not locked:
            self.setCheckable(True)
            self.setChecked(document.has_section(section.key))
            self.toggled.connect(self._on_toggled)
        elif locked:
            self.setToolTip(document.section_lock_reason(section.key) or '')

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_header())

        # Everything below the header lives in one widget so the section can be
        # folded away: with nine settings groups and as many config sections,
        # the window is mostly scrolling otherwise.
        self.body = QWidget(self)
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.body)

        help_text = self.override.help if self.override else None
        if help_text:
            caption = QLabel(help_text, self.body)
            caption.setWordWrap(True)
            caption.setStyleSheet(SECTION_HELP)
            if self.override.detail:
                caption.setToolTip(self.override.detail)
            body_layout.addWidget(caption)

        self.form = QFormLayout()
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        body_layout.addLayout(self.form)

        # A free-form section has user-chosen keys rather than model fields, so
        # introspection yields nothing to render and it would otherwise appear
        # as an empty box.
        self.pair_editor: Optional[PairOverrideEditor] = None
        if section.key == 'lj_override':
            self.pair_editor = PairOverrideEditor(self.body)
            self.pair_editor.changed.connect(self._on_field_changed)
            body_layout.addWidget(self.pair_editor)

        hidden = hidden_fields(document.model, section.key)
        for spec in _ordered_fields(section, self.override, hidden):
            self._add_field(spec)

        self.load()
        self._apply_dependencies()

    def _add_field(self, spec: FieldSpec) -> None:
        """One labelled row: control, required marker, and an info bubble."""
        if spec.kind is FieldKind.NESTED:
            self._add_nested_row(spec)
            return

        field_override = self.override.fields.get(spec.key) if self.override else None

        editor = build_editor(
            spec, self,
            choices=(field_override.choices if field_override else None),
            editor=(field_override.editor if field_override else None))
        editor.changed.connect(self._on_field_changed)
        self.editors[spec.key] = editor

        # Required fields are marked, so "which of these must I fill in?" is
        # answerable by looking rather than by pressing Render.
        text = _label_for(spec, self.override)
        label = QLabel(f"{text}{REQUIRED_MARK}" if spec.required else text, self)
        if spec.required:
            label.setToolTip('Required')

        help_text = field_override.help if field_override else None
        detail_text = field_override.detail if field_override else None
        hint = loopability_hint(self.document, self.section.key, spec.key)
        tooltip_text = _compose_tooltip(help_text, detail_text, spec.doc, hint, spec.default)
        enable_hint = field_override.enable_hint if field_override else None

        cell = QWidget(self)
        cell_layout = QHBoxLayout(cell)
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(4)
        cell_layout.addWidget(editor, 1)

        # A bubble exists whenever there is something to show -- either the
        # composed tooltip, or (a hint-only field like twist_strain_tol) just
        # an enable_if reason for when the field is disabled.
        if tooltip_text or enable_hint:
            bubble = QToolButton(cell)
            bubble.setText('\N{CIRCLED LATIN SMALL LETTER I}')
            bubble.setAutoRaise(True)
            bubble.setToolTip(tooltip_text)
            bubble.clicked.connect(
                lambda _checked=False, b=bubble: QToolTip.showText(QCursor.pos(), b.toolTip(), b))
            cell_layout.addWidget(bubble)
            self._bubbles[spec.key] = bubble
            self._composed_tooltips[spec.key] = tooltip_text

        if tooltip_text:
            label.setToolTip(tooltip_text)
            # Not editor.setToolTip() directly: set_error(None) -- called on
            # every editor by show_errors(), including when there is nothing
            # to report -- would otherwise blank it right back out.
            editor.set_base_tooltip(tooltip_text)

        self._rows[spec.key] = (label, cell)
        self.form.addRow(label, cell)

    def _add_nested_row(self, spec: FieldSpec) -> None:
        """A nested sub-model, shown but not editable.

        Rendering it through the text fallback would stringify a dict and
        destroy it on the next commit. It is deliberately kept out of
        `editors`, which `commit` and `load` iterate, so the value passes
        through untouched -- but omitting the row entirely made the field
        indistinguishable from one that does not exist.
        """
        text = _label_for(spec, self.override)
        label = QLabel(text, self)
        label.setEnabled(False)

        caption = QLabel('Nested settings — passed through unchanged; '
                         'edit in the YAML.', self)
        caption.setWordWrap(True)
        caption.setStyleSheet(CAPTION_SMALL)
        caption.setEnabled(False)

        self._rows[spec.key] = (label, caption)
        self.form.addRow(label, caption)

    def _build_header(self) -> QHBoxLayout:
        """The fold control and the summary shown when folded."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        self.fold_button = QToolButton(self)
        self.fold_button.setAutoRaise(True)
        self.fold_button.setArrowType(Qt.DownArrow)
        self.fold_button.setToolTip('Collapse this section')
        self.fold_button.clicked.connect(self.toggle_collapsed)
        row.addWidget(self.fold_button)

        # Only visible when collapsed: something to decide by without unfolding.
        self.summary = QLabel('', self)
        self.summary.setStyleSheet(CAPTION_SMALL)
        self.summary.hide()
        row.addWidget(self.summary)

        row.addStretch(1)
        return row

    # ------------------------------------------------------------- collapsing

    def is_collapsed(self) -> bool:
        # Tracked rather than read back from the widget: isVisible() is False
        # for everything until the top-level window is shown, so deriving the
        # state from it reports every section as collapsed before that.
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Fold or unfold the section body."""
        self._collapsed = collapsed
        self.body.setVisible(not collapsed)
        self.fold_button.setArrowType(Qt.RightArrow if collapsed else Qt.DownArrow)
        self.fold_button.setToolTip(
            'Expand this section' if collapsed else 'Collapse this section')
        self.summary.setVisible(collapsed)
        if collapsed:
            self.summary.setText(self.summarise())
        self.collapsed_changed.emit(self.section.key, collapsed)

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self.is_collapsed())

    def summarise(self) -> str:
        """One line describing the section's state, for when it is folded."""
        if not self.document.has_section(self.section.key):
            return 'not included'

        values = self.document.values.get(self.section.key) or {}
        if self.pair_editor is not None:
            return f"{len(values)} override(s)"

        filled = sum(
            1 for key in self.editors
            if values.get(key) not in (None, '', [])
        )
        missing = [
            key for key, spec in self.section.field_map().items()
            if spec.required and key in self.editors
            and values.get(key) in (None, '', [])
        ]
        text = f"{filled} of {len(self.editors)} set"
        if missing:
            text += f" — {len(missing)} required still empty"
        return text

    def _is_required(self) -> bool:
        """Whether this section must be present for the model to build."""
        return (not self.section.optional
                or self.document.is_section_locked(self.section.key))

    # ---------------------------------------------------------------- state

    def load(self) -> None:
        """Populate every editor from the document."""
        self._loading = True
        try:
            for key, editor in self.editors.items():
                editor.set_value(self.document.get(self.section.key, key))
            if self.pair_editor is not None:
                self.pair_editor.set_value(
                    self.document.values.get(self.section.key) or {})

            material_editor = self.editors.get('materials_list')
            if self.section.key == '2D' and hasattr(material_editor, 'is_sweeping'):
                structures, potentials, extension = self.document.material_directories()
                if structures and potentials:
                    # material_directories() gives back paths as stored -- relative
                    # to the config's own directory, per _relative_to_config -- so
                    # they have to be rejoined against base_dir before scanning the
                    # actual filesystem, or a relative 'cif' resolves against the
                    # process cwd instead of the run directory and scans empty.
                    base = self.document.base_dir
                    if base is not None:
                        structures = structures if structures.is_absolute() \
                            else base / structures
                        potentials = potentials if potentials.is_absolute() \
                            else base / potentials
                    material_editor.set_directories(structures, potentials, extension)

                # Whether the box is ticked is decided by whether a sweep is
                # actually configured, not by whether the paths happen to be
                # templated -- set_directories only populates the pickers. A
                # {mat} config with no materials_list is unfinished, not
                # swept, and must not have the box invented for it.
                material_editor.set_sweeping(self.document.has_material_list())

                names = self.document.material_names()
                if names:
                    # An existing list is the selection; the directories
                    # only decide what is available to tick.
                    material_editor.set_selected(names)
                elif self.document.has_material_list():
                    # Configured but unreadable -- missing file, empty file,
                    # bad path. Leaving the default "everything usable"
                    # selection in place is how a typo'd path becomes a full
                    # sweep the moment anything is committed; an empty
                    # selection matches what is actually on disk (nothing)
                    # and commits as a no-op instead.
                    material_editor.set_selected([])
        finally:
            self._loading = False
        self._apply_dependencies()

    def commit(self) -> None:
        """Write the editors' values back to the document.

        A key absent from the config is left absent unless the user actually
        gave it a value. Writing every editor unconditionally would invent
        entries -- most visibly for booleans, whose editors always report True
        or False and so would turn "not specified" into an explicit False.

        The `[2D]` material sweep is handled before the generic loop, not
        after: `_commit_material_sweep` writes materials_list and, when a
        sweep is actually configured, the mat/cif_path/pot_path placeholders,
        straight into the document -- bypassing their editors, which still
        hold whatever text was on screen before. Running the generic loop
        afterwards used to let it re-read that stale editor text and write it
        right back over what the sweep logic had just decided, which both
        discarded same-commit edits to those fields and replaced a correctly
        *relative* materials_list with the absolute path note_written() had
        been (wrongly) given. materials_list itself is excluded from the
        generic loop entirely below, for the same reason: it is always fully
        owned by _commit_material_sweep.
        """
        if not self.document.has_section(self.section.key):
            # A section that cannot be switched off must exist before anything
            # can be written to it; otherwise the user fills the substrate in
            # and every value is silently dropped.
            if not self._is_required():
                return
            self.document.enable_section(self.section.key, True)

        if self.pair_editor is not None:
            # A free-form section *is* its values; replace it wholesale.
            self.document.values[self.section.key] = self.pair_editor.value()
            self.document.dirty = True
            return

        material_editor = self.editors.get('materials_list')
        is_material_section = (self.section.key == '2D'
                               and hasattr(material_editor, 'is_sweeping'))

        if is_material_section:
            wrote_to_document = self._commit_material_sweep(material_editor)
            if wrote_to_document:
                # Keep the owned editors in sync with what the document now
                # holds. Gated on whether the document actually changed this
                # call, not on the box being ticked: an unresolved sweep (box
                # ticked, no directories chosen yet) must leave
                # mat/cif_path/pot_path exactly as editable as any other
                # field, or a same-commit edit to one of them is silently
                # reverted before it ever reaches the document.
                for key in self.document.TEMPLATED_KEYS:
                    templated_editor = self.editors.get(key)
                    if templated_editor is not None:
                        templated_editor.set_value(
                            self.document.get(self.section.key, key))

        existing = self.document.values.get(self.section.key, {})
        specs = self.section.field_map()

        for key, editor in self.editors.items():
            if is_material_section and key == 'materials_list':
                continue
            value = editor.value()
            if key not in existing:
                spec = specs.get(key)
                if value is None or (spec is not None and value == spec.default):
                    continue
            self.document.set(self.section.key, key, value)

    def _commit_material_sweep(self, editor) -> bool:
        """Write the three things a sweep needs, when they actually changed.

        Unconditional writes are dangerous here: this runs on every commit of
        the section, including commits triggered by unrelated fields.

        Returns:
            Whether this call wrote (or cleared) something in the document --
            i.e. whether mat/cif_path/pot_path may have just changed under
            their editors' feet and need resyncing.
        """
        if not editor.is_sweeping():
            wrote = False
            if self.document.has_material_list():
                self.document.write_material_names([])
                editor.note_written(None)
                # Unticking must not leave the config invalid:
                # apply_material_directories' placeholders are only
                # meaningful with a sweep to substitute them, so they come
                # out with it. Nested inside has_material_list(), not called
                # unconditionally: this branch runs on *every* commit while
                # the box is unticked, including one triggered by an
                # unrelated keystroke on a config that was never a sweep at
                # all -- examples/sheet_config.ini ships with {mat} in all
                # three fields and no materials_list, and clearing them on
                # sight would destroy a documented, shipped configuration
                # form on the first commit.
                if self.document.clear_material_directories():
                    wrote = True
            return wrote

        structures, potentials, extension = editor.directories()
        if not (structures and potentials and extension):
            return False

        if not editor.has_usable_names():
            # Nothing usable was found on disk. scan_material_directories
            # reports orphans too -- names present in only one directory --
            # so a non-empty listing does not mean anything here is usable:
            # when only one of the two directories resolves (a moved run
            # directory, an unmounted share), the listing is full of
            # unusable orphans from the one that did. Treating that as a
            # real, empty selection would delete a working sweep the scan
            # simply cannot see right now.
            return False

        self.document.apply_material_directories(
            structures, potentials, extension)

        selected = editor.selected_names()
        # Order-insensitive: selected_names() comes back in
        # scan_material_directories' alphabetical order, but a hand-maintained
        # .txt is under no obligation to be sorted. Comparing as lists would
        # treat that as a change and retarget a file that nothing actually
        # asked to change.
        if set(selected) != set(self.document.material_names()):
            self.document.write_material_names(selected)
            # Read back what the document actually holds -- a relative path,
            # per _relative_to_config -- rather than reusing
            # write_material_names' return value, which is the absolute
            # target it wrote to disk. The two are not the same, and the
            # generic loop used to write whichever one note_written was
            # given straight back into the document.
            editor.note_written(self.document.get('2D', 'materials_list'))

        return True

    def show_errors(self, issues) -> None:
        """Decorate the fields named by validation issues."""
        for editor in self.editors.values():
            editor.set_error(None)
        for issue in issues:
            if issue.section == self.section.key and issue.key in self.editors:
                self.editors[issue.key].set_error(issue.message)

        for editor in self.editors.values():
            if editor.error() is None and editor.warning():
                editor.set_error(editor.warning())

    # --------------------------------------------------------------- events

    def _apply_dependencies(self) -> None:
        """Grey out controls whose prerequisite is not set.

        A disabled control says "this does nothing yet" plainly; a live one
        whose value is quietly ignored says nothing at all.
        """
        if not self.override:
            return

        for key, editor in self.editors.items():
            field_override = self.override.fields.get(key)
            if not field_override or not field_override.enable_if:
                continue

            other_key, expected = field_override.enable_if
            other = self.editors.get(other_key)
            value = other.value() if other else self.document.get(
                self.section.key, other_key)

            enabled = bool(value) if expected is ENABLED else value == expected
            editor.setEnabled(enabled)

            label, _cell = self._rows.get(key, (None, None))
            if label is not None:
                label.setEnabled(enabled)
            bubble = self._bubbles.get(key)
            if bubble is not None and field_override.enable_hint:
                # The composed tooltip carries the loopability hint, which the
                # override alone does not -- recomputing from `help` drops it.
                bubble.setToolTip(
                    field_override.enable_hint if not enabled
                    else self._composed_tooltips.get(key, ''))

    def _on_field_changed(self) -> None:
        if self._loading:
            return
        self.commit()
        self._apply_dependencies()
        if self.is_collapsed():
            self.summary.setText(self.summarise())
        self.changed.emit()

    def _on_toggled(self, enabled: bool) -> None:
        if self._loading:
            return
        self.document.enable_section(self.section.key, enabled)
        if enabled:
            self.load()
        # Switching a section on is a request to work on it.
        if enabled and self.is_collapsed():
            self.set_collapsed(False)
        elif self.is_collapsed():
            self.summary.setText(self.summarise())
        self.section_toggled.emit(self.section.key, enabled)
        self.changed.emit()


class ConfigForm(QWidget):
    """Every section of a config."""

    changed = Signal()

    def __init__(self, document: ConfigDocument,
                 parent: Optional[QWidget] = None,
                 start_collapsed: bool = False) -> None:
        super().__init__(parent)
        self.document = document
        #: Open every section folded. Used for settings, where all nine groups
        #: are populated and none demands attention.
        self.start_collapsed = start_collapsed
        self.sections: Dict[str, SectionForm] = {}
        #: Fold state survives a rebuild -- switching mode should not reopen
        #: everything the user just tidied away.
        self._collapsed: Dict[str, bool] = {}

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
            form.collapsed_changed.connect(self._remember_collapsed)
            self.sections[section.key] = form
            self._layout.addWidget(form)

            collapsed = self._collapsed.get(
                section.key,
                self.start_collapsed or _starts_collapsed(section, self.document))
            form.set_collapsed(collapsed)

        self._layout.addStretch(1)

    # ------------------------------------------------------------- collapsing

    def _remember_collapsed(self, key: str, collapsed: bool) -> None:
        self._collapsed[key] = collapsed

    def set_all_collapsed(self, collapsed: bool) -> None:
        """Fold or unfold every section."""
        for form in self.sections.values():
            form.set_collapsed(collapsed)

    def collapsed_sections(self) -> List[str]:
        return sorted(k for k, f in self.sections.items() if f.is_collapsed())

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
