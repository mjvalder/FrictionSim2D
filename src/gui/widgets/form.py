"""Assembling editors into a form.

One group box per config section, one row per field, ordered and labelled from
the override table. Optional sections carry a checkbox that adds or removes the
whole section -- which is how the models express "no substrate", "no flake": an
absent section, not a flag.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ...core.material_templates import has_template_marker
from ..controllers.document import ConfigDocument
from ..controllers.elements import scan_config_elements
from ..schema.introspect import FieldKind, FieldSpec, SectionSpec
from ..schema.overrides import (
    ENABLED,
    SET,
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
                 parent: Optional[QWidget] = None,
                 settings: Optional[object] = None) -> None:
        overrides = overrides_for(document.model)
        self.override = overrides.get(section.key)
        title = (self.override.label if self.override and self.override.label
                 else _prettify(section.key))

        super().__init__(title, parent)
        self.section = section
        self.document = document
        #: A `GlobalSettings`, needed only by fields whose requiredness
        #: depends on a Settings choice rather than another field in this
        #: same section (that is what `enable_if` is for) -- e.g. the driving
        #: spring only matters when the virtual-atom drive method is
        #: selected. None on the Settings tab's own form, which has no
        #: settings to consult.
        self.settings = settings
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
        #: One checkbox per gate declared in the section override (e.g. "Set
        #: tip position"), keyed by gate name. Ticking it shows the gated
        #: fields and lets them commit; unticking hides them and drops their
        #: keys from the document.
        self._gate_boxes: Dict[str, QCheckBox] = {}
        #: Form-layout row each gate's checkbox was inserted at, recorded
        #: while adding fields below -- the checkbox belongs right before the
        #: first field it gates, not tacked on at the end of the section.
        self._gate_first_row: Dict[str, int] = {}
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
            self.pair_editor.set_base_dir(document.base_dir)
            body_layout.addWidget(self.pair_editor)

        hidden = hidden_fields(document.model, section.key)
        for spec in _ordered_fields(section, self.override, hidden):
            self._add_field(spec)
        self._insert_material_list_row()
        self._build_gates()
        self._wire_material_sweep()

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
            editor=(field_override.editor if field_override else None),
            choice_labels=(field_override.choice_labels if field_override else None))
        editor.changed.connect(self._on_field_changed)
        self.editors[spec.key] = editor

        # Required fields are marked, so "which of these must I fill in?" is
        # answerable by looking rather than by pressing Render.
        text = _label_for(spec, self.override)
        required = self.effective_required(spec.key)
        label = QLabel(f"{text}{REQUIRED_MARK}" if required else text, self)
        if required:
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

        if field_override and field_override.gate:
            self._gate_first_row.setdefault(field_override.gate, self.form.rowCount())

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

    def _insert_material_list_row(self) -> None:
        """Slot the tick list in as its own row, right after `mat`.

        `materials_list`'s own row (added by the ordered-fields loop above,
        at the override's order=0 slot) holds only MaterialListEditor's
        checkbox -- the .txt picker and tick list are `list_panel`, a
        second widget the editor builds but never adds to its own layout
        (see materials.py). Inserting it here, after every field has its
        row, is what lets the checkbox stay first (so it visibly governs
        cif_path/pot_type/pot_path below it) while the list itself sits
        after `mat`, per the user's chosen layout -- one FieldSpec, two
        rows, without `_ordered_fields`' single (order, key) sort needing to
        know about it.
        """
        material_editor = self.editors.get('materials_list')
        if self.section.key != '2D' or not hasattr(material_editor, 'list_panel'):
            return
        mat_label, _mat_cell = self._rows.get('mat', (None, None))
        if mat_label is None:
            return
        row, _role = self.form.getWidgetPosition(mat_label)
        if row < 0:
            return
        self.form.insertRow(row + 1, material_editor.list_panel)

    def _build_gates(self) -> None:
        """One checkbox per declared gate, inserted just above the first
        field it controls rather than appended after every field is in."""
        if not self.override or not self.override.gates:
            return
        # Highest row first: inserting low-to-high would shift a
        # not-yet-inserted gate's recorded row out from under it.
        ordered = sorted(
            self.override.gates.items(),
            key=lambda item: self._gate_first_row.get(item[0], -1),
            reverse=True,
        )
        for gate_key, label in ordered:
            row = self._gate_first_row.get(gate_key)
            if row is None:
                continue
            checkbox = QCheckBox(label, self)
            checkbox.toggled.connect(
                lambda _checked=False, g=gate_key: self._on_gate_toggled(g))
            self.form.insertRow(row, checkbox)
            self._gate_boxes[gate_key] = checkbox

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
            key for key in self.section.field_map()
            if key in self.editors and self.effective_required(key)
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

    def effective_required(self, key: str) -> bool:
        """Whether `key` must be filled in, resolving the settings-dependent
        case: a `required_if_settings` override wins over a plain `required`
        override, which in turn wins over the model's own required flag."""
        fo = self.override.fields.get(key) if self.override else None
        spec = self.section.field_map().get(key)
        if fo and fo.required_if_settings and self.settings is not None:
            path, expected = fo.required_if_settings
            value = self.settings
            for part in path.split('.'):
                value = getattr(value, part, None)
            return value == expected
        if fo and fo.required is not None:
            return fo.required
        return bool(spec and spec.required)

    def set_settings(self, settings: Optional[object]) -> None:
        """Re-derive every row's star in place, e.g. after Settings changes
        the drive method. Rebuilding the whole form would work too, but would
        also discard whatever the user has open/collapsed/typed elsewhere.

        Also re-runs the required-field prefill (`load()`'s, extracted into
        `_prefill_required`): a field that becomes required only because of
        this settings change (e.g. `driving_spring` once `drive_method`
        flips to `virtual_atom`) would otherwise sit starred-and-blank,
        blocking Generate until the user notices and fills it in by hand --
        contradicting documentation/gui.md's promise that a starred field
        with a default is pre-filled with it. Wrapped in `_loading` so the
        prefill's own `editor.set_value` calls do not fire a commit here;
        the next real commit (validation, Render, Generate) picks the
        prefilled value up normally.
        """
        self.settings = settings
        for key in self.editors:
            spec = self.section.field_map().get(key)
            if spec is None:
                continue
            label, _cell = self._rows.get(key, (None, None))
            if label is None:
                continue
            text = _label_for(spec, self.override)
            if self.effective_required(key):
                label.setText(f"{text}{REQUIRED_MARK}")
            else:
                label.setText(text)

        self._loading = True
        try:
            self._prefill_required()
        finally:
            self._loading = False

    # ---------------------------------------------------------------- state

    def load(self) -> None:
        """Populate every editor from the document."""
        self._loading = True
        try:
            for key, editor in self.editors.items():
                editor.set_value(self.document.get(self.section.key, key))
            if self.pair_editor is not None:
                self.pair_editor.set_base_dir(self.document.base_dir)
                self.pair_editor.set_known_elements(
                    scan_config_elements(self.document))
                self.pair_editor.set_value(
                    self.document.values.get(self.section.key) or {})

            material_editor = self.editors.get('materials_list')
            if self.section.key == '2D' and hasattr(material_editor, 'is_sweeping'):
                # The generic loop above has already put the document's raw
                # cif_path/pot_path text into their editors, so
                # _sweep_directories() (reading those editors) sees exactly
                # what a freshly opened config carries -- templated or not.
                names = self.document.material_names()
                material_editor.set_document_names(names)
                material_editor.rescan(*self._sweep_directories())

                # Whether the box is ticked is decided by whether a sweep is
                # actually configured, not by whether the paths happen to be
                # templated -- rescan() only populates the listing. A {mat}
                # config with no materials_list is unfinished, not swept, and
                # must not have the box invented for it.
                material_editor.set_sweeping(self.document.has_material_list())

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

                self._apply_sweep_mode()

            self._load_gates()
            self._prefill_required()
        finally:
            self._loading = False
        self._apply_dependencies()

    def _gate_is_open(self, key: str) -> bool:
        """Whether `key`'s row is currently visible, i.e. not hidden behind
        an unticked gate checkbox. A field with no gate (the common case) is
        always open."""
        field_override = self.override.fields.get(key) if self.override else None
        if not field_override or not field_override.gate:
            return True
        box = self._gate_boxes.get(field_override.gate)
        return box is None or box.isChecked()

    def _prefill_required(self) -> None:
        """Fill every starred-but-empty editor with its spec default.

        A starred field left empty has no way to show the user what will
        actually be built without pre-filling it: unlike the ghost
        placeholder on an optional field, a required field's blank box
        blocks Render entirely, so the default has to actually land in
        the control (and, on commit, the document) rather than just be
        suggested. Run after `_load_gates()` so `_gate_is_open` sees each
        gate's real, just-loaded state.

        Skips a gated-but-invisible field: prefilling into a hidden editor
        would only have `commit()` pop it right back out (the gate is off),
        while `missing_required` still reported it as blocking with no
        visible row for the user to fix it on. Not reachable by any gated
        field today -- none is also `required=True` -- but cheap to guard
        against a future one that is.
        """
        specs = self.section.field_map()
        for key, editor in self.editors.items():
            spec = specs.get(key)
            if spec is None or spec.default is None:
                continue
            if not self.effective_required(key):
                continue
            if not self._gate_is_open(key):
                continue
            if editor.value() in (None, '', []):
                editor.set_value(spec.default)

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

            field_override = self.override.fields.get(key) if self.override else None
            if field_override and field_override.gate:
                gate_box = self._gate_boxes.get(field_override.gate)
                if gate_box is not None and not gate_box.isChecked():
                    # The gate is off: this key must not survive in the
                    # document, or a value typed before unticking (or one
                    # loaded from a hand-written config) would silently
                    # reappear on the next commit.
                    removed = self.document.values.get(
                        self.section.key, {}).pop(key, None)
                    spec = specs.get(key)
                    # Popping a value that just equals the field's own spec
                    # default is a no-op, not an edit: a hand-written config
                    # with an explicit `tip_x_offset = 0.0` (see
                    # `_load_gates`'s docstring) reads as untouched, ticks no
                    # gate, and would otherwise mark a freshly-opened,
                    # never-edited config dirty on the very first commit --
                    # which `_validate()` runs on every Render.
                    if removed is not None and (spec is None or removed != spec.default):
                        self.document.dirty = True
                    continue

            value = editor.value()
            if key not in existing:
                spec = specs.get(key)
                if value is None or (spec is not None and value == spec.default
                                     and not self.effective_required(key)):
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
            written = self.document.write_material_names(selected)
            # Read back what the document actually holds -- a relative path,
            # per _relative_to_config -- rather than reusing
            # write_material_names' return value, which is the absolute
            # target it wrote to disk. The two are not the same, and the
            # generic loop used to write whichever one note_written was
            # given straight back into the document.
            editor.note_written(self.document.get('2D', 'materials_list'))
            if written is None and selected:
                # None is ambiguous on its own: write_material_names also
                # returns it for an empty `selected` (a legitimate clear, not
                # a problem). Only a real, non-empty selection failing to
                # land anywhere -- an unsaved document with no workspace to
                # fall back to -- is the decline this warns about. The
                # document is unchanged either way; note_written above just
                # read that back, so the selection on screen has nothing
                # behind it until the user saves.
                self._warn_material_list_not_written()

        return True

    def _warn_material_list_not_written(self) -> None:
        """Tell the user why a ticked selection did not reach disk.

        Local import, like `MaterialListEditor._browse_list`'s QFileDialog:
        this module already depends on Qt throughout, so pulling the dialog
        class in here rather than routing a message through a new signal
        keeps the fix contained to the one place that needs it.
        """
        from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415

        QMessageBox.warning(
            self, 'Material list not saved',
            'This configuration has not been saved yet and no workspace is '
            'open, so there is nowhere to write the material list.\n\n'
            'Save the configuration first, then tick materials again.')

    # ----------------------------------------------------------- sweep mode

    def _sweep_directories(self) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
        """The (structures, potentials, extension) triple the [2D] path
        editors currently describe, resolved against the config's own
        directory.

        Reads the editors rather than the document: while sweeping, the
        cif_path/pot_path fields hold a bare directory (folder mode), not yet
        the `{mat}`-templated path only `apply_material_directories` writes on
        commit -- so a templated value's directory is its parent, and anything
        else is taken as a directory outright. The base_dir rejoin mirrors
        what `load()` used to do reading `document.material_directories()`: a
        relative 'cif' has to resolve against the config's own directory, not
        the process cwd, or it scans empty.
        """
        cif_editor = self.editors.get('cif_path')
        pot_editor = self.editors.get('pot_path')
        cif_text = (cif_editor.value() if cif_editor else None) or ''
        pot_text = (pot_editor.value() if pot_editor else None) or ''

        structures = self._directory_from_field(cif_text)
        potentials = self._directory_from_field(pot_text)
        extension = Path(pot_text).suffix if has_template_marker(pot_text) else None

        base = self.document.base_dir
        if base is not None:
            if structures is not None and not structures.is_absolute():
                structures = base / structures
            if potentials is not None and not potentials.is_absolute():
                potentials = base / potentials
        return structures, potentials, (extension or None)

    @staticmethod
    def _directory_from_field(text: str) -> Optional[Path]:
        text = (text or '').strip()
        if not text:
            return None
        return Path(text).parent if has_template_marker(text) else Path(text)

    def _apply_sweep_mode(self) -> None:
        """Folder pickers while sweeping, single-file pickers otherwise.

        The sweep state decides what Structure/Potential mean -- a directory
        to scan when ticked, a single file otherwise -- not whichever mode the
        widgets happened to be left in. The mat field is only ever set to
        '{mat}' by the post-apply resync in commit(), never here: ticking the
        box alone must not manufacture a {mat}-with-no-materials_list config.
        """
        material_editor = self.editors.get('materials_list')
        if self.section.key != '2D' or not hasattr(material_editor, 'is_sweeping'):
            return
        sweeping = material_editor.is_sweeping()
        for key in ('cif_path', 'pot_path'):
            path_editor = self.editors.get(key)
            if path_editor is not None and hasattr(path_editor, 'set_directory_mode'):
                path_editor.set_directory_mode(sweeping)
        mat_editor = self.editors.get('mat')
        if mat_editor is not None:
            mat_editor.setEnabled(not sweeping)

    def _wire_material_sweep(self) -> None:
        """Connect the sweep checkbox and the two path editors'
        editingFinished to rescans -- in the [2D] section only, and never
        through `changed`/`_on_field_changed`.

        MaterialListEditor.rescan() ends by emitting `changed`, and
        `_on_field_changed` is wired to every editor's `changed` (including
        this one) to commit the section. Triggering a rescan from
        `_on_field_changed` would therefore recurse forever: rescan ->
        changed -> _on_field_changed -> rescan -> ... These two connections
        are the only rescan triggers besides `load()` and a browse-dialog
        selection (which raises editingFinished itself, see _PathEditor).

        The materials_list `.txt` picker's own editingFinished is wired here
        too, to `_on_materials_path_finished` -- not a rescan (it reads a
        file, not the two directories), but the same rule applies: it must
        never run from `_on_field_changed`, or focusing away from the path
        box mid-edit would recommit on every keystroke's eventual
        editingFinished.
        """
        material_editor = self.editors.get('materials_list')
        if self.section.key != '2D' or not hasattr(material_editor, 'is_sweeping'):
            return
        material_editor.sweep_box.toggled.connect(self._on_sweep_box_toggled)
        for key in ('cif_path', 'pot_path'):
            path_editor = self.editors.get(key)
            if path_editor is not None:
                path_editor.edit.editingFinished.connect(self._on_sweep_paths_finished)
        material_editor.path_edit.editingFinished.connect(
            self._on_materials_path_finished)

    def _on_sweep_box_toggled(self, _checked: bool) -> None:
        self._apply_sweep_mode()
        if self._loading:
            # load() issues its own rescan once the document has been fully
            # read; a tick fired by set_sweeping() while loading must not
            # duplicate it with one built from a still-partially-applied form.
            return
        material_editor = self.editors.get('materials_list')
        if material_editor is not None and material_editor.is_sweeping():
            material_editor.rescan(*self._sweep_directories())

    def _on_materials_path_finished(self) -> None:
        """Point `materials_list` at whatever `.txt` the user just typed or
        browsed to, and load its names as the selection.

        Writes the document directly, like `apply_material_directories`
        does for cif_path/pot_path -- not through the generic per-field
        commit loop, which never touches materials_list at all (see
        `commit`'s docstring: it is always fully owned by
        `_commit_material_sweep`). Guarded against a same-text refire (a
        plain focus-out re-raises editingFinished without the text having
        changed) and against an unreadable path -- a typo mid-edit must not
        wipe out whatever was already ticked.
        """
        material_editor = self.editors.get('materials_list')
        if material_editor is None or not material_editor.is_sweeping():
            return
        text = material_editor.path_edit.text().strip()
        if not text or text == self.document.get('2D', 'materials_list'):
            return
        self.document.set('2D', 'materials_list', text)
        names = self.document.material_names()
        if names:
            material_editor.load_names(names)
        material_editor.note_written(text)

    def _on_sweep_paths_finished(self) -> None:
        material_editor = self.editors.get('materials_list')
        if material_editor is None or not material_editor.is_sweeping():
            return
        material_editor.rescan(*self._sweep_directories())

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

            if expected is ENABLED:
                enabled = bool(value)
            elif expected is SET:
                # Holds a value at all, as opposed to any truthy value: 0 /
                # 0.0 is a real scan angle (a straight-line scan), not "unset".
                enabled = value not in (None, '', [])
            else:
                enabled = value == expected
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

            # A greyed row with no on-screen explanation reads as a missing
            # feature (only the ⓘ bubble said why, and nothing prompts anyone
            # to hover it). Put the same reason in the control itself.
            if field_override.enable_hint:
                editor.set_disabled_hint(
                    None if enabled else field_override.enable_hint)

    def _load_gates(self) -> None:
        """Tick each gate iff a field it guards actually differs from its
        spec default.

        Presence alone cannot be the test: `_defaults_for` already keeps
        gated keys out of a blank config, but an *existing* config loaded
        from disk still carries whatever `_defaults_for` wrote when it was
        first created, plus anything hand-edited since. Comparing against
        the spec default is what tells "never touched" from "deliberately
        set to the default value" apart.

        One accepted consequence: a hand-written config with an explicit
        `tip_x_offset = 0.0` (equal to its own spec default) reads as
        untouched, ticks no gate, and loses that key on the next save. This
        is a no-op semantically -- the builder already treats an absent
        offset as 0.0 -- so the round trip changes nothing about how the
        config builds, only how it looks on disk.
        """
        if not self.override or not self.override.gates:
            return
        specs = self.section.field_map()
        for gate_key, checkbox in self._gate_boxes.items():
            gated_keys = [key for key, fo in self.override.fields.items()
                         if fo.gate == gate_key]
            on = False
            for key in gated_keys:
                spec = specs.get(key)
                value = self.document.get(self.section.key, key)
                if value not in (None, '', []) and (
                        spec is None or value != spec.default):
                    on = True
                    break
            checkbox.setChecked(on)
        self._apply_gates()

    def _apply_gates(self) -> None:
        """Show or hide each gate's fields to match its checkbox.

        `setRowVisible` (not hiding label and widget separately) is what
        keeps a hidden row from leaving its slice of form spacing behind.
        """
        if not self.override or not self.override.gates:
            return
        for gate_key, checkbox in self._gate_boxes.items():
            on = checkbox.isChecked()
            for key, field_override in self.override.fields.items():
                if field_override.gate != gate_key:
                    continue
                label, _cell = self._rows.get(key, (None, None))
                if label is None:
                    continue
                row, _role = self.form.getWidgetPosition(label)
                if row < 0:
                    continue
                self.form.setRowVisible(row, on)

    def _on_gate_toggled(self, _gate_key: str) -> None:
        self._apply_gates()
        self._on_field_changed()

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
                 start_collapsed: bool = False,
                 settings: Optional[object] = None) -> None:
        super().__init__(parent)
        self.document = document
        #: Forwarded to every `SectionForm`; None on the Settings tab's own
        #: form, which keeps passing nothing -- it has no settings of its own
        #: to consult.
        self.settings = settings
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
        self.setUpdatesEnabled(False)
        try:
            while self._layout.count():
                item = self._layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.sections.clear()

            for section in self.document.sections():
                form = SectionForm(section, self.document, self, settings=self.settings)
                form.changed.connect(self.changed)
                form.collapsed_changed.connect(self._remember_collapsed)
                self.sections[section.key] = form
                self._layout.addWidget(form)

                collapsed = self._collapsed.get(
                    section.key,
                    self.start_collapsed or _starts_collapsed(section, self.document))
                form.set_collapsed(collapsed)

            # A material change lives in a component section, not
            # [lj_override], but it changes which elements the pair combos
            # should offer. This must NOT go through `changed` -- that is
            # `_on_field_changed`'s per-keystroke commit signal (see
            # `_wire_material_sweep`'s docstring above for the same hazard
            # with rescan()), and scanning re-reads CIFs off disk. So this
            # listens for the same kind of settled-edit points rescan()
            # itself uses: `editingFinished` on the mat/cif_path text
            # editors, a materials-list change (MaterialListEditor's own
            # `changed`, emitted only from rescan()/a checkbox click, never a
            # keystroke), and a section being switched on or off.
            self._wire_lj_known_elements_refresh()

            self._layout.addStretch(1)
        finally:
            self.setUpdatesEnabled(True)

    #: Sections `scan_config_elements` actually reads; only these need to
    #: trigger a refresh.
    _LJ_ELEMENT_SECTIONS = ('2D', '2D-2', 'tip', 'sub', 'flake')

    def _wire_lj_known_elements_refresh(self) -> None:
        if 'lj_override' not in self.sections:
            return
        for key in self._LJ_ELEMENT_SECTIONS:
            form = self.sections.get(key)
            if form is None:
                continue
            for field_key in ('mat', 'cif_path'):
                edit_widget = getattr(form.editors.get(field_key), 'edit', None)
                if edit_widget is not None:
                    edit_widget.editingFinished.connect(
                        self._refresh_lj_known_elements)
            material_editor = form.editors.get('materials_list')
            if material_editor is not None:
                material_editor.changed.connect(self._refresh_lj_known_elements)
            form.section_toggled.connect(
                lambda _key, _enabled: self._refresh_lj_known_elements())

    def _refresh_lj_known_elements(self) -> None:
        section_form = self.sections.get('lj_override')
        pair_editor = section_form.pair_editor if section_form else None
        if pair_editor is not None:
            pair_editor.set_known_elements(scan_config_elements(self.document))

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

    def set_settings(self, settings: Optional[object]) -> None:
        """Fan a (possibly new) settings object out to every section, so a
        star that depends on a Settings choice moves without a rebuild."""
        self.settings = settings
        for form in self.sections.values():
            form.set_settings(settings)

    def missing_required(self) -> List[Tuple[str, str, str, bool]]:
        """Starred-but-empty fields, as ``(section, key, label, blocks)``.

        Only sections actually in play are checked: an optional section that
        is switched off has nothing to fill in, and its editors' values do
        not reflect the document (there is nothing there to reflect).
        """
        missing: List[Tuple[str, str, str, bool]] = []
        for section_key, form in self.sections.items():
            section_required = (not form.section.optional
                                or self.document.is_section_locked(section_key))
            if not (self.document.has_section(section_key) or section_required):
                continue
            field_map = form.section.field_map()
            for key, spec in field_map.items():
                if key not in form.editors or not form.effective_required(key):
                    continue
                if not form._gate_is_open(key):
                    continue
                if self.document.get(section_key, key) not in (None, '', []):
                    continue
                label = _label_for(spec, form.override)
                field_override = form.override.fields.get(key) if form.override else None
                blocks = field_override.required_blocks if field_override else True
                missing.append((section_key, key, label, blocks))
        return missing
