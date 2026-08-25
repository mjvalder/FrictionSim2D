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

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ...core.material_templates import has_template_marker
from ..controllers.document import ConfigDocument, DocumentError
from ..controllers.elements import scan_config_elements
from ..schema.introspect import FieldKind, FieldSpec, SectionSpec
from ..schema.overrides import (
    ENABLED,
    SET,
    SectionOverride,
    field_override,
    hidden_fields,
    override_for_section,
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
#: in run.naming cannot drift apart.
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
    key_override = override.fields.get(spec.key) if override else None
    if key_override and key_override.label:
        label = key_override.label
    else:
        label = spec.key.replace('_', ' ').strip().capitalize()

    if key_override and key_override.unit:
        label = f"{label} ({key_override.unit})"
    return label


def _refusal_line(section_key: str, exc: Exception) -> str:
    """One section's refusal, ready to show, whichever layer refused it.

    Two things can refuse a commit and both are the user's to see:

    * `DocumentError` -- the document cannot represent what the form is
      holding. Its message is already a sentence written for the user.
    * `OSError` -- the material list is the one part of an edit that reaches
      disk as it is made (`ConfigDocument.write_material_names`), so it is
      the one part the filesystem can refuse: a read-only config directory,
      a share that dropped, a full disk. It arrives as an errno, which says
      nothing about what the user was doing, so it is given a sentence here.

    The directory permission is worth naming. The list is written to a
    sibling temp file and renamed into place (`_write_atomically`), so the
    write now needs permission to create a file *in the directory* -- the
    in-place truncation it replaced only needed the list file writable, and
    "but the file is writable" is otherwise a fair objection.
    """
    if isinstance(exc, DocumentError):
        return f"[{section_key}] {exc}"
    return (
        f"[{section_key}] the material list could not be written ({exc}). "
        "The sweep on screen is not saved anywhere. The list goes next to "
        "the config as a temporary file that is then renamed into place, so "
        "the config's directory has to be writable, not only the list file."
    )


def _ordered_fields(section: SectionSpec,
                    override: Optional[SectionOverride],
                    hidden: Tuple[str, ...] = ()) -> List[FieldSpec]:
    """Fields in display order, minus any this model's templates never read."""
    def key(spec: FieldSpec):
        key_override = override.fields.get(spec.key) if override else None
        return (key_override.order if key_override else 100, spec.key)

    visible = [f for f in section.fields if f.key not in hidden]
    return sorted(visible, key=key)


#: The `[2D]` button's two states. Adding a second material is a change to the
#: *shape* of the config, not to a value, which is why it is a button and not
#: a field: it adds a whole section.
ADD_MATERIAL = 'Add a material'
REMOVE_MATERIAL = 'Remove material 2'


class SectionForm(QGroupBox):
    """One config section."""

    changed = Signal()
    section_toggled = Signal(str, bool)
    collapsed_changed = Signal(str, bool)
    #: The set of sections changed -- a second material was added or dropped --
    #: so the whole form has to be rebuilt, not merely reloaded. Distinct from
    #: `changed`, which reports a value edit and never moves a section.
    structure_changed = Signal()
    #: `[2D]` only: the material sweep was ticked or unticked. The second
    #: sheet's path cells follow it into (and out of) folder mode, and only
    #: `ConfigForm` owns both sections.
    sweep_mode_changed = Signal(bool)
    #: `[2D]` only: this commit rewrote the `{mat}` placeholders, which for a
    #: pair sweep includes `[2D-2]`'s -- straight into the document, behind
    #: its editors' backs. They have to be resynced before that section
    #: commits, or it writes its stale text back over them.
    templates_written = Signal()
    #: The document refused this section's commit. Carries a message ready to
    #: show; deliberately NOT a dialog, see `_on_field_changed`.
    refused = Signal(str)

    def __init__(self, section: SectionSpec, document: ConfigDocument,
                 parent: Optional[QWidget] = None,
                 settings: Optional[object] = None) -> None:
        # The same resolution `ConfigDocument.comments()` uses to write this
        # text back into the file -- one rule, so the two cannot drift.
        self.override = override_for_section(document.model, section.key)
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

        #: The Add/Remove control for the second material -- `[2D]` only.
        self.add_material_button: Optional[QPushButton] = None
        #: The `hetero_stacking` proxy -- `[2D-2]` only. Deliberately not a
        #: `FieldEditor` in `self.editors`: it writes to `[general]`.
        self.stacking_box: Optional[QComboBox] = None

        hidden = hidden_fields(document.model, section.key)
        for spec in _ordered_fields(section, self.override, hidden):
            self._add_field(spec)
        self._insert_material_list_row()
        self._build_gates()
        self._wire_material_sweep()
        self._build_stacking_proxy()
        self._build_material_button(body_layout)

        self.load()
        self._apply_dependencies()

    # ------------------------------------------------------ second material

    def _build_material_button(self, body_layout: QVBoxLayout) -> None:
        """The `[2D]` section's own Add/Remove control for the second material.

        It sits under the sheet rather than beside the family/mode selectors
        because a heterostructure is a property of the sheet, not of the run --
        every mode builds one.

        Always enabled, including while a material sweep is configured. The
        pair sweep (one ordered pair per list line) is precisely the case that
        needs a second sheet, so excluding it while sweeping would make that
        configuration unreachable from the form.
        """
        if self.section.key != '2D' or not hasattr(self.document, 'has_second_material'):
            return
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.add_material_button = QPushButton(self.body)
        self.add_material_button.clicked.connect(self._on_material_button_clicked)
        row.addWidget(self.add_material_button)
        row.addStretch(1)
        body_layout.addLayout(row)
        self._update_material_button()

    def _update_material_button(self) -> None:
        if self.add_material_button is None:
            return
        second = self.document.has_second_material()
        self.add_material_button.setText(
            REMOVE_MATERIAL if second else ADD_MATERIAL)
        self.add_material_button.setToolTip(
            'Drop the second material and build a single-material stack.'
            if second else
            'Stack a second material with this one to build a heterostructure.')

    def _on_material_button_clicked(self) -> None:
        """Add or drop the second sheet, then ask the form to rebuild.

        `commit()` runs first, and that ordering is the whole point.
        `add_second_material` seeds the new sheet's `{mat2}` by reading the
        placeholders back off the *document*, so a sweep whose directories have
        only reached the editors so far would leave `[2D-2]` untemplated -- and
        an untemplated second sheet under a pair list is not a visible failure:
        expansion leaves it pinned and every member of the sweep builds the
        same second material, inside correctly-named heterostructure
        directories.

        The rebuild is left to `structure_changed`: this handler runs from a
        button that the rebuild deletes, so it cannot do it itself.

        Removing is the half that can refuse. A pair material list has to be
        collapsed to its first members, and that reaches disk immediately --
        unlike every other edit here, which the user can still throw away by
        closing without saving. So it is asked about first, and a failed
        write leaves the document untouched (the document rolls itself back)
        and this handler returns before touching the button or the form: the
        alternative was a form showing a second sheet the document no longer
        had.
        """
        self.commit()
        if self.document.has_second_material():
            if not self._confirm_pair_list_collapse():
                return
            try:
                self.document.remove_second_material()
            except DocumentError as exc:
                self._warn(
                    'Could not remove the second material', str(exc))
                return
        else:
            self.document.add_second_material()
        self._update_material_button()
        self.structure_changed.emit()
        self.changed.emit()

    def _confirm_pair_list_collapse(self) -> bool:
        """Ask before discarding the second material of every pair.

        True when there is nothing to ask about -- no pair list, nothing lost
        -- because a prompt on every Remove is how a real warning becomes one
        nobody reads.
        """
        from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415

        planner = getattr(self.document, 'pair_list_collapse_plan', None)
        plan = planner() if planner is not None else None
        if plan is None:
            return True
        if plan.target is None:
            # Nowhere to write the collapsed list, so `remove_second_material`
            # will refuse outright and this handler reports that. Asking first
            # would be two dialogs for one refusal -- and a question whose
            # 'Yes' cannot be honoured.
            return True

        dropped = ', '.join(plan.dropped[:6])
        if len(plan.dropped) > 6:
            dropped += ', …'
        target = plan.target.name if plan.target else 'the config'
        same_file = (plan.source is not None and plan.target is not None
                     and Path(plan.source) == Path(plan.target))
        where = (f"The single-material list will be written to '{target}'"
                 + ('.' if same_file else
                    f", leaving '{plan.source.name}' as it is."
                    if plan.source else '.'))
        return QMessageBox.question(
            self, 'Remove the second material?',
            f"The material list names {plan.pairs} ordered pair(s). Keeping "
            f"only the first material of each drops the second one from the "
            f"sweep: {dropped}.\n\n{where}\n\nRemove the second material?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes

    def _warn(self, title: str, text: str) -> None:
        """A modal warning. Local import, like every other dialog here."""
        from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415

        QMessageBox.warning(self, title, text)

    def _build_stacking_proxy(self) -> None:
        """Show `[general] hetero_stacking` on the second sheet.

        It orders the layers of a two-material stack, so this is the only place
        it means anything -- `hidden_fields` takes it out of Run parameters for
        every mode so it is not offered twice.

        A plain `QComboBox` and deliberately *not* an entry in `self.editors`:
        everything in there is committed to this section, and this value
        belongs to `[general]`. Committing it here would write a
        `hetero_stacking` key into `[2D-2]`, which `SheetConfig` does not
        have. The choices come from the model's own Literal, so adding a
        stacking there adds it here.
        """
        if self.section.key != getattr(self.document, 'SECOND_SHEET', None):
            return
        general = self.document.section('general')
        spec = general.field_map().get('hetero_stacking') if general else None
        if spec is None:
            return

        override = field_override(self.document.model, 'general', 'hetero_stacking')
        self.stacking_box = QComboBox(self)
        for choice in spec.choices:
            self.stacking_box.addItem(str(choice), choice)
        # Connected after populating: addItem() moves the current index off -1
        # and would otherwise write the first choice over whatever the config
        # actually holds before `load()` ever reads it.
        self.stacking_box.currentTextChanged.connect(self._on_stacking_changed)

        text = override.label if override and override.label else 'Layer stacking'
        label = QLabel(text, self)
        tooltip = _compose_tooltip(
            override.help if override else None,
            override.detail if override else None,
            spec.doc, '', spec.default)
        if tooltip:
            label.setToolTip(tooltip)
            self.stacking_box.setToolTip(tooltip)
        self.form.addRow(label, self.stacking_box)

    def _on_stacking_changed(self, text: str) -> None:
        if self._loading:
            return
        self.document.set('general', 'hetero_stacking', text)
        self.changed.emit()

    def _add_field(self, spec: FieldSpec) -> None:
        """One labelled row: control, required marker, and an info bubble."""
        if spec.kind is FieldKind.NESTED:
            self._add_nested_row(spec)
            return

        key_override = self.override.fields.get(spec.key) if self.override else None

        editor = build_editor(
            spec, self,
            choices=(key_override.choices if key_override else None),
            editor=(key_override.editor if key_override else None),
            choice_labels=(key_override.choice_labels if key_override else None))
        editor.changed.connect(self._on_field_changed)
        self.editors[spec.key] = editor

        # Required fields are marked, so "which of these must I fill in?" is
        # answerable by looking rather than by pressing Render.
        text = _label_for(spec, self.override)
        required = self.effective_required(spec.key)
        label = QLabel(f"{text}{REQUIRED_MARK}" if required else text, self)
        if required:
            label.setToolTip('Required')

        help_text = key_override.help if key_override else None
        detail_text = key_override.detail if key_override else None
        hint = loopability_hint(self.document, self.section.key, spec.key)
        tooltip_text = _compose_tooltip(help_text, detail_text, spec.doc, hint, spec.default)
        enable_hint = key_override.enable_hint if key_override else None

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

        if key_override and key_override.gate:
            self._gate_first_row.setdefault(key_override.gate, self.form.rowCount())

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
                # Pairs are invisible to material_names(), which is exactly
                # why they are read separately: read_material_list refuses an
                # 'a + b' line and the document reports [] for it.
                pairs = [e for e in self.document.material_entries()
                         if len(e) == 2]
                material_editor.set_document_names(names)
                material_editor.rescan(*self._sweep_directories())

                # Whether the box is ticked is decided by whether a sweep is
                # actually configured, not by whether the paths happen to be
                # templated -- rescan() only populates the listing. A {mat}
                # config with no materials_list is unfinished, not swept, and
                # must not have the box invented for it.
                material_editor.set_sweeping(self.document.has_material_list())

                if pairs and self._pair_sweeping():
                    # A pair list is the selection outright: which material
                    # is sheet 1 and which is sheet 2 is written down in it,
                    # and no scan of the directories can reconstruct that.
                    #
                    # Only when there is a second sheet to hold it. A pair
                    # list on a one-sheet config -- hand-edited, or written
                    # before the second sheet was dropped -- is exactly the
                    # stale state `remove_second_material`'s collapse exists
                    # for, and it commits down the single-material branch:
                    # adopting the pairs here would skip the empty-selection
                    # fallback below and leave the tick list at the scan's
                    # "everything usable" default, overwriting the curated
                    # file with three unrelated materials on the first commit.
                    material_editor.set_selected_pairs(pairs)
                elif names:
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

            self._load_stacking()
            self._load_gates()
            self._prefill_required()
        finally:
            self._loading = False
        self._apply_dependencies()

    def _load_stacking(self) -> None:
        """Point the stacking proxy at what `[general]` currently holds.

        A value the model does not offer (a hand-edited config) leaves the box
        alone rather than silently rewriting it to the first choice: validation
        reports it, and the form must not quietly repair what it is reporting.
        """
        if self.stacking_box is None:
            return
        index = self.stacking_box.findData(
            self.document.get('general', 'hetero_stacking'))
        if index >= 0:
            self.stacking_box.setCurrentIndex(index)

    def _gate_is_open(self, key: str) -> bool:
        """Whether `key`'s row is currently visible, i.e. not hidden behind
        an unticked gate checkbox. A field with no gate (the common case) is
        always open."""
        key_override = self.override.fields.get(key) if self.override else None
        if not key_override or not key_override.gate:
            return True
        box = self._gate_boxes.get(key_override.gate)
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
                self.sync_templated_editors()
                # The same resync is owed to `[2D-2]`, whose placeholders a
                # pair sweep writes from here too -- but that section is
                # another SectionForm, so ConfigForm does it.
                self.templates_written.emit()

        existing = self.document.values.get(self.section.key, {})
        specs = self.section.field_map()

        for key, editor in self.editors.items():
            if is_material_section and key == 'materials_list':
                continue

            key_override = self.override.fields.get(key) if self.override else None
            if key_override and key_override.gate:
                gate_box = self._gate_boxes.get(key_override.gate)
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

    def sync_templated_editors(self) -> None:
        """Re-read mat/cif_path/pot_path from the document into their editors.

        The sweep writes those three keys straight into the document, bypassing
        the editors, which still hold whatever text was on screen before. Any
        commit of this section afterwards would read that stale text back and
        write it over what the sweep just decided.
        """
        for key in self.document.TEMPLATED_KEYS:
            editor = self.editors.get(key)
            if editor is not None:
                editor.set_value(self.document.get(self.section.key, key))

    def _pair_sweeping(self) -> bool:
        """Whether this section's sweep names ordered pairs.

        The document's shape decides it, not the widget's: each line of the
        list has to name exactly as many materials as there are sheet sections
        to hold them (`run.expand` refuses either mismatch outright).
        """
        has_second = getattr(self.document, 'has_second_material', None)
        return bool(has_second is not None and has_second())

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

        # The widget's own mode, not the document's shape re-derived: those
        # two are what have to agree, and stating the one the user is actually
        # looking at is the only way the document's check has anything to
        # catch. A pair table committed into a config with no [2D-2] would
        # otherwise write 'a + b' lines under a single-material {mat}
        # template -- refused by `run.expand` much later, with the GUI insisting
        # throughout that the config is fine.
        pair_mode = editor.is_pair_mode()
        self.document.apply_material_directories(
            structures, potentials, extension, pair_mode=pair_mode)

        if pair_mode:
            self._commit_pair_list(editor)
        else:
            self._commit_single_list(editor)
        return True

    def _commit_single_list(self, editor) -> None:
        """Write the ticked material names, when they actually changed."""
        selected = editor.selected_names()
        # Order-insensitive: selected_names() comes back in
        # scan_material_directories' alphabetical order, but a hand-maintained
        # .txt is under no obligation to be sorted. Comparing as lists would
        # treat that as a change and retarget a file that nothing actually
        # asked to change.
        if set(selected) != set(self.document.material_names()):
            self._write_material_list(editor, selected)

    def _commit_pair_list(self, editor) -> None:
        """Write the table's ordered pairs as ``'mat1 + mat2'`` lines.

        Compared against the configured list's *pairs only*, which is what
        makes "the table has nothing in it yet" different from "the user
        emptied the table". Pressing *Add a material* on a live
        single-material sweep lands exactly here on the very next commit --
        pair mode on, no pair chosen -- and comparing against the whole list
        would read as a change from three single materials to nothing, and
        delete the list the user is in the middle of pairing up. Against the
        pairs alone it is [] vs [], which is no change at all.

        Emptying a table that *did* hold pairs still clears the sweep, exactly
        as unticking every material does in the single-material branch.

        Order matters and is preserved: the pairs are compared as sets only to
        decide whether anything changed, exactly as the single-material branch
        does, but written in table order.
        """
        selected = editor.selected_pairs()
        current = [e for e in self.document.material_entries() if len(e) == 2]
        if set(selected) != set(current):
            self._write_material_list(editor, selected)

    def _write_material_list(self, editor, entries) -> None:
        written = self.document.write_material_names(entries)
        # Read back what the document actually holds -- a relative path,
        # per _relative_to_config -- rather than reusing
        # write_material_names' return value, which is the absolute
        # target it wrote to disk. The two are not the same, and the
        # generic loop used to write whichever one note_written was
        # given straight back into the document.
        editor.note_written(self.document.get('2D', 'materials_list'))
        if written is None and entries:
            # None is ambiguous on its own: write_material_names also
            # returns it for an empty `entries` (a legitimate clear, not
            # a problem). Only a real, non-empty selection failing to
            # land anywhere -- an unsaved document with no workspace to
            # fall back to -- is the decline this warns about. The
            # document is unchanged either way; note_written above just
            # read that back, so the selection on screen has nothing
            # behind it until the user saves.
            self._warn_material_list_not_written()

    def _warn_material_list_not_written(self) -> None:
        """Tell the user why a ticked selection did not reach disk.

        This is the *declined* write -- nowhere legitimate to put the list --
        which `write_material_names` reports by returning None. Its sibling is
        the write that was attempted and refused by the filesystem, which
        raises `OSError` instead; that one is reported through `refused` (see
        `_refusal_line`), not through a dialog, because it can arrive on the
        per-keystroke path where a modal cannot.

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

    def is_material_sweeping(self) -> bool:
        """Whether this section's sweep box is ticked. False for any section
        that has no material list at all."""
        editor = self.editors.get('materials_list')
        return bool(hasattr(editor, 'is_sweeping') and editor.is_sweeping())

    #: Why `[2D-2]`'s folder pickers are greyed while a pair sweep is on.
    #: Shown in the control itself, not only as a tooltip: a greyed row with
    #: nothing on screen reads as a missing feature (see
    #: `FieldEditor.set_disabled_hint`).
    SHARED_DIRECTORY_HINT = ('Chosen on the first material -- both sheets of a '
                             'pair sweep share one folder')

    def set_material_sweep_mode(self, sweeping: bool,
                                owns_directories: bool = True) -> None:
        """Folder pickers and a sweep-owned `mat`, or hand-edited ones.

        Applies to any sheet section whose material the sweep writes -- `[2D]`
        drives its own, and `ConfigForm` drives `[2D-2]`'s from it, because a
        pair sweep writes both sheets from one list.

        `owns_directories` is False for `[2D-2]`, which does not pick its own:
        `ConfigDocument._write_material_templates` builds *both* sheets'
        cif_path/pot_path out of the one structure and one potential directory
        `[2D]` holds, on every commit. A live picker there accepts a folder
        that the next commit silently reverts -- and that commit is usually
        fired by an unrelated keystroke, so the value disappears with no act of
        the user's to blame it on. Greyed, with the reason in the control, is
        the honest version: the choice is one row up, in `[2D]`.

        Unticking the sweep hands the pickers back, because without one each
        sheet names its own file again.
        """
        for key in ('cif_path', 'pot_path'):
            path_editor = self.editors.get(key)
            if path_editor is None or not hasattr(path_editor, 'set_directory_mode'):
                continue
            path_editor.set_directory_mode(sweeping)
            enabled = owns_directories or not sweeping
            path_editor.setEnabled(enabled)
            path_editor.set_disabled_hint(
                None if enabled else self.SHARED_DIRECTORY_HINT)
            label, _cell = self._rows.get(key, (None, None))
            if label is not None:
                label.setEnabled(enabled)
        mat_editor = self.editors.get('mat')
        if mat_editor is not None:
            mat_editor.setEnabled(not sweeping)

    def _apply_sweep_mode(self) -> None:
        """Folder pickers while sweeping, single-file pickers otherwise.

        The sweep state decides what Structure/Potential mean -- a directory
        to scan when ticked, a single file otherwise -- not whichever mode the
        widgets happened to be left in. The mat field is only ever set to
        '{mat}' by the post-apply resync in commit(), never here: ticking the
        box alone must not manufacture a {mat}-with-no-materials_list config.

        The same state decides whether the selection is a tick list or a pair
        table: pair mode is sweeping AND a second sheet, because that is
        exactly when each line of the list has to name two materials. Adding
        or dropping the second material rebuilds the form, which re-enters
        here, so the mode follows the button with nothing else to wire up.
        """
        material_editor = self.editors.get('materials_list')
        if self.section.key != '2D' or not hasattr(material_editor, 'is_sweeping'):
            return
        sweeping = material_editor.is_sweeping()
        self.set_material_sweep_mode(sweeping)
        # The document's shape, not the mode itself: the widget derives the
        # mode from this and its own tick, so that a tick can never commit
        # while the two disagree (see `MaterialListEditor.set_pair_capable`).
        material_editor.set_pair_capable(self._pair_sweeping())
        self.sweep_mode_changed.emit(sweeping)

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

        A `.txt` of the *other* kind -- ordered pairs while this config has
        one sheet, single names while it has two -- is neither adopted nor
        overwritten, the same rule `load()` follows for a list already
        configured (see its `set_selected([])` fallback). The fallback is the
        whole point, and this path had the two branches without it: the line
        above has already repointed `materials_list` at the curated file, so
        leaving the previous selection standing means the next commit -- any
        field edit, Render, Generate or Save -- compares a full selection
        against the *nothing* an unusable list reports, calls that a change,
        and rewrites the curated file into the other kind of list. Clearing
        the live selection makes that comparison empty against empty, i.e. a
        no-op, and the note says so on the widget rather than leaving the
        picker looking like it did nothing at all.

        This is the only path that repoints `materials_list` from the picker:
        `MaterialListEditor._browse_list` raises `editingFinished` itself
        rather than writing the document, so a browse and a typed path both
        arrive here, and the only other writers of that key are `load()`
        (which reads, never writes) and `_write_material_list`.
        """
        material_editor = self.editors.get('materials_list')
        if material_editor is None or not material_editor.is_sweeping():
            return
        text = material_editor.path_edit.text().strip()
        if not text or text == self.document.get('2D', 'materials_list'):
            return
        self.document.set('2D', 'materials_list', text)
        pairs = [e for e in self.document.material_entries() if len(e) == 2]
        names = self.document.material_names()
        if material_editor.is_pair_mode():
            if pairs:
                material_editor.load_pairs(pairs)
            else:
                material_editor.clear_selection(self._unusable_list_note(
                    text, wanted_pairs=True, found_other=bool(names)))
        elif names:
            material_editor.load_names(names)
        else:
            material_editor.clear_selection(self._unusable_list_note(
                text, wanted_pairs=False, found_other=bool(pairs)))
        material_editor.note_written(text)

    @staticmethod
    def _unusable_list_note(text: str, wanted_pairs: bool,
                            found_other: bool) -> str:
        """Why the `.txt` just chosen was not loaded as the selection.

        Names the file and the shape this config needs, because the two ways
        of getting here want opposite fixes: a list of the wrong kind means
        either the file or the number of sheets is wrong, while an unreadable
        one is a path problem.
        """
        name = Path(text).name or text
        wanted = ("ordered pairs ('mat1 + mat2'), one per line"
                  if wanted_pairs else 'one material name per line')
        if found_other:
            found = ('single material names'
                     if wanted_pairs else "ordered pairs ('mat1 + mat2')")
            return (f"'{name}' names {found}, but this config needs {wanted}. "
                    f"Nothing was selected; the file was left as it is.")
        return (f"'{name}' could not be read as {wanted}. Nothing was "
                f"selected; the file was left as it is.")

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
            key_override = self.override.fields.get(key)
            if not key_override or not key_override.enable_if:
                continue

            other_key, expected = key_override.enable_if
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
            if bubble is not None and key_override.enable_hint:
                # The composed tooltip carries the loopability hint, which the
                # override alone does not -- recomputing from `help` drops it.
                bubble.setToolTip(
                    key_override.enable_hint if not enabled
                    else self._composed_tooltips.get(key, ''))

            # A greyed row with no on-screen explanation reads as a missing
            # feature (only the ⓘ bubble said why, and nothing prompts anyone
            # to hover it). Put the same reason in the control itself.
            if key_override.enable_hint:
                editor.set_disabled_hint(
                    None if enabled else key_override.enable_hint)

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
            for key, key_override in self.override.fields.items():
                if key_override.gate != gate_key:
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
        try:
            self.commit()
        except (DocumentError, OSError) as exc:
            # A refusal has to be said out loud. This is a Qt slot: an
            # exception out of here is printed to stderr by the signal
            # machinery (app.py installs no excepthook) and the edit silently
            # does nothing -- indistinguishable, from the user's side, from a
            # GUI that ignored them. `commit()` itself keeps raising, so the
            # document's contract stays as loud as it was.
            #
            # `OSError` for the same reason and by the same route: committing
            # a sweep writes the material list to disk, so a read-only config
            # directory refuses the edit here, in a slot, where the traceback
            # goes to a terminal nobody is reading.
            #
            # Reported through a signal and NOT a modal, unlike the collapse
            # confirmation, which genuinely needs a blocking yes/no. This slot
            # runs on ordinary field edits: a dialog here interrupts typing,
            # and -- with no event loop to dismiss it -- turns any future
            # regression on this path into a hang instead of a test failure.
            self.refused.emit(_refusal_line(self.section.key, exc))
            return
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
    #: A section's commit was refused, with a message ready to show. Raised by
    #: `_commit_section` and forwarded from every `SectionForm.refused`, so a
    #: window has one place to listen. See `_record_refusal`.
    refused = Signal(str)

    def __init__(self, document: ConfigDocument,
                 parent: Optional[QWidget] = None,
                 start_collapsed: bool = False,
                 settings: Optional[object] = None) -> None:
        super().__init__(parent)
        self._document: Optional[ConfigDocument] = None
        #: Whether `document` has been pointed at a different config since the
        #: last rebuild. See the property setter below.
        self._document_swapped = False
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
        #: Refusals since the last `commit()`. The signal alone is not enough:
        #: `_validate` runs a commit and then rewrites the issues panel from
        #: scratch, so a refusal that only ever existed as a signal would be
        #: erased by the very validation that provoked it.
        self._refusals: List[str] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.rebuild()

    @property
    def document(self) -> ConfigDocument:
        return self._document

    @document.setter
    def document(self, document: ConfigDocument) -> None:
        """Point the form at another config.

        A property purely so a *swap* is observable. `app._open_config`
        assigns here and then calls `rebuild()`, which carries a ticked sweep
        box across -- and that tick belongs to the config being closed. Left
        standing on the config being opened it turned a `{mat}`-templated
        config with no `materials_list` into a full sweep on the `_validate()`
        of the very next line: a `.txt` written beside a file the user had
        only just opened, with no action of theirs at all.
        """
        if document is not self._document:
            self._document_swapped = True
        self._document = document

    def rebuild(self) -> None:
        """Recreate every section form, e.g. after the model changed."""
        # Read before anything is torn down: a ticked sweep box that has not
        # been configured yet writes nothing to the document, so the rebuilt
        # form has nothing to restore it from. Adding a second material is
        # precisely when that happens -- the button rebuilds the form -- and
        # the sweep silently switching itself off there took the two
        # directories picked next down with it, saving them as literal paths.
        # Only within one document, though: see the `document` setter.
        sweeping = self._material_sweep_state() and not self._document_swapped
        self._document_swapped = False
        self.setUpdatesEnabled(False)
        try:
            while self._layout.count():
                item = self._layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.sections.clear()

            for section in self.document.sections():
                self._add_section(section)
                if section.key == '2D':
                    second = self._second_material_section()
                    if second is not None:
                        # Directly after the first sheet: a config whose two
                        # materials are pages apart is one nobody can read.
                        self._add_section(second)

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
            if sweeping:
                self._restore_material_sweep()
            self._wire_lj_known_elements_refresh()
            self._wire_second_sheet_to_the_sweep()

            self._layout.addStretch(1)
        finally:
            self.setUpdatesEnabled(True)

    def _add_section(self, section: SectionSpec) -> None:
        """One SectionForm, wired and folded to its remembered state."""
        form = SectionForm(section, self.document, self, settings=self.settings)
        form.changed.connect(self.changed)
        form.collapsed_changed.connect(self._remember_collapsed)
        form.structure_changed.connect(self._on_structure_changed)
        form.refused.connect(self._record_refusal)
        self.sections[section.key] = form
        self._layout.addWidget(form)

        collapsed = self._collapsed.get(
            section.key,
            self.start_collapsed or _starts_collapsed(section, self.document))
        form.set_collapsed(collapsed)

    def _second_material_section(self) -> Optional[SectionSpec]:
        """The synthetic `[2D-2]` spec, or None when there is no second sheet.

        Not part of `document.sections()`, because the second sheet has no
        field on the root model at all -- a model validator folds numbered
        sections into a derived `sheets` list, so introspection of the root
        model can never produce it. It is described from `SheetConfig` on
        demand instead.

        Forced non-optional: an optional section renders with a group-box tick
        that routes through `enable_section('2D-2')`, which the document
        refuses (there is no such section spec to enable). The second material
        is added and dropped by the `[2D]` button, not by a tick.
        """
        has_second = getattr(self.document, 'has_second_material', None)
        if has_second is None or not has_second():
            return None
        spec = self.document.second_material_spec()
        spec.optional = False
        return spec

    def _material_sweep_state(self) -> bool:
        """Whether `[2D]`'s sweep box is currently ticked, before a rebuild
        throws the widget away. False when there is no `[2D]` form yet."""
        primary = self.sections.get('2D')
        return primary is not None and primary.is_material_sweeping()

    def _restore_material_sweep(self) -> None:
        """Re-tick the sweep on the rebuilt `[2D]`.

        Only ever re-ticks: `load()` has already ticked the box for any sweep
        the document actually holds, and this restores the one state the
        document cannot express -- ticked, nothing chosen yet.
        """
        primary = self.sections.get('2D')
        editor = primary.editors.get('materials_list') if primary else None
        if editor is not None and hasattr(editor, 'set_sweeping'):
            editor.set_sweeping(True)

    def _second_sheet_form(self) -> Optional[SectionForm]:
        return self.sections.get(
            getattr(self.document, 'SECOND_SHEET', None) or '')

    def _wire_second_sheet_to_the_sweep(self) -> None:
        """Let `[2D]`'s material sweep drive `[2D-2]`, which it also writes.

        A pair sweep fills both sheets from one list, so the second sheet's
        Structure/Potential cells mean a directory exactly when the first
        one's do, and its placeholders are written from the first section's
        commit -- behind its own editors' backs. Only this class owns both
        sections, so both connections are made here rather than between two
        `SectionForm`s that cannot see each other.

        The initial `set_material_sweep_mode` call is not redundant with the
        signal: `[2D]` applies its own sweep mode during construction, before
        `[2D-2]` exists to hear about it.
        """
        primary = self.sections.get('2D')
        if primary is None or self._second_sheet_form() is None:
            return
        primary.sweep_mode_changed.connect(self._on_sweep_mode_changed)
        primary.templates_written.connect(self._on_templates_written)
        self._on_sweep_mode_changed(primary.is_material_sweeping())

    def _on_sweep_mode_changed(self, sweeping: bool) -> None:
        second = self._second_sheet_form()
        if second is not None:
            # `owns_directories=False`: the sweep writes [2D-2]'s paths from
            # the folders [2D] holds, so its own pickers would be reverted.
            second.set_material_sweep_mode(sweeping, owns_directories=False)

    def _on_templates_written(self) -> None:
        second = self._second_sheet_form()
        if second is not None:
            second.sync_templated_editors()

    def _on_structure_changed(self) -> None:
        """Rebuild after the *set* of sections changed, not just a value.

        Deferred by a zero timer because the request comes from a button inside
        the very SectionForm `rebuild()` is about to delete; tearing that widget
        down from inside its own signal handler is a crash, not an error.
        """
        QTimer.singleShot(0, self.rebuild)

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
        """Commit every section, `[2D]` first.

        The order is load-bearing for a pair sweep: `[2D]`'s commit writes
        `[2D-2]`'s `{mat2}` placeholders straight into the document, and
        `[2D-2]`'s own commit would write its (pre-sweep) editor text back
        over them. Committing the sweep first means the resync
        `templates_written` triggers has already happened by the time the
        second sheet's turn comes.
        """
        self._refusals = []
        primary = self.sections.get('2D')
        if primary is not None:
            self._commit_section(primary)
        for form in self.sections.values():
            if form is not primary:
                self._commit_section(form)

    def refusals(self) -> List[str]:
        """What the last `commit()` could not apply, ready to show.

        Read by the window after its own `commit()`, so the refusals land in
        the issues panel alongside the validation messages and stay readable
        instead of scrolling past in the status bar.
        """
        return list(self._refusals)

    def _record_refusal(self, message: str) -> None:
        """Remember a refusal and pass it on."""
        self._refusals.append(message)
        self.refused.emit(message)

    def _commit_section(self, form: SectionForm) -> None:
        """Commit one section, reporting a refusal rather than raising it.

        `commit()` is called from Render, Generate and Save -- all of them Qt
        slots in `app.py`, none with a handler of its own -- so a
        `DocumentError` escaping here is a stderr traceback and a button that
        appears to have done nothing.

        `OSError` travels the same way and for the same reason: committing a
        material sweep writes the list to disk, and a config directory the
        user cannot write refuses it. Left uncaught it crashed out of those
        three slots rather than merely doing nothing.

        Reported, not shown: this used to raise a modal per failing section,
        which meant a config refusing in two sections put two dialogs in front
        of one mistake, and -- since Render, Generate and Save all come
        through here -- any headless or offscreen driver of `commit()` blocked
        forever on a dialog nothing could dismiss. Collecting them lets the
        window say all of them at once, in the panel that already holds the
        other reasons a config will not build.
        """
        try:
            form.commit()
        except (DocumentError, OSError) as exc:
            self._record_refusal(_refusal_line(form.section.key, exc))

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
                key_override = form.override.fields.get(key) if form.override else None
                blocks = key_override.required_blocks if key_override else True
                missing.append((section_key, key, label, blocks))
        return missing
