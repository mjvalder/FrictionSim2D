"""Choosing the materials a run sweeps over.

A sweep needs three things agreeing with each other: a `.txt` of names in
``[2D] materials_list``, and `{mat}` placeholders in `mat`, `cif_path` and
`pot_path` so expansion has somewhere to substitute. Setting them by hand is
why the feature was effectively undiscoverable -- and the placeholder half was
invisible in the GUI entirely.

Both halves come from one question instead: which structure directory, and
which potential directory. The material list is the intersection, which is what
``preflight_materials`` works out at build time anyway. The two directories
themselves are no longer this widget's own text fields -- they are the [2D]
section's `cif_path`/`pot_path` pickers, switched into folder mode while
sweeping; `SectionForm` reads them and calls `rescan()` here.

This widget renders as two rows, not one: the checkbox (``materials_list``'s
own row, carrying no text of its own -- the form label already says "Sweep
over materials") and, placed later by `SectionForm` (after `mat`, see
`_insert_material_list_row`), the ``.txt`` path picker plus the tick list
itself, exposed as `list_panel`. They stay one editor -- one `value()`, one
set of sweep-state guards -- split only for where Qt draws it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ..controllers.materials import (
    infer_potential_extension,
    scan_material_directories,
)
from ..schema.introspect import FieldSpec
from ..style import CAPTION_SMALL
from .fields import FieldEditor, register_editor

__all__ = ['MaterialListEditor']


class MaterialListEditor(FieldEditor):
    """A sweep toggle and the resulting material list.

    ``value()`` is the ``materials_list`` path, because that is what the config
    holds. Writing the `.txt` and the `{mat}` templates spans other fields in
    the section, so it happens in ``SectionForm.commit``. The two directories
    that decide what is available live in the [2D] section's own `cif_path`/
    `pot_path` editors -- ``rescan()`` is how their current values reach here.
    """

    def __init__(self, spec: FieldSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(spec, parent)
        self._path: Optional[str] = None
        self._extension: Optional[str] = None
        #: Whether the user has ever changed a checkbox themselves (directly,
        #: or via Select all / None). Until then, a rescan defaults to
        #: selecting everything usable; after, an empty selection is a real
        #: choice and must survive a rescan rather than being read as "nothing
        #: has been picked yet" and silently reset to "everything".
        self._selection_touched = False
        #: Reentrancy guard: rescan() emits `changed` while this is still
        #: True (the emit sits inside the guarded region, not after it), so a
        #: listener that calls back into rescan() synchronously -- which
        #: nothing upstream should ever do, see SectionForm's rescan wiring --
        #: is refused rather than recursing.
        self._rescanning = False
        #: The (structures, potentials, extension) triple last passed to
        #: rescan(), or None before the first one. This widget no longer
        #: keeps its own directory text -- SectionForm's path editors do --
        #: so this is what `directories()` reports back to
        #: `_commit_material_sweep`.
        self._scanned: Optional[Tuple[Optional[Path], Optional[Path],
                                      Optional[str]]] = None
        #: Whether the selection is a table of ordered pairs rather than a
        #: tick list of single materials. Always `_pair_capable` AND
        #: `is_sweeping()`; never set independently of those two.
        self._pair_mode = False
        #: Whether this config has a second sheet to hold the second member
        #: of each pair. SectionForm's to know (it reads the document) and
        #: this widget's to remember, because the widget is what has to keep
        #: `_pair_mode` true to it the instant the box is ticked -- see
        #: `_on_sweep_toggled`.
        self._pair_capable = False
        #: material_names() as of the last load(), passed in via
        #: set_document_names. The floor a rescan's "keep the selection"
        #: logic falls back to when the widget's own selection has gone empty
        #: -- e.g. an intermediate scan of a directory that does not resolve
        #: -- so a curated .txt cannot be silently read as "nothing selected".
        self._document_names: List[str] = []

        # This editor's own row holds only the checkbox -- no text of its
        # own, matching every other checkbox editor (_BoolEditor): the row
        # label ("Sweep over materials", from the override) already names
        # it, and repeating that on the checkbox was the duplicated wording
        # the user's screenshot flagged.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.sweep_box = QCheckBox(self)
        self.sweep_box.toggled.connect(self._on_sweep_toggled)
        layout.addWidget(self.sweep_box)
        layout.addStretch(1)

        # The .txt picker, tick list and Select all/None -- deliberately
        # NOT part of this widget's own layout. `SectionForm` places it as a
        # second, later form row (after `mat`, see
        # `SectionForm._insert_material_list_row`) so the checkbox --
        # first, and in its own row -- visibly governs the file/folder cells
        # in between, per the user's explicit layout choice.
        self.list_panel = QWidget()
        panel = QVBoxLayout(self.list_panel)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(3)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(self.list_panel)
        self.path_edit.setPlaceholderText(
            'Point at an existing materials .txt to load it, or leave blank')
        path_row.addWidget(self.path_edit, 1)
        self.browse_button = QPushButton('…', self.list_panel)
        self.browse_button.setFixedWidth(30)
        self.browse_button.setToolTip('Browse for a materials .txt')
        self.browse_button.clicked.connect(self._browse_list)
        path_row.addWidget(self.browse_button)
        panel.addLayout(path_row)

        self.extension_label = QLabel('', self.list_panel)
        self.extension_label.setStyleSheet(CAPTION_SMALL)
        panel.addWidget(self.extension_label)

        self.summary = QLabel('', self.list_panel)
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(CAPTION_SMALL)
        panel.addWidget(self.summary)

        self.listing = QListWidget(self.list_panel)
        self.listing.setMaximumHeight(160)
        self.listing.itemChanged.connect(self._on_item_changed)
        panel.addWidget(self.listing)

        self.list_buttons = self._button_row(
            (('Select all', self.select_all), ('None', self.select_none)))
        panel.addWidget(self.list_buttons)

        # The pair half: same widget, other mode. A pair sweep names an
        # ORDERED pair per line, and which material is sheet 1 and which is
        # sheet 2 is the whole content of a heterostructure -- a set of ticks
        # cannot say it, so pair mode swaps the list for a two-column table.
        self.pair_table = QTableWidget(0, 2, self.list_panel)
        self.pair_table.setHorizontalHeaderLabels(['Material 1', 'Material 2'])
        self.pair_table.verticalHeader().setVisible(False)
        self.pair_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.pair_table.setMaximumHeight(160)
        self.pair_table.setVisible(False)
        panel.addWidget(self.pair_table)

        self.pair_buttons = self._button_row(
            (('Add pair', self.add_pair), ('Remove pair', self.remove_pair),
             ('Generate pairs…', self._open_generate_dialog)))
        self.pair_buttons.setVisible(False)
        panel.addWidget(self.pair_buttons)

        self.list_panel.setVisible(False)

    def _button_row(self, entries) -> QWidget:
        """One row of push buttons, in a widget of its own so the row can be
        shown or hidden as a unit when the mode changes."""
        row = QWidget(self.list_panel)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for label, slot in entries:
            button = QPushButton(label, row)
            button.clicked.connect(slot)
            layout.addWidget(button)
        layout.addStretch(1)
        return row

    # ----------------------------------------------------------------- state

    def is_sweeping(self) -> bool:
        return self.sweep_box.isChecked()

    def set_sweeping(self, sweeping: bool) -> None:
        self.sweep_box.setChecked(sweeping)

    def set_document_names(self, names: List[str]) -> None:
        """Record the document's current material list, for rescan()'s
        selection-keeping fallback. See ``_document_names``."""
        self._document_names = list(names or [])

    def directories(self) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
        """The (structures, potentials, extension) triple last passed to
        rescan(), or all-None before one ever happened."""
        return self._scanned or (None, None, None)

    def listed_names(self) -> List[str]:
        return [self.listing.item(i).data(Qt.UserRole)
                for i in range(self.listing.count())]

    def has_usable_names(self) -> bool:
        """Whether anything in the current listing could actually be swept.

        scan_material_directories reports orphans too -- names present in
        only one directory -- so listed_names() being non-empty does not mean
        anything here is usable. That distinction matters when only one of
        the two directories resolves (a moved run directory, an unmounted
        share): the listing is full of unusable orphans from the directory
        that *did* resolve, and treating that as a real (empty) selection
        would delete a working sweep the scan simply cannot see right now.
        """
        return any(
            self.listing.item(i).flags() & Qt.ItemIsUserCheckable
            for i in range(self.listing.count())
        )

    def selected_names(self) -> List[str]:
        return [
            self.listing.item(i).data(Qt.UserRole)
            for i in range(self.listing.count())
            if self.listing.item(i).checkState() == Qt.Checked
        ]

    def set_selected(self, names: List[str]) -> None:
        wanted = set(names)
        for index in range(self.listing.count()):
            item = self.listing.item(index)
            if not (item.flags() & Qt.ItemIsUserCheckable):
                continue
            item.setCheckState(
                Qt.Checked if item.data(Qt.UserRole) in wanted else Qt.Unchecked)

    def select_all(self) -> None:
        self.set_selected(self.listed_names())

    def select_none(self) -> None:
        self.set_selected([])

    def summary_text(self) -> str:
        return self.summary.text()

    # ------------------------------------------------------------ pair mode

    def is_pair_mode(self) -> bool:
        return self._pair_mode

    def set_pair_capable(self, capable: bool) -> None:
        """Record whether this config has a second sheet, and re-derive the
        mode from it.

        The mode itself is never handed in from outside any more. It used to
        be, and the ordering that made that wrong is worth stating: this
        widget connects its own `toggled` handler in the constructor, before
        `SectionForm` connects the one that would have re-applied the mode,
        and Qt delivers slots in connection order. So the commit that a
        re-tick fires ran while the mode was still whatever the *untick* left
        behind -- and a stale mode is not a cosmetic disagreement, it is what
        `apply_material_directories`' `pair_mode` refuses outright. Keeping
        the ingredients here and deriving the mode from them means the widget
        cannot be caught mid-update by its own signal.
        """
        self._pair_capable = bool(capable)
        self.set_pair_mode(self._pair_capable and self.is_sweeping())

    def set_pair_mode(self, pair_mode: bool) -> None:
        """Swap the tick list for the pair table, or back.

        The two selections are kept side by side rather than converted into
        each other: adding a second material and changing one's mind about it
        is one click each way, and a conversion would have to invent the
        member the other mode never had.
        """
        self._pair_mode = bool(pair_mode)
        self.listing.setVisible(not self._pair_mode)
        self.list_buttons.setVisible(not self._pair_mode)
        self.pair_table.setVisible(self._pair_mode)
        self.pair_buttons.setVisible(self._pair_mode)
        self._update_summary()

    def usable_names(self) -> List[str]:
        """Listed names that have both a structure and a potential file."""
        return [
            self.listing.item(index).data(Qt.UserRole)
            for index in range(self.listing.count())
            if self.listing.item(index).flags() & Qt.ItemIsUserCheckable
        ]

    def pair_pool(self) -> List[str]:
        """The materials `generate_pairs` builds from.

        The ticked ones, falling back to everything usable when nothing is
        ticked: the tick list is hidden in pair mode, so an empty selection
        there is far more likely to mean "not used" than "deliberately none",
        and a Generate that produced nothing at all would just look broken.
        """
        return self.selected_names() or self.usable_names()

    def selected_pairs(self) -> List[Tuple[str, str]]:
        """The ordered pairs currently in the table, in table order."""
        pairs: List[Tuple[str, str]] = []
        for row in range(self.pair_table.rowCount()):
            first = self.pair_table.cellWidget(row, 0)
            second = self.pair_table.cellWidget(row, 1)
            if first is None or second is None:
                continue
            mat1, mat2 = first.currentText().strip(), second.currentText().strip()
            if mat1 and mat2:
                pairs.append((mat1, mat2))
        return pairs

    def set_selected_pairs(self, pairs) -> None:
        """Replace the table's rows. Emits nothing: this is how a document is
        loaded into the widget, and committing on load is how an open turns
        into a write."""
        self.pair_table.setRowCount(0)
        for pair in pairs:
            members = list(pair)
            self._append_pair_row(members[0] if members else None,
                                  members[1] if len(members) > 1 else None)
        self._update_summary()

    def add_pair(self) -> None:
        """Append a row, defaulted to the first two materials available."""
        pool = self.pair_pool()
        self._append_pair_row(pool[0] if pool else None,
                              pool[1] if len(pool) > 1 else
                              (pool[0] if pool else None))
        self._selection_touched = True
        self._update_summary()
        self.changed.emit()

    def remove_pair(self) -> None:
        """Drop the selected row, or the last one when nothing is selected."""
        if not self.pair_table.rowCount():
            return
        row = self.pair_table.currentRow()
        if row < 0:
            row = self.pair_table.rowCount() - 1
        self.pair_table.removeRow(row)
        self._selection_touched = True
        self._update_summary()
        self.changed.emit()

    def generate_pairs(self, include_reversed: bool = False,
                       include_same: bool = False) -> None:
        """Fill the table with every ordered pair over `pair_pool()`.

        `include_reversed` is off by default even though AB and BA really are
        different stacks: doubling the sweep is a decision, not a default.
        `include_same` adds the homostructure bilayers (``a + a``), which are
        a legitimate reference run but not what "pair these up" asks for.
        """
        pool = self.pair_pool()
        pairs = [
            (first, second)
            for i, first in enumerate(pool)
            for j, second in enumerate(pool)
            if not (i == j and not include_same)
            and not (j < i and not include_reversed)
        ]
        self.set_selected_pairs(pairs)
        self._selection_touched = True
        self.changed.emit()

    def _append_pair_row(self, first: Optional[str] = None,
                         second: Optional[str] = None) -> None:
        row = self.pair_table.rowCount()
        self.pair_table.insertRow(row)
        for column, name in ((0, first), (1, second)):
            self.pair_table.setCellWidget(row, column, self._pair_combo(name))

    def _pair_combo(self, current: Optional[str]) -> QComboBox:
        """One cell: the usable materials, plus whatever this cell already
        holds.

        A curated list may name a material neither directory holds on this
        machine -- a moved share, a potential not yet generated. Offering only
        what the last scan saw would silently rewrite that member to something
        else the moment the combo was built, which is the same silent
        narrowing the tick list's keep-selection logic exists to prevent.

        The signal is connected only after the current item is set, so
        populating a row never reads as the user editing it.
        """
        combo = QComboBox(self.pair_table)
        names = self.usable_names()
        if current and current not in names:
            names = [current] + names
        combo.addItems(names)
        if current:
            index = combo.findText(current)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.currentTextChanged.connect(self._on_pair_changed)
        return combo

    def _on_pair_changed(self, _text: str) -> None:
        self._selection_touched = True
        self._update_summary()
        self.changed.emit()

    def _open_generate_dialog(self) -> None:
        """Ask for the two options, then build the pairs.

        Local import, like `_browse_list`'s QFileDialog: the dialog is only
        ever needed when the button is actually pressed.
        """
        from PySide6.QtWidgets import (  # noqa: PLC0415
            QDialog, QDialogButtonBox,
        )

        pool = self.pair_pool()
        dialog = QDialog(self.list_panel)
        dialog.setWindowTitle('Generate pairs')
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(
            f"Pairs will be built from the {len(pool)} material(s) currently "
            f"available: {', '.join(pool) or '(none)'}.", dialog))
        reversed_box = QCheckBox('Include the reversed order (b + a)', dialog)
        same_box = QCheckBox('Include same-material pairs (a + a)', dialog)
        layout.addWidget(reversed_box)
        layout.addWidget(same_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            self.generate_pairs(include_reversed=reversed_box.isChecked(),
                                include_same=same_box.isChecked())

    # --------------------------------------------------------------- scanning

    def rescan(self, structures, potentials,
              extension: Optional[str] = None) -> None:
        """Re-read (structures, potentials, extension) and rebuild the listing.

        A no-op when the triple is identical to the last call and the listing
        is already populated: `editingFinished` fires even when the text did
        not actually change (a plain focus-out), and rebuilding -- and
        re-emitting `changed` -- on every one of those would be pure noise for
        no new information.

        `self._rescanning` guards against a reentrant call landing here while
        one is already in progress -- true from before the listing is
        rebuilt until after `changed` has been emitted and every connected
        slot has returned, so a slot that calls back into rescan() while
        handling that signal is refused rather than recursing. See the class
        docstring's note on `_rescanning`.
        """
        # Resolved, not the raw argument: a folder-mode field never carries an
        # extension of its own (there is no `{mat}` to read a suffix off of),
        # so the triple `directories()` reports back to `_commit_material_sweep`
        # has to be the one actually used to scan, or a sweep built purely
        # through folder-mode editing would look unconfigured at commit time
        # (extension=None) even though the listing is fully populated.
        resolved_extension = extension or infer_potential_extension(potentials)
        triple = (structures, potentials, resolved_extension)
        if self._rescanning:
            return
        if triple == self._scanned and self.listing.count():
            return

        self._rescanning = True
        try:
            self._scanned = triple
            self._extension = resolved_extension
            # Only "nobody has decided yet" defaults to everything usable.
            # Using an empty `keep` for that would also fire the first time a
            # user re-picks a directory after explicitly clicking None: their
            # empty selection is a real choice, not an unset default. When the
            # widget's own selection has gone empty for some other reason --
            # an intermediate scan of a directory that does not resolve --
            # fall back to the document's list rather than reading that as
            # "the user emptied it".
            keep = set(self.selected_names()) or set(self._document_names)
            select_by_default = not self._selection_touched

            candidates = scan_material_directories(
                structures, potentials, self._extension)

            self.listing.blockSignals(True)
            try:
                self.listing.clear()
                for candidate in candidates:
                    text = (candidate.name if candidate.usable
                            else f"{candidate.name}   {candidate.reason}")
                    item = QListWidgetItem(text, self.listing)
                    item.setData(Qt.UserRole, candidate.name)
                    if candidate.usable:
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        # First scan selects everything usable; a later one
                        # keeps whatever was ticked (even nothing), so
                        # re-picking a directory does not silently undo the
                        # user's selection.
                        item.setCheckState(
                            Qt.Checked if (select_by_default or candidate.name in keep)
                            else Qt.Unchecked)
                    else:
                        item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable
                                      & ~Qt.ItemIsEnabled)
                        item.setCheckState(Qt.Unchecked)
            finally:
                self.listing.blockSignals(False)

            # Rebuild the pair combos against the new listing, keeping the
            # pairs themselves: a scan changes what is *available*, not what
            # the user has chosen.
            self.set_selected_pairs(self.selected_pairs())

            self.extension_label.setText(self._extension or '')
            if candidates:
                self._update_summary()
            else:
                self.summary.setText('Choose both directories.')

            # Emitted inside the guarded region, not after: `_rescanning` is
            # what a reentrant rescan() call checks, so it must still be True
            # for the whole time `changed` is out and its listeners are
            # running -- otherwise a listener that calls rescan() again would
            # sail straight through the guard the moment it is released.
            self.changed.emit()
        finally:
            self._rescanning = False

    def _on_item_changed(self, _item) -> None:
        # Only reached from a real click: programmatic changes during rescan()
        # happen under blockSignals(). This is what distinguishes "the user
        # has never touched the selection" from "the user touched it and
        # ended up at empty".
        self._selection_touched = True
        self._update_summary()
        self.changed.emit()

    def _update_summary(self) -> None:
        """What this selection would build, counted the way the current mode
        actually expands: one config per ticked material, or one per pair."""
        if self._pair_mode:
            count = len(self.selected_pairs())
            self.summary.setText(f"{count} pair(s) → {count} configs")
            return
        total = self.listing.count()
        usable = sum(
            1 for i in range(total)
            if self.listing.item(i).flags() & Qt.ItemIsUserCheckable)
        self.summary.setText(
            f"{usable} of {total} usable → {len(self.selected_names())} configs")

    def _on_sweep_toggled(self, sweeping: bool) -> None:
        self.list_panel.setVisible(sweeping)
        # Before the emit, always: `changed` is what commits the section, and
        # a commit reads `is_pair_mode()`. See `set_pair_capable`.
        self.set_pair_mode(self._pair_capable and sweeping)
        self.changed.emit()

    # ------------------------------------------------------------- .txt path

    def _browse_list(self) -> None:
        from PySide6.QtWidgets import QFileDialog  # noqa: PLC0415

        chosen, _ = QFileDialog.getOpenFileName(
            self.list_panel, 'Select a materials list',
            filter='Text files (*.txt);;All files (*)')
        if chosen:
            self.path_edit.setText(chosen)
            # A rescan-style trigger, not a per-keystroke one: setText()
            # alone does not fire editingFinished, so a browse selection has
            # to raise it explicitly -- mirrors _PathEditor._browse, and is
            # what SectionForm's own editingFinished wiring for this field
            # depends on (see SectionForm._wire_material_sweep's docstring
            # for why that rule matters).
            self.path_edit.editingFinished.emit()

    def load_names(self, names: List[str]) -> None:
        """Adopt `names` as the selection -- e.g. from a curated `.txt` the
        path row was just pointed at (SectionForm reads the file and calls
        this once it has resolved something).

        Marks the selection touched exactly as a real click would: a curated
        list the user just loaded is a deliberate choice, not "nothing
        decided yet", so a later rescan of the structure/potential
        directories must not silently widen it back to "everything usable".

        If the listing is currently empty -- no structure/potential
        directory has resolved to anything yet, whatever the reason -- there
        is nothing real in `self.listing` to tick against: `set_selected`
        would silently no-op and the summary would read "0 of 0 usable", as
        if pointing the picker at the file had done nothing at all. Showing
        the loaded names as unusable-until-scanned placeholders instead
        makes that legible without pretending any of them have actually been
        checked against a directory -- `has_usable_names()` still reports
        False for every one of them, so the orphan-scan guard in
        `_commit_material_sweep` is untouched: there is still nothing here a
        commit could destroy.
        """
        self._selection_touched = True
        self.set_document_names(names)
        if self.listing.count() == 0:
            self._seed_placeholder_listing(names)
        else:
            self.set_selected(names)
            self._update_summary()
        self.changed.emit()

    def load_pairs(self, pairs) -> None:
        """Adopt `pairs` as the selection -- the pair-mode sibling of
        `load_names`, for a curated `.txt` of ``'mat1 + mat2'`` lines the
        path row was just pointed at.

        No placeholder listing to seed: the table's combos already carry a
        member the last scan could not see (see `_pair_combo`), so a curated
        pair list is legible before either directory has resolved.
        """
        self._selection_touched = True
        self.set_selected_pairs(pairs)
        self.changed.emit()

    def _seed_placeholder_listing(self, names: List[str]) -> None:
        """Show `names` as loaded-but-unchecked, before any directory scan.

        Every item is unusable (not `Qt.ItemIsUserCheckable`), exactly like a
        real orphan from `scan_material_directories` -- `has_usable_names()`
        must keep reporting False here, the same as an empty listing would,
        since nothing has actually been verified against a structure or
        potential directory yet. A real `rescan()` later discards these
        placeholders outright (it clears and rebuilds the listing from
        `scan_material_directories`'s candidates), so this state is only ever
        seen between loading a `.txt` and choosing both directories.
        """
        self.listing.blockSignals(True)
        try:
            self.listing.clear()
            for name in names:
                item = QListWidgetItem(
                    f"{name}   (choose both directories to check this)",
                    self.listing)
                item.setData(Qt.UserRole, name)
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable
                              & ~Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked)
        finally:
            self.listing.blockSignals(False)
        self.extension_label.setText('')
        self.summary.setText(
            f"{len(names)} material(s) loaded from the list -- choose "
            f"both directories to check which are usable.")

    # ----------------------------------------------------------------- value

    def value(self):
        """The ``materials_list`` path, or None when not sweeping."""
        return self._path if self.is_sweeping() else None

    def set_value(self, value) -> None:
        self._path = value if isinstance(value, str) and value else None
        self.path_edit.setText(self._path or '')
        self.set_sweeping(bool(self._path))

    def note_written(self, path: Optional[Path]) -> None:
        """Record where SectionForm.commit put the list."""
        self._path = str(path) if path else None
        self.path_edit.setText(self._path or '')


register_editor('material_list', MaterialListEditor)
