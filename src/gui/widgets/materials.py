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
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
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
        #: material_names() as of the last load(), passed in via
        #: set_document_names. The floor a rescan's "keep the selection"
        #: logic falls back to when the widget's own selection has gone empty
        #: -- e.g. an intermediate scan of a directory that does not resolve
        #: -- so a curated .txt cannot be silently read as "nothing selected".
        self._document_names: List[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.sweep_box = QCheckBox('Sweep over many materials', self)
        self.sweep_box.toggled.connect(self._on_sweep_toggled)
        layout.addWidget(self.sweep_box)

        self.body = QWidget(self)
        body = QVBoxLayout(self.body)
        body.setContentsMargins(16, 0, 0, 0)
        body.setSpacing(3)

        self.extension_label = QLabel('', self.body)
        self.extension_label.setStyleSheet(CAPTION_SMALL)
        body.addWidget(self.extension_label)

        self.summary = QLabel('', self.body)
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(CAPTION_SMALL)
        body.addWidget(self.summary)

        self.listing = QListWidget(self.body)
        self.listing.setMaximumHeight(160)
        self.listing.itemChanged.connect(self._on_item_changed)
        body.addWidget(self.listing)

        buttons = QHBoxLayout()
        for label, slot in (('Select all', self.select_all),
                            ('None', self.select_none)):
            button = QPushButton(label, self.body)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        body.addLayout(buttons)

        layout.addWidget(self.body)
        self.body.setVisible(False)

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

            usable = sum(1 for c in candidates if c.usable)
            self.extension_label.setText(self._extension or '')
            self.summary.setText(
                f"{usable} of {len(candidates)} usable → {len(self.selected_names())} "
                f"configs" if candidates else 'Choose both directories.')

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
        self._rescan_summary()
        self.changed.emit()

    def _rescan_summary(self) -> None:
        total = self.listing.count()
        usable = sum(
            1 for i in range(total)
            if self.listing.item(i).flags() & Qt.ItemIsUserCheckable)
        self.summary.setText(
            f"{usable} of {total} usable → {len(self.selected_names())} configs")

    def _on_sweep_toggled(self, sweeping: bool) -> None:
        self.body.setVisible(sweeping)
        self.changed.emit()

    # ----------------------------------------------------------------- value

    def value(self):
        """The ``materials_list`` path, or None when not sweeping."""
        return self._path if self.is_sweeping() else None

    def set_value(self, value) -> None:
        self._path = value if isinstance(value, str) and value else None
        self.set_sweeping(bool(self._path))

    def note_written(self, path: Optional[Path]) -> None:
        """Record where SectionForm.commit put the list."""
        self._path = str(path) if path else None


register_editor('material_list', MaterialListEditor)
