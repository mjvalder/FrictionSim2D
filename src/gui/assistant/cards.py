"""Proposal cards: what the agent wants to change, with Apply and Discard."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout)

from ...agent.changes import describe_change


class _Card(QFrame):
    """Shared body; subclasses declare `discarded` and their apply signal."""

    def __init__(self, change, parent=None) -> None:
        super().__init__(parent)
        self.change = change
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        self.body = QLabel(describe_change(change), self)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.body)
        self.status = QLabel('', self)
        self.status.hide()
        layout.addWidget(self.status)
        self.buttons = QHBoxLayout()
        layout.addLayout(self.buttons)
        self.discard_button = QPushButton('Discard', self)
        self.discard_button.clicked.connect(self._discard)

    def _discard(self) -> None:
        self.mark_applied('Discarded.')
        self.discarded.emit(self.change)  # pylint: disable=no-member

    def mark_applied(self, message: str) -> None:
        """Freeze the card and show what happened."""
        for button in self.findChildren(QPushButton):
            button.setEnabled(False)
        self.status.setText(message)
        self.status.show()


class ChangeCard(_Card):
    """A config/settings change set."""

    apply_requested = Signal(object, bool)
    discarded = Signal(object)

    def __init__(self, change, parent=None) -> None:
        super().__init__(change, parent)
        self.apply_button = QPushButton('Apply', self)
        self.apply_new_button = QPushButton('Apply as new config', self)
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit(self.change, False))
        self.apply_new_button.clicked.connect(lambda: self.apply_requested.emit(self.change, True))
        for button in (self.apply_button, self.apply_new_button, self.discard_button):
            self.buttons.addWidget(button)
        if not change.ok:
            self.apply_button.setText('Apply anyway')


class FsCard(_Card):
    """A folder/copy change set."""

    apply_requested = Signal(object)
    discarded = Signal(object)

    def __init__(self, change, parent=None) -> None:
        super().__init__(change, parent)
        self.apply_button = QPushButton('Apply', self)
        self.apply_button.setEnabled(change.ok)
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit(self.change))
        for button in (self.apply_button, self.discard_button):
            self.buttons.addWidget(button)
