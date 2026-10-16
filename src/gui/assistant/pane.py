"""The Assistant side pane: chat, provider picker, pending cards, Undo lines.

Laid out for a narrow dock beside the main tabs: header rows on top, the chat,
a "Pending" strip of cards (hidden when empty), then the input.

Config and settings changes are handed to the window as they arrive
(`change_proposed`) unless "Ask before applying" is on; file operations always
wait on a card. Each applied change gets one chat line with an Undo link, whose
state the window drives (`set_undo_state`).
"""
from __future__ import annotations

import html
import logging
from typing import Callable, Optional

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (QColor, QKeySequence, QShortcut, QTextBlockFormat,
                           QTextCharFormat, QTextCursor, QTextFormat)
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QTextBrowser,
                               QVBoxLayout, QWidget)

from ...agent.changes import FsChangeSet
from ...agent.providers import ollama_setup_help, ollama_start_command, ollama_status
from ...agent.settings import AgentSettings, gemini_key, save_agent_settings, store_gemini_key
from .cards import ChangeCard, FsCard
from .worker import AgentWorker

logger = logging.getLogger(__name__)
_ORPHAN_THREADS = []  # keeps a still-running QThread alive past its pane

CONSENT_TEXT = (
    'Gemini runs in Google\'s cloud. Your messages and the assistant\'s tool results '
    '(config contents, file names, and later unpublished results tables) are sent to '
    'Google. On the free tier Google may use them to improve its products.\n\n'
    'Use Gemini for the Assistant?')

#: Cards shown in the Pending strip before it scrolls.
PENDING_VISIBLE = 3
#: Char-format property tagging an applied line's Undo control with its id.
_UNDO_ID = QTextFormat.Property.UserProperty.value + 1
#: (text, tooltip, colour, is a link) per Undo state.
_UNDO_LOOK = {
    'enabled': ('Undo', 'Put the Builder back as it was before this change', '#0b57d0', True),
    'blocked': ('Undo', 'undo newer changes first', 'gray', False),
    'undone': ('undone', 'This change was undone', 'gray', False),
    'locked': ('Undo unavailable', 'The config was saved (or replaced) since; '
               'undo no longer applies', 'gray', False),
}


class AssistantPane(QWidget):  # pylint: disable=too-many-instance-attributes
    """Chat with the assistant; its changes land directly or wait as cards."""

    apply_requested = Signal(object, bool)
    fs_apply_requested = Signal(object)
    #: A config/settings change set to apply now (ask-before-applying is off).
    change_proposed = Signal(object)
    #: The Undo link of applied change ``id`` was clicked while enabled.
    undo_requested = Signal(int)
    #: Accept: clear the assistant's highlights.
    accept_requested = Signal()
    #: New chat: the applied lines (and their Undo links) are gone.
    chat_cleared = Signal()
    turn_finished = Signal()
    _start_turn = Signal(str, object, object, int)

    def __init__(self, session, deps_factory: Callable[[], object],
                 settings: AgentSettings, parent=None, *,
                 save_settings=save_agent_settings, status_fn=ollama_status,
                 ask_key: Optional[Callable[[], Optional[str]]] = None,
                 confirm_consent: Optional[Callable[[], bool]] = None,
                 store_key=store_gemini_key,
                 start_command_fn=ollama_start_command) -> None:
        super().__init__(parent)
        self.session = session
        self.settings = settings
        self._deps_factory = deps_factory
        self._save_settings = save_settings
        self._store_key = store_key
        self._status_fn = status_fn
        self._start_command_fn = start_command_fn
        self._ollama_ready = False
        self._ask_key = ask_key or self._default_ask_key
        self._confirm_consent = confirm_consent or self._default_confirm_consent
        self._busy = False
        self._text_open = False
        self._last_text = ''
        self._requests = 0
        self._turn_id = 0  # events from any other turn are dropped
        self.cards = {}
        self._undo_states = {}
        self._build_ui()
        self._thread = QThread(self)
        self._worker = AgentWorker(session)
        self._worker.moveToThread(self._thread)
        self._start_turn.connect(self._worker.run_turn)
        self._worker.event.connect(self._on_event)
        self._worker.finished.connect(self._on_finished)
        # Direct: `finished` is emitted on the worker thread, whose own event
        # loop has stopped, so a queued call would never run.
        self._thread.finished.connect(self._worker.close, Qt.DirectConnection)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()
        self._refresh_header()

    # ------------------------------------------------------------------ ui

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        self.context_label = QLabel('', self)
        self.context_label.setToolTip('The tab you are on (shown here only).')
        self.requests_label = QLabel('requests: 0', self)
        self.ask_box = QCheckBox('Ask before applying', self)
        self.ask_box.setToolTip('Off: the assistant\'s config and settings changes land in '
                                'the Builder at once (yellow, with Undo). On: they wait as '
                                'cards. File operations always wait.')
        self.ask_box.setChecked(self.settings.ask_before_apply)
        self.ask_box.toggled.connect(self._on_ask_toggled)
        top.addWidget(self.context_label)
        top.addStretch(1)
        top.addWidget(self.requests_label)
        layout.addLayout(top)
        # Its own row, with Accept: the dock is narrow and must stay so.
        review = QHBoxLayout()
        self.accept_button = QPushButton('Accept', self)
        self.accept_button.setToolTip('Clear the yellow marks on fields the assistant set')
        self.accept_button.setEnabled(False)
        self.accept_button.clicked.connect(self.accept_requested.emit)
        review.addWidget(self.ask_box)
        review.addStretch(1)
        review.addWidget(self.accept_button)
        layout.addLayout(review)

        picker = QHBoxLayout()
        self.provider_box = QComboBox(self)
        self.provider_box.addItem('Gemini (cloud)', 'gemini')
        self.provider_box.addItem('Ollama (local, experimental)', 'ollama')
        self.provider_box.setCurrentIndex(self.provider_box.findData(self.settings.provider))
        self.provider_box.currentIndexChanged.connect(self._on_provider_changed)
        self.model_edit = QLineEdit(self)
        self.model_edit.setToolTip('Model')
        self.model_edit.editingFinished.connect(self._on_model_edited)
        picker.addWidget(self.provider_box)
        picker.addWidget(self.model_edit, 1)
        layout.addLayout(picker)

        status = QHBoxLayout()
        self.cloud_badge = QLabel('cloud: data leaves this machine', self)
        self.status_label = QLabel('', self)
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status.addWidget(self.cloud_badge)
        status.addWidget(self.status_label, 1)
        layout.addLayout(status)

        self.chat = QTextBrowser(self)
        self.chat.setOpenExternalLinks(False)
        self.chat.setOpenLinks(False)  # undo: links are handled, never followed
        self.chat.anchorClicked.connect(self._on_anchor)
        layout.addWidget(self.chat, 1)

        self.pending_box = QFrame(self)
        pending = QVBoxLayout(self.pending_box)
        pending.setContentsMargins(0, 0, 0, 0)
        pending.addWidget(QLabel('Pending', self.pending_box))
        card_host = QWidget()
        self._card_layout = QVBoxLayout(card_host)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self.pending_scroll = QScrollArea(self.pending_box)
        self.pending_scroll.setWidgetResizable(True)
        self.pending_scroll.setWidget(card_host)
        self.pending_scroll.setFrameShape(QFrame.NoFrame)
        pending.addWidget(self.pending_scroll)
        self.pending_box.hide()
        layout.addWidget(self.pending_box)

        self.input = QPlainTextEdit(self)
        self.input.setPlaceholderText('Ask the assistant (Ctrl+Enter to send)')
        self.input.setMaximumHeight(90)
        layout.addWidget(self.input)
        row = QHBoxLayout()
        self.send_button = QPushButton('Send', self)
        self.stop_button = QPushButton('Stop', self)
        self.retry_button = QPushButton('Retry', self)
        self.new_button = QPushButton('New chat', self)
        self.send_button.clicked.connect(lambda: self.send())
        self.stop_button.clicked.connect(self.stop)
        self.retry_button.clicked.connect(lambda: self.send(self._last_text))
        self.new_button.clicked.connect(self.new_chat)
        self.retry_button.hide()
        self.stop_button.setEnabled(False)
        for button in (self.send_button, self.stop_button, self.retry_button, self.new_button):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        QShortcut(QKeySequence('Ctrl+Return'), self.input, activated=self.send)

    def set_context_label(self, text: str) -> None:
        """Show which main tab is active. Display only: never sent to the model."""
        self.context_label.setText(f'On: {text}' if text else '')

    def _refresh_header(self) -> None:
        provider = self.settings.provider
        model = getattr(self.settings, provider).model
        self.model_edit.setText(model)
        self.cloud_badge.setVisible(provider == 'gemini')
        if provider == 'gemini':
            self.status_label.setText('key set' if gemini_key(self.settings)
                                      else 'no API key yet (you will be asked)')
        else:
            self._ollama_ready, message = self._status_fn(self.settings)
            if not self._ollama_ready:
                message += ('\nIf it is not running, start it in a terminal with:\n'
                            + self._start_command_fn(self.settings))
            self.status_label.setText(message)

    def _append(self, text: str, style: str = '') -> None:
        self._text_open = False
        self.chat.moveCursor(QTextCursor.End)
        escaped = html.escape(text).replace('\n', '<br>')
        self.chat.append(f'<span style="{style}">{escaped}</span>' if style else escaped)

    def _append_inline(self, text: str) -> None:
        self.chat.moveCursor(QTextCursor.End)
        if not self._text_open:
            cursor = self.chat.textCursor()
            cursor.insertBlock(QTextBlockFormat(), QTextCharFormat())
            self.chat.setTextCursor(cursor)
            self._text_open = True
        self.chat.insertPlainText(text)

    # ------------------------------------------------------------ provider

    def _on_provider_changed(self) -> None:
        self.settings.provider = self.provider_box.currentData()
        self._save()
        self.session.switch(self.settings)
        self._refresh_header()
        self._append(f'Switched to {self.provider_box.currentText()} (history kept as text only).',
                     'color:gray')
        if self.settings.provider == 'ollama' and not self._ollama_ready:
            help_text = ollama_setup_help(
                self.settings, start_command=self._start_command_fn(self.settings))
            self._append(help_text, 'color:gray')

    def _on_model_edited(self) -> None:
        provider = getattr(self.settings, self.settings.provider)
        if self.model_edit.text().strip() and self.model_edit.text().strip() != provider.model:
            provider.model = self.model_edit.text().strip()
            self._save()
            self.session.switch(self.settings)

    def _on_ask_toggled(self, checked: bool) -> None:
        self.settings.ask_before_apply = checked
        self._save()

    def _save(self) -> None:
        """Save the assistant settings; a file error is shown, not raised."""
        try:
            self._save_settings(self.settings)
        except OSError as exc:
            self._append(f'Could not save the assistant settings: {exc}', 'color:#b00')

    def _default_ask_key(self) -> Optional[str]:
        key, ok = QInputDialog.getText(self, 'Gemini API key',
                                       'Paste your Gemini API key (stored owner-only in '
                                       '~/.config/FrictionSim2D/agent.secrets):',
                                       QLineEdit.Password)
        return key.strip() if ok and key.strip() else None

    def _default_confirm_consent(self) -> bool:
        answer = QMessageBox.question(self, 'Use Gemini?', CONSENT_TEXT,
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def _ready(self) -> bool:
        if self.settings.provider != 'gemini':
            return True
        if not self.settings.gemini.consent_given:
            if not self._confirm_consent():
                self._append('Gemini not used: consent declined. Switch to Ollama or ask again.',
                             'color:gray')
                return False
            self.settings.gemini.consent_given = True
            self._save()
        if not gemini_key(self.settings):
            key = self._ask_key()
            if not key:
                self._append('No Gemini API key: get one at aistudio.google.com/apikey.',
                             'color:gray')
                return False
            try:
                self._store_key(key)
            except OSError as exc:
                self._append(f'Could not store the Gemini API key: {exc}', 'color:#b00')
                return False
            self.session.switch(self.settings)
            self._refresh_header()
        return True

    # ---------------------------------------------------------------- turns

    def send(self, text: Optional[str] = None) -> None:
        """Start a turn with ``text`` (default: the input box)."""
        text = (text if text is not None else self.input.toPlainText()).strip()
        if not text or self._busy or not self._ready():
            return
        self.input.clear()
        self._last_text = text
        self.retry_button.hide()
        self._append(f'You: {text}', 'font-weight:bold')
        self._set_busy(True)
        token = self.session.new_token()
        self._worker.arm(token)
        self._turn_id += 1
        deps = self._deps_factory()
        if hasattr(deps, 'direct_apply'):
            deps.direct_apply = not self.settings.ask_before_apply
        self._start_turn.emit(text, deps, token, self._turn_id)

    def stop(self) -> None:
        """Cancel the running turn."""
        self._worker.cancel()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.send_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        for widget in (self.provider_box, self.model_edit, self.new_button, self.ask_box):
            widget.setEnabled(not busy)

    def _on_event(self, turn_id: int, event) -> None:
        if turn_id != self._turn_id:
            return  # a turn from before New chat / a workspace change
        if event.kind == 'text':
            self._append_inline(event.text)
        elif event.kind == 'tool_call':
            self._append(f'· {event.text}', 'color:gray;font-size:small')
        elif event.kind == 'tool_result':
            self._append(f'  → {event.text}', 'color:gray;font-size:small')
        elif event.kind == 'proposal':
            self.receive_proposal(event.payload)
        elif event.kind == 'done':
            self._requests += int(event.payload or 0)
            self.requests_label.setText(f'requests: {self._requests}')
        elif event.kind == 'error':
            self._append(event.text, 'color:#b00')
            if event.payload == 'rate_limited':
                self.retry_button.setEnabled(False)
                self.retry_button.show()
                QTimer.singleShot(int(event.retry_after * 1000), self.retry_button,
                                  lambda: self.retry_button.setEnabled(True))

    def _on_finished(self, _turn_id: int) -> None:
        self._set_busy(False)  # the worker is free whichever turn this was
        self.turn_finished.emit()

    # ------------------------------------------------------------ changes

    def receive_proposal(self, change) -> None:
        """Route a change set: file operations (and everything, when asking
        first) wait as a card; a config/settings change goes to the window."""
        if isinstance(change, FsChangeSet) or self.settings.ask_before_apply:
            self._add_card(change)
        else:
            self.change_proposed.emit(change)

    def queue_card(self, change, note: str = '') -> None:
        """Keep a change for the user to apply, whatever the apply mode."""
        self._add_card(change)
        if note:
            self.note(note)

    def note(self, text: str, error: bool = False) -> None:
        """One line in the chat: gray, or red for something that failed."""
        self._append(text, 'color:#b00' if error else 'color:gray')

    def _add_card(self, change) -> None:
        if isinstance(change, FsChangeSet):
            card = FsCard(change)
            card.apply_requested.connect(self.fs_apply_requested.emit)
        else:
            card = ChangeCard(change)
            card.apply_requested.connect(self.apply_requested.emit)
        card.discarded.connect(self._on_discarded)
        self.cards[change.id] = card
        self._card_layout.addWidget(card)
        self._fit_pending()

    def _on_discarded(self, change) -> None:
        self.finish_card(change, f'Discarded: {change.title}')

    def finish_card(self, change, message: str, error: bool = False) -> None:
        """Take a card out of the Pending strip; ``message`` goes to the chat."""
        card = self.cards.pop(change.id, None)
        if card is not None:
            self._card_layout.removeWidget(card)
            card.hide()
            card.deleteLater()
            self._fit_pending()
        if message:
            self.note(message, error)

    def replace_change(self, old, new, note: str) -> None:
        """Swap a stale card for a re-checked one."""
        self.finish_card(old, '')
        self.note(note)
        self._add_card(new)

    def _fit_pending(self) -> None:
        """Show the strip only with cards in it, sized to (up to three of) them."""
        cards = list(self.cards.values())
        self.pending_box.setVisible(bool(cards))
        if not cards:
            return
        shown = cards[:PENDING_VISIBLE]
        spacing = max(self._card_layout.spacing(), 0)
        height = sum(card.sizeHint().height() for card in shown) + spacing * (len(shown) - 1)
        self.pending_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff if len(cards) <= PENDING_VISIBLE else Qt.ScrollBarAsNeeded)
        self.pending_scroll.setFixedHeight(height + 2 * self.pending_scroll.frameWidth())

    def set_accept_enabled(self, enabled: bool) -> None:
        """Whether there are highlights for Accept to clear."""
        self.accept_button.setEnabled(enabled)

    # ------------------------------------------------------------ undo lines

    def add_applied_line(self, entry_id: int, text: str) -> None:
        """One chat line for an applied change, ending in its Undo link."""
        self.chat.moveCursor(QTextCursor.End)
        cursor = self.chat.textCursor()
        cursor.insertBlock(QTextBlockFormat(), QTextCharFormat())
        plain = QTextCharFormat()
        plain.setForeground(QColor('#2e7d32'))
        cursor.insertText(f'✓ {text}  ', plain)
        self._undo_states[entry_id] = 'enabled'
        cursor.insertText(*self._undo_fragment(entry_id, 'enabled'))
        # A plain character after the link: text typed or streamed at the end
        # takes the format of the character before it, which must not be the
        # anchor.
        cursor.insertText(' ', QTextCharFormat())
        self.chat.setTextCursor(cursor)
        self._text_open = False

    @staticmethod
    def _undo_fragment(entry_id: int, state: str):
        text, tip, colour, link = _UNDO_LOOK[state]
        fmt = QTextCharFormat()
        fmt.setProperty(_UNDO_ID, entry_id)
        fmt.setForeground(QColor(colour))
        fmt.setToolTip(tip)
        if link:
            fmt.setAnchor(True)
            fmt.setAnchorHref(f'undo:{entry_id}')
            fmt.setFontUnderline(True)
        return text, fmt

    def set_undo_state(self, entry_id: int, state: str) -> None:
        """'enabled', 'blocked' (a newer change is on top), 'undone' or 'locked'."""
        if self._undo_states.get(entry_id) in (None, state):
            return
        span = self._find_undo(entry_id)
        if span is None:
            return
        self._undo_states[entry_id] = state
        cursor = QTextCursor(self.chat.document())
        cursor.setPosition(span[0])
        cursor.setPosition(span[1], QTextCursor.KeepAnchor)
        cursor.insertText(*self._undo_fragment(entry_id, state))

    def undo_state(self, entry_id: int) -> Optional[str]:
        """The Undo state of an applied line, or None when it is not shown."""
        return self._undo_states.get(entry_id)

    def undo_tooltip(self, entry_id: int) -> str:
        """The tooltip on an applied line's Undo control ('' when not shown)."""
        state = self._undo_states.get(entry_id)
        return _UNDO_LOOK[state][1] if state else ''

    def _find_undo(self, entry_id: int):
        block = self.chat.document().begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                if fragment.charFormat().property(_UNDO_ID) == entry_id:
                    return fragment.position(), fragment.position() + fragment.length()
                it += 1
            block = block.next()
        return None

    def _on_anchor(self, url: QUrl) -> None:
        if url.scheme() != 'undo':
            return
        try:
            entry_id = int(url.path())
        except ValueError:
            return
        if self._undo_states.get(entry_id) == 'enabled':
            self.undo_requested.emit(entry_id)

    def new_chat(self) -> None:
        """Cancel a running turn, forget the conversation and clear the cards."""
        self._worker.cancel()
        self._turn_id += 1
        self.session.new_chat()
        self.chat.clear()
        for card in self.cards.values():
            card.deleteLater()
        self.cards = {}
        self._undo_states = {}
        self._fit_pending()
        self.chat_cleared.emit()

    def shutdown(self) -> None:
        """Stop the worker thread (window close)."""
        self._worker.cancel()
        self._thread.quit()
        if self._thread.wait(3000):
            return
        logger.warning('assistant worker thread did not stop within 3 s')
        if self._worker.busy:
            self._thread.wait()  # the cancelled turn ends at its next await
        if self._thread.isRunning():
            self._thread.setParent(None)
            _ORPHAN_THREADS.append(self._thread)
