"""The Assistant tab: chat, provider picker, proposal cards."""
from __future__ import annotations

import html
import logging
from typing import Callable, Optional

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import (QKeySequence, QShortcut, QTextBlockFormat, QTextCharFormat,
                           QTextCursor)
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QInputDialog, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QSplitter, QTextBrowser,
                               QVBoxLayout, QWidget)

from ...agent.changes import FsChangeSet
from ...agent.providers import OLLAMA_SETUP_HELP, ollama_modelfile, ollama_status
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


class AssistantPane(QWidget):  # pylint: disable=too-many-instance-attributes
    """Chat with the assistant; its proposals appear as cards to apply."""

    apply_requested = Signal(object, bool)
    fs_apply_requested = Signal(object)
    turn_finished = Signal()
    _start_turn = Signal(str, object, object, int)

    def __init__(self, session, deps_factory: Callable[[], object],
                 settings: AgentSettings, parent=None, *,
                 save_settings=save_agent_settings, status_fn=ollama_status,
                 ask_key: Optional[Callable[[], Optional[str]]] = None,
                 confirm_consent: Optional[Callable[[], bool]] = None,
                 store_key=store_gemini_key) -> None:
        super().__init__(parent)
        self.session = session
        self.settings = settings
        self._deps_factory = deps_factory
        self._save_settings = save_settings
        self._store_key = store_key
        self._status_fn = status_fn
        self._ask_key = ask_key or self._default_ask_key
        self._confirm_consent = confirm_consent or self._default_confirm_consent
        self._busy = False
        self._text_open = False
        self._last_text = ''
        self._requests = 0
        self._turn_id = 0  # events from any other turn are dropped
        self.cards = {}
        self._build_ui()
        self._thread = QThread(self)
        self._worker = AgentWorker(session)
        self._worker.moveToThread(self._thread)
        self._start_turn.connect(self._worker.run_turn)
        self._worker.event.connect(self._on_event)
        self._worker.finished.connect(self._on_finished)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()
        self._refresh_header()

    # ------------------------------------------------------------------ ui

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.provider_box = QComboBox(self)
        self.provider_box.addItem('Gemini (cloud)', 'gemini')
        self.provider_box.addItem('Ollama (local, experimental)', 'ollama')
        self.provider_box.setCurrentIndex(self.provider_box.findData(self.settings.provider))
        self.provider_box.currentIndexChanged.connect(self._on_provider_changed)
        self.model_edit = QLineEdit(self)
        self.model_edit.editingFinished.connect(self._on_model_edited)
        self.cloud_badge = QLabel('cloud: data leaves this machine', self)
        self.status_label = QLabel('', self)
        self.requests_label = QLabel('requests: 0', self)
        for widget in (QLabel('Provider:', self), self.provider_box, QLabel('Model:', self),
                       self.model_edit, self.cloud_badge, self.status_label):
            header.addWidget(widget)
        header.addStretch(1)
        header.addWidget(self.requests_label)
        layout.addLayout(header)

        split = QSplitter(self)
        self.chat = QTextBrowser(split)
        self.chat.setOpenExternalLinks(False)
        card_host = QWidget()
        self._card_layout = QVBoxLayout(card_host)
        self._card_layout.addStretch(1)
        scroll = QScrollArea(split)
        scroll.setWidgetResizable(True)
        scroll.setWidget(card_host)
        layout.addWidget(split, 1)

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

    def _refresh_header(self) -> None:
        provider = self.settings.provider
        model = getattr(self.settings, provider).model
        self.model_edit.setText(model)
        self.cloud_badge.setVisible(provider == 'gemini')
        if provider == 'gemini':
            self.status_label.setText('key set' if gemini_key(self.settings)
                                      else 'no API key yet (you will be asked)')
        else:
            self.status_label.setText(self._status_fn(self.settings)[1])

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
        if self.settings.provider == 'ollama' and 'ready' not in self.status_label.text():
            self._append(OLLAMA_SETUP_HELP + '\n\nModelfile:\n' + ollama_modelfile(self.settings),
                         'color:gray')

    def _on_model_edited(self) -> None:
        provider = getattr(self.settings, self.settings.provider)
        if self.model_edit.text().strip() and self.model_edit.text().strip() != provider.model:
            provider.model = self.model_edit.text().strip()
            self._save()
            self.session.switch(self.settings)

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
        self._start_turn.emit(text, self._deps_factory(), token, self._turn_id)

    def stop(self) -> None:
        """Cancel the running turn."""
        self._worker.cancel()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.send_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        for widget in (self.provider_box, self.model_edit, self.new_button):
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
            self._add_card(event.payload)
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

    # ---------------------------------------------------------------- cards

    def _add_card(self, change) -> None:
        if isinstance(change, FsChangeSet):
            card = FsCard(change)
            card.apply_requested.connect(self.fs_apply_requested.emit)
        else:
            card = ChangeCard(change)
            card.apply_requested.connect(self.apply_requested.emit)
        self.cards[change.id] = card
        self._card_layout.insertWidget(self._card_layout.count() - 1, card)

    def mark_applied(self, change, message: str) -> None:
        """Freeze a card after Apply (or a refusal) with what happened."""
        card = self.cards.get(change.id)
        if card is not None:
            card.mark_applied(message)

    def replace_change(self, old, new, note: str) -> None:
        """Swap a stale card for a re-checked one."""
        self.mark_applied(old, 'Superseded by a re-check.')
        self._append(note, 'color:gray')
        self._add_card(new)

    def new_chat(self) -> None:
        """Cancel a running turn, forget the conversation and clear the cards."""
        self._worker.cancel()
        self._turn_id += 1
        self.session.new_chat()
        self.chat.clear()
        for card in self.cards.values():
            card.deleteLater()
        self.cards = {}

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
