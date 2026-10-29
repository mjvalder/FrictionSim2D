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

from PySide6.QtCore import Qt, QThread, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import (QColor, QKeySequence, QShortcut, QTextBlockFormat,
                           QTextCharFormat, QTextCursor, QTextFormat)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea, QTextBrowser,
                               QVBoxLayout, QWidget)

from ...agent.changes import FsChangeSet
from ...agent.plot_changes import PlotChange
from ...agent.ollama_server import is_local
from ...agent.providers import (model_installed, ollama_binary, ollama_models,
                                ollama_start_command)
from ...agent.settings import (AgentSettings, api_key, fallback_ollama_model,
                               save_agent_settings, store_api_key)
from .cards import ChangeCard, FsCard, PlotCard
from .worker import AgentWorker

logger = logging.getLogger(__name__)
_ORPHAN_THREADS = []  # keeps a still-running QThread alive past its pane

CONSENT_TEXT = (
    'Gemini runs in Google\'s cloud. Your messages and the assistant\'s tool results '
    '(config contents, file names, and later unpublished results tables) are sent to '
    'Google. On the free tier Google may use them to improve its products.\n\n'
    'Use Gemini for the Assistant?')
ANTHROPIC_CONSENT_TEXT = (
    'Claude runs in Anthropic\'s cloud. Your messages and the assistant\'s tool results '
    '(config contents, file names, and later unpublished results tables) are sent to '
    'Anthropic. API use is billed to your Anthropic account.\n\n'
    'Use Claude for the Assistant?')
#: Per cloud provider: (name in messages, consent text, where to get a key).
CLOUD = {'gemini': ('Gemini', CONSENT_TEXT, 'aistudio.google.com/apikey'),
         'anthropic': ('Claude', ANTHROPIC_CONSENT_TEXT, 'platform.claude.com')}

#: How often the idle check runs (a server the pane started stops when idle).
IDLE_CHECK_MS = 60_000
#: SIGTERM grace for our Ollama server when the window closes.
SHUTDOWN_GRACE = 1.0
#: Characters the model box always shows ("llama3.1:8b" and shorter fit).
MODEL_BOX_CHARS = 14
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
    #: A plot change's card was applied.
    plot_apply_requested = Signal(object)
    #: An applied line's "Show" link: switch to the tab with this title.
    show_tab_requested = Signal(str)
    #: A config/settings change set to apply now (ask-before-applying is off).
    change_proposed = Signal(object)
    #: The Undo link of applied change ``id`` was clicked while enabled.
    undo_requested = Signal(int)
    #: Accept: clear the assistant's highlights.
    accept_requested = Signal()
    #: New chat: the applied lines (and their Undo links) are gone.
    chat_cleared = Signal()
    turn_finished = Signal()
    #: An Ollama start/check finished: whether it is ready for a turn.
    ollama_checked = Signal(bool)
    _start_turn = Signal(str, object, object, int)
    #: A pool task finished: (kind, ok, message, installed models).
    _ollama_result = Signal(str, bool, str, object)

    def __init__(self, session, deps_factory: Callable[[], object],
                 settings: AgentSettings, parent=None, *,
                 save_settings=save_agent_settings,
                 ask_key: Optional[Callable[[], Optional[str]]] = None,
                 confirm_consent: Optional[Callable[[], bool]] = None,
                 store_key=store_api_key,
                 start_command_fn=ollama_start_command,
                 ollama_server, models_fn=ollama_models) -> None:
        super().__init__(parent)
        self.session = session
        self.settings = settings
        self._deps_factory = deps_factory
        self._save_settings = save_settings
        self._store_key = store_key
        self._start_command_fn = start_command_fn
        self._server = ollama_server  # an OllamaServer (the window's) or a test fake
        self._models_fn = models_fn
        self._ollama_ready = False
        self._ollama_starting = False
        self._ollama_checking = False
        self._idle_checking = False
        self._ollama_problem = ''  # why it is not ready, for Details
        self._installed = []
        self._pending_text: Optional[str] = None  # sent once Ollama is up
        self._ask_key = ask_key or self._default_ask_key
        self._confirm_consent = confirm_consent or self._default_confirm_consent
        self._busy = False
        self._text_open = False
        self._last_text = ''
        self._requests = 0
        self._turn_id = 0  # events from any other turn are dropped
        self.cards = {}
        self._undo_states = {}
        self._undo_tips = {}  # entry id -> its enabled Undo's tooltip, when not the default
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
        self._ollama_result.connect(self._on_ollama_result)
        self.idle_timer = QTimer(self)
        self.idle_timer.setInterval(IDLE_CHECK_MS)
        self.idle_timer.timeout.connect(self._check_idle)
        self.idle_timer.start()
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
        self.provider_box.addItem('Ollama (local)', 'ollama')
        self.provider_box.setItemData(1, 'Offline; experimental and slow on a laptop GPU',
                                      Qt.ToolTipRole)
        self.provider_box.addItem('Claude (cloud)', 'anthropic')
        self.provider_box.setItemData(2, 'Anthropic API; needs a paid API key',
                                      Qt.ToolTipRole)
        self.provider_box.setCurrentIndex(self.provider_box.findData(self.settings.provider))
        self.provider_box.currentIndexChanged.connect(self._on_provider_changed)
        self.model_box = QComboBox(self)
        self.model_box.setEditable(True)
        self.model_box.setInsertPolicy(QComboBox.NoInsert)
        self.model_box.setToolTip('Model (Ollama: the installed ones; any name can be typed)')
        self.model_box.setMinimumContentsLength(MODEL_BOX_CHARS)
        self.model_box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.model_box.activated.connect(self._on_model_chosen)
        self.model_box.lineEdit().editingFinished.connect(self._on_model_chosen)
        picker.addWidget(self.provider_box)
        picker.addWidget(self.model_box, 1)
        layout.addLayout(picker)

        status = QHBoxLayout()
        self.cloud_badge = QLabel('cloud: data leaves this machine', self)
        self.status_label = QLabel('', self)
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.check_button = QPushButton('Check again', self)
        self.check_button.setToolTip('Ask Ollama again whether it is up (never starts it)')
        self.check_button.clicked.connect(self._check_ollama)
        self.details_button = QPushButton('Details', self)
        self.details_button.setCheckable(True)
        status.addWidget(self.cloud_badge)
        status.addWidget(self.status_label, 1)
        status.addWidget(self.check_button)
        status.addWidget(self.details_button)
        layout.addLayout(status)

        self.details_box = QFrame(self)
        details = QVBoxLayout(self.details_box)
        details.setContentsMargins(0, 0, 0, 0)
        self.details_label = QLabel('', self.details_box)
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.copy_button = QPushButton('Copy start command', self.details_box)
        self.copy_button.clicked.connect(self._copy_start_command)
        details.addWidget(self.details_label)
        copy_row = QHBoxLayout()
        copy_row.addWidget(self.copy_button)
        copy_row.addStretch(1)
        details.addLayout(copy_row)
        self.details_box.hide()
        self.details_button.toggled.connect(self.details_box.setVisible)
        layout.addWidget(self.details_box)

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

    def _refresh_header(self, check: bool = True) -> None:
        """Model box, badge and status for the current provider.

        ``check``: ask Ollama now (quick, never starts it).
        """
        provider = self.settings.provider
        ollama = provider == 'ollama'
        self.cloud_badge.setVisible(not ollama)
        self.check_button.setVisible(ollama)
        self.details_button.setVisible(ollama)
        if not ollama:
            self.details_button.setChecked(False)
            self._fill_models([])
            name = CLOUD[provider][0]
            self._set_status(f'{name}: key set' if api_key(self.settings, provider)
                             else f'{name}: no API key yet (you will be asked)')
        elif check:
            self._check_ollama()
        else:
            self._fill_models(self._installed)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self._fill_details()

    def _fill_details(self) -> None:
        lines = [self._ollama_problem] if self._ollama_problem else []
        lines += ['The Assistant starts Ollama when needed and stops it after '
                  f'{self.settings.ollama.idle_minutes} min idle. To start it yourself:',
                  self._start_command_fn(self.settings),
                  f'Log: {self._server.log_path}']
        self.details_label.setText('\n'.join(lines))

    def _copy_start_command(self) -> None:
        QApplication.clipboard().setText(self._start_command_fn(self.settings))

    def _fill_models(self, installed) -> None:
        """Offer the installed models (Ollama) or the current one; keep it selected."""
        current = getattr(self.settings, self.settings.provider).model
        names = list(installed) if self.settings.provider == 'ollama' else []
        if current not in names:
            names.append(current)
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItems(names)
        self.model_box.setCurrentText(current)
        self.model_box.blockSignals(False)

    # -------------------------------------------------------------- ollama

    def _run_ollama_task(self, kind: str, work: Callable[[], tuple]) -> None:
        """Run ``work`` -> (ok, message, installed) on a pool thread, never the GUI's."""
        done = self._ollama_result

        def task() -> None:
            try:
                ok, message, installed = work()
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception('Ollama %s failed', kind)
                ok, message, installed = False, f'Ollama {kind} failed: {exc}', None
            try:
                done.emit(kind, ok, message, installed)
            except RuntimeError:  # the pane is gone (window closed meanwhile)
                pass
        QThreadPool.globalInstance().start(task)

    def _check_ollama(self) -> None:
        """Ask a running Ollama for its models (start-up, Check again); never starts it."""
        if self._ollama_starting or self._ollama_checking:
            return
        self._ollama_checking = True
        self._set_status('Checking Ollama…')
        settings, models_fn = self.settings, self._models_fn
        self._run_ollama_task('check', lambda: (True, '', models_fn(settings)))

    def _on_ollama_result(self, kind: str, ok: bool, message: str, installed) -> None:
        if kind == 'start':
            self._on_ollama_started(ok, message, installed)
            return
        if kind == 'check':
            self._ollama_checking = False
            if self.settings.provider == 'ollama' and not self._ollama_starting:
                self._show_ollama(installed)
        elif kind == 'idle':
            self._idle_checking = False
            if not ok:
                return
            self._ollama_ready = False
            if self.settings.provider == 'ollama':
                self._set_status(f'Ollama stopped after {self.settings.ollama.idle_minutes} '
                                 'min idle (starts again on your next message)')
        self.ollama_checked.emit(self._ollama_ready)

    def _show_ollama(self, installed) -> None:
        """Status from the installed model list (None: not running)."""
        if installed is None:
            self._ollama_ready = False
            self._ollama_problem = ''
            self._fill_models(self._installed)
            self._set_status('Ollama not running — starts on your next message'
                             if is_local(self.settings) else
                             f'Ollama at {self.settings.ollama.base_url} is unreachable')
            return
        self._installed = list(installed)
        note = fallback_ollama_model(self.settings, self._installed)
        if note:
            self._save()
            self.session.switch(self.settings)
            self.note(note)
        self._fill_models(self._installed)
        self._update_ollama_ready()

    def _update_ollama_ready(self) -> None:
        model = self.settings.ollama.model
        self._ollama_ready = model_installed(model, self._installed)
        if self._ollama_ready:
            self._ollama_problem = ''
            self._set_status(f'Ollama ready ({model})')
        else:
            self._ollama_problem = (f'Fetch the model once with: '
                                    f'{ollama_binary()} pull {model}')
            self._set_status(f'Ollama is running but has no model {model} — see Details')

    def _ensure_ollama(self, pending_text: Optional[str] = None) -> None:
        """Start Ollama (if needed) on a pool thread; ``pending_text`` is sent after."""
        if pending_text is not None:
            self._pending_text = pending_text
        if self._ollama_starting:
            return
        self._ollama_starting = True
        self._set_status('Starting Ollama…')
        server, settings, models_fn = self._server, self.settings, self._models_fn

        def work() -> tuple:
            ok, message = server.ensure_running(settings)
            installed = models_fn(settings) if ok else None
            if ok and installed is None:
                ok, message = False, 'Ollama started but does not list its models.'
            return ok, message, installed
        self._run_ollama_task('start', work)

    def _on_ollama_started(self, ok: bool, message: str, installed) -> None:
        self._ollama_starting = False
        self._ollama_checking = False
        self._idle_checking = False
        pending, self._pending_text = self._pending_text, None
        if self.settings.provider != 'ollama':
            if pending and not self.input.toPlainText().strip():
                self.input.setPlainText(pending)
            self.ollama_checked.emit(False)
            return
        if ok:
            self._show_ollama(installed)
        else:
            self._ollama_ready = False
            self._ollama_problem = message
            self._set_status('Ollama could not start — see Details')
        if pending:
            if self._ollama_ready:
                self.send(pending)
            else:
                reason = message if not ok else self.status_label.text()
                self._append(f'Not sent: {reason}', 'color:#b00')
                if not self.input.toPlainText().strip():
                    self.input.setPlainText(pending)
        self.ollama_checked.emit(self._ollama_ready)

    def _check_idle(self) -> None:
        """Timer tick: stop our idle server on a pool thread (it waits for the exit)."""
        if self._busy or self._ollama_starting or self._idle_checking:
            return
        self._idle_checking = True
        server = self._server
        self._run_ollama_task('idle', lambda: (server.stop_if_idle(), '', None))

    def _append(self, text: str, style: str = '', indent: int = 0) -> None:
        """A new chat line; ``indent`` non-breaking spaces lead it (HTML would
        collapse plain ones)."""
        self._text_open = False
        self.chat.moveCursor(QTextCursor.End)
        escaped = '&nbsp;' * indent + html.escape(text).replace('\n', '<br>')
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
        ollama = self.settings.provider == 'ollama'
        self._refresh_header(check=not ollama)
        self._append(f'Switched to {self.provider_box.currentText()} (history kept as text only).',
                     'color:gray')
        if ollama:
            self._ensure_ollama()

    def _on_model_chosen(self, *_args) -> None:
        text = self.model_box.currentText().strip()
        provider = getattr(self.settings, self.settings.provider)
        if not text or text == provider.model:
            return
        provider.model = text
        self._save()
        self.session.switch(self.settings)
        if self.settings.provider == 'ollama' and self._installed:
            self._update_ollama_ready()

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
        name = CLOUD[self.settings.provider][0]
        key, ok = QInputDialog.getText(self, f'{name} API key',
                                       f'Paste your {name} API key (stored owner-only in '
                                       '~/.config/FrictionSim2D/agent.secrets):',
                                       QLineEdit.Password)
        return key.strip() if ok and key.strip() else None

    def _default_confirm_consent(self) -> bool:
        name, text, _ = CLOUD[self.settings.provider]
        answer = QMessageBox.question(self, f'Use {name}?', text,
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def _ready(self) -> bool:
        provider = self.settings.provider
        if provider not in CLOUD:
            return self._ollama_ready
        name, _, key_url = CLOUD[provider]
        cloud = getattr(self.settings, provider)
        if not cloud.consent_given:
            if not self._confirm_consent():
                self._append(f'{name} not used: consent declined. Switch to Ollama or ask '
                             'again.', 'color:gray')
                return False
            cloud.consent_given = True
            self._save()
        if not api_key(self.settings, provider):
            key = self._ask_key()
            if not key:
                self._append(f'No {name} API key: get one at {key_url}.', 'color:gray')
                return False
            try:
                self._store_key(key, provider)
            except OSError as exc:
                self._append(f'Could not store the {name} API key: {exc}', 'color:#b00')
                return False
            self.session.switch(self.settings)
            self._refresh_header()
        return True

    # ---------------------------------------------------------------- turns

    def send(self, text: Optional[str] = None) -> None:
        """Start a turn with ``text`` (default: the input box)."""
        text = (text if text is not None else self.input.toPlainText()).strip()
        if not text or self._busy:
            return
        if self.settings.provider == 'ollama' and not self._ollama_ready:
            self.input.clear()
            self.note('Starting Ollama; your message goes as soon as it is up.')
            self._ensure_ollama(text)
            return
        if not self._ready():
            return
        if self.settings.provider == 'ollama':
            self._server.touch()
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
        for widget in (self.provider_box, self.model_box, self.new_button, self.ask_box):
            widget.setEnabled(not busy)

    def _on_event(self, turn_id: int, event) -> None:
        if turn_id != self._turn_id:
            return  # a turn from before New chat / a workspace change
        if event.kind == 'text':
            self._append_inline(event.text)
        elif event.kind == 'tool_call':
            self._append(f'· {event.text}', 'color:gray;font-size:small')
        elif event.kind == 'tool_result':
            self._append(f'→ {event.text}', 'color:gray;font-size:small', indent=2)
        elif event.kind == 'proposal':
            self.receive_proposal(event.payload)
        elif event.kind == 'done':
            self._requests += int(event.payload or 0)
            self.requests_label.setText(f'requests: {self._requests}')
        elif event.kind == 'error':
            self._append(event.text, 'color:#b00')
            if event.payload in ('unreachable', 'unavailable') \
                    and self.settings.provider == 'ollama':
                self._show_ollama(None)  # not ready: the next Send starts it again
            if event.payload == 'rate_limited':
                self.retry_button.setEnabled(False)
                self.retry_button.show()
                QTimer.singleShot(int(event.retry_after * 1000), self.retry_button,
                                  lambda: self.retry_button.setEnabled(True))

    def _on_finished(self, _turn_id: int) -> None:
        self._set_busy(False)  # the worker is free whichever turn this was
        if self.settings.provider == 'ollama':
            self._server.touch()  # idle time counts from the end of a turn
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
        elif isinstance(change, PlotChange):
            card = PlotCard(change)
            card.apply_requested.connect(self.plot_apply_requested.emit)
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

    def add_applied_line(self, entry_id: int, text: str, show_tab: str = '',
                         undo_tip: str = '') -> None:
        """One chat line for an applied change, ending in its Undo link --
        and, with ``show_tab``, a "Show" link that switches to that tab.
        ``undo_tip`` replaces the enabled Undo's tooltip (default: the Builder's)."""
        self.chat.moveCursor(QTextCursor.End)
        cursor = self.chat.textCursor()
        cursor.insertBlock(QTextBlockFormat(), QTextCharFormat())
        plain = QTextCharFormat()
        plain.setForeground(QColor('#2e7d32'))
        cursor.insertText(f'✓ {text}  ', plain)
        self._undo_states[entry_id] = 'enabled'
        if undo_tip:
            self._undo_tips[entry_id] = undo_tip
        cursor.insertText(*self._undo_fragment(entry_id, 'enabled'))
        if show_tab:
            cursor.insertText('  ', QTextCharFormat())
            cursor.insertText('Show', self._link_format(f'show:{show_tab}',
                                                        f'Switch to the {show_tab} tab'))
        # A plain character after the link: text typed or streamed at the end
        # takes the format of the character before it, which must not be the
        # anchor.
        cursor.insertText(' ', QTextCharFormat())
        self.chat.setTextCursor(cursor)
        self._text_open = False

    @staticmethod
    def _link_format(href: str, tip: str) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor('#0b57d0'))
        fmt.setToolTip(tip)
        fmt.setAnchor(True)
        fmt.setAnchorHref(href)
        fmt.setFontUnderline(True)
        return fmt

    def _undo_fragment(self, entry_id: int, state: str):
        text, tip, colour, link = _UNDO_LOOK[state]
        if state == 'enabled':
            tip = self._undo_tips.get(entry_id, tip)
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
        if state == 'enabled' and entry_id in self._undo_tips:
            return self._undo_tips[entry_id]
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
        if url.scheme() == 'show':
            self.show_tab_requested.emit(url.path())
            return
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
        self._undo_tips = {}
        self._fit_pending()
        self.chat_cleared.emit()
        if self.settings.provider == 'ollama':
            self._ensure_ollama()

    def stop_ollama(self) -> None:
        """Stop the Ollama server the pane started (never one it did not)."""
        self.idle_timer.stop()
        try:
            self._server.stop(grace=SHUTDOWN_GRACE)
        except Exception:  # pylint: disable=broad-except
            logger.exception('stopping Ollama failed')

    def shutdown(self) -> None:
        """Stop the worker thread and our Ollama server (window close)."""
        _ORPHAN_THREADS[:] = [t for t in _ORPHAN_THREADS if t.isRunning()]
        self.stop_ollama()
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
