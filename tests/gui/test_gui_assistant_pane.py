"""The Assistant pane with a fake session (no network, no pydantic-ai run).

No pytest-qt here: the loop is pumped by `_run_until`, as in
test_gui_plots_pane.py. Nothing may open a modal: every dialog is injected.
"""
from __future__ import annotations

import asyncio
import os

import pytest

pytest.importorskip('PySide6')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.agent.changes import Op, build_changeset  # noqa: E402
from src.agent.events import TurnEvent  # noqa: E402
from src.agent.settings import AgentSettings  # noqa: E402
from src.gui.assistant.pane import AssistantPane  # noqa: E402
from tests.helpers.agent import make_deps  # noqa: E402


def _run_until(signal, timeout_ms=10000):
    loop, got = QEventLoop(), []
    def capture(*args):
        got.append(args)
        loop.quit()
    signal.connect(capture)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    signal.disconnect(capture)
    return got


class FakeToken:
    def __init__(self):
        self.cancelled = False
    def cancel(self):
        self.cancelled = True


class FakeSession:
    def __init__(self, events):
        self.events, self.switched, self.cleared, self.texts = events, [], 0, []
    def new_token(self):
        return FakeToken()
    async def run_turn(self, text, deps, emit, token):
        self.texts.append(text)
        for event in self.events:
            emit(event)
    def switch(self, settings):
        self.switched.append(settings.provider)
    def new_chat(self):
        self.cleared += 1


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


_PANES = []


@pytest.fixture(autouse=True)
def _shutdown_panes():
    yield
    while _PANES:
        _PANES.pop().shutdown()


class SlowSession(FakeSession):
    async def run_turn(self, text, deps, emit, token):
        self.texts.append(text)
        await asyncio.sleep(0.3)
        for event in self.events:
            emit(event)


@pytest.fixture(name='env')
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')


def _pane(events, *, consent=True, settings=None, ask_key=lambda: None, session=None, status_fn=None):
    s = settings or AgentSettings()
    s.gemini.consent_given = consent
    pane = AssistantPane(session or FakeSession(events), lambda: object(), s,
                         save_settings=lambda *a, **k: None, ask_key=ask_key,
                         confirm_consent=lambda: True,
                         **({'status_fn': status_fn} if status_fn else {}))
    _PANES.append(pane)
    return pane


def test_streams_text_and_tool_lines(qt_app, env):
    pane = _pane([TurnEvent('tool_call', 'search_structures(MoS2)'),
                  TurnEvent('text', 'Found '), TurnEvent('text', 'h-MoS2.'),
                  TurnEvent('done', payload=2)])
    pane.send('find MoS2')
    assert _run_until(pane.turn_finished)
    text = pane.chat.toPlainText()
    assert 'find MoS2' in text and 'search_structures(MoS2)' in text and 'Found h-MoS2.' in text
    pane.shutdown()


def test_proposal_card_apply_emits(qt_app, env, tmp_path, mock_settings):
    deps = make_deps(tmp_path, mock_settings)
    change = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    pane = _pane([TurnEvent('proposal', payload=change), TurnEvent('done', payload=1)])
    seen = []
    pane.apply_requested.connect(lambda c, new: seen.append((c, new)))
    pane.send('loads 5')
    _run_until(pane.turn_finished)
    card = pane.cards[change.id]
    card.apply_button.click()
    assert seen == [(change, False)]
    pane.mark_applied(change, 'Applied: loads')
    assert not card.apply_button.isEnabled()
    pane.shutdown()


def test_rate_limited_shows_retry(qt_app, env):
    pane = _pane([TurnEvent('error', 'busy; retry in 0 s', payload='rate_limited', retry_after=0)])
    assert pane.retry_button.isHidden()
    pane.send('x')
    _run_until(pane.turn_finished)
    assert not pane.retry_button.isHidden()
    pane.shutdown()


def test_no_consent_no_turn(qt_app, env):
    session = FakeSession([TurnEvent('done')])
    s = AgentSettings()
    pane = AssistantPane(session, lambda: object(), s, save_settings=lambda *a, **k: None,
                         ask_key=lambda: None, confirm_consent=lambda: False)
    pane.send('x')
    QApplication.processEvents()
    assert session.texts == [] and not s.gemini.consent_given
    pane.shutdown()


def test_missing_key_declined_no_turn(qt_app, env, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY')
    session = FakeSession([TurnEvent('done')])
    pane = AssistantPane(session, lambda: object(), AgentSettings(),
                         save_settings=lambda *a, **k: None, ask_key=lambda: None,
                         confirm_consent=lambda: True)
    pane.send('x')
    QApplication.processEvents()
    assert session.texts == []
    assert 'API key' in pane.chat.toPlainText()
    pane.shutdown()


def test_provider_switch_and_new_chat(qt_app, env, monkeypatch):
    pane = _pane([], status_fn=lambda s: (False, 'Ollama is not running'))
    pane.provider_box.setCurrentIndex(pane.provider_box.findData('ollama'))
    assert pane.session.switched == ['ollama']
    assert 'not running' in pane.status_label.text()
    pane.new_chat()
    assert pane.session.cleared == 1 and pane.chat.toPlainText() == ''
    pane.shutdown()


def test_reply_text_starts_its_own_line(qt_app, env):
    pane = _pane([TurnEvent('tool_call', 'search(MoS2)'), TurnEvent('text', 'Found it.'),
                  TurnEvent('done', payload=1)])
    pane.send('hi')
    _run_until(pane.turn_finished)
    lines = pane.chat.toPlainText().splitlines()
    assert any('search(MoS2)' in ln and 'Found' not in ln for ln in lines)
    assert 'Found it.' in lines


def test_controls_locked_mid_turn(qt_app, env):
    pane = _pane([TurnEvent('done')], session=SlowSession([TurnEvent('done')]))
    pane.send('x')
    for widget in (pane.provider_box, pane.model_edit, pane.new_button):
        assert not widget.isEnabled()
    _run_until(pane.turn_finished)
    for widget in (pane.provider_box, pane.model_edit, pane.new_button):
        assert widget.isEnabled()


def test_shutdown_mid_turn_stops_thread(qt_app, env):
    pane = _pane([], session=SlowSession([]))
    pane.send('x')
    pane.shutdown()
    assert not pane._thread.isRunning()


def test_stop_right_after_send_cancels_token(qt_app, env):
    session = SlowSession([])
    tokens = []
    orig = session.new_token
    session.new_token = lambda: tokens.append(orig()) or tokens[-1]
    pane = _pane([], session=session)
    pane.send('x')
    pane.stop()
    assert tokens and tokens[0].cancelled


def test_new_chat_mid_turn_cancels_and_drops_its_events(qt_app, env, tmp_path, mock_settings):
    deps = make_deps(tmp_path, mock_settings)
    change = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    session = SlowSession([TurnEvent('text', 'old workspace reply'),
                           TurnEvent('proposal', payload=change), TurnEvent('done', payload=1)])
    tokens = []
    orig = session.new_token
    session.new_token = lambda: tokens.append(orig()) or tokens[-1]
    pane = _pane([], session=session)
    pane.send('x')
    pane.new_chat()
    assert tokens[0].cancelled and session.cleared == 1
    assert _run_until(pane.turn_finished)
    assert 'old workspace reply' not in pane.chat.toPlainText()
    assert pane.cards == {} and pane.requests_label.text() == 'requests: 0'
    assert pane.send_button.isEnabled()


def test_save_errors_show_in_chat(qt_app, env, monkeypatch):
    def broken(*_a, **_k):
        raise PermissionError('read-only config')
    s = AgentSettings()
    s.gemini.consent_given = True
    pane = AssistantPane(FakeSession([]), lambda: object(), s, save_settings=broken,
                         ask_key=lambda: None, confirm_consent=lambda: True,
                         status_fn=lambda s: (False, 'down'))
    _PANES.append(pane)
    pane.provider_box.setCurrentIndex(pane.provider_box.findData('ollama'))
    assert 'read-only config' in pane.chat.toPlainText()


def test_key_store_error_shows_in_chat(qt_app, env, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY')
    def broken(_key):
        raise OSError('disk full')
    session = FakeSession([TurnEvent('done')])
    s = AgentSettings()
    s.gemini.consent_given = True
    pane = AssistantPane(session, lambda: object(), s, save_settings=lambda *a, **k: None,
                         ask_key=lambda: 'k', confirm_consent=lambda: True, store_key=broken)
    _PANES.append(pane)
    pane.send('x')
    QApplication.processEvents()
    assert session.texts == [] and 'disk full' in pane.chat.toPlainText()


def test_narrow_layout_stacks_chat_above_cards(qt_app, env):
    from PySide6.QtCore import Qt
    pane = _pane([])
    split = pane.chat.parent()
    assert split.orientation() == Qt.Vertical
    assert split.indexOf(pane.chat) == 0 and split.count() == 2


def test_context_label_is_display_only(qt_app, env):
    session = FakeSession([TurnEvent('done', payload=1)])
    pane = _pane([], session=session)
    pane.set_context_label('Plots')
    assert 'Plots' in pane.context_label.text()
    pane.send('hi')
    _run_until(pane.turn_finished)
    assert session.texts == ['hi']


class LoopBoundSession(FakeSession):
    """Like the real session: an object made in turn 1 is bound to turn 1's loop."""

    def __init__(self, events):
        super().__init__(events)
        self._event = None

    async def run_turn(self, text, deps, emit, token):
        self.texts.append(text)
        if self._event is None:
            self._event = asyncio.Event()
        waiter = asyncio.ensure_future(self._event.wait())
        await asyncio.sleep(0)  # the waiter blocks, binding the Event to this loop
        self._event.set()
        await waiter  # binds the Event to the running loop; a new loop raises
        self._event.clear()
        for event in self.events:
            emit(event)


def test_consecutive_turns_share_the_worker_loop(qt_app, env):
    pane = _pane([], session=LoopBoundSession([TurnEvent('done', payload=1)]))
    for text in ('one', 'two', 'three'):
        pane.send(text)
        assert _run_until(pane.turn_finished)
    assert pane.session.texts == ['one', 'two', 'three']
    assert 'error' not in pane.chat.toPlainText().lower()
    assert pane.requests_label.text() == 'requests: 3'
    loop = pane._worker._runner._loop  # pylint: disable=protected-access
    assert loop is not None and not loop.is_closed()
    pane.shutdown()
    assert loop.is_closed()  # closed on the worker thread as it finished


def test_ollama_not_ready_shows_a_selectable_start_command(qt_app, env):
    from PySide6.QtCore import Qt
    s = AgentSettings(provider='ollama')
    pane = AssistantPane(FakeSession([]), lambda: object(), s,
                         save_settings=lambda *a, **k: None, ask_key=lambda: None,
                         confirm_consent=lambda: True,
                         status_fn=lambda _s: (False, 'Ollama is not running at here.'),
                         start_command_fn=lambda _s: 'OLLAMA_IGPU_ENABLE=1 /x/ollama serve')
    _PANES.append(pane)
    text = pane.status_label.text()
    assert 'not running' in text and 'OLLAMA_IGPU_ENABLE=1 /x/ollama serve' in text
    assert pane.status_label.textInteractionFlags() & Qt.TextSelectableByMouse


def test_ollama_ready_shows_no_start_command(qt_app, env):
    s = AgentSettings(provider='ollama')
    pane = AssistantPane(FakeSession([]), lambda: object(), s,
                         save_settings=lambda *a, **k: None, ask_key=lambda: None,
                         confirm_consent=lambda: True,
                         status_fn=lambda _s: (True, 'Ollama ready (m).'),
                         start_command_fn=lambda _s: 'SERVE-CMD')
    _PANES.append(pane)
    assert 'SERVE-CMD' not in pane.status_label.text()


def test_switch_to_ollama_help_uses_the_start_command(qt_app, env):
    pane = AssistantPane(FakeSession([]), lambda: object(), AgentSettings(),
                         save_settings=lambda *a, **k: None, ask_key=lambda: None,
                         confirm_consent=lambda: True,
                         status_fn=lambda _s: (False, 'Ollama is not running at here.'),
                         start_command_fn=lambda _s: 'SERVE-CMD')
    _PANES.append(pane)
    pane.provider_box.setCurrentIndex(pane.provider_box.findData('ollama'))
    assert 'SERVE-CMD' in pane.chat.toPlainText()
