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


def _pane(events, *, consent=True, settings=None, ask_key=lambda: None, session=None):
    s = settings or AgentSettings()
    s.gemini.consent_given = consent
    pane = AssistantPane(session or FakeSession(events), lambda: object(), s,
                         save_settings=lambda *a, **k: None, ask_key=ask_key,
                         confirm_consent=lambda: True)
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
    monkeypatch.setattr('src.gui.assistant.pane.ollama_status',
                        lambda s: (False, 'Ollama is not running'))
    pane = _pane([])
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
