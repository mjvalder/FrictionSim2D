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

from frictionsim2d.agent.changes import Op, build_changeset  # noqa: E402
from frictionsim2d.agent.events import TurnEvent  # noqa: E402
from frictionsim2d.agent.settings import AgentSettings  # noqa: E402
from frictionsim2d.gui.assistant.pane import AssistantPane  # noqa: E402
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


class FakeServer:
    """Stands in for OllamaServer: never starts a process."""

    def __init__(self, ok=True, message='Ollama started.', on_ensure=None):
        self.ok, self.message, self.on_ensure = ok, message, on_ensure
        self.ensured = self.touched = self.stopped = 0
        self.idle = False
        self.log_path = '/fake/cache/ollama.log'

    def ensure_running(self, settings):
        self.ensured += 1
        if self.on_ensure:
            self.on_ensure()
        return self.ok, self.message

    def touch(self):
        self.touched += 1

    def stop_if_idle(self, now=None):
        return self.idle

    def stop(self, grace=5.0):
        self.stopped += 1
        self.grace = grace


def _settings(**kwargs):
    """Settings naming a stock model these tests install: no fine-tune involved."""
    s = AgentSettings(**kwargs)
    s.ollama.model = 'qwen3:4b'
    return s


def _pane(events, *, consent=True, settings=None, ask_key=lambda: None, session=None,
          server=None, models_fn=lambda s: None, save_settings=lambda *a, **k: None,
          schema_hash_fn=lambda: 'b8b78921bc6dd497', **kwargs):
    s = settings or _settings()
    s.gemini.consent_given = consent
    pane = AssistantPane(session or FakeSession(events), lambda: object(), s,
                         save_settings=save_settings, ask_key=ask_key,
                         confirm_consent=lambda: True, ollama_server=server or FakeServer(),
                         models_fn=models_fn, schema_hash_fn=schema_hash_fn, **kwargs)
    _PANES.append(pane)
    if s.provider == 'ollama':
        assert _run_until(pane.ollama_checked)  # the status check runs off the GUI thread
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


def test_config_proposal_is_handed_over_without_a_card(qt_app, env, tmp_path, mock_settings):
    deps = make_deps(tmp_path, mock_settings)
    change = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    pane = _pane([TurnEvent('proposal', payload=change), TurnEvent('done', payload=1)])
    seen = []
    pane.change_proposed.connect(seen.append)
    pane.send('loads 5')
    _run_until(pane.turn_finished)
    assert seen == [change] and pane.cards == {}
    assert pane.pending_box.isHidden()


def test_ask_before_apply_shows_a_card_in_the_pending_strip(qt_app, env, tmp_path, mock_settings):
    deps = make_deps(tmp_path, mock_settings)
    change = build_changeset(deps, 'loads', [Op('set', 'general', 'force', [5])])
    pane = _pane([TurnEvent('proposal', payload=change), TurnEvent('done', payload=1)],
                 settings=AgentSettings(ask_before_apply=True))
    direct, seen = [], []
    pane.change_proposed.connect(direct.append)
    pane.apply_requested.connect(lambda c, new: seen.append((c, new)))
    pane.send('loads 5')
    _run_until(pane.turn_finished)
    assert direct == [] and not pane.pending_box.isHidden()
    pane.cards[change.id].apply_button.click()
    assert seen == [(change, False)]
    pane.finish_card(change, '')
    assert pane.cards == {} and pane.pending_box.isHidden()


def test_ask_switch_is_saved(qt_app, env):
    saved = []
    s = AgentSettings()
    s.gemini.consent_given = True
    pane = AssistantPane(FakeSession([]), lambda: object(), s,
                         save_settings=lambda settings: saved.append(settings.ask_before_apply),
                         ask_key=lambda: None, confirm_consent=lambda: True,
                         ollama_server=FakeServer())
    _PANES.append(pane)
    assert not pane.ask_box.isChecked()
    pane.ask_box.setChecked(True)
    assert s.ask_before_apply is True and saved == [True]


def test_ask_switch_locked_mid_turn(qt_app, env):
    pane = _pane([], session=SlowSession([TurnEvent('done')]))
    pane.send('x')
    assert not pane.ask_box.isEnabled()
    _run_until(pane.turn_finished)
    assert pane.ask_box.isEnabled()


def test_send_tells_the_deps_the_apply_mode(qt_app, env, tmp_path, mock_settings):
    seen = []

    class Recording(FakeSession):
        async def run_turn(self, text, deps, emit, token):
            seen.append(deps.direct_apply)
    s = AgentSettings()
    s.gemini.consent_given = True
    pane = AssistantPane(Recording([]), lambda: make_deps(tmp_path, mock_settings), s,
                         save_settings=lambda *a, **k: None, ask_key=lambda: None,
                         confirm_consent=lambda: True, ollama_server=FakeServer())
    _PANES.append(pane)
    pane.send('a')
    _run_until(pane.turn_finished)
    pane.ask_box.setChecked(True)
    pane.send('b')
    _run_until(pane.turn_finished)
    assert seen == [True, False]


def test_file_card_discard_removes_it_and_says_so(qt_app, env, tmp_path, mock_settings):
    from frictionsim2d.agent.changes import build_fs_changeset
    change = build_fs_changeset(make_deps(tmp_path, mock_settings), 'mk', [('mkdir', 'materials')])
    pane = _pane([])
    assert pane.pending_box.isHidden()
    pane.receive_proposal(change)
    assert not pane.pending_box.isHidden() and change.id in pane.cards
    pane.cards[change.id].discard_button.click()
    assert pane.cards == {} and pane.pending_box.isHidden()
    assert 'Discarded: mk' in pane.chat.toPlainText()


def test_pending_strip_scrolls_only_past_three_cards(qt_app, env, tmp_path, mock_settings):
    from PySide6.QtCore import Qt
    from frictionsim2d.agent.changes import build_fs_changeset
    deps = make_deps(tmp_path, mock_settings)
    pane = _pane([])
    changes = [build_fs_changeset(deps, f'mk{i}', [('mkdir', f'd{i}')]) for i in range(4)]
    for change in changes[:3]:
        pane.receive_proposal(change)
    assert pane.pending_scroll.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    pane.receive_proposal(changes[3])
    assert pane.pending_scroll.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded


def test_applied_lines_offer_undo_for_the_latest_only(qt_app, env):
    from PySide6.QtCore import QUrl
    pane = _pane([])
    asked = []
    pane.undo_requested.connect(asked.append)
    pane.add_applied_line(1, 'Set 1 field: [general] force')
    pane.add_applied_line(2, 'Set 1 field: [general] temp')
    pane.set_undo_state(1, 'blocked')
    assert pane.undo_state(1) == 'blocked' and pane.undo_state(2) == 'enabled'
    pane.chat.anchorClicked.emit(QUrl('undo:1'))
    assert asked == []
    pane.chat.anchorClicked.emit(QUrl('undo:2'))
    assert asked == [2]
    pane.set_undo_state(2, 'undone')
    text = pane.chat.toPlainText()
    assert 'Set 1 field: [general] force' in text and 'undone' in text
    assert pane.chat.toPlainText().count('undo newer changes first') == 0  # tooltip only
    assert pane.undo_tooltip(1) == 'undo newer changes first'


def test_accept_button_emits(qt_app, env):
    pane = _pane([])
    accepted = []
    pane.accept_requested.connect(lambda: accepted.append(1))
    assert not pane.accept_button.isEnabled()
    pane.set_accept_enabled(True)
    pane.accept_button.click()
    assert accepted == [1]


def test_new_chat_announces_it(qt_app, env):
    pane = _pane([])
    cleared = []
    pane.chat_cleared.connect(lambda: cleared.append(1))
    pane.new_chat()
    assert cleared == [1]


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
                         ask_key=lambda: None, confirm_consent=lambda: False,
                         ollama_server=FakeServer())
    pane.send('x')
    QApplication.processEvents()
    assert session.texts == [] and not s.gemini.consent_given
    pane.shutdown()


def test_missing_key_declined_no_turn(qt_app, env, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY')
    session = FakeSession([TurnEvent('done')])
    pane = AssistantPane(session, lambda: object(), AgentSettings(),
                         save_settings=lambda *a, **k: None, ask_key=lambda: None,
                         confirm_consent=lambda: True, ollama_server=FakeServer())
    pane.send('x')
    QApplication.processEvents()
    assert session.texts == []
    assert 'API key' in pane.chat.toPlainText()
    pane.shutdown()


def test_provider_switch_and_new_chat(qt_app, env):
    pane = _pane([], server=FakeServer(ok=False, message='Ollama is not installed'))
    pane.provider_box.setCurrentIndex(pane.provider_box.findData('ollama'))
    assert pane.session.switched == ['ollama']
    assert _run_until(pane.ollama_checked)
    assert 'could not start' in pane.status_label.text()
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
    for widget in (pane.provider_box, pane.model_box, pane.new_button):
        assert not widget.isEnabled()
    _run_until(pane.turn_finished)
    for widget in (pane.provider_box, pane.model_box, pane.new_button):
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


def test_save_errors_show_in_chat(qt_app, env):
    def broken(*_a, **_k):
        raise PermissionError('read-only config')
    s = AgentSettings()
    s.gemini.consent_given = True
    pane = AssistantPane(FakeSession([]), lambda: object(), s, save_settings=broken,
                         ask_key=lambda: None, confirm_consent=lambda: True,
                         ollama_server=FakeServer(ok=False), models_fn=lambda s: None)
    _PANES.append(pane)
    pane.provider_box.setCurrentIndex(pane.provider_box.findData('ollama'))
    assert 'read-only config' in pane.chat.toPlainText()
    _run_until(pane.ollama_checked)


def test_key_store_error_shows_in_chat(qt_app, env, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY')
    def broken(_key, _provider):
        raise OSError('disk full')
    session = FakeSession([TurnEvent('done')])
    s = AgentSettings()
    s.gemini.consent_given = True
    pane = AssistantPane(session, lambda: object(), s, save_settings=lambda *a, **k: None,
                         ask_key=lambda: 'k', confirm_consent=lambda: True, store_key=broken,
                         ollama_server=FakeServer())
    _PANES.append(pane)
    pane.send('x')
    QApplication.processEvents()
    assert session.texts == [] and 'disk full' in pane.chat.toPlainText()


def test_narrow_layout_stacks_chat_pending_input(qt_app, env):
    from PySide6.QtWidgets import QSplitter
    pane = _pane([])
    assert pane.findChildren(QSplitter) == []
    layout = pane.layout()
    order = [layout.indexOf(w) for w in (pane.chat, pane.pending_box, pane.input)]
    assert order == sorted(order) and -1 not in order


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


def test_ollama_not_running_is_one_line_with_collapsed_details(qt_app, env):
    from PySide6.QtCore import Qt
    pane = _pane([], settings=_settings(provider='ollama'),
                 start_command_fn=lambda _s: 'OLLAMA_IGPU_ENABLE=1 /x/ollama serve')
    status = pane.status_label.text()
    assert status.startswith('Ollama not running') and '\n' not in status
    assert 'serve' not in status
    assert pane.details_box.isHidden() and not pane.details_button.isHidden()
    pane.details_button.click()
    assert not pane.details_box.isHidden()
    details = pane.details_label.text()
    assert 'OLLAMA_IGPU_ENABLE=1 /x/ollama serve' in details
    assert '/fake/cache/ollama.log' in details
    assert pane.details_label.textInteractionFlags() & Qt.TextSelectableByMouse
    pane.copy_button.click()
    assert QApplication.clipboard().text() == 'OLLAMA_IGPU_ENABLE=1 /x/ollama serve'
    pane.details_button.click()
    assert pane.details_box.isHidden()


def test_ollama_ready_status_names_the_model(qt_app, env):
    pane = _pane([], settings=_settings(provider='ollama'),
                 models_fn=lambda s: ['qwen3:4b', 'qwen3:8b'])
    assert pane.status_label.text() == 'Ollama ready (qwen3:4b)'


def test_check_again_reruns_the_status_check(qt_app, env):
    state = {'models': None}
    server = FakeServer()
    pane = _pane([], settings=_settings(provider='ollama'),
                 models_fn=lambda s: state['models'], server=server)
    assert pane.status_label.text().startswith('Ollama not running')
    state['models'] = ['qwen3:4b']
    pane.check_button.click()
    assert _run_until(pane.ollama_checked) == [(True,)]
    assert pane.status_label.text() == 'Ollama ready (qwen3:4b)'
    assert server.ensured == 0  # checking never starts anything


def test_selecting_ollama_starts_it_off_the_gui_thread(qt_app, env):
    import threading
    threads = []
    server = FakeServer(on_ensure=lambda: threads.append(threading.current_thread()))
    state = {'models': None}
    server.on_ensure = lambda: (threads.append(threading.current_thread()),
                                state.update(models=['qwen3:4b']))
    pane = _pane([], server=server, models_fn=lambda s: state['models'])
    pane.provider_box.setCurrentIndex(pane.provider_box.findData('ollama'))
    assert pane.status_label.text() == 'Starting Ollama…'
    assert _run_until(pane.ollama_checked) == [(True,)]
    assert server.ensured == 1 and threads[0] is not threading.main_thread()
    assert pane.status_label.text() == 'Ollama ready (qwen3:4b)'


def test_send_while_not_ready_starts_then_sends(qt_app, env):
    state = {'models': None}
    server = FakeServer(on_ensure=lambda: state.update(models=['qwen3:4b']))
    session = FakeSession([TurnEvent('done', payload=1)])
    pane = _pane([], settings=_settings(provider='ollama'), session=session,
                 server=server, models_fn=lambda s: state['models'])
    pane.input.setPlainText('hello')
    pane.send()
    assert session.texts == [] and pane.status_label.text() == 'Starting Ollama…'
    assert _run_until(pane.turn_finished)
    assert server.ensured == 1
    assert session.texts == ['hello'] and 'You: hello' in pane.chat.toPlainText()
    assert server.touched >= 2  # the turn's start and end


def test_send_when_start_fails_says_why_and_keeps_the_text(qt_app, env):
    server = FakeServer(ok=False, message='Ollama did not answer within 20 s; see /l.log.')
    session = FakeSession([TurnEvent('done')])
    pane = _pane([], settings=_settings(provider='ollama'), session=session,
                 server=server)
    pane.send('hello')
    assert _run_until(pane.ollama_checked) == [(False,)]
    assert session.texts == []
    assert 'did not answer within 20 s' in pane.chat.toPlainText()
    assert pane.input.toPlainText() == 'hello'
    assert 'could not start' in pane.status_label.text()
    assert 'did not answer' in pane.details_label.text()


def test_new_chat_with_ollama_ensures_it(qt_app, env):
    server = FakeServer()
    pane = _pane([], settings=_settings(provider='ollama'), server=server,
                 models_fn=lambda s: ['qwen3:4b'])
    pane.new_chat()
    assert _run_until(pane.ollama_checked)
    assert server.ensured == 1


def test_idle_timer_stop_updates_the_status(qt_app, env):
    server = FakeServer()
    s = _settings(provider='ollama')
    s.ollama.idle_minutes = 7
    pane = _pane([], settings=s, server=server, models_fn=lambda _s: ['qwen3:4b'])
    assert pane.idle_timer.isActive() and pane.idle_timer.interval() == 60_000
    pane.idle_timer.timeout.emit()
    assert pane.status_label.text() == 'Ollama ready (qwen3:4b)'
    server.idle = True
    pane.idle_timer.timeout.emit()
    assert _run_until(pane.ollama_checked) == [(False,)]
    assert pane.status_label.text() == ('Ollama stopped after 7 min idle '
                                        '(starts again on your next message)')
    pane.send('x')  # not ready any more: starting again comes first
    assert pane.status_label.text() == 'Starting Ollama…'
    assert _run_until(pane.turn_finished)
    assert server.ensured == 1 and pane.session.texts == ['x']


def test_shutdown_stops_our_server(qt_app, env):
    server = FakeServer()
    pane = AssistantPane(FakeSession([]), lambda: object(), AgentSettings(),
                         save_settings=lambda *a, **k: None, ask_key=lambda: None,
                         confirm_consent=lambda: True, ollama_server=server)
    pane.shutdown()
    assert server.stopped == 1 and server.grace <= 1.0  # closing never waits 5 s


def test_ollama_server_is_a_required_keyword(qt_app, env):
    with pytest.raises(TypeError):
        AssistantPane(FakeSession([]), lambda: object(), AgentSettings(),
                      save_settings=lambda *a, **k: None)


class SlowServer(FakeServer):
    def ensure_running(self, settings):
        import time
        time.sleep(0.5)
        return super().ensure_running(settings)

    def stop_if_idle(self, now=None):
        import time
        time.sleep(0.5)
        return True


def _slow_models(_settings):
    import time
    time.sleep(0.5)
    return ['qwen3:4b']


def test_ollama_checks_never_block_the_gui(qt_app, env):
    import time
    server = SlowServer()
    t0 = time.monotonic()
    pane = AssistantPane(FakeSession([]), lambda: object(), _settings(provider='ollama'),
                         save_settings=lambda *a, **k: None, ollama_server=server,
                         models_fn=_slow_models)
    _PANES.append(pane)
    assert time.monotonic() - t0 < 0.2
    assert pane.status_label.text() == 'Checking Ollama…'
    assert _run_until(pane.ollama_checked) == [(True,)]
    for action in (pane.check_button.click, pane.idle_timer.timeout.emit, pane.new_chat):
        t0 = time.monotonic()
        action()
        assert time.monotonic() - t0 < 0.2, action
        assert _run_until(pane.ollama_checked)


def test_unreachable_turn_resets_ready_so_the_next_send_starts_it(qt_app, env):
    server = FakeServer()
    session = FakeSession([TurnEvent('error', 'Could not reach the ollama server.',
                                     payload='unreachable')])
    pane = _pane([], settings=_settings(provider='ollama'), session=session,
                 server=server, models_fn=lambda s: ['qwen3:4b'])
    pane.send('one')
    assert _run_until(pane.turn_finished) and session.texts == ['one']
    assert pane.status_label.text().startswith('Ollama not running')
    session.events = [TurnEvent('done', payload=1)]
    pane.send('two')
    assert _run_until(pane.turn_finished)
    assert server.ensured == 1 and session.texts == ['one', 'two']


def test_remote_server_status_says_unreachable(qt_app, env):
    s = _settings(provider='ollama')
    s.ollama.base_url = 'http://gpu-box.lan:11434'
    pane = _pane([], settings=s)
    assert pane.status_label.text() == 'Ollama at http://gpu-box.lan:11434 is unreachable'


def test_model_combo_lists_installed_models_and_fits(qt_app, env):
    from PySide6.QtWidgets import QComboBox
    saved = []
    s = _settings(provider='ollama')
    pane = _pane([], settings=s, models_fn=lambda _s: ['qwen3:4b', 'qwen3:8b', 'llama3.1:8b'],
                 save_settings=lambda settings: saved.append(settings.ollama.model))
    box = pane.model_box
    assert isinstance(box, QComboBox) and box.isEditable()
    assert [box.itemText(i) for i in range(box.count())] == ['qwen3:4b', 'qwen3:8b',
                                                              'llama3.1:8b']
    assert box.currentText() == 'qwen3:4b'
    assert box.minimumContentsLength() >= len('llama3.1:8b')
    pane.resize(pane.minimumSizeHint().width(), 600)
    pane.show()
    QApplication.processEvents()
    assert pane.minimumSizeHint().width() <= 300  # the dock's narrow width still fits
    assert box.lineEdit().width() > box.fontMetrics().horizontalAdvance('llama3.1:8b')
    pane.hide()
    box.setCurrentIndex(1)
    box.activated.emit(1)
    assert s.ollama.model == 'qwen3:8b' and saved[-1] == 'qwen3:8b'
    assert pane.session.switched[-1] == 'ollama'
    assert pane.status_label.text() == 'Ollama ready (qwen3:8b)'


def test_typed_model_name_is_kept(qt_app, env):
    s = AgentSettings()
    pane = _pane([], settings=s)
    pane.model_box.setEditText('gemini-9-pro')
    pane.model_box.lineEdit().editingFinished.emit()
    assert s.gemini.model == 'gemini-9-pro'


def test_missing_default_finetune_is_kept_and_offered_for_download(qt_app, env):
    saved = []
    s = AgentSettings(provider='ollama')  # the default: fs2d-assistant:qwen3.5-4b
    pane = _pane([], settings=s, models_fn=lambda _s: ['qwen3.5:4b'],
                 save_settings=lambda settings: saved.append(settings.ollama.model))
    assert s.ollama.model == 'fs2d-assistant:qwen3.5-4b' and saved == []  # no stock fallback
    assert not pane.install_button.isHidden() and pane.install_button.text() == 'Download'
    assert pane.chat.toPlainText() == ''


def test_old_experiment_names_are_left_alone(qt_app, env):
    s = AgentSettings(provider='ollama')
    s.ollama.model = 'fs2d-assistant:q35-4b'
    pane = _pane([], settings=s, models_fn=lambda _s: ['qwen3.5:4b'])
    assert s.ollama.model == 'fs2d-assistant:q35-4b'
    assert pane.install_button.isHidden() and 'pull fs2d-assistant:q35-4b' in pane.details_label.text()


def test_missing_model_is_named_and_details_say_how_to_pull(qt_app, env):
    s = _settings(provider='ollama')
    s.ollama.model = 'mistral:7b'
    pane = _pane([], settings=s, models_fn=lambda _s: ['qwen3:4b'])
    assert 'no model mistral:7b' in pane.status_label.text()
    assert 'pull mistral:7b' in pane.details_label.text()
    assert pane.model_box.currentText() == 'mistral:7b'


def test_gemini_is_unaffected_by_the_ollama_controls(qt_app, env):
    server = FakeServer()
    session = FakeSession([TurnEvent('done', payload=1)])
    pane = _pane([], session=session, server=server)
    assert pane.status_label.text() == 'Gemini: key set'
    assert pane.details_button.isHidden() and pane.check_button.isHidden()
    assert pane.details_box.isHidden()
    assert [pane.model_box.itemText(i) for i in range(pane.model_box.count())] == [
        'gemini-3.5-flash-lite']
    pane.send('x')
    _run_until(pane.turn_finished)
    pane.new_chat()
    pane.idle_timer.timeout.emit()
    assert session.texts == ['x'] and server.ensured == 0


def _anchor_of(chat, text):
    block = chat.document().begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            fragment = it.fragment()
            if text in fragment.text():
                return fragment.charFormat().isAnchor()
            it += 1
        block = block.next()
    raise AssertionError(f'{text!r} not in the chat')


def test_text_after_an_undo_line_is_not_part_of_the_link(qt_app, env):
    from PySide6.QtGui import QTextCursor
    pane = _pane([])
    pane.add_applied_line(1, 'Set 1 field: [general] force')
    pane.chat.moveCursor(QTextCursor.End)
    pane.chat.insertPlainText('tail')
    assert _anchor_of(pane.chat, 'Undo') and not _anchor_of(pane.chat, 'tail')
    pane._append_inline('streamed')  # pylint: disable=protected-access
    assert not _anchor_of(pane.chat, 'streamed')


def test_queue_card_keeps_a_change_as_a_card_with_a_note(qt_app, env, tmp_path, mock_settings):
    change = build_changeset(make_deps(tmp_path, mock_settings), 'loads',
                             [Op('set', 'general', 'force', [5])])
    pane = _pane([])
    pane.queue_card(change, 'kept as a card because you were editing')
    assert change.id in pane.cards and not pane.pending_box.isHidden()
    assert 'kept as a card because you were editing' in pane.chat.toPlainText()
    assert not hasattr(pane.cards[change.id], 'status')


def test_tool_result_is_indented(qt_app, env):
    pane = _pane([TurnEvent('tool_call', 'search(MoS2)'),
                  TurnEvent('tool_result', 'search: found'), TurnEvent('done', payload=1)])
    pane.send('hi')
    _run_until(pane.turn_finished)
    line = next(ln for ln in pane.chat.toPlainText().splitlines() if 'search: found' in ln)
    assert line.startswith('  →')  # nbsp survives HTML; toPlainText shows spaces


def test_shutdown_prunes_finished_orphan_threads(qt_app, env):
    from frictionsim2d.gui.assistant import pane as pane_module

    class Finished:
        def isRunning(self):
            return False

    stale = Finished()
    pane_module._ORPHAN_THREADS.append(stale)
    pane = _pane([])
    pane.shutdown()
    assert stale not in pane_module._ORPHAN_THREADS


def _plot_change(tmp_path, mock_settings):
    from frictionsim2d.agent.plot_changes import build_plot_change
    deps = make_deps(tmp_path, mock_settings)
    outputs = deps.roots.workspace / 'campaign' / 'outputs'
    outputs.mkdir(parents=True)
    (outputs / 'output_full_100x100y.json').write_text('{"runs": []}')
    return build_plot_change(deps, 'COF vs load', 'summary', {'y_axis': 'cof'}, ['campaign'])


def test_plot_proposal_is_handed_over_without_a_card(qt_app, env, tmp_path, mock_settings):
    change = _plot_change(tmp_path, mock_settings)
    pane = _pane([TurnEvent('proposal', payload=change), TurnEvent('done', payload=1)])
    seen = []
    pane.change_proposed.connect(seen.append)
    pane.send('plot it')
    _run_until(pane.turn_finished)
    assert seen == [change] and pane.cards == {}


def test_plot_card_in_ask_mode_requests_a_plot_apply(qt_app, env, tmp_path, mock_settings):
    change = _plot_change(tmp_path, mock_settings)
    pane = _pane([TurnEvent('proposal', payload=change), TurnEvent('done', payload=1)],
                 settings=AgentSettings(ask_before_apply=True))
    configs, plots = [], []
    pane.apply_requested.connect(lambda c, new: configs.append(c))
    pane.plot_apply_requested.connect(plots.append)
    pane.send('plot it')
    _run_until(pane.turn_finished)
    card = pane.cards[change.id]
    assert 'Plot: summary' in card.body.text() and 'workspace:/campaign' in card.body.text()
    card.apply_button.click()
    assert plots == [change] and configs == []


def test_applied_line_can_offer_a_show_link(qt_app, env):
    from PySide6.QtCore import QUrl
    pane = _pane([])
    shown, undone = [], []
    pane.show_tab_requested.connect(shown.append)
    pane.undo_requested.connect(undone.append)
    pane.add_applied_line(1, 'Added plot: summary on campaign', show_tab='Plots')
    assert 'Show' in pane.chat.toPlainText()
    pane.chat.anchorClicked.emit(QUrl('show:Plots'))
    assert shown == ['Plots']
    pane.set_undo_state(1, 'undone')  # the Show link is not the Undo control
    assert 'Show' in pane.chat.toPlainText() and undone == []


def test_claude_asks_for_and_stores_its_own_key(qt_app, env, monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    stored = []
    session = FakeSession([TurnEvent('done')])
    s = AgentSettings(provider='anthropic')
    s.anthropic.consent_given = True
    pane = AssistantPane(session, lambda: object(), s, save_settings=lambda *a, **k: None,
                         ask_key=lambda: 'sk-ant', confirm_consent=lambda: True,
                         store_key=lambda key, provider: stored.append((key, provider)),
                         ollama_server=FakeServer())
    _PANES.append(pane)
    assert pane.provider_box.currentData() == 'anthropic'
    assert pane.status_label.text() == 'Claude: no API key yet (you will be asked)'
    assert not pane.cloud_badge.isHidden() and pane.check_button.isHidden()
    pane.send('x')
    QApplication.processEvents()
    assert stored == [('sk-ant', 'anthropic')]


def test_claude_offers_opus_sonnet_and_haiku(qt_app, env):
    s = AgentSettings(provider='anthropic')
    pane = _pane([], settings=s)
    assert [pane.model_box.itemText(i) for i in range(pane.model_box.count())] == [
        'claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5']
    pane.model_box.setCurrentIndex(2)
    pane.model_box.activated.emit(2)
    assert s.anthropic.model == 'claude-haiku-5-5'


def test_openai_asks_for_and_stores_its_own_key(qt_app, env, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    stored, consent = [], []
    session = FakeSession([TurnEvent('done')])
    s = AgentSettings(provider='openai')
    pane = AssistantPane(session, lambda: object(), s, save_settings=lambda *a, **k: None,
                         ask_key=lambda: 'sk-o', confirm_consent=lambda: consent.append(1) or True,
                         store_key=lambda key, provider: stored.append((key, provider)),
                         ollama_server=FakeServer())
    _PANES.append(pane)
    assert pane.provider_box.currentData() == 'openai'
    assert pane.status_label.text() == 'OpenAI: no API key yet (you will be asked)'
    assert not pane.cloud_badge.isHidden()
    pane.send('x')
    QApplication.processEvents()
    assert consent == [1] and s.openai.consent_given
    assert stored == [('sk-o', 'openai')]


def test_openai_offers_its_model_list(qt_app, env):
    from frictionsim2d.agent.settings import OPENAI_MODELS
    s = AgentSettings(provider='openai')
    pane = _pane([], settings=s)
    assert tuple(pane.model_box.itemText(i)
                 for i in range(pane.model_box.count())) == OPENAI_MODELS
    pane.model_box.setCurrentIndex(1)
    pane.model_box.activated.emit(1)
    assert s.openai.model == OPENAI_MODELS[1]


# ---------------------------------------------------------------- fine-tunes

def _tune_pane(*, fetch=None, models=None, settings=None, **kwargs):
    """Ollama pane whose selected model is the 4B fine-tune, not yet installed."""
    from frictionsim2d.agent.fetch import FetchError
    state = {'models': ['qwen3.5:4b'] if models is None else models, 'tags': []}
    s = settings or AgentSettings(provider='ollama')

    def fake_fetch(tag, progress, settings=None):
        state['tags'].append(tag)
        progress('pulling manifest 50%')
        error = fetch(tag) if fetch else None
        if error:
            raise FetchError(error)
        state['models'] = state['models'] + [f'fs2d-assistant:{tag}']
        return f'fs2d-assistant:{tag}'
    pane = _pane([], settings=s, models_fn=lambda _s: state['models'],
                 fetch_fn=fake_fetch, **kwargs)
    return pane, state


def test_missing_finetune_offers_download(qt_app, env):
    pane, _ = _tune_pane()
    assert pane.settings.ollama.model == 'fs2d-assistant:qwen3.5-4b'
    assert pane.status_label.text() == ('fs2d-assistant:qwen3.5-4b is not installed — '
                                        'Download fetches it')
    assert not pane.install_button.isHidden() and pane.install_button.isEnabled()
    assert 'huggingface.co/mjvalder/fs2d-assistant-qwen3.5-4b' in pane.details_label.text()


def test_unknown_model_offers_no_download(qt_app, env):
    pane = _pane([], settings=_settings(provider='ollama'), models_fn=lambda s: ['qwen3:8b'])
    assert pane.install_button.isHidden()
    assert 'see Details' in pane.status_label.text() and 'pull qwen3:4b' in pane.details_label.text()


def test_download_success_is_ready(qt_app, env):
    saved = []
    pane, state = _tune_pane(save_settings=lambda s: saved.append(s.ollama.model_copy()))
    pane.install_button.click()
    assert pane.status_label.text().startswith('Downloading fs2d-assistant:qwen3.5-4b')
    assert _run_until(pane.ollama_checked) == [(True,)]
    assert state['tags'] == ['qwen3.5-4b']
    assert pane.status_label.text() == 'Ollama ready (fs2d-assistant:qwen3.5-4b)'
    assert pane.install_button.isHidden() and pane.model_box.isEnabled()


def test_download_failure_keeps_settings(qt_app, env):
    pane, _ = _tune_pane(fetch=lambda tag: 'Error: unknown renderer "qwen3.5"')
    before = pane.settings.model_dump()
    pane.install_button.click()
    assert _run_until(pane.ollama_checked) == [(False,)]
    assert pane.status_label.text() == 'Download failed — see Details'
    assert 'unknown renderer' in pane.details_label.text()
    assert pane.settings.model_dump() == before
    assert pane.install_button.isEnabled() and pane.model_box.isEnabled()


def test_download_button_disabled_while_running(qt_app, env):
    import threading
    gate, calls = threading.Event(), []

    def fetch(tag):
        calls.append(1)
        gate.wait(5)
    pane, _ = _tune_pane(fetch=fetch)
    pane.install_button.click()
    assert not pane.install_button.isEnabled() and not pane.model_box.isEnabled()
    pane._install()  # pylint: disable=protected-access  # a second click that got through
    gate.set()
    assert _run_until(pane.ollama_checked) == [(True,)]
    assert calls == [1]


def test_download_result_after_provider_switch_changes_nothing(qt_app, env):
    import threading
    gate = threading.Event()

    def fetch(tag):
        gate.wait(5)
        return 'Error: boom'
    pane, _ = _tune_pane(fetch=fetch)
    pane.install_button.click()
    pane.provider_box.setCurrentIndex(pane.provider_box.findData('gemini'))
    gate.set()
    assert _run_until(pane.ollama_checked) == [(False,)]
    assert pane.settings.provider == 'gemini' and pane.status_label.text().startswith('Gemini')
    assert pane.install_button.isHidden() and pane.model_box.isEnabled()


def test_remote_base_url_no_download(qt_app, env):
    s = AgentSettings(provider='ollama')
    s.ollama.base_url = 'http://gpu-box:11434'
    pane, _ = _tune_pane(settings=s)
    assert pane.install_button.isHidden()
    assert 'pull fs2d-assistant:qwen3.5-4b' in pane.details_label.text()


def test_older_schema_is_said_before_download_and_logged_once(qt_app, env, caplog):
    with caplog.at_level('WARNING', logger='frictionsim2d.gui.assistant.pane'):
        pane, _ = _tune_pane(schema_hash_fn=lambda: 'bbbb')
        assert 'older tool schema' in pane.details_label.text()
        assert 'older tool schema' in pane.install_button.toolTip()
        pane.model_box.setCurrentText('qwen3.5:4b')
        pane._on_model_chosen()  # pylint: disable=protected-access
        pane.model_box.setCurrentText('fs2d-assistant:qwen3.5-4b')
        pane._on_model_chosen()  # pylint: disable=protected-access
    assert sum('tool schemas' in r.message for r in caplog.records) == 1
    assert 'schema' not in pane.chat.toPlainText()


def test_schema_match_no_log(qt_app, env, caplog):
    with caplog.at_level('WARNING', logger='frictionsim2d.gui.assistant.pane'):
        pane, _ = _tune_pane(models=['fs2d-assistant:qwen3.5-4b'])
    assert pane.status_label.text() == 'Ollama ready (fs2d-assistant:qwen3.5-4b)'
    assert not any('tool schema' in r.message for r in caplog.records)
    assert 'older' not in pane.install_button.toolTip()


def test_idle_timer_never_stops_ollama_during_a_download(qt_app, env):
    import threading
    gate, checks = threading.Event(), []

    class CountingServer(FakeServer):
        def stop_if_idle(self, now=None):
            checks.append(1)
            return True

    pane, _ = _tune_pane(fetch=lambda tag: gate.wait(5) and None, server=CountingServer())
    pane.install_button.click()
    pane.idle_timer.timeout.emit()
    gate.set()
    assert _run_until(pane.ollama_checked) == [(True,)]
    assert checks == []
