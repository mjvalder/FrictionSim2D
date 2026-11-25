"""Several turns of a real AgentSession over HTTP, run the way the worker runs them.

The model's async HTTP client is cached on the agent, so its pooled
connections belong to the loop of the first turn. A fresh loop per turn
(``asyncio.run``) made every other turn fail with 'Event loop is closed'.
The stub below is an OpenAI-compatible SSE server on 127.0.0.1 only.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip('pydantic_ai')

from frictionsim2d.agent.loop import TurnRunner  # noqa: E402
from frictionsim2d.agent.runtime import AgentSession  # noqa: E402
from frictionsim2d.agent.settings import AgentSettings  # noqa: E402
from tests.helpers.agent import make_deps  # noqa: E402


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'  # keep-alive: connections are pooled

    def log_message(self, *_args):  # pylint: disable=arguments-differ
        pass

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers['Content-Length']))
        chunks = [
            {'id': 'x', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'm',
             'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'hello'},
                          'finish_reason': None}]},
            {'id': 'x', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'm',
             'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]},
        ]
        body = (''.join(f'data: {json.dumps(c)}\n\n' for c in chunks)
                + 'data: [DONE]\n\n').encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(name='stub_url')
def _stub_url():
    server = ThreadingHTTPServer(('127.0.0.1', 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_address[1]}'
    server.shutdown()
    server.server_close()


def test_three_turns_on_one_runner_all_finish(stub_url, tmp_path, mock_settings):
    settings = AgentSettings(provider='ollama')
    settings.ollama.base_url = stub_url
    session = AgentSession(settings)
    runner = TurnRunner()
    outcomes = []
    try:
        for turn in range(3):
            events = []
            deps = make_deps(tmp_path / str(turn), mock_settings)
            runner.run(session.run_turn('hi', deps, events.append, session.new_token()))
            outcomes.append([(e.kind, e.text) for e in events if e.kind in ('done', 'error')])
    finally:
        runner.close()
    assert outcomes == [[('done', '')]] * 3


def test_runner_reuses_one_loop_and_closes_it():
    import asyncio
    runner = TurnRunner()

    async def current():
        return asyncio.get_running_loop()

    first, second = runner.run(current()), runner.run(current())
    assert first is second and not first.is_closed()
    runner.close()
    assert first.is_closed()
    runner.close()  # idempotent


def test_close_cancels_a_task_left_pending():
    import asyncio
    runner = TurnRunner()

    async def leave_a_task():
        return asyncio.ensure_future(asyncio.sleep(60))

    task = runner.run(leave_a_task())
    assert not task.done()
    runner.close()
    assert task.cancelled()
