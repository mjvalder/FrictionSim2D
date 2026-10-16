"""OllamaServer: start, poll, idle stop -- every process and HTTP call is a fake."""
from pathlib import Path

import httpx
import pytest

from src.agent.ollama_server import OllamaServer
from src.agent.settings import AgentSettings


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeProcess:
    def __init__(self, exit_code=None, ignores_term=False):
        self.exit_code, self.ignores_term = exit_code, ignores_term
        self.terminated = self.killed = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True
        if not self.ignores_term:
            self.exit_code = 0

    def kill(self):
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout=None):
        if self.exit_code is None:
            import subprocess
            raise subprocess.TimeoutExpired('ollama', timeout)
        return self.exit_code


class FakeHttp:
    """Answers /api/version once ``up`` is true (or after ``up_after`` calls)."""

    def __init__(self, up=False, up_after=None):
        self.up, self.up_after, self.calls = up, up_after, []

    def __call__(self, url, timeout=None):
        self.calls.append(url)
        if self.up_after is not None and len(self.calls) > self.up_after:
            self.up = True
        if not self.up:
            raise httpx.ConnectError('refused')
        return httpx.Response(200, json={'version': '0.12'},
                              request=httpx.Request('GET', url))


class FakePopen:
    def __init__(self, process=None):
        self.process = process or FakeProcess()
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return self.process


def _server(tmp_path, http, popen=None, clock=None, which=None):
    clock = clock or FakeClock()
    return OllamaServer(which=which or (lambda name: f'/usr/bin/{name}'), home=tmp_path,
                        popen=popen or FakePopen(), http_get=http, clock=clock,
                        sleep=clock.sleep, log_path=tmp_path / 'cache' / 'ollama.log')


def test_already_up_starts_nothing(tmp_path):
    popen = FakePopen()
    server = _server(tmp_path, FakeHttp(up=True), popen)
    ok, message = server.ensure_running(AgentSettings())
    assert ok and popen.calls == [] and not server.started_by_us
    assert 'running' in message


def test_starts_with_throttled_env_and_polls_until_up(tmp_path):
    popen = FakePopen()
    settings = AgentSettings()
    settings.ollama.context_tokens = 4096
    server = _server(tmp_path, FakeHttp(up_after=3), popen)
    ok, message = server.ensure_running(settings)
    assert ok and server.started_by_us, message
    (cmd, kwargs), = popen.calls
    assert cmd == ['/usr/bin/nice', '-n', '10', '/usr/bin/ollama', 'serve']
    env = kwargs['env']
    for key, value in {'OLLAMA_IGPU_ENABLE': '1', 'OLLAMA_VULKAN': '1',
                       'OLLAMA_MAX_LOADED_MODELS': '1', 'OLLAMA_NUM_PARALLEL': '1',
                       'OLLAMA_KEEP_ALIVE': '2m', 'OLLAMA_CONTEXT_LENGTH': '4096'}.items():
        assert env[key] == value
    assert 'PATH' in env  # os.environ is kept
    assert kwargs['start_new_session'] is True
    assert (tmp_path / 'cache' / 'ollama.log').is_file()


def test_without_nice_runs_the_binary_directly(tmp_path):
    popen = FakePopen()
    which = lambda name: '/opt/ollama' if name == 'ollama' else None  # noqa: E731
    server = _server(tmp_path, FakeHttp(up_after=1), popen, which=which)
    assert server.ensure_running(AgentSettings())[0]
    assert popen.calls[0][0] == ['/opt/ollama', 'serve']


def test_user_local_binary_is_used_when_not_on_path(tmp_path):
    binary = tmp_path / '.local' / 'ollama' / 'bin' / 'ollama'
    binary.parent.mkdir(parents=True)
    binary.write_text('')
    popen = FakePopen()
    which = lambda name: '/usr/bin/nice' if name == 'nice' else None  # noqa: E731
    server = _server(tmp_path, FakeHttp(up_after=1), popen, which=which)
    assert server.ensure_running(AgentSettings())[0]
    assert popen.calls[0][0][-2:] == [str(binary), 'serve']


def test_no_binary_is_a_clear_failure(tmp_path):
    popen = FakePopen()
    server = _server(tmp_path, FakeHttp(), popen, which=lambda name: None)
    ok, message = server.ensure_running(AgentSettings())
    assert not ok and popen.calls == [] and 'not installed' in message


def test_poll_timeout_names_the_log(tmp_path):
    clock = FakeClock()
    server = _server(tmp_path, FakeHttp(), clock=clock)
    ok, message = server.ensure_running(AgentSettings())
    assert not ok and str(tmp_path / 'cache' / 'ollama.log') in message
    assert 20 <= clock.now - 1000.0 <= 25


def test_process_exiting_at_start_is_a_failure(tmp_path):
    server = _server(tmp_path, FakeHttp(), FakePopen(FakeProcess(exit_code=1)))
    ok, message = server.ensure_running(AgentSettings())
    assert not ok and 'exited' in message and 'ollama.log' in message


def test_stop_if_idle_only_after_the_idle_time(tmp_path):
    clock = FakeClock()
    process = FakeProcess()
    settings = AgentSettings()
    settings.ollama.idle_minutes = 10
    server = _server(tmp_path, FakeHttp(up_after=1), FakePopen(process), clock=clock)
    assert server.ensure_running(settings)[0]
    clock.now += 9 * 60
    assert not server.stop_if_idle(clock()) and not process.terminated
    server.touch()
    clock.now += 9 * 60
    assert not server.stop_if_idle(clock()) and not process.terminated
    clock.now += 2 * 60
    assert server.stop_if_idle(clock()) and process.terminated
    assert not server.started_by_us


def test_stop_if_idle_never_touches_a_foreign_server(tmp_path):
    clock = FakeClock()
    popen = FakePopen()
    server = _server(tmp_path, FakeHttp(up=True), popen, clock=clock)
    server.ensure_running(AgentSettings())
    clock.now += 10 ** 6
    assert not server.stop_if_idle(clock())
    server.stop()
    assert not popen.process.terminated and not popen.process.killed


def test_stop_kills_after_five_seconds_of_ignoring_sigterm(tmp_path):
    process = FakeProcess(ignores_term=True)
    server = _server(tmp_path, FakeHttp(up_after=1), FakePopen(process))
    assert server.ensure_running(AgentSettings())[0]
    server.stop()
    assert process.terminated and process.killed


def test_stop_without_a_start_is_a_no_op(tmp_path):
    _server(tmp_path, FakeHttp()).stop()


def test_log_path_honours_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xc'))
    assert OllamaServer().log_path == tmp_path / 'xc' / 'FrictionSim2D' / 'ollama.log'
    monkeypatch.delenv('XDG_CACHE_HOME')
    assert OllamaServer().log_path == Path.home() / '.cache' / 'FrictionSim2D' / 'ollama.log'
