"""OllamaServer: start, poll, idle stop -- every process and HTTP call is a fake."""
import signal
import sys
from pathlib import Path

import httpx
import pytest

from frictionsim2d.agent.ollama_server import OllamaServer
from frictionsim2d.agent.settings import AgentSettings


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeProcess:
    def __init__(self, exit_code=None, ignores_term=False, pid=4242):
        self.exit_code, self.ignores_term, self.pid = exit_code, ignores_term, pid
        self.terminated = self.killed = False
        self.waits = []

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
        self.waits.append(timeout)
        if self.exit_code is None:
            import subprocess
            raise subprocess.TimeoutExpired('ollama', timeout)
        return self.exit_code


class FakeHttp:
    """Answers /api/version once ``up`` is true (or after ``up_after`` calls)."""

    def __init__(self, up=False, up_after=None, version_body=None, ps_models=()):
        self.up, self.up_after, self.calls = up, up_after, []
        self.version_body = {'version': '0.12'} if version_body is None else version_body
        self.ps_models = list(ps_models)
        self.status = 200

    def __call__(self, url, timeout=None):
        self.calls.append(url)
        if url.endswith('/api/version') and self.up_after is not None \
                and len([c for c in self.calls if c.endswith('/api/version')]) > self.up_after:
            self.up = True
        if not self.up:
            raise httpx.ConnectError('refused')
        request = httpx.Request('GET', url)
        if url.endswith('/api/ps'):
            return httpx.Response(200, json={'models': [{'name': n} for n in self.ps_models]},
                                  request=request)
        if isinstance(self.version_body, str):
            return httpx.Response(self.status, text=self.version_body, request=request)
        return httpx.Response(self.status, json=self.version_body, request=request)


class FakePopen:
    def __init__(self, process=None):
        self.process = process or FakeProcess()
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return self.process


class FakeProcTable:
    """pid -> cmdline of 'live' processes, plus the signals sent to them."""

    def __init__(self, table=None, identity=None, deny=False):
        self.table = dict(table or {})
        self.identity = dict(identity or {})  # pid -> (uid, start ticks)
        self.deny = deny  # kill raises PermissionError (someone else's pid)
        self.signals = []

    def read_identity(self, pid):
        return self.identity.get(pid) if pid in self.table else None

    def read_cmdline(self, pid):
        return self.table.get(pid)

    def kill(self, pid, sig):
        import signal
        if self.deny:
            raise PermissionError(1, 'Operation not permitted')
        self.signals.append((pid, sig))
        if sig in (signal.SIGTERM, signal.SIGKILL):
            self.table.pop(pid, None)


def _server(tmp_path, http, popen=None, clock=None, which=None, procs=None, exits=None,
            **kwargs):
    clock = clock or FakeClock()
    procs = procs or FakeProcTable()
    exits = exits if exits is not None else []
    return OllamaServer(which=which or (lambda name: f'/usr/bin/{name}'), home=tmp_path,
                        popen=popen or FakePopen(), http_get=http, clock=clock,
                        sleep=clock.sleep, log_path=tmp_path / 'cache' / 'ollama.log',
                        read_cmdline=procs.read_cmdline, kill=procs.kill,
                        register_exit=exits.append, read_identity=procs.read_identity,
                        getuid=lambda: 1000, **kwargs)


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
    if sys.platform.startswith('linux'):
        assert callable(kwargs['preexec_fn'])  # PR_SET_PDEATHSIG: dies with the GUI


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
    assert 'port in use?' in message


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
    # an idle stop is not a close: the next message starts it again
    assert server.ensure_running(settings)[0]


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


# ---------------------------------------------------------------- review fixes

def test_stop_before_the_start_refuses_to_start(tmp_path):
    popen = FakePopen()
    server = _server(tmp_path, FakeHttp(), popen)
    server.stop()  # the window closed while the start task was still queued
    ok, message = server.ensure_running(AgentSettings())
    assert not ok and popen.calls == [] and 'closing' in message


def test_stop_during_the_poll_ends_it_and_terminates_our_process(tmp_path):
    clock = FakeClock()
    process = FakeProcess()
    server = None

    def sleep(seconds):
        clock.sleep(seconds)
        server.stop()  # the window closes mid-poll
    server = OllamaServer(which=lambda n: f'/usr/bin/{n}', home=tmp_path,
                          popen=FakePopen(process), http_get=FakeHttp(), clock=clock,
                          sleep=sleep, log_path=tmp_path / 'ollama.log',
                          read_cmdline=lambda pid: None, kill=lambda pid, sig: None,
                          register_exit=lambda f: None)
    ok, _message = server.ensure_running(AgentSettings())
    assert not ok and process.terminated
    assert clock.now - 1000.0 < 1  # returned at once, not after the 20 s timeout


def test_first_start_registers_stop_at_exit_and_writes_a_pidfile(tmp_path):
    exits = []
    process = FakeProcess(pid=777)
    procs = FakeProcTable({777: ['ollama', 'serve']}, {777: (1000, 98765)})
    server = _server(tmp_path, FakeHttp(up_after=1), FakePopen(process), exits=exits,
                     procs=procs)
    assert server.ensure_running(AgentSettings())[0]
    assert exits == [server.stop]
    pidfile = tmp_path / 'cache' / 'ollama.pid'
    assert pidfile.read_text().split() == ['777', '98765']  # pid and its start time
    server.stop()
    assert not pidfile.exists()


def test_a_later_session_adopts_our_server_from_the_pidfile(tmp_path):
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache' / 'ollama.pid').write_text('555 98765\n')
    procs = FakeProcTable({555: ['/home/u/.local/ollama/bin/ollama', 'serve']},
                          {555: (1000, 98765)})
    clock = FakeClock()
    popen = FakePopen()
    exits = []
    server = _server(tmp_path, FakeHttp(up=True), popen, clock=clock, procs=procs,
                     exits=exits)
    assert server.ensure_running(AgentSettings())[0]
    assert popen.calls == [] and server.started_by_us and exits == [server.stop]
    clock.now += 11 * 60
    assert server.stop_if_idle(clock())
    assert procs.signals[0] == (555, signal.SIGTERM)
    assert not (tmp_path / 'cache' / 'ollama.pid').exists()


def test_a_pidfile_naming_another_program_is_not_adopted(tmp_path):
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache' / 'ollama.pid').write_text('555 98765\n')
    procs = FakeProcTable({555: ['/usr/bin/python', 'worker.py']},  # pid reused
                          {555: (1000, 98765)})
    server = _server(tmp_path, FakeHttp(up=True), procs=procs)
    assert server.ensure_running(AgentSettings())[0]
    assert not server.started_by_us
    server.stop()
    assert procs.signals == []


def test_idle_stop_skips_while_a_model_is_loaded(tmp_path):
    clock = FakeClock()
    process = FakeProcess()
    http = FakeHttp(up_after=1, ps_models=['qwen3:4b'])  # another GUI used it lately
    server = _server(tmp_path, http, FakePopen(process), clock=clock)
    assert server.ensure_running(AgentSettings())[0]
    clock.now += 60 * 60
    assert not server.stop_if_idle(clock()) and not process.terminated
    http.ps_models = []
    assert server.stop_if_idle(clock()) and process.terminated


def test_idle_stop_never_runs_during_a_start(tmp_path):
    clock = FakeClock()
    process = FakeProcess()
    server = None
    stopped = []

    def sleep(seconds):
        clock.sleep(seconds)
        stopped.append(server.stop_if_idle(clock() + 60 * 60))  # "an hour later", mid-poll
    server = OllamaServer(which=lambda n: f'/usr/bin/{n}', home=tmp_path,
                          popen=FakePopen(process), http_get=FakeHttp(up_after=2),
                          clock=clock, sleep=sleep, log_path=tmp_path / 'ollama.log',
                          read_cmdline=lambda pid: None, kill=lambda pid, sig: None,
                          register_exit=lambda f: None)
    assert server.ensure_running(AgentSettings())[0]
    assert stopped and not any(stopped) and not process.terminated


def test_stop_grace_is_configurable(tmp_path):
    process = FakeProcess(ignores_term=True)
    server = _server(tmp_path, FakeHttp(up_after=1), FakePopen(process))
    assert server.ensure_running(AgentSettings())[0]
    server.stop(grace=1.0)
    assert process.waits[0] == 1.0 and process.killed


@pytest.mark.parametrize('body', ['<html>not ollama</html>', {'hello': 'world'}])
def test_port_held_by_something_else_is_reported_not_started(tmp_path, body):
    popen = FakePopen()
    server = _server(tmp_path, FakeHttp(up=True, version_body=body), popen)
    ok, message = server.ensure_running(AgentSettings())
    assert not ok and popen.calls == []
    assert 'port 11434 is in use by something that is not Ollama' in message


def test_port_held_answering_404_is_reported_not_started(tmp_path):
    http = FakeHttp(up=True, version_body='nope')
    http.status = 404
    popen = FakePopen()
    ok, message = _server(tmp_path, http, popen).ensure_running(AgentSettings())
    assert not ok and popen.calls == [] and 'not Ollama' in message


def test_remote_server_is_never_started(tmp_path):
    popen = FakePopen()
    settings = AgentSettings()
    settings.ollama.base_url = 'http://gpu-box.lan:11434'
    ok, message = _server(tmp_path, FakeHttp(), popen).ensure_running(settings)
    assert not ok and popen.calls == [] and 'unreachable' in message


@pytest.mark.parametrize('url', ['http://localhost:11434', 'http://127.0.0.1:9000',
                                 'http://[::1]:11434'])
def test_loopback_hosts_count_as_local(url):
    from frictionsim2d.agent.ollama_server import is_local
    settings = AgentSettings()
    settings.ollama.base_url = url
    assert is_local(settings)


def test_big_log_is_rotated_at_start(tmp_path):
    log = tmp_path / 'cache' / 'ollama.log'
    log.parent.mkdir()
    log.write_bytes(b'x' * (5 * 1024 * 1024 + 1))
    server = _server(tmp_path, FakeHttp(up_after=1))
    assert server.ensure_running(AgentSettings())[0]
    assert (log.parent / 'ollama.log.1').stat().st_size > 5 * 1024 * 1024
    assert log.stat().st_size == 0


def _adoption(tmp_path, identity):
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache' / 'ollama.pid').write_text('555 98765\n')
    procs = FakeProcTable({555: ['/usr/bin/ollama', 'serve']}, {555: identity})
    server = _server(tmp_path, FakeHttp(up=True), procs=procs)
    assert server.ensure_running(AgentSettings())[0]
    server.stop()
    return server, procs


def test_recycled_pid_with_another_start_time_is_not_adopted(tmp_path):
    server, procs = _adoption(tmp_path, (1000, 11111))  # a later ollama serve
    assert procs.signals == [] and not server.started_by_us


def test_another_users_server_is_not_adopted(tmp_path):
    server, procs = _adoption(tmp_path, (1001, 98765))
    assert procs.signals == [] and not server.started_by_us


def test_permission_error_at_stop_is_swallowed(tmp_path):
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache' / 'ollama.pid').write_text('555 98765\n')
    procs = FakeProcTable({555: ['/usr/bin/ollama', 'serve']}, {555: (1000, 98765)})
    clock = FakeClock()
    server = _server(tmp_path, FakeHttp(up=True), procs=procs, clock=clock)
    assert server.ensure_running(AgentSettings())[0] and server.started_by_us
    procs.deny = True
    server.stop(grace=0.1)  # SIGTERM and SIGKILL both refused: no exception
    server.send_signal(999, signal.SIGTERM)


def test_overlapping_starts_keep_starting_until_both_finish(tmp_path):
    import threading
    import time
    clock = FakeClock()
    process = FakeProcess()
    http = FakeHttp()
    server = None
    observed = []
    gate = threading.Event()
    second_done = threading.Event()

    def second():
        server.ensure_running(AgentSettings())
        second_done.set()

    calls = {'n': 0}

    def sleep(seconds):
        if threading.current_thread() is not threading.main_thread():
            gate.set()  # the second call is polling; its clock stays put
            time.sleep(0.001)
            return
        clock.sleep(seconds)
        calls['n'] += 1
        if calls['n'] == 1:
            worker = threading.Thread(target=second)
            worker.start()
            gate.wait(0.5)  # let the second call enter its poll
            http.up = True
            second_done.wait(2)
            # the second ensure has finished; the first is still polling
            observed.append(server.stop_if_idle(clock() + 3600))
    server = OllamaServer(which=lambda n: f'/usr/bin/{n}', home=tmp_path,
                          popen=FakePopen(process), http_get=http, clock=clock,
                          sleep=sleep, log_path=tmp_path / 'ollama.log',
                          read_cmdline=lambda pid: None, kill=lambda pid, sig: None,
                          register_exit=lambda f: None, read_identity=lambda pid: None,
                          getuid=lambda: 1000)
    assert server.ensure_running(AgentSettings())[0]
    assert observed == [False] and not process.terminated


def test_server_env_pins_the_throttle_and_the_host(monkeypatch):
    from frictionsim2d.agent.ollama_server import server_env, throttle_env
    monkeypatch.setenv('OLLAMA_HOST', 'elsewhere:1')
    settings = AgentSettings(ollama={'base_url': 'http://127.0.0.1:11500'})
    env = server_env(settings)
    assert set(throttle_env(settings)) == {
        'OLLAMA_IGPU_ENABLE', 'OLLAMA_VULKAN', 'OLLAMA_MAX_LOADED_MODELS',
        'OLLAMA_NUM_PARALLEL', 'OLLAMA_KEEP_ALIVE', 'OLLAMA_CONTEXT_LENGTH'}
    assert env['OLLAMA_HOST'] == '127.0.0.1:11500'  # the settings win over the shell
    assert env.items() >= throttle_env(settings).items()


def test_cache_dir_honours_xdg_cache_home(monkeypatch, tmp_path):
    from frictionsim2d.agent.ollama_server import cache_dir
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    assert cache_dir() == tmp_path / 'FrictionSim2D'
    monkeypatch.delenv('XDG_CACHE_HOME')
    assert cache_dir() == Path.home() / '.cache' / 'FrictionSim2D'
