"""Starting a local Ollama server on demand, and stopping it when idle.

Qt-free so it is tested with fakes: every process, HTTP call, clock read and
signal goes through an injected callable. A server this object did not start
(or adopt from its own pidfile) is never stopped -- the user may be using it
for something else.

A started server must not outlive the GUI: `stop` runs at window close and at
interpreter exit (atexit), and on Linux the child also gets
PR_SET_PDEATHSIG=SIGTERM. That signal fires when the *thread* that forked the
child exits, so every start forks from one long-lived spawner thread, never
from a short-lived pool thread. A hard crash that defeats all three leaves a
pidfile; the next session adopts the server from it.
"""
from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .providers import ollama_binary
from .settings import AgentSettings

#: How long a fresh ``ollama serve`` gets to answer /api/version.
START_TIMEOUT = 20.0
#: How long a stopped server gets between SIGTERM and SIGKILL.
STOP_GRACE = 5.0
#: A log bigger than this is moved to ``ollama.log.1`` at the next start.
LOG_ROTATE_BYTES = 5 * 1024 * 1024
LOOPBACK_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})
_POLL_EVERY = 0.25
_PR_SET_PDEATHSIG = 1


def cache_dir() -> Path:
    """FrictionSim2D's per-user cache directory, honouring XDG_CACHE_HOME."""
    base = os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
    return Path(base) / 'FrictionSim2D'


def is_local(settings: AgentSettings) -> bool:
    """Whether ``base_url`` names this machine (only then is a server started)."""
    return (urlparse(settings.ollama.base_url).hostname or '') in LOOPBACK_HOSTS


def server_env(settings: AgentSettings) -> dict:
    """The environment ``ollama serve`` runs with: throttled for a laptop iGPU.

    OLLAMA_IGPU_ENABLE=1 lets Ollama use an integrated GPU such as the
    Radeon 780M; OLLAMA_VULKAN is honoured though not listed by
    ``ollama serve --help``.
    """
    env = dict(os.environ)
    env.update({'OLLAMA_IGPU_ENABLE': '1', 'OLLAMA_VULKAN': '1',
                'OLLAMA_MAX_LOADED_MODELS': '1', 'OLLAMA_NUM_PARALLEL': '1',
                'OLLAMA_KEEP_ALIVE': settings.ollama.keep_alive,
                'OLLAMA_CONTEXT_LENGTH': str(settings.ollama.context_tokens)})
    host = urlparse(settings.ollama.base_url).netloc
    if host:
        env['OLLAMA_HOST'] = host
    return env


def _die_with_parent() -> None:  # pragma: no cover  (runs in the forked child)
    """preexec_fn: SIGTERM this child when the thread that forked it exits."""
    try:
        import ctypes  # pylint: disable=import-outside-toplevel
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
    except Exception:  # pylint: disable=broad-except
        pass


def read_cmdline(pid: int) -> Optional[List[str]]:
    """A live process's command line from /proc; None when there is none."""
    try:
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return None
    return [part.decode(errors='replace') for part in raw.split(b'\0') if part] or None


def read_identity(pid: int) -> Optional[Tuple[int, int]]:
    """(owner uid, start time in clock ticks) of a live process; None if unknown.

    The start time is field 22 of /proc/<pid>/stat, counted after the
    parenthesised command name (which may itself contain spaces).
    """
    try:
        uid = os.stat(f'/proc/{pid}').st_uid
        stat = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8', errors='replace')
        return uid, int(stat.rsplit(')', 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def _is_ollama_serve(cmdline: Optional[List[str]]) -> bool:
    return bool(cmdline) and 'serve' in cmdline and any('ollama' in p for p in cmdline)


class _PidProcess:
    """A Popen-like handle on a server adopted from the pidfile."""

    def __init__(self, pid: int, server: 'OllamaServer') -> None:
        self.pid = pid
        self._server = server

    def poll(self) -> Optional[int]:
        """None while the pid is still an ``ollama serve``."""
        return None if _is_ollama_serve(self._server.read_cmdline(self.pid)) else 0

    def terminate(self) -> None:
        """SIGTERM."""
        self._server.send_signal(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        """SIGKILL."""
        self._server.send_signal(self.pid, signal.SIGKILL)

    def wait(self, timeout: Optional[float] = None) -> int:
        """Until the process is gone; TimeoutExpired after ``timeout``."""
        deadline = None if timeout is None else self._server.clock() + timeout
        while self.poll() is None:
            if deadline is not None and self._server.clock() >= deadline:
                raise subprocess.TimeoutExpired('ollama serve', timeout)
            self._server.sleep(0.05)
        return 0


class OllamaServer:  # pylint: disable=too-many-instance-attributes
    """One ``ollama serve`` process this app may own.

    `ensure_running` may block for up to `START_TIMEOUT` and `stop_if_idle`
    for an HTTP call plus the SIGTERM grace: call both off the GUI thread.
    `stop` is final (window close): later starts are refused.
    """

    def __init__(self, *, which: Callable[[str], Optional[str]] = shutil.which,  # pylint: disable=too-many-arguments
                 home: Optional[Path] = None, popen: Callable[..., Any] = subprocess.Popen,
                 http_get: Callable[..., Any] = httpx.get,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 log_path: Optional[Path] = None,
                 start_timeout: float = START_TIMEOUT,
                 read_cmdline: Callable[[int], Optional[List[str]]] = read_cmdline,  # pylint: disable=redefined-outer-name
                 kill: Callable[[int, int], None] = os.kill,
                 register_exit: Callable[[Callable], Any] = atexit.register,
                 read_identity: Callable[[int], Optional[Tuple[int, int]]] = read_identity,  # pylint: disable=redefined-outer-name
                 getuid: Callable[[], int] = os.getuid) -> None:
        self._which = which
        self._home = home
        self._popen = popen
        self._http_get = http_get
        self.clock = clock
        self.sleep = sleep
        self.read_cmdline = read_cmdline
        self._kill = kill
        self._register_exit = register_exit
        self._read_identity = read_identity
        self._getuid = getuid
        self._log_path = log_path
        self._start_timeout = start_timeout
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._starting = 0  # ensure_running calls polling right now
        self._not_stopping = threading.Event()  # cleared while an idle stop terminates
        self._not_stopping.set()
        self._exit_registered = False
        self._spawner: Optional[ThreadPoolExecutor] = None
        self._process = None
        self._settings: Optional[AgentSettings] = None
        self._last_activity = clock()

    # ------------------------------------------------------------ helpers

    @property
    def log_path(self) -> Path:
        """Where the server's stdout and stderr go."""
        return self._log_path or cache_dir() / 'ollama.log'

    @property
    def pid_path(self) -> Path:
        """Pid and start time of the server this app started, next to the log."""
        return self.log_path.with_name('ollama.pid')

    @property
    def started_by_us(self) -> bool:
        """True while a server this object started (or adopted) is running."""
        process = self._process
        return process is not None and process.poll() is None

    def send_signal(self, pid: int, sig: int) -> None:
        """Signal ``pid``; one that is gone or not ours never raises (stop
        runs at exit, where an exception helps no one)."""
        try:
            self._kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _url(self, settings: AgentSettings, path: str) -> str:
        return settings.ollama.base_url.rstrip('/') + path

    def probe(self, settings: AgentSettings) -> str:
        """'up', 'down', or 'foreign' (something that is not Ollama answers)."""
        try:
            response = self._http_get(self._url(settings, '/api/version'), timeout=1.0)
        except httpx.HTTPError:
            return 'down'
        try:
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return 'foreign'
        return 'up' if isinstance(body, dict) and 'version' in body else 'foreign'

    def is_up(self, settings: AgentSettings) -> bool:
        """Whether Ollama answers /api/version at the configured URL."""
        return self.probe(settings) == 'up'

    def _models_loaded(self) -> bool:
        """Whether /api/ps lists a loaded model (someone used it lately)."""
        if self._settings is None:
            return False
        try:
            response = self._http_get(self._url(self._settings, '/api/ps'), timeout=1.0)
            response.raise_for_status()
            return bool(response.json().get('models'))
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    def touch(self) -> None:
        """Record activity: a turn started or ended."""
        self._last_activity = self.clock()

    def _own(self, process) -> None:
        """Take ``process`` as ours (lock held)."""
        self._process = process
        if not self._exit_registered:
            self._register_exit(self.stop)
            self._exit_registered = True

    # -------------------------------------------------------------- start

    def ensure_running(self, settings: AgentSettings) -> Tuple[bool, str]:
        """Make sure a server answers, starting one when none does."""
        self._settings = settings
        self.touch()
        self._not_stopping.wait(STOP_GRACE + 1.0)  # an idle stop finishes first
        if self._closed.is_set():
            return False, 'The Assistant is closing; Ollama was not started.'
        state = self.probe(settings)
        if state == 'foreign':
            port = urlparse(settings.ollama.base_url).port or 11434
            return False, (f'port {port} is in use by something that is not Ollama; '
                           'stop it or change ollama.base_url in agent.yaml.')
        if state == 'up':
            with self._lock:
                if not self.started_by_us:
                    self._adopt()
            return True, f'Ollama is running at {settings.ollama.base_url}.'
        if not is_local(settings):
            return False, (f'Ollama at {settings.ollama.base_url} is unreachable; a '
                           'remote server is never started from here.')
        with self._lock:
            if self._closed.is_set():
                return False, 'The Assistant is closing; Ollama was not started.'
            if not self.started_by_us:
                failure = self._start(settings)
                if failure:
                    return False, failure
            process = self._process
            self._starting += 1
        try:
            return self._poll(settings, process)
        finally:
            with self._lock:
                self._starting -= 1

    def _poll(self, settings: AgentSettings, process) -> Tuple[bool, str]:
        deadline = self.clock() + self._start_timeout
        while self.clock() < deadline:
            if self._closed.is_set():
                self._terminate(process, 1.0)
                return False, 'The Assistant is closing; Ollama was stopped.'
            code = process.poll()
            if code is not None:
                return False, (f'Ollama exited while starting (code {code}; port in use?); '
                               f'see {self.log_path}.')
            if self.is_up(settings):
                self.touch()
                return True, 'Ollama started.'
            self.sleep(_POLL_EVERY)
        return False, (f'Ollama did not answer within {self._start_timeout:.0f} s; '
                       f'see {self.log_path}.')

    def _adopt(self) -> None:
        """Own the running server if it is the very process our pidfile records.

        Same pid, same start time (so not a recycled pid), owned by this user,
        and still an ``ollama serve`` (lock held).
        """
        try:
            pid_text, start_text = self.pid_path.read_text(encoding='utf-8').split()[:2]
            pid, start = int(pid_text), int(start_text)
        except (OSError, ValueError):
            return
        if (self._read_identity(pid) == (self._getuid(), start)
                and _is_ollama_serve(self.read_cmdline(pid))):
            self._own(_PidProcess(pid, self))
        else:
            self.pid_path.unlink(missing_ok=True)  # stale: not our server any more

    def _rotate_log(self) -> None:
        log = self.log_path
        if log.is_file() and log.stat().st_size > LOG_ROTATE_BYTES:
            log.replace(log.with_name(log.name + '.1'))

    def _spawn(self, cmd: List[str], **kwargs) -> Any:
        """Popen from the long-lived spawner thread (see the module docstring)."""
        if self._spawner is None:
            self._spawner = ThreadPoolExecutor(1, thread_name_prefix='ollama-spawn')
        return self._spawner.submit(self._popen, cmd, **kwargs).result()

    def _start(self, settings: AgentSettings) -> Optional[str]:
        """Launch ``ollama serve`` (lock held); a message on failure, else None."""
        binary = ollama_binary(self._which, self._home)
        if binary == 'ollama' and not self._which('ollama'):
            return ('Ollama is not installed: no ollama on PATH or in '
                    '~/.local/ollama/bin (see documentation/assistant.md).')
        nice = self._which('nice')
        cmd = ([nice, '-n', '10'] if nice else []) + [binary, 'serve']
        extra = {'preexec_fn': _die_with_parent} if sys.platform.startswith('linux') else {}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_log()
            with open(self.log_path, 'ab') as log:
                process = self._spawn(
                    cmd, env=server_env(settings), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, **extra)
        except OSError as exc:
            return f'Could not start Ollama ({exc}); see {self.log_path}.'
        self._own(process)
        identity = self._read_identity(process.pid)
        try:
            if identity is not None:  # no /proc: nothing a later session could verify
                self.pid_path.write_text(f'{process.pid} {identity[1]}\n', encoding='utf-8')
        except OSError:
            pass  # adoption by a later session is a convenience
        return None

    # --------------------------------------------------------------- stop

    def stop_if_idle(self, now: Optional[float] = None) -> bool:
        """Stop our server after ``idle_minutes`` without activity; True if stopped.

        Never while a start is polling, and never while /api/ps lists a loaded
        model: with OLLAMA_KEEP_ALIVE=2m that means another client (another
        GUI window) used it lately.
        """
        with self._lock:
            if self._starting or not self.started_by_us or self._settings is None:
                return False
            now = self.clock() if now is None else now
            if now - self._last_activity <= self._settings.ollama.idle_minutes * 60:
                return False
            if self._models_loaded():
                return False
            process, self._process = self._process, None
            self._not_stopping.clear()  # a start waits until this one has exited
        try:
            self._terminate(process, STOP_GRACE)
        finally:
            self._not_stopping.set()
        return True

    def stop(self, grace: float = STOP_GRACE) -> None:
        """Final stop (window close, exit): end our server; refuse later starts."""
        with self._lock:
            self._closed.set()
            process, self._process = self._process, None
        self._terminate(process, grace)
        if self._spawner is not None:
            self._spawner.shutdown(wait=False)

    def _terminate(self, process, grace: float) -> None:
        """SIGTERM, then SIGKILL after ``grace``; drop the pidfile."""
        if process is None:
            return
        self.pid_path.unlink(missing_ok=True)
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:  # e.g. not ours to signal: give up
                pass
