"""Starting a local Ollama server on demand, and stopping it when idle.

Qt-free so it is tested with fakes: every process, HTTP call and clock read
goes through an injected callable. A server this object did not start is
never stopped -- the user may be using it for something else.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .providers import ollama_binary
from .settings import AgentSettings

#: How long a fresh ``ollama serve`` gets to answer /api/version.
START_TIMEOUT = 20.0
#: How long a stopped server gets between SIGTERM and SIGKILL.
STOP_GRACE = 5.0
_POLL_EVERY = 0.25


def cache_dir() -> Path:
    """FrictionSim2D's per-user cache directory, honouring XDG_CACHE_HOME."""
    base = os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
    return Path(base) / 'FrictionSim2D'


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


class OllamaServer:  # pylint: disable=too-many-instance-attributes
    """One ``ollama serve`` process this app may own.

    `ensure_running` may block for up to `START_TIMEOUT` (call it off the GUI
    thread); `touch`, `stop_if_idle` and `stop` are quick except for the
    SIGTERM grace period of a real stop.
    """

    def __init__(self, *, which: Callable[[str], Optional[str]] = shutil.which,
                 home: Optional[Path] = None, popen: Callable[..., Any] = subprocess.Popen,
                 http_get: Callable[..., Any] = httpx.get,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 log_path: Optional[Path] = None,
                 start_timeout: float = START_TIMEOUT) -> None:
        self._which = which
        self._home = home
        self._popen = popen
        self._http_get = http_get
        self._clock = clock
        self._sleep = sleep
        self._log_path = log_path
        self._start_timeout = start_timeout
        self._lock = threading.Lock()
        self._process = None
        self._settings: Optional[AgentSettings] = None
        self._last_activity = clock()

    @property
    def log_path(self) -> Path:
        """Where the server's stdout and stderr go."""
        return self._log_path or cache_dir() / 'ollama.log'

    @property
    def started_by_us(self) -> bool:
        """True while a server this object started is still running."""
        process = self._process
        return process is not None and process.poll() is None

    def is_up(self, settings: AgentSettings) -> bool:
        """Whether a server answers /api/version at the configured URL."""
        url = settings.ollama.base_url.rstrip('/') + '/api/version'
        try:
            response = self._http_get(url, timeout=1.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def touch(self) -> None:
        """Record activity: a turn started or ended."""
        self._last_activity = self._clock()

    def ensure_running(self, settings: AgentSettings) -> Tuple[bool, str]:
        """Make sure a server answers, starting one when none does."""
        self._settings = settings
        self.touch()
        if self.is_up(settings):
            return True, f'Ollama is running at {settings.ollama.base_url}.'
        with self._lock:
            if not self.started_by_us:
                failure = self._start(settings)
                if failure:
                    return False, failure
            process = self._process
        deadline = self._clock() + self._start_timeout
        while self._clock() < deadline:
            code = process.poll()
            if code is not None:
                return False, (f'Ollama exited while starting (code {code}); '
                               f'see {self.log_path}.')
            if self.is_up(settings):
                self.touch()
                return True, 'Ollama started.'
            self._sleep(_POLL_EVERY)
        return False, (f'Ollama did not answer within {self._start_timeout:.0f} s; '
                       f'see {self.log_path}.')

    def _start(self, settings: AgentSettings) -> Optional[str]:
        """Launch ``ollama serve``; a message on failure, else None."""
        binary = ollama_binary(self._which, self._home)
        if binary == 'ollama' and not self._which('ollama'):
            return ('Ollama is not installed: no ollama on PATH or in '
                    '~/.local/ollama/bin (see documentation/assistant.md).')
        nice = self._which('nice')
        cmd = ([nice, '-n', '10'] if nice else []) + [binary, 'serve']
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'ab') as log:
                self._process = self._popen(
                    cmd, env=server_env(settings), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            return f'Could not start Ollama ({exc}); see {self.log_path}.'
        return None

    def stop_if_idle(self, now: Optional[float] = None) -> bool:
        """Stop our server after ``idle_minutes`` without activity; True if stopped."""
        if not self.started_by_us or self._settings is None:
            return False
        now = self._clock() if now is None else now
        if now - self._last_activity <= self._settings.ollama.idle_minutes * 60:
            return False
        self.stop()
        return True

    def stop(self) -> None:
        """Stop the server this object started (SIGTERM, SIGKILL after 5 s)."""
        with self._lock:
            process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=STOP_GRACE)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
