"""Running database queries and downloads, and reporting the result.

Every request runs in a subprocess (see :mod:`src.gui.db_worker`) for the
reasons `PreviewController` does: `import aiida` plus `load_profile` is slow
and process-global, psycopg2 and httpx can hang, and a hung request must be
killable without taking the window with it. The process lifecycle here --
including the `errorOccurred(FailedToStart)` carve-out -- is that controller's,
deliberately.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from ..workspace import Workspace

logger = logging.getLogger(__name__)

__all__ = ['DatabaseController', 'DatabaseRequest', 'central_configured',
          'request_key']


def request_key(kind: str, source: str, payload: Dict[str, Any]) -> str:
    """Identity of one request: what was asked, of which store, with what."""
    digest = hashlib.sha256()
    digest.update(kind.encode())
    digest.update(source.encode())
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:24]


#: What `DatabaseSettings.api_url` is when nobody has set it.
DEFAULT_API_URL = 'http://localhost:8000'


def central_configured(settings: Any) -> bool:
    """Has this installation been pointed at a central database at all?

    There is a difference between "the central database is down" and "there
    is no central database", and only the second is true today. The field
    default is a localhost URL, so a bare default with no key and no host
    beside it means nobody has configured anything -- and every central action
    would end as a ConnectionError string against a server that was never
    supposed to be there.
    """
    database = getattr(settings, 'database', None)
    if database is None:
        return False
    url = (getattr(database, 'api_url', '') or '').strip()
    if not url:
        return False
    if url != DEFAULT_API_URL:
        return True
    return bool((getattr(database, 'api_key', '') or '').strip()
                or (getattr(database, 'host', '') or '').strip())


@dataclass
class DatabaseRequest:
    """One database request: a kind, a source, and the kind's own fields."""

    kind: str
    source: str = 'local'
    payload: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return request_key(self.kind, self.source, self.payload)


class DatabaseController(QObject):
    """Runs one database request at a time and reports what came back."""

    started = Signal(str)              # kind
    progress = Signal(str)             # message
    finished = Signal(str, object)     # kind, the worker's result dict
    failed = Signal(str, str)          # kind, message
    cancelled = Signal(str)            # kind

    #: The API key travels in the child's environment, never in `request.json`.
    API_KEY_ENV = 'FRICTIONSIM2D_API_KEY'

    #: Kinds that write outside the workspace. Killing one mid-flight leaves a
    #: half-written destination directory behind, so the window refuses to
    #: start anything while one of these is running.
    WRITE_KINDS: Tuple[str, ...] = ('export', 'import')

    def __init__(self, workspace: Workspace, settings: Any,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        #: The live `GlobalSettings`. Re-pointed by the window whenever the
        #: settings document is applied, so a request always uses the profile
        #: and API URL currently on screen.
        self.settings = settings
        self._process: Optional[QProcess] = None
        self._result_file: Optional[Path] = None
        self._kind = ''
        self._cancelled = False

    @property
    def running(self) -> bool:
        return self._process is not None

    @property
    def current_kind(self) -> str:
        """What is running, or '' when nothing is."""
        return self._kind if self.running else ''

    def request_dir(self, key: str) -> Path:
        """Where this request's files live. Under `.preview` because it is
        the same kind of disposable, workspace-local scratch."""
        return self.workspace.preview_dir / '_db' / key

    # ------------------------------------------------------------------ run

    def start(self, request: DatabaseRequest) -> None:
        """Run a request, replacing any request already in flight."""
        if self.running:
            self.cancel()

        directory = self.request_dir(request.key())
        directory.mkdir(parents=True, exist_ok=True)
        result_file = directory / 'result.json'

        payload = self._payload(request)
        payload['result_file'] = str(result_file)
        request_file = directory / 'request.json'
        request_file.write_text(json.dumps(payload, indent=2, default=str),
                                encoding='utf-8')

        self._kind = request.kind
        self._result_file = result_file
        self._cancelled = False
        self.started.emit(request.kind)

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(['-m', 'src.gui.db_worker', str(request_file)])
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setProcessEnvironment(self._environment(request))
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        process.start()

    def cancel(self) -> None:
        """Stop a running request."""
        if self._process is None:
            return
        self._cancelled = True
        self._process.kill()
        self._process.waitForFinished(3000)
        self._process = None
        self.cancelled.emit(self._kind)

    # -------------------------------------------------------------- helpers

    def _payload(self, request: DatabaseRequest) -> Dict[str, Any]:
        """The request as the worker reads it: the pane's fields plus the
        connection settings, which the pane never sees (D9). The API key is
        deliberately absent -- see `_environment`."""
        payload: Dict[str, Any] = {'kind': request.kind, 'source': request.source}
        payload.update(request.payload)
        if request.source == 'central':
            payload['api_url'] = self.settings.database.api_url
            payload['configured'] = central_configured(self.settings)
        else:
            payload['profile'] = self.settings.aiida.profile
        return payload

    def _environment(self, request: DatabaseRequest) -> QProcessEnvironment:
        """The child's environment, carrying the API key for a central request.

        Not `request.json`: that file stays in the workspace as the D13 cache
        and is never pruned, so a key written into it outlives the session.
        """
        environment = QProcessEnvironment.systemEnvironment()
        key = (self.settings.database.api_key or '').strip() \
            if request.source == 'central' else ''
        if key:
            environment.insert(self.API_KEY_ENV, key)
        else:
            environment.remove(self.API_KEY_ENV)
        return environment

    # -------------------------------------------------------------- signals

    def _on_output(self) -> None:
        if self._process is None:
            return
        raw = bytes(self._process.readAllStandardOutput()).decode(errors='replace')
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('kind') == 'progress':
                self.progress.emit(record.get('message', ''))

    def _on_finished(self) -> None:
        process, self._process = self._process, None
        if self._cancelled or process is None:
            return
        if self._result_file and self._result_file.exists():
            self._deliver(self._result_file)
        else:
            self.failed.emit(self._kind, 'The database request produced no result.')

    def _on_error(self, error: QProcess.ProcessError) -> None:
        """A process that Qt will never call `_on_finished` for.

        `finished` is not emitted for exactly one reason -- the process could
        not be started at all. Every other error is still followed by
        `finished`, which already turns a missing result into `failed`;
        reacting to those here would emit `failed` twice.
        """
        if error != QProcess.FailedToStart:
            return
        process, self._process = self._process, None
        if self._cancelled or process is None:
            return
        self.failed.emit(self._kind, 'The database request process failed to start.')

    def _deliver(self, result_file: Path) -> None:
        try:
            result = json.loads(result_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            self.failed.emit(self._kind, f"Could not read the database result: {exc}")
            return
        if not result.get('ok'):
            self.failed.emit(self._kind,
                             result.get('error', 'the database request failed'))
            return
        self.finished.emit(self._kind, result)
