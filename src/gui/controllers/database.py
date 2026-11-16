"""Running database queries and downloads, and reporting the result.

Every request runs in a subprocess (see :mod:`src.gui.db_worker`) for the
reasons `PreviewController` does: `import aiida` plus `load_profile` is slow
and process-global, psycopg2 can hang, and a hung request must be
killable without taking the window with it. The process lifecycle here --
including the `errorOccurred(FailedToStart)` carve-out -- is that controller's,
deliberately.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QProcess, Signal

from ..workspace import Workspace

logger = logging.getLogger(__name__)

__all__ = ['CACHE_KEEP', 'DatabaseController', 'DatabaseRequest',
          'request_key']

#: How many keyed query-cache directories `.preview/_db` keeps. A cached
#: `list` result is a few hundred kB of rows; ten covers the handful of
#: queries a session cycles between and still bounds the directory.
CACHE_KEEP = 10


def request_key(kind: str, source: str, payload: Dict[str, Any]) -> str:
    """Identity of one request: what was asked, of which store, with what."""
    digest = hashlib.sha256()
    digest.update(kind.encode())
    digest.update(source.encode())
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:24]


def _cached_success(result_file: Path) -> bool:
    """Whether a cached result exists and records a successful request."""
    try:
        return bool(json.loads(result_file.read_text(encoding='utf-8')).get('ok'))
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


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

    #: Kinds that write outside the workspace. Killing one mid-flight leaves a
    #: half-written destination directory behind, so the window refuses to
    #: start anything while one of these is running.
    WRITE_KINDS: Tuple[str, ...] = ('export', 'import')

    #: Kinds whose result may be served from disk. A download or an import
    #: must always run: its result file describes work, not an answer.
    CACHED_KINDS: Tuple[str, ...] = ('list',)

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
        """Where the files of the request keyed `key` live. Under `.preview`
        because it is the same kind of disposable, workspace-local scratch."""
        return self._db_dir / key

    def directory_for(self, request: DatabaseRequest) -> Path:
        """Where `request`'s `request.json` and `result.json` go.

        A cached kind gets its own directory, keyed by `cache_key`. Every
        other kind reuses one scratch directory per kind: a `detail` fires on
        each row click, and its result is never read back from disk, so a
        directory per click would only grow `.preview/_db` without bound.

        Every distinct query is a new keyed directory, though, and nothing
        else deletes them: `Workspace.prune_previews` skips `_`-prefixed
        entries (it must, or eight renders would throw the cache away) and
        `_invalidate_cached_lists` only unlinks the result files. So a new
        keyed directory prunes its siblings to the newest `CACHE_KEEP`.
        """
        if request.kind in self.CACHED_KINDS:
            directory = self.request_dir(self.cache_key(request))
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                self._prune_cached_results()
            return directory
        return self._db_dir / '_scratch' / request.kind

    def _prune_cached_results(self, keep: int = CACHE_KEEP) -> List[Path]:
        """Delete all but the newest `keep` keyed cache directories.

        `_scratch` is not one of them -- it is where every uncached kind
        writes, and removing it under a running request would drop the result
        file the worker is about to write.
        """
        entries = sorted(
            (p for p in self._db_dir.iterdir()
             if p.is_dir() and p.name != '_scratch'),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        removed = []
        for stale in entries[keep:]:
            shutil.rmtree(stale, ignore_errors=True)
            removed.append(stale)
        return removed

    def cache_key(self, request: DatabaseRequest) -> str:
        """Identity of a result: the request exactly as the worker reads it.

        That is `_payload`, connection included -- the same filters against
        another profile are another store's answer.
        """
        payload = self._payload(request)
        return request_key(request.kind, request.source, payload)

    @property
    def _db_dir(self) -> Path:
        return self.workspace.preview_dir / '_db'

    # ------------------------------------------------------------------ run

    def start(self, request: DatabaseRequest, *, use_cache: bool = True) -> None:
        """Run a request, replacing any request already in flight.

        A successful `list` result is keyed by `cache_key` and reused, because
        a query over a store with thousands of nodes takes seconds. Refresh
        passes `use_cache=False`. Either way a result that is not reused is
        deleted before the worker starts, so a failed or killed re-query can
        never fall back to it -- and, as in `PreviewController`, a cached
        failure is never replayed: its cause (a profile, a server) can lie
        outside the key, and replaying it would outlive the fix.
        """
        if self.running:
            self.cancel()

        directory = self.directory_for(request)
        directory.mkdir(parents=True, exist_ok=True)
        result_file = directory / 'result.json'

        if (use_cache and request.kind in self.CACHED_KINDS
                and _cached_success(result_file)):
            self._kind = request.kind
            self._result_file = result_file
            self._cancelled = False
            self.started.emit(request.kind)
            self.progress.emit('reusing the cached query result')
            self._deliver(result_file, cached=True)
            return
        result_file.unlink(missing_ok=True)

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
        connection settings, which the pane never sees (D9)."""
        payload: Dict[str, Any] = {'kind': request.kind, 'source': request.source}
        payload.update(request.payload)
        payload['profile'] = self.settings.aiida.profile
        return payload

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

    def _deliver(self, result_file: Path, *, cached: bool = False) -> None:
        """Hand one result to the window, saying whether it came off disk.

        `elapsed_s` is the original query's, so a replayed result otherwise
        reports a fresh-looking timing for a store nobody has touched since.
        The window says so in the message panel; the flag is set here because
        this is the only place that knows.
        """
        try:
            result = json.loads(result_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            self.failed.emit(self._kind, f"Could not read the database result: {exc}")
            return
        if not result.get('ok'):
            self.failed.emit(self._kind,
                             result.get('error', 'the database request failed'))
            return
        if self._kind == 'import':
            self._invalidate_cached_lists()
        result['cached'] = cached
        self.finished.emit(self._kind, result)

    def _invalidate_cached_lists(self) -> None:
        """Forget every cached query result: an import changed the store they
        describe, and a Search that replayed one would hide the new rows."""
        if not self._db_dir.is_dir():
            return
        for directory in self._db_dir.iterdir():
            if directory.is_dir() and directory.name != '_scratch':
                (directory / 'result.json').unlink(missing_ok=True)
