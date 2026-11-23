"""Running plot renders and reporting what they produced.

The render runs in a subprocess (see :mod:`frictionsim2d.gui.plot_worker`) so it can be
cancelled by killing it, and so matplotlib never installs a second Qt backend
beside ovito's -- G4's "no live canvas" is enforced here, by never importing
matplotlib in this process at all.

The wiring is deliberately the same as `PreviewController`'s, `errorOccurred`
branch included: a process that fails to start never emits `finished`, and an
unwired one leaves `running` True forever.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QProcess, Signal

from ...postprocessing.plot_config import PlotDocument
from ..workspace import Workspace

logger = logging.getLogger(__name__)

__all__ = ['PlotController', 'PlotRequest', 'PlotFigure']


@dataclass(frozen=True)
class PlotFigure:
    """One rendered plot and the files it wrote."""
    index: int
    plot_type: str
    files: List[Path]


@dataclass
class PlotRequest:
    """What to render, and how."""
    document: PlotDocument
    preset: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    plot_indices: Optional[List[int]] = None

    def key(self) -> str:
        """Identity of this render -- the scratch directory's name."""
        digest = hashlib.sha256()
        digest.update(self.document.model_dump_json().encode())
        digest.update(str(self.preset).encode())
        digest.update(json.dumps(self.settings, sort_keys=True,
                                 default=str).encode())
        digest.update(str(self.plot_indices).encode())
        return digest.hexdigest()[:24]


class PlotController(QObject):
    """Renders a plot document and reports the figures."""

    started = Signal()
    progress = Signal(str)
    finished = Signal(object)      # List[PlotFigure]
    failed = Signal(str)
    cancelled = Signal()
    warned = Signal(list)          # List[str]

    def __init__(self, workspace: Workspace,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.plots_dir = workspace.preview_dir / '_plots'
        self._process: Optional[QProcess] = None
        self._result_file: Optional[Path] = None
        self._cancelled = False

    @property
    def running(self) -> bool:
        return self._process is not None

    # ------------------------------------------------------------------ run

    def start(self, request: PlotRequest) -> None:
        """Render this request, replacing any render already in flight."""
        if self.running:
            self.cancel()

        directory = self.plots_dir / request.key()
        directory.mkdir(parents=True, exist_ok=True)
        document_file = directory / 'plots.json'
        document_file.write_text(request.document.model_dump_json(
            indent=2, exclude_unset=True), encoding='utf-8')

        result_file = directory / 'result.json'
        payload = {
            'document_file': str(document_file),
            'output_dir': str(directory / 'figures'),
            'preset': request.preset,
            'settings': request.settings,
            'plot_indices': request.plot_indices,
            'result_file': str(result_file),
        }
        request_file = directory / 'request.json'
        request_file.write_text(json.dumps(payload, indent=2),
                                encoding='utf-8')

        self._result_file = result_file
        self._cancelled = False
        self.started.emit()

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(['-m', 'frictionsim2d.gui.plot_worker', str(request_file)])
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        process.start()

    def cancel(self) -> None:
        """Stop a running render."""
        if self._process is None:
            return
        self._cancelled = True
        self._process.kill()
        self._process.waitForFinished(3000)
        self._process = None
        self.cancelled.emit()

    # -------------------------------------------------------------- signals

    def _on_output(self) -> None:
        if self._process is None:
            return
        raw = bytes(
            self._process.readAllStandardOutput()).decode(errors='replace')
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
            self.failed.emit('The plot render produced no result.')

    def _on_error(self, error: QProcess.ProcessError) -> None:
        """Handle a process Qt will never call `_on_finished` for.

        `finished` is only *not* emitted when the process could not start at
        all. Every other error is still followed by `finished`, which
        `_on_finished` turns into `failed`; reacting to those here too would
        emit `failed` twice for one render.
        """
        if error != QProcess.FailedToStart:
            return
        process, self._process = self._process, None
        if self._cancelled or process is None:
            return
        self.failed.emit('The plot render process failed to start.')

    def _deliver(self, result_file: Path) -> None:
        """Parse a completed render and emit its figures."""
        try:
            result = json.loads(result_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            self.failed.emit(f"Could not read the plot result: {exc}")
            return

        if not result.get('ok'):
            self.failed.emit(result.get('error', 'the plot render failed'))
            return

        warnings = list(result.get('warnings') or [])
        if warnings:
            self.warned.emit(warnings)
        self.finished.emit([
            PlotFigure(index=int(entry['index']),
                       plot_type=str(entry['plot_type']),
                       files=[Path(f) for f in entry['files']])
            for entry in result.get('figures', [])])
