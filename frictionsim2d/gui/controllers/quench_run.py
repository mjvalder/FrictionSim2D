"""Running a generated quench root here ("Run locally").

Only ever started by the user's explicit action: the Builder offers it on the
Amorphous block family, unticked, and ``start`` is called from nowhere else.
The run is a subprocess (``gui.quench_worker``) so Cancel can kill ``lmp``;
a cancelled run imports nothing.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QProcess, Signal

from ..workspace import Workspace

logger = logging.getLogger(__name__)

__all__ = ['QuenchRunController']


class QuenchRunController(QObject):
    """Runs the members of one quench root serially, then imports the blocks."""

    started = Signal()
    progress = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, workspace: Workspace, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self._process: Optional[QProcess] = None
        self._result_file: Optional[Path] = None
        self._cancelled = False
        self._pending = ''

    @property
    def running(self) -> bool:
        return self._process is not None

    def start(self, root: Path) -> None:
        """Run ``root``'s members. Hours per member; call only on a user's request."""
        if self.running:
            self.cancel()

        staging = self.workspace.preview_dir / '_quench_run'
        staging.mkdir(parents=True, exist_ok=True)
        result_file = staging / 'result.json'
        result_file.unlink(missing_ok=True)
        request_file = staging / 'request.json'
        request_file.write_text(json.dumps(
            {'root': str(root), 'result_file': str(result_file)}, indent=2),
            encoding='utf-8')

        self._result_file = result_file
        self._cancelled = False
        self._pending = ''
        self.started.emit()

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(['-m', 'frictionsim2d.gui.quench_worker', str(request_file)])
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        process.start()

    def cancel(self) -> None:
        """Stop the run: SIGTERM lets the worker kill ``lmp``; nothing is imported."""
        if self._process is None:
            return
        self._cancelled = True
        process, self._process = self._process, None
        process.terminate()
        if not process.waitForFinished(5000):
            process.kill()
            process.waitForFinished(3000)
        self.cancelled.emit()

    def _on_output(self) -> None:
        if self._process is None:
            return
        raw = bytes(self._process.readAllStandardOutput()).decode(errors='replace')
        # A chunk can end mid-line: keep the unfinished tail for the next one.
        *lines, self._pending = (self._pending + raw).split('\n')
        for line in lines:
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if record.get('kind') == 'progress':
                self.progress.emit(record.get('message', ''))

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.FailedToStart:
            return
        process, self._process = self._process, None
        if self._cancelled or process is None:
            return
        self.failed.emit('The quench worker failed to start.')

    def _on_finished(self) -> None:
        process, self._process = self._process, None
        if self._cancelled or process is None:
            return
        if not (self._result_file and self._result_file.exists()):
            self.failed.emit('The quench run produced no result (the worker stopped early).')
            return
        try:
            result = json.loads(self._result_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            self.failed.emit(f"Could not read the quench result: {exc}")
            return
        self.finished.emit(result)
