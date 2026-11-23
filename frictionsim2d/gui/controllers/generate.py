"""Generating the real simulation package.

Unlike the preview this is the deliverable, so it runs the full sweep, may emit
HPC scripts, and is allowed to generate a missing amorphous structure -- that is
a deliberate act here, rather than a side effect of clicking Render.

Runs in a subprocess for the same reasons as the preview: it can be cancelled,
and a LAMMPS abort cannot take the application down.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QProcess, Signal

from ...core.config import write_config
from .staged_paths import absolutize_staged_paths
from ..workspace import Workspace

logger = logging.getLogger(__name__)

__all__ = ['GenerateController', 'GenerateRequest']


@dataclass
class GenerateRequest:
    """One package generation."""

    config: Dict[str, Any]
    model: str
    generate_hpc: bool = False
    settings_file: Optional[Path] = None
    root_name: Optional[str] = None
    #: The *original* document's directory (`ConfigDocument.base_dir`) --
    #: where relative `cif_path`/`pot_path`/`materials_list` values are meant
    #: to resolve against, as opposed to the `_generate` staging directory
    #: this request's config is actually written into. See
    #: `staged_paths.absolutize_staged_paths`, which `GenerateController.start`
    #: uses this for.
    base_dir: Optional[Path] = None


class GenerateController(QObject):
    """Generates a package into the workspace."""

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

    @property
    def running(self) -> bool:
        return self._process is not None

    def start(self, request: GenerateRequest) -> None:
        if self.running:
            self.cancel()

        staging = self.workspace.preview_dir / '_generate'
        staging.mkdir(parents=True, exist_ok=True)
        # The config is about to be written somewhere other than where it
        # lived when its relative paths were written -- absolutise them
        # against the *original* document's directory first, or search_paths
        # would resolve them (wrongly) against `staging` instead. See
        # `staged_paths` for why.
        staged_config = absolutize_staged_paths(request.config, request.base_dir)
        config_path = write_config(staged_config, staging / 'config.ini')
        result_file = staging / 'result.json'
        if result_file.exists():
            result_file.unlink()

        payload = {
            'config_file': str(config_path),
            'model': request.model,
            'output_root': str(self.workspace.runs_dir),
            'generate_hpc': request.generate_hpc,
            'settings_file': str(request.settings_file) if request.settings_file else None,
            'root_name': request.root_name,
            'result_file': str(result_file),
        }
        request_file = staging / 'request.json'
        request_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')

        self._result_file = result_file
        self._cancelled = False
        self.started.emit()

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(['-m', 'frictionsim2d.gui.generate_worker', str(request_file)])
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        self._process = process
        process.start()

    def cancel(self) -> None:
        if self._process is None:
            return
        self._cancelled = True
        self._process.kill()
        self._process.waitForFinished(3000)
        self._process = None
        self.cancelled.emit()

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

        if not (self._result_file and self._result_file.exists()):
            self.failed.emit('Generation produced no result.')
            return

        try:
            result = json.loads(self._result_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            self.failed.emit(f"Could not read the generation result: {exc}")
            return

        if result.get('ok'):
            self.finished.emit(result)
        else:
            self.failed.emit(result.get('error', 'generation failed'))
