"""Running preview builds and turning them into something displayable.

The build runs in a subprocess (see :mod:`src.gui.preview_worker`) so it can be
cancelled by killing it, and so a LAMMPS abort cannot take the application down.

Results are cached by a key that includes a fingerprint of the builder and
template sources: without that, editing a template and re-rendering would show
a system built by the previous code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QObject, QProcess, Signal

from ...core.config_io import write_config
from ..preview.assemble import AssembledSystem, assemble
from ..preview.init_script import parse_assembly
from ..preview.settings_file import parse_identity
from ..workspace import Workspace

logger = logging.getLogger(__name__)

__all__ = ['PreviewController', 'PreviewRequest', 'cache_key']

#: Source trees whose contents change what a build produces.
_CODE_ROOTS = ('builders', 'templates')

_code_fingerprint_cache: Optional[str] = None


def code_fingerprint() -> str:
    """A digest of the code that turns a config into a package.

    Included in the cache key because "same config means same output" is only
    true if the code is part of the input -- and on this project the builders
    are under active development.
    """
    global _code_fingerprint_cache  # pylint: disable=global-statement
    if _code_fingerprint_cache is not None:
        return _code_fingerprint_cache

    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[2]
    for name in _CODE_ROOTS:
        root = package_root / name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.lmp', '.j2'):
                digest.update(path.name.encode())
                digest.update(str(path.stat().st_mtime_ns).encode())

    _code_fingerprint_cache = digest.hexdigest()[:16]
    return _code_fingerprint_cache


def cache_key(config: Dict[str, Any], model: str, layer: Optional[int],
              referenced_files: Optional[List[Path]] = None) -> str:
    """Identity of a preview build."""
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True, default=str).encode())
    digest.update(model.encode())
    digest.update(str(layer).encode())
    digest.update(code_fingerprint().encode())

    for path in sorted(referenced_files or []):
        try:
            stat = Path(path).stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        except OSError:
            digest.update(f"{path}:missing".encode())

    return digest.hexdigest()[:24]


@dataclass
class PreviewRequest:
    """One preview build."""

    config: Dict[str, Any]
    model: str
    layer: Optional[int] = None
    settings_file: Optional[Path] = None
    referenced_files: List[Path] = field(default_factory=list)

    def key(self) -> str:
        return cache_key(self.config, self.model, self.layer, self.referenced_files)


class PreviewController(QObject):
    """Builds a preview and reports the result."""

    started = Signal()
    progress = Signal(str)
    finished = Signal(object)      # AssembledSystem
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

    # ------------------------------------------------------------------ run

    def start(self, request: PreviewRequest) -> None:
        """Build (or reuse) a preview for this request."""
        if self.running:
            self.cancel()

        key = request.key()
        directory = self.workspace.preview_path(key)
        result_file = directory / 'preview.json'

        self.started.emit()

        if result_file.exists():
            self.progress.emit('reusing cached preview')
            self._deliver(result_file)
            return

        directory.mkdir(parents=True, exist_ok=True)
        config_path = write_config(request.config, directory / 'config.ini')

        payload = {
            'config_file': str(config_path),
            'model': request.model,
            'output_root': str(directory),
            'layer': request.layer,
            'settings_file': str(request.settings_file) if request.settings_file else None,
            'result_file': str(result_file),
        }
        request_file = directory / 'request.json'
        request_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')

        self._result_file = result_file
        self._cancelled = False

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(['-m', 'src.gui.preview_worker', str(request_file)])
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        self._process = process
        process.start()

    def cancel(self) -> None:
        """Stop a running build."""
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
            self.failed.emit('The preview build produced no result.')

    def _deliver(self, result_file: Path) -> None:
        """Parse a completed build and emit the assembled system."""
        try:
            result = json.loads(result_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            self.failed.emit(f"Could not read the preview result: {exc}")
            return

        if not result.get('ok'):
            self.failed.emit(result.get('error', 'the preview build failed'))
            return

        try:
            assembly = parse_assembly(
                Path(result['init_script']), Path(result['simulation_root']))
            identity = parse_identity(Path(result['settings_file'])) \
                if result.get('settings_file') else None
            system = assemble(assembly, identity)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to assemble the preview")
            self.failed.emit(f"Could not read the generated system: {exc}")
            return

        self.finished.emit(system)
