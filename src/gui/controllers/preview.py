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
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QProcess, Signal

from ...core.config_io import write_config
from ...core.run import expand_config_sweeps
from .staged_paths import absolutize_staged_paths
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


def _first_sweep_member(
        config: Dict[str, Any],
        base_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], Optional[str]]:
    """The first member of a sweep to preview, and a label naming it.

    A material (or other) sweep expands to one config per combination.
    Handing the whole thing to ``generate_package`` -- as the preview used
    to -- built every member just to show one: slow, and which member landed
    first in ``result.created`` was arbitrary from the user's point of view.

    This picks the first member deterministically, through the same
    ``expand_config_sweeps`` a real run uses rather than a re-implemented
    selection rule, so it is genuinely one the real run would produce:
    placeholders substituted, ``materials_list`` stripped.

    Args:
        base_dir: The document's own directory, i.e. what a relative
            ``materials_list`` is relative to. Without this, a swept config
            opened from anywhere other than the current working directory
            silently fails to resolve its material list here (see
            ``expand_config_sweeps``, which only logs and returns no
            materials on a resolution failure) and previews as if it had no
            sweep at all -- ``{mat}`` unsubstituted.

    Returns ``config`` unchanged, with no label, when it carries no sweep --
    including when expansion itself fails (e.g. an unreadable
    materials_list): the real build hits the same error and reports it
    through the existing failure path, so silently falling back here is
    just deferring to that, not hiding anything.
    """
    try:
        members = expand_config_sweeps(config, base_dir)
    except Exception:  # pylint: disable=broad-except
        return config, None
    if len(members) <= 1:
        return config, None

    member = members[0]
    material = member.get('2D', {}).get('mat')
    shown = f"material {material}" if material else "the first configuration"
    label = f"Previewing {shown} (1 of {len(members)} in the sweep)."
    return member, label


@dataclass
class PreviewRequest:
    """One preview build."""

    config: Dict[str, Any]
    model: str
    layer: Optional[int] = None
    settings_file: Optional[Path] = None
    referenced_files: List[Path] = field(default_factory=list)
    #: The *original* document's directory (`ConfigDocument.base_dir`) --
    #: where relative `cif_path`/`pot_path`/`materials_list` values are meant
    #: to resolve against. Not the directory the preview is staged into; see
    #: `staged_paths.absolutize_staged_paths`, which `PreviewController.start`
    #: uses this for.
    base_dir: Optional[Path] = None
    #: Memoized (single-member config, label) -- see `_resolve`. Not part of
    #: the request's identity: two requests with the same `config` must
    #: resolve to the same member regardless of whether either has resolved
    #: yet.
    _resolved: Optional[Tuple[Dict[str, Any], Optional[str]]] = field(
        default=None, init=False, repr=False, compare=False)

    def _resolve(self) -> Tuple[Dict[str, Any], Optional[str]]:
        if self._resolved is None:
            self._resolved = _first_sweep_member(self.config, self.base_dir)
        return self._resolved

    def single_member(self) -> Dict[str, Any]:
        """The config to actually build.

        `config` unchanged, or the first member of the sweep it configures --
        never the whole sweep, which is what made the previewed material
        arbitrary and the build take one pass per material.
        """
        return self._resolve()[0]

    def preview_label(self) -> Optional[str]:
        """Which sweep member `single_member` is, or None outside a sweep."""
        return self._resolve()[1]

    def key(self) -> str:
        return cache_key(
            self.single_member(), self.model, self.layer, self.referenced_files)


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
        #: Which sweep member `finished`/`failed` refer to, or None outside a
        #: sweep -- set in `start`, read back by the caller once the build
        #: settles, so the shown material is never a silent, arbitrary choice.
        self.current_label: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._process is not None

    # ------------------------------------------------------------------ run

    def start(self, request: PreviewRequest) -> None:
        """Build (or reuse) a preview for this request.

        A sweep is reduced to its first member here, before anything is
        written to disk: the worker builds exactly what `request.key()` was
        computed from, so the cache and the build always agree on which
        member this is.
        """
        if self.running:
            self.cancel()

        config = request.single_member()
        self.current_label = request.preview_label()

        key = request.key()
        directory = self.workspace.preview_path(key)
        result_file = directory / 'preview.json'

        self.started.emit()
        if self.current_label:
            self.progress.emit(self.current_label)

        if result_file.exists():
            self.progress.emit('reusing cached preview')
            self._deliver(result_file)
            return

        directory.mkdir(parents=True, exist_ok=True)
        # The config is about to be written into the preview cache directory,
        # not wherever the document itself lives -- absolutise its relative
        # paths against the *original* document's directory first, or
        # search_paths would resolve them (wrongly) against `directory`
        # instead. See `staged_paths` for why.
        staged_config = absolutize_staged_paths(config, request.base_dir)
        config_path = write_config(staged_config, directory / 'config.ini')

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
        process.errorOccurred.connect(self._on_error)
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

    def _on_error(self, error: QProcess.ProcessError) -> None:
        """Handle a process that Qt will never call `_on_finished` for.

        `finished` is only *not* emitted for one reason: the process could
        not be started at all (bad interpreter path, missing module,
        permission denied) -- Qt's docs call this out explicitly. Every
        other `QProcess.ProcessError` (crash, timeout, I/O) is still
        followed by `finished`, which `_on_finished` already turns into
        `failed` when no result file appears; reacting to those here too
        would emit `failed` twice for one build.

        Without this, a failed-to-start process left `_process` set and
        `running` True forever -- indistinguishable, from the Render
        button's point of view, from a build that runs forever.
        """
        if error != QProcess.FailedToStart:
            return
        process, self._process = self._process, None
        if self._cancelled or process is None:
            return
        self.failed.emit('The preview build process failed to start.')

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
