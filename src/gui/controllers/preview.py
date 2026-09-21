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

from ...core.config import write_config
from ...run.expand import expand_config_sweeps
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

    Whether a material sweep actually happened is decided by whether
    ``materials_list`` disappeared between ``config`` and the returned
    member -- not by ``len(members)`` -- because those are different
    questions. A sweep of exactly one selected material still substitutes
    ``{mat}`` and strips ``materials_list``, so ``expand_config_sweeps``
    correctly returns a single, already-substituted member; treating "one
    member" as "no sweep" (the previous rule here) discarded that
    substitution and previewed the original config with a literal ``{mat}``
    still in ``cif_path``/``pot_path``, which the builder cannot resolve.
    Conversely, a config with no ``materials_list`` at all also produces
    exactly one, unsubstituted member -- ``materials_list`` was never there
    to strip -- and that case must still fall back to ``config`` unchanged.

    The sheet section is looked up the way ``expand_config_sweeps`` itself
    looks it up -- a plain ``[2D]`` if there is one, else the first numbered
    ``[2D-N]`` (see ``_primary_sheet_key``). It used to check ``[2D]`` alone,
    on the since-outdated grounds that expansion only ever read
    ``base_config['2D']``; it was taught the numbered fallback, and a
    heterostructure config has ``[2D-1]``/``[2D-2]`` and never a plain
    ``[2D]``, so the old rule reported ``material_swept`` False for *every*
    pair sweep. Two or more pairs were saved by the ``len(members)`` term;
    exactly one pair -- add a material, add one pair, Render -- had both
    terms False and previewed the original config, literal ``{mat1}`` in
    ``cif_path`` and all.
    """
    try:
        members = expand_config_sweeps(config, base_dir)
    except Exception:  # pylint: disable=broad-except
        return config, None

    member = members[0]
    sheet = _primary_sheet_key(config)
    material_swept = (
        'materials_list' in config.get(sheet, {})
        and 'materials_list' not in member.get(sheet, {})
    )
    if len(members) <= 1 and not material_swept:
        return config, None

    material = _member_material(member, sheet)
    # What the sweep is over, so a pair sweep does not read as "the only
    # material in the sweep" while naming two of them.
    noun = 'pair' if material and ' + ' in material else 'material'
    if material_swept and len(members) == 1:
        shown = f"{noun} {material}" if material else "the configured material"
        label = f"Previewing {shown} (the only {noun} in the sweep)."
    else:
        shown = f"{noun} {material}" if material else "the first configuration"
        label = f"Previewing {shown} (1 of {len(members)} in the sweep)."
    return member, label


def _primary_sheet_key(config: Dict[str, Any]) -> str:
    """The section a ``materials_list`` lives in: ``[2D]``, else ``[2D-1]``.

    Mirrors ``expand_config_sweeps``' own choice of primary key (plain
    ``'2D'`` when present, otherwise the lowest-numbered ``2D-N``) so the two
    are asking about the same section. Falls back to ``'2D'`` when there is
    no sheet section at all, which reads the same as the old behaviour: a
    missing key means no sweep.
    """
    if '2D' in config:
        return '2D'
    numbered = sorted(
        (k for k in config if isinstance(k, str) and k.startswith('2D-')),
        key=lambda key: int(key.split('-', 1)[1])
        if key.split('-', 1)[1].isdigit() else 10**9,
    )
    return numbered[0] if numbered else '2D'


def _member_material(member: Dict[str, Any], sheet: str) -> Optional[str]:
    """What to call the member on screen: ``mat``, or ``mat1 + mat2``.

    A pair sweep's member is a stack of two materials and naming only the
    first would be actively misleading -- ``h_MoS2`` on ``h_WS2`` and
    ``h_MoS2`` on ``h_graphene`` are different runs with the same first
    sheet.
    """
    first = member.get(sheet, {}).get('mat')
    second = member.get('2D-2', {}).get('mat') if sheet != '2D-2' else None
    if first and second:
        return f"{first} + {second}"
    return first


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


def _cached_success(result_file: Path) -> bool:
    """Whether a cached preview result exists and records a successful build."""
    try:
        return bool(json.loads(result_file.read_text(encoding='utf-8')).get('ok'))
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


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

        if _cached_success(result_file):
            self.progress.emit('reusing cached preview')
            self._deliver(result_file)
            return
        # A failure is never reused: its cause can lie outside the cache key
        # (a path staged wrongly, a potential file added since), so replaying
        # it would outlive the fix.
        result_file.unlink(missing_ok=True)

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
