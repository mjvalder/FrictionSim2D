"""Import-only CalcJob and Parser for finished FrictionSim2D runs (spec §5.1).

One :class:`LammpsFrictionCalcJob` per LAMMPS deck of a run. It is launched
with ``engine.run`` and a ``remote_folder`` pointing at the run directory, so
aiida-core skips upload and submit and only retrieves and parses: LAMMPS is
never launched through AiiDA. The import path hard-codes ``LocalTransport``
(``CalcJob._perform_import``), so the run directory must be readable on this
machine; the ``RemoteData``'s Computer (unconfigured, carrying the real host
and scheduler, see :func:`src.aiida.store.machine_computer`) only records
where it came from.

What each deck's node carries:

* inputs -- the deck itself, the files it ``include``\\ s (``includes``:
  ``lammps/system.in.settings``), the run's ``provenance/`` +
  ``provenance.json``, its parameter blocks and potential files (all
  content-addressed through :mod:`src.aiida.store`, so runs share them), the
  run's ``derived`` values and the family's ``schema`` (units, kinds,
  defaults), and ``structures``: the ``build/*.lmp`` files the first deck
  reads and, for a slide deck, the very ``load_*.data`` nodes the system
  deck output for its loads. That last link is the system -> slide handoff in the graph.
* outputs -- per load point ``points.<key>`` (window + raw tiers, and the
  indent retract curve), ``peratom.<key>`` (the per-atom tier) and
  ``snapshots.<key>`` (the end ``final_*.data``); ``structures`` and ``aux``
  for system decks; ``run_info`` and ``log``.

Retrieval goes through ``retrieve_temporary_list`` and the parser builds the
stored nodes itself, so it can drop Syncthing ``*.sync-conflict-*`` copies
and always keep the original file -- a plain ``retrieve_list`` glob cannot.
Every output file goes through :func:`src.aiida.data.redact.stage_file`
(:func:`src.aiida.store.new_file` / :func:`~src.aiida.store.new_folder`).
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)
from uuid import UUID, uuid5

from aiida import orm
from aiida.common.datastructures import CalcInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser

from .data_keys import content_hash
from .decks import Deck, discover_decks, points_for
from .retention import retrieve_patterns
from .store import (
    StoreIndex, calcjob_extras, get_or_create_file, lammps_code, new_file, new_folder,
)

logger = logging.getLogger(__name__)

__all__ = ['LammpsFrictionCalcJob', 'LammpsFrictionParser', 'MissingRunUUID',
           'RunInputs', 'check_run', 'deck_inputs', 'import_deck', 'import_run',
           'included_files', 'point_label', 'point_uuid', 'read_run_uuid',
           'run_inputs', 'structures_for']


class MissingRunUUID(ValueError):
    """A run's ``provenance.json`` has no ``run_uuid`` (a pre-2.2 build)."""

_CONFLICT = '.sync-conflict-'
_PES_FAMILIES = ('pes_tip', 'pes_sheet')
_PES_REFINE_NAME = 'pes_scan_refine.csv'
_TOL = 1e-9

#: Tiers (keys of `LoadPoint.tiers`) stored on the point node itself; the
#: per-atom tier gets its own node, `peratom.<key>`.
_POINT_TIERS = ('window', 'raw', 'retract')
_READ_TIERS = ('window', 'raw', 'peratom')

#: What each kind of deck retrieves, as remote globs of
#: :func:`~src.aiida.retention.retrieve_patterns`: a system deck writes the
#: loaded structures (and the AFM hold traces), a slide-like deck the results
#: and end snapshots.
_SYSTEM_RETRIEVE = ('data/load_*.data', 'data/hold_*.txt')
_SLIDE_RETRIEVE = ('results/*', 'data/final_*.data')

_WALL_RE = re.compile(r'^Total wall time:\s*(\d+):(\d+):(\d+)', re.M)
_PROCS_RE = re.compile(r'^Loop time of \S+ on (\d+) procs', re.M)
_INCLUDE_RE = re.compile(r'^\s*include\s+(\S+)', re.M | re.I)
_LOAD_FILE_RE = re.compile(r'^load_(?P<load>[-+0-9.eE]+)N\.data$')


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _num(value: float) -> str:
    """``f'{value:g}'`` with ``-`` -> ``m`` and ``.`` -> ``p``."""
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def point_label(point: Any) -> str:
    """The key of one load point, e.g. ``force10_a0_s2``.

    The load-condition half of :func:`src.aiida.ingest._point_key`, spelled
    as a valid link label (alphanumerics and single underscores).
    """
    return (f'{point.load_kind}{_num(point.load)}'
            f'_a{_num(point.angle)}_s{_num(point.speed)}')


def point_uuid(run_uuid: str, label: str) -> str:
    """A point's stable identity: ``uuid5(UUID(run_uuid), label)``.

    Path-free, so the same point of the same run gets the same uuid wherever
    the run directory sits, and under whatever set label it is imported.
    """
    return str(uuid5(UUID(run_uuid), label))


def _file_label(name: str) -> str:
    """A link-label-safe key for a file, from its FULL name (``MoS.sw`` ->
    ``MoS_sw``), so files sharing a stem stay apart."""
    text = PurePosixPath(name).name.replace('-', 'm')
    label = re.sub(r'_+', '_', re.sub(r'\W', '_', text)).strip('_') or 'file'
    return label if label[0].isalpha() else f'f_{label}'


def _unique(pairs: Iterable[Tuple[str, Any]], what: str) -> Dict[str, Any]:
    """``dict(pairs)``, but a repeated key is an error, never a silent loss."""
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'{what}: two entries map to the label {key!r}')
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# CalcJob
# ---------------------------------------------------------------------------

class LammpsFrictionCalcJob(CalcJob):
    """One LAMMPS deck of a finished run, imported -- never executed."""

    @classmethod
    def define(cls, spec):
        """Inputs, outputs and exit codes of an imported deck."""
        super().define(spec)

        spec.input('deck', valid_type=orm.SinglefileData,
                   help='The LAMMPS input deck (lammps/<name>.in).')
        spec.input_namespace('includes', valid_type=orm.SinglefileData,
                             dynamic=True, required=False,
                             help='Files the deck includes (system.in.settings).')
        spec.input('provenance', valid_type=orm.FolderData, required=False,
                   help="The run's provenance/ directory plus provenance.json.")
        spec.input_namespace('blocks', valid_type=orm.Dict, dynamic=True,
                             required=False, help='Parameter blocks (spec §5.2).')
        spec.input('derived', valid_type=orm.Dict, required=False,
                   help="The run's derived values (drive_k, lat_c, ...), not hashed.")
        spec.input('schema', valid_type=orm.Dict, required=False,
                   help='The block schema: kind, unit and default of every field.')
        spec.input_namespace('potentials', valid_type=orm.SinglefileData,
                             dynamic=True, required=False,
                             help='Potential files the run used.')
        spec.input_namespace('structures', valid_type=orm.SinglefileData,
                             dynamic=True, required=False,
                             help='Structure files the deck reads: build/*.lmp, '
                                  "and the system deck's load_*.data outputs.")
        spec.input('parameters', valid_type=orm.Dict,
                   help='deck_name, rel_path, run_uuid, point_filter, family, '
                        'n_layers, log_name, writes_structures, block_names, '
                        'build_level.')

        spec.input('metadata.options.resources', valid_type=dict,
                   default={'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        spec.input('metadata.options.parser_name', valid_type=str,
                   default='friction2d.lammps')

        spec.output_namespace('points', valid_type=orm.FolderData, dynamic=True,
                              required=False,
                              help='One FolderData per load point: window + raw tiers.')
        spec.output_namespace('peratom', valid_type=orm.FolderData, dynamic=True,
                              required=False,
                              help='The per-atom tier of a load point.')
        spec.output_namespace('snapshots', valid_type=orm.SinglefileData,
                              dynamic=True, required=False,
                              help="A load point's end snapshot (final_*.data).")
        spec.output_namespace('structures', valid_type=orm.SinglefileData,
                              dynamic=True, required=False,
                              help='load_*.data written by a system deck.')
        spec.output_namespace('aux', valid_type=orm.SinglefileData,
                              dynamic=True, required=False,
                              help='Other retained files of a system deck '
                                   '(hold_*.txt).')
        spec.output('run_info', valid_type=orm.Dict, required=False,
                    help='lammps_version, wall_time_s, n_procs, log_missing, '
                         'log_has_version, not_stored (files over the cap).')
        spec.output('log', valid_type=orm.SinglefileData, required=False,
                    help="The deck's LAMMPS log.")

        spec.exit_code(320, 'ERROR_NO_RESULTS',
                       message='The deck produced neither load points nor structures.')
        spec.exit_code(330, 'ERROR_PARSING_FAILED',
                       message='Parsing the deck outputs failed: {error}')
        spec.exit_code(410, 'ERROR_LAMMPS_NOT_FINISHED',
                       message='LAMMPS did not finish (no Total wall time in the log).')

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Only the retrieve lists: nothing is uploaded and nothing runs."""
        params = self.inputs.parameters.get_dict()  # type: ignore[attr-defined]
        # The run's and the deck's identity and the run's path tokens, as
        # extras: found and filtered on without loading `parameters`.
        # Set here (presubmit runs on import too) so every launch route has it.
        extras = calcjob_extras(params)
        if extras:
            self.node.base.extras.set_many(extras)
        calcinfo = CalcInfo()
        calcinfo.uuid = str(self.uuid)
        calcinfo.codes_info = []
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []
        log_name = params.get('log_name')
        calcinfo.retrieve_list = ([(log_name, '.', len(log_name.split('/')))]
                                  if log_name else [])
        calcinfo.retrieve_temporary_list = _temporary_list(
            bool(params.get('writes_structures')))
        return calcinfo

    def parse_scheduler_output(self, retrieved: orm.Node):
        """An imported deck has no scheduler output; skip it, warning-free."""
        return None


def _temporary_list(writes_structures: bool) -> List[Tuple[str, str, Optional[int]]]:
    """The retention rule, narrowed to what this kind of deck writes.

    Narrowed by category only, not by load: result filenames spell loads with
    LAMMPS' own number formatting, so a per-load glob could silently miss a
    file. The parser selects this deck's points (:func:`decks.points_for`).
    """
    wanted = _SYSTEM_RETRIEVE if writes_structures else _SLIDE_RETRIEVE
    return [entry for entry in retrieve_patterns() if entry[0] in wanted]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _drop_conflict_copies(root: Path) -> None:
    """Delete Syncthing conflict copies from the scratch retrieval."""
    for path in list(root.rglob(f'*{_CONFLICT}*')):
        if path.is_file():
            logger.info('Ignoring Syncthing conflict copy %s', path.name)
            path.unlink()


def _scratch_locator(root: Path, rel_path: str, family: str):
    """A ``RunLocator`` over the scratch retrieval of one run."""
    from dataclasses import replace  # pylint: disable=import-outside-toplevel

    from ..run.naming import parse_run_path  # pylint: disable=import-outside-toplevel
    from ..run.tree import RunLocator  # pylint: disable=import-outside-toplevel

    return RunLocator(
        tokens=replace(parse_run_path(rel_path), family=family), run_dir=root,
        provenance_file=root / 'provenance.json',
        provenance_dir=root / 'provenance', results_dir=root / 'results',
        data_dir=root / 'data', lammps_dir=root / 'lammps')


def _read_frame(path: Path, family: str, n_layers: int):
    """One tier file as a frame, the way the DataReader reads it."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    from ..postprocessing.read_data import read_results_file  # pylint: disable=import-outside-toplevel

    if path.suffix == '.csv':
        return pd.read_csv(path, comment='#')
    return read_results_file(path, family, n_layers)


def _status(tiers: Mapping[str, Path], family: str, n_layers: int,
            expected: Mapping[str, int]) -> Tuple[str, str]:
    """``(status, detail)`` of a point that left files, as the DataReader
    gives it: ``unreadable`` when the window (or every file) fails to parse,
    else :func:`run_status.classify` on the rows read."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    from ..postprocessing.run_status import classify  # pylint: disable=import-outside-toplevel

    n_rows: Dict[str, int] = {}
    failures: List[str] = []
    window_error = None
    for tier, path in sorted(tiers.items()):
        if tier not in _READ_TIERS:
            continue
        try:
            n_rows[tier] = len(_read_frame(path, family, n_layers))
        except (ValueError, OSError, KeyError, IndexError,
                pd.errors.EmptyDataError) as exc:
            failures.append(f'{tier} unreadable: {exc}')
            if tier == 'window':
                window_error = str(exc) or type(exc).__name__
    if window_error is not None or not n_rows:
        return 'unreadable', window_error or '; '.join(failures)
    status, detail = classify(n_rows, dict(expected))
    return status, '; '.join(x for x in (*failures, detail) if x)


def _first(value: Any, default: float) -> float:
    """A scalar config value, or the first entry of a list one."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return default if value is None else float(value)


def _configured_keys(config: Optional[Mapping], family: str) -> List[Tuple]:
    """Every point the run's config asked for, as `run_status.point_key`s.

    `expected_chains` for the slide and PES families. Indent is not modelled
    there (no chains), but its points are still configured: one hold +
    retract per `general.force`, at the run's single angle and speed (the
    template takes a scalar angle and the first speed).
    """
    from ..postprocessing.run_status import expected_chains, point_key  # pylint: disable=import-outside-toplevel

    if family != 'indent':
        return [key for chain in (expected_chains(config, family) or [])
                for key in chain]
    general = (config or {}).get('general')
    if not isinstance(general, Mapping) or general.get('force') is None:
        return []
    try:
        forces = general['force']
        loads = forces if isinstance(forces, (list, tuple)) else [forces]
        angle = _first(general.get('scan_angle'), 0.0)
        speed = _first(general.get('scan_speed'), 2.0)
        return [point_key('force', load, angle, speed) for load in loads]
    except (TypeError, ValueError):
        return []


def _log_text(retrieved: Optional[orm.Node], log_name: Optional[str]) -> Optional[str]:
    if retrieved is None or not log_name:
        return None
    try:
        raw = retrieved.base.repository.get_object_content(log_name, mode='rb')
    except (FileNotFoundError, OSError):
        return None
    return raw.decode('utf-8', errors='replace')


def _provenance_input(node) -> Tuple[Optional[dict], dict]:
    """``(config.json, provenance.json)`` from the ``provenance`` input."""
    folder = node.inputs.provenance if 'provenance' in node.inputs else None
    if folder is None:
        return None, {}

    def load(name):
        try:
            value = json.loads(folder.base.repository.get_object_content(name, mode='r'))
        except (FileNotFoundError, OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    return load('provenance/config.json'), load('provenance.json') or {}


class LammpsFrictionParser(Parser):
    """Turn one deck's retrieved files into point, structure and info nodes."""

    def __init__(self, node):
        super().__init__(node)
        self._cap: Optional[int] = None
        self._not_stored: List[Dict[str, Any]] = []

    def parse(self, **kwargs):  # type: ignore[override]
        """Parse the scratch retrieval and the retrieved log."""
        params = self.node.inputs.parameters.get_dict()
        config, record = _provenance_input(self.node)
        finished, info = self._run_info(params, record)
        self._cap = _cap_bytes(params.get('max_stored_file_mb'))
        self._not_stored = []

        scratch = kwargs.get('retrieved_temporary_folder')
        root = Path(scratch) if scratch else None
        if root is not None:
            _drop_conflict_copies(root)

        failure = None
        try:
            if params.get('writes_structures'):
                produced = self._structures(root)
            else:
                produced = self._points(root, params, config, record)
        except Exception as exc:  # pylint: disable=broad-except
            # Caught, not raised: an exception would discard every output
            # (the log and run_info included) with the excepted process.
            self.logger.exception('Parsing the outputs of %s failed', params.get('deck_name'))
            failure, produced = f'{type(exc).__name__}: {exc}', 0
        if self._not_stored:
            info['not_stored'] = list(self._not_stored)
        self.out('run_info', orm.Dict(info))
        if failure is not None:
            return self.exit_codes.ERROR_PARSING_FAILED.format(error=failure)
        if not produced:
            return self.exit_codes.ERROR_NO_RESULTS
        if finished is False:
            return self.exit_codes.ERROR_LAMMPS_NOT_FINISHED
        return None

    # -- run_info and log ------------------------------------------------------

    def _run_info(self, params: Mapping[str, Any],
                  record: Mapping[str, Any]) -> Tuple[Optional[bool], Dict[str, Any]]:
        """``run_info`` from the log, each field falling back to
        ``provenance.json``. A log without a version line never blanks the
        recorded version: ``parse_log_extras`` only sets keys it found, and
        ``log_has_version`` keeps the gap visible.

        Returns ``(finished, run_info)``: whether LAMMPS finished (a ``Total
        wall time`` line; None without a log) and the ``run_info`` content,
        which :meth:`parse` outputs once the other outputs are known."""
        from ..data.provenance import parse_log_extras  # pylint: disable=import-outside-toplevel

        log_name = params.get('log_name')
        text = _log_text(self.retrieved, log_name)
        info: Dict[str, Any] = {
            'lammps_version': record.get('lammps_version'),
            'wall_time_s': record.get('wall_time_s'),
            'n_procs': record.get('n_procs'),
            'log_missing': text is None,
            'log_has_version': False,
        }
        if text is not None:
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / PurePosixPath(log_name).name
                path.write_text(text, encoding='utf-8')
                extras = parse_log_extras(path)
                info['log_has_version'] = 'lammps_version' in extras
                info.update(extras)
                self.out('log', new_file(path, name=log_name))
            wall = _WALL_RE.findall(text)
            if wall:
                hours, minutes, seconds = (int(x) for x in wall[-1])
                info['wall_time_s'] = float(hours * 3600 + minutes * 60 + seconds)
            procs = _PROCS_RE.findall(text)
            if procs:
                info['n_procs'] = int(procs[-1])
        return (None if text is None else bool(_WALL_RE.search(text))), info

    # -- the stored-file cap ---------------------------------------------------

    def _over_cap(self, name: str, path: Path, point: Optional[str] = None) -> bool:
        """Whether ``path`` exceeds ``max_stored_file_mb``; if so it is recorded
        in ``self._not_stored`` instead of stored: ``name`` (its path inside the
        node it would have gone to), ``size_bytes``, ``sha256`` and, for a
        point's file, the ``point`` label."""
        size = Path(path).stat().st_size
        if self._cap is None or size <= self._cap:
            return False
        from ..data.provenance import sha256_file  # pylint: disable=import-outside-toplevel
        entry = {'name': name, 'size_bytes': size, 'sha256': sha256_file(path)}
        if point is not None:
            entry['point'] = point
        self._not_stored.append(entry)
        logger.info('%s is %d bytes, over max_stored_file_mb: recorded, not stored',
                    name, size)
        return True

    # -- system decks ----------------------------------------------------------

    def _structures(self, root: Optional[Path]) -> int:
        """``load_*.data`` as ``structures``, ``hold_*.txt`` as ``aux``."""
        if root is None or not (root / 'data').is_dir():
            return 0
        outputs: Dict[str, List[Tuple[str, Path]]] = {'structures': [], 'aux': []}
        for path in sorted((root / 'data').iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith('load_') and path.suffix == '.data':
                outputs['structures'].append((_file_label(path.name), path))
            elif path.name.startswith('hold_'):
                outputs['aux'].append((_file_label(path.name), path))
        for namespace, pairs in outputs.items():
            for label, path in _unique(pairs, namespace).items():
                if namespace == 'aux' and self._over_cap(f'data/{path.name}', path):
                    continue
                self.out(f'{namespace}.{label}', new_file(path, name=f'data/{path.name}'))
        return len(outputs['structures'])

    # -- slide-like decks ------------------------------------------------------

    def _points(self, root: Optional[Path], params: Mapping[str, Any],
                config: Optional[dict], record: Mapping[str, Any]) -> int:
        """One ``points`` node per point this deck covers, configured or present,
        plus its ``peratom`` and ``snapshots`` nodes when it has them."""
        # pylint: disable=import-outside-toplevel
        from ..core.observables import expected_tier_rows
        from .block_schema import POINT_UNITS
        from ..postprocessing.post_channels import final_data_path
        from ..postprocessing.run_status import halt_detail, point_key, read_halt_registry
        # `ingest` drives this module (`import_root` -> `import_deck`); looked
        # up at call time so the two do not form an import cycle.
        load_points = import_module(f'{__package__}.ingest').load_points

        family = params['family']
        rel_path = params['rel_path']
        if params.get('n_layers') is None:
            raise ValueError(f'{rel_path}: parameters carry no n_layers')
        n_layers = int(params['n_layers'])
        deck = Deck(name=params['deck_name'], path=Path(params['deck_name']),
                    log_path=None, point_filter=dict(params.get('point_filter') or {}))

        if root is None:
            locator, found, halts = None, [], []
        else:
            locator = _scratch_locator(root, rel_path, family)
            found = points_for(deck, load_points(locator))
            if family in _PES_FAMILIES and deck.name.endswith('_refine'):
                found = _refine_points(found, locator)
            halts = read_halt_registry(locator)
        on_disk = {point_key(p.load_kind, p.load, p.angle, p.speed): p for p in found}

        configured = [key for key in _configured_keys(config, family)
                      if points_for(deck, [SimpleNamespace(load=key[1], angle=key[2],
                                                           speed=key[3])])]
        expected = expected_tier_rows(config, family) if config else {}
        layers_token = _layers_token(rel_path)

        labels = _unique(((point_label(SimpleNamespace(
            load_kind=k[0], load=k[1], angle=k[2], speed=k[3])), k)
            for k in dict.fromkeys(configured + list(on_disk))), 'points')
        for label, key in labels.items():
            point = on_disk.get(key)
            files: Dict[str, Path] = {}
            peratom: Dict[str, Path] = {}
            final = None
            if point is None:
                status, detail = 'no_output', ''
                ident = SimpleNamespace(load_kind=key[0], load=key[1], angle=key[2],
                                        speed=key[3], layers=layers_token)
            else:
                ident = point
                for tier, path in point.tiers.items():
                    target = peratom if tier == 'peratom' else files
                    if tier == 'peratom' or tier in _POINT_TIERS:
                        target[f'results/{path.name}'] = path
                final = final_data_path(locator.data_dir, point.load_kind,
                                        point.load, point.angle)
                status, detail = _status(point.tiers, family, n_layers, expected)
                halted = halt_detail(halts, key)
                if halted is not None:
                    status, detail = 'halted', halted
            kept = {name: path for name, path in peratom.items()
                    if not self._over_cap(name, path, point=label)}
            skipped = [n for n in self._not_stored if n.get('point') == label]
            node = new_folder(files)
            if skipped:
                node.base.attributes.set('not_stored', skipped)
            node.base.attributes.set_many({
                **_point_identity(params.get('run_uuid'), label, ident, record,
                                  POINT_UNITS),
                'family': family, 'load_kind': ident.load_kind,
                'load': float(ident.load), 'angle': float(ident.angle),
                'speed': float(ident.speed), 'layers': ident.layers,
                'status': status, 'status_detail': detail,
                'content_sha256': content_hash(list(files.values())),
                'rel_path': rel_path, 'deck_name': deck.name,
                'has_snapshot': final is not None, 'has_peratom': bool(kept),
            })
            self.out(f'points.{label}', node)
            if kept:
                folder = new_folder(kept)
                if skipped:
                    folder.base.attributes.set('not_stored', skipped)
                self.out(f'peratom.{label}', folder)
            if final is not None:
                self.out(f'snapshots.{label}', new_file(final, name=f'data/{final.name}'))
        return len(labels)


def _cap_bytes(max_mb: Any) -> Optional[int]:
    """``max_stored_file_mb`` in bytes; None (no cap) when unset."""
    if max_mb is None:
        return None
    return int(float(max_mb) * 1024 * 1024)


def _max_stored_file_mb() -> int:
    """``settings.aiida.max_stored_file_mb`` (default 20)."""
    from ..core.config import load_settings  # pylint: disable=import-outside-toplevel
    try:
        return int(load_settings().aiida.max_stored_file_mb)
    except (OSError, ValueError, AttributeError):
        return 20


def _point_identity(run_uuid: Optional[str], label: str, ident: Any,
                    record: Mapping[str, Any],
                    point_units: Mapping[str, Mapping[str, str]]) -> Dict[str, Any]:
    """``point_uuid``, ``units`` and the per-load ``load_achieved`` /
    ``ramp_converged`` from ``provenance.json`` (None when not recorded)."""
    from ..data.provenance import load_key  # pylint: disable=import-outside-toplevel

    key = load_key(ident.load)
    achieved = (record.get('load_achieved_by_load') or {}).get(key)
    ramp = (record.get('ramp_by_load') or {}).get(key)
    return {
        'point_uuid': point_uuid(run_uuid, label) if run_uuid else None,
        'units': dict(point_units[ident.load_kind]),
        'load_achieved': None if achieved is None else float(achieved),
        'ramp_converged': ramp.get('converged') if isinstance(ramp, Mapping) else None,
    }


def _layers_token(rel_path: str) -> Optional[int]:
    from ..run.naming import parse_run_path  # pylint: disable=import-outside-toplevel
    return parse_run_path(rel_path).layers


def _refine_points(found, locator):
    """A PES ``*_refine`` deck's point: its own refine grid, not the base grid."""
    from dataclasses import replace  # pylint: disable=import-outside-toplevel

    refine = locator.results_dir / _PES_REFINE_NAME
    if not found or not refine.is_file():
        return []
    return [replace(found[0], tiers={'window': refine},
                    content_sha256=content_hash([refine]))]


# ---------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    import yaml  # pylint: disable=import-outside-toplevel
    try:
        value = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _files(directory: Path, pattern: str) -> List[Path]:
    if not directory.is_dir():
        return []
    return [p for p in sorted(directory.glob(pattern))
            if p.is_file() and _CONFLICT not in p.name]


def _build_dir(locator) -> Path:
    """``build/`` of the run, or of the level above ``L{n}`` (AFM)."""
    here = Path(locator.run_dir) / 'build'
    return here if here.is_dir() else Path(locator.run_dir).parent / 'build'


def _sim_root(locator) -> Path:
    """The simulation root the run's ``rel_path`` is relative to."""
    run_dir = Path(locator.run_dir)
    depth = len(PurePosixPath(locator.tokens.rel_path).parts)
    return run_dir.parents[depth - 1] if depth else run_dir


def _run_uuid(locator, record: Mapping[str, Any]) -> str:
    """``record['run_uuid']``, or :class:`MissingRunUUID` naming the file when
    it is absent or not a UUID."""
    run_uuid = record.get('run_uuid')
    if not run_uuid:
        raise MissingRunUUID(
            f'{locator.provenance_file} has no run_uuid (a run built before '
            f'provenance schema 2.2.0). Stamp the legacy runs with '
            f'`FrictionSim2D aiida stamp-uuid {_sim_root(locator)}`, or rebuild.')
    try:
        UUID(str(run_uuid))
    except ValueError as exc:
        raise MissingRunUUID(
            f'{locator.provenance_file}: run_uuid {run_uuid!r} is not a UUID. '
            f'Fix or remove it, then stamp with `FrictionSim2D aiida stamp-uuid '
            f'{_sim_root(locator)}`, or rebuild.') from exc
    return str(run_uuid)


def read_run_uuid(locator) -> str:
    """The run's ``provenance.json`` ``run_uuid``.

    Raises:
        MissingRunUUID: it is absent or not a UUID.
    """
    return _run_uuid(locator, _read_json(locator.provenance_file))


def included_files(locator, deck: Deck) -> List[Path]:
    """The files the deck ``include``\\ s that exist on disk.

    Deck paths are simulation-root relative (``<rel_path>/lammps/
    system.in.settings``); a path relative to the run, or the bare name in
    ``lammps/``, is accepted too. A ``$``-variable path is skipped.
    """
    run_dir = Path(locator.run_dir)
    sim_root = _sim_root(locator)
    try:
        text = Path(deck.path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    found: List[Path] = []
    for ref in _INCLUDE_RE.findall(text):
        if '$' in ref:
            continue
        for candidate in (sim_root / ref, run_dir / ref,
                          Path(locator.lammps_dir) / PurePosixPath(ref).name):
            if candidate.is_file():
                if candidate not in found:
                    found.append(candidate)
                break
    return found


@dataclass(frozen=True)
class RunInputs:
    """What every deck of one run shares, computed once per run (:func:`run_inputs`).

    Attributes:
        blocks: Block name (``material__sheet_0``) -> stored block ``Dict``.
        derived: The run's derived values (``drive_k``, ``geometry.lat_c``,
            ...) under ``'fields'`` (a top-level attribute key cannot hold a
            ``.``), never hashed. Unstored until the first deck uses it; then
            that same stored node is every deck's input.
        schema: The stored schema document (units, kinds, defaults) of the family.
        provenance: The stored ``provenance.json`` + ``provenance/`` folder.
        potentials: Link label -> stored potential file.
        run_uuid: ``provenance.json`` ``run_uuid``.
        config, record, settings: ``provenance/config.json``,
            ``provenance.json``, ``provenance/settings.yaml``.
        max_stored_file_mb: ``settings.aiida.max_stored_file_mb``: an aux or
            per-atom file over it is recorded (name, sha256, size) in
            ``run_info['not_stored']`` -- and, for a point's per-atom file, on
            the point node (and its ``peratom`` node, if any file was kept) --
            not stored.
    """
    blocks: Dict[str, orm.Dict]
    derived: orm.Dict
    schema: orm.Dict
    provenance: orm.FolderData
    potentials: Dict[str, orm.SinglefileData]
    run_uuid: str
    config: dict
    record: dict
    settings: dict
    max_stored_file_mb: int = 20


def _split_run(locator) -> Tuple[Any, dict, dict, dict, str]:
    """``(blocks, config, record, settings, run_uuid)`` of one run, with
    ``derived.drive_k`` checked against its unit; stores nothing."""
    # pylint: disable=import-outside-toplevel
    from ..postprocessing.read_data import resolve_n_layers
    from ..postprocessing.run_status import read_run_config
    from . import blocks as _blocks
    from .block_schema import check_units

    prov_dir = Path(locator.provenance_dir)
    config = read_run_config(locator) or {}
    record = _read_json(locator.provenance_file)
    run_uuid = _run_uuid(locator, record)
    settings = _read_yaml(prov_dir / 'settings.yaml')
    split = _blocks.split_blocks(config, settings, record, family=locator.tokens.family,
                                 n_layers=resolve_n_layers(locator, config or None),
                                 provenance_dir=prov_dir)
    check_units(split, record=record)
    return split, config, record, settings, run_uuid


def check_run(locator) -> str:
    """What :func:`run_inputs` checks, storing nothing: the run's ``run_uuid``.

    Raises:
        MissingRunUUID: ``provenance.json`` has no ``run_uuid``.
        UnitMismatch: ``derived.drive_k`` is not ``driving_spring`` in eV/A^2.
    """
    return _split_run(locator)[4]


def run_inputs(locator, *, index: Optional[StoreIndex] = None) -> RunInputs:
    """The inputs every deck of the run at ``locator`` shares.

    Splits the run into blocks once, checks ``derived.drive_k`` against its
    unit with the run's own record (a legacy AFM run used ``/ 16.02``), and
    stores the blocks, the schema node, the provenance folder and the
    potentials through ``index`` (loaded here when absent).

    Raises:
        MissingRunUUID: ``provenance.json`` has no ``run_uuid``.
        UnitMismatch: ``derived.drive_k`` is not ``driving_spring`` in eV/A^2.
    """
    # pylint: disable=import-outside-toplevel
    from .block_schema import schema_document
    from .store import get_or_create_block, get_or_create_provenance, get_or_create_schema

    index = index if index is not None else StoreIndex.load()
    prov_dir = Path(locator.provenance_dir)
    split, config, record, settings, run_uuid = _split_run(locator)
    return RunInputs(
        blocks={name: get_or_create_block(content, index=index)
                for name, content in split.blocks.items()},
        derived=orm.Dict({'fields': dict(split.derived)}),
        schema=get_or_create_schema(schema_document(split.family), index=index),
        provenance=get_or_create_provenance(locator.provenance_file, prov_dir,
                                            index=index),
        potentials=_unique(((_file_label(p.name), get_or_create_file(p, index=index))
                            for p in _files(prov_dir / 'potentials', '*')),
                           'potentials'),
        run_uuid=run_uuid, config=config, record=record, settings=settings,
        max_stored_file_mb=_max_stored_file_mb())


_LMP_TOKEN_RE = re.compile(r'(?:^|\s)(\S*/)?(lmp\S*)(?=\s|$)')
_CD_RE = re.compile(r'^\s*cd\s+(/[^\s"\']+)\s*$', re.M)


def _job_scripts(locator) -> List[Path]:
    hpc_dir = _sim_root(locator) / 'hpc'
    return sorted(p for ext in ('*.pbs', '*.slurm', '*.sh') for p in hpc_dir.glob(ext)
                  if p.is_file() and not p.name.startswith('preflight'))


def _executable_from_job_script(locator) -> Optional[str]:
    """The first ``lmp*`` token of an ``mpirun``/``srun``/``lmp`` line of the
    run's job script (``<simulation root>/hpc/*.pbs|*.slurm|*.sh``), else None."""
    for script in _job_scripts(locator):
        try:
            text = script.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        for line in text.splitlines():
            if line.lstrip().startswith('#'):
                continue
            if re.match(r'\s*(\S*mpirun|\S*mpiexec|srun|lmp\S*)\b', line):
                match = _LMP_TOKEN_RE.search(line)
                if match:
                    return (match.group(1) or '') + match.group(2)
    return None


def _hpc_run_path(locator) -> Optional[str]:
    """The HPC-side run path when a job script ``cd``s to an absolute directory
    (the generated scripts do not), else None."""
    for script in _job_scripts(locator):
        try:
            text = script.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        match = _CD_RE.search(text)
        if match:
            return (PurePosixPath(match.group(1)) / locator.tokens.rel_path).as_posix()
    return None


def _log_info(log_path: Optional[Path]) -> Dict[str, Any]:
    """``lammps_version`` and ``n_procs`` of a deck's log (keys only when found)."""
    from ..data.provenance import parse_log_extras  # pylint: disable=import-outside-toplevel

    if log_path is None:
        return {}
    info: Dict[str, Any] = {}
    version = parse_log_extras(Path(log_path)).get('lammps_version')
    if version:
        info['lammps_version'] = version
    procs = _PROCS_RE.findall(Path(log_path).read_text(encoding='utf-8', errors='replace'))
    if procs:
        info['n_procs'] = int(procs[-1])
    return info


def _hpc(run: RunInputs) -> Mapping[str, Any]:
    hpc = (run.settings or {}).get('hpc')
    return hpc if isinstance(hpc, Mapping) else {}


def _options(hpc: Mapping[str, Any], n_procs: int) -> Dict[str, Any]:
    """Real ``metadata.options``: resources from the log's process count."""
    nodes = hpc.get('num_nodes')
    machines = nodes if isinstance(nodes, int) and nodes > 0 and n_procs % nodes == 0 else 1
    options: Dict[str, Any] = {
        'resources': {'num_machines': machines,
                      'num_mpiprocs_per_machine': math.ceil(n_procs / machines)},
        'withmpi': n_procs > 1}
    walltime = hpc.get('walltime_hours')
    if walltime:
        options['max_wallclock_seconds'] = int(walltime * 3600)
    return options


def _deck_code(locator, computer: orm.Computer,
               version: Optional[str]) -> Optional[orm.InstalledCode]:
    """The LAMMPS Code of (``computer``, ``version``); None without a version,
    so no "unknown" Code ever merges unrelated binaries."""
    if not version:
        return None
    return lammps_code(computer, version, _executable_from_job_script(locator))


def deck_inputs(locator, deck: Deck, *, computer: orm.Computer,
                run: Optional[RunInputs] = None,
                structures: Optional[Mapping[str, orm.SinglefileData]] = None,
                decks: Optional[Sequence[Deck]] = None,
                code: Optional[orm.AbstractCode] = None,
                index: Optional[StoreIndex] = None) -> Dict[str, Any]:
    """The inputs of one deck's import-only CalcJob.

    ``run`` is the run's shared inputs (:func:`run_inputs`, computed here when
    absent, through ``index``). ``structures`` are the system deck's
    ``structures`` outputs, handed to a slide deck. The ``build/*.lmp`` files
    go to the system deck, or to every deck of a run that has none.
    ``decks`` is the run's deck list, when the caller already has it.

    Block link labels are the block names with ``__`` -> ``_`` (``__`` is
    AiiDA's namespace separator); ``parameters['block_names']`` maps each
    label back to the unambiguous name [C2].

    Raises:
        MissingRunUUID: ``provenance.json`` has no ``run_uuid``.
    """
    from ..postprocessing.read_data import resolve_n_layers  # pylint: disable=import-outside-toplevel

    if run is None:
        index = index if index is not None else StoreIndex.load()
        run = run_inputs(locator, index=index)
    run_dir = Path(locator.run_dir)
    n_layers = resolve_n_layers(locator, run.config or None)
    decks = list(decks) if decks is not None else discover_decks(locator)

    reads_build = deck.writes_structures or not any(d.writes_structures for d in decks)
    builds = _files(_build_dir(locator), '*.lmp') if reads_build else []
    handed = _unique([(f'build_{_file_label(p.name)}', get_or_create_file(p, index=index))
                      for p in builds]
                     + list((structures or {}).items()), 'structures')
    labels = _unique(((name.replace('__', '_'), name) for name in run.blocks), 'blocks')

    log_name = (Path(deck.log_path).relative_to(run_dir).as_posix()
                if deck.log_path is not None else None)
    log_info = _log_info(deck.log_path)
    n_procs = log_info.get('n_procs') or run.record.get('n_procs') or 1
    hpc = _hpc(run)
    if code is None:
        code = _deck_code(locator, computer, log_info.get('lammps_version')
                          or run.record.get('lammps_version'))
    inputs: Dict[str, Any] = {
        'remote_folder': orm.RemoteData(computer=computer,
                                        remote_path=str(run_dir.resolve())),
        'deck': get_or_create_file(deck.path, index=index),
        'provenance': run.provenance,
        'blocks': {label: run.blocks[name] for label, name in labels.items()},
        'derived': run.derived,
        'schema': run.schema,
        'potentials': dict(run.potentials),
        'parameters': orm.Dict({
            'deck_name': deck.name, 'rel_path': locator.tokens.rel_path,
            'run_uuid': run.run_uuid,
            'point_filter': dict(deck.point_filter),
            'family': locator.tokens.family,
            'n_layers': n_layers,
            'log_name': log_name, 'writes_structures': deck.writes_structures,
            'block_names': labels,
            'hpc_run_path': _hpc_run_path(locator),
            'max_stored_file_mb': run.max_stored_file_mb,
            # Where the run's build/ sits: the integrity step looks the bond
            # reference up where the DataReader would (`bond_reference`).
            'build_level': 'run' if (run_dir / 'build').is_dir() else 'parent',
        }),
        'metadata': {'computer': computer,
                     'label': f'{locator.tokens.rel_path}:{deck.name}',
                     'options': _options(hpc, int(n_procs))},
    }
    includes = _unique(((_file_label(p.name), get_or_create_file(p, index=index))
                        for p in included_files(locator, deck)), 'includes')
    if includes:
        inputs['includes'] = includes
    if handed:
        inputs['structures'] = handed
    if code is not None:
        inputs['code'] = code
    return inputs


def import_deck(locator, deck: Deck, *, computer: orm.Computer,
                run: Optional[RunInputs] = None,
                structures: Optional[Mapping[str, orm.SinglefileData]] = None,
                decks: Optional[Sequence[Deck]] = None,
                code: Optional[orm.AbstractCode] = None,
                index: Optional[StoreIndex] = None):
    """Run one deck's import-only CalcJob in-process: ``(outputs, node)``."""
    from aiida.engine import run_get_node  # pylint: disable=import-outside-toplevel

    return run_get_node(LammpsFrictionCalcJob,
                        **deck_inputs(locator, deck, computer=computer, run=run,
                                      structures=structures, decks=decks,
                                      code=code, index=index))


def structures_for(deck: Deck, structures: Mapping[str, orm.SinglefileData]
                   ) -> Dict[str, orm.SinglefileData]:
    """The handed-off ``load_<F>N.data`` nodes this deck reads: those whose
    load matches its ``point_filter['load']``, or all without a load filter."""
    load = deck.point_filter.get('load')
    if load is None:
        return dict(structures)
    out = {}
    for key, node in structures.items():
        match = _LOAD_FILE_RE.match(node.filename)
        if match and abs(float(match['load']) - float(load)) <= _TOL:
            out[key] = node
    return out


def import_run(locator, *, computer: orm.Computer,
               index: Optional[StoreIndex] = None,
               code_for: Optional[Callable[[Deck], Optional[orm.AbstractCode]]] = None,
               ) -> List[orm.CalcJobNode]:
    """Every deck of one run, system decks first; their ``structures``
    outputs become the following decks' ``structures`` inputs.

    The run's blocks, schema, provenance and potentials are computed once
    (:func:`run_inputs`); ``index`` is the import's lookup memo (loaded here
    when absent -- pass one to share it across runs). ``code_for(deck)``
    gives a deck's ``code`` input, if any.
    """
    index = index if index is not None else StoreIndex.load()
    run = run_inputs(locator, index=index)
    nodes: List[orm.CalcJobNode] = []
    handed: Dict[str, orm.SinglefileData] = {}
    decks = discover_decks(locator)
    for deck in decks:
        given = None if deck.writes_structures else structures_for(deck, handed)
        _outputs, node = import_deck(locator, deck, computer=computer, run=run,
                                     structures=given, decks=decks,
                                     code=code_for(deck) if code_for else None,
                                     index=index)
        if deck.writes_structures and 'structures' in node.outputs:
            handed.update(dict(node.outputs.structures.items()))
        nodes.append(node)
    return nodes
