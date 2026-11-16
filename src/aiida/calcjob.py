"""Import-only CalcJob and Parser for finished FrictionSim2D runs (spec §5.1).

One :class:`LammpsFrictionCalcJob` per LAMMPS deck of a run. It is launched
with ``engine.run`` and a ``remote_folder`` pointing at the run directory, so
aiida-core skips upload and submit and only retrieves and parses: LAMMPS is
never launched through AiiDA. The import path hard-codes ``LocalTransport``
(``CalcJob._perform_import``), so the run directory must be readable on this
machine; the ``RemoteData``'s Computer (a ``core.local`` one, see
:func:`src.aiida.store.machine_computer`) only records where it came from.

What each deck's node carries:

* inputs -- the deck itself, the run's ``provenance/`` + ``provenance.json``,
  its parameter blocks and potential files (all content-addressed through
  :mod:`src.aiida.store`, so runs share them), and ``structures``: the
  ``build/*.lmp`` files the first deck reads and, for a slide deck, the very
  ``load_*.data`` nodes the system deck output. That last link is the
  system -> slide handoff in the graph.
* outputs -- ``points`` (one ``FolderData`` per configured or present load
  point), ``structures`` and ``aux`` (system decks), ``run_info`` and ``log``.

Retrieval goes through ``retrieve_temporary_list`` and the parser builds the
stored nodes itself, so it can drop Syncthing ``*.sync-conflict-*`` copies
and always keep the original file -- a plain ``retrieve_list`` glob cannot.
"""
from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from aiida import orm
from aiida.common.datastructures import CalcInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser

from .data_keys import TIER_KEYS, content_hash
from .decks import Deck, discover_decks, points_for
from .retention import retrieve_patterns

logger = logging.getLogger(__name__)

__all__ = ['LammpsFrictionCalcJob', 'LammpsFrictionParser', 'deck_inputs',
           'import_deck', 'import_run', 'point_label']

_CONFLICT = '.sync-conflict-'
_PES_FAMILIES = ('pes_tip', 'pes_sheet')
_PES_REFINE_NAME = 'pes_scan_refine.csv'

#: What each kind of deck retrieves, as remote globs of
#: :func:`~src.aiida.retention.retrieve_patterns`: a system deck writes the
#: loaded structures (and the AFM hold traces), a slide-like deck the results
#: and end snapshots.
_SYSTEM_RETRIEVE = ('data/load_*.data', 'data/hold_*.txt')
_SLIDE_RETRIEVE = ('results/*', 'data/final_*.data')

_WALL_RE = re.compile(r'^Total wall time:\s*(\d+):(\d+):(\d+)', re.M)
_PROCS_RE = re.compile(r'^Loop time of \S+ on (\d+) procs', re.M)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _num(value: float) -> str:
    """``f'{value:g}'`` with ``-`` -> ``m`` and ``.`` -> ``p``."""
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def point_label(point: Any) -> str:
    """The ``points`` key of one load point, e.g. ``force10_a0_s2``.

    The load-condition half of :func:`src.aiida.ingest._point_key`, spelled
    as a valid link label (alphanumerics and single underscores).
    """
    return (f'{point.load_kind}{_num(point.load)}'
            f'_a{_num(point.angle)}_s{_num(point.speed)}')


def _file_label(name: str) -> str:
    """A link-label-safe key for a file: its stem, ``.``/``-`` respelled."""
    stem = PurePosixPath(name).stem.replace('-', 'm').replace('.', 'p')
    label = re.sub(r'_+', '_', re.sub(r'\W', '_', stem)).strip('_') or 'file'
    return label if label[0].isalpha() else f'f_{label}'


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
        spec.input('provenance', valid_type=orm.FolderData, required=False,
                   help="The run's provenance/ directory plus provenance.json.")
        spec.input_namespace('blocks', valid_type=orm.Dict, dynamic=True,
                             required=False, help='Parameter blocks (spec §5.2).')
        spec.input_namespace('potentials', valid_type=orm.SinglefileData,
                             dynamic=True, required=False,
                             help='Potential files the run used.')
        spec.input_namespace('structures', valid_type=orm.SinglefileData,
                             dynamic=True, required=False,
                             help='Structure files the deck reads: build/*.lmp, '
                                  "and the system deck's load_*.data outputs.")
        spec.input('parameters', valid_type=orm.Dict,
                   help='deck_name, rel_path, point_filter, family, n_layers, '
                        'log_name, writes_structures.')

        spec.input('metadata.options.resources', valid_type=dict,
                   default={'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        spec.input('metadata.options.parser_name', valid_type=str,
                   default='friction2d.lammps')

        spec.output_namespace('points', valid_type=orm.FolderData, dynamic=True,
                              required=False,
                              help='One FolderData per load point.')
        spec.output_namespace('structures', valid_type=orm.SinglefileData,
                              dynamic=True, required=False,
                              help='load_*.data written by a system deck.')
        spec.output_namespace('aux', valid_type=orm.SinglefileData,
                              dynamic=True, required=False,
                              help='Other retained files of a system deck '
                                   '(hold_*.txt).')
        spec.output('run_info', valid_type=orm.Dict, required=False,
                    help='lammps_version, wall_time_s, n_procs, log_missing.')
        spec.output('log', valid_type=orm.SinglefileData, required=False,
                    help="The deck's LAMMPS log.")

        spec.exit_code(320, 'ERROR_NO_RESULTS',
                       message='The deck produced neither load points nor structures.')

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Only the retrieve lists: nothing is uploaded and nothing runs."""
        params = self.inputs.parameters.get_dict()  # type: ignore[attr-defined]
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
        if tier not in TIER_KEYS:
            continue
        try:
            n_rows[tier] = len(_read_frame(path, family, n_layers))
        except (ValueError, OSError, pd.errors.EmptyDataError) as exc:
            failures.append(f'{tier} unreadable: {exc}')
            if tier == 'window':
                window_error = str(exc)
    if window_error is not None or not n_rows:
        return 'unreadable', window_error or '; '.join(failures)
    status, detail = classify(n_rows, dict(expected))
    return status, '; '.join(x for x in (*failures, detail) if x)


def _folder(files: Mapping[str, Path]) -> orm.FolderData:
    node = orm.FolderData()
    for name, path in files.items():
        node.base.repository.put_object_from_file(str(path), name)
    return node


def _log_text(retrieved: Optional[orm.Node], log_name: Optional[str]) -> Optional[str]:
    if retrieved is None or not log_name:
        return None
    try:
        return retrieved.base.repository.get_object_content(log_name, mode='r')
    except (FileNotFoundError, OSError):
        return None


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

    def parse(self, **kwargs):  # type: ignore[override]
        """Parse the scratch retrieval and the retrieved log."""
        params = self.node.inputs.parameters.get_dict()
        config, record = _provenance_input(self.node)
        self._run_info(params, record)

        scratch = kwargs.get('retrieved_temporary_folder')
        root = Path(scratch) if scratch else None
        if root is not None:
            _drop_conflict_copies(root)

        if params.get('writes_structures'):
            produced = self._structures(root)
        else:
            produced = self._points(root, params, config)
        if not produced:
            return self.exit_codes.ERROR_NO_RESULTS
        return None

    # -- run_info and log ------------------------------------------------------

    def _run_info(self, params: Mapping[str, Any], record: Mapping[str, Any]) -> None:
        from ..data.provenance import parse_log_extras  # pylint: disable=import-outside-toplevel

        log_name = params.get('log_name')
        text = _log_text(self.retrieved, log_name)
        info: Dict[str, Any] = {
            'lammps_version': record.get('lammps_version'),
            'wall_time_s': record.get('wall_time_s'),
            'n_procs': record.get('n_procs'),
            'log_missing': text is None,
        }
        if text is not None:
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / 'log.lammps'
                path.write_text(text, encoding='utf-8')
                info.update(parse_log_extras(path))
            wall = _WALL_RE.findall(text)
            if wall:
                hours, minutes, seconds = (int(x) for x in wall[-1])
                info['wall_time_s'] = float(hours * 3600 + minutes * 60 + seconds)
            procs = _PROCS_RE.findall(text)
            if procs:
                info['n_procs'] = int(procs[-1])
            self.out('log', orm.SinglefileData(
                io.BytesIO(text.encode('utf-8')),
                filename=PurePosixPath(log_name).name))
        self.out('run_info', orm.Dict(info))

    # -- system decks ----------------------------------------------------------

    def _structures(self, root: Optional[Path]) -> int:
        """``load_*.data`` as ``structures``, ``hold_*.txt`` as ``aux``."""
        if root is None or not (root / 'data').is_dir():
            return 0
        count = 0
        for path in sorted((root / 'data').iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith('load_') and path.suffix == '.data':
                namespace = 'structures'
                count += 1
            elif path.name.startswith('hold_'):
                namespace = 'aux'
            else:
                continue
            self.out(f'{namespace}.{_file_label(path.name)}',
                     orm.SinglefileData(str(path)))
        return count

    # -- slide-like decks ------------------------------------------------------

    def _points(self, root: Optional[Path], params: Mapping[str, Any],
                config: Optional[dict]) -> int:
        """One ``FolderData`` per point this deck covers, configured or present."""
        # pylint: disable=import-outside-toplevel
        from ..core.observables import expected_tier_rows
        from ..postprocessing.post_channels import final_data_path
        from ..postprocessing.run_status import (expected_chains, halt_detail,
                                                 point_key, read_halt_registry)
        from .ingest import LoadPoint, load_points

        family = params['family']
        rel_path = params['rel_path']
        n_layers = int(params.get('n_layers') or 2)
        deck = Deck(name=params['deck_name'], path=Path(params['deck_name']),
                    log_path=None, point_filter=dict(params.get('point_filter') or {}))
        locator = _scratch_locator(root or Path('/nonexistent'), rel_path, family)

        found: List[LoadPoint] = points_for(deck, load_points(locator)) if root else []
        if family in _PES_FAMILIES and deck.name.endswith('_refine'):
            found = _refine_points(found, locator)
        on_disk = {point_key(p.load_kind, p.load, p.angle, p.speed): p for p in found}

        configured = [key for chain in (expected_chains(config, family) or [])
                      for key in chain]
        configured = [key for key in configured
                      if points_for(deck, [SimpleNamespace(load=key[1], angle=key[2],
                                                           speed=key[3])])]
        expected = expected_tier_rows(config, family) if config else {}
        halts = read_halt_registry(locator) if root else []

        count = 0
        for key in dict.fromkeys(configured + list(on_disk)):
            point = on_disk.get(key)
            if point is None:
                files: Dict[str, Path] = {}
                status, detail = 'no_output', ''
                ident = SimpleNamespace(load_kind=key[0], load=key[1], angle=key[2],
                                        speed=key[3], layers=locator.tokens.layers)
            else:
                ident = point
                files = {f'results/{p.name}': p for p in point.tiers.values()}
                final = final_data_path(locator.data_dir, point.load_kind,
                                        point.load, point.angle)
                if final is not None:
                    files[f'data/{final.name}'] = final
                status, detail = _status(point.tiers, family, n_layers, expected)
                halted = halt_detail(halts, key)
                if halted is not None:
                    status, detail = 'halted', halted
            node = _folder(files)
            node.base.attributes.set_many({
                'family': family, 'load_kind': ident.load_kind,
                'load': float(ident.load), 'angle': float(ident.angle),
                'speed': float(ident.speed), 'layers': ident.layers,
                'status': status, 'status_detail': detail,
                'content_sha256': content_hash(list(files.values())),
                'rel_path': rel_path, 'deck_name': deck.name,
            })
            self.out(f'points.{point_label(ident)}', node)
            count += 1
        return count


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


def deck_inputs(locator, deck: Deck, *, computer: orm.Computer,
                structures: Optional[Mapping[str, orm.SinglefileData]] = None
                ) -> Dict[str, Any]:
    """The inputs of one deck's import-only CalcJob.

    ``structures`` are the system deck's ``structures`` outputs, handed to a
    slide deck. The ``build/*.lmp`` files go to the system deck, or to every
    deck of a run that has none.
    """
    # pylint: disable=import-outside-toplevel
    from ..postprocessing.read_data import resolve_n_layers
    from ..postprocessing.run_status import read_run_config
    from .blocks import split_blocks
    from .store import get_or_create_block, get_or_create_file, get_or_create_provenance

    run_dir = Path(locator.run_dir)
    prov_dir = Path(locator.provenance_dir)
    config = read_run_config(locator) or {}
    record = _read_json(locator.provenance_file)
    settings = _read_yaml(prov_dir / 'settings.yaml')
    blocks = split_blocks(config, settings, record, provenance_dir=prov_dir)

    reads_build = deck.writes_structures or not any(
        d.writes_structures for d in discover_decks(locator))
    handed = {f'build_{_file_label(p.name)}': get_or_create_file(p)
              for p in (_files(_build_dir(locator), '*.lmp') if reads_build else [])}
    handed.update(structures or {})

    log_name = (Path(deck.log_path).relative_to(run_dir).as_posix()
                if deck.log_path is not None else None)
    inputs: Dict[str, Any] = {
        'remote_folder': orm.RemoteData(computer=computer,
                                        remote_path=str(run_dir.resolve())),
        'deck': get_or_create_file(deck.path),
        'provenance': get_or_create_provenance(locator.provenance_file, prov_dir),
        'blocks': {name.replace('__', '_'): get_or_create_block(name, content)
                   for name, content in blocks.items()},
        'potentials': {_file_label(p.name): get_or_create_file(p)
                       for p in _files(prov_dir / 'potentials', '*')},
        'parameters': orm.Dict({
            'deck_name': deck.name, 'rel_path': locator.tokens.rel_path,
            'point_filter': dict(deck.point_filter),
            'family': locator.tokens.family,
            'n_layers': resolve_n_layers(locator, config or None),
            'log_name': log_name, 'writes_structures': deck.writes_structures,
        }),
        'metadata': {'computer': computer,
                     'label': f'{locator.tokens.rel_path}:{deck.name}'},
    }
    if handed:
        inputs['structures'] = handed
    return inputs


def import_deck(locator, deck: Deck, *, computer: orm.Computer,
                structures: Optional[Mapping[str, orm.SinglefileData]] = None):
    """Run one deck's import-only CalcJob in-process: ``(outputs, node)``."""
    from aiida.engine import run_get_node  # pylint: disable=import-outside-toplevel

    return run_get_node(LammpsFrictionCalcJob,
                        **deck_inputs(locator, deck, computer=computer,
                                      structures=structures))


def import_run(locator, *, computer: orm.Computer) -> List[orm.CalcJobNode]:
    """Every deck of one run, system decks first; their ``structures``
    outputs become the following decks' ``structures`` inputs."""
    nodes: List[orm.CalcJobNode] = []
    handed: Dict[str, orm.SinglefileData] = {}
    for deck in discover_decks(locator):
        _outputs, node = import_deck(locator, deck, computer=computer,
                                     structures=None if deck.writes_structures
                                     else handed)
        if deck.writes_structures and 'structures' in node.outputs:
            handed.update(dict(node.outputs.structures.items()))
        nodes.append(node)
    return nodes
