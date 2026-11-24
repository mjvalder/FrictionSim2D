"""Getting data back out of the store: trees, rebuilds and archives.

* :func:`export_results` writes a set's CalcJob graph back into an ordinary
  simulation tree, so ``postprocess read`` gives the statuses it gave on the source.
* :func:`rebuild_set` hands each run's stored ``provenance/config.json`` to
  ``run.artifacts.generate_package``, the router ``FrictionSim2D run`` uses.
  ``generate_package`` is imported at module level so tests can patch it.
* :func:`export_archive` writes one campaign as an ``.aiida`` archive, gated on every
  live point having a verdict and on the closure staying inside the campaign, with
  per-atom data opt-in, HPC details anonymised by default, JSON sidecars
  ``<archive>.params/`` (:func:`frictionsim2d.aiida.compare.load_blocks_from_sidecar`),
  a README, and the stored analysis results of the set's datasets.
* :func:`export_dataset` writes one dataset as its own archive.

Every aiida import is deferred, so importing this module needs no aiida-core.

Anonymisation (what is scrubbed: :data:`HPC_SCRUBBED_KEYS` and the README) is a rewrite of
the written archive, not of the store, because stored nodes are immutable. A rewritten
provenance folder keeps its UUID, so a re-import into the source profile still
deduplicates onto the full local node (a new UUID would collide with its link labels);
its ``friction2d_folder_sha256`` is recomputed and it gains ``friction2d_anonymised``,
so ``store.verify`` agrees on the far side. Dropping the ``peratom.<key>`` nodes is the
same rewrite: the export rules always follow a CalcJob's CREATE links, so they cannot be
left out of the traversal.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import shutil
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Type,
    get_args, get_origin,
)

from pydantic import BaseModel

from ..core.config import AFMSimulationConfig, SheetOnSheetSimulationConfig
from ..run.artifacts import generate_package
from ..run.naming import AFM_LIKE_MODELS
from ..run.tree import provenance_dir_for
from .layout import ANALYSIS_GROUP_PREFIX, DATASET_PREFIX, is_superseded

logger = logging.getLogger(__name__)

__all__ = ['ArchiveFormatError', 'ExportClosureError', 'UncheckedPointsError', 'export_archive',
           'export_dataset', 'export_provenance_bundle', 'export_results',
           'import_archive', 'rebuild_set', 'resolve_set', 'sidecar_documents']

#: The extra a rewritten (anonymised) node carries in the archive.
ANONYMISED_EXTRA = 'friction2d_anonymised'
#: ``settings.hpc`` keys dropped on export: paths on the HPC machine, and the account.
HPC_SCRUBBED_KEYS = ('hpc_home', 'scratch_dir', 'log_dir', 'conda_env', 'account')
#: Keys dropped anywhere under a ``database`` mapping on export.
DATABASE_SCRUBBED_KEYS = ('user', 'host')

_CHUNK = 500
_HALT_DETAIL_RE = re.compile(r'^(?P<stage>\S+) halted at T=(?P<T>\S+) step=(?P<step>\S+)$')


class UncheckedPointsError(ValueError):
    """Live points without a live verdict: the archive would publish them unjudged.

    Attributes:
        points: ``'<rel_path>:<deck> <point label> (<point_uuid>)'`` per point.
    """

    def __init__(self, what: str, points: List[str]) -> None:
        self.points = points
        super().__init__(
            f'{what}: {len(points)} live point(s) have no verdict (import the set '
            f'again, or pass force=True):\n  ' + '\n  '.join(points))


class ExportClosureError(ValueError):
    """The export closure reaches CalcJobs outside the exported set.

    Attributes:
        calcjobs: ``'<uuid> <label>'`` of each foreign CalcJob.
    """

    def __init__(self, label: str, calcjobs: List[str]) -> None:
        self.calcjobs = calcjobs
        super().__init__(
            f'set {label!r}: the archive would carry {len(calcjobs)} CalcJob(s) of '
            f'other sets:\n  ' + '\n  '.join(calcjobs))


def _load_profile(profile: Optional[str]) -> None:
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel

    load_aiida_profile(profile)


def resolve_set(label: str):
    """The Group ``friction2d/<label>`` (``LookupError`` when there is none)."""
    from .store import resolve_set as _resolve  # pylint: disable=import-outside-toplevel
    return _resolve(label)


def set_calcjobs(group) -> List[Any]:
    """Every CalcJob of ``group``, in pk order."""
    from .store import set_calcjobs as _calcjobs  # pylint: disable=import-outside-toplevel
    return _calcjobs(group)


def _live_calcjobs(group) -> List[Any]:
    return [calc for calc in set_calcjobs(group) if not is_superseded(calc)]


def _live_points(calcs: Iterable[Any]) -> List[Tuple[Any, str, Any]]:
    """``(calc, point label, point)`` of every live point of ``calcs``."""
    out = []
    for calc in calcs:
        if 'points' in calc.outputs:
            out += [(calc, key, point) for key, point in sorted(calc.outputs.points.items())
                    if not is_superseded(point)]
    return out


# ---------------------------------------------------------------------------
# export_results
# ---------------------------------------------------------------------------

@dataclass
class _Tree:
    """One :func:`export_results` call: where files go, what is written."""
    dest: Path
    written: List[Path] = field(default_factory=list)
    seen: Set[Path] = field(default_factory=set)

    def put(self, out: Path, content: bytes) -> None:
        """Write ``content`` to ``out`` once per call (a shared provenance
        folder reaches every layer of an AFM run)."""
        if out in self.seen:
            return
        self.seen.add(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)
        self.written.append(out)

    def put_file(self, node, out: Path) -> None:
        """A ``SinglefileData``'s bytes at ``out``."""
        self.put(out, node.base.repository.get_object_content(node.filename, mode='rb'))

    def put_folder(self, node, mapper: Callable[[str], Path]) -> None:
        """Every file of a ``FolderData`` at ``mapper(its stored path)``."""
        repository = node.base.repository
        for root, _dirs, files in repository.walk():
            for name in files:
                rel = (root / name).as_posix()
                self.put(mapper(rel), repository.get_object_content(rel, mode='rb'))


def _provenance_mapper(run_dest: Path, family: str) -> Callable[[str], Path]:
    """``provenance.json`` in the run; ``provenance/**`` one level up for the
    AFM-like families (``run.tree.provenance_dir_for``)."""
    above = run_dest.parent if family in AFM_LIKE_MODELS else run_dest

    def mapper(rel: str) -> Path:
        return above / rel if rel.startswith('provenance/') else run_dest / rel

    return mapper


def _write_inputs(tree: _Tree, calc, params: Mapping[str, Any], run_dest: Path) -> None:
    inputs = calc.inputs
    if 'provenance' in inputs:
        tree.put_folder(inputs.provenance, _provenance_mapper(run_dest, params.get('family')))
    tree.put_file(inputs.deck, run_dest / 'lammps' / inputs.deck.filename)
    for node in (inputs.includes.values() if 'includes' in inputs else ()):
        tree.put_file(node, run_dest / 'lammps' / node.filename)
    build = run_dest if params.get('build_level') == 'run' else run_dest.parent
    for key, node in (inputs.structures.items() if 'structures' in inputs else ()):
        where = build / 'build' if key.startswith('build_') else run_dest / 'data'
        tree.put_file(node, where / node.filename)


def _write_outputs(tree: _Tree, calc, params: Mapping[str, Any], run_dest: Path) -> None:
    outputs = calc.outputs
    for namespace in ('points', 'peratom'):
        for node in (outputs[namespace].values() if namespace in outputs else ()):
            if not is_superseded(node):
                tree.put_folder(node, lambda rel: run_dest / rel)
    for namespace in ('snapshots', 'structures', 'aux'):
        for node in (outputs[namespace].values() if namespace in outputs else ()):
            tree.put_file(node, run_dest / 'data' / node.filename)
    if 'log' in outputs:
        log_name = params.get('log_name') or f'lammps/{outputs.log.filename}'
        tree.put_file(outputs.log, run_dest / log_name)
    info = outputs.run_info.get_dict() if 'run_info' in outputs else {}
    for entry in info.get('not_stored') or []:
        logger.warning('%s: %s was not stored (%s bytes, sha256=%s) and is missing '
                       'from the export.', params.get('rel_path'), entry.get('name'),
                       entry.get('size_bytes'), entry.get('sha256'))


def _halt_line(point) -> Optional[str]:
    """The ``# HALT`` line that gives ``point`` its ``halted`` status again
    (``run_status.halt_detail`` is its inverse); None for any other status.

    The halt registry (``results/halted.txt``) is read by the parser, not
    stored, so the export writes back what each halted point recorded.
    """
    attrs = point.base.attributes
    if attrs.get('status') != 'halted':
        return None
    match = _HALT_DETAIL_RE.match(str(attrs.get('status_detail') or ''))
    if match is None:
        return None
    stage = match['stage']
    tokens = [f'stage={stage}']
    if stage != 'system_equil' and attrs.get('load') is not None:
        key = 'load' if stage == 'system' or attrs.get('load_kind') == 'force' else 'pressure'
        tokens.append(f"{key}={float(attrs.get('load')):g}")
    if stage == 'slide':
        tokens += [f"{name}={float(attrs.get(name)):g}" for name in ('angle', 'speed')
                   if attrs.get(name) is not None]
    tokens += [f"T={match['T']}", f"step={match['step']}"]
    return '# HALT ' + ' '.join(tokens)


def _write_halts(tree: _Tree, points: List[Tuple[Any, str, Any]]) -> None:
    from ..postprocessing.run_status import HALT_RECORD  # pylint: disable=import-outside-toplevel

    lines: Dict[Path, List[str]] = {}
    for calc, _key, point in points:
        line = _halt_line(point)
        if line is not None:
            run_dest = tree.dest / calc.inputs.parameters['rel_path']
            lines.setdefault(run_dest / 'results' / HALT_RECORD, []).append(line)
    for path, entries in sorted(lines.items()):
        text = '\n'.join(dict.fromkeys(entries)) + '\n'
        tree.put(path, text.encode('utf-8'))


def _live_rulings(points: List[Any]) -> Dict[str, List[Any]]:
    """point_uuid -> its live ruled verdicts (every generation), oldest first."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .steps import KIND_EXTRA, POINT_UUID_EXTRA  # pylint: disable=import-outside-toplevel

    uuids = sorted({p.base.attributes.get('point_uuid') for p in points} - {None})
    out: Dict[str, List[Any]] = {}
    for begin in range(0, len(uuids), _CHUNK):
        qb = orm.QueryBuilder().append(orm.Dict, tag='v', project='*', filters={
            f'extras.{KIND_EXTRA}': 'ruled_verdict',
            f'extras.{POINT_UUID_EXTRA}': {'in': uuids[begin:begin + _CHUNK]}})
        qb.order_by({'v': {'id': 'asc'}})
        for node in qb.all(flat=True):
            if not is_superseded(node):
                out.setdefault(node.base.extras.get(POINT_UUID_EXTRA), []).append(node)
    return out


def _write_rulings(tree: _Tree, points: List[Tuple[Any, str, Any]]) -> None:
    """``integrity_rulings.yaml`` from the live user rulings of the points, so
    the DataReader applies the same rulings again.

    A point's entry is the ruling the policy selects when that is a ruled
    verdict; otherwise its newest live ruled verdict of any generation (a
    ruling on an older thresholds generation is still in force for it). A
    withdrawn ruling is superseded, so it is never written.
    """
    import yaml  # pylint: disable=import-outside-toplevel

    from ..postprocessing.integrity_rulings import RULINGS_FILE  # pylint: disable=import-outside-toplevel
    from .steps import latest_verdicts  # pylint: disable=import-outside-toplevel

    nodes = [point for _c, _k, point in points]
    selected = latest_verdicts(nodes)
    rulings = _live_rulings(nodes)
    entries = []
    for point in nodes:
        ruled = rulings.get(point.base.attributes.get('point_uuid'), [])
        if not ruled:
            continue
        chosen = selected.get(point.pk)
        verdict = chosen if chosen is not None and chosen.pk in {r.pk for r in ruled} \
            else ruled[-1]
        content = verdict.creator.inputs.ruling.get_dict()
        entries.append({k: content[k] for k in ('point', 'ruling', 'note', 'date', 'override')
                        if k in content})
    if entries:
        entries.sort(key=lambda e: str(e.get('point')))
        tree.put(tree.dest / RULINGS_FILE,
                 yaml.safe_dump(entries, sort_keys=False).encode('utf-8'))


def export_results(label: str, dest: Path, *,
                   profile: Optional[str] = None) -> List[Path]:
    """Write the set ``friction2d/<label>`` back into a simulation tree at ``dest``.

    For every live CalcJob: ``provenance.json`` + ``provenance/`` from its
    ``provenance`` input (``provenance/`` one level up for the AFM-like
    families), ``lammps/*.in`` from ``deck`` and ``includes``, ``build/*.lmp``
    and the handed ``data/load_*.data`` from ``structures``; from its outputs
    the live points' ``results/*``, the per-atom files (``peratom``),
    ``data/final_*.data`` (``snapshots``), ``data/load_*.data`` and
    ``data/hold_*.txt`` (``structures``, ``aux``) and the log. Halted points
    get their ``results/halted.txt`` line back, and user rulings their
    ``integrity_rulings.yaml`` entry, so ``postprocess read`` gives the source
    tree's statuses. A file the import did not store (over
    ``max_stored_file_mb``) is logged. Returns every file written, sorted.
    """
    _load_profile(profile)
    calcs = _live_calcjobs(resolve_set(label))
    tree = _Tree(Path(dest))
    tree.dest.mkdir(parents=True, exist_ok=True)
    for calc in calcs:
        params = calc.inputs.parameters.get_dict()
        run_dest = tree.dest / params['rel_path']
        _write_inputs(tree, calc, params, run_dest)
        _write_outputs(tree, calc, params, run_dest)
    points = _live_points(calcs)
    _write_halts(tree, points)
    _write_rulings(tree, points)
    logger.info('Exported %d files for set %r to %s', len(tree.written), label, tree.dest)
    return sorted(tree.written)


# ---------------------------------------------------------------------------
# rebuild_set
# ---------------------------------------------------------------------------

def _nested_model_cls(annotation: Any) -> Optional[Type[BaseModel]]:
    """The ``BaseModel`` subclass ``annotation`` names, through ``Optional``/``List``."""
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        return _nested_model_cls(args[0]) if args else None
    if origin is not None:  # Union (incl. Optional) and any other generic
        none_type = type(None)
        for arg in get_args(annotation):
            if arg is none_type:  # the `None` half of `Optional[X]`
                continue
            found = _nested_model_cls(arg)
            if found is not None:
                return found
        return None
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _normalize_config_aliases(data: Dict[str, Any], model_cls: Type[BaseModel]) -> None:
    """Rename ``model_cls``'s fields back to the aliases their stored JSON dropped.

    ``run_simulations`` stores ``model_dump_json()`` without ``by_alias``, so a
    ``Field(alias=X)`` round-trips under its Python name (``sheet`` for
    ``'2D'``, ``flake`` for ``'Flake'``). The raw-dict readers
    (``expand_config_sweeps``, ``output_dir_for``) and Pydantic itself (for a
    class without ``populate_by_name``) want the alias; a field-name key would
    silently validate to its default. Recurses into nested models that need it;
    mutates ``data``; idempotent.
    """
    tolerant = model_cls.model_config.get('populate_by_name', False) or \
        model_cls.model_config.get('validate_by_name', False)
    for name, info in model_cls.model_fields.items():
        alias = info.alias
        key = name
        if alias and alias != name and not tolerant:
            if name in data and alias not in data:
                data[alias] = data.pop(name)
            key = alias
        nested_cls = _nested_model_cls(info.annotation)
        if nested_cls is None or key not in data or data[key] is None:
            continue
        if isinstance(data[key], list):
            for item in data[key]:
                if isinstance(item, dict):
                    _normalize_config_aliases(item, nested_cls)
        elif isinstance(data[key], dict):
            _normalize_config_aliases(data[key], nested_cls)


def _fix_config_aliases(config_json: Path, model: str) -> None:
    """Normalise one staged ``config.json`` back to alias form, against the
    config class ``run_simulations`` builds for ``model``. Rewrites the staged
    copy only."""
    config_cls = AFMSimulationConfig if model in AFM_LIKE_MODELS else SheetOnSheetSimulationConfig
    data = json.loads(config_json.read_text(encoding='utf-8'))
    _normalize_config_aliases(data, config_cls)
    config_json.write_text(json.dumps(data, indent=2), encoding='utf-8')


def _model_of(rel_path: str) -> str:
    from ..run.naming import parse_run_path  # pylint: disable=import-outside-toplevel
    try:
        return parse_run_path(rel_path).model
    except ValueError:
        return PurePosixPath(rel_path).parts[0]


def rebuild_set(label: str, dest: Path, *, profile: Optional[str] = None,
                generate_hpc: bool = False, failures: Optional[List[str]] = None
                ) -> Tuple[List[Path], Path]:
    """Regenerate a set's LAMMPS inputs through the real router.

    Exports the set (:func:`export_results`) into a staging directory outside
    ``dest`` -- a staged ``lammps/`` inside ``dest`` would be found by
    ``JobManifest``'s ``rglob('lammps')`` and duplicate every job -- and hands
    each DISTINCT staged ``provenance/config.json`` (by path: every ``L{n}`` of
    an AFM run shares one) to ``run.artifacts.generate_package``, alias-fixed
    (:func:`_fix_config_aliases`), with ``allow_generate_amorphous=False``
    and ``simulation_root_name='.'`` so each run keeps its ``rel_path``. The
    run's stored ``provenance/settings.yaml`` is the settings file. Returns
    ``(created paths, dest)``; a run that could not be rebuilt (its amorphous block is
    missing, say) is logged and, when ``failures`` is a list, appended to it.
    """
    _load_profile(profile)
    rel_paths = sorted({calc.inputs.parameters['rel_path']
                        for calc in _live_calcjobs(resolve_set(label))})
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    created: List[Path] = []
    seen: Set[Path] = set()
    with tempfile.TemporaryDirectory(prefix='frictionsim2d-rebuild-') as tmp_dir:
        staging = Path(tmp_dir)
        export_results(label, staging, profile=profile)
        for rel_path in rel_paths:
            prov_dir = provenance_dir_for(staging / rel_path)
            config_json = prov_dir / 'config.json'
            if not config_json.is_file():
                logger.warning('No config.json stored for %s -- skipping', rel_path)
                continue
            if config_json.resolve() in seen:
                continue
            seen.add(config_json.resolve())
            model = _model_of(rel_path)
            _fix_config_aliases(config_json, model)
            settings_file = prov_dir / 'settings.yaml'
            result = generate_package(
                config_json, model, output_root=dest,
                settings_file=settings_file if settings_file.is_file() else None,
                generate_hpc=generate_hpc,
                simulation_root_name='.', allow_generate_amorphous=False)
            for failure in result.failures:
                logger.error('Rebuild failed: %s', failure.render())
                if failures is not None:
                    failures.append(f'{rel_path}: {failure.render()}')
            created.extend(result.created)
    return created, dest


# ---------------------------------------------------------------------------
# export_archive: entities, gate, closure
# ---------------------------------------------------------------------------

def _point_name(point) -> str:
    """``'<rel_path>:<deck> points__<key> (<point_uuid>)'``."""
    from aiida.common.links import LinkType  # pylint: disable=import-outside-toplevel

    attrs = point.base.attributes
    links = point.base.links.get_incoming(link_type=LinkType.CREATE).all()
    key = links[0].link_label if links else '?'
    return (f"{attrs.get('rel_path')}:{attrs.get('deck_name')} {key} "
            f"({attrs.get('point_uuid')})")


def _without_verdict(points: List[Any]) -> List[str]:
    """``points`` that have no live verdict, named."""
    from .steps import latest_verdicts  # pylint: disable=import-outside-toplevel

    verdicts = latest_verdicts(points)
    return [_point_name(point) for point in points if verdicts.get(point.pk) is None]


def _unchecked(group) -> List[str]:
    """Every live point of the set without a live verdict."""
    return _without_verdict([point for _c, _k, point in _live_points(_live_calcjobs(group))])


def _created_by(pks: Iterable[int], source) -> Set[int]:
    """pks of the Data that ``source`` nodes (``pks``) CREATE-link to."""
    from aiida import orm  # pylint: disable=import-outside-toplevel
    from aiida.common.links import LinkType  # pylint: disable=import-outside-toplevel

    out: Set[int] = set()
    ordered = sorted(pks)
    for start in range(0, len(ordered), _CHUNK):
        qb = orm.QueryBuilder().append(
            source, filters={'id': {'in': ordered[start:start + _CHUNK]}}, tag='s')
        qb.append(orm.Data, with_incoming='s', project='id',
                  edge_filters={'type': LinkType.CREATE.value})
        out.update(qb.all(flat=True))
    return out


def _consumers(pks: Iterable[int], labels: Optional[Sequence[str]] = None) -> Set[int]:
    """pks of the calcfunctions with an input among ``pks`` (only those whose
    ``process_label`` is in ``labels``, when given)."""
    from aiida import orm  # pylint: disable=import-outside-toplevel
    from aiida.common.links import LinkType  # pylint: disable=import-outside-toplevel

    filters = {} if labels is None else {'attributes.process_label': {'in': list(labels)}}
    out: Set[int] = set()
    ordered = sorted(pks)
    for start in range(0, len(ordered), _CHUNK):
        qb = orm.QueryBuilder().append(
            orm.Data, filters={'id': {'in': ordered[start:start + _CHUNK]}}, tag='d')
        qb.append(orm.CalcFunctionNode, with_incoming='d', project='id', filters=filters,
                  edge_filters={'type': LinkType.INPUT_CALC.value})
        out.update(qb.all(flat=True))
    return out


def _reachable_functions(frontier: Set[int],
                         labels: Optional[Sequence[str]] = None) -> Set[int]:
    """Every calcfunction reached from ``frontier`` through calcfunction
    inputs and outputs, transitively."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    functions: Set[int] = set()
    seen = set(frontier)
    while frontier:
        new = _consumers(frontier, labels) - functions
        functions |= new
        frontier = _created_by(new, orm.CalcFunctionNode) - seen
        seen |= frontier
    return functions


def _peratom_pks(calc_pks: Iterable[int]) -> Set[int]:
    from aiida import orm  # pylint: disable=import-outside-toplevel

    out: Set[int] = set()
    ordered = sorted(calc_pks)
    for start in range(0, len(ordered), _CHUNK):
        qb = orm.QueryBuilder().append(
            orm.CalcJobNode, filters={'id': {'in': ordered[start:start + _CHUNK]}}, tag='c')
        qb.append(orm.FolderData, with_incoming='c', project='id',
                  edge_filters={'label': {'like': 'peratom__%'}})
        out.update(qb.all(flat=True))
    return out


def _start_pks(group, *, include_peratom: bool = False) -> Set[int]:
    """The set's CalcJobs plus every calcfunction reached from their outputs
    (verdicts, rulings, summaries), transitively; the per-atom nodes are not
    followed unless ``include_peratom``."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    calcs = {calc.pk for calc in set_calcjobs(group)}
    frontier = _created_by(calcs, orm.CalcJobNode)
    if not include_peratom:
        frontier -= _peratom_pks(calcs)
    return calcs | _reachable_functions(frontier)


def _analysis_groups(label: str) -> List[Any]:
    """The analysis Groups (``friction2d/analyses/<dataset>``) of every
    dataset built from the set ``label``, alone or with other sets, in label
    order. A result's CSV is the only place another set's points appear: its
    inputs are the dataset's table ``Dict`` (no creator), a params ``Dict``
    and the results it read, so no foreign CalcJob is reached."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .steps import DATASET_EXTRA, KIND_EXTRA  # pylint: disable=import-outside-toplevel

    qb = orm.QueryBuilder().append(
        orm.Dict, filters={f'extras.{KIND_EXTRA}': 'dataset'},
        project=[f'extras.{DATASET_EXTRA}', 'attributes.labels'])
    names = sorted({name for name, labels in qb.all() if label in (labels or ())})
    if not names:
        return []
    found = orm.QueryBuilder().append(orm.Group, filters={
        'label': {'in': [f'{ANALYSIS_GROUP_PREFIX}{name}' for name in names]}})
    return sorted(found.all(flat=True), key=lambda g: g.label)


def _check_closure(label: str, group, start: Set[int]) -> None:
    """Raise :class:`ExportClosureError` when the default export closure of
    ``start`` and the group's own nodes holds a CalcJob not in ``group``."""
    from aiida import orm  # pylint: disable=import-outside-toplevel
    from aiida.tools.graph.graph_traversers import get_nodes_export  # pylint: disable=import-outside-toplevel

    members = {node.pk for node in group.nodes}
    closure = get_nodes_export(sorted(start | members))['nodes']
    foreign = []
    ordered = sorted(closure)
    for begin in range(0, len(ordered), _CHUNK):
        qb = orm.QueryBuilder().append(
            orm.CalcJobNode, filters={'id': {'in': ordered[begin:begin + _CHUNK]}},
            project=['id', 'uuid', 'label'])
        foreign += [f'{uuid} {name}' for pk, uuid, name in qb.all() if pk not in members]
    if foreign:
        raise ExportClosureError(label, sorted(foreign))


# ---------------------------------------------------------------------------
# export_archive: rewriting the written archive
# ---------------------------------------------------------------------------

_DB = 'db.sqlite3'
_META = 'metadata.json'
_REPO = 'repo/'
_CALCJOB_TYPE = 'process.calculation.calcjob.CalcJobNode.'
_WINDOWS_ABS = re.compile(r'^[A-Za-z]:[\\/]')


class ArchiveFormatError(RuntimeError):
    """The written archive is not the format :func:`_rewrite_archive` edits."""


def _connect(path: Path) -> sqlite3.Connection:
    """The archive database connection (a seam for the secure-delete test)."""
    return sqlite3.connect(path)


def _blob_key(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _abs_basename(value: Any) -> Optional[str]:
    """The basename of an absolute path string, or None for anything else."""
    if not isinstance(value, str) or not (value.startswith(('/', '~/'))
                                          or _WINDOWS_ABS.match(value)):
        return None
    name = (PureWindowsPath(value) if '\\' in value else PurePosixPath(value)).name
    return name if name and name != value else None


def _scrub_hpc(hpc: Dict[str, Any], hosts: Mapping[str, str]) -> bool:
    """Drop :data:`HPC_SCRUBBED_KEYS` from one ``hpc`` mapping and turn its
    ``hpc_host`` into the machine label (dropped when unknown)."""
    changed = False
    for key in HPC_SCRUBBED_KEYS:
        if hpc.get(key) not in (None, ''):
            hpc.pop(key)
            changed = True
    host = hpc.get('hpc_host')
    if host is not None and hosts.get(host) != host:
        if host in hosts:
            hpc['hpc_host'] = hosts[host]
        else:
            hpc.pop('hpc_host')
        changed = True
    return changed


def _scrub(value: Any, hosts: Mapping[str, str], in_database: bool = False) -> bool:
    """Anonymise one parsed provenance document in place: every ``hpc``
    mapping (:func:`_scrub_hpc`), ``user``/``host`` under ``database``, and
    every absolute path string -> its basename."""
    changed = False
    if isinstance(value, dict):
        for key in list(value):
            item = value[key]
            if key == 'hpc' and isinstance(item, dict):
                changed = _scrub_hpc(item, hosts) or changed
            if in_database and key in DATABASE_SCRUBBED_KEYS and item not in (None, ''):
                value.pop(key)
                changed = True
                continue
            name = _abs_basename(item)
            if name is not None:
                value[key] = name
                changed = True
                continue
            changed = _scrub(item, hosts, in_database or key == 'database') or changed
    elif isinstance(value, list):
        for i, item in enumerate(value):
            name = _abs_basename(item)
            if name is not None:
                value[i] = name
                changed = True
            else:
                changed = _scrub(item, hosts, in_database) or changed
    return changed


def _scrub_document(name: str, content: bytes, hosts: Mapping[str, str]) -> Optional[bytes]:
    """The anonymised bytes of one provenance document, or None if unchanged."""
    import yaml  # pylint: disable=import-outside-toplevel

    suffix = PurePosixPath(name).suffix.lower()
    try:
        if suffix == '.json':
            data = json.loads(content.decode('utf-8'))
        elif suffix in ('.yaml', '.yml'):
            data = yaml.safe_load(content.decode('utf-8'))
        else:
            return None
    except (UnicodeDecodeError, ValueError, yaml.YAMLError):
        return None
    if not _scrub(data, hosts):
        return None
    if suffix == '.json':
        return json.dumps(data, indent=2).encode('utf-8')
    return yaml.safe_dump(data, sort_keys=False).encode('utf-8')


def _walk_metadata(tree: Mapping[str, Any], prefix: str = '') -> Iterable[Tuple[str, dict]]:
    """``(stored path, entry)`` of every file in a node's repository metadata."""
    for name, entry in (tree.get('o') or {}).items():
        path = f'{prefix}{name}'
        if 'k' in entry:
            yield path, entry
        else:
            yield from _walk_metadata(entry, f'{path}/')


@dataclass
class _Archive:
    """The archive being rewritten: its db, its blob reader and new blobs."""
    con: sqlite3.Connection
    read: Callable[[str], bytes]
    blobs: Dict[str, bytes] = field(default_factory=dict)

    def rows(self, sql: str, args: Tuple = ()) -> List[Tuple]:
        """All rows of one query."""
        return self.con.execute(sql, args).fetchall()

    def set_json(self, pk: int, column: str, value: Any, table: str = 'db_dbnode') -> None:
        """Write one JSON column of a node (or group) row."""
        if column not in ('attributes', 'extras', 'repository_metadata') or \
                table not in ('db_dbnode', 'db_dbgroup'):
            raise ValueError(f'{table}.{column}')
        self.con.execute(f'UPDATE {table} SET {column} = ? WHERE id = ?',
                         (json.dumps(value), pk))


def _inputs_labelled(archive: _Archive, label: str) -> List[Tuple]:
    return archive.rows(
        'SELECT DISTINCT n.id, n.attributes, n.extras, n.repository_metadata '
        'FROM db_dbnode n JOIN db_dblink l ON l.input_id = n.id '
        "WHERE l.type = 'input_calc' AND l.label = ?", (label,))


def _anonymise_folder(archive: _Archive, row: Tuple, hosts: Mapping[str, str]) -> bool:
    """Scrub the provenance documents of one provenance ``FolderData``."""
    from .store import FOLDER_SHA_EXTRA  # pylint: disable=import-outside-toplevel

    pk, _attributes, extras, metadata = row
    metadata = json.loads(metadata)
    members, changed = [], False
    for name, entry in sorted(_walk_metadata(metadata)):
        if not name.startswith(('provenance/cif/', 'provenance/potentials/')):
            new = _scrub_document(name, archive.read(entry['k']), hosts)
            if new is not None:
                entry['k'] = _blob_key(new)
                archive.blobs[entry['k']] = new
                changed = True
        members.append(f"{name}\0{entry['k']}\n")
    if not changed:
        return False
    extras = json.loads(extras or '{}')
    extras.pop('_aiida_hash', None)
    if FOLDER_SHA_EXTRA in extras:
        extras[FOLDER_SHA_EXTRA] = hashlib.sha256(''.join(members).encode('utf-8')).hexdigest()
    extras[ANONYMISED_EXTRA] = True
    archive.set_json(pk, 'repository_metadata', metadata)
    archive.set_json(pk, 'extras', extras)
    return True


def _mark(archive: _Archive, pk: int, extras: Optional[str], attributes: Dict[str, Any]) -> None:
    content = json.loads(extras or '{}')
    content.pop('_aiida_hash', None)
    content[ANONYMISED_EXTRA] = True
    archive.set_json(pk, 'attributes', attributes)
    archive.set_json(pk, 'extras', content)


def _anonymise_computers(archive: _Archive) -> Tuple[int, Dict[str, str], Dict[int, str]]:
    """Hostname -> label on every Computer: ``(changed, hostname -> label, id -> label)``."""
    computers = archive.rows('SELECT id, label, hostname FROM db_dbcomputer')
    changed = 0
    for pk, label, hostname in computers:
        if hostname != label:
            archive.con.execute('UPDATE db_dbcomputer SET hostname = ? WHERE id = ?',
                                (label, pk))
            changed += 1
    return (changed, {hostname: label for _pk, label, hostname in computers},
            {pk: label for pk, label, _hostname in computers})


def _anonymise_codes(archive: _Archive) -> int:
    changed = 0
    for pk, attributes, extras in archive.rows(
            "SELECT id, attributes, extras FROM db_dbnode WHERE node_type LIKE 'data.core.code.%'"):
        attributes = json.loads(attributes or '{}')
        executable = attributes.get('filepath_executable')
        if executable and PurePosixPath(executable).name != executable:
            attributes['filepath_executable'] = PurePosixPath(executable).name
            _mark(archive, pk, extras, attributes)
            changed += 1
    return changed


def _anonymise_parameters(archive: _Archive) -> int:
    changed = 0
    for pk, attributes, extras, _meta in _inputs_labelled(archive, 'parameters'):
        attributes = json.loads(attributes or '{}')
        if 'hpc_run_path' in attributes:
            attributes.pop('hpc_run_path')
            _mark(archive, pk, extras, attributes)
            changed += 1
    return changed


def _anonymise_rulings(archive: _Archive) -> int:
    """A ruling Dict's ``source`` (the rulings file it came from) -> its basename."""
    changed = 0
    for pk, attributes, extras, _meta in _inputs_labelled(archive, 'ruling'):
        attributes = json.loads(attributes or '{}')
        name = _abs_basename(attributes.get('source'))
        if name is not None:
            attributes['source'] = name
            _mark(archive, pk, extras, attributes)
            changed += 1
    return changed


def _anonymise_remote(archive: _Archive, labels: Mapping[int, str]) -> int:
    """``remote_workdir`` of each CalcJob and ``remote_path`` of its
    ``RemoteData`` -> ``<machine label>:<rel_path>``."""
    changed = 0
    for pk, attributes, extras, computer in archive.rows(
            'SELECT id, attributes, extras, dbcomputer_id FROM db_dbnode WHERE node_type = ?',
            (_CALCJOB_TYPE,)):
        params = archive.rows(
            'SELECT n.attributes FROM db_dbnode n JOIN db_dblink l ON l.input_id = n.id '
            "WHERE l.output_id = ? AND l.label = 'parameters'", (pk,))
        rel_path = json.loads(params[0][0] or '{}').get('rel_path') if params else None
        where = f"{labels.get(computer, 'unknown')}:{rel_path or '?'}"
        attributes = json.loads(attributes or '{}')
        if attributes.get('remote_workdir') not in (None, where):
            attributes['remote_workdir'] = where
            _mark(archive, pk, extras, attributes)
            changed += 1
        for rpk, rattributes, rextras in archive.rows(
                'SELECT n.id, n.attributes, n.extras FROM db_dbnode n JOIN db_dblink l '
                'ON (l.input_id = n.id AND l.output_id = ?) OR (l.output_id = n.id AND '
                "l.input_id = ?) WHERE n.node_type = 'data.core.remote.RemoteData.'",
                (pk, pk)):
            rattributes = json.loads(rattributes or '{}')
            if rattributes.get('remote_path') not in (None, where):
                rattributes['remote_path'] = where
                _mark(archive, rpk, rextras, rattributes)
                changed += 1
    return changed


def _anonymise_groups(archive: _Archive, hosts: Mapping[str, str]) -> int:
    """A set Group's ``source_root`` -> its basename, ``source_host`` -> the
    machine label, or ``'unknown'``, and each source path of the ruling
    ledger its basename (:func:`_basenamed_ledger`)."""
    # pylint: disable=import-outside-toplevel,protected-access
    from .steps import LEDGER_EXTRA
    changed = 0
    for pk, extras in archive.rows('SELECT id, extras FROM db_dbgroup'):
        content = json.loads(extras or '{}')
        before = dict(content)
        if content.get('source_root'):
            content['source_root'] = PurePosixPath(str(content['source_root'])).name
        if content.get('source_host'):
            content['source_host'] = hosts.get(content['source_host'], 'unknown')
        if content.get(LEDGER_EXTRA):
            content[LEDGER_EXTRA] = _basenamed_ledger(content[LEDGER_EXTRA])
        if content != before:
            archive.set_json(pk, 'extras', content, table='db_dbgroup')
            changed += 1
    return changed


def _basenamed_ledger(ledger: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The ruling ledger with every source path its basename; two sources of
    one basename become ``x.yaml`` and ``x.yaml (2)``, the same name for the
    same path in every point."""
    names: Dict[str, str] = {}
    for source in sorted({s for by_source in ledger.values() for s in by_source}):
        name = PurePosixPath(source).name
        n, unique = 1, name
        while unique in names.values():
            n += 1
            unique = f'{name} ({n})'
        names[source] = unique
    return {pid: {names[s]: entry for s, entry in by_source.items()}
            for pid, by_source in ledger.items()}


#: An absolute path inside free text (not ``eV/angstrom``: a word character
#: before the slash is not a path start).
_TEXT_PATH = re.compile(r'(?<![\w.~])(~?/[^\s\'"()\[\],;]+)')
#: Log-record metadata that is a traceback or a frame, never kept.
_LOG_DROPPED = ('exc_info', 'exc_text', 'stack_info', 'args', 'msg')


def _basenamed_text(text: str) -> str:
    """``text`` with every absolute path reduced to its basename."""
    return _TEXT_PATH.sub(lambda m: PurePosixPath(m.group(1)).name or m.group(1), text)


def _anonymise_logs(archive: _Archive) -> int:
    """Node logs keep their level and the first line of their message, with
    absolute paths reduced to basenames; the traceback body and the
    traceback/frame metadata are dropped (a record's ``pathname`` is basenamed)."""
    changed = 0
    for pk, message, metadata in archive.rows('SELECT id, message, metadata FROM db_dblog'):
        lines = (message or '').splitlines()
        new_message = _basenamed_text(lines[0]) if lines else ''
        content = json.loads(metadata or '{}')
        new_metadata = {k: v for k, v in content.items() if k not in _LOG_DROPPED}
        _scrub(new_metadata, {})
        for key, value in new_metadata.items():
            if isinstance(value, str):
                new_metadata[key] = _basenamed_text(value)
        if new_message != message or new_metadata != content:
            archive.con.execute('UPDATE db_dblog SET message = ?, metadata = ? WHERE id = ?',
                                (new_message, json.dumps(new_metadata), pk))
            changed += 1
    return changed


#: Process-node attributes that carry free text a failure wrote: kept as
#: their first line, with absolute paths reduced to basenames.
_PROCESS_TEXT = ('exit_message', 'process_status', 'exception')


def _anonymise_processes(archive: _Archive) -> int:
    """``exit_message``/``process_status``/``exception`` of every process node."""
    changed = 0
    for pk, attributes, extras in archive.rows(
            "SELECT id, attributes, extras FROM db_dbnode WHERE node_type LIKE 'process.%'"):
        attributes = json.loads(attributes or '{}')
        before = dict(attributes)
        for key in _PROCESS_TEXT:
            if isinstance(attributes.get(key), str):
                lines = attributes[key].splitlines()
                attributes[key] = _basenamed_text(lines[0]) if lines else ''
        if attributes != before:
            _mark(archive, pk, extras, attributes)
            changed += 1
    return changed


def _anonymise(archive: _Archive) -> int:
    """Rewrite the archive's HPC and local details (see the module
    docstring); returns the number of rows changed."""
    changed, hosts, labels = _anonymise_computers(archive)
    changed += _anonymise_codes(archive) + _anonymise_parameters(archive)
    changed += _anonymise_rulings(archive)
    changed += _anonymise_remote(archive, labels) + _anonymise_groups(archive, hosts)
    changed += _anonymise_logs(archive) + _anonymise_processes(archive)
    for row in _inputs_labelled(archive, 'provenance'):
        changed += _anonymise_folder(archive, row, hosts)
    return changed


def _drop_peratom(archive: _Archive) -> int:
    """Delete the ``peratom.<key>`` nodes and everything that refers to them."""
    pks = [pk for (pk,) in archive.rows(
        "SELECT output_id FROM db_dblink WHERE type = 'create' "
        "AND substr(label, 1, 9) = 'peratom__'")]
    for begin in range(0, len(pks), _CHUNK):
        chunk = pks[begin:begin + _CHUNK]
        marks = ','.join('?' * len(chunk))
        for sql in (f'DELETE FROM db_dblink WHERE input_id IN ({marks}) OR output_id IN ({marks})',
                    f'DELETE FROM db_dbgroup_dbnodes WHERE dbnode_id IN ({marks})',
                    f'DELETE FROM db_dbcomment WHERE dbnode_id IN ({marks})',
                    f'DELETE FROM db_dblog WHERE dbnode_id IN ({marks})',
                    f'DELETE FROM db_dbnode WHERE id IN ({marks})'):
            archive.con.execute(sql, tuple(chunk) * sql.count('IN ('))
    return len(pks)


def _referenced(archive: _Archive) -> Set[str]:
    keys: Set[str] = set()
    for (metadata,) in archive.rows('SELECT repository_metadata FROM db_dbnode'):
        keys.update(entry['k'] for _n, entry in _walk_metadata(json.loads(metadata or '{}')))
    return keys


def _write_zip(source: zipfile.ZipFile, target: Path, db_path: Path,
               keep: Set[str], blobs: Mapping[str, bytes]) -> None:
    """A new archive: the old metadata, the edited db, the kept and new blobs.
    ``metadata.json`` and ``db.sqlite3`` stay the first two entries, where
    aiida's reader looks for them."""
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        out.writestr(source.getinfo(_META), source.read(_META))
        out.write(db_path, _DB)
        written: Set[str] = set()
        for info in source.infolist():
            name = info.filename
            if not name.startswith(_REPO) or name in written:
                continue
            key = name[len(_REPO):]
            if key in keep and key not in blobs:
                fresh = zipfile.ZipInfo(name, date_time=info.date_time)
                fresh.compress_type = info.compress_type
                with source.open(info) as src, out.open(fresh, 'w', force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst)
                written.add(name)
        for key, content in sorted(blobs.items()):
            if key in keep and f'{_REPO}{key}' not in written:
                out.writestr(f'{_REPO}{key}', content)


def _check_format(source: zipfile.ZipFile) -> None:
    """Refuse an archive whose layout :func:`_rewrite_archive` does not know."""
    from aiida.storage.sqlite_zip.migrator import get_schema_version_head  # pylint: disable=import-outside-toplevel

    try:
        metadata = json.loads(source.read(_META))
    except (KeyError, ValueError) as exc:
        raise ArchiveFormatError(f'archive has no readable {_META}: {exc}') from exc
    expected = get_schema_version_head()
    if metadata.get('export_version') != expected or metadata.get('key_format') != 'sha256':
        raise ArchiveFormatError(
            f"archive export_version {metadata.get('export_version')!r} / key_format "
            f"{metadata.get('key_format')!r}; the anonymising rewrite knows only "
            f"{expected!r} / 'sha256' -- update frictionsim2d/aiida/export.py for this aiida-core")


def _edit_db(db_path: Path, read: Callable[[str], bytes], *, anonymise: bool,
             drop_peratom: bool) -> Tuple[Dict[str, int], Set[str], Dict[str, bytes]]:
    """Edit the extracted archive db: ``(counts, blob keys kept, new blobs)``.

    ``secure_delete`` zeroes what is deleted or overwritten, and ``VACUUM``
    rebuilds the file, so no scrubbed byte survives in a free page."""
    counts = {'anonymised': 0, 'peratom_dropped': 0}
    con = _connect(db_path)
    try:
        con.execute('PRAGMA secure_delete=ON')
        archive = _Archive(con, read)
        if drop_peratom:
            counts['peratom_dropped'] = _drop_peratom(archive)
        if anonymise:
            counts['anonymised'] = _anonymise(archive)
        con.commit()
        keep = _referenced(archive)
        con.execute('VACUUM')
    finally:
        con.close()
    return counts, keep, archive.blobs


def _rewrite_archive(path: Path, *, anonymise: bool, drop_peratom: bool) -> Dict[str, int]:
    """Anonymise and/or drop the per-atom nodes of a written archive, in place.
    Returns the counts of what changed."""
    with tempfile.TemporaryDirectory(prefix='frictionsim2d-archive-') as tmp:
        work = Path(tmp)
        db_path = work / _DB
        with zipfile.ZipFile(path) as source:
            _check_format(source)
            if not (anonymise or drop_peratom):
                return {'anonymised': 0, 'peratom_dropped': 0}
            db_path.write_bytes(source.read(_DB))
            counts, keep, blobs = _edit_db(
                db_path, lambda key: source.read(f'{_REPO}{key}'),
                anonymise=anonymise, drop_peratom=drop_peratom)
            if not any(counts.values()):
                return counts
            _write_zip(source, work / 'new.aiida', db_path, keep, blobs)
        shutil.move(str(work / 'new.aiida'), str(path))
    return counts


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _write_archive(entities: List[Any], output_path: Path, *, anonymise: bool,
                   drop_peratom: bool, readme: Callable[[int], str],
                   sidecars: Optional[Mapping[str, Mapping[str, Any]]] = None
                   ) -> Dict[str, int]:
    """Write the archive, its README (``readme(size in bytes)``) and its
    sidecars (``<output_path>.params/``) as partial files beside
    ``output_path``, then move all of them into place: on any failure before
    that, nothing is left at any target and every partial file is removed."""
    from aiida.tools.archive import create_archive  # pylint: disable=import-outside-toplevel

    targets = [output_path, Path(f'{output_path}.README.md')]
    if sidecars is not None:
        targets.append(Path(f'{output_path}.params'))
    for target in targets:
        if target.exists():
            raise FileExistsError(f'{target} exists; remove it or choose another path')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partials = [target.with_name(f'.{target.name}.partial') for target in targets]
    for partial in partials:
        _remove(partial)
    try:
        create_archive(entities, filename=partials[0])
        counts = _rewrite_archive(partials[0], anonymise=anonymise, drop_peratom=drop_peratom)
        partials[1].write_text(readme(partials[0].stat().st_size), encoding='utf-8')
        if sidecars is not None:
            partials[2].mkdir()
            _put_sidecars(sidecars, partials[2])
    except BaseException:
        for partial in partials:
            _remove(partial)
        raise
    for partial, target in zip(partials, targets):
        partial.replace(target)
    return counts


# ---------------------------------------------------------------------------
# export_archive: sidecars and README
# ---------------------------------------------------------------------------

def _sidecar(calc) -> Dict[str, Any]:
    """One deck's ``{block name: block content, 'derived': ...}``."""
    from .compare import block_name  # pylint: disable=import-outside-toplevel

    names = calc.inputs.parameters.get_dict().get('block_names') or {}
    document: Dict[str, Any] = {}
    for label, node in (calc.inputs.blocks.items() if 'blocks' in calc.inputs else ()):
        content = node.get_dict()
        document[names.get(label) or block_name(label, str(content.get('kind')))] = content
    if 'derived' in calc.inputs:
        document['derived'] = calc.inputs.derived.get_dict()
    return document


def _merge_schemas(documents: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """block_schema version -> one schema document over every family exported
    at that version (``fields`` united). ``compare``'s loader picks a schema by
    version alone, so two families that define one path differently at one
    version raise ``ValueError`` rather than lose a definition."""
    merged: Dict[str, Dict[str, Any]] = {}
    for doc in documents:
        version = str(doc.get('block_schema'))
        into = merged.setdefault(version, {**doc, 'family': [], 'fields': {}})
        if doc.get('family') not in into['family']:
            into['family'].append(doc.get('family'))
        for path, spec in (doc.get('fields') or {}).items():
            if into['fields'].setdefault(path, spec) != spec:
                raise ValueError(
                    f'block_schema {version}: field {path!r} differs between families '
                    f'{sorted(str(f) for f in into["family"])}; one schema-{version}.json '
                    f'cannot hold both')
    for doc in merged.values():
        doc['family'] = sorted(str(f) for f in doc['family'])
    return merged


def sidecar_documents(calcs: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    """Every sidecar, by its path relative to ``<archive>.params/``:
    ``<rel_path>/<deck>.json`` per live CalcJob with blocks, plus
    ``schema-<version>.json`` (:func:`_merge_schemas`, which may raise)."""
    documents: Dict[str, Dict[str, Any]] = {}
    schemas = []
    for calc in calcs:
        if is_superseded(calc) or 'blocks' not in calc.inputs:
            continue
        params = calc.inputs.parameters
        documents[f"{params['rel_path']}/{params['deck_name']}.json"] = _sidecar(calc)
        if 'schema' in calc.inputs:
            schemas.append(calc.inputs.schema.get_dict())
    for version, document in _merge_schemas(schemas).items():
        documents[f'schema-{version}.json'] = document
    return documents


def _put_sidecars(documents: Mapping[str, Mapping[str, Any]], params_dir: Path) -> List[Path]:
    written = []
    for rel, document in sorted(documents.items()):
        path = Path(params_dir) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding='utf-8')
        written.append(path)
    return written


def _schema_nodes(calcs: Iterable[Any]) -> List[str]:
    found = {calc.inputs.schema.uuid: calc.inputs.schema.get_dict()
             for calc in calcs if 'schema' in calc.inputs}
    return [f"`{uuid}` (family {doc.get('family')}, block_schema {doc.get('block_schema')})"
            for uuid, doc in sorted(found.items())]


_README = """# FrictionSim2D archive `{name}`

{title}: {n_calcs} CalcJob(s), one per LAMMPS deck of a finished run, imported
with the FrictionSim2D plugin (`aiida.calculations:friction2d.lammps`). Archive size:
{size}.

## Reading it without the plugin

Every node is a core AiiDA type (`Dict`, `FolderData`, `SinglefileData`, `RemoteData`),
so the archive imports and its data loads without FrictionSim2D installed; only
`calc.process_class` and `calc.res` need the plugin. Link labels:

* `points__<key>` (output, `FolderData`): one load point, `<key>` =
  `<load kind><load>_a<angle>_s<speed>` (`force10_a0_s2`; `m` = minus, `p` = point).
  Attributes: `status`, `load`, `angle`, `speed`, `units`, `point_uuid`, `rel_path`.
* `snapshots__<key>` (output, `SinglefileData`): the point's end `final_*.data`.
* `peratom__<key>` (output, `FolderData`): the per-atom tier. {peratom}
* `structures__*`, `aux__*`, `log`, `run_info` (outputs of a system deck).
* `blocks__<label>` (input, `Dict` = `{{kind, block_schema, fields}}`): the parameter
  blocks. A label is the block name with `__` -> `_`; `parameters['block_names']`
  maps each label back to its unambiguous name (`material_sheet_0` ->
  `material__sheet_0`).
* `derived`, `schema`, `provenance`, `deck`, `includes__*`, `potentials__*`,
  `structures__*`, `parameters` (inputs).
* Calcfunctions on the points: `check_integrity` (verdict `Dict`), `apply_ruling` /
  `withdraw_ruling` (user rulings), `summarise` (summary `Dict` with `units`).
* Analyses (set archives): Groups `friction2d/analyses/<dataset>` of `SinglefileData`
  CSV results, each made by a `record_analysis` calcfunction from the dataset's table
  `Dict` and a params `Dict` (`analysis`, `version`, `params`).

Schema node(s), giving the unit, kind and default of every field:
{schemas}
{sidecars}
## Personal data

* Included, for attribution: the AiiDA User (name and email) of whoever imported
  the runs, on every node.
* {details}
* Anonymised nodes keep their UUIDs. An anonymised archive and a
  `keep_hpc_paths=True` archive of the same runs therefore carry the same nodes:
  whichever is imported into a profile first is the version that profile keeps; the
  other adds only nodes the first lacked.
"""

_SIDECARS = """
The same blocks are in `{name}.params/<rel_path>/<deck>.json`, with
`schema-<version>.json`, readable by `FrictionSim2D compare` without AiiDA.
"""

_ANONYMISED = (
    'Anonymised: Computer hostnames are their machine labels; Code executables their '
    'basenames; `RemoteData.remote_path` and the CalcJobs\' `remote_workdir` are '
    '`<machine label>:<rel_path>`; `hpc_run_path` is dropped from `parameters`; the '
    'set Group\'s `source_root` is its basename and `source_host` the machine label '
    '(or `unknown`). In the provenance documents `hpc.hpc_home`/`scratch_dir`/'
    '`log_dir`/`conda_env`/`account` and `database.user`/`host` are dropped, '
    '`hpc.hpc_host` is the machine label, and every absolute path is its basename. '
    'Rewritten nodes carry the extra `friction2d_anonymised`.')
_KEPT = ('Kept as recorded (`keep_hpc_paths=True`): Computer hostnames, Code '
         'executable paths, local `RemoteData` paths and `remote_workdir`, '
         '`hpc_run_path`, the Group\'s `source_root`/`source_host`, and every path, '
         'account and database host/user in the provenance documents.')


def _readme_text(name: str, size: int, *, title: str, calcs: List[Any],
                 include_peratom: bool, keep_hpc_paths: bool, sidecars: bool = True) -> str:
    """The README of archive ``name`` (``size`` bytes)."""
    return _README.format(
        name=name, title=title, n_calcs=len(calcs),
        size=f'{size / 1e6:.1f} MB ({size} bytes)',
        peratom=('Included.' if include_peratom
                 else 'Left out of this archive (`include_peratom=False`).'),
        schemas='\n'.join(f'* {line}' for line in _schema_nodes(calcs)) or '* none',
        sidecars=_SIDECARS.format(name=name) if sidecars else '',
        details=_KEPT if keep_hpc_paths else _ANONYMISED)


def _log_size(path: Path, what: str) -> None:
    size = path.stat().st_size
    logger.info('%s: %s is %.1f MB (%d bytes)', what, path, size / 1e6, size)


def export_archive(label: str, output_path: Path, *, profile: Optional[str] = None,
                   force: bool = False, include_peratom: bool = False,
                   keep_hpc_paths: bool = False) -> Path:
    """Archive the set ``friction2d/<label>``.

    The entities are the set Group (so an import re-creates it), its CalcJobs
    and every calcfunction reached from their outputs (verdicts, rulings,
    summaries), and the analysis Groups ``friction2d/analyses/<dataset>`` of
    every dataset built from the set (:func:`_analysis_groups`), so the
    stored analysis results and their ``record_analysis`` calcfunctions
    travel too; aiida's export closure adds their inputs and created outputs
    -- for an analysis, the dataset's table Dict and the results it read.
    Dataset Groups are never included (:func:`export_dataset`). The archive
    is written beside ``output_path`` and moved into place only once
    rewritten.

    Args:
        label: The set.
        output_path: The ``.aiida`` file; ``<output_path>.params/`` (block
            sidecars) and ``<output_path>.README.md`` are written beside it.
        profile: The AiiDA profile (default: the configured one).
        force: Write even when live points have no verdict.
        include_peratom: Keep the ``peratom.<key>`` nodes (dropped by default).
        keep_hpc_paths: Do not anonymise (module docstring).

    Raises:
        LookupError: No such set.
        FileExistsError: ``output_path`` exists.
        UncheckedPointsError: A live point has no live verdict (unless ``force``).
        ExportClosureError: The closure holds a CalcJob of another set.
        ArchiveFormatError: aiida-core wrote an archive layout this module
            does not know how to rewrite.
    """
    from aiida import orm  # pylint: disable=import-outside-toplevel

    _load_profile(profile)
    group = resolve_set(label)
    if not force:
        unchecked = _unchecked(group)
        if unchecked:
            raise UncheckedPointsError(f'set {label!r}', unchecked)
    analyses = _analysis_groups(label)
    start = _start_pks(group, include_peratom=include_peratom)
    start |= {node.pk for g in analyses for node in g.nodes}
    _check_closure(label, group, start)
    calcs = set_calcjobs(group)
    members = {calc.pk for calc in calcs}
    output_path = Path(output_path)
    sidecars = sidecar_documents(calcs)

    def readme(size: int) -> str:
        return _readme_text(output_path.name, size, title=f'Set `{group.label}`',
                            calcs=calcs, include_peratom=include_peratom,
                            keep_hpc_paths=keep_hpc_paths)

    counts = _write_archive([group, *analyses,
                             *(orm.load_node(pk) for pk in sorted(start - members))],
                            output_path, anonymise=not keep_hpc_paths,
                            drop_peratom=not include_peratom, readme=readme,
                            sidecars=sidecars)
    _log_size(output_path, f'set {label!r} ({len(start)} start nodes, {counts})')
    return output_path


#: The calcfunctions that make, rule or withdraw a point's verdict.
_VERDICT_FUNCTIONS = ('check_integrity', 'apply_ruling', 'withdraw_ruling')


def export_dataset(name: str, output_path: Path, *, profile: Optional[str] = None,
                   force: bool = False, include_peratom: bool = False,
                   keep_hpc_paths: bool = False) -> Path:
    """Archive the dataset ``friction2d/datasets/<name>`` on its own.

    The entities are its Group, its table Dict and summaries, and the verdict
    calcfunctions of every member point (``check_integrity``,
    ``apply_ruling``, ``withdraw_ruling``), so the far side can select each
    point's verdict; the closure adds the points, their CalcJobs and inputs,
    across every set the dataset was built from. Gated, anonymised,
    per-atom-free and documented (``<output_path>.README.md``) like
    :func:`export_archive`.

    Raises:
        LookupError: No such dataset.
        UncheckedPointsError: A member point has no live verdict (unless ``force``).
    """
    from aiida import orm  # pylint: disable=import-outside-toplevel

    _load_profile(profile)
    found = orm.Group.collection.find(filters={'label': f'{DATASET_PREFIX}{name}'})
    if not found:
        raise LookupError(f'no dataset {name!r} ({DATASET_PREFIX}{name})')
    group = found[0]
    points = list({node.creator.inputs.point.pk: node.creator.inputs.point
                   for node in group.nodes
                   if node.creator is not None and 'point' in node.creator.inputs}.values())
    if not force:
        unchecked = _without_verdict(points)
        if unchecked:
            raise UncheckedPointsError(f'dataset {name!r}', unchecked)
    functions = _reachable_functions({p.pk for p in points}, _VERDICT_FUNCTIONS)
    output_path = Path(output_path)
    calcs = list({p.creator.pk: p.creator for p in points}.values())

    def readme(size: int) -> str:
        return _readme_text(output_path.name, size, title=f'Dataset `{group.label}`',
                            calcs=calcs, include_peratom=include_peratom,
                            keep_hpc_paths=keep_hpc_paths, sidecars=False)

    _write_archive([group, *(orm.load_node(pk) for pk in sorted(functions))], output_path,
                   anonymise=not keep_hpc_paths, drop_peratom=not include_peratom,
                   readme=readme)
    _log_size(output_path, f'dataset {name!r}')
    return output_path


# ---------------------------------------------------------------------------
# Provenance bundle and import
# ---------------------------------------------------------------------------

_BUNDLE_EXACT = ('provenance.json', 'provenance/config.json', 'provenance/settings.yaml')
_BUNDLE_PREFIXES = ('provenance/cif/', 'provenance/potentials/')


def export_provenance_bundle(run, dest: Path) -> List[Path]:
    """Write ONE run's provenance files into ``dest``, and nothing else.

    ``run`` is a deck CalcJob (its ``provenance`` input is read) or a node
    with the provenance files in its own repository. Exactly
    ``provenance.json``, ``provenance/config.json``, ``provenance/settings.yaml``,
    ``provenance/cif/*`` and ``provenance/potentials/*``, at their stored
    paths -- the GUI's "download provenance" action. Returns the paths, sorted.
    """
    from aiida import orm  # pylint: disable=import-outside-toplevel

    node = run.inputs.provenance if isinstance(run, orm.CalcJobNode) else run
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for root, _dirs, files in node.base.repository.walk():
        for name in files:
            rel = (root / name).as_posix()
            if rel not in _BUNDLE_EXACT and not rel.startswith(_BUNDLE_PREFIXES):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(node.base.repository.get_object_content(rel, mode='rb'))
            written.append(out)
    logger.info('Wrote %d provenance file(s) to %s', len(written), dest)
    return sorted(written)


def import_archive(archive_path: Path, *, profile: Optional[str] = None) -> int:
    """Import an archive into the current profile; returns the new-node count.

    ``create_group=False``: the archive carries its own ``friction2d/<label>``
    Group, so aiida's default import group would only duplicate it on every
    import. The count is the profile's node count across the call
    (``aiida.tools.archive.import_archive`` returns a group pk, not a count).
    """
    from aiida.orm import Node, QueryBuilder  # pylint: disable=import-outside-toplevel
    from aiida.tools.archive import import_archive as _import  # pylint: disable=import-outside-toplevel

    _load_profile(profile)
    before = QueryBuilder().append(Node).count()
    _import(Path(archive_path), create_group=False)
    n_new = QueryBuilder().append(Node).count() - before
    logger.info('Imported %d new node(s) from %s', n_new, archive_path)
    return n_new
