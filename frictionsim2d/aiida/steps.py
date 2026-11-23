"""Tracked steps over imported points: integrity verdicts and user rulings.

* :func:`check_integrity` -- a calcfunction: one point's verdict from the
  ported checker, :func:`frictionsim2d.postprocessing.integrity.check_files`, on the
  point's stored files. The verdict is ``IntegrityVerdict.as_dict()`` plus
  ``thresholds_version``, ``code_version`` and ``point_uuid``; the thresholds
  are the integrity module's own (dimensionless, nothing rescaled here).
* :func:`check_group` -- runs it on every live point of a set that has no
  live verdict yet; :func:`~frictionsim2d.aiida.ingest.import_root` calls it.
* :func:`recheck_set` -- ``check_group(recheck=True)`` then the rulings
  pass: re-checks the points checked under other integrity settings.
* :func:`apply_ruling` -- a calcfunction: a verdict as one user ruling decides
  it (:func:`frictionsim2d.postprocessing.integrity_rulings.rule_verdict`, the rulings
  layer's own decision).
* :func:`apply_rulings` -- the idempotent rulings pass: reads the set's
  rulings files into its ruling ledger (:mod:`frictionsim2d.aiida.rulings`) and gives
  every live verdict the ruling the ledger calls for.
* :func:`latest_verdict` -- the verdict a :class:`~frictionsim2d.aiida.rulings.Policy`
  selects for a point, by declared version (never ctime); provisional or
  not is :func:`is_provisional`.
* :func:`summarise` -- a calcfunction: one point's ``RunSummary`` with units;
  :func:`postprocess_group` runs it on every point whose selected verdict is
  ``complete``.
* :func:`build_dataset` -- a Group ``friction2d/datasets/<name>`` of the
  selected summaries plus one table Dict naming the policy.
* :func:`delete_simulation_set`, :func:`clear_all_nodes` -- delete one set
  (keeping whatever another set still holds) or every friction2d node and
  Group; :func:`delete_dataset` deletes one dataset.

Every Dict made only to feed a step (``context``, ``settings``, ``ruling``,
``options``, ``parse``)
carries the extra ``friction2d_step_input``, so deleting a set or clearing
the store can collect them once their step is gone.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

from aiida import orm
from aiida.engine import calcfunction

from .. import __version__ as CODE_VERSION
from .rulings import Policy, select

logger = logging.getLogger(__name__)

__all__ = ['CODE_VERSION', 'DATASET_PREFIX', 'DatasetDeletion', 'RecheckReport', 'RulingReport',
           'apply_ruling', 'apply_rulings', 'build_dataset', 'check_group', 'check_integrity',
           'delete_dataset', 'is_provisional', 'latest_summaries', 'latest_verdict',
           'latest_verdicts', 'normalise_options', 'options_hash', 'postprocess_group',
           'recheck_set', 'rule_group', 'summarise', 'summary_options', 'withdraw_ruling']

#: Marks every Dict made only as a step's input (value: what it is).
STEP_INPUT_EXTRA = 'friction2d_step_input'
#: ``'verdict'``, ``'ruled_verdict'`` or ``'withdrawn_verdict'`` on a step's
#: output Dict.
KIND_EXTRA = 'friction2d_kind'
POINT_UUID_EXTRA = 'friction2d_point_uuid'
#: ``f'{verdict uuid}|{sha256 of the canonical ruling}'`` on a ruled verdict.
RULING_KEY_EXTRA = 'friction2d_ruling_key'
#: The uuid of the checker verdict a ruled or withdrawn verdict stands for.
VERDICT_UUID_EXTRA = 'friction2d_verdict_uuid'
#: On a set Group: the ruling ledger, ``{point id: {source path: entry}}``
#: (:mod:`frictionsim2d.aiida.rulings`): every ruling of every rulings file, the
#: ones not in force included. Every rulings pass reconciles the store with it.
LEDGER_EXTRA = 'friction2d_rulings_ledger'
#: The Group extras the ledger replaced: read once to seed the ledger of a
#: set ruled before it existed, then deleted (and dropped from archives).
_OLD_EXTRAS = ('friction2d_pending_rulings', 'friction2d_ruling_conflicts')
#: The ' (2)' an anonymised archive appends to a repeated basename.
_ARCHIVED_SUFFIX = re.compile(r' \(\d+\)$')
#: True on a verdict checked while its stage-2 status was unknown (a
#: ``_stage2`` failure in its run or (family, size) group): re-checked next time.
PROVISIONAL_EXTRA = 'friction2d_provisional'
#: ``'user'`` on a ruling Dict: a human, not the checker, decided it.
AUTHOR_EXTRA = 'friction2d_author'
#: The uuid of the verdict a summary was made under (not
#: ``friction2d_verdict_uuid``: that one marks a verdict's successors).
SUMMARY_VERDICT_EXTRA = 'friction2d_summary_verdict'
#: A summary's ``options_hash`` (sha256 of its canonical ``options``).
OPTIONS_HASH_EXTRA = 'friction2d_options_hash'
#: The dataset name on a dataset's table Dict.
DATASET_EXTRA = 'friction2d_dataset'
#: Datasets are the Groups ``friction2d/datasets/<name>``.
DATASET_PREFIX = 'friction2d/datasets'

#: Statuses with no frames in the DataReader: the checker gets no window.
_NO_FRAMES = ('no_output', 'unreadable', 'not_reached')


# ---------------------------------------------------------------------------
# check_integrity
# ---------------------------------------------------------------------------

def _materialise(node: Optional[orm.SinglefileData], directory: Path) -> Optional[Path]:
    """``node``'s file under ``directory`` (by its filename), or None."""
    if node is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / node.filename
    path.write_bytes(node.get_content(mode='rb'))
    return path


def _window_name(point: orm.FolderData) -> Optional[str]:
    """The window-tier file of a point node (``results/<name>``), or None."""
    # pylint: disable=import-outside-toplevel
    from ..postprocessing.read_data import tier_of
    from .ingest import _INDENT_RE  # pylint: disable=protected-access

    repo = point.base.repository
    if 'results' not in repo.list_object_names():
        return None
    names = [n for n in sorted(repo.list_object_names('results'))
             if tier_of(n) == 'window' and not _INDENT_RE.match(n)]
    return f'results/{names[0]}' if names else None


def _window(point: orm.FolderData, directory: Path, family: str, n_layers: int):
    """The point's window frame, read the way the DataReader reads it."""
    from .calcjob import _read_frame  # pylint: disable=import-outside-toplevel,protected-access

    name = _window_name(point)
    if name is None:
        return None
    path = directory / Path(name).name
    path.write_bytes(point.base.repository.get_object_content(name, mode='rb'))
    return _read_frame(path, family, n_layers)


@calcfunction
def check_integrity(point: orm.FolderData, snapshot: Optional[orm.SinglefileData] = None,
                    reference: Optional[orm.SinglefileData] = None,
                    context: Optional[orm.Dict] = None,
                    settings: Optional[orm.Dict] = None) -> orm.Dict:
    """One point's integrity verdict (``integrity.check_files``).

    ``context``: ``status`` (the stage-2 status the DataReader would give:
    the parser's, after legacy truncation and ``not_reached``), ``family``,
    ``n_layers``, ``T_target`` [K], ``layers`` (layer name -> atom types),
    ``multi_speed`` and ``writes_snapshot``. ``settings``: an
    ``IntegritySettings`` dump. ``reference`` is the bond reference the
    DataReader would use, for every status (it gives ``n_heat``).
    """
    from ..core.config import IntegritySettings  # pylint: disable=import-outside-toplevel
    from ..postprocessing.integrity import check_files  # pylint: disable=import-outside-toplevel

    if context is None or settings is None:
        raise ValueError('check_integrity needs `context` and `settings`')
    ctx = context.get_dict()
    thresholds = IntegritySettings(**settings.get_dict())
    status = ctx.get('status') or point.base.attributes.get('status')
    family, n_layers = ctx['family'], int(ctx['n_layers'])
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        window = (None if status in _NO_FRAMES
                  else _window(point, tmp_dir, family, n_layers))
        verdict = check_files(
            status, window, family, n_layers, float(ctx['T_target']),
            _materialise(snapshot, tmp_dir / 'final'),
            _materialise(reference, tmp_dir / 'reference'),
            {name: set(types) for name, types in ctx['layers'].items()},
            thresholds, multi_speed=bool(ctx['multi_speed']),
            writes_snapshot=bool(ctx['writes_snapshot']))
    return orm.Dict({**verdict.as_dict(), 'thresholds_version': thresholds.version,
                     'code_version': CODE_VERSION,
                     'point_uuid': point.base.attributes.get('point_uuid')})


# ---------------------------------------------------------------------------
# What a point's check needs, from the graph
# ---------------------------------------------------------------------------

def _live(node) -> bool:
    return not node.base.extras.get('superseded', False)


def _creator(point: orm.Node) -> orm.CalcJobNode:
    return point.creator


class _RunView:  # pylint: disable=too-few-public-methods
    """What every point of one run shares, read from the run's live CalcJobs:
    its provenance record, decks, includes, logs and structures, laid out in
    ``directory`` the way the run directory holds them."""

    def __init__(self, run_uuid: str, directory: Path) -> None:
        # pylint: disable=import-outside-toplevel
        from .calcjob import _provenance_input  # pylint: disable=protected-access
        from .store import CALCJOB_PROCESS_TYPE, RUN_UUID_EXTRA

        self.run_dir = directory / 'run' / 'L'
        calcs = [c for c in orm.QueryBuilder().append(
            orm.CalcJobNode, filters={'process_type': CALCJOB_PROCESS_TYPE,
                                      f'extras.{RUN_UUID_EXTRA}': run_uuid}
        ).all(flat=True) if _live(c)]
        self.lammps_dir = self.run_dir / 'lammps'
        self.data_dir = self.run_dir / 'data'
        self.record: Dict[str, Any] = {}
        self.speeds = set()
        self.has_snapshots = False
        self.structures: Dict[Path, orm.SinglefileData] = {}
        for calc in calcs:
            if not self.record:
                self.record = _provenance_input(calc)[1]
            self._lay_out(calc)
            outputs = calc.outputs
            if 'points' in outputs:
                self.speeds |= {float(p.base.attributes.get('speed'))
                                for p in outputs.points.values() if _live(p)}
            self.has_snapshots = self.has_snapshots or (
                'snapshots' in outputs and bool(dict(outputs.snapshots.items())))
        self._writes: Optional[bool] = None

    def _lay_out(self, calc: orm.CalcJobNode) -> None:
        """The calc's deck, includes and log in ``lammps/``; its structures
        (in and out) in ``data/`` (``*.data``) or ``build/`` (``*.lmp``)."""
        inputs, outputs = calc.inputs, calc.outputs
        files = [inputs.deck] if 'deck' in inputs else []
        if 'includes' in inputs:
            files += list(inputs.includes.values())
        if 'log' in outputs:
            files.append(outputs.log)
        for node in files:
            _materialise(node, self.lammps_dir)
        structures = list(inputs.structures.values()) if 'structures' in inputs else []
        if 'structures' in outputs:
            structures += list(outputs.structures.values())
        build = (self.run_dir if _build_level(calc) == 'run' else self.run_dir.parent) / 'build'
        for node in structures:
            where = build if node.filename.endswith('.lmp') else self.data_dir
            path = where / node.filename
            if path not in self.structures:
                _materialise(node, where)
                self.structures[path] = node

    @property
    def writes_snapshot(self) -> bool:
        """``integrity.writes_end_snapshot`` over the stored decks and logs."""
        if self._writes is None:
            from ..postprocessing.integrity import writes_end_snapshot  # pylint: disable=import-outside-toplevel
            self._writes = self.has_snapshots or writes_end_snapshot(self.lammps_dir)
        return self._writes


def _build_level(calc: orm.CalcJobNode) -> str:
    """``'run'`` or ``'parent'``: where the run's ``build/`` sat
    (``parameters['build_level']``; before it was recorded, the family's
    layout: ``L{n}/../build`` for the AFM-like families)."""
    from ..postprocessing.integrity import AFM_LIKE  # pylint: disable=import-outside-toplevel

    params = calc.inputs.parameters.get_dict()
    level = params.get('build_level')
    if level in ('run', 'parent'):
        return level
    return 'parent' if params.get('family') in AFM_LIKE else 'run'


def _point_inputs(point: orm.FolderData, run: _RunView, status: str) -> Dict[str, Any]:
    """``check_integrity``'s inputs for ``point`` at stage-2 ``status``
    (``context`` as a plain dict)."""
    # pylint: disable=import-outside-toplevel
    from ..postprocessing.integrity import bond_reference, resolve_layers
    from ..run.naming import parse_run_path

    calc = _creator(point)
    params = calc.inputs.parameters.get_dict()
    attrs = point.base.attributes.all
    family, n_layers = params['family'], int(params['n_layers'])
    inputs: Dict[str, Any] = {'point': point}
    key = _output_key(calc, point)
    if 'snapshots' in calc.outputs and key in calc.outputs.snapshots:
        inputs['snapshot'] = calc.outputs.snapshots[key]
    # Every status, as `check_point` does: the reference gives `n_heat`.
    ref_path = bond_reference(
        SimpleNamespace(run_dir=run.run_dir, data_dir=run.data_dir), family,
        SimpleNamespace(load_kind=attrs.get('load_kind'), load=attrs.get('load')))
    if ref_path is not None:
        inputs['reference'] = run.structures[ref_path]
    layers = resolve_layers(SimpleNamespace(lammps_dir=run.lammps_dir), run.record,
                            ref_path, n_layers, family=family)
    inputs['context'] = {
        'status': status, 'family': family, 'n_layers': n_layers,
        'T_target': float(parse_run_path(params['rel_path']).temperature),
        'layers': {name: sorted(int(t) for t in types) for name, types in layers.items()},
        'multi_speed': len(run.speeds) > 1,
        'writes_snapshot': run.writes_snapshot,
    }
    return inputs


def _output_key(calc: orm.CalcJobNode, point: orm.Node) -> str:
    """The ``points.<key>`` key under which ``calc`` output ``point``."""
    for key, node in calc.outputs.points.items():
        if node.pk == point.pk:
            return key
    raise KeyError(f'point {point.uuid} is not an output of CalcJob {calc.pk}')


def _step_input(content: Dict[str, Any], kind: str) -> orm.Dict:
    node = orm.Dict(content).store()
    node.base.extras.set(STEP_INPUT_EXTRA, kind)
    return node


# ---------------------------------------------------------------------------
# The set's points and verdicts
# ---------------------------------------------------------------------------

def _set_group(label: str) -> orm.Group:
    from .ingest import group_label  # pylint: disable=import-outside-toplevel
    return orm.Group.collection.get(label=group_label(label))


def _live_points(group: orm.Group) -> List[orm.FolderData]:
    """Every live point node of the set's live CalcJobs, in pk order."""
    qb = orm.QueryBuilder().append(orm.Group, filters={'id': group.pk}, tag='g')
    qb.append(orm.CalcJobNode, with_group='g', tag='c', project='extras.superseded')
    qb.append(orm.FolderData, with_incoming='c', tag='p',
              edge_filters={'label': {'like': 'points__%'}},
              project=['id', 'extras.superseded'])
    pks = {pk for calc_dead, pk, dead in qb.all() if not calc_dead and not dead}
    return [orm.load_node(pk) for pk in sorted(pks)]


def _live_verdicts(points: List[orm.FolderData]) -> Dict[int, List[orm.Dict]]:
    """point pk -> its live ``check_integrity`` verdicts, oldest first."""
    out: Dict[int, List[orm.Dict]] = {p.pk: [] for p in points}
    if not points:
        return out
    qb = orm.QueryBuilder().append(
        orm.FolderData, filters={'id': {'in': sorted(out)}}, tag='p', project='id')
    qb.append(orm.CalcFunctionNode, with_incoming='p', tag='f',
              filters={'attributes.process_label': 'check_integrity'})
    qb.append(orm.Dict, with_incoming='f', project='*')
    qb.order_by({orm.Dict: {'id': 'asc'}})
    for pk, verdict in qb.all():
        if _live(verdict):
            out[pk].append(verdict)
    return out


def _strict_settings(warnings: Optional[List[str]], fallback: str,
                     settings_file: Optional[Path] = None):
    """The ``GlobalSettings`` of the settings file in effect (``settings_file``
    when given, else the per-user one), validated strictly; None without a
    file, or -- with ``f'settings file <path> unreadable (<error>);
    {fallback}'`` in ``warnings``, or logged as a WARNING when ``warnings``
    is None -- when it cannot be read.

    ``load_settings`` skips an unreadable file silently (a log line), so the
    file it would use (``settings_origin``) is read and validated here.
    """
    # pylint: disable=import-outside-toplevel
    import yaml

    from ..core.config import GlobalSettings, settings_origin
    path = settings_origin(settings_file)
    if path is None:
        return None
    try:
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        return GlobalSettings.model_validate(data)
    except Exception as exc:  # pylint: disable=broad-except
        line = f'settings file {path} unreadable ({type(exc).__name__}: {exc}); {fallback}'
        if warnings is None:
            logger.warning('%s', line)
        else:
            warnings.append(line)
        return None


def _settings_dump(warnings: Optional[List[str]] = None,
                   settings_file: Optional[Path] = None) -> Dict[str, Any]:
    """``settings.integrity`` of the settings file in effect (see
    :func:`_strict_settings`); the defaults, with a warning in ``warnings``,
    when that file cannot be read."""
    from ..core.config import IntegritySettings  # pylint: disable=import-outside-toplevel
    default = IntegritySettings()
    settings = _strict_settings(
        warnings, f'integrity uses the default thresholds v{default.version}',
        settings_file)
    return (default if settings is None else settings.integrity).model_dump(mode='json')


def _run_uuid(point: orm.Node) -> Optional[str]:
    from .store import RUN_UUID_EXTRA  # pylint: disable=import-outside-toplevel
    return _creator(point).base.extras.get(RUN_UUID_EXTRA)


def _size_group(point: orm.Node) -> Tuple[str, Any]:
    """``(family, (size_x, size_y))``: the legacy truncation's comparison group."""
    from ..run.naming import parse_run_path  # pylint: disable=import-outside-toplevel
    attrs = point.base.attributes.all
    tokens = parse_run_path(attrs['rel_path'])
    return attrs['family'], (tokens.size_x, tokens.size_y)


def _pid(point: orm.Node) -> str:
    """The point's ruling id (``integrity_rulings.format_point_id``)."""
    from ..postprocessing.integrity_rulings import format_point_id  # pylint: disable=import-outside-toplevel
    attrs = point.base.attributes.all
    return format_point_id(attrs['rel_path'], attrs['load_kind'], attrs['load'],
                           attrs['angle'], attrs['speed'])


def _window_rows(point: orm.FolderData, directory: Path) -> int:
    """Rows of the point's window tier; 0 without one (`RunFrames.window`)."""
    params = _creator(point).inputs.parameters.get_dict()
    directory.mkdir(parents=True, exist_ok=True)
    frame = _window(point, directory, params['family'], int(params['n_layers']))
    return 0 if frame is None else len(frame)


def _stage2(points: List[orm.FolderData], directory: Path,
            problems: List[str]) -> Tuple[Dict[int, str], Set[int]]:
    """``(pk -> status, unknown pks)``: the status the DataReader gives each
    point before stage 3 -- the parser's, then the legacy truncation over the
    set's points by (family, size) (`run_status.legacy_truncated`), then
    `mark_not_reached` over each run's configured chains, in the DataReader's
    order.

    A run whose part of this fails is left out of the truncation's longest
    windows, with one entry in ``problems``. Its points, and every point of a
    (family, size) group it belongs to, are *unknown*: their status is the
    parser's, and it must not replace a stored verdict.
    """
    # pylint: disable=import-outside-toplevel
    from ..core.observables import expected_tier_rows
    from ..postprocessing.run_status import (
        UNWINDOWED_FAMILIES, expected_chains, legacy_truncated, mark_not_reached,
        point_key)
    from .calcjob import _provenance_input  # pylint: disable=protected-access

    parser = {p.pk: p.base.attributes.get('status') for p in points}
    status = dict(parser)
    by_run: Dict[Optional[str], List[orm.FolderData]] = {}
    for point in points:
        by_run.setdefault(_run_uuid(point), []).append(point)
    broken: Dict[Optional[str], str] = {}

    def fail(run_uuid, exc):
        if run_uuid not in broken:
            broken[run_uuid] = f'{type(exc).__name__}: {exc}'

    configs: Dict[Optional[str], Any] = {}
    framed, rows = [], []
    for run_uuid, run_points in by_run.items():
        try:
            configs[run_uuid] = _provenance_input(_creator(run_points[0]))[0]
            for point in run_points:
                if status[point.pk] in _NO_FRAMES:
                    continue
                family, size = _size_group(point)
                rows.append([family, size, 0,
                             bool(configs[run_uuid]
                                  and expected_tier_rows(configs[run_uuid], family)),
                             status[point.pk]])
                framed.append((run_uuid, point))
        except Exception as exc:  # pylint: disable=broad-except
            fail(run_uuid, exc)
    # Window rows are read only where some point is judged at all.
    judged = {(r[0], r[1]) for r in rows if not r[3] and r[4] == 'complete'
              and r[0] not in UNWINDOWED_FAMILIES}
    for (run_uuid, point), row in zip(framed, rows):
        if (row[0], row[1]) in judged and run_uuid not in broken:
            try:
                row[2] = _window_rows(point, directory)
            except Exception as exc:  # pylint: disable=broad-except
                fail(run_uuid, exc)
    # A broken run's rows are unknown, not zero: they must not set a longest.
    kept = [i for i, (run_uuid, _p) in enumerate(framed) if run_uuid not in broken]
    for j in legacy_truncated([tuple(rows[i]) for i in kept]):
        status[framed[kept[j]][1].pk] = 'truncated'

    for run_uuid, run_points in by_run.items():
        if run_uuid in broken:
            continue
        try:
            keyed = {point_key(*(p.base.attributes.get(k) for k in
                                 ('load_kind', 'load', 'angle', 'speed'))): p.pk
                     for p in run_points}
            current = {key: status[pk] for key, pk in keyed.items()}
            family = run_points[0].base.attributes.get('family')
            for chain in expected_chains(configs[run_uuid], family) or []:
                mark_not_reached(chain, current)
            status.update({pk: current[key] for key, pk in keyed.items()})
        except Exception as exc:  # pylint: disable=broad-except
            fail(run_uuid, exc)

    unknown_groups = set()
    for run_uuid, why in broken.items():
        logger.warning('stage-2 status of run %s failed: %s', run_uuid, why)
        problems.append(f'run {run_uuid}: stage-2 status failed ({why}); its points '
                        f'and their (family, size) groups keep their verdicts')
        for point in by_run[run_uuid]:
            try:
                unknown_groups.add(_size_group(point))
            except ValueError:
                pass
    unknown = {p.pk for run_uuid in broken for p in by_run[run_uuid]}
    for point in points:
        if point.pk not in unknown:
            try:
                if _size_group(point) in unknown_groups:
                    unknown.add(point.pk)
            except ValueError:
                pass
    for pk in unknown:
        status[pk] = parser[pk]
    return status, unknown


def _uuid_of(node: Optional[orm.Node]) -> Optional[str]:
    return None if node is None else node.uuid


def _unchanged(verdict: orm.Dict, inputs: Dict[str, Any], point: orm.Node) -> bool:
    """Whether ``verdict``'s check had the context, snapshot and reference
    ``inputs`` would give it now. A context from before ``status`` was
    recorded stands for the parser status; a provisional one (checked while
    stage 2 was unknown) is always stale."""
    func = verdict.creator
    old = func.inputs.context.get_dict()
    if old.get('status_provisional'):
        return False                       # checked while stage 2 was unknown
    old.setdefault('status', point.base.attributes.get('status'))
    return (old == inputs['context']
            and all(_uuid_of(func.inputs[k] if k in func.inputs else None)
                    == _uuid_of(inputs.get(k)) for k in ('snapshot', 'reference')))


def _scope(points: List[orm.FolderData], new: List[orm.FolderData],
           problems: List[str]) -> List[orm.FolderData]:
    """The points whose stage-2 context this import may have changed: those
    of every run with a new point, and of every (family, size) group holding
    one."""
    runs = {_run_uuid(p) for p in new}
    groups = set()
    for point in new:
        try:
            groups.add(_size_group(point))
        except ValueError as exc:
            problems.append(f'point {point.uuid}: {exc}')

    def touched(point) -> bool:
        if _run_uuid(point) in runs:
            return True
        try:
            return _size_group(point) in groups
        except ValueError:
            return False
    return [p for p in points if touched(p)]


def _stale_settings(verdict: orm.Dict, current: Dict[str, Any]) -> bool:
    """Whether ``verdict`` was checked under integrity settings other than
    ``current``: another ``thresholds_version``, or other stored content."""
    if verdict.get('thresholds_version') != current.get('version'):
        return True
    func = verdict.creator
    return 'settings' not in func.inputs or func.inputs.settings.get_dict() != current


def check_group(group: orm.Group, *, settings: Optional[Dict[str, Any]] = None,
                failures: Optional[List[Tuple[str, str]]] = None,
                warnings: Optional[List[str]] = None,
                rechecked: Optional[List[Tuple[str, str, str]]] = None,
                settings_file: Optional[Path] = None,
                recheck: bool = False) -> Dict[str, int]:
    """Give every live point of ``group`` the verdict its current context
    calls for; return the new verdicts' status counts.

    A point with no live verdict is checked. So is every point of the scope
    this import touched (:func:`_scope`) whose stage-2 context -- status
    (:func:`_stage2`, over every live point of the set), layers,
    ``multi_speed``, ``writes_snapshot``, snapshot and reference -- differs
    from its live verdict's: the old verdict is superseded with its ruled and
    withdrawn successors, and ``(point id, old status, new status)`` goes to
    ``rechecked`` when the status changed. ``settings`` is an
    ``IntegritySettings`` dump (default: that of ``settings_file``, else of
    the per-user settings file; a fallback to the defaults goes to
    ``warnings``). A
    point whose check raises goes to ``failures`` as ``(point uuid,
    message)`` and the others go on.

    With ``recheck``, every live point whose live verdict was checked under
    other integrity settings (:func:`_stale_settings`: another
    ``thresholds_version``, or other stored settings content) is checked
    too, and superseded as a context change is -- ``aiida recheck``. The
    rulings pass that follows (:func:`rule_group`) re-applies every ruling
    from the set's ledger to the new verdicts, whatever file it came from.
    """
    points = _live_points(group)
    verdicts = _live_verdicts(points)
    # A point with no verdict, or whose verdict was provisional.
    new = [p for p in points if not verdicts[p.pk]
           or verdicts[p.pk][-1].base.extras.get(PROVISIONAL_EXTRA, False)]
    stale: Set[int] = set()
    if recheck:
        settings = settings or _settings_dump(warnings, settings_file=settings_file)
        stale = {p.pk for p in points
                 if verdicts[p.pk] and _stale_settings(verdicts[p.pk][-1], settings)}
    if not new and not stale:
        return {}
    problems: List[str] = []
    counts: Counter = Counter()
    settings_node = None
    runs: Dict[str, _RunView] = {}
    with TemporaryDirectory() as tmp:
        status, unknown = _stage2(points, Path(tmp) / 'rows', problems)
        scope = {p.pk for p in _scope(points, new, problems)} | stale
        for point in [p for p in points if p.pk in scope]:
            old = verdicts[point.pk]
            if point.pk in unknown and old:
                continue                    # stage 2 unknown: keep the verdict
            try:
                run_uuid = _run_uuid(point)
                if run_uuid not in runs:
                    runs[run_uuid] = _RunView(run_uuid, Path(tmp) / str(len(runs)))
                inputs = _point_inputs(point, runs[run_uuid], status[point.pk])
                if point.pk in unknown:
                    inputs['context']['status_provisional'] = True
                    problems.append(f'point {_pid(point)}: checked provisionally at '
                                    f'its parser status {status[point.pk]!r} '
                                    f'(stage 2 unknown); the next import re-checks it')
                elif old and point.pk not in stale and _unchanged(old[-1], inputs, point):
                    continue
                if settings_node is None:
                    settings_node = _step_input(
                        settings or _settings_dump(warnings, settings_file=settings_file),
                        'integrity_settings')
                inputs['context'] = _step_input(inputs['context'], 'integrity_context')
                verdict = check_integrity(**inputs, settings=settings_node)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception('Integrity check failed for point %s', point.uuid)
                if failures is not None:
                    failures.append((point.uuid, f'{type(exc).__name__}: {exc}'))
                continue
            verdict.base.extras.set_many({
                KIND_EXTRA: 'verdict', PROVISIONAL_EXTRA: point.pk in unknown,
                POINT_UUID_EXTRA: point.base.attributes.get('point_uuid')})
            counts[verdict['status']] += 1
            for dead in old:
                for successor in _successors(dead):
                    _supersede(successor)
                _supersede(dead)
            if old and rechecked is not None and old[-1]['status'] != verdict['status']:
                rechecked.append((_pid(point), old[-1]['status'], verdict['status']))
    if warnings is not None:
        warnings += problems
    return dict(counts)


# ---------------------------------------------------------------------------
# Rulings
# ---------------------------------------------------------------------------

def _verdict_of(content: Dict[str, Any]):
    """An ``IntegrityVerdict`` from a stored verdict's content."""
    from ..postprocessing.integrity import IntegrityVerdict  # pylint: disable=import-outside-toplevel
    names = {f.name for f in fields(IntegrityVerdict)}
    return IntegrityVerdict(**{k: v for k, v in content.items() if k in names})


def _ruling_of(content: Dict[str, Any]):
    from ..postprocessing.integrity_rulings import Ruling  # pylint: disable=import-outside-toplevel
    return Ruling(ruling=content['ruling'], note=content.get('note', ''),
                  date=content.get('date', ''), override=bool(content.get('override')))


@calcfunction
def apply_ruling(verdict: orm.Dict, ruling: orm.Dict) -> orm.Dict:
    """``verdict`` as the user's ``ruling`` decides it
    (``integrity_rulings.rule_verdict``): status, ``ruled_by_user``,
    ``overridden``, ``original_status``, ``note`` and ``date``. The verdict's
    other keys (versions, ``point_uuid``) are carried over."""
    from ..postprocessing.integrity_rulings import rule_verdict  # pylint: disable=import-outside-toplevel

    content = verdict.get_dict()
    ruled, why = rule_verdict(_verdict_of(content), _ruling_of(ruling.get_dict()))
    if ruled is None:
        raise ValueError(f'ruling does not apply: {why}')
    return orm.Dict({**content, **ruled.as_dict()})


@calcfunction
def withdraw_ruling(ruled: orm.Dict, verdict: orm.Dict) -> orm.Dict:
    """The checker's ``verdict`` back in force once the ruling that made
    ``ruled`` is gone from the rulings file: its content, stamped
    ``withdrawn_ruling=<ruled uuid>``."""
    return orm.Dict({**verdict.get_dict(), 'withdrawn_ruling': ruled.uuid})


@dataclass
class RulingReport:  # pylint: disable=too-many-instance-attributes
    """What one rulings pass (:func:`rule_group`) did, or would do (``dry_run``).

    ``applied`` counts ruled verdicts made where none was in force,
    ``replaced`` those made over another ruling, ``skipped`` the verdicts
    whose ruling in force already is the one the ledger calls for.
    ``withdrawn`` are the points whose ruling was withdrawn because no
    source rules them any more. ``unmatched`` are ruled point ids naming no
    live point of the set; ``not_quarantined`` those on a point that is not
    quarantined and that lack ``override: true`` or a note (or would pass a
    point with no data). ``overrides`` are ``(point id, original status,
    new status, note)`` of every live overridden verdict of the set, read
    from the store.

    ``conflicts`` are ``(point id, (source, ruling) in force or None, why,
    [(source, ruling) not applied])``: two or more sources rule the point
    with different decisions. ``pending`` are ``(point id, source,
    'missing')``: a ruling of a file that is gone, kept in the ledger and
    still counted. ``pending_withdrawn`` are ``(point id, source)`` dropped
    by ``--withdraw-pending``; ``retracted`` ``(point id, source)`` entries a
    file no longer makes. ``own_source`` is the set's own rulings file.
    """
    label: str
    rulings_path: str = ''
    own_source: str = ''
    dry_run: bool = False
    applied: int = 0
    replaced: int = 0
    skipped: int = 0
    unmatched: List[str] = field(default_factory=list)
    not_quarantined: List[str] = field(default_factory=list)
    withdrawn: List[str] = field(default_factory=list)
    overrides: List[Tuple[str, str, str, str]] = field(default_factory=list)
    conflicts: List[Tuple[str, Optional[Tuple[str, str]], str, List[Tuple[str, str]]]] = \
        field(default_factory=list)
    pending: List[Tuple[str, str, str]] = field(default_factory=list)
    pending_withdrawn: List[Tuple[str, str]] = field(default_factory=list)
    retracted: List[Tuple[str, str]] = field(default_factory=list)
    archived: List[Tuple[str, str]] = field(default_factory=list)
    changed: List[Tuple[str, str, str, str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def render(self) -> str:
        """The report as text."""
        lines = [f'{"DRY RUN: " if self.dry_run else ""}rulings for set {self.label!r} '
                 f'({self.rulings_path}): applied {self.applied}, replaced {self.replaced}, '
                 f'skipped {self.skipped}, withdrawn {len(self.withdrawn)}']
        lines += [f'  {line}' for line in self.changed_lines()]
        lines += [f'  override {pid}: {old} -> {new} ({note})'
                  for pid, old, new, note in self.overrides]
        lines += [f'  withdrawn: {pid}' for pid in self.withdrawn]
        lines += [f'  unmatched: {pid}' for pid in self.unmatched]
        lines += [f'  not quarantined: {pid}' for pid in self.not_quarantined]
        lines += [f'  conflict {line}' for line in self.conflict_lines()]
        lines += [f'  {line}' for line in self.pending_lines()]
        lines += [f'  WARNING {w}' for w in self.warnings]
        return '\n'.join(lines)

    def pending_lines(self) -> List[str]:
        """Each of ``pending``, ``pending_withdrawn`` and ``retracted`` as text."""
        lines = []
        for pid, src, reason in self.pending:
            if src == self.own_source:
                lines.append(f'ruling pending: {src} {reason}: {pid}; the set\'s own rulings '
                             f'file is unreachable, its ruling still counts')
            else:
                lines.append(f'ruling pending: {src} {reason}: {pid}; it still counts. '
                             f'Restore it with `FrictionSim2D aiida rule {self.label} '
                             f'--rulings {src}`, or drop it with `--withdraw-pending {src}`')
        lines += [f'pending ruling withdrawn: {pid} ({src})'
                  for pid, src in self.pending_withdrawn]
        lines += [f'ruling retracted by its file: {pid} ({src})' for pid, src in self.retracted]
        lines += [f'ruling archived: {pid} ({src}): it came with an archive and still '
                  f'counts; no file is read for it' for pid, src in self.archived]
        return lines

    def changed_lines(self) -> List[str]:
        """Each of changed as text: every point whose decision in force
        this pass changed."""
        return [f'ruling changed: {pid} {old} -> {new} ({src}, {why})'
                for pid, old, new, src, why in self.changed]

    def conflict_lines(self) -> List[str]:
        """Each of ``conflicts`` as text."""
        lines = []
        for pid, in_force, why, others in self.conflicts:
            head = (f'{pid}: {in_force[0]} rules {in_force[1]} (in force: {why})'
                    if in_force else f'{pid}: no ruling in force ({why})')
            lines.append('; '.join([head] + [f'{src} rules {ruling} (not applied)'
                                             for src, ruling in others]))
        return lines


def _canonical(content: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'))
                          .encode('utf-8')).hexdigest()


def _successors(verdict: orm.Dict) -> List[orm.Dict]:
    """The live ruled and withdrawn verdicts of ``verdict``, oldest first."""
    qb = orm.QueryBuilder().append(
        orm.Dict, filters={f'extras.{VERDICT_UUID_EXTRA}': verdict.uuid}, tag='d')
    qb.order_by({'d': {'id': 'asc'}})
    return [n for n in qb.all(flat=True) if _live(n)]


def _ruled_by_verdict(uuids: List[str]) -> Dict[str, List[orm.Dict]]:
    """verdict uuid -> its live ruled verdicts, in one query."""
    out: Dict[str, List[orm.Dict]] = {}
    if not uuids:
        return out
    qb = orm.QueryBuilder().append(orm.Dict, tag='d', filters={
        f'extras.{KIND_EXTRA}': 'ruled_verdict',
        f'extras.{VERDICT_UUID_EXTRA}': {'in': sorted(set(uuids))}})
    qb.order_by({'d': {'id': 'asc'}})
    for node in qb.all(flat=True):
        if _live(node):
            out.setdefault(node.base.extras.get(VERDICT_UUID_EXTRA), []).append(node)
    return out


def _supersede(node: orm.Node) -> None:
    node.base.extras.set_many({'superseded': True,
                               'superseded_at': datetime.now(timezone.utc).isoformat()})


def _tag(node: orm.Dict, kind: str, verdict: orm.Dict, **extra: Any) -> None:
    node.base.extras.set_many({KIND_EXTRA: kind, VERDICT_UUID_EXTRA: verdict.uuid,
                               POINT_UUID_EXTRA: verdict.get('point_uuid'), **extra})


def _was_ruled_quarantined(verdict: orm.Dict) -> bool:
    """Whether an earlier, now superseded, ruled verdict of ``verdict``'s
    point was ruled while the point was quarantined."""
    qb = orm.QueryBuilder().append(orm.Dict, filters={
        f'extras.{KIND_EXTRA}': 'ruled_verdict',
        f'extras.{POINT_UUID_EXTRA}': verdict.get('point_uuid'),
        'attributes.original_status': 'quarantined'})
    return any(not _live(n) for n in qb.all(flat=True))


def _refusal(pid: str, source: str, entry: Dict[str, Any],
             verdict: orm.Dict) -> Optional[str]:
    """The warning why the ledger ``entry`` of ``source`` cannot rule
    ``verdict`` (``decide``); None when it can."""
    from ..postprocessing.integrity_rulings import decide  # pylint: disable=import-outside-toplevel

    original = verdict['status']
    _status, why = decide(original, _ruling_of(entry))
    if why is None:
        return None
    if (original != 'quarantined' and not entry.get('override')
            and _was_ruled_quarantined(verdict)):
        return (f'integrity ruling no longer applies: {pid} ({source}): re-checked from '
                f'quarantined to {original}; the point is no longer quarantined, '
                f'so its ruling now needs `override: true` and a note')
    return f'integrity ruling ignored: {pid} ({source}): {why}'


def _matches_source(recorded: Optional[str], source: str, own_source: Optional[str]) -> bool:
    """Whether a ruling Dict's ``source`` (``recorded``) is the ledger's
    ``source``. A ruling Dict without one predates the field and came from
    the set's own file. A ledger source that is not absolute comes from an
    anonymised archive: a basename, maybe disambiguated as ``x.yaml (2)``,
    matched against the recorded source's basename."""
    recorded = recorded or own_source
    if recorded is None:
        return False
    if Path(source).is_absolute():
        return recorded == source
    return PurePosixPath(recorded).name == _ARCHIVED_SUFFIX.sub('', source)


def _in_force_is(node: orm.Dict, source: str, entry: Dict[str, Any],
                 own_source: Optional[str]) -> bool:
    """Whether the ruled verdict ``node`` was made from ``source``'s ``entry``."""
    from .rulings import CONTENT  # pylint: disable=import-outside-toplevel

    content = node.creator.inputs.ruling.get_dict()
    content['override'] = bool(content.get('override'))
    return (_matches_source(content.get('source'), source, own_source)
            and all(content.get(k, '') == entry[k] for k in CONTENT))


def _apply(pid: str, source: str, entry: Dict[str, Any], verdict: orm.Dict,
           own_source: Optional[str]) -> None:
    """A ruled verdict of ``verdict`` from ``source``'s ledger ``entry``; it
    supersedes the verdict's earlier ruled and withdrawn ones."""
    from .rulings import CONTENT  # pylint: disable=import-outside-toplevel

    content = {'point': pid, **{k: entry[k] for k in CONTENT}, 'source': source}
    node = _step_input({**content, 'own_file': source == own_source}, 'ruling')
    node.base.extras.set(AUTHOR_EXTRA, 'user')
    ruled = apply_ruling(verdict=verdict, ruling=node)
    _tag(ruled, 'ruled_verdict', verdict,
         **{RULING_KEY_EXTRA: f'{verdict.uuid}|{_canonical(content)}'})
    for older in _successors(verdict):
        if older.pk != ruled.pk:
            _supersede(older)


def _withdraw(report: RulingReport, pid: str, verdict: orm.Dict,
              ruled_nodes: List[orm.Dict], dry_run: bool) -> None:
    """Withdraw ``ruled_nodes``, the live ruled verdicts of ``verdict``."""
    if not dry_run:
        for ruled in ruled_nodes:
            back = withdraw_ruling(ruled=ruled, verdict=verdict)
            _tag(back, 'withdrawn_verdict', verdict)
            _supersede(ruled)
    if pid not in report.withdrawn:
        report.withdrawn.append(pid)


def _own_source(group: orm.Group, root: Optional[Path],
                warnings: List[str]) -> Optional[str]:
    """The set's own rulings file, resolved: under ``root`` (the root being
    imported) when given, else under the set's ``source_root``. None
    without one, or when it is not an absolute path (a set loaded from an
    anonymised archive): a relative path is never read from disk."""
    from ..postprocessing.integrity_rulings import RULINGS_FILE  # pylint: disable=import-outside-toplevel

    if root is not None:
        return str((Path(root) / RULINGS_FILE).resolve())
    stored = group.base.extras.get('source_root', None)
    if not stored:
        warnings.append('the set records no source_root: its own rulings file is not read')
        return None
    if not Path(stored).is_absolute():
        warnings.append(f'the set\'s source_root {stored!r} is not a path on this machine '
                        f'(an archive): its own rulings file is not read')
        return None
    return str((Path(stored) / RULINGS_FILE).resolve())


def _resolve_source(path: Path, own: Optional[str], known: Set[str]) -> str:
    """The source ``path`` names: the file, or ``DIR/integrity_rulings.yaml``
    for a directory -- also for one that does not exist when that is the
    source the ledger knows (an unmounted drive)."""
    from ..postprocessing.integrity_rulings import RULINGS_FILE  # pylint: disable=import-outside-toplevel

    path = Path(path)
    as_file, in_dir = str(path.resolve()), str((path / RULINGS_FILE).resolve())
    if own in (as_file, in_dir):
        return own
    if path.is_dir() or (in_dir in known and as_file not in known):
        return in_dir
    return as_file


def _load_source(source: str):
    """``load_rulings`` of ``source``; None when the file does not exist. A
    file with no YAML document at all (0 bytes, blank, only comments) is
    unreadable, not "no rulings": an editor that truncated it must not
    withdraw everything. ``[]`` says "no rulings"."""
    # pylint: disable=import-outside-toplevel
    import yaml

    from ..postprocessing.integrity_rulings import Ruling, load_rulings
    path = Path(source)
    if not path.is_file():
        return None
    try:
        empty = yaml.safe_load(path.read_text(encoding='utf-8')) is None
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        empty = False                          # load_rulings reports it
    if empty:
        return {'<file>': Ruling(error='the file holds no entries (empty); write `[]` '
                                       'to withdraw every ruling')}
    return load_rulings(path)


def _former_own(ledger, own: Optional[str], report: RulingReport) -> None:
    """Entries the own file made at a former root (``own_file`` at another
    path) rank as another file's from now on, at their old path: the new
    own file overrides them only where it rules a point. Reported once."""
    if own is None:
        return
    former = sorted({s for by_source in ledger.values() for s, e in by_source.items()
                     if s != own and e.get('own_file') and Path(s).is_absolute()})
    for source in former:
        for by_source in ledger.values():
            if source in by_source:
                by_source[source]['own_file'] = False
        report.warnings.append(f'former own rulings file {source} now ranks as another '
                               f'file: the own file {own} overrides it only where it '
                               f'rules a point')


def _update_ledger(ledger, report: RulingReport, own: Optional[str],
                   explicit: Optional[str], dropped: Optional[str]) -> None:
    """Read every source into ``ledger`` (:func:`.rulings.read_source`): the
    sources the ledger already holds (one that is gone is marked missing),
    then the set's own file, then ``explicit`` (``--rulings``), which is
    therefore seen last. A source that is not an absolute path (from an
    anonymised archive) is never read: it is marked ``archived``."""
    from .rulings import Clock, read_source  # pylint: disable=import-outside-toplevel

    clock = Clock()

    def read(source, rulings, is_own):
        for pid in read_source(ledger, source, rulings, own=is_own, now=clock(),
                               warnings=report.warnings):
            report.retracted.append((pid, source))

    for pid in sorted(ledger):
        for source, entry in sorted(ledger[pid].items()):
            if not Path(source).is_absolute() and entry.get('status') != 'archived':
                entry['status'] = 'archived'
                report.archived.append((pid, source))
    _former_own(ledger, own, report)
    others = sorted({s for by_source in ledger.values() for s in by_source
                     if Path(s).is_absolute()} - {own, explicit, dropped})
    for source in others:                       # a file that is gone: missing
        read(source, _load_source(source), False)
    if own is not None:
        if Path(own).parent.is_dir():
            # An absent file inside an existing campaign root means "no rulings".
            rulings = _load_source(own)
            read(own, {} if rulings is None else rulings, True)
        else:
            # An unreachable root (an unmounted drive) withdraws nothing.
            read(own, None, True)
            report.warnings.append(f'no ruling withdrawn: {own} is not reachable (its '
                                   f'directory {Path(own).parent} is missing)')
    if explicit is not None and explicit != own:
        rulings = _load_source(explicit)
        if rulings is None and not any(explicit in e for e in ledger.values()):
            report.warnings.append(f'{explicit} does not exist: nothing read')
        else:
            read(explicit, rulings, False)


def _seed_entry(content: Dict[str, Any], own_file: bool, when) -> Dict[str, Any]:
    from .rulings import stamp  # pylint: disable=import-outside-toplevel
    return {'ruling': content['ruling'], 'note': content.get('note', '') or '',
            'date': content.get('date', '') or '', 'override': bool(content.get('override')),
            'own_file': own_file, 'author': 'user', 'seen_at': stamp(when),
            'status': 'present'}


def _seed_ledger(group: orm.Group, own: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """The ledger of a set ruled before the ledger existed: every live ruled
    verdict's ruling (by its ruling Dict's ``source``; none means the own
    file), then every ruling of the old extras ``friction2d_pending_rulings``
    and ``friction2d_ruling_conflicts`` (their full ruling from the latest
    ruling Dict of that point and source, when one was ever applied)."""
    ledger: Dict[str, Dict[str, Any]] = {}
    points = _live_points(group)
    verdicts = _live_verdicts(points)
    live = _ruled_by_verdict([v.uuid for vs in verdicts.values() for v in vs])
    for point in points:
        for verdict in verdicts[point.pk]:
            for node in live.get(verdict.uuid, []):
                content = node.creator.inputs.ruling.get_dict()
                source = content.get('source') or own
                if source is not None:
                    entry = _seed_entry(content, source == own, node.ctime)
                    ledger.setdefault(_pid(point), {})[source] = {**entry, 'in_force': True}
    for key in _OLD_EXTRAS:
        for old in group.base.extras.get(key, []) or []:
            pid, source = old.get('point'), old.get('source')
            if not pid or not source or source in ledger.get(pid, {}):
                continue
            qb = orm.QueryBuilder().append(orm.Dict, tag='r', filters={
                f'extras.{STEP_INPUT_EXTRA}': 'ruling', 'attributes.point': pid,
                'attributes.source': source})
            qb.order_by({'r': {'id': 'desc'}})
            full = qb.first(flat=True)
            ledger.setdefault(pid, {})[source] = _seed_entry(
                full.get_dict() if full is not None else old, source == own,
                full.ctime if full is not None else datetime.now(timezone.utc))
    return ledger


def _record_refusals(report: RulingReport, pid: str, refusals: Dict[str, Optional[str]],
                     order: List[str], chosen: Optional[str], verdict: orm.Dict) -> None:
    """Warn about every entry ranked above ``chosen`` (all, without one)
    that does not apply: it is why ``chosen`` is in force. A lower-ranked
    entry that does not apply changes nothing and is not reported."""
    for source in order:
        if source == chosen:
            return
        line = refusals[source]
        if line is None:
            continue
        if verdict['status'] != 'quarantined' and pid not in report.not_quarantined:
            report.not_quarantined.append(pid)
        if line not in report.warnings:
            report.warnings.append(line)


def _reconcile(group: orm.Group, ledger, previous, report: RulingReport,
               own: Optional[str]) -> None:
    """Make every live verdict's ruling in force the one the ledger calls
    for (the first source of :func:`.rulings.ruling_order` that applies):
    apply it, replace another, or
    withdraw one no source makes any more; mark it ``in_force`` in the
    ledger; then fill the report's changes, conflicts, pending and
    overrides. ``previous`` is ``point id -> (source, ruling)`` in force
    before this pass. With ``report.dry_run`` nothing is made: the counts
    say what would be."""
    # pylint: disable=import-outside-toplevel
    from .rulings import conflicts, ruling_order, why_in_force

    points = _live_points(group)
    verdicts = _live_verdicts(points)
    by_id: Dict[str, List[orm.Dict]] = {}
    for point in points:
        by_id.setdefault(_pid(point), []).extend(verdicts[point.pk])
    live = _ruled_by_verdict([v.uuid for vs in by_id.values() for v in vs])
    for pid in sorted(set(by_id) | set(ledger) | set(previous)):
        entries = ledger.get(pid, {})
        pid_verdicts = by_id.get(pid, [])
        if not pid_verdicts:
            if entries:
                report.unmatched.append(pid)
            continue
        # A set loaded from an archive has no own file to read; the own
        # file's archived entry still ranks first.
        rank_own = own if own is not None else next(
            (s for s, e in sorted(entries.items())
             if e.get('own_file') and not Path(s).is_absolute()), None)
        order = ruling_order(entries, rank_own)
        chosen = None
        for verdict in pid_verdicts:
            refusals = {s: _refusal(pid, s, entries[s], verdict) for s in order}
            chosen = next((s for s in order if refusals[s] is None), None)
            _record_refusals(report, pid, refusals, order, chosen, verdict)
            in_force = live.get(verdict.uuid, [])
            if chosen is None:
                if in_force:
                    _withdraw(report, pid, verdict, in_force, report.dry_run)
            elif len(in_force) == 1 and _in_force_is(in_force[0], chosen, entries[chosen],
                                                     own):
                report.skipped += 1
            else:
                if not report.dry_run:
                    _apply(pid, chosen, entries[chosen], verdict, own)
                if in_force:
                    report.replaced += 1
                else:
                    report.applied += 1
        for source, entry in entries.items():
            if source == chosen:
                entry['in_force'] = True
            else:
                entry.pop('in_force', None)
        before = previous.get(pid)
        after = entries[chosen]['ruling'] if chosen is not None else 'none'
        if (before[1] if before else 'none') != after:
            report.changed.append((pid, before[1] if before else 'none', after,
                                   chosen or (before[0] if before else ''),
                                   why_in_force(entries, rank_own, chosen)))
        found = conflicts(entries, rank_own, chosen)
        if found is not None:
            report.conflicts.append((pid, *found))
        report.pending += [(pid, s, 'missing') for s in sorted(entries)
                           if entries[s].get('status') == 'missing']
    live = _ruled_by_verdict([v.uuid for vs in by_id.values() for v in vs])
    report.overrides = [
        (pid, node['original_status'], node['status'], node.get('note', ''))
        for pid, pid_verdicts in sorted(by_id.items()) for verdict in pid_verdicts
        for node in live.get(verdict.uuid, []) if node.get('overridden')]


def rule_group(group: orm.Group, rulings_path: Optional[Path] = None, *,
               withdraw_pending: Optional[Path] = None,
               dry_run: bool = False, root: Optional[Path] = None) -> RulingReport:
    """:func:`apply_rulings` on a loaded set Group; with ``dry_run``, the
    report of what it would do, and nothing written. ``root`` is the root
    being imported, when it is not yet the set's ``source_root`` (a dry run)."""
    from .ingest import GROUP_PREFIX  # pylint: disable=import-outside-toplevel
    from .rulings import drop_source  # pylint: disable=import-outside-toplevel

    report = RulingReport(label=group.label[len(GROUP_PREFIX) + 1:], dry_run=dry_run)
    own = _own_source(group, root, report.warnings)
    extras = group.base.extras.all
    stored = extras.get(LEDGER_EXTRA)
    if stored is None:
        ledger = _seed_ledger(group, own)
    else:
        ledger = json.loads(json.dumps(stored))             # a deep copy
    previous = {pid: next(((s, e['ruling']) for s, e in by_source.items()
                           if e.get('in_force')), None)
                for pid, by_source in ledger.items()}
    previous = {pid: v for pid, v in previous.items() if v is not None}
    known = {s for by_source in ledger.values() for s in by_source}
    explicit = None if rulings_path is None else _resolve_source(rulings_path, own, known)
    report.rulings_path, report.own_source = explicit or own or '', own or ''
    dropped = None
    if withdraw_pending is not None:
        dropped = _resolve_source(withdraw_pending, own, known)
        if dropped == own:
            report.warnings.append(f"{own} is the set's own rulings file: it is read on "
                                   f"every pass, so edit it instead of dropping it")
            dropped = None
        else:
            report.pending_withdrawn = [(pid, dropped) for pid in drop_source(ledger, dropped)]
            if not report.pending_withdrawn:
                report.warnings.append(f'the ledger holds no ruling of {dropped}: '
                                       f'nothing dropped')
    _update_ledger(ledger, report, own, explicit, dropped)
    _reconcile(group, ledger, previous, report, own)
    if not dry_run:
        if ledger != stored:
            group.base.extras.set(LEDGER_EXTRA, ledger)
        for key in _OLD_EXTRAS:
            if key in extras:
                group.base.extras.delete(key)
    return report


def apply_rulings(label: str, rulings_path: Optional[Path] = None, *,
                  withdraw: Optional[bool] = None,
                  withdraw_pending: Optional[Path] = None,
                  profile: Optional[str] = None) -> RulingReport:
    """Reconcile the set ``friction2d/<label>``'s rulings with its ruling
    ledger (:mod:`frictionsim2d.aiida.rulings`, Group extra :data:`LEDGER_EXTRA`).

    Every pass reads the set's own ``integrity_rulings.yaml``
    (``<source_root>/integrity_rulings.yaml``), ``rulings_path`` when given
    (the file or its directory), and every other rulings file the ledger
    already holds that exists. A file that parses replaces its entries: a
    point it no longer rules is withdrawn by it. A malformed entry is
    skipped (the others count, nothing is withdrawn); an unreadable file
    changes nothing. A file that is gone keeps its entries, marked
    ``missing``: they still count and every report lists them as pending.
    ``withdraw_pending`` (resolved as ``rulings_path`` is; it need not
    exist) deletes that file's entries. ``withdraw`` is accepted for
    compatibility and changes nothing: every pass follows the files.

    Per point the ruling in force is the own file's, else the most recently
    seen (added or changed) entry of another file -- the first of those that
    applies (``decide``: a quarantined point takes either ruling, any other
    needs ``override: true`` and a note). Each live verdict gets it
    (:func:`apply_ruling`, superseding the earlier ruled and withdrawn
    ones), loses one no source makes (:func:`withdraw_ruling`), or keeps the
    one in force. Idempotent: a second pass makes no node and leaves the
    ledger as it was. Each ruling Dict records ``source`` (the resolved
    path) and ``own_file``.
    """
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel

    del withdraw                    # every pass follows the files
    load_aiida_profile(profile)
    return rule_group(_set_group(label), rulings_path, withdraw_pending=withdraw_pending)


@dataclass
class RecheckReport:
    """What one :func:`recheck_set` pass did: the integrity settings
    ``version`` it checked under, the new verdicts' status counts, the
    re-checked points whose status changed (``(point id, old, new)``), the
    rulings pass that followed, failures and warnings."""
    label: str
    version: str = ''
    verdict_counts: Dict[str, int] = field(default_factory=dict)
    rechecked: List[Tuple[str, str, str]] = field(default_factory=list)
    rulings: Optional[RulingReport] = None
    failures: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def render(self) -> str:
        """The report as text."""
        total = sum(self.verdict_counts.values())
        lines = [f'set {self.label!r}: {total} point(s) checked under integrity '
                 f'thresholds v{self.version}']
        lines += [f'  verdict {status}: {n}' for status, n in sorted(self.verdict_counts.items())]
        lines += [f'  re-checked {pid}: {old} -> {new}' for pid, old, new in self.rechecked]
        lines += [f'  FAILED {who}: {why}' for who, why in self.failures]
        lines += [f'  WARNING {w}' for w in self.warnings]
        if self.rulings is not None:
            lines.append(self.rulings.render())
        return '\n'.join(lines)


def recheck_set(label: str, *, settings_file: Optional[Path] = None,
                profile: Optional[str] = None) -> RecheckReport:
    """Re-check the set ``friction2d/<label>`` under the current integrity
    settings (those of ``settings_file``, else of the per-user file).

    :func:`check_group` with ``recheck=True``: every live point whose live
    verdict was checked with another ``thresholds_version``, or with other
    stored settings content, gets a new verdict; the old one is superseded
    with its ruled and withdrawn verdicts, as a context change does. Then
    the rulings pass (:func:`rule_group`) gives the new verdicts the rulings
    of the set's ledger, from every file. A second pass checks nothing.
    """
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel

    load_aiida_profile(profile)
    group = _set_group(label)
    report = RecheckReport(label=label)
    settings = _settings_dump(report.warnings, settings_file=settings_file)
    report.version = settings['version']
    report.verdict_counts = check_group(
        group, settings=settings, failures=report.failures, warnings=report.warnings,
        rechecked=report.rechecked, recheck=True)
    report.rulings = rule_group(group)
    return report


# ---------------------------------------------------------------------------
# The latest verdict (by policy, never by ctime)
# ---------------------------------------------------------------------------

def _verdict_candidates(points: List[orm.FolderData]
                        ) -> Tuple[Dict[int, List[orm.Dict]], Dict[str, orm.Dict]]:
    """``(point pk -> its live checker verdicts plus their live ruled and
    withdrawn verdicts, checker uuid -> checker verdict)``, in two queries."""
    checker = _live_verdicts(points)
    by_uuid = {v.uuid: v for nodes in checker.values() for v in nodes}
    successors: Dict[str, List[orm.Dict]] = {}
    if by_uuid:
        qb = orm.QueryBuilder().append(orm.Dict, filters={
            f'extras.{KIND_EXTRA}': {'in': ['ruled_verdict', 'withdrawn_verdict']},
            f'extras.{VERDICT_UUID_EXTRA}': {'in': sorted(by_uuid)}})
        for node in qb.all(flat=True):
            if _live(node):
                successors.setdefault(node.base.extras.get(VERDICT_UUID_EXTRA, None),
                                      []).append(node)
    return ({pk: nodes + [s for v in nodes for s in successors.get(v.uuid, [])]
             for pk, nodes in checker.items()}, by_uuid)


def _select_node(nodes: List[orm.Dict], kind: str, policy: Policy,
                 digest: Optional[str] = None) -> Optional[orm.Dict]:
    """The node of ``nodes`` that :func:`policy.select` picks (on the
    versions, the ``friction2d_kind`` extra, the options hash and the uuid)."""
    by_uuid = {n.uuid: n for n in nodes}
    candidates = [{'uuid': n.uuid, 'kind': n.base.extras.get(KIND_EXTRA, 'verdict'),
                   'thresholds_version': n.get('thresholds_version'),
                   'code_version': n.get('code_version'),
                   'options_hash': n.get('options_hash')} for n in nodes]
    chosen = select(candidates, kind, policy, options_hash=digest)
    return None if chosen is None else by_uuid[chosen['uuid']]


def _provisional(verdict: orm.Dict, checkers: Dict[str, orm.Dict]) -> bool:
    """:func:`is_provisional` with the checker verdicts already loaded."""
    if verdict.base.extras.get(PROVISIONAL_EXTRA, False):
        return True
    source = verdict.base.extras.get(VERDICT_UUID_EXTRA, None)
    if not source:
        return False
    checker = checkers.get(source)
    if checker is None:
        checker = orm.load_node(source)
    return bool(checker.base.extras.get(PROVISIONAL_EXTRA, False))


def _latest(points: List[orm.FolderData], policy: Policy
            ) -> Dict[int, Tuple[Optional[orm.Dict], bool]]:
    """point pk -> (its selected verdict, whether that is provisional)."""
    candidates, checkers = _verdict_candidates(points)
    out: Dict[int, Tuple[Optional[orm.Dict], bool]] = {}
    for pk, nodes in candidates.items():
        verdict = _select_node(nodes, 'verdict', policy)
        out[pk] = (verdict, verdict is not None and _provisional(verdict, checkers))
    return out


def latest_verdicts(points: List[orm.FolderData],
                    policy: Policy = Policy()) -> Dict[int, Optional[orm.Dict]]:
    """point pk -> :func:`latest_verdict` of each point, in bulk."""
    return {pk: verdict for pk, (verdict, _p) in _latest(points, policy).items()}


def latest_verdict(point: orm.FolderData, policy: Policy = Policy()) -> Optional[orm.Dict]:
    """The verdict ``policy`` selects for ``point``, or None.

    Among the point's live verdicts -- the checker's (``check_integrity``),
    the ruled (``apply_ruling``) and the withdrawn (``withdraw_ruling``) --
    :func:`frictionsim2d.aiida.rulings.select` takes the newest generation
    (``thresholds_version``, then ``code_version``; or the pinned one), and
    within it a ruled verdict over a withdrawn one over the checker's; then
    the uuid. Never the creation time. Whether the choice is provisional is
    :func:`is_provisional`.
    """
    return latest_verdicts([point], policy)[point.pk]


def is_provisional(verdict: orm.Dict) -> bool:
    """Whether ``verdict`` rests on a check made while stage 2 was unknown:
    its own ``friction2d_provisional`` extra, or (a ruled or withdrawn
    verdict) that of the checker verdict it stands for."""
    return _provisional(verdict, {})


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------

def _point_frames(point: orm.FolderData, directory: Path, family: str,
                  n_layers: int) -> Dict[str, Any]:
    """The point's stored tiers as frames, the way the DataReader reads them:
    ``indent_f*N_l*.csv`` is the retract tier, the rest by filename."""
    # pylint: disable=import-outside-toplevel
    from ..postprocessing.read_data import tier_of
    from .calcjob import _read_frame  # pylint: disable=protected-access
    from .ingest import _INDENT_RE  # pylint: disable=protected-access

    repo = point.base.repository
    if 'results' not in repo.list_object_names():
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    frames: Dict[str, Any] = {}
    for name in sorted(repo.list_object_names('results')):
        tier = 'retract' if _INDENT_RE.match(name) else tier_of(name)
        path = directory / name
        path.write_bytes(repo.get_object_content(f'results/{name}', mode='rb'))
        frames.setdefault(tier, _read_frame(path, family, n_layers))
    return frames


def _post(point: orm.FolderData, snapshot: Optional[orm.SinglefileData],
          provenance: Optional[orm.FolderData], directory: Path) -> Dict[str, float]:
    """``post_channels`` of the point's end snapshot, as
    ``post_channels_for`` gives them to the DataReader; ``{}`` without a
    snapshot or a provenance record."""
    # pylint: disable=import-outside-toplevel
    from ..data.provenance import read_provenance
    from ..postprocessing.post_channels import post_channels
    from ..run.naming import parse_run_path

    if snapshot is None or provenance is None:
        return {}
    try:
        text = provenance.base.repository.get_object_content('provenance.json', mode='rb')
    except (FileNotFoundError, OSError):
        return {}
    record_path = directory / 'provenance.json'
    record_path.write_bytes(text)
    attrs = point.base.attributes.all
    tokens = parse_run_path(attrs['rel_path'])
    try:
        record = read_provenance(record_path, family=attrs['family'], model=tokens.model,
                                 material_name=tokens.materials[0] if tokens.materials
                                 else '')
    except (OSError, ValueError) as exc:
        logger.warning('No provenance for %s (%s); post channels skipped',
                       attrs['rel_path'], exc)
        return {}
    return post_channels(_materialise(snapshot, directory / 'final'), record,
                         attrs['family'])


@calcfunction
def summarise(point: orm.FolderData, options: orm.Dict, parse: orm.Dict,
              snapshot: Optional[orm.SinglefileData] = None,
              provenance: Optional[orm.FolderData] = None) -> orm.Dict:
    """One point's summary: ``summarise_run`` over its stored tiers.

    ``options`` are the summary's own choices (:func:`summary_options`:
    ``skip_fraction``, the leading fraction of the window dropped); their
    :func:`options_hash` is part of the summary's identity. ``parse`` says
    how the point's files are read (``n_layers``). ``snapshot`` (the end
    state) and ``provenance`` (the run's record) give the post-stream
    channels, as the DataReader computes them. The result is
    ``RunSummary.as_dict()`` plus ``units`` (``block_schema.summary_units``:
    it refuses a number without a unit), ``options``, ``options_hash``,
    ``code_version`` and ``point_uuid``.
    """
    # pylint: disable=import-outside-toplevel
    from ..data.models import summarise_run
    from .block_schema import summary_units

    opts = options.get_dict()
    family = point.base.attributes.get('family')
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frames = _point_frames(point, tmp_dir / 'results', family,
                               int(parse['n_layers']))
        post = _post(point, snapshot, provenance, tmp_dir)
    summary = summarise_run(frames, family, skip_fraction=float(opts['skip_fraction']),
                            post=post).as_dict()
    return orm.Dict({**summary, 'units': summary_units(family, summary),
                     'options': opts, 'options_hash': options_hash(opts),
                     'code_version': CODE_VERSION,
                     'point_uuid': point.base.attributes.get('point_uuid')})


def summary_options(warnings: Optional[List[str]] = None, *,
                    settings_file: Optional[Path] = None) -> Dict[str, Any]:
    """The summary options of the settings file in effect (``settings_file``
    when given, else the per-user one):
    ``{'skip_fraction': settings.aiida.skip_fraction}``. The file is read
    strictly; when it cannot be, the default, with a warning in ``warnings``."""
    from ..core.config import AiidaSettings  # pylint: disable=import-outside-toplevel
    default = AiidaSettings().skip_fraction
    settings = _strict_settings(
        warnings, f'summaries use the default skip_fraction {default}', settings_file)
    return {'skip_fraction': float(default if settings is None
                                   else settings.aiida.skip_fraction)}


#: Summary option -> how it is normalised before hashing.
_OPTION_TYPES = {'skip_fraction': float}


def normalise_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """``options`` with known keys only, each of its type
    (``skip_fraction`` a float), so ``{'skip_fraction': 1}`` and
    ``{'skip_fraction': 1.0}`` are one identity. An unknown key, or a
    missing one, raises ``ValueError``."""
    unknown = sorted(set(options) - set(_OPTION_TYPES))
    if unknown:
        raise ValueError(f'unknown summary options {unknown}; known: '
                         f'{sorted(_OPTION_TYPES)}')
    missing = sorted(set(_OPTION_TYPES) - set(options))
    if missing:
        raise ValueError(f'summary options lack {missing}')
    return {key: kind(options[key]) for key, kind in _OPTION_TYPES.items()}


def options_hash(options: Dict[str, Any]) -> str:
    """sha256 of the canonical JSON of a summary's normalised ``options``."""
    return _canonical(normalise_options(options))


def _live_summaries(points: List[orm.FolderData]) -> Dict[int, List[orm.Dict]]:
    """point pk -> its live ``summarise`` outputs, oldest first."""
    out: Dict[int, List[orm.Dict]] = {p.pk: [] for p in points}
    if not points:
        return out
    qb = orm.QueryBuilder().append(
        orm.FolderData, filters={'id': {'in': sorted(out)}}, tag='p', project='id')
    qb.append(orm.CalcFunctionNode, with_incoming='p', tag='f',
              filters={'attributes.process_label': 'summarise'})
    qb.append(orm.Dict, with_incoming='f', project='*')
    qb.order_by({orm.Dict: {'id': 'asc'}})
    for pk, summary in qb.all():
        if _live(summary):
            out[pk].append(summary)
    return out


def latest_summaries(points: List[orm.FolderData], policy: Policy = Policy(), *,
                     digest: Optional[str] = None) -> Dict[int, Optional[orm.Dict]]:
    """point pk -> the summary ``policy`` selects among the point's live
    summaries (:func:`frictionsim2d.aiida.rulings.select`): only those made with the
    options whose :func:`options_hash` is ``digest`` (any, when None), then
    the highest ``code_version``, then the uuid. In one query."""
    return {pk: _select_node(nodes, 'summary', policy, digest)
            for pk, nodes in _live_summaries(points).items()}


class _SummaryInputs:  # pylint: disable=too-few-public-methods
    """``summarise``'s inputs, with the options and parse Dicts made once
    (the options Dict lazily, so a pass that summarises nothing stores
    nothing)."""

    def __init__(self, options: Dict[str, Any]) -> None:
        self._options = options
        self._nodes: Dict[Any, orm.Dict] = {}

    def _node(self, key: Any, content: Dict[str, Any], kind: str) -> orm.Dict:
        if key not in self._nodes:
            self._nodes[key] = _step_input(content, kind)
        return self._nodes[key]

    def __call__(self, point: orm.FolderData) -> Dict[str, Any]:
        calc = _creator(point)
        n_layers = int(calc.inputs.parameters.get_dict()['n_layers'])
        inputs: Dict[str, Any] = {
            'point': point,
            'options': self._node('options', self._options, 'summary_options'),
            'parse': self._node(('parse', n_layers), {'n_layers': n_layers},
                                'summary_parse'),
        }
        key = _output_key(calc, point)
        if 'snapshots' in calc.outputs and key in calc.outputs.snapshots:
            inputs['snapshot'] = calc.outputs.snapshots[key]
        if 'provenance' in calc.inputs:
            inputs['provenance'] = calc.inputs.provenance
        return inputs


def _report_versions(label: str, verdicts: Dict[int, Tuple[Optional[orm.Dict], bool]],
                     current: str, warnings: List[str]) -> None:
    """One warning per thresholds version, other than ``current``, that a
    selected verdict of the set was checked with."""
    stale: Counter = Counter(v.get('thresholds_version') for v, _p in verdicts.values()
                             if v is not None and v.get('thresholds_version') != current)
    for version, count in sorted(stale.items(), key=lambda kv: str(kv[0])):
        warnings.append(
            f'set {label!r}: {count} point(s) have verdicts checked with integrity '
            f'thresholds v{version}, not the current v{current}; summaries and '
            f'datasets use those verdicts until `aiida recheck {label}` re-checks them')


def postprocess_group(label: str, *, policy: Policy = Policy(),
                      skip_failed: bool = True,
                      options: Optional[Dict[str, Any]] = None,
                      warnings: Optional[List[str]] = None,
                      profile: Optional[str] = None,
                      settings_file: Optional[Path] = None) -> int:
    """Summarise every live point of ``friction2d/<label>`` whose selected
    verdict (:func:`latest_verdict` under ``policy``) is ``complete`` --
    checked, ruled or overridden so; return the number of new summaries.

    A summary's identity is (point, :func:`options_hash` of ``options``,
    ``code_version``); ``options`` default to :func:`summary_options` (of
    ``settings_file``, else of the per-user settings file). A
    point that already has a live summary with that identity is skipped, so
    a second pass makes none; new options or a ``code_version`` bump make
    new summaries beside the old ones, which stay live (``policy`` and the
    options choose between them). Each summary carries the extras
    ``friction2d_kind='summary'``, ``friction2d_point_uuid``,
    ``friction2d_options_hash``, ``friction2d_summary_verdict`` (the verdict
    it was made under) and ``friction2d_provisional``.

    To ``warnings``: every point summarised in this pass on a provisional
    verdict, every thresholds version other than the current
    ``IntegritySettings.version`` a selected verdict was checked with
    (reported; :func:`recheck_set` re-checks them), an unreadable settings file, and, with
    ``skip_failed``, every point whose summary raised (otherwise the error
    propagates).
    """
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel

    load_aiida_profile(profile)
    problems: List[str] = []
    points = _live_points(_set_group(label))
    verdicts = _latest(points, policy)
    _report_versions(label, verdicts,
                     _settings_dump(problems, settings_file=settings_file)['version'],
                     problems)
    opts = normalise_options(options if options is not None
                             else summary_options(problems, settings_file=settings_file))
    digest = options_hash(opts)
    existing = _live_summaries(points)
    inputs_of = _SummaryInputs(opts)
    made = 0
    for point in points:
        verdict, provisional = verdicts[point.pk]
        if verdict is None or verdict['status'] != 'complete':
            continue
        if any(s.get('code_version') == CODE_VERSION and s.get('options_hash') == digest
               for s in existing[point.pk]):
            continue
        try:
            summary = summarise(**inputs_of(point))
        except Exception as exc:  # pylint: disable=broad-except
            if not skip_failed:
                raise
            logger.exception('Summary failed for point %s', point.uuid)
            problems.append(f'point {_pid(point)}: summary failed '
                            f'({type(exc).__name__}: {exc})')
            continue
        summary.base.extras.set_many({
            KIND_EXTRA: 'summary', POINT_UUID_EXTRA: point.base.attributes.get('point_uuid'),
            OPTIONS_HASH_EXTRA: digest, SUMMARY_VERDICT_EXTRA: verdict.uuid,
            PROVISIONAL_EXTRA: provisional})
        if provisional:
            problems.append(f'point {_pid(point)}: summarised on a provisional verdict '
                            f'(stage 2 was unknown at its check); its summary is '
                            f'marked friction2d_provisional')
        made += 1
    for line in problems:
        logger.warning('%s', line)
    if warnings is not None:
        warnings += problems
    return made


# ---------------------------------------------------------------------------
# Datasets: a Group of selected summaries plus one table Dict (critique M3)
# ---------------------------------------------------------------------------

#: What each key of a dataset row holds.
DATASET_COLUMNS: Dict[str, str] = {
    'set': 'label of the set (friction2d/<label>) the point was taken from',
    'family': 'the point family; its units are units[family]',
    'point_uuid': 'the point (FolderData) uuid',
    'calc_uuid': 'uuid of the CalcJob whose points output it is',
    'summary_uuid': 'uuid of the summary Dict the policy selected',
    'code_version': "the summary's code_version",
    'options_hash': "the summary's options_hash (policy.options_hash)",
    'skip_fraction': "the summary's skip_fraction",
    'verdict_uuid': 'uuid of the verdict the policy selected',
    'verdict_status': "that verdict's status (always complete)",
    'thresholds_version': "that verdict's integrity thresholds version",
    'provisional': 'whether that verdict was checked while stage 2 was unknown',
    'block_hashes': "CalcJob blocks namespace key -> the block's friction2d_block_hash",
}


def merge_units(units: Dict[str, Dict[str, str]], family: str,
                summary_units: Dict[str, str]) -> None:
    """Fold one summary's ``units`` into ``units[family]``; a key whose unit
    differs between two summaries of a family raises ``ValueError``."""
    table = units.setdefault(family, {})
    for key, unit in summary_units.items():
        if table.setdefault(key, unit) != unit:
            raise ValueError(f'family {family!r}: summaries disagree on the unit of '
                             f'{key!r} ({table[key]!r} vs {unit!r}); a dataset never '
                             f'mixes units')


def _block_hashes(calc: orm.CalcJobNode) -> Dict[str, Optional[str]]:
    from .store import BLOCK_HASH_EXTRA  # pylint: disable=import-outside-toplevel
    if 'blocks' not in calc.inputs:
        return {}
    return {name: node.base.extras.get(BLOCK_HASH_EXTRA, None)
            for name, node in sorted(calc.inputs.blocks.items())}


def _dataset_rows(name: str, labels: Sequence[str], policy: Policy, digest: str
                  ) -> Tuple[List[Dict[str, Any]], List[orm.Node], Dict[str, Dict[str, str]]]:
    """``(rows, summary nodes, units)`` of a dataset over the sets ``labels``."""
    rows: List[Dict[str, Any]] = []
    members: List[orm.Node] = []
    units: Dict[str, Dict[str, str]] = {}
    seen: Set[int] = set()
    for label in labels:
        points = [p for p in _live_points(_set_group(label)) if p.pk not in seen]
        verdicts = _latest(points, policy)
        summaries = _live_summaries(points)
        for point in points:
            verdict, provisional = verdicts[point.pk]
            if verdict is None or verdict['status'] != 'complete':
                continue
            summary = _select_node(summaries[point.pk], 'summary', policy, digest)
            if summary is None:
                logger.info('dataset %r: point %s has no summary under %s, options %s',
                            name, point.uuid, policy.as_dict(), digest)
                continue
            seen.add(point.pk)
            family = point.base.attributes.get('family')
            merge_units(units, family, summary.get('units', {}))
            calc = _creator(point)
            rows.append({
                'set': label, 'family': family,
                'point_uuid': point.base.attributes.get('point_uuid'),
                'calc_uuid': calc.uuid, 'summary_uuid': summary.uuid,
                'code_version': summary.get('code_version'),
                'options_hash': summary.get('options_hash'),
                'skip_fraction': summary.get('skip_fraction'),
                'verdict_uuid': verdict.uuid, 'verdict_status': verdict['status'],
                'thresholds_version': verdict.get('thresholds_version'),
                'provisional': provisional,
                'block_hashes': _block_hashes(calc),
            })
            members.append(summary)
    return rows, members, units


def build_dataset(name: str, labels: Sequence[str], *, policy: Policy = Policy(),
                  options: Optional[Dict[str, Any]] = None,
                  warnings: Optional[List[str]] = None,
                  profile: Optional[str] = None,
                  settings_file: Optional[Path] = None) -> orm.Dict:
    """Make the dataset ``friction2d/datasets/<name>`` from the sets
    ``labels``: a Group holding, for every live point whose selected verdict
    is ``complete``, the summary ``policy`` selects among those made with
    ``options`` (default :func:`summary_options`), plus one table ``Dict``
    (also in the Group, extra ``friction2d_kind='dataset'``).

    The Dict holds ``name``, ``labels``, ``policy`` (``Policy.as_dict()``
    plus ``summary_options`` and their ``options_hash``), ``units`` (family
    -> key -> unit, from the summaries; disagreeing units raise),
    ``columns`` (:data:`DATASET_COLUMNS`) and ``rows``, one per point. No
    calcfunction: the Group, not thousands of input links, ties the rows to
    their summaries. A point with no summary the policy selects is left out
    (logged). An existing dataset of that name raises ``ValueError``. The
    rows are computed before anything is stored; the Group is stored first,
    then the table, and a failure while storing removes both again, so
    nothing is left orphaned. :func:`delete_dataset` undoes a dataset.

    ``options`` are normalised (:func:`normalise_options`; unknown keys
    raise); they default to those of ``settings_file``, else of the per-user
    settings file. An unreadable settings file, when ``options`` default,
    goes to ``warnings`` (logged as a WARNING without a list).
    """
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel

    load_aiida_profile(profile)
    group_label = f'{DATASET_PREFIX}/{name}'
    if orm.QueryBuilder().append(orm.Group, filters={'label': group_label}).count():
        raise ValueError(f'dataset {name!r} exists ({group_label}); datasets are '
                         f'not rewritten -- build it under a new name, or '
                         f'delete_dataset({name!r}) first')
    opts = normalise_options(options if options is not None
                             else summary_options(warnings, settings_file=settings_file))
    digest = options_hash(opts)
    rows, members, units = _dataset_rows(name, labels, policy, digest)
    group = orm.Group(label=group_label).store()
    table: Optional[orm.Dict] = None
    try:
        table = orm.Dict({'name': name, 'labels': list(labels),
                          'policy': {**policy.as_dict(), 'summary_options': opts,
                                     'options_hash': digest},
                          'units': units, 'columns': dict(DATASET_COLUMNS),
                          'rows': rows}).store()
        table.base.extras.set_many({KIND_EXTRA: 'dataset', DATASET_EXTRA: name})
        group.add_nodes(members + [table])
    except Exception:
        # pylint: disable=import-outside-toplevel
        from aiida.tools import delete_nodes
        orm.Group.collection.delete(group.pk)
        if table is not None and table.is_stored:
            delete_nodes([table.pk], dry_run=False)
        raise
    return table


class DatasetDeletion(NamedTuple):
    """What :func:`delete_dataset` deleted: the dataset's table ``Dict`` nodes
    and the analysis results (``frictionsim2d.analysis``) computed from them."""

    tables: int
    analyses: int


#: The Group of a dataset's analysis results is this plus the dataset name
#: (``frictionsim2d.analysis.store.ANALYSIS_GROUP_PREFIX``).
_ANALYSIS_GROUP = 'friction2d/analyses/'


def delete_dataset(name: str, *, profile: Optional[str] = None) -> DatasetDeletion:
    """Delete the dataset ``friction2d/datasets/<name>``: its Group and its
    table Dict. The summaries it held are kept (their points' sets own them).

    Every analysis result computed from the table goes with it (the table is
    an input of its ``record_analysis``): they are counted, logged as a
    WARNING, and their emptied Group ``friction2d/analyses/<name>`` removed.
    Returns the counts (:class:`DatasetDeletion`)."""
    # pylint: disable=import-outside-toplevel
    from aiida.tools import delete_nodes

    from . import load_aiida_profile

    load_aiida_profile(profile)
    found = orm.QueryBuilder().append(
        orm.Group, filters={'label': f'{DATASET_PREFIX}/{name}'}).all(flat=True)
    if not found:
        raise ValueError(f'no dataset {name!r} ({DATASET_PREFIX}/{name})')
    (group,) = found
    tables = [n.pk for n in group.nodes
              if isinstance(n, orm.Dict) and n.base.extras.get(DATASET_EXTRA, None) == name]
    analyses = 0
    if tables:
        doomed, _ = delete_nodes(tables, dry_run=True)
        analyses = orm.QueryBuilder().append(orm.SinglefileData, filters={
            'id': {'in': list(doomed)}, 'extras': {'has_key': 'analysis'}}).count()
    if analyses:
        logger.warning('deleting dataset %r also deletes the %d analysis results '
                       'computed from it', name, analyses)
    orm.Group.collection.delete(group.pk)
    if tables:
        delete_nodes(tables, dry_run=False)
    for left in orm.QueryBuilder().append(
            orm.Group, filters={'label': f'{_ANALYSIS_GROUP}{name}'}).all(flat=True):
        if not left.count():
            orm.Group.collection.delete(left.pk)
    return DatasetDeletion(len(tables), analyses)


# ---------------------------------------------------------------------------
# Deleting sets, clearing the store
# ---------------------------------------------------------------------------
#
# A set is the Group ``friction2d/<label>`` that :func:`frictionsim2d.aiida.ingest.import_root`
# fills with deck CalcJobs. One CalcJob can sit in several sets (a tree imported
# under two labels), and every content-addressed input -- blocks, files,
# provenance, schema, Code -- is shared across runs, so deleting a set removes
# only what no other set still holds.

def _set_groups() -> List[Any]:
    """Every ``friction2d/<label>`` set Group (dataset and analysis Groups excluded)."""
    from .ingest import GROUP_PREFIX  # pylint: disable=import-outside-toplevel

    qb = orm.QueryBuilder().append(
        orm.Group, filters={'label': {'like': f'{GROUP_PREFIX}/%'}})
    return [g for g in qb.all(flat=True)
            if not g.label.startswith((f'{GROUP_PREFIX}/datasets/',
                                       f'{GROUP_PREFIX}/analyses/'))]


def _calcjobs_of(group) -> List[Any]:
    qb = orm.QueryBuilder().append(orm.Group, filters={'id': group.pk}, tag='g')
    qb.append(orm.CalcJobNode, with_group='g')
    return qb.all(flat=True)


def _held_elsewhere(group) -> Set[int]:
    """pks of CalcJobs some OTHER set Group holds."""
    return {calc.pk for other in _set_groups() if other.pk != group.pk
            for calc in _calcjobs_of(other)}


def _dataset_held() -> Dict[int, Set[str]]:
    """node pk -> names of the datasets (``friction2d/datasets/<name>``)
    whose Group holds it."""
    qb = orm.QueryBuilder().append(
        orm.Group, filters={'label': {'like': f'{DATASET_PREFIX}/%'}}, tag='g',
        project='label')
    qb.append(orm.Node, with_group='g', project='id')
    held: Dict[int, Set[str]] = {}
    for label, pk in qb.all():
        held.setdefault(pk, set()).add(label[len(DATASET_PREFIX) + 1:])
    return held


def _delete_closure(pks: Iterable[int]) -> Set[int]:
    """What ``delete_nodes`` would remove with ``pks``: their outputs and every
    process consuming those, forward only -- inputs are never followed."""
    from aiida.tools import delete_nodes  # pylint: disable=import-outside-toplevel

    pks = sorted(set(pks))
    if not pks:
        return set()
    closure, _deleted = delete_nodes(pks, dry_run=True)
    return set(closure)


def delete_simulation_set(label: str, profile: Optional[str] = None) -> int:
    """Delete the set ``friction2d/<label>``: its Group, and every CalcJob of
    it that no other set holds, with that CalcJob's outputs and their
    calcfunction descendants.

    Shared inputs (blocks, files, provenance, schema, Codes, Computers) are
    never deleted; neither is a CalcJob another set holds, nor one whose
    deletion would cascade into such a CalcJob. The input Dicts of the steps
    it deletes (integrity context and settings, rulings, summary options) go
    with them when no remaining process uses them. Returns the number of
    nodes deleted.

    A set a dataset draws on is refused whole, before anything is deleted:
    deleting its CalcJobs would cascade (``input_calc_forward``) into the
    summaries a ``friction2d/datasets/<name>`` Group holds, leaving that
    dataset's rows pointing at nothing. ``ValueError`` names the datasets
    and the :func:`frictionsim2d.aiida.steps.delete_dataset` calls that release the set.
    """
    # pylint: disable=import-outside-toplevel
    from aiida.tools import delete_nodes

    from . import load_aiida_profile
    from .ingest import group_label

    load_aiida_profile(profile)

    group = orm.Group.collection.get(label=group_label(label))
    protected = _held_elsewhere(group)
    in_datasets = _dataset_held()
    doomed: Set[int] = set()
    for calc in _calcjobs_of(group):
        if calc.pk in protected:
            continue
        closure = _delete_closure([calc.pk])
        if closure & protected:
            logger.warning('Keeping CalcJob %s of %r: deleting it would delete '
                           'CalcJobs another set holds', calc.pk, label)
            continue
        doomed |= closure
    datasets = sorted({name for pk in doomed & set(in_datasets)
                       for name in in_datasets[pk]})
    if datasets:
        raise ValueError(
            f'set {label!r} is not deleted: dataset(s) {", ".join(map(repr, datasets))} '
            f'hold summaries of its points, which deleting it would delete too. '
            f'Delete those datasets first: '
            + '; '.join(f'frictionsim2d.aiida.steps.delete_dataset({n!r})' for n in datasets))
    stranded = _stranded_sets(group, doomed)
    step_inputs = _step_inputs(doomed)
    if doomed:
        delete_nodes(sorted(doomed), dry_run=False)
    orphans = _orphans(step_inputs - doomed)
    if orphans:
        delete_nodes(orphans, dry_run=False)
    orm.Group.collection.delete(group.pk)
    if stranded:
        logger.warning(
            'Deleting set %r removed the live CalcJobs of %d deck(s) that set(s) %s '
            'hold only as superseded copies. Recover by re-importing the tree under '
            'each surviving label (FrictionSim2D aiida import <root> --label <label>).',
            label, len(stranded), ', '.join(repr(s) for s in sorted(
                {other for others in stranded.values() for other in others})))
    return len(doomed) + len(orphans)


def _stranded_sets(group, doomed: Set[int]) -> Dict[str, Set[str]]:
    """deck key -> labels of the other sets that hold that deck only as a
    superseded CalcJob, once the live CalcJobs in ``doomed`` are deleted."""
    from .ingest import GROUP_PREFIX  # pylint: disable=import-outside-toplevel
    from .store import DECK_KEY_EXTRA  # pylint: disable=import-outside-toplevel

    removed = {c.base.extras.get(DECK_KEY_EXTRA) for c in _calcjobs_of(group)
               if _live(c) and c.pk in doomed}
    removed.discard(None)
    stranded: Dict[str, Set[str]] = {}
    for other in _set_groups():
        if other.pk == group.pk:
            continue
        calcs = _calcjobs_of(other)
        live = {c.base.extras.get(DECK_KEY_EXTRA) for c in calcs if _live(c)}
        for calc in calcs:
            key = calc.base.extras.get(DECK_KEY_EXTRA)
            if key in removed and key not in live:
                stranded.setdefault(key, set()).add(other.label[len(GROUP_PREFIX) + 1:])
    return stranded


def _friction2d_calcjobs() -> List[int]:
    from .store import CALCJOB_PROCESS_TYPE  # pylint: disable=import-outside-toplevel

    return orm.QueryBuilder().append(
        orm.CalcJobNode, filters={'process_type': CALCJOB_PROCESS_TYPE},
        project='id').all(flat=True)


def _inputs_of(calc_pks: List[int]) -> Set[int]:
    if not calc_pks:
        return set()
    qb = orm.QueryBuilder().append(orm.CalcJobNode, filters={'id': {'in': calc_pks}},
                                   tag='c')
    qb.append(orm.Node, with_outgoing='c', project='id')
    return set(qb.all(flat=True))


def _orphans(candidates: Set[int]) -> List[int]:
    """The ``candidates`` that still exist and no process uses as an input."""
    if not candidates:
        return []
    ids = {'id': {'in': sorted(candidates)}}
    existing = set(orm.QueryBuilder().append(orm.Node, filters=ids,
                                             project='id').all(flat=True))
    qb = orm.QueryBuilder().append(orm.Node, filters=ids, tag='n', project='id')
    qb.append(orm.ProcessNode, with_incoming='n')
    return sorted(existing - set(qb.all(flat=True)))


def _step_inputs(within: Optional[Set[int]] = None) -> Set[int]:
    """pks of the Dicts made only to feed a step (:data:`STEP_INPUT_EXTRA`:
    integrity context and settings, user rulings); with ``within``, only
    those some process in ``within`` takes as an input."""
    qb = orm.QueryBuilder().append(
        orm.Dict, filters={'extras': {'has_key': STEP_INPUT_EXTRA}}, tag='d', project='id')
    if within is not None:
        if not within:
            return set()
        qb.append(orm.ProcessNode, with_incoming='d',
                  filters={'id': {'in': sorted(within)}})
    return set(qb.all(flat=True))


def _dataset_tables() -> Set[int]:
    """pks of the datasets' table Dicts (:data:`DATASET_EXTRA`)."""
    return set(orm.QueryBuilder().append(
        orm.Dict, filters={'extras': {'has_key': DATASET_EXTRA}},
        project='id').all(flat=True))


def _content_addressed() -> Set[int]:
    """pks of every content-addressed node (blocks, files, provenance folders,
    schemas) and every LAMMPS Code (``store``'s index extras)."""
    # pylint: disable=import-outside-toplevel
    from .store import (BLOCK_HASH_EXTRA, FILE_SHA_EXTRA, FOLDER_SHA_EXTRA,
                        SCHEMA_HASH_EXTRA, VERSION_EXTRA)

    pks: Set[int] = set()
    for extra in (BLOCK_HASH_EXTRA, FILE_SHA_EXTRA, FOLDER_SHA_EXTRA, SCHEMA_HASH_EXTRA,
                  VERSION_EXTRA):
        pks |= set(orm.QueryBuilder().append(
            orm.Node, filters={'extras': {'has_key': extra}}, project='id').all(flat=True))
    return pks


#: ``node_type`` prefix of the removed ``friction2d.*`` ``aiida.data`` node
#: types (the old set/run/results nodes) an older profile may still hold.
_LEGACY_NODE_TYPE = 'data.friction2d.%'


def _legacy_nodes() -> List[int]:
    """The pk of every node of a removed ``friction2d.*`` Data type, matched
    on ``node_type`` (the classes and their entry points are gone)."""
    return orm.QueryBuilder().append(
        orm.Node, filters={'node_type': {'like': _LEGACY_NODE_TYPE}},
        project='id').all(flat=True)


def clear_all_nodes(profile: Optional[str] = None) -> int:
    """Delete every FrictionSim2D node in the profile and every
    ``friction2d/*`` Group (spec U6).

    That is every friction2d CalcJob with its descendants, then the inputs
    they leave unused -- content-addressed blocks, files, provenance, schema,
    LAMMPS Codes, ``derived`` Dicts, remote folders, the integrity, ruling
    and summary steps' input Dicts, the datasets' table Dicts (their summaries
    go with the CalcJobs' descendants) -- so a re-import builds
    everything afresh. Computers are kept. Returns the number of nodes deleted.
    """
    # pylint: disable=import-outside-toplevel
    from aiida.tools import delete_nodes

    from . import load_aiida_profile
    from .ingest import GROUP_PREFIX

    load_aiida_profile(profile)

    calcs = _friction2d_calcjobs()
    inputs = (_inputs_of(calcs) | _content_addressed() | _step_inputs()
              | _dataset_tables())
    doomed = _delete_closure(calcs + _legacy_nodes())
    if doomed:
        delete_nodes(sorted(doomed), dry_run=False)
    orphans = _orphans(inputs - doomed)
    if orphans:
        delete_nodes(orphans, dry_run=False)
    qb = orm.QueryBuilder().append(
        orm.Group, filters={'label': {'like': f'{GROUP_PREFIX}/%'}}, project='id')
    for group_pk in qb.all(flat=True):
        orm.Group.collection.delete(group_pk)
    return len(doomed) + len(orphans)
