"""Tracked steps over imported points: integrity verdicts and user rulings.

* :func:`check_integrity` -- a calcfunction: one point's verdict from the
  ported checker, :func:`src.postprocessing.integrity.check_files`, on the
  point's stored files. The verdict is ``IntegrityVerdict.as_dict()`` plus
  ``thresholds_version``, ``code_version`` and ``point_uuid``; the thresholds
  are the integrity module's own (dimensionless, nothing rescaled here).
* :func:`check_group` -- runs it on every live point of a set that has no
  live verdict yet; :func:`~src.aiida.ingest.import_root` calls it.
* :func:`apply_ruling` -- a calcfunction: a verdict as one user ruling decides
  it (:func:`src.postprocessing.integrity_rulings.rule_verdict`, the rulings
  layer's own decision).
* :func:`apply_rulings` -- the idempotent pass over a set's
  ``integrity_rulings.yaml``: one ``apply_ruling`` per (verdict, ruling)
  pair, keyed by the extra ``friction2d_ruling_key``.

Every Dict made only to feed a step (``context``, ``settings``, ``ruling``)
carries the extra ``friction2d_step_input``, so deleting a set or clearing
the store can collect them once their step is gone.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from aiida import orm
from aiida.engine import calcfunction

from .. import __version__ as CODE_VERSION

logger = logging.getLogger(__name__)

__all__ = ['CODE_VERSION', 'RulingReport', 'apply_ruling', 'apply_rulings',
           'check_group', 'check_integrity', 'rule_group', 'withdraw_ruling']

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
#: ``'user'`` on a ruling Dict: a human, not the checker, decided it.
AUTHOR_EXTRA = 'friction2d_author'

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


def _settings_dump(warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    """``settings.integrity`` of the global settings; the defaults, with a
    warning in ``warnings``, when they cannot be read."""
    # pylint: disable=import-outside-toplevel
    from ..core.config import IntegritySettings, load_settings
    try:
        thresholds = load_settings().integrity
    except (OSError, ValueError, AttributeError) as exc:
        thresholds = IntegritySettings()
        if warnings is not None:
            warnings.append(f'settings.yaml unreadable ({type(exc).__name__}: {exc}); '
                            f'integrity uses the default thresholds '
                            f'v{thresholds.version}')
    return thresholds.model_dump(mode='json')


def _run_uuid(point: orm.Node) -> Optional[str]:
    from .store import RUN_UUID_EXTRA  # pylint: disable=import-outside-toplevel
    return _creator(point).base.extras.get(RUN_UUID_EXTRA)


def _window_rows(point: orm.FolderData, directory: Path) -> int:
    """Rows of the point's window tier; 0 without one (`RunFrames.window`)."""
    params = _creator(point).inputs.parameters.get_dict()
    directory.mkdir(parents=True, exist_ok=True)
    frame = _window(point, directory, params['family'], int(params['n_layers']))
    return 0 if frame is None else len(frame)


def _stage2(points: List[orm.FolderData], directory: Path) -> Dict[int, str]:
    """pk -> the status the DataReader gives each point before stage 3: the
    parser's, then the legacy truncation over the set's points by (family,
    size) (`run_status.legacy_truncated`), then `mark_not_reached` over each
    run's configured chains -- in the DataReader's order."""
    # pylint: disable=import-outside-toplevel
    from ..core.observables import expected_tier_rows
    from ..postprocessing.run_status import (
        UNWINDOWED_FAMILIES, expected_chains, legacy_truncated, mark_not_reached,
        point_key)
    from ..run.naming import parse_run_path
    from .calcjob import _provenance_input  # pylint: disable=protected-access

    status = {p.pk: p.base.attributes.get('status') for p in points}
    by_run: Dict[Optional[str], List[orm.FolderData]] = {}
    for point in points:
        by_run.setdefault(_run_uuid(point), []).append(point)
    configs = {u: _provenance_input(_creator(ps[0]))[0] for u, ps in by_run.items()}

    framed, rows = [], []
    for run_uuid, run_points in by_run.items():
        config = configs[run_uuid]
        for point in run_points:
            if status[point.pk] in _NO_FRAMES:
                continue
            attrs = point.base.attributes.all
            tokens = parse_run_path(attrs['rel_path'])
            family = attrs['family']
            framed.append(point)
            rows.append([family, (tokens.size_x, tokens.size_y), 0,
                         bool(config and expected_tier_rows(config, family)),
                         status[point.pk]])
    # Window rows are read only where some point is judged at all.
    judged = {(r[0], r[1]) for r in rows if not r[3] and r[4] == 'complete'
              and r[0] not in UNWINDOWED_FAMILIES}
    for point, row in zip(framed, rows):
        if (row[0], row[1]) in judged:
            row[2] = _window_rows(point, directory)
    for i in legacy_truncated([tuple(r) for r in rows]):
        status[framed[i].pk] = 'truncated'

    for run_uuid, run_points in by_run.items():
        family = run_points[0].base.attributes.get('family')
        keyed = {}
        for point in run_points:
            attrs = point.base.attributes.all
            keyed[point_key(attrs['load_kind'], attrs['load'], attrs['angle'],
                            attrs['speed'])] = point.pk
        current = {key: status[pk] for key, pk in keyed.items()}
        for chain in expected_chains(configs[run_uuid], family) or []:
            mark_not_reached(chain, current)
        for key, pk in keyed.items():
            status[pk] = current[key]
    return status


def check_group(group: orm.Group, *, settings: Optional[Dict[str, Any]] = None,
                failures: Optional[List[Tuple[str, str]]] = None,
                warnings: Optional[List[str]] = None) -> Dict[str, int]:
    """Run :func:`check_integrity` on every live point of ``group`` that has no
    live verdict; return the new verdicts' status counts.

    Each point is checked at the stage-2 status the DataReader gives it
    (:func:`_stage2`, over every live point of the set). ``settings`` is an
    ``IntegritySettings`` dump (default: the global settings'; a fallback to
    the defaults is reported in ``warnings``). A point whose check raises goes
    to ``failures`` as ``(point uuid, message)`` and the others go on.
    """
    points = _live_points(group)
    verdicts = _live_verdicts(points)
    todo = [p for p in points if not verdicts[p.pk]]
    counts: Counter = Counter()
    if not todo:
        return {}
    settings_node = _step_input(settings or _settings_dump(warnings), 'integrity_settings')
    runs: Dict[str, _RunView] = {}
    with TemporaryDirectory() as tmp:
        status = _stage2(points, Path(tmp) / 'rows')
        for point in todo:
            try:
                run_uuid = _run_uuid(point)
                if run_uuid not in runs:
                    runs[run_uuid] = _RunView(run_uuid, Path(tmp) / str(len(runs)))
                inputs = _point_inputs(point, runs[run_uuid], status[point.pk])
                inputs['context'] = _step_input(inputs['context'], 'integrity_context')
                verdict = check_integrity(**inputs, settings=settings_node)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception('Integrity check failed for point %s', point.uuid)
                if failures is not None:
                    failures.append((point.uuid, f'{type(exc).__name__}: {exc}'))
                continue
            verdict.base.extras.set_many({
                KIND_EXTRA: 'verdict',
                POINT_UUID_EXTRA: point.base.attributes.get('point_uuid')})
            counts[verdict['status']] += 1
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
    """What one :func:`apply_rulings` pass did.

    ``applied``/``skipped`` count (verdict, ruling) pairs: a new
    ``apply_ruling``, or one already in the store. ``unmatched`` are ruling
    ids naming no live point of the set; ``not_quarantined`` those on a point
    that is not quarantined and that lack ``override: true`` or a note (or
    would pass a point with no data). ``withdrawn`` are the points whose
    ruled verdict was withdrawn because no applicable ruling names them any
    more. ``overrides`` are ``(point id, original status, new status, note)``
    of every live overridden verdict of the set, read from the store.
    """
    label: str
    rulings_path: str = ''
    applied: int = 0
    skipped: int = 0
    unmatched: List[str] = field(default_factory=list)
    not_quarantined: List[str] = field(default_factory=list)
    withdrawn: List[str] = field(default_factory=list)
    overrides: List[Tuple[str, str, str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def render(self) -> str:
        """The report as text."""
        lines = [f'rulings for set {self.label!r} ({self.rulings_path}): '
                 f'applied {self.applied}, skipped {self.skipped}, '
                 f'withdrawn {len(self.withdrawn)}']
        lines += [f'  override {pid}: {old} -> {new} ({note})'
                  for pid, old, new, note in self.overrides]
        lines += [f'  withdrawn: {pid}' for pid in self.withdrawn]
        lines += [f'  unmatched: {pid}' for pid in self.unmatched]
        lines += [f'  not quarantined: {pid}' for pid in self.not_quarantined]
        lines += [f'  WARNING {w}' for w in self.warnings]
        return '\n'.join(lines)


def _canonical(content: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'))
                          .encode('utf-8')).hexdigest()


def _ruled(key: str) -> List[orm.Dict]:
    """The live ruled verdicts carrying ``friction2d_ruling_key == key``."""
    return [n for n in orm.QueryBuilder().append(
        orm.Dict, filters={f'extras.{RULING_KEY_EXTRA}': key}).all(flat=True)
        if _live(n)]


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


def _rule(report: RulingReport, pid: str, ruling, verdict: orm.Dict) -> bool:
    """One (ruling, verdict) pair: decide, skip or apply. True when the
    ruling applies to the verdict (applied now or already)."""
    from ..postprocessing.integrity_rulings import decide  # pylint: disable=import-outside-toplevel

    original = verdict['status']
    _status, why = decide(original, ruling)
    if why is not None:
        if original != 'quarantined' and pid not in report.not_quarantined:
            report.not_quarantined.append(pid)
        report.warnings.append(f'integrity ruling ignored: {pid}: {why}')
        return False
    content = {'point': pid, 'ruling': ruling.ruling, 'note': ruling.note,
               'date': ruling.date, 'override': ruling.override}
    key = f'{verdict.uuid}|{_canonical(content)}'
    if _ruled(key):
        report.skipped += 1
        return True
    node = _step_input(content, 'ruling')
    node.base.extras.set(AUTHOR_EXTRA, 'user')
    ruled = apply_ruling(verdict=verdict, ruling=node)
    _tag(ruled, 'ruled_verdict', verdict, **{RULING_KEY_EXTRA: key})
    for older in _successors(verdict):
        if older.pk != ruled.pk:
            _supersede(older)
    report.applied += 1
    return True


def _withdraw(report: RulingReport, pid: str, verdict: orm.Dict,
              ruled_nodes: List[orm.Dict]) -> None:
    """Withdraw ``ruled_nodes``, the live ruled verdicts of ``verdict``."""
    for ruled in ruled_nodes:
        back = withdraw_ruling(ruled=ruled, verdict=verdict)
        _tag(back, 'withdrawn_verdict', verdict)
        _supersede(ruled)
        if pid not in report.withdrawn:
            report.withdrawn.append(pid)


def _authoritative(group: orm.Group, path: Path) -> bool:
    """Whether ``path`` is the set's own rulings file (or its directory)."""
    from ..postprocessing.integrity_rulings import RULINGS_FILE  # pylint: disable=import-outside-toplevel

    root = group.base.extras.get('source_root', None)
    if not root:
        return False
    own = (Path(root) / RULINGS_FILE).resolve()
    path = Path(path)
    return own in (path.resolve(), (path / RULINGS_FILE).resolve())


def _unreadable(rulings) -> bool:
    """Whether the file, or an entry of it, could not be read: its point is
    unknown, so nothing may be withdrawn on the strength of its absence."""
    return any(key == '<file>' or key.startswith('<malformed') for key in rulings)


def rule_group(group: orm.Group, rulings_path: Optional[Path] = None, *,
               withdraw: Optional[bool] = None) -> RulingReport:
    """:func:`apply_rulings` on a loaded set Group."""
    # pylint: disable=import-outside-toplevel
    from ..postprocessing.integrity_rulings import RULINGS_FILE, format_point_id, load_rulings
    from .ingest import GROUP_PREFIX

    if rulings_path is None:
        root = group.base.extras.get('source_root', None)
        rulings_path = Path(root) / RULINGS_FILE if root else None
    report = RulingReport(label=group.label[len(GROUP_PREFIX) + 1:],
                          rulings_path=str(rulings_path or ''))
    if rulings_path is None:
        report.warnings.append('the set records no source_root; pass the rulings file')
        return report
    if withdraw is None:
        withdraw = _authoritative(group, Path(rulings_path))
    rulings = load_rulings(Path(rulings_path))
    points = _live_points(group)
    verdicts = _live_verdicts(points)
    by_id: Dict[str, List[orm.Dict]] = {}
    for point in points:
        attrs = point.base.attributes.all
        pid = format_point_id(attrs['rel_path'], attrs['load_kind'], attrs['load'],
                              attrs['angle'], attrs['speed'])
        by_id.setdefault(pid, []).extend(verdicts[point.pk])
    ruled: set = set()
    for pid, ruling in rulings.items():
        if ruling.error:
            report.warnings.append(f'integrity ruling ignored: {ruling.error}')
        elif not by_id.get(pid):
            report.unmatched.append(pid)
        else:
            ruled |= {v.uuid for v in by_id[pid] if _rule(report, pid, ruling, v)}
    if withdraw and _unreadable(rulings):
        report.warnings.append('no ruling withdrawn: the rulings file has entries '
                               'that cannot be read')
    elif withdraw:
        live = _ruled_by_verdict([v.uuid for vs in by_id.values() for v in vs])
        for pid, pid_verdicts in sorted(by_id.items()):
            for verdict in pid_verdicts:
                if verdict.uuid not in ruled and live.get(verdict.uuid):
                    _withdraw(report, pid, verdict, live[verdict.uuid])
    live = _ruled_by_verdict([v.uuid for vs in by_id.values() for v in vs])
    report.overrides = [
        (pid, node['original_status'], node['status'], node.get('note', ''))
        for pid, pid_verdicts in sorted(by_id.items()) for verdict in pid_verdicts
        for node in live.get(verdict.uuid, []) if node.get('overridden')]
    return report


def apply_rulings(label: str, rulings_path: Optional[Path] = None, *,
                  withdraw: Optional[bool] = None,
                  profile: Optional[str] = None) -> RulingReport:
    """Apply ``integrity_rulings.yaml`` to the set ``friction2d/<label>``.

    Reads ``<group.extras.source_root>/integrity_rulings.yaml`` by default
    (``integrity_rulings.load_rulings``; ``rulings_path`` may be the file or
    its directory). For each live verdict of a point a ruling names it runs
    :func:`apply_ruling` -- unless a live ruled verdict already carries
    ``friction2d_ruling_key == f'{verdict.uuid}|{sha256(canonical ruling)}'``.
    A new ruled verdict supersedes the verdict's earlier ruled and withdrawn
    ones. A quarantined point takes either ruling; any other needs
    ``override: true`` and a note.

    Overrides are never permanent: reading the set's own rulings file (the
    default), every live ruled verdict whose point no applicable ruling names
    any more -- entry deleted, ``override: true`` or the note removed -- is
    withdrawn (:func:`withdraw_ruling`), so the latest verdict follows the
    file. An explicit ``rulings_path`` that is not that file withdraws
    nothing unless ``withdraw=True``; neither does a file with unreadable
    entries.
    """
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel

    load_aiida_profile(profile)
    return rule_group(_set_group(label), rulings_path, withdraw=withdraw)
