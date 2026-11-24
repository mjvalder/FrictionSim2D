"""Import a campaign tree into the local AiiDA store: :func:`import_root`.

One import-only CalcJob (:mod:`frictionsim2d.aiida.calcjob`) per LAMMPS deck of every
run under the root. Identity never comes from a path or a label: a run is its
``provenance.json`` ``run_uuid``, a deck is ``f'{run_uuid}|{deck_name}'`` (extra
``friction2d_deck_key``), and the label only names the Group ``friction2d/<label>``.

Each deck carries a fingerprint (extra ``friction2d_deck_sha256``,
:func:`deck_fingerprint`). On re-import an unchanged deck is skipped; a changed one
gets a new CalcJob and the old one is :func:`supersede`\\ d, locally, as extras.

The load-point helpers (:class:`LoadPoint`, :func:`load_points`) are aiida-free, and
so is this module at import time: every aiida import is deferred.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple,
)

from ..core.observables import PES_BASE_NAME, PES_FAMILIES
from ..data.provenance import sha256_file
from ..postprocessing.read_data import (
    FILE_PATTERN_SHEET, FILE_PATTERN_SHEET_ALT, FILE_PATTERN_TIP,
    FILE_PATTERN_TIP_ALT, tier_of,
)
from ..run.naming import AFM_LIKE_MODELS
from . import aiida_setting
from .layout import (
    GROUP_PREFIX, LOAD_TOL, TIER_KEYS, build_dir, group_label, is_conflict, is_superseded,
    plain_files, reads_build,
)

if TYPE_CHECKING:  # pragma: no cover
    from aiida import orm

    from .layout import Deck

logger = logging.getLogger(__name__)

#: Every key `LoadPoint.tiers` may hold. `'retract'` is the indent family's
#: force-distance curve -- not a sampling tier, but it belongs to the same
#: load point and must be hashed with it, so it rides in the same dict.
POINT_KEYS = TIER_KEYS + ('retract',)

#: `indent_f30N_l1.csv` -- the retract force-distance curve. Not a DataReader
#: pattern (the DataReader never reads retract curves), so it is not lifted
#: from anywhere.
_INDENT_RE = re.compile(r'^indent_f(?P<load>\d+\.?\d*)N_l(?P<layers>\d+)\.csv$')


@dataclass(frozen=True)
class LoadPoint:
    """One results row of the store: everything that shares a load condition.

    `tiers` is keyed by :data:`POINT_KEYS`; the CalcJob parser stores the
    window/raw/retract entries on the point node and the per-atom tier on
    its own node.
    """
    family: str
    load_kind: str
    load: float
    angle: float
    speed: float
    layers: Optional[int]
    tiers: Dict[str, Path]


def _match_point(name: str, model: str):
    """`(load_kind, load, angle, speed, layers)` parsed from a results
    filename, or `None` when `name` matches none of DataReader's patterns.

    Reuses DataReader's own filename patterns (module level in
    `frictionsim2d.postprocessing.read_data`) so a file this
    grouping accepts is never one the DataReader itself would skip.
    `raw_`/`peratom_` share the new-format grammar with `friction_` apart
    from the prefix, so they are matched by substituting the `friction_`
    prefix DataReader's ALT patterns expect before testing them; the legacy
    `fc_ave_slide_*` form never had raw/peratom tiers, so it is only tried
    for the window tier.
    """
    tier = tier_of(name)
    probe = name
    if tier == 'raw':
        probe = 'friction_' + name[len('raw_'):]
    elif tier == 'peratom':
        probe = 'friction_' + name[len('peratom_'):]

    is_tip = model in AFM_LIKE_MODELS

    match = (FILE_PATTERN_TIP_ALT if is_tip else FILE_PATTERN_SHEET_ALT).match(probe)
    if match is not None:
        if is_tip:
            load, angle, speed, layers = match.groups()
            return 'force', float(load), float(angle), float(speed), int(layers)
        load, angle, speed = match.groups()
        return 'pressure', float(load), float(angle), float(speed), None

    if tier != 'window':
        return None
    match = (FILE_PATTERN_TIP if is_tip else FILE_PATTERN_SHEET).match(probe)
    if match is None:
        return None
    if is_tip:
        load, angle, speed, layers = match.groups()
        return 'force', float(load), float(angle), float(speed), int(layers)
    load, unit, angle, speed = match.groups()
    return ('pressure' if unit == 'GPa' else 'force',
            float(load), float(angle), float(speed), None)


def load_points(locator) -> List[LoadPoint]:
    """Group a run's results files into load points.

    The ONLY family-aware function in this module: a table keyed on
    `frictionsim2d.core.observables.FAMILIES`. Everything else -- set, run, results,
    retention, export -- is family-agnostic.
    """
    family = locator.tokens.family
    if not locator.results_dir.is_dir():
        return []

    if family in PES_FAMILIES:
        base = locator.results_dir / PES_BASE_NAME
        if not base.is_file():
            return []
        return [LoadPoint(family=family, load_kind='none', load=0.0, angle=0.0,
                          speed=0.0, layers=locator.tokens.layers,
                          tiers={'window': base})]

    model = locator.tokens.model
    grouped: Dict[Tuple[str, float, float, float, Optional[int]], Dict[str, Path]] = {}
    for path in sorted(locator.results_dir.iterdir()):
        if is_conflict(path):
            continue   # a Syncthing copy: the original is the result
        matched = _match_point(path.name, model)
        if matched is None:
            continue
        kind, load, angle, speed, layers = matched
        key = (kind, load, angle, speed, layers)
        bucket = grouped.setdefault(key, {})
        tier = tier_of(path.name)
        existing = bucket.get(tier)
        if existing is not None and existing != path:
            logger.warning(
                "%s: %r and %r both match the %r tier of load point "
                "(kind=%s, load=%s, angle=%s, speed=%s, layers=%s); keeping "
                "%r, ignoring %r.",
                locator.run_dir, existing.name, path.name, tier,
                kind, load, angle, speed, layers, existing.name, path.name)
            continue
        bucket[tier] = path

    if family == 'indent':
        for path in sorted(locator.results_dir.glob('indent_f*N_l*.csv')):
            match = _INDENT_RE.match(path.name)
            if match is None:
                continue
            load = float(match.group('load'))
            for key, tiers in grouped.items():
                if key[1] == load:
                    tiers['retract'] = path

    points = []
    for (kind, load, angle, speed, layers), tiers in sorted(grouped.items()):
        ordered = {t: tiers[t] for t in POINT_KEYS if t in tiers}
        points.append(LoadPoint(
            family=family, load_kind=kind, load=load, angle=angle, speed=speed,
            layers=layers if layers is not None else locator.tokens.layers,
            tiers=ordered))
    return points


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

class SetRootMismatch(ValueError):
    """A label already names a non-empty Group holding none of this root's runs.

    Compared on ``run_uuid`` values, never on paths: the same runs rsynced to
    a new path import under their label as before (``source_root`` is
    updated), while a different campaign under a used label is refused --
    merging it silently would give the label two unrelated campaigns.
    """


@dataclass
class ImportReport:
    """What one :func:`import_root` did, deck by deck.

    Deck entries are ``'<rel_path>:<deck_name>'``. ``points`` holds the uuid
    of every point node a new CalcJob created.
    ``failures`` are ``(rel_path or deck, message)``: a :class:`MissingRunUUID`
    or :class:`UnitMismatch` for one run lands here and the other runs go on.
    ``verdict_counts`` are the status counts of the verdicts this import's
    integrity pass made (:func:`frictionsim2d.aiida.steps.check_group`); ``rulings`` is
    the rulings pass that follows (:class:`~frictionsim2d.aiida.steps.RulingReport`)
    and ``overrides`` every live expert override of the set, read from the
    store, ``(point id, original status, new status, note)``. ``rechecked``
    are ``(point id, old status, new status)`` of the points whose stored
    verdict this import's changes outdated (their context changed).
    ``no_decks`` are the runs with no ``lammps/*.in`` deck: nothing of them
    is imported (not a failure).
    ``superseded_elsewhere`` are ``(deck, other label)``: a CalcJob this
    import superseded that another set also holds -- that set now holds a
    superseded deck.
    """
    label: str
    group_uuid: str = ''
    created_decks: List[str] = field(default_factory=list)
    skipped_decks: List[str] = field(default_factory=list)
    superseded_decks: List[str] = field(default_factory=list)
    already_imported: List[str] = field(default_factory=list)
    points: List[str] = field(default_factory=list)
    verdict_counts: Dict[str, int] = field(default_factory=dict)
    log_missing: List[str] = field(default_factory=list)
    no_version: List[str] = field(default_factory=list)
    unparseable: List[str] = field(default_factory=list)
    no_decks: List[str] = field(default_factory=list)
    failures: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    superseded_elsewhere: List[Tuple[str, str]] = field(default_factory=list)
    overrides: List[Tuple[str, str, str, str]] = field(default_factory=list)
    rechecked: List[Tuple[str, str, str]] = field(default_factory=list)
    rulings: Any = None
    dry_run: bool = False

    def ok(self) -> bool:
        """No run or deck failed."""
        return not self.failures

    def render(self) -> str:
        """The report as text: counts, then every notable entry."""
        head = f"{'DRY RUN: ' if self.dry_run else ''}set {self.label!r}"
        if self.group_uuid:
            head += f' ({self.group_uuid})'
        lines = [
            head,
            f'  decks created:       {len(self.created_decks)}',
            f'  decks skipped:       {len(self.skipped_decks)}',
            f'  decks superseded:    {len(self.superseded_decks)}',
            f'  already imported:    {len(self.already_imported)}',
            f'  points created:      {len(self.points)}',
            f'  log missing:         {len(self.log_missing)}',
            f'  no LAMMPS version:   {len(self.no_version)}',
            f'  unparseable runs:    {len(self.unparseable)}',
            f'  runs with no decks:  {len(self.no_decks)}',
        ]
        lines += [f'  verdict {k}: {v}' for k, v in sorted(self.verdict_counts.items())]
        lines += [f'  re-checked {pid}: {old} -> {new}'
                  for pid, old, new in self.rechecked]
        if self.rulings is not None:
            lines.append(f'  rulings applied:     {self.rulings.applied}')
            lines.append(f'  rulings replaced:    {self.rulings.replaced}')
            lines.append(f'  rulings skipped:     {self.rulings.skipped}')
            lines.append(f'  rulings withdrawn:   {len(self.rulings.withdrawn)}')
            lines += [f'  {line}' for line in self.rulings.changed_lines()]
            lines += [f'  ruling withdrawn: {pid}' for pid in self.rulings.withdrawn]
            lines += [f'  ruling unmatched: {pid}' for pid in self.rulings.unmatched]
            lines += [f'  ruling on a point not quarantined (needs override: true '
                      f'and a note): {pid}' for pid in self.rulings.not_quarantined]
            lines += [f'  ruling conflict {line}' for line in self.rulings.conflict_lines()]
            lines += [f'  {line}' for line in self.rulings.pending_lines()]
        lines += [f'  override {pid}: {old} -> {new} ({note})'
                  for pid, old, new, note in self.overrides]
        lines += [f'  already imported elsewhere: {d}' for d in self.already_imported]
        lines += [f'  no LAMMPS version (no Code): {d}' for d in self.no_version]
        lines += [f'  log missing: {d}' for d in self.log_missing]
        lines += [f'  unparseable: {d}' for d in self.unparseable]
        lines += [f'  no decks (nothing imported): {d}' for d in self.no_decks]
        lines += [f'  superseded {deck}, also held by set {other!r}'
                  for deck, other in self.superseded_elsewhere]
        lines += [f'  WARNING {w}' for w in self.warnings]
        lines += [f'  FAILED {where}: {message}' for where, message in self.failures]
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

_LOAD_DATA_RE = re.compile(r'^load_.+\.data$')


def _rel(path: Path, run_dir: Path) -> str:
    """``path`` relative to the run directory: the same wherever the tree sits."""
    return Path(os.path.relpath(path, run_dir)).as_posix()


def _covers(deck: 'Deck', load: float, angle: float, speed: float) -> bool:
    from .layout import points_for  # pylint: disable=import-outside-toplevel
    return bool(points_for(deck, [SimpleNamespace(load=load, angle=angle, speed=speed)]))


def _slide_outputs(locator, deck: 'Deck') -> List[Path]:
    """The output files a slide-like deck's CalcJob is built from: the result
    files of its points and their end snapshots. A results file that names no
    point (a PES grid, ``halted.txt``) belongs to every slide deck of the run;
    an indent retract curve to the deck covering its load."""
    from ..postprocessing.post_channels import final_data_path  # pylint: disable=import-outside-toplevel

    found: List[Path] = []
    model = locator.tokens.model
    for path in plain_files(locator.results_dir, '*'):
        matched = _match_point(path.name, model)
        indent = _INDENT_RE.match(path.name)
        if matched is not None:
            kind, load, angle, speed, _layers = matched
            if not _covers(deck, load, angle, speed):
                continue
            found.append(path)
            final = final_data_path(locator.data_dir, kind, load, angle)
            if final is not None:
                found.append(final)
        elif indent is None or 'load' not in deck.point_filter or \
                abs(float(indent['load']) - deck.point_filter['load']) <= LOAD_TOL:
            found.append(path)
    if locator.tokens.family in PES_FAMILIES:
        final = final_data_path(locator.data_dir, 'none', 0.0, 0.0)
        if final is not None:
            found.append(final)
    return sorted(set(found))


def _structure_files(locator, deck: 'Deck', decks: Sequence['Deck']) -> List[Tuple[Path, str]]:
    """``(path, stored name)`` of every structure the deck's CalcJob gets as
    ``structures``: ``build/*.lmp`` for the deck that reads the build, and for
    a slide deck the ``load_*.data`` the run's system deck outputs for it."""
    # pylint: disable=import-outside-toplevel
    from .calcjob import structures_for

    out: List[Tuple[Path, str]] = []
    if reads_build(deck, decks):
        out += [(p, p.name) for p in plain_files(build_dir(locator), '*.lmp')]
    if not deck.writes_structures and any(d.writes_structures for d in decks):
        loads = {p.name: p for p in plain_files(locator.data_dir, 'load_*.data')
                 if _LOAD_DATA_RE.match(p.name)}
        named = {name: SimpleNamespace(filename=name) for name in loads}
        out += [(loads[name], f'data/{name}') for name in structures_for(deck, named)]
    return out


def deck_fingerprint(locator, deck: 'Deck', decks: Sequence['Deck']) -> str:
    """The deck's fingerprint (extra ``friction2d_deck_sha256``).

    A sha256 over ``(run-relative name, sha256)`` of: the deck file, the
    files it includes, its log (or its absence), its retained output files
    (``load_*.data`` + ``hold_*.txt`` for a system deck, its points' result
    files and snapshots otherwise), the run's provenance folder key, and the
    stored sha256 of every structure handed to it. Path-free, so a moved
    tree has the same fingerprints; a log that arrives later changes it.
    """
    # pylint: disable=import-outside-toplevel
    from .calcjob import included_files, run_config_file
    from .store import provenance_key, staged_sha256

    run_dir = Path(locator.run_dir)
    if deck.writes_structures:
        outputs = (plain_files(locator.data_dir, 'load_*.data')
                   + plain_files(locator.data_dir, 'hold_*.txt'))
    else:
        outputs = _slide_outputs(locator, deck)
    entries = [('deck', sha256_file(deck.path))]
    entries += [(f'include:{_rel(p, run_dir)}', sha256_file(p))
                for p in included_files(locator, deck)]
    entries.append(('log', sha256_file(deck.log_path) if deck.log_path else '-'))
    entries += [(f'output:{_rel(p, run_dir)}', sha256_file(p)) for p in outputs]
    entries.append(('provenance', provenance_key(locator.provenance_file,
                                                 locator.provenance_dir,
                                                 run_config_file(locator))))
    entries += [('structure', sha)
                for sha in sorted(staged_sha256(p, name)
                                  for p, name in _structure_files(locator, deck, decks))]
    digest = hashlib.sha256()
    for name, sha in sorted(entries):
        digest.update(f'{name}\0{sha}\n'.encode('utf-8'))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Supersession (local only)
# ---------------------------------------------------------------------------

def _superseded_closure(calc: 'orm.CalcJobNode') -> Set[int]:
    """pks of ``calc``'s CREATE outputs and, transitively, of the outputs of
    every calcfunction that consumed one. INPUT_CALC consumers that are
    CalcJobs (a slide deck reading a structure) are not followed."""
    # pylint: disable=import-outside-toplevel
    from aiida import orm
    from aiida.common.links import LinkType

    create = {'type': LinkType.CREATE.value}
    qb = orm.QueryBuilder().append(orm.CalcJobNode, filters={'id': calc.pk}, tag='c')
    qb.append(orm.Data, with_incoming='c', edge_filters=create, project='id')
    seen = set(qb.all(flat=True))
    frontier = set(seen)
    while frontier:
        qb = orm.QueryBuilder().append(orm.Data, filters={'id': {'in': sorted(frontier)}},
                                       tag='d')
        qb.append(orm.CalcFunctionNode, with_incoming='d', tag='f',
                  edge_filters={'type': LinkType.INPUT_CALC.value})
        qb.append(orm.Data, with_incoming='f', edge_filters=create, project='id')
        frontier = set(qb.all(flat=True)) - seen
        seen |= frontier
    return seen


def supersede(calc: 'orm.CalcJobNode') -> List[int]:
    """Mark ``calc`` replaced, with its CREATE outputs and their calcfunction
    descendants: extras ``superseded=True``, ``superseded_at`` (ISO, UTC).

    Local to this profile: extras are an index, not provenance. Returns the
    pks marked.
    """
    # pylint: disable=import-outside-toplevel
    from datetime import datetime, timezone

    from aiida import orm

    extras = {'superseded': True,
              'superseded_at': datetime.now(timezone.utc).isoformat()}
    pks = [calc.pk, *sorted(_superseded_closure(calc))]
    for pk in pks:
        orm.load_node(pk).base.extras.set_many(extras)
    return pks


# ---------------------------------------------------------------------------
# import_root
# ---------------------------------------------------------------------------

@dataclass
class _Known:
    """A stored deck CalcJob, from the one up-front projection query."""
    pk: int
    fingerprint: Optional[str]


@dataclass
class _Import:
    """The state of one :func:`import_root` call."""
    report: ImportReport
    machine: Optional[str]
    group: Any = None
    group_pks: Set[int] = field(default_factory=set)
    known: Dict[str, List[_Known]] = field(default_factory=dict)
    index: Any = None
    computers: Dict[str, Any] = field(default_factory=dict)
    settings_file: Optional[Path] = None
    integrity: Optional[Mapping[str, Any]] = None


def _group_runs(label: str) -> Tuple[Any, Set[int], Set[str]]:
    """``(group or None, CalcJob pks in it, run_uuids in it)``."""
    # pylint: disable=import-outside-toplevel
    from aiida import orm
    from aiida.common.exceptions import NotExistent

    from .store import RUN_UUID_EXTRA

    try:
        group = orm.Group.collection.get(label=group_label(label))
    except NotExistent:
        return None, set(), set()
    qb = orm.QueryBuilder().append(orm.Group, filters={'id': group.pk}, tag='g')
    qb.append(orm.CalcJobNode, with_group='g',
              project=['id', f'extras.{RUN_UUID_EXTRA}'])
    rows = qb.all()
    return group, {pk for pk, _u in rows}, {u for _pk, u in rows if u}


def _check_root(label: str, group: Any, held: Set[str], run_uuids: Set[str]) -> None:
    """Raise :class:`SetRootMismatch` when ``label`` names a non-empty Group
    sharing no ``run_uuid`` with this root."""
    if group is None or not run_uuids or not group.count() or held & run_uuids:
        return
    raise SetRootMismatch(
        f'set {label!r} already holds {len(held)} other run(s) and none of this '
        f"tree's {len(run_uuids)}: it is a different campaign. Pass --label to "
        f'give this tree its own set, or delete the old one with '
        f'`FrictionSim2D aiida delete {label}`.')


def _known_decks(run_uuids: Iterable[str]) -> Dict[str, List[_Known]]:
    """deck key -> the live (non-superseded) CalcJobs of it, oldest first."""
    # pylint: disable=import-outside-toplevel
    from aiida import orm

    from .store import CALCJOB_PROCESS_TYPE, DECK_KEY_EXTRA, DECK_SHA_EXTRA, RUN_UUID_EXTRA

    uuids = sorted(set(run_uuids))
    known: Dict[str, List[_Known]] = {}
    if not uuids:
        return known
    qb = orm.QueryBuilder().append(
        orm.CalcJobNode, tag='c',
        filters={'process_type': CALCJOB_PROCESS_TYPE,
                 f'extras.{RUN_UUID_EXTRA}': {'in': uuids}},
        project=['id', f'extras.{DECK_KEY_EXTRA}', f'extras.{DECK_SHA_EXTRA}',
                 'extras.superseded'])
    qb.order_by({'c': {'id': 'asc'}})
    for pk, key, sha, superseded in qb.iterall():
        if key and not superseded:
            known.setdefault(key, []).append(_Known(pk, sha))
    return known


def _group_extras(root: Path, records: Sequence[dict]) -> Dict[str, Any]:
    """The Group extras: what the campaign is and where it came from."""
    import socket  # pylint: disable=import-outside-toplevel

    def one_or_all(values):
        values = sorted({str(v) for v in values if v is not None})
        return values[0] if len(values) == 1 else values

    materials = {m.get('name') for r in records for m in (r.get('materials') or [])
                 if isinstance(m, dict)}
    dirty = {bool(r.get('git_dirty')) for r in records if 'git_dirty' in r}
    return {
        'model': one_or_all(r.get('model') for r in records),
        'materials': sorted(m for m in materials if m),
        'source_root': str(Path(root).resolve()),
        'source_host': socket.gethostname(),
        'git_commit': one_or_all(r.get('git_commit') for r in records),
        'git_dirty': any(dirty),
    }


def _open_group(label: str, root: Path, records: Sequence[dict]):
    from datetime import datetime, timezone  # pylint: disable=import-outside-toplevel

    from aiida import orm  # pylint: disable=import-outside-toplevel

    group, created = orm.Group.collection.get_or_create(label=group_label(label))
    extras = _group_extras(root, records)
    if created or 'created' not in group.base.extras.all:
        extras['created'] = datetime.now(timezone.utc).isoformat()
    group.base.extras.set_many(extras)
    return group


def _computer(state: _Import, settings: dict, config: Optional[dict] = None):
    """The machine Computer of a run: ``import_root(machine=)`` >
    ``settings.aiida.machine`` (both in ``state.machine``) > the run's
    ``settings.hpc.hpc_host`` > ``'localhost'``. The run's settings are
    ``provenance/settings.yaml``, else ``config['settings']`` (as
    ``blocks.split_blocks`` reads them). A label already registered with
    another host or scheduler is reused, with a warning in the report."""
    from .store import machine_computer, machine_mismatch  # pylint: disable=import-outside-toplevel

    settings = settings or (config or {}).get('settings') or {}
    hpc = settings.get('hpc') if isinstance(settings.get('hpc'), dict) else {}
    name = state.machine or hpc.get('hpc_host') or 'localhost'
    if name not in state.computers:
        problem = machine_mismatch(name, hpc)
        if problem and problem not in state.report.warnings:
            state.report.warnings.append(problem)
        state.computers[name] = machine_computer(name, hpc)
    return state.computers[name]


def _other_labels(pk: int, label: str) -> List[str]:
    """The other ``friction2d/*`` set labels holding node ``pk``."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    qb = orm.QueryBuilder().append(orm.Node, filters={'id': pk}, tag='n')
    qb.append(orm.Group, with_node='n', project='label',
              filters={'label': {'like': f'{GROUP_PREFIX}%'}})
    return sorted(g[len(GROUP_PREFIX):] for g in qb.all(flat=True)
                  if g != group_label(label))


def _has_dead_structures(calc) -> bool:
    """Whether any of ``calc``'s ``structures`` inputs is superseded."""
    if 'structures' not in calc.inputs:
        return False
    return any(is_superseded(node)
               for node in calc.inputs.structures.values())


def _record_new(state: _Import, name: str, node) -> None:
    """Report what a new deck CalcJob produced."""
    report = state.report
    report.created_decks.append(name)
    info = node.outputs.run_info.get_dict() if 'run_info' in node.outputs else {}
    if info.get('log_missing', True):
        report.log_missing.append(name)
    if 'code' not in node.inputs:
        report.no_version.append(name)
    if 'points' in node.outputs:
        report.points += [point.uuid for _label, point in sorted(node.outputs.points.items())]


class _RunState:  # pylint: disable=too-few-public-methods
    """One run's progress through :func:`_import_run`."""

    def __init__(self, locator, run_uuid: str) -> None:
        self.locator = locator
        self.run_uuid = run_uuid
        self.inputs = None          # calcjob.RunInputs, built on first need
        self.handed: Dict[str, Any] = {}
        self.system_replaced = False


def _deck_name(locator, deck) -> str:
    return f'{locator.tokens.rel_path}:{deck.name}'


def _reuse(state: _Import, run: _RunState, deck, name: str, live: _Known) -> None:
    """An unchanged deck: skipped in this group, or added to it from another."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    calc = orm.load_node(live.pk)
    if live.pk in state.group_pks:
        state.report.skipped_decks.append(name)
    else:
        state.report.already_imported.append(name)
        if not state.report.dry_run:
            state.group.add_nodes([calc])
            state.group_pks.add(live.pk)
    if deck.writes_structures and 'structures' in calc.outputs:
        run.handed.update(dict(calc.outputs.structures.items()))


#: ``LammpsFrictionCalcJob``'s ``ERROR_PARSING_FAILED``.
_PARSING_FAILED = 330


def _import_deck(state: _Import, run: _RunState, deck, decks,
                 fingerprint: str, stale: List[_Known]) -> bool:
    """Import one deck as a new CalcJob; supersede ``stale``. False on failure."""
    # pylint: disable=import-outside-toplevel
    from aiida import orm

    from .calcjob import import_deck, run_inputs, structures_for
    from .store import DECK_KEY_EXTRA, DECK_SHA_EXTRA, deck_key

    name = _deck_name(run.locator, deck)
    if run.inputs is None:
        run.inputs = run_inputs(run.locator, index=state.index)
    computer = _computer(state, run.inputs.settings, run.inputs.config)
    given = None if deck.writes_structures else structures_for(deck, run.handed)
    try:
        _outputs, node = import_deck(run.locator, deck, computer=computer,
                                     run=run.inputs, structures=given, decks=decks,
                                     index=state.index)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception('Import failed for deck %s', name)
        state.report.failures.append((name, f'{type(exc).__name__}: {exc}'))
        return False
    if node.is_excepted or not node.is_finished:
        state.report.failures.append((name, f'CalcJob {node.pk} did not finish'))
        return False
    key = {DECK_KEY_EXTRA: deck_key(run.run_uuid, deck.name)}
    state.group.add_nodes([node])
    state.group_pks.add(node.pk)
    if node.exit_status == _PARSING_FAILED:
        # No fingerprint and nothing superseded: the next import retries the
        # deck, and a good earlier CalcJob stays live meanwhile.
        node.base.extras.set_many(key)
        state.report.failures.append((name, node.exit_message or 'parsing failed'))
        return not deck.writes_structures
    node.base.extras.set_many({**key, DECK_SHA_EXTRA: fingerprint})
    for old in stale:
        supersede(orm.load_node(old.pk))
        state.report.superseded_elsewhere += [
            (name, other) for other in _other_labels(old.pk, state.report.label)]
    if stale:
        state.report.superseded_decks.append(name)
        run.system_replaced = run.system_replaced or deck.writes_structures
    if deck.writes_structures and 'structures' in node.outputs:
        run.handed.update(dict(node.outputs.structures.items()))
    _record_new(state, name, node)
    return True


def _plan_deck(state: _Import, run: _RunState, deck, stale: List[_Known]) -> None:
    """Dry run: classify a deck that would get a new CalcJob."""
    name = _deck_name(run.locator, deck)
    state.report.created_decks.append(name)
    for old in stale:
        state.report.superseded_elsewhere += [
            (name, other) for other in _other_labels(old.pk, state.report.label)]
    if stale:
        state.report.superseded_decks.append(name)
        run.system_replaced = run.system_replaced or deck.writes_structures


def _import_run(state: _Import, run: _RunState) -> None:
    """Every deck of one run, system decks first."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .layout import discover_decks  # pylint: disable=import-outside-toplevel
    from .store import deck_key  # pylint: disable=import-outside-toplevel

    decks = discover_decks(run.locator)
    if not decks:
        # No lammps/*.in (a results-only run): nothing to import, but never silently.
        state.report.no_decks.append(run.locator.tokens.rel_path)
        return
    for deck in decks:
        name = _deck_name(run.locator, deck)
        fingerprint = deck_fingerprint(run.locator, deck, decks)
        live = state.known.get(deck_key(run.run_uuid, deck.name), [])
        current = live[-1] if live else None
        # A slide deck reading a replaced system deck's structures re-imports
        # even when its own files are unchanged: a live deck never consumes
        # superseded inputs.
        cascaded = (not deck.writes_structures and run.system_replaced
                    and _receives_structures(deck, decks))
        if (current is not None and current.fingerprint == fingerprint and not cascaded
                and (state.report.dry_run
                     or not _has_dead_structures(orm.load_node(current.pk)))):
            _reuse(state, run, deck, name, current)
            continue
        if state.report.dry_run:
            _plan_deck(state, run, deck, live)
        elif not _import_deck(state, run, deck, decks, fingerprint, live):
            if deck.writes_structures:
                state.report.failures.append(
                    (run.locator.tokens.rel_path,
                     f'system deck {deck.name!r} failed; the run\'s other decks '
                     f'were not imported'))
                return


def _receives_structures(deck, decks) -> bool:
    """Whether a slide deck is handed structures by a system deck of its run."""
    return not deck.writes_structures and any(d.writes_structures for d in decks)


def _read_runs(locators, report: ImportReport) -> List[Tuple[Any, str, dict]]:
    """``(locator, run_uuid, record)`` of every run with a usable identity
    and a config matching its ``config_hash``
    (:func:`~frictionsim2d.aiida.calcjob.run_config_file`; a replacement config is a
    warning); the others go to ``failures``. A dry run also checks the units
    here (a real import does it in :func:`~frictionsim2d.aiida.calcjob.run_inputs`,
    splitting each run once)."""
    # pylint: disable=import-outside-toplevel
    import json

    from .block_schema import UnitMismatch
    from .calcjob import ConfigMismatch, MissingRunUUID, check_run, read_run_uuid, run_config_file

    runs = []
    for locator in locators:
        rel = locator.tokens.rel_path
        try:
            run_uuid = (check_run(locator) if report.dry_run
                        else read_run_uuid(locator))
            record = json.loads(Path(locator.provenance_file).read_text(encoding='utf-8'))
            config_file = run_config_file(locator, record)
        except (MissingRunUUID, UnitMismatch, ConfigMismatch) as exc:
            report.failures.append((rel, f'{type(exc).__name__}: {exc}'))
            continue
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception('Could not read run %s', rel)
            report.failures.append((rel, f'{type(exc).__name__}: {exc}'))
            continue
        if config_file is not None:
            report.warnings.append(
                f'{rel}: provenance/config.json does not match the config_hash of '
                f'provenance.json; using {config_file}, which does')
        runs.append((locator, run_uuid, record))
    return runs


def import_root(root: Path, label: Optional[str] = None, *,
                machine: Optional[str] = None, profile: Optional[str] = None,
                dry_run: bool = False,
                settings_file: Optional[Path] = None,
                integrity: Optional[Mapping[str, Any]] = None) -> ImportReport:
    """Import every run under ``root`` into ``friction2d/<label>``: one
    CalcJob per deck, keyed on ``run_uuid|deck_name``.

    Idempotent: a deck whose fingerprint (:func:`deck_fingerprint`) is
    unchanged is skipped, or -- when it was imported under another label --
    added to this Group (``already_imported``). A changed deck gets a new
    CalcJob and the old one is :func:`supersede`\\ d. A run without a
    ``run_uuid``, or whose spring is not in eV/A^2, is reported in
    ``failures`` and the other runs go on.

    Args:
        root: The simulation root (``simulation_<stamp>/``).
        label: The set's name (default: ``root``'s name).
        machine: The machine label; default ``settings.aiida.machine``, else
            each run's ``settings.hpc.hpc_host``, else ``'localhost'``.
        profile: The AiiDA profile (default: the configured one).
        dry_run: Classify every deck against the store; create nothing.
        settings_file: The settings file whose ``aiida.machine`` and
            ``integrity`` thresholds apply (default: the per-user one).
        integrity: Single integrity thresholds overriding those of the settings
            (``{'hole_bonds': 2.0, 'margins': {'rule': 0.1}}``).

    Raises:
        SetRootMismatch: ``label`` names a non-empty set holding none of the
            root's runs.
    """
    # pylint: disable=import-outside-toplevel
    from ..run.tree import discover_runs, unparseable_runs
    from . import load_aiida_profile
    from .store import StoreIndex

    root = Path(root)
    label = label or root.name
    report = ImportReport(label=label, dry_run=dry_run)
    report.unparseable = unparseable_runs(root)
    locators = discover_runs(root)
    if not locators:
        report.failures.append((str(root), 'no provenance.json found'))
        return report

    load_aiida_profile(profile)
    runs = _read_runs(locators, report)
    group, group_pks, held = _group_runs(label)
    _check_root(label, group, held, {u for _loc, u, _r in runs})

    state = _Import(report=report,
                    machine=machine or aiida_setting('machine', None, settings_file) or None,
                    group_pks=group_pks,
                    known=_known_decks(u for _loc, u, _r in runs),
                    settings_file=settings_file, integrity=integrity)
    if not dry_run:
        state.group = _open_group(label, root, [r for _loc, _u, r in runs])
        state.index = StoreIndex.load()
    elif group is not None:
        state.group = group
    if state.group is not None:
        report.group_uuid = str(state.group.uuid)

    for locator, run_uuid, _record in runs:
        try:
            _import_run(state, _RunState(locator, run_uuid))
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception('Import failed for run %s', locator.tokens.rel_path)
            report.failures.append((locator.tokens.rel_path,
                                    f'{type(exc).__name__}: {exc}'))
    if not dry_run:
        _integrity_pass(state)
    elif state.group is not None:
        _dry_rulings(state, root)
    return report


def _dry_rulings(state: _Import, root: Path) -> None:
    """A dry run's rulings report: what the rulings pass would do to the
    set as stored, with ``root`` (the root being imported) as its root (no
    check is run, nothing is written)."""
    rule_group = import_module(f'{__package__}.steps').rule_group
    report = state.report
    report.rulings = rule_group(state.group, dry_run=True, root=root)
    report.overrides = list(report.rulings.overrides)
    report.warnings += report.rulings.warnings


def _integrity_pass(state: _Import) -> None:
    """A verdict for every live point of the set that has none, then the
    set's rulings (:mod:`frictionsim2d.aiida.steps`)."""
    # `steps` imports aiida at module level and this module must stay importable
    # without it, so it is looked up at call time (which also avoids the import
    # cycle, as `calcjob` does with `load_points`).
    steps = import_module(f'{__package__}.steps')
    check_group, rule_group = steps.check_group, steps.rule_group

    report = state.report
    failed: List[Tuple[str, str]] = []
    report.verdict_counts = check_group(state.group, failures=failed,
                                        warnings=report.warnings,
                                        rechecked=report.rechecked,
                                        settings_file=state.settings_file,
                                        integrity=state.integrity)
    report.failures += [(f'point {uuid}', f'integrity check: {message}')
                        for uuid, message in failed]
    report.rulings = rule_group(state.group)
    report.overrides = list(report.rulings.overrides)
    report.warnings += report.rulings.warnings
