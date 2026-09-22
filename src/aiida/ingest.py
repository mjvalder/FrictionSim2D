"""One ingest path, idempotent.

Replaces `integration.py`'s three overlapping importers (`register_*`,
`import_results_to_aiida`, `import_simulation_set`) with a single function that
`run --aiida`, `aiida import` and the CalcJob parser all call.

No process nodes are faked on import: the nodes link by `set_uuid`/`run_uuid`
attributes and share an AiiDA `Group`. The CalcJob path produces real provenance
links for free; import does not pretend to. That is a deliberate choice
(spec §4.1).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..postprocessing.read_data import (
    FILE_PATTERN_SHEET, FILE_PATTERN_SHEET_ALT, FILE_PATTERN_TIP,
    FILE_PATTERN_TIP_ALT, tier_of,
)
from ..run.naming import AFM_LIKE_MODELS
from .data_keys import TIER_KEYS, content_hash, is_superseded

logger = logging.getLogger(__name__)

GROUP_PREFIX = 'friction2d'

#: Every key `LoadPoint.tiers` may hold. `'retract'` is the indent family's
#: force-distance curve -- not a sampling tier, but it belongs to the same
#: load point and must be hashed with it, so it rides in the same dict.
#: `FrictionResultsData.add_result_files` (Task 11) iterates `TIER_KEYS` only,
#: so a `'retract'` entry is stored via `aux`; Task 16's `create_results`
#: splits `point.tiers` on exactly that tuple.
POINT_KEYS = TIER_KEYS + ('retract',)

#: `indent_f30N_l1.csv` -- the retract force-distance curve. Not a DataReader
#: pattern (the DataReader never reads retract curves), so it is not lifted
#: from anywhere.
_INDENT_RE = re.compile(r'^indent_f(?P<load>\d+\.?\d*)N_l(?P<layers>\d+)\.csv$')
#: The base grid. `src.builders.sheetonsheet.pes_sheet._grid_variants` also
#: writes `_PES_REFINE_NAME` alongside it when `pes.grid_n_refine` is set --
#: a SECOND grid at a finer resolution for the same (family, load=0) point,
#: not a second point: both would carry the identical identity (load_kind=
#: 'none', load=angle=speed=0, layers=None), so treating it as its own point
#: would make each ingest supersede the other and never settle. It rides
#: along as an aux file on the one point instead.
_PES_BASE_NAME = 'pes_scan.csv'
_PES_REFINE_NAME = 'pes_scan_refine.csv'
_PES_FAMILIES = ('pes_tip', 'pes_sheet')


def group_label(label: str) -> str:
    """The AiiDA group every node of a set belongs to."""
    return f'{GROUP_PREFIX}/{label}'


def _max_stored_file_mb(settings_file=None) -> int:
    """`settings.aiida.max_stored_file_mb`, defaulting to 20 (spec §4.1)."""
    from ..core.config import load_settings  # pylint: disable=import-outside-toplevel
    try:
        return int(load_settings(settings_file=settings_file).aiida.max_stored_file_mb)
    except (OSError, ValueError, AttributeError):
        return 20


def _group_for(label: str):
    from aiida.orm import Group  # pylint: disable=import-outside-toplevel
    group, _created = Group.collection.get_or_create(label=group_label(label))
    return group


class SetRootMismatch(ValueError):
    """A label already names a set built from a different tree.

    Merging them silently is the failure this exists to prevent: `n_runs`,
    `materials`, `git_commit` and the stored `settings.yaml` freeze from
    whichever tree arrived first; a `rel_path` present in both returns tree A's
    run node with `created=False`, so `add_run_files` never re-runs and tree
    B's inputs are discarded while its results attach anyway; and `to_record`
    then stamps tree A's `git_commit` and `pot_sha256` onto tree B's central
    rows. It also fires on the ordinary "I edited the config and re-ran into a
    fresh timestamped root", which is exactly when it matters most.
    """


def _find_set(root: Path, label: str):
    """The set node for `label`, if it already exists; `None` otherwise.

    Read-only -- never creates a node or a Group. Still raises
    :class:`SetRootMismatch` when the existing set was built from a different
    `source_root`, exactly like `find_or_create_set`: a dry run must surface
    the same conflict a real ingest would, not silently classify every run
    under the mismatched label as "would create".
    """
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    from .data import FrictionSimulationSetData  # pylint: disable=import-outside-toplevel

    root = Path(root).resolve()
    qb = QueryBuilder()
    qb.append(FrictionSimulationSetData, filters={'attributes.label': label})
    existing = qb.first(flat=True)
    if existing is not None and existing.source_root and existing.source_root != str(root):
        raise SetRootMismatch(
            f"set {label!r} was ingested from {existing.source_root!r}; "
            f"{str(root)!r} is a different tree. Pass --label to give this "
            f"tree its own set, or delete the old one with "
            f"`FrictionSim2D aiida delete {label}`.")
    return existing


def find_or_create_set(root: Path, label: str):
    """The set node for `label`, created from `root` if it does not exist.

    Raises :class:`SetRootMismatch` when `label` already names a set built from
    a different `source_root`. Re-ingesting the SAME tree under the same label
    is the idempotent path and is allowed.
    """
    existing = _find_set(root, label)
    if existing is not None:
        return existing

    from .data import FrictionSimulationSetData  # pylint: disable=import-outside-toplevel

    node = FrictionSimulationSetData()
    node.set_from_root(Path(root).resolve(), label)
    node.store()
    _group_for(label).add_nodes([node])
    return node


def _find_run(set_node, locator) -> Optional[object]:
    """The run node for `(set_uuid, rel_path)`, if it already exists; `None`
    otherwise. Read-only -- never creates a node."""
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    from .data import FrictionSimulationData  # pylint: disable=import-outside-toplevel

    qb = QueryBuilder()
    qb.append(FrictionSimulationData, filters={
        'attributes.set_uuid': str(set_node.uuid),
        'attributes.rel_path': locator.tokens.rel_path,
    })
    return qb.first(flat=True)


def find_or_create_run(set_node, locator) -> Tuple[object, bool]:
    """The run node for `(set_uuid, rel_path)`; `(node, created)`."""
    existing = _find_run(set_node, locator)
    if existing is not None:
        return existing, False

    from .data import FrictionSimulationData  # pylint: disable=import-outside-toplevel

    node = FrictionSimulationData()
    node.set_from_locator(locator, set_uuid=str(set_node.uuid),
                          set_label=set_node.label)
    node.add_run_files(locator, max_mb=_max_stored_file_mb())
    node.store()
    node.status = 'imported'
    _group_for(set_node.label).add_nodes([node])
    return node, True


def find_run_by_rel_path(run_dir: Path, root: Path) -> Optional[object]:
    """The stored run node for a built directory, or None.

    `submit_batch` used to CREATE a node per directory at submit time; the
    nodes now exist already (`run --aiida` -> `ingest_root`), so submitting
    looks one up. Returning None is not an error: a directory submitted
    without `--aiida` simply has no node to track against.

    Scoped to the set(s) whose `source_root` is `root`: `rel_path` alone is
    not unique across sets (two different trees can each have a run at, say,
    `K300/afm_.../layer1`), so an unscoped lookup would return an arbitrary
    node instead of the one that actually belongs to this tree.
    """
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    from .data import FrictionSimulationData, FrictionSimulationSetData  # pylint: disable=import-outside-toplevel

    root = Path(root).resolve()
    rel = Path(run_dir).resolve().relative_to(root).as_posix()

    set_qb = QueryBuilder()
    set_qb.append(FrictionSimulationSetData,
                  filters={'attributes.source_root': str(root)})
    set_uuids = [str(node.uuid) for node in set_qb.all(flat=True)]
    if not set_uuids:
        return None

    qb = QueryBuilder()
    qb.append(FrictionSimulationData, filters={
        'attributes.rel_path': rel,
        'attributes.set_uuid': {'in': set_uuids},
    })
    return qb.first(flat=True)


def find_results(run_node, point) -> Optional[object]:
    """The stored results node for this run at `point`'s CONDITIONS, if any.

    Keyed on `(load_kind, load, angle, speed, layers)`, not on the content
    hash, because those five ARE the load point's identity. Hash-keying let a
    re-run of the same conditions create a second node with nothing superseding
    the first; `time_series_hash TEXT UNIQUE` then rejected one of them at
    insert time (and every incomplete row carries `time_series_hash = ''`, so
    the second incomplete row in the whole table collided too), which
    `db upload` swallowed as `rejected` with a warning.

    A `{'attributes.layers': None}` QueryBuilder filter raises `TypeError` on
    SQLite, so a sheet-family point (`layers is None`) omits that filter from
    the query and matches it in Python on the candidates instead (R4).
    Superseded nodes (R9) are excluded the same way: a stale node must never
    be returned again, or a re-ingest would compare against it, or supersede
    it a second time.

    Candidates are ordered newest-first (`ctime desc`): if a create-then-
    supersede pair was ever interrupted between the two (a crash, a killed
    process) and more than one non-superseded node exists for the same
    conditions, the newest one is the one a subsequent ingest should compare
    against and, if still stale, supersede -- not an arbitrary DB-order pick.
    """
    live = _live_results(run_node, point)
    return live[0] if live else None


def _live_results(run_node, point) -> List:
    """Every non-superseded results node of `run_node` at `point`'s
    conditions, newest first. `point` is anything with `load_kind`, `load`,
    `angle`, `speed` and `layers` -- a `LoadPoint` or a results node."""
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    from .data import FrictionResultsData  # pylint: disable=import-outside-toplevel

    filters = {
        'attributes.run_uuid': str(run_node.uuid),
        'attributes.load_kind': point.load_kind,
        'attributes.load': point.load,
        'attributes.angle': point.angle,
        'attributes.speed': point.speed,
    }
    if point.layers is not None:
        filters['attributes.layers'] = point.layers
    qb = QueryBuilder()
    qb.append(FrictionResultsData, filters=filters)
    qb.order_by({FrictionResultsData: [{'ctime': 'desc'}]})
    return [node for node in qb.all(flat=True)
            if node.layers == point.layers and not is_superseded(node)]


def supersede_older(run_node, nodes) -> List:
    """Supersede every other live results node at the conditions of `nodes`.

    The CalcJob path's counterpart to `ingest_run`'s `find_results` +
    `supersede`: the parser hands the engine fresh nodes without looking at
    what the run already holds, so after a resubmit two live nodes would
    exist per load point. Called from `LammpsFrictionCalcJob.on_terminated`
    once the new nodes are stored. `nodes` themselves are never superseded.
    Returns the nodes it superseded.
    """
    keep = {str(node.uuid) for node in nodes}
    superseded = []
    for node in nodes:
        for other in _live_results(run_node, node):
            if str(other.uuid) in keep:
                continue
            supersede(other, run_node)
            keep.add(str(other.uuid))
            superseded.append(other)
    return superseded


def results_exist(run_node, point) -> bool:
    """True when this run already holds an up-to-date node for `point`.

    Up to date means "same conditions AND same bytes". Same conditions with
    different bytes is a re-run: the caller supersedes rather than skips.
    """
    existing = find_results(run_node, point)
    return existing is not None and existing.content_sha256 == point.content_sha256


def supersede(existing, run_node) -> None:  # pylint: disable=unused-argument
    """Mark an older results node as replaced.

    Attributes are immutable once stored, so this is an EXTRA. The node is not
    deleted: it may already have been uploaded centrally and its `result_uuid`
    is the pointer a reader was given. `Friction2DDB.query` filters it out.
    """
    from datetime import datetime, timezone  # pylint: disable=import-outside-toplevel

    existing.base.extras.set('superseded', True)
    existing.base.extras.set('superseded_at',
                             datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class LoadPoint:
    """One results row of the store: everything that shares a load condition.

    `tiers` is keyed by :data:`POINT_KEYS`; `create_results` stores the
    :data:`TIER_KEYS` entries as tier files and `'retract'` as an aux file.
    """
    family: str
    load_kind: str
    load: float
    angle: float
    speed: float
    layers: Optional[int]
    tiers: Dict[str, Path]
    aux: Tuple[Path, ...]
    content_sha256: str


def _aux_files(locator) -> Tuple[Path, ...]:
    """Logs and hold/final artefacts that ride along with every point of a run.

    Returned as absolute paths under the run root (not basenames), per R30:
    Task 16 stores them via `path.relative_to(run_dir)`, which needs the full
    path to compute.
    """
    out = sorted(locator.run_dir.glob('log*.lammps'))
    out += sorted(locator.lammps_dir.glob('run_*.log'))
    if locator.data_dir.is_dir():
        out += sorted(locator.data_dir.glob('hold_*.txt'))
        out += sorted(locator.data_dir.glob('final_*.data'))
    return tuple(out)


def _match_point(name: str, model: str):
    """`(load_kind, load, angle, speed, layers)` parsed from a results
    filename, or `None` when `name` matches none of DataReader's patterns.

    Reuses DataReader's own filename patterns (module level in
    `src.postprocessing.read_data`; R29/U7 "one parser") so a file this
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
    `src.core.observables.FAMILIES`. Everything else -- set, run, results,
    retention, export -- is family-agnostic.
    """
    family = locator.tokens.family
    if not locator.results_dir.is_dir():
        return []
    aux = _aux_files(locator)

    if family in _PES_FAMILIES:
        base = locator.results_dir / _PES_BASE_NAME
        if not base.is_file():
            return []
        refine = locator.results_dir / _PES_REFINE_NAME
        point_aux = aux + ((refine,) if refine.is_file() else ())
        return [LoadPoint(family=family, load_kind='none', load=0.0, angle=0.0,
                          speed=0.0, layers=locator.tokens.layers,
                          tiers={'window': base}, aux=point_aux,
                          content_sha256=content_hash([base]))]

    model = locator.tokens.model
    grouped: Dict[Tuple[str, float, float, float, Optional[int]], Dict[str, Path]] = {}
    for path in sorted(locator.results_dir.iterdir()):
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
            tiers=ordered, aux=aux,
            content_sha256=content_hash(ordered.values())))
    return points


@dataclass
class IngestReport:
    """What one ingest did, run by run and point by point.

    `aiida import` (Task 20) prints `render()` and exits non-zero when
    `ok()` is False; it adds its OWN non-zero exit on `empty_runs` /
    `unparseable` (R8) -- `ok()` here stays failures-only, because an
    all-empty tree (a run directory `run --aiida` has just created, with no
    results written yet) must be able to ingest successfully.

    Every list is `field(default_factory=list)`, never a bare `[]` default --
    a mutable default shared across every `IngestReport` instance is exactly
    the bug dataclasses exist to prevent.
    """
    label: str
    set_uuid: str = ''
    created_runs: List[str] = field(default_factory=list)
    created_results: List[str] = field(default_factory=list)
    skipped_results: List[str] = field(default_factory=list)
    superseded_results: List[str] = field(default_factory=list)
    updated_provenance: List[str] = field(default_factory=list)
    incomplete_results: List[str] = field(default_factory=list)
    empty_runs: List[str] = field(default_factory=list)
    unparseable: List[str] = field(default_factory=list)
    failures: List[Tuple[str, str]] = field(default_factory=list)
    dry_run: bool = False

    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        head = f"{'DRY RUN: ' if self.dry_run else ''}set {self.label!r}"
        if self.set_uuid:
            head += f" ({self.set_uuid})"
        lines = [
            head,
            f"  runs created:        {len(self.created_runs)}",
            f"  results created:     {len(self.created_results)}",
            f"  results skipped:     {len(self.skipped_results)}",
            f"  results superseded:  {len(self.superseded_results)}",
            f"  provenance updated:  {len(self.updated_provenance)}",
            f"  results incomplete:  {len(self.incomplete_results)}",
            f"  empty runs:          {len(self.empty_runs)}",
            f"  unparseable runs:    {len(self.unparseable)}",
        ]
        for rel, message in self.failures:
            lines.append(f"  FAILED {rel}: {message}")
        return '\n'.join(lines)


def _point_key(rel_path: str, point: LoadPoint) -> str:
    return (f"{rel_path}#{point.load_kind}{point.load:g}"
            f"_a{point.angle:g}_s{point.speed:g}")


def _frames_for(point: LoadPoint) -> Dict[str, Any]:
    """The frames `summarise_run` consumes for one load point.

    A `.csv` member (a PES grid, the indent retract curve) is a plain comma
    table with a `# family=` first line, not a LAMMPS `ave/time` file, so it
    is read with `pd.read_csv(comment='#')`; everything else goes through
    `read_data.read_results_file`, the same parser the DataReader uses.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    from ..postprocessing.read_data import read_results_file  # pylint: disable=import-outside-toplevel

    frames: Dict[str, Any] = {}
    for tier in TIER_KEYS:
        path = point.tiers.get(tier)
        if path is None:
            continue
        frames[tier] = (pd.read_csv(path, comment='#') if path.suffix == '.csv'
                        else read_results_file(path, point.family,
                                               n_layers=point.layers or 2))
    retract = point.tiers.get('retract')
    if retract is not None:
        frames['retract'] = pd.read_csv(retract, comment='#')
    return frames


def create_results(run_node, point: LoadPoint, run_dir: Path, *,
                   skip_fraction: float = 0.2, store: bool = True):
    """Build one load point as a `FrictionResultsData` and return it.

    `run_dir` is the run's own root directory (`locator.run_dir`): every path
    in `point.tiers`/`point.aux` lives under it, and `add_result_files` needs
    it to compute each stored member's run-relative path (R18).

    `store=False` (the CalcJob parser's path, via `ingest_retrieved`) returns
    the node UNSTORED and skips the group: `Parser.out()` requires unstored
    nodes so the engine can attach them to the calculation with real
    provenance links. The engine does NOT add anything to a `Group` on its
    own once it stores those outputs -- `LammpsFrictionCalcJob.on_terminated`
    files them into the set's group afterwards, via `add_results_to_group`.
    """
    from ..data.models import summarise_run  # pylint: disable=import-outside-toplevel

    from .data import FrictionResultsData  # pylint: disable=import-outside-toplevel

    node = FrictionResultsData()
    node.family = point.family
    node.load_kind = point.load_kind
    node.load = float(point.load)
    node.angle = float(point.angle)
    node.speed = float(point.speed)
    node.layers = point.layers
    node.run_uuid = str(run_node.uuid)
    node.set_uuid = str(run_node.set_uuid)

    # The 'retract' entry of `point.tiers` is not a sampling tier -- it rides
    # to `add_result_files` as an aux file (R11c), alongside every other aux
    # file the point carries (logs, hold/final artefacts, pes_scan_refine).
    tiers = {t: p for t, p in point.tiers.items() if t in TIER_KEYS}
    retract = point.tiers.get('retract')
    aux = point.aux + ((retract,) if retract is not None else ())
    node.add_result_files(tiers, run_dir=run_dir, aux=aux,
                          max_mb=_max_stored_file_mb(),
                          content_sha256=point.content_sha256)
    node.set_summary(summarise_run(_frames_for(point), point.family,
                                   skip_fraction=skip_fraction))
    if not store:
        return node
    node.store()
    _group_for(run_node.set_label or run_node.set_uuid).add_nodes([node])
    return node


def locator_for(run_dir: Path, rel_path: str):
    """A `RunLocator` for a bare run directory whose `rel_path` is known.

    `discover_runs` needs a whole simulation root; a retrieved CalcJob folder
    is one run with no `<model>/` above it, so the CalcJob parser builds its
    locator from the `rel_path` recorded on the run node instead.
    """
    from dataclasses import replace  # pylint: disable=import-outside-toplevel
    import json as _json  # pylint: disable=import-outside-toplevel

    from ..run.naming import parse_run_path  # pylint: disable=import-outside-toplevel
    from ..run.tree import RunLocator  # pylint: disable=import-outside-toplevel

    run_dir = Path(run_dir)
    tokens = parse_run_path(rel_path)
    prov_file = run_dir / 'provenance.json'
    family = tokens.model
    if prov_file.is_file():
        family = _json.loads(prov_file.read_text(encoding='utf-8')).get(
            'family') or tokens.model
    # NOT `provenance_dir_for(run_dir)`: for an AFM-family run that helper
    # falls back to `run_dir.parent / 'provenance'` when `run_dir/provenance`
    # does not exist -- correct for a real tree, wrong for a scratch directory
    # that IS the whole retrieved tree. `run_dir.parent` there is arbitrary
    # (whatever the CalcJob's `TemporaryDirectory` happens to sit under, or
    # nothing at all if `run_dir` is the scratch root itself), never a real
    # `provenance/` sibling, so this pins it to the one place that could
    # possibly exist inside the retrieval.
    return RunLocator(
        tokens=replace(tokens, family=family), run_dir=run_dir,
        provenance_file=prov_file, provenance_dir=run_dir / 'provenance',
        results_dir=run_dir / 'results', data_dir=run_dir / 'data',
        lammps_dir=run_dir / 'lammps')


def ingest_retrieved(run_dir: Path, rel_path: str, run_node, *,
                     skip_fraction: float = 0.2) -> List:
    """UNSTORED `FrictionResultsData` for a retrieved CalcJob folder.

    A `Parser` must hand `self.out()` unstored nodes so the engine can attach
    them to the calculation with real provenance links. Same `load_points` and
    same `summarise_run` as the import path -- the nodes are indistinguishable
    from ones `ingest_run` would have created.

    Degrades per point, exactly like `ingest_run`'s main loop: a truncated or
    unparseable results file for ONE load point is logged and skipped, not
    allowed to abort every other point of the same run (a PBS array where one
    load's write got cut off must not also lose the 19 that finished cleanly).
    The caller (the parser) treats an all-empty return the same as "no results
    at all" -- `ERROR_NO_LOAD_POINTS` -- since there is no `IngestReport` here
    to record a partial failure against.
    """
    locator = locator_for(run_dir, rel_path)
    nodes = []
    for point in load_points(locator):
        try:
            nodes.append(create_results(run_node, point, locator.run_dir,
                                        skip_fraction=skip_fraction,
                                        store=False))
        except Exception:  # pylint: disable=broad-except
            logger.exception("Ingest failed for a load point of %s", rel_path)
    return nodes


def add_results_to_group(label: str, nodes) -> None:
    """File already-stored results nodes into `label`'s AiiDA group.

    The counterpart to `create_results`'s own `_group_for(...).add_nodes(...)`
    call for the CalcJob-native path: `ingest_retrieved` hands its nodes to
    the parser UNSTORED (`store=False`), and the AiiDA engine stores and links
    them for real once the process finishes -- but the engine never adds
    anything to a `Group` on its own. `LammpsFrictionCalcJob.on_terminated`
    calls this once those nodes are stored, so `export_archive`'s group-based
    "everything this set produced" query includes CalcJob-native results too.
    """
    nodes = list(nodes)
    if not nodes:
        return
    _group_for(label).add_nodes(nodes)


def _plan_run(rel: str, set_node, locator, points: List[LoadPoint],
             report: IngestReport) -> None:
    """Classify one run's points for a dry run, using READ-ONLY queries only.

    No node and no Group is ever created here. `set_node` is `None` when no
    set named `label` exists yet in the store (a brand-new import): every run
    and point under it is then "would create", because there is nothing to
    compare against. When a set DOES exist, a run/point that is not found in
    it is likewise "would create" -- but one that IS found is classified
    exactly as a real ingest would (`results_exist` / `find_results`), never
    unconditionally reported as "created". Getting this wrong is what made a
    dry run after a real ingest still claim N new results instead of N
    skipped.
    """
    run_node = _find_run(set_node, locator) if set_node is not None else None
    if run_node is None:
        report.created_runs.append(rel)
        report.created_results.extend(_point_key(rel, p) for p in points)
        return
    if run_node.provenance_changed(locator):
        report.updated_provenance.append(rel)

    for point in points:
        key = _point_key(rel, point)
        try:
            if results_exist(run_node, point):
                report.skipped_results.append(key)
            elif find_results(run_node, point) is not None:
                report.superseded_results.append(key)
            else:
                report.created_results.append(key)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Dry-run classification failed for %s", key)
            report.failures.append((key, f"{type(exc).__name__}: {exc}"))


def ingest_run(locator, set_node, report: IngestReport, *,
               skip_fraction: float = 0.2) -> None:
    """Ingest one run into `set_node`, recording the outcome in `report`.

    One `try` per LOAD POINT, not one per run. With the try around the whole
    loop, a single truncated file at load 3 of 20 aborted loads 4-20, named
    none of them in the report, and on re-ingest hash-skipped 1-2 and raised
    again on 3 -- 4-20 were unreachable until the user found the bad file by
    hand. That is the remote-manual mode's normal failure (a PBS array where
    some jobs hit walltime), so it must degrade per point.
    """
    rel = locator.tokens.rel_path
    try:
        points = load_points(locator)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Could not enumerate load points for %s", rel)
        report.failures.append((rel, f"{type(exc).__name__}: {exc}"))
        return

    if not points:
        # A run directory with no results at all -- e.g. `run --aiida` has
        # just created it and the job has not finished yet. Not a failure
        # (R8): the caller decides whether an empty tree is acceptable.
        report.empty_runs.append(rel)

    if report.dry_run:
        try:
            _plan_run(rel, set_node, locator, points, report)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Could not look up the run node for %s", rel)
            report.failures.append((rel, f"{type(exc).__name__}: {exc}"))
        return

    try:
        run_node, created = find_or_create_run(set_node, locator)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Could not create the run node for %s", rel)
        report.failures.append((rel, f"{type(exc).__name__}: {exc}"))
        return
    if created:
        report.created_runs.append(rel)
    else:
        try:
            if run_node.refresh_provenance(locator):
                report.updated_provenance.append(rel)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Could not record the updated provenance of %s", rel)
            report.failures.append((rel, f"{type(exc).__name__}: {exc}"))

    for point in points:
        key = _point_key(rel, point)
        try:
            if results_exist(run_node, point):
                report.skipped_results.append(key)
                continue
            stale = find_results(run_node, point)
            node = create_results(run_node, point, locator.run_dir,
                                  skip_fraction=skip_fraction)
            if stale is not None:
                supersede(stale, run_node)
                report.superseded_results.append(key)
            else:
                report.created_results.append(key)
            if not node.is_complete:
                report.incomplete_results.append(key)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Ingest failed for %s", key)
            report.failures.append((key, f"{type(exc).__name__}: {exc}"))


def ingest_root(root: Path, label: Optional[str] = None,
                profile: Optional[str] = None, *,
                dry_run: bool = False) -> IngestReport:
    """Ingest every run under `root`. Idempotent: re-running adds only new or
    changed load points, and creates nothing at all when nothing changed.

    `profile` is loaded via `load_aiida_profile` before anything touches the
    store. `dry_run` still loads it: a plan must classify each run/point
    against what is ALREADY in the store (existing set found by label, run
    found by `(set_uuid, rel_path)`, `results_exist`/`find_results` per
    point), read-only and via `_find_set`/`_find_run` rather than
    `find_or_create_*` -- nothing is stored and no Group is created, but a
    dry run after a real ingest must report "0 created, N skipped", not
    silently re-claim everything as new.
    """
    from ..core.config import load_settings  # pylint: disable=import-outside-toplevel
    from ..run.tree import discover_runs, unparseable_runs  # pylint: disable=import-outside-toplevel

    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel

    root = Path(root)
    label = label or root.name
    report = IngestReport(label=label, dry_run=dry_run)
    report.unparseable = unparseable_runs(root)

    locators = discover_runs(root)
    if not locators:
        report.failures.append((str(root), 'no provenance.json found'))
        return report

    load_aiida_profile(profile)

    settings = load_settings()
    # `DatabaseSettings.skip_fraction` is a plain, always-present field
    # (Task 31 made `DatabaseSettings` central-only but kept this field).
    skip_fraction = settings.database.skip_fraction

    if dry_run:
        set_node = _find_set(root, label)
    else:
        set_node = find_or_create_set(root, label)
    if set_node is not None:
        report.set_uuid = str(set_node.uuid)

    for locator in locators:
        ingest_run(locator, set_node, report, skip_fraction=skip_fraction)
    return report
