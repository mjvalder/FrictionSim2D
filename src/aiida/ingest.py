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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..postprocessing.read_data import (
    FILE_PATTERN_SHEET, FILE_PATTERN_SHEET_ALT, FILE_PATTERN_TIP,
    FILE_PATTERN_TIP_ALT, _tier_of,
)
from ..run.naming import AFM_LIKE_MODELS
from .data.results import TIER_KEYS, _content_hash

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
_PES_GLOB = 'pes_scan*.csv'
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


def find_or_create_set(root: Path, label: str):
    """The set node for `label`, created from `root` if it does not exist.

    Raises :class:`SetRootMismatch` when `label` already names a set built from
    a different `source_root`. Re-ingesting the SAME tree under the same label
    is the idempotent path and is allowed.
    """
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    from .data import FrictionSimulationSetData  # pylint: disable=import-outside-toplevel

    root = Path(root).resolve()
    qb = QueryBuilder()
    qb.append(FrictionSimulationSetData, filters={'attributes.label': label})
    existing = qb.first(flat=True)
    if existing is not None:
        if existing.source_root and existing.source_root != str(root):
            raise SetRootMismatch(
                f"set {label!r} was ingested from {existing.source_root!r}; "
                f"{str(root)!r} is a different tree. Pass --label to give this "
                f"tree its own set, or delete the old one with "
                f"`FrictionSim2D aiida delete {label}`.")
        return existing

    node = FrictionSimulationSetData()
    node.set_from_root(root, label)
    node.store()
    _group_for(label).add_nodes([node])
    return node


def find_or_create_run(set_node, locator) -> Tuple[object, bool]:
    """The run node for `(set_uuid, rel_path)`; `(node, created)`."""
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    from .data import FrictionSimulationData  # pylint: disable=import-outside-toplevel

    qb = QueryBuilder()
    qb.append(FrictionSimulationData, filters={
        'attributes.set_uuid': str(set_node.uuid),
        'attributes.rel_path': locator.tokens.rel_path,
    })
    existing = qb.first(flat=True)
    if existing is not None:
        return existing, False

    node = FrictionSimulationData()
    node.set_from_locator(locator, set_uuid=str(set_node.uuid),
                          set_label=set_node.label)
    node.add_run_files(locator, max_mb=_max_stored_file_mb())
    node.store()
    node.status = 'imported'
    _group_for(set_node.label).add_nodes([node])
    return node, True


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
    """
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
    for node in qb.all(flat=True):
        if node.layers != point.layers:
            continue
        if node.base.extras.get('superseded', False):
            continue
        return node
    return None


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
    tier = _tier_of(name)
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
        return [
            LoadPoint(family=family, load_kind='none', load=0.0, angle=0.0,
                      speed=0.0, layers=locator.tokens.layers,
                      tiers={'window': csv}, aux=aux,
                      content_sha256=_content_hash([csv]))
            for csv in sorted(locator.results_dir.glob(_PES_GLOB))
        ]

    model = locator.tokens.model
    grouped: Dict[Tuple[str, float, float, float, Optional[int]], Dict[str, Path]] = {}
    for path in sorted(locator.results_dir.iterdir()):
        matched = _match_point(path.name, model)
        if matched is None:
            continue
        kind, load, angle, speed, layers = matched
        grouped.setdefault((kind, load, angle, speed, layers), {})[
            _tier_of(path.name)] = path

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
            content_sha256=_content_hash(ordered.values())))
    return points
