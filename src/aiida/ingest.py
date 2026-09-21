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
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

GROUP_PREFIX = 'friction2d'


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
