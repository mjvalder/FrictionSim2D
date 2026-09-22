"""Set-level bookkeeping over the AiiDA store.

Importing, exporting and rebuilding moved to `ingest.py` and `export.py`; what
is left is the three operations that act on a whole set: list, delete, clear.
"""
import logging
from typing import Any, Dict, List, Optional

from . import AIIDA_AVAILABLE

logger = logging.getLogger(__name__)


def _require_aiida():
    if not AIIDA_AVAILABLE:
        raise RuntimeError(
            "AiiDA is not installed. Install with: pip install 'FrictionSim2D[aiida]'")


def _ensure_aiida_profile(profile: Optional[str] = None) -> None:
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel
    load_aiida_profile(profile)


def list_simulation_sets(profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """One row per stored set: label, model, created, materials, counts."""
    _require_aiida()
    _ensure_aiida_profile(profile)
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    from .data import (FrictionResultsData, FrictionSimulationData,  # pylint: disable=import-outside-toplevel
                       FrictionSimulationSetData)

    rows = []
    qb = QueryBuilder()
    qb.append(FrictionSimulationSetData)
    for set_node in qb.all(flat=True):
        runs = QueryBuilder()
        runs.append(FrictionSimulationData,
                    filters={'attributes.set_uuid': str(set_node.uuid)})
        results = QueryBuilder()
        results.append(FrictionResultsData,
                       filters={'attributes.set_uuid': str(set_node.uuid)})
        row = set_node.to_dict()
        row['n_runs'] = runs.count()
        row['n_results'] = results.count()
        rows.append(row)
    return sorted(rows, key=lambda r: r['label'])


def delete_simulation_set(label: str, profile: Optional[str] = None) -> int:
    """Delete every node of `friction2d/<label>` and the group itself."""
    _require_aiida()
    _ensure_aiida_profile(profile)
    from aiida.orm import Group  # pylint: disable=import-outside-toplevel
    from aiida.tools import delete_nodes  # pylint: disable=import-outside-toplevel

    from .ingest import group_label  # pylint: disable=import-outside-toplevel

    group = Group.collection.get(label=group_label(label))
    pks = [node.pk for node in group.nodes]
    _deleted, was_deleted = delete_nodes(pks, dry_run=False)
    Group.collection.delete(group.pk)
    return len(pks) if was_deleted else 0


def clear_all_nodes(profile: Optional[str] = None) -> int:
    """Delete every FrictionSim2D node in the profile (spec U6), and every
    `friction2d/*` group with them -- a group left behind would keep offering
    its set label (`get_available_conditions`, the GUI) for a set that no
    longer exists."""
    _require_aiida()
    _ensure_aiida_profile(profile)
    from aiida.orm import Group, QueryBuilder  # pylint: disable=import-outside-toplevel
    from aiida.tools import delete_nodes  # pylint: disable=import-outside-toplevel

    from .data import (FrictionResultsData, FrictionSimulationData,  # pylint: disable=import-outside-toplevel
                       FrictionSimulationSetData)
    from .ingest import GROUP_PREFIX  # pylint: disable=import-outside-toplevel

    pks = []
    for cls in (FrictionResultsData, FrictionSimulationData,
                FrictionSimulationSetData):
        qb = QueryBuilder()
        qb.append(cls, project=['id'])
        pks.extend(pk for (pk,) in qb.all())
    if pks:
        delete_nodes(pks, dry_run=False)
    groups = QueryBuilder()
    groups.append(Group, filters={'label': {'like': f'{GROUP_PREFIX}/%'}},
                  project=['id'])
    for (group_pk,) in groups.all():
        Group.collection.delete(group_pk)
    return len(pks)
