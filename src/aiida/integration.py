"""Set-level bookkeeping over the AiiDA store: list, delete, clear.

A set is the Group ``friction2d/<label>`` that :func:`src.aiida.ingest.import_root`
fills with deck CalcJobs. One CalcJob can sit in several sets (a tree imported
under two labels), and every content-addressed input -- blocks, files,
provenance, schema, Code -- is shared across runs, so deleting a set removes
only what no other set still holds.
"""
import logging
from typing import Any, Dict, Iterable, List, Optional, Set

from . import AIIDA_AVAILABLE

logger = logging.getLogger(__name__)

#: The extras that mark a content-addressed friction2d node (``store``).
_CONTENT_EXTRAS = ('friction2d_block_hash', 'friction2d_sha256',
                   'friction2d_folder_sha256', 'friction2d_schema_hash')


def _require_aiida():
    if not AIIDA_AVAILABLE:
        raise RuntimeError(
            "AiiDA is not installed. Install with: pip install 'FrictionSim2D[aiida]'")


def _ensure_aiida_profile(profile: Optional[str] = None) -> None:
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel
    load_aiida_profile(profile)


def _set_groups() -> List[Any]:
    """Every ``friction2d/<label>`` set Group (datasets excluded)."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .ingest import GROUP_PREFIX  # pylint: disable=import-outside-toplevel

    qb = orm.QueryBuilder().append(
        orm.Group, filters={'label': {'like': f'{GROUP_PREFIX}/%'}})
    return [g for g in qb.all(flat=True)
            if not g.label.startswith(f'{GROUP_PREFIX}/datasets/')]


def _calcjobs_of(group) -> List[Any]:
    from aiida import orm  # pylint: disable=import-outside-toplevel

    qb = orm.QueryBuilder().append(orm.Group, filters={'id': group.pk}, tag='g')
    qb.append(orm.CalcJobNode, with_group='g')
    return qb.all(flat=True)


def _live(nodes: Iterable[Any]) -> List[Any]:
    return [n for n in nodes if not n.base.extras.get('superseded', False)]


def list_simulation_sets(profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """One row per set: label, live decks, live points, source root, and the
    Group extras (model, materials, created, git_commit, source_host).

    ``n_runs`` (distinct ``run_uuid`` of live decks) and ``n_results`` (live
    points) keep the old column names the CLI prints.
    """
    _require_aiida()
    _ensure_aiida_profile(profile)
    from .ingest import GROUP_PREFIX  # pylint: disable=import-outside-toplevel
    from .store import RUN_UUID_EXTRA  # pylint: disable=import-outside-toplevel

    rows = []
    for group in _set_groups():
        decks = _live(_calcjobs_of(group))
        points = [p for calc in decks if 'points' in calc.outputs
                  for p in calc.outputs.points.values()]
        extras = group.base.extras.all
        n_points = len(_live(points))
        rows.append({
            'label': group.label[len(GROUP_PREFIX) + 1:],
            'uuid': group.uuid, 'pk': group.pk,
            'model': extras.get('model'), 'materials': extras.get('materials'),
            'created': extras.get('created'), 'git_commit': extras.get('git_commit'),
            'source_host': extras.get('source_host'),
            'source_root': extras.get('source_root'),
            'n_decks': len(decks), 'n_points': n_points,
            'n_runs': len({c.base.extras.get(RUN_UUID_EXTRA) for c in decks}),
            'n_results': n_points,
        })
    return sorted(rows, key=lambda r: r['label'])


def _held_elsewhere(group) -> Set[int]:
    """pks of CalcJobs some OTHER set Group holds."""
    return {calc.pk for other in _set_groups() if other.pk != group.pk
            for calc in _calcjobs_of(other)}


def _dataset_held() -> Dict[int, Set[str]]:
    """node pk -> names of the datasets (``friction2d/datasets/<name>``)
    whose Group holds it."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .steps import DATASET_PREFIX  # pylint: disable=import-outside-toplevel

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
    dataset's rows pointing at nothing. ``ValueError`` names the datasets;
    delete those Groups first if the set must go.
    """
    _require_aiida()
    _ensure_aiida_profile(profile)
    from aiida import orm  # pylint: disable=import-outside-toplevel
    from aiida.tools import delete_nodes  # pylint: disable=import-outside-toplevel

    from .ingest import group_label  # pylint: disable=import-outside-toplevel

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
            f'Delete those dataset Groups first.')
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

    removed = {c.base.extras.get(DECK_KEY_EXTRA) for c in _live(_calcjobs_of(group))
               if c.pk in doomed}
    removed.discard(None)
    stranded: Dict[str, Set[str]] = {}
    for other in _set_groups():
        if other.pk == group.pk:
            continue
        calcs = _calcjobs_of(other)
        live = {c.base.extras.get(DECK_KEY_EXTRA) for c in _live(calcs)}
        for calc in calcs:
            key = calc.base.extras.get(DECK_KEY_EXTRA)
            if key in removed and key not in live:
                stranded.setdefault(key, set()).add(other.label[len(GROUP_PREFIX) + 1:])
    return stranded


def _friction2d_calcjobs() -> List[int]:
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .store import CALCJOB_PROCESS_TYPE  # pylint: disable=import-outside-toplevel

    return orm.QueryBuilder().append(
        orm.CalcJobNode, filters={'process_type': CALCJOB_PROCESS_TYPE},
        project='id').all(flat=True)


def _inputs_of(calc_pks: List[int]) -> Set[int]:
    from aiida import orm  # pylint: disable=import-outside-toplevel

    if not calc_pks:
        return set()
    qb = orm.QueryBuilder().append(orm.CalcJobNode, filters={'id': {'in': calc_pks}},
                                   tag='c')
    qb.append(orm.Node, with_outgoing='c', project='id')
    return set(qb.all(flat=True))


def _orphans(candidates: Set[int]) -> List[int]:
    """The ``candidates`` that still exist and no process uses as an input."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    if not candidates:
        return []
    ids = {'id': {'in': sorted(candidates)}}
    existing = set(orm.QueryBuilder().append(orm.Node, filters=ids,
                                             project='id').all(flat=True))
    qb = orm.QueryBuilder().append(orm.Node, filters=ids, tag='n', project='id')
    qb.append(orm.ProcessNode, with_incoming='n')
    return sorted(existing - set(qb.all(flat=True)))


def _step_inputs(within: Optional[Set[int]] = None) -> Set[int]:
    """pks of the Dicts made only to feed a step (``steps.STEP_INPUT_EXTRA``:
    integrity context and settings, user rulings); with ``within``, only
    those some process in ``within`` takes as an input."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .steps import STEP_INPUT_EXTRA  # pylint: disable=import-outside-toplevel

    qb = orm.QueryBuilder().append(
        orm.Dict, filters={'extras': {'has_key': STEP_INPUT_EXTRA}}, tag='d', project='id')
    if within is not None:
        if not within:
            return set()
        qb.append(orm.ProcessNode, with_incoming='d',
                  filters={'id': {'in': sorted(within)}})
    return set(qb.all(flat=True))


def _dataset_tables() -> Set[int]:
    """pks of the datasets' table Dicts (``steps.DATASET_EXTRA``)."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .steps import DATASET_EXTRA  # pylint: disable=import-outside-toplevel

    return set(orm.QueryBuilder().append(
        orm.Dict, filters={'extras': {'has_key': DATASET_EXTRA}},
        project='id').all(flat=True))


def _content_addressed() -> Set[int]:
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .store import VERSION_EXTRA  # pylint: disable=import-outside-toplevel

    pks: Set[int] = set()
    for extra in (*_CONTENT_EXTRAS, VERSION_EXTRA):
        pks |= set(orm.QueryBuilder().append(
            orm.Node, filters={'extras': {'has_key': extra}}, project='id').all(flat=True))
    return pks


def _legacy_nodes() -> List[int]:
    """The pre-Task-6 FrictionSimulation*/FrictionResults data nodes."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .data import (FrictionResultsData, FrictionSimulationData,  # pylint: disable=import-outside-toplevel
                       FrictionSimulationSetData)

    pks: List[int] = []
    for cls in (FrictionResultsData, FrictionSimulationData, FrictionSimulationSetData):
        pks += orm.QueryBuilder().append(cls, project='id').all(flat=True)
    return pks


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
    _require_aiida()
    _ensure_aiida_profile(profile)
    from aiida import orm  # pylint: disable=import-outside-toplevel
    from aiida.tools import delete_nodes  # pylint: disable=import-outside-toplevel

    from .ingest import GROUP_PREFIX  # pylint: disable=import-outside-toplevel

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
