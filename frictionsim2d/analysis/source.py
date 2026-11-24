"""The points an analysis reads: one dataset's integrity-pass rows, and the
stored files behind each of them.

:class:`RunSource` holds the rows; what it reads per row goes through a
fetcher -- :class:`StoreFetcher` against the loaded AiiDA profile, or any
object with the same methods in a test.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ['NO_WINDOW', 'PASS_STATUS', 'RunSource', 'StoreFetcher', 'dataset_table',
           'material_names']

#: The verdict status of an integrity-pass point (`steps._dataset_rows` keeps
#: only these; `steps.postprocess_group` summarises only these).
PASS_STATUS = 'complete'
#: The drop reason of a point with no window tier to read (halted, not reached, ...).
NO_WINDOW = 'no_window_tier'

SnapshotKind = Literal['load', 'final']

#: `calcjob._LOAD_FILE_RE`: a system deck's loaded structure.
_LOAD_FILE_RE = re.compile(r'^load_(?P<load>[-+0-9.eE]+)N\.data$')
#: `calcjob._TOL`.
_TOL = 1e-9
#: The settings file a deck includes (`calcjob.included_files`).
_SETTINGS_NAME = 'system.in.settings'
#: The link-label prefix of a CalcJob's ``points`` namespace.
_POINTS_PREFIX = 'points__'


class RunSource:
    """The integrity-pass rows of one dataset, for one family.

    ``rows`` (any rows; non-pass ones are dropped here) is a DataFrame of
    :func:`frictionsim2d.aiida.query.dataset_rows` rows. ``family`` is checked against
    the rows' ``family`` column, or taken from it when None. ``fetch`` reads
    a row's stored files (:class:`StoreFetcher` by default).

    Raises:
        ValueError: the rows hold more than one family, or not ``family``.
    """

    def __init__(self, rows: pd.DataFrame, family: Optional[str] = None,
                 fetch: Any = None) -> None:
        found = sorted(set(rows['family'].dropna())) if 'family' in rows else []
        if len(found) > 1:
            raise ValueError(f'a dataset mixes families {found}; an analysis reads '
                             f'one family -- build one dataset per family')
        if family is None:
            if not found:
                raise ValueError('no family given and the rows carry none')
            family = found[0]
        elif found and found != [family]:
            raise ValueError(f'rows are family {found[0]!r}, not {family!r}')
        self.family: str = family
        self.all_rows = rows.reset_index(drop=True)
        passed = self.all_rows['verdict_status'] == PASS_STATUS
        self.rows = self.all_rows[passed].reset_index(drop=True)
        self.dropped: List[Tuple[str, str]] = [
            (str(r.point_uuid), str(r.verdict_status))
            for r in self.all_rows[~passed].itertuples()]
        self._fetch = fetch if fetch is not None else StoreFetcher()
        #: The dataset the rows are (set by :meth:`from_dataset`; None otherwise),
        #: for an analysis that reads another stored result of it.
        self.dataset: Optional[str] = None
        #: What an analysis records about its table beside the drops (a
        #: column it left out and why); stored with the result.
        self.notes: List[str] = []
        #: The stored results the analysis reads (``Analysis.depends_on``), set
        #: by :func:`~frictionsim2d.analysis.store.result_node`; None: look them up.
        self.depends: Optional[Dict[str, Any]] = None

    def __enter__(self) -> 'RunSource':
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release what the fetcher holds (its snapshot copies)."""
        close = getattr(self._fetch, 'close', None)
        if close is not None:
            close()

    @classmethod
    def from_dataset(cls, name: str) -> 'RunSource':
        """The source of the dataset ``friction2d/datasets/<name>`` (the
        profile must be loaded).

        A dataset holds only ``complete`` points (`steps._dataset_rows`), so
        the rest of its sets' live points are added here, with the verdict
        the dataset's own policy selects: they land in ``all_rows`` and
        ``dropped``, never in ``rows``.
        """
        from ..aiida.query import dataset_rows  # pylint: disable=import-outside-toplevel
        rows, _units = dataset_rows(name)
        excluded = _excluded_rows(dataset_table(name),
                                  {(r['calc_uuid'], r['point_uuid']) for r in rows})
        source = cls(pd.DataFrame(rows + excluded), None, StoreFetcher())
        source.dataset = name
        return source

    def frames(self, row: Any) -> Dict[str, pd.DataFrame]:
        """``tier -> DataFrame`` of one row's point."""
        return self._fetch.frames(row)

    def verdict(self, row: Any) -> Dict[str, Any]:
        """The stored integrity verdict of the row's point (``status``,
        ``reasons``, ``measures``, ...), or ``{}`` for a point without one."""
        return self._fetch.verdict(row)

    def snapshot(self, row: Any, kind: SnapshotKind) -> Optional[Path]:
        """A temporary copy of the row's ``load`` or ``final`` snapshot, or
        None when the store has none."""
        if kind not in ('load', 'final'):
            raise ValueError(f'unknown snapshot kind {kind!r} (load or final)')
        return self._fetch.snapshot(row, kind)

    def settings_text(self, row: Any) -> str:
        """The run's ``system.in.settings``."""
        return self._fetch.settings_text(row)

    def structure_text(self, row: Any) -> Optional[str]:
        """The run's built ``<mat>_<n>.lmp``, or None when not stored."""
        return self._fetch.structure_text(row)

    def run_fields(self, row: Any) -> Dict[str, Any]:
        """The run's parameter fields: every block field (``settings.simulation.
        timestep``, ``general.driving_spring``, ...), an elided one refilled
        from the run's stored schema default, plus ``derived.<name>`` for the
        derived values (``derived.drive_k``)."""
        return self._fetch.run_fields(row)


class StoreFetcher:
    """Reads a row's files from the loaded AiiDA profile.

    Snapshots are copied into one temporary directory that lives as long as
    this fetcher.
    """

    def __init__(self) -> None:
        self._tmp: Optional[tempfile.TemporaryDirectory] = None
        self._fields: Dict[str, Dict[str, Any]] = {}

    def close(self) -> None:
        """Remove the snapshot copies handed out so far."""
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def frames(self, row: Any) -> Dict[str, pd.DataFrame]:
        """`frictionsim2d.aiida.query.point_frames` of the row, plus its ``peratom``
        tier: the CalcJob outputs that as its own ``peratom.<point label>``
        node beside the point, read the same way."""
        # pylint: disable=import-outside-toplevel
        from aiida import orm

        from ..aiida.query import point_frames
        from ..aiida.steps import point_frames as stored_frames
        frames = point_frames(row.calc_uuid, row.node_uuid)
        if 'peratom' in frames:
            return frames
        calc = orm.load_node(row.calc_uuid)
        if 'peratom' not in calc.outputs:
            return frames
        point = orm.load_node(row.node_uuid)
        label = _point_label(point)
        if label not in calc.outputs.peratom:
            return frames
        n_layers = int(calc.inputs.parameters.get_dict()['n_layers'])
        with tempfile.TemporaryDirectory() as tmp:
            frames.update(stored_frames(calc.outputs.peratom[label], Path(tmp) / 'results',
                                        point.base.attributes.get('family'), n_layers))
        return frames

    def verdict(self, row: Any) -> Dict[str, Any]:
        """The verdict Dict the row's ``verdict_uuid`` names (the one the
        dataset's policy selected; `steps.check_integrity` stores its
        ``measures``), or ``{}``."""
        from aiida import orm  # pylint: disable=import-outside-toplevel
        uuid = getattr(row, 'verdict_uuid', None)
        if uuid is None or pd.isna(uuid):
            return {}
        return orm.load_node(uuid).get_dict()

    def snapshot(self, row: Any, kind: SnapshotKind) -> Optional[Path]:
        """``final``: the CalcJob's ``snapshots.<point label>`` output.
        ``load``: the ``load_<F>N.data`` structure of
        the row's load, an input or output ``structures`` node of the run's
        CalcJobs."""
        from aiida import orm  # pylint: disable=import-outside-toplevel
        calc = orm.load_node(row.calc_uuid)
        node = None
        if kind == 'final':
            label = _point_label(orm.load_node(row.node_uuid))
            if 'snapshots' in calc.outputs and label in calc.outputs.snapshots:
                node = calc.outputs.snapshots[label]
        else:
            node = next((n for n in _structure_nodes(calc)
                         if _is_load_file(n.filename, float(row.load))), None)
        if node is None:
            return None
        if self._tmp is None:
            # Lives as long as the fetcher (`close`): the caller reads the
            # returned path after this returns, so no `with`.
            self._tmp = tempfile.TemporaryDirectory(  # pylint: disable=consider-using-with
                prefix='friction2d-analysis-')
        path = Path(self._tmp.name) / node.uuid / node.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(node.get_content(mode='rb'))
        return path

    def settings_text(self, row: Any) -> str:
        """``system.in.settings`` from the ``includes`` of the run's CalcJobs.

        Raises:
            LookupError: no CalcJob of the run includes it.
        """
        from aiida import orm  # pylint: disable=import-outside-toplevel
        for calc in _run_calcs(orm.load_node(row.calc_uuid)):
            if 'includes' not in calc.inputs:
                continue
            for node in calc.inputs.includes.values():
                if node.filename == _SETTINGS_NAME:
                    return node.get_content()
        raise LookupError(f'run {row.run_uuid}: no CalcJob includes {_SETTINGS_NAME}')

    def structure_text(self, row: Any) -> Optional[str]:
        """The built ``<mat>_<n_layers>.lmp``
        among the ``structures`` inputs of the run's CalcJobs, or None."""
        from aiida import orm  # pylint: disable=import-outside-toplevel
        calc = orm.load_node(row.calc_uuid)
        n_layers = int(calc.inputs.parameters.get_dict()['n_layers'])
        pattern = re.compile(rf'^.+_{n_layers}\.lmp$')
        hits = {n.filename: n for c in _run_calcs(calc) if 'structures' in c.inputs
                for n in c.inputs.structures.values() if pattern.match(n.filename)}
        if not hits:
            return None
        if len(hits) > 1:
            named = [h for m in _materials(row) for h in hits if h == f'{m}_{n_layers}.lmp']
            if len(named) != 1:
                raise LookupError(f'run {row.run_uuid}: {sorted(hits)} all match '
                                  f'<mat>_{n_layers}.lmp')
            return hits[named[0]].get_content()
        return next(iter(hits.values())).get_content()

    def run_fields(self, row: Any) -> Dict[str, Any]:
        """The ``schema`` defaults of the row's CalcJob, overlaid with its
        ``blocks`` fields, plus its ``derived`` fields as ``derived.<name>``.
        Read once per CalcJob."""
        if row.calc_uuid not in self._fields:
            from aiida import orm  # pylint: disable=import-outside-toplevel
            calc = orm.load_node(row.calc_uuid)
            out: Dict[str, Any] = {}
            if 'schema' in calc.inputs:
                out.update({path: spec.get('default') for path, spec in
                            (calc.inputs.schema.get_dict().get('fields') or {}).items()})
            if 'blocks' in calc.inputs:
                for block in calc.inputs.blocks.values():
                    out.update(block.get_dict().get('fields') or {})
            if 'derived' in calc.inputs:
                out.update({f'derived.{k}': v for k, v in
                            (calc.inputs.derived.get_dict().get('fields') or {}).items()})
            self._fields[row.calc_uuid] = out
        return dict(self._fields[row.calc_uuid])


def dataset_table(name: str) -> Any:
    """The table ``Dict`` of the dataset ``friction2d/datasets/<name>`` (the
    query of :func:`frictionsim2d.aiida.query.dataset_rows`).

    Raises:
        LookupError: no dataset of that name.
    """
    # pylint: disable=import-outside-toplevel
    from aiida import orm

    from ..aiida.layout import DATASET_PREFIX
    from ..aiida.steps import DATASET_EXTRA
    qb = orm.QueryBuilder().append(
        orm.Group, tag='g', filters={'label': f'{DATASET_PREFIX}{name}'})
    qb.append(orm.Dict, with_group='g', filters={f'extras.{DATASET_EXTRA}': name},
              project=['*'])
    tables = qb.all(flat=True)
    if not tables:
        raise LookupError(f'no dataset {name!r} ({DATASET_PREFIX}{name})')
    return tables[0]


def _excluded_rows(table: Any, kept: set) -> List[Dict[str, Any]]:
    """Rows for the live points of the dataset's sets that it left out for
    their verdict: the identity columns `query._points` gives every point,
    plus ``set``, ``verdict_uuid`` and ``verdict_status`` (``None`` for a
    point with no verdict). The selection is `steps._dataset_rows`' own
    under the dataset's stored policy.

    A ``complete`` point the dataset left out (it had no summary under the
    dataset's options) is not a drop for its verdict; it is logged and left out.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from ..aiida import query, steps
    from ..aiida.rulings import Policy
    from ..aiida.store import resolve_set
    policy = Policy.from_dict(table.get('policy') or {})
    identity = {(r['calc_uuid'], r['point_uuid']): r
                for r in query._points(include_superseded=True).rows}
    out: List[Dict[str, Any]] = []
    seen = set()
    for label in table.get('labels') or []:
        points = [p for p in steps._live_points(resolve_set(label))
                  if p.pk not in seen]
        verdicts = steps.latest_verdicts(points, policy)
        for point in points:
            seen.add(point.pk)
            key = (point.creator.uuid, point.base.attributes.get('point_uuid'))
            if key in kept:
                continue
            verdict = verdicts.get(point.pk)
            status = None if verdict is None else verdict['status']
            if status == PASS_STATUS:
                logger.info('dataset %r: complete point %s is not in it (no summary)',
                            table.get('name'), key[1])
                continue
            row = {k: v for k, v in identity.get(key, {}).items()
                   if k not in ('superseded', 'set_labels')}
            row.update({'set': label, 'calc_uuid': key[0], 'point_uuid': key[1],
                        'node_uuid': point.uuid,
                        'family': point.base.attributes.get('family'),
                        'verdict_uuid': None if verdict is None else verdict.uuid,
                        'verdict_status': status or 'no_verdict'})
            out.append(row)
    return out


def _point_label(point: Any) -> str:
    """The ``points`` namespace key the point was output under."""
    from aiida.common.links import LinkType  # pylint: disable=import-outside-toplevel
    for link in point.base.links.get_incoming(link_type=LinkType.CREATE).all():
        if link.link_label.startswith(_POINTS_PREFIX):
            return link.link_label[len(_POINTS_PREFIX):]
    raise LookupError(f'point node {point.uuid} is no points output')


def _run_calcs(calc: Any) -> List[Any]:
    """``calc`` first, then the run's other live friction2d CalcJobs."""
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from ..aiida.layout import is_superseded  # pylint: disable=import-outside-toplevel
    from ..aiida.store import RUN_UUID_EXTRA  # pylint: disable=import-outside-toplevel
    run_uuid = calc.base.extras.get(RUN_UUID_EXTRA, None)
    others: List[Any] = []
    if run_uuid is not None:
        qb = orm.QueryBuilder().append(
            orm.CalcJobNode, filters={f'extras.{RUN_UUID_EXTRA}': run_uuid,
                                      'uuid': {'!==': calc.uuid}},
            project=['*'])
        others = sorted((n for n in qb.all(flat=True) if not is_superseded(n)),
                        key=lambda n: n.uuid)
    return [calc, *others]


def _structure_nodes(calc: Any) -> List[Any]:
    """The ``structures`` inputs and outputs of the run's CalcJobs."""
    nodes: List[Any] = []
    for each in _run_calcs(calc):
        for links in (each.inputs, each.outputs):
            if 'structures' in links:
                nodes += list(links.structures.values())
    return nodes


def _is_load_file(filename: str, load: float) -> bool:
    match = _LOAD_FILE_RE.match(filename)
    return bool(match) and abs(float(match['load']) - load) <= _TOL


def material_names(value: Any) -> List[str]:
    """A row's ``materials`` (one name or a list of them) as a list."""
    if isinstance(value, str):
        return [value]
    return list(value or [])


def _materials(row: Any) -> List[str]:
    return material_names(getattr(row, 'materials', None))
