"""Database query interface for FrictionSim2D (spec §4.4).

`Friction2DDB.query` filters on `RunTokens` (Tasks 3-4) plus `family`,
`model`, `set_label` and `potential`, and returns a `QueryResult` of matching
run nodes and their (non-superseded) result nodes -- the summary scalars a
`FrictionResultsData` carries are `RunSummary.as_dict()`'s fields (Task 8),
written as attributes by `FrictionResultsData.set_summary` (Task 11).

Runs and results are linked by a `run_uuid` ATTRIBUTE, not a provenance edge
(spec §4.1 -- import does not fake process nodes), so this is two
`QueryBuilder` passes, not one chained `append(..., with_X=...)`: the runs are
found first, then the results are found among `attributes.run_uuid in
<those runs>`. A single `append` of both classes with no link would be a
cross product -- every run paired with every result -- and the run-level
filters would not scope the returned rows at all.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .data_keys import is_superseded

try:
    from aiida.orm import QueryBuilder as _QueryBuilder
    from aiida.orm import load_node as _load_node
    AIIDA_AVAILABLE = True
except ImportError:
    AIIDA_AVAILABLE = False
    _QueryBuilder = None  # type: ignore[assignment]
    _load_node = None  # type: ignore[assignment]


@dataclass
class QueryResult:
    """Matching runs and their results. One row per RESULT, not per run.

    `runs` and `results` are aligned by `run_uuid` -- not index-parallel, and
    not necessarily the same length: a run with no matching result (every
    load point excluded by a range filter) is dropped from `runs` too, so
    every run in this list has at least one result in it.
    """

    runs: List[Any]
    results: List[Any]
    total_count: int
    query_params: Dict[str, Any]

    def to_dataframe(self):
        """One row per result: the run's `to_dict()` plus the result's own
        attributes (the `RunSummary` scalars among them), never a
        hand-written subset -- a column this misses is a column the GUI
        table and every CSV/JSON export miss too.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        by_uuid = {str(run.uuid): run for run in self.runs}
        records = []
        for result in self.results:
            run = by_uuid.get(result.run_uuid)
            record = run.to_dict() if run is not None else {}
            # The raw stored attributes go in FIRST: they include the result's
            # own `tiers` (a plain list, e.g. `['window', 'raw']`) and `layers`
            # (may differ from the run's if that ever became possible), which
            # the explicit identity/derived fields below must be allowed to
            # override -- most importantly `tiers`, joined into a single
            # ``'window;raw'`` string. Applying `attributes.all` LAST would
            # silently put the raw list back and undo that join.
            record.update(result.base.attributes.all)
            record.update({
                'run_uuid': result.run_uuid,
                'result_uuid': str(result.uuid),
                'set_label': run.set_label if run is not None else '',
                'load_kind': result.load_kind, 'load': result.load,
                'angle': result.angle, 'speed': result.speed,
                'tiers': ';'.join(result.tiers),
            })
            records.append(record)
        return pd.DataFrame(records)

    def export_csv(self, filepath: Path) -> Path:
        """Export the one-row-per-result table to a CSV file."""
        filepath = Path(filepath)
        self.to_dataframe().to_csv(filepath, index=False)
        return filepath

    def export_json(self, filepath: Path) -> Path:
        """Export the query params and the matching RUNS (not results) as JSON."""
        filepath = Path(filepath)
        filepath.write_text(json.dumps({
            'query_params': self.query_params,
            'total_count': self.total_count,
            'runs': [run.to_dict() for run in self.runs],
        }, indent=2, default=str), encoding='utf-8')
        return filepath


#: RunTokens field (plus family/model/set_label/potential) -> the run-node
#: attribute it filters on. Equality filters only; the temperature range is
#: handled separately below, and `layers`/`materials` need their own
#: `{'in': ...}`/`{'contains': ...}` shape so they are not in this table.
_RUN_EQ_FILTERS = {
    'family': 'family', 'model': 'model', 'set_label': 'set_label',
    'potential': 'pot_token', 'substrate': 'substrate',
    'substrate_amorphous': 'substrate_amorphous', 'tip_material': 'tip_material',
    'tip_radius': 'tip_radius', 'flake_material': 'flake_material',
    'size_x': 'size_x', 'size_y': 'size_y', 'rel_path': 'rel_path',
}

#: Batch size for the `attributes.run_uuid IN (...)` filter in `_fetch_results`.
#: Results are ALWAYS scoped to the run uuids `query()` just found -- an
#: orphaned `FrictionResultsData` (a `run_uuid` naming no live run: the run was
#: deleted, or the node predates a rename) must never leak into a query just
#: because no run-level filter happened to be given. A single `{'in': [...]}`
#: over the whole list would do that safely, but a real store's run count can
#: run into the thousands (13k rows is the scale this store replaces) and
#: SQLite's default `SQLITE_MAX_VARIABLE_NUMBER` is 999 -- well under that, so
#: one IN-clause over every run can fail the query outright rather than merely
#: being slow. Kept comfortably under that limit; a test may monkeypatch this
#: down to exercise the chunking loop itself without building 500+ runs.
_RUN_UUID_CHUNK_SIZE = 500


def _range_filter(span: Tuple[Optional[float], Optional[float]]) -> Optional[Dict[str, Any]]:
    """One QueryBuilder operator dict for a `(lo, hi)` range, never two calls.

    R5: two `add_filter` calls on the SAME key overwrite each other on this
    backend -- the second call replaces the first rather than ANDing with it,
    so a naive `add_filter(lo); add_filter(hi)` silently drops the lower
    bound. Every range in this module goes through this helper instead, which
    folds both bounds into a single `{'and': [...]}` filter (or a single
    operator when one side of the range is open-ended, e.g. `(10.0, None)`).

    Returns `None` for a fully-open range (`(None, None)`) -- there is no
    constraint to express, and callers must skip `add_filter` entirely rather
    than pass `None` as a QueryBuilder operator value (which is exactly the
    `{'attributes.x': None}` shape R4 says never to construct).
    """
    lo, hi = span
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None:
        return {'and': [{'>=': lo}, {'<=': hi}]}
    if lo is not None:
        return {'>=': lo}
    return {'<=': hi}


def _result_sort_key(result, key: str) -> Any:
    """`order_by`'s raw sort value for one result node.

    `key` is usually a `FrictionResultsData` property (`load`, `angle`,
    `speed`, ...), but `order_by` may also name a bare `RunSummary` scalar
    (`mean_cof`) that has no property of its own -- those are read straight
    off the stored attribute instead.
    """
    if hasattr(type(result), key):
        return getattr(result, key)
    return result.base.attributes.get(key)


def _none_safe(value: Any) -> Tuple[bool, Any]:
    """A sort key that never compares `None` against a real value.

    `sorted()`/`list.sort()` raise `TypeError` comparing `None` to a float or
    a string -- a real risk here since several `RunSummary` extras and a
    sheet run's `layers`/`tip_radius`/etc. are legitimately `None`. Wrapping
    every key in `(value is None, value_or_placeholder)` sorts on the first
    element whenever the two values disagree on nullness, so the real second
    element is only ever compared against another real value of the same
    kind (or another placeholder of the same kind), never against `None`.
    """
    return (value is None, value if value is not None else 0)


def _default_sort_key(result, run_by_uuid: Dict[str, Any]) -> Tuple[Any, ...]:
    """Deterministic order for a `query()` with no `order_by`: by the owning
    run's `rel_path`, then `load`, `angle`, `speed` -- so two calls with the
    same filters always return results in the same order, instead of
    whatever order the backend happens to return them in.
    """
    run = run_by_uuid.get(result.run_uuid)
    rel_path = run.rel_path if run is not None else ''
    return (_none_safe(rel_path), _none_safe(result.load),
            _none_safe(result.angle), _none_safe(result.speed))


class Friction2DDB:
    """Query interface for the Friction2D simulation database.

    Wraps AiiDA's ``QueryBuilder`` with domain-specific methods for querying
    friction simulation data.

    Example::

        db = Friction2DDB()
        results = db.query(materials=['h-MoS2'], force_range=(5.0, 15.0))
        for run in results.runs:
            print(f"{run.materials} L{run.layers}: {run.rel_path}")

    Raises:
        RuntimeError: If AiiDA is not available or no profile is loaded.
    """

    def __init__(self):
        if not AIIDA_AVAILABLE:
            raise RuntimeError(
                "AiiDA is not available. Install with: pip install 'FrictionSim2D[aiida]'"
            )
        if _QueryBuilder is None or _load_node is None:
            raise RuntimeError("AiiDA ORM is not available")
        self._query_builder_cls = _QueryBuilder
        self._load_node = _load_node
        # Lazy import to avoid circular dependencies
        from .data import (  # pylint: disable=import-outside-toplevel
            FrictionSimulationData,
            FrictionResultsData,
        )
        self._simulation_data_cls = FrictionSimulationData
        self._results_data_cls = FrictionResultsData
        from .data import FrictionSimulationSetData  # pylint: disable=import-outside-toplevel
        self._simulation_set_cls = FrictionSimulationSetData

    # -- Simulation set queries -----------------------------------------------

    def list_sets(self) -> List[Dict[str, Any]]:
        """Every `FrictionSimulationSetData` node, newest first, as dicts."""
        qb = self._query_builder_cls()
        qb.append(self._simulation_set_cls, project=['*'])
        qb.order_by({self._simulation_set_cls: [{'ctime': 'desc'}]})
        return [node.to_dict() for node in qb.all(flat=True)]

    # -- The query --------------------------------------------------------

    def _fetch_results(self, run_uuids: List[str], *,
                       force_range: Optional[Tuple[Optional[float], Optional[float]]],
                       pressure_range: Optional[Tuple[Optional[float], Optional[float]]],
                       angle_range: Optional[Tuple[Optional[float], Optional[float]]],
                       speed_range: Optional[Tuple[Optional[float], Optional[float]]],
                       ) -> List[Any]:
        """Every result node (superseded ones included) whose `run_uuid` is in
        `run_uuids`, plus the four result-level range filters.

        ALWAYS filters on `run_uuid` -- never called with "skip the filter,
        every run matched anyway", because a result node whose `run_uuid`
        names no live run (orphaned: the run was deleted, or predates a
        rename) would otherwise be returned regardless of what `query()` was
        actually asked for. `run_uuids` is chunked into batches of
        `_RUN_UUID_CHUNK_SIZE` -- one QueryBuilder query per batch, results
        concatenated -- so this stays correct however large the run count
        gets, per-key range filters unchanged (see `query()`'s docstring for
        why `force_range`/`pressure_range` also gate on `load_kind`).
        """
        results: List[Any] = []
        for start in range(0, len(run_uuids), _RUN_UUID_CHUNK_SIZE):
            chunk = run_uuids[start:start + _RUN_UUID_CHUNK_SIZE]
            res_qb = self._query_builder_cls()
            res_qb.append(self._results_data_cls, tag='res', project=['*'],
                          filters={'attributes.run_uuid': {'in': chunk}})
            if force_range is not None:
                res_qb.add_filter('res', {'attributes.load_kind': 'force'})
                force_filter = _range_filter(force_range)
                if force_filter is not None:
                    res_qb.add_filter('res', {'attributes.load': force_filter})
            if pressure_range is not None:
                res_qb.add_filter('res', {'attributes.load_kind': 'pressure'})
                pressure_filter = _range_filter(pressure_range)
                if pressure_filter is not None:
                    res_qb.add_filter('res', {'attributes.load': pressure_filter})
            if angle_range is not None:
                angle_filter = _range_filter(angle_range)
                if angle_filter is not None:
                    res_qb.add_filter('res', {'attributes.angle': angle_filter})
            if speed_range is not None:
                speed_filter = _range_filter(speed_range)
                if speed_filter is not None:
                    res_qb.add_filter('res', {'attributes.speed': speed_filter})
            results.extend(res_qb.all(flat=True))
        return results

    def query(  # pylint: disable=too-many-arguments,too-many-locals
        self, *, materials: Optional[List[str]] = None,
        family: Optional[str] = None, model: Optional[str] = None,
        set_label: Optional[str] = None, potential: Optional[str] = None,
        layers: Optional[Union[int, List[int]]] = None,
        substrate: Optional[str] = None,
        substrate_amorphous: Optional[bool] = None,
        tip_material: Optional[str] = None, tip_radius: Optional[float] = None,
        flake_material: Optional[str] = None,
        size_x: Optional[float] = None, size_y: Optional[float] = None,
        rel_path: Optional[str] = None,
        temperature_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        force_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        pressure_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        angle_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        speed_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        limit: Optional[int] = None, order_by: Optional[str] = None,
    ) -> QueryResult:
        """Filter on every RunTokens axis plus family, model, set and potential.

        Every parameter defaults to `None`, meaning "do not filter on this" --
        this API never builds a QueryBuilder filter for an explicit `None`
        (R4: `{'attributes.x': None}` raises `TypeError` on SQLite), so a
        sheet-family run -- whose `layers`/`tip_radius`/`substrate`/
        `flake_material` attributes ARE `None` -- is never the target of a
        crashing filter; it is simply unaffected by an axis nobody asked to
        filter on.

        `force_range`/`pressure_range`/`angle_range`/`speed_range` filter the
        RESULT nodes; every other filter (including `set_label`, which is an
        attribute of the run, not a provenance edge to the set) filters the
        RUN nodes first, and only results of a matching run are considered
        (R9 also drops any result marked `is_superseded`). `limit`/`order_by`
        apply to the final result list, so a superseded row is never counted
        against `limit`.
        """
        run_eq = {
            'family': family, 'model': model, 'set_label': set_label,
            'potential': potential, 'substrate': substrate,
            'substrate_amorphous': substrate_amorphous,
            'tip_material': tip_material, 'tip_radius': tip_radius,
            'flake_material': flake_material, 'size_x': size_x,
            'size_y': size_y, 'rel_path': rel_path,
        }

        run_qb = self._query_builder_cls()
        run_qb.append(self._simulation_data_cls, tag='run', project=['*'])
        if materials:
            # ALL semantics (spec): a run matches only if it names EVERY
            # material given, not merely one of them -- `'contains'` is a
            # containment check (the run's own `materials` list is a
            # superset of `materials`), not a `{'in': ...}` any-of check.
            run_qb.add_filter('run',
                              {'attributes.materials': {'contains': materials}})
        for name, value in run_eq.items():
            if value is not None:
                run_qb.add_filter('run',
                                  {f'attributes.{_RUN_EQ_FILTERS[name]}': value})
        if layers is not None:
            run_qb.add_filter('run', {'attributes.layers':
                                      layers if isinstance(layers, int)
                                      else {'in': list(layers)}})
        if temperature_range is not None:
            temp_filter = _range_filter(temperature_range)
            if temp_filter is not None:
                run_qb.add_filter('run', {'attributes.temperature': temp_filter})
        runs = sorted(run_qb.all(flat=True), key=lambda n: n.rel_path)
        run_uuids = [str(run.uuid) for run in runs]
        run_by_uuid = dict(zip(run_uuids, runs))

        results: List[Any] = []
        if runs:
            fetched = self._fetch_results(
                run_uuids, force_range=force_range, pressure_range=pressure_range,
                angle_range=angle_range, speed_range=speed_range)
            # R9: a superseded result must never be returned by a query --
            # excluded here, in Python, the same way `export._results_of`
            # and `ingest.find_results` do (the one shared `is_superseded`
            # predicate, not a second QueryBuilder JSONB-extras filter).
            results = [r for r in fetched if not is_superseded(r)]

        if order_by:
            key = order_by[1:] if order_by.startswith('-') else order_by
            results.sort(key=lambda r: _none_safe(_result_sort_key(r, key)),
                        reverse=order_by.startswith('-'))
        else:
            # Deterministic even with no explicit order_by -- two identical
            # queries never return results in a different order just because
            # the backend felt like it.
            results.sort(key=lambda r: _default_sort_key(r, run_by_uuid))
        if limit:
            results = results[:limit]

        kept_run_uuids = {r.run_uuid for r in results}
        return QueryResult(
            runs=[run for run in runs if str(run.uuid) in kept_run_uuids],
            results=results, total_count=len(results),
            query_params={
                'materials': materials, 'family': family, 'model': model,
                'set_label': set_label, 'potential': potential,
                'layers': layers, 'substrate': substrate,
                'substrate_amorphous': substrate_amorphous,
                'tip_material': tip_material, 'tip_radius': tip_radius,
                'flake_material': flake_material, 'size_x': size_x,
                'size_y': size_y, 'rel_path': rel_path,
                'temperature_range': temperature_range,
                'force_range': force_range, 'pressure_range': pressure_range,
                'angle_range': angle_range, 'speed_range': speed_range,
                'limit': limit, 'order_by': order_by})

    def query_by_set(self, label: str) -> QueryResult:
        """Every run (and result) of one set, by its label."""
        return self.query(set_label=label)

    # -- Aggregation ----------------------------------------------------------

    def get_available_conditions(self) -> Dict[str, List[Any]]:
        """Distinct values behind each GUI filter dropdown.

        Ten keys, one per combo in the GUI's filter bar (its database-pane
        spec D3 populates the whole bar from this ONE call, so a filter with
        no key here has no way to be populated). `potentials` are the
        `pot_token` values carried on the run nodes; `set_labels` are the
        labels of the `friction2d/<label>` groups, which is the authoritative
        list -- a set exists as a group even before any run is attached to
        it. A superseded result (R9) never contributes a value here, the same
        way it is never returned by `query()`.
        """
        from aiida.orm import Group  # pylint: disable=import-outside-toplevel

        from .ingest import GROUP_PREFIX  # pylint: disable=import-outside-toplevel

        qb = self._query_builder_cls()
        qb.append(self._results_data_cls, project=['*'])
        forces, pressures, angles, speeds, families = (set(), set(), set(),
                                                        set(), set())
        for (node,) in qb.all():
            if is_superseded(node):
                continue
            # A PES grid's `load_kind` is 'none' and its load/angle/speed are
            # placeholder zeros, not conditions anyone ran at: they must not
            # show up as a 0 GPa pressure (or a 0 deg / 0 speed) option.
            if node.load_kind == 'force':
                forces.add(node.load)
            elif node.load_kind == 'pressure':
                pressures.add(node.load)
            if node.load_kind in ('force', 'pressure'):
                angles.add(node.angle)
                speeds.add(node.speed)
            if node.family:
                families.add(node.family)

        qb2 = self._query_builder_cls()
        qb2.append(self._simulation_data_cls, project=['*'])
        temps, layers, models, potentials = set(), set(), set(), set()
        for (node,) in qb2.all():
            temps.add(node.temperature)
            if node.layers is not None:
                layers.add(node.layers)
            if node.model:
                models.add(node.model)
            if node.pot_token:
                potentials.add(node.pot_token)

        qb3 = self._query_builder_cls()
        qb3.append(Group, filters={'label': {'like': f'{GROUP_PREFIX}/%'}},
                  project=['label'])
        prefix = f'{GROUP_PREFIX}/'
        set_labels = {label[len(prefix):] for (label,) in qb3.all() if label}

        return {'forces': sorted(forces), 'pressures': sorted(pressures),
                'angles': sorted(angles), 'speeds': sorted(speeds),
                'temperatures': sorted(temps), 'layers': sorted(layers),
                'families': sorted(families), 'models': sorted(models),
                'potentials': sorted(potentials),
                'set_labels': sorted(set_labels)}

    def get_available_materials(self) -> List[str]:
        """Sorted, deduplicated list of every material named on a run."""
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, project=['attributes.materials'])
        return sorted({m for (mats,) in qb.all() for m in (mats or [])})

    def get_statistics(self) -> Dict[str, Any]:
        """Counts by status, model, family and material across every run."""
        stats: Dict[str, Any] = {
            'total_simulations': 0, 'by_status': {}, 'by_model': {},
            'by_family': {}, 'by_material': {}, 'n_materials': 0,
        }
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, project=['*'])
        for (node,) in qb.all():
            stats['total_simulations'] += 1
            stats['by_status'][node.status] = stats['by_status'].get(
                node.status, 0) + 1
            stats['by_model'][node.model] = stats['by_model'].get(
                node.model, 0) + 1
            stats['by_family'][node.family] = stats['by_family'].get(
                node.family, 0) + 1
            for material in node.materials:
                stats['by_material'][material] = stats['by_material'].get(
                    material, 0) + 1
        stats['n_materials'] = len(stats['by_material'])
        return stats

    # -- Provenance and reproducibility ---------------------------------------

    def export_for_reproduction(self, run, output_dir: Path) -> Path:
        """Export one run's WHOLE SET back to a tree.

        Every run of the set and every results file, not just `run` -- this
        is the "recreate the run tree" action. For one run's provenance files
        alone, use `src.aiida.export.export_provenance_bundle(run, dest)`.
        """
        from .export import export_results  # pylint: disable=import-outside-toplevel

        if isinstance(run, str):
            run = self._load_node(run)
        export_results(run.set_label, Path(output_dir))
        return Path(output_dir)
