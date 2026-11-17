"""Subprocess entry point for database queries and downloads.

Everything the Database pane asks for runs here, for the reason the preview
worker exists: ``import aiida`` plus ``load_profile`` is slow and
process-global, and psycopg2 can hang. A hung or crashed query kills
this process, not the window.

Invoked as ``python -m src.gui.db_worker <request.json>``. Progress lines go to
stdout as JSON; the result is written to ``request['result_file']``.

Every heavy import (aiida, pandas) is inside a function: ``src.gui``
must stay importable on a machine that has none of them.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

#: The kwargs `Friction2DDB.query` accepts. Anything else in a request's
#: `filters` block is dropped rather than passed through -- a typo in the pane
#: must not become a TypeError in the store. `Friction2DDB.query` has no
#: `is_complete` parameter (there is no Completeness filter in the pane), so
#: it is not in this tuple.
FILTER_KEYS = (
    'materials', 'family', 'model', 'set_label', 'potential', 'layers',
    'force_range', 'pressure_range', 'temperature_range', 'angle_range',
    'speed_range', 'substrate', 'tip_material', 'limit', 'order_by',
)

#: The combo-box choices a `list` response carries back, beside `materials`.
FACET_KEYS = ('families', 'models', 'potentials', 'layers', 'temperatures',
              'set_labels')


def _emit(kind: str, **payload: Any) -> None:
    """Write one progress record to stdout."""
    print(json.dumps({'kind': kind, **payload}), flush=True)


def _load_profile(request: Dict[str, Any]):
    """Load the AiiDA profile this request runs against."""
    from ..aiida import load_aiida_profile  # pylint: disable=import-outside-toplevel
    return load_aiida_profile(request.get('profile'))


def _local_db(request: Dict[str, Any]):
    """The local store's query interface, profile loaded."""
    _load_profile(request)
    from ..aiida.query import Friction2DDB  # pylint: disable=import-outside-toplevel
    return Friction2DDB()


def _records(frame) -> List[Dict[str, Any]]:
    """Row dicts that ``json.dumps`` accepts.

    ``DataFrame.to_dict`` hands back numpy scalars and NaN; the first is not
    JSON-serialisable and the second is not valid JSON. ``to_json`` handles
    both -- NaN becomes null -- so the round trip through pandas is the
    conversion rather than a hand-written one.
    """
    return json.loads(frame.to_json(orient='records', date_format='iso'))


def _facet_lists(conditions: Dict[str, Any]) -> Dict[str, List[Any]]:
    """The FACET_KEYS lists, sorted and None-free, from whatever is offered.

    Tolerant of a store that does not offer every key: a missing facet comes
    back empty and `_facets_from_records` backfills it.
    """
    out: Dict[str, List[Any]] = {}
    for key in FACET_KEYS:
        values = conditions.get(key) if isinstance(conditions, dict) else None
        out[key] = sorted({v for v in (values or []) if v is not None}, key=str)
    return out


#: Which record field(s) each facet is the distinct values of. The potential
#: axis is `pot_token` in the local store and `potential_type` in older
#: records, so a facet may read more than one field.
_FACET_FIELDS = (('families', ('family',)), ('models', ('model',)),
                 ('potentials', ('pot_token', 'potential_type')),
                 ('set_labels', ('set_label',)), ('layers', ('layers',)),
                 ('temperatures', ('temperature',)))


def _facets_from_records(records: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Choices read off the rows just fetched.

    The fallback for any facet the local store does not offer. Only as
    complete as the last query, which is why it is a fallback and not the
    primary source.
    """
    out: Dict[str, List[Any]] = {}
    for facet, fields in _FACET_FIELDS:
        values = {r.get(field) for r in records for field in fields
                  if r.get(field) is not None}
        out[facet] = sorted(values, key=str)
    return out


def _backfill_facets(facets: Dict[str, List[Any]],
                     records: List[Dict[str, Any]]) -> List[str]:
    """Fill any facet the store did not offer from the fetched rows.

    Returns the facets it had to fill, so the caller can say so: a choice list
    derived from one page of rows is not the same promise as one derived from
    the store, and a user narrowing by it deserves to know.
    """
    derived = _facets_from_records(records)
    filled = []
    for key, values in derived.items():
        if not facets.get(key):
            facets[key] = values
            filled.append(key)
    return filled


def _facet_warnings(filled: List[str], n_records: int) -> List[str]:
    """Say which filter choices came from the rows rather than the store."""
    if not filled:
        return []
    names = ' and '.join(f.replace('_', ' ') for f in sorted(filled))
    return [f"the {names} choices come from the {n_records} rows fetched, "
            f"not from the store"]


def _set_sizes(sets: Any) -> Dict[str, int]:
    """How many runs each set holds, for the run_tree confirmation.

    Tolerates both shapes the query API has had: a list of dicts and a list of
    set nodes. A set it cannot measure is simply absent, and the confirmation
    says the count is unknown rather than inventing one.
    """
    out: Dict[str, int] = {}
    for entry in sets or []:
        if isinstance(entry, dict):
            label, runs = entry.get('label'), entry.get('n_runs')
        else:
            label, runs = getattr(entry, 'label', None), getattr(entry, 'n_runs', None)
        if label and isinstance(runs, int):
            out[str(label)] = runs
    return out


def _list(request: Dict[str, Any]) -> Dict[str, Any]:
    """Rows for the table, plus the choices the filter combos offer."""
    filters = {k: v for k, v in (request.get('filters') or {}).items()
               if k in FILTER_KEYS and v is not None}
    started = time.monotonic()

    limit = filters.get('limit')
    db = _local_db(request)
    _emit('progress', message='querying the local store')
    # One projection of the store's points serves the rows and the facets.
    snapshot = getattr(db, 'snapshot', None)
    with snapshot() if snapshot is not None else nullcontext():
        result = db.query(**filters)
        records = _records(result.to_dataframe())
        facets = {'materials': list(db.get_available_materials()),
                  **_facet_lists(db.get_available_conditions())}
    total = result.total_count
    # The run counts the run_tree confirmation quotes.
    sizes = _set_sizes(db.list_sets())

    filled = _backfill_facets(facets, records)
    return {
        'ok': True, 'kind': 'list', 'source': 'local',
        'records': records,
        'columns': list(records[0]) if records else [],
        'facets': facets,
        'set_sizes': sizes,
        'total_count': total, 'returned': len(records), 'limit': limit,
        'elapsed_s': round(time.monotonic() - started, 3),
        'warnings': _facet_warnings(filled, len(records)),
    }


def _related_kinds() -> Tuple[str, ...]:
    """What `detail` counts `related` points along: every block kind
    (`realization` among them) and the swept axes `load` and `speed`."""
    from ..aiida.block_schema import BLOCK_KINDS  # pylint: disable=import-outside-toplevel
    return tuple(BLOCK_KINDS) + ('load', 'speed')


def _node_dict(node) -> Optional[Dict[str, Any]]:
    """Identity plus every stored attribute of one node; None for no node."""
    if node is None:
        return None
    return {'uuid': node.uuid, 'node_type': node.node_type,
            'attributes': dict(node.base.attributes.all)}


def _repository_files(node, which: str) -> List[Dict[str, Any]]:
    """The node's stored files with their sizes.

    `as_path` materialises the repository as a real directory, which is the
    only way to get a size without reading every file into memory.
    """
    out: List[Dict[str, Any]] = []
    with node.base.repository.as_path() as root:
        for path in sorted(p for p in root.rglob('*') if p.is_file()):
            out.append({'node': which, 'path': str(path.relative_to(root)),
                        'bytes': path.stat().st_size})
    return out


def _point(item: Dict[str, Any]):
    """The point `FolderData` a row names, by its `node_uuid`.

    Not by `point_uuid`: a re-import supersedes a point and keeps its
    `point_uuid`, so only the node uuid names exactly the row on screen.
    """
    from aiida.orm import load_node  # pylint: disable=import-outside-toplevel
    return load_node(uuid=item['node_uuid'])


def _related_counts(db, point) -> Dict[str, int]:
    """How many current points differ from `point` only along each kind,
    from one `point_table()`. Empty when the point is not in the table (its
    blocks carry no hash extras: `FrictionSim2D aiida reindex`)."""
    table = db.point_table()
    counts: Dict[str, int] = {}
    for kind in _related_kinds():
        try:
            counts[kind] = len(db.related(point, kind, table=table))
        except LookupError:
            return {}
    return counts


def _detail(request: Dict[str, Any]) -> Dict[str, Any]:
    """Everything about one selected row -- without downloading anything:
    the point's attributes and stored files, the verdict and summary the
    policy selects, and the `related` counts."""
    db = _local_db(request)
    point = _point(request)
    files = _repository_files(point, 'point')
    return {
        'ok': True, 'kind': 'detail', 'source': 'local',
        'point': _node_dict(point),
        'verdict': _node_dict(db.latest(point, 'verdict')),
        'summary': _node_dict(db.latest(point, 'summary')),
        'related': _related_counts(db, point),
        'files': files, 'total_bytes': sum(f['bytes'] for f in files),
    }


#: What a download can be. All four map to a function the AiiDA rework
#: already defines -- nothing about the store's layout is reimplemented here.
EXPORT_KINDS = ('results', 'provenance', 'summary_csv', 'run_tree')


def _csv_value(value: Any) -> Any:
    """A cell a CSV reader gets back unchanged."""
    if isinstance(value, (list, tuple)):
        return '+'.join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return '' if value is None else value


def _export_response(request: Dict[str, Any], dest, files: List[Any],
                     warnings: Any = ()) -> Dict[str, Any]:
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    paths = sorted({Path(p) for p in files})
    return {
        'ok': True, 'kind': 'export', 'what': request['what'], 'dest': str(dest),
        'files': [str(p) for p in paths],
        'bytes': sum(p.stat().st_size for p in paths if p.is_file()),
        'warnings': list(warnings),
    }


def _export_summary_csv(request: Dict[str, Any], dest) -> Dict[str, Any]:
    """The rows on screen, exactly as the table shows them."""
    import csv  # pylint: disable=import-outside-toplevel
    records = request.get('records') or []
    columns = request.get('columns') or (list(records[0]) if records else [])
    target = dest / 'summary.csv'
    with target.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow({c: _csv_value(record.get(c)) for c in columns})
    return _export_response(request, dest, [target])


def _stored_paths(node) -> List[str]:
    """Every file path in a node's repository."""
    return [(root / name).as_posix()
            for root, _dirs, files in node.base.repository.walk() for name in files]


def _export_result_files(request: Dict[str, Any], dest) -> Dict[str, Any]:
    """Each selected point's stored tier files (window and raw), verbatim,
    under <dest>/<rel_path>/. `run_tree` is the download with the per-atom
    files, snapshots and logs."""
    written: List[Any] = []
    warnings: List[str] = []
    for item in request.get('selection') or []:
        point = _point(item)
        rel_path = point.base.attributes.get('rel_path')
        target = dest / rel_path
        target.mkdir(parents=True, exist_ok=True)
        _emit('progress', message=f"copying {rel_path}")
        point.base.repository.copy_tree(target)
        stored = _stored_paths(point)
        if not stored:
            # A file over `settings.aiida.max_stored_file_mb` is recorded by
            # hash instead of stored, so a point can be in the store and have
            # nothing to hand back. Silence here reads as a failed download.
            warnings.append(f"{rel_path}: nothing stored (files over the "
                            f"size limit are recorded by hash)")
        written += [target / path for path in stored]
    return _export_response(request, dest, written, warnings)


def _export_provenance(request: Dict[str, Any], dest) -> Dict[str, Any]:
    """provenance.json, config.json, settings.yaml, cif and potentials,
    copied out of the `provenance` input of each selected point's CalcJob.

    One bundle per selected run and nothing else -- deliberately not
    `export_results`, which exports the run's whole set and would turn
    "download the provenance of this row" into a multi-gigabyte copy of
    every run beside it. Two points of one run share one bundle.
    """
    from ..aiida.export import (  # pylint: disable=import-outside-toplevel
        export_provenance_bundle)
    written: List[Any] = []
    done = set()
    for item in request.get('selection') or []:
        calc = _point(item).creator
        if calc.uuid in done:
            continue
        done.add(calc.uuid)
        rel_path = calc.inputs.parameters.get('rel_path')
        _emit('progress', message=f"provenance for {rel_path}")
        written += list(export_provenance_bundle(calc, dest / rel_path))
    return _export_response(request, dest, written)


def _export_run_tree(request: Dict[str, Any], dest) -> Dict[str, Any]:
    """A tree `postprocess read` can read, optionally with lammps/ rebuilt."""
    from ..aiida.export import (  # pylint: disable=import-outside-toplevel
        export_results, rebuild_set)
    labels = sorted({item.get('set_label') for item in (request.get('selection') or [])
                     if item.get('set_label')})
    written: List[Any] = []
    warnings: List[str] = []
    for label in labels:
        _emit('progress', message=f"exporting set {label}")
        written += list(export_results(label, dest / label,
                                       profile=request.get('profile')))
        # The unit of this download is the SET, not the selected row -- the
        # response says so, because the destination will hold runs the user
        # never selected.
        warnings.append(f"set {label!r} was exported in full, not only the "
                        f"selected rows")
        if request.get('rebuild'):
            _emit('progress', message=f"rebuilding lammps/ for {label}")
            files, _root = rebuild_set(label, dest / label,
                                       profile=request.get('profile'))
            written += list(files)
    return _export_response(request, dest, written, warnings)


def _export(request: Dict[str, Any]) -> Dict[str, Any]:
    """One download. Raises; `main` turns that into the error shape."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    what = request.get('what')
    if what not in EXPORT_KINDS:
        raise ValueError(f"unknown export kind: {what!r}")
    dest = Path(request['dest']).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    if what == 'summary_csv':
        return _export_summary_csv(request, dest)
    _load_profile(request)
    if what == 'results':
        return _export_result_files(request, dest)
    if what == 'provenance':
        return _export_provenance(request, dest)
    return _export_run_tree(request, dest)


def _ingest_root(root, label, profile, dry_run):
    """Indirection so a test can replace the import without a live store."""
    from ..aiida.ingest import import_root  # pylint: disable=import-outside-toplevel
    return import_root(root, label=label, profile=profile, dry_run=dry_run)


def _import(request: Dict[str, Any]) -> Dict[str, Any]:
    """Import a run directory into the local store (`import_root`).

    The one write action the pane offers: it is the step a remote-manual user
    does on every return trip. `ok` says the import ran; `ingest_ok` says it
    ran clean -- a report with failures is still a report worth reading.
    """
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    _load_profile(request)
    root = Path(request['root']).expanduser()
    _emit('progress', message=f"importing {root}")
    report = _ingest_root(root, request.get('label') or None,
                          request.get('profile'), bool(request.get('dry_run')))
    return {
        'ok': True, 'kind': 'import', 'ingest_ok': bool(report.ok()),
        'label': report.label, 'group_uuid': report.group_uuid,
        'created_decks': list(report.created_decks),
        'skipped_decks': list(report.skipped_decks),
        'superseded_decks': list(report.superseded_decks),
        'already_imported': list(report.already_imported),
        'n_points': len(report.points),
        'verdict_counts': dict(report.verdict_counts),
        # Decks whose log has not arrived yet: not a failure, but something
        # the user wants to see before trusting the numbers that just arrived.
        'log_missing': list(report.log_missing),
        'no_version': list(report.no_version),
        # A run whose `provenance.json` could not even be parsed. `aiida
        # import` exits non-zero on this (N8); the pane must not bury it either.
        'unparseable': list(report.unparseable),
        'failures': [list(f) for f in report.failures],
        'warnings': list(report.warnings),
        'dry_run': bool(report.dry_run),
        'report': report.render(),
    }


def _check(request: Dict[str, Any]) -> Dict[str, Any]:
    """Can the local store be reached, and with what.

    A fifth request kind beside D8's four: loading an AiiDA profile is
    exactly the work that must not run on the UI thread, so the check goes
    through the same subprocess everything else does. `ok` says the check
    ran; `ready` says the store is usable.
    """
    from importlib.util import find_spec  # pylint: disable=import-outside-toplevel
    source = request.get('source', 'local')
    checks: List[Dict[str, Any]] = []
    notes: List[str] = []

    has_aiida = find_spec('aiida') is not None
    checks.append({'name': 'aiida installed', 'ok': has_aiida,
                   'detail': '' if has_aiida else "pip install -e '.[aiida]'"})
    if not has_aiida:
        notes.append("The local store needs aiida-core: pip install -e '.[aiida]'")
    else:
        try:
            profile = _load_profile(request)
            checks.append({'name': 'profile loaded', 'ok': True,
                           'detail': str(profile)})
            checks.append({'name': 'sets in the store', 'ok': True,
                           'detail': str(len(_local_db(request).list_sets()))})
        except Exception as exc:  # pylint: disable=broad-except
            checks.append({'name': 'profile loaded', 'ok': False,
                           'detail': f"{type(exc).__name__}: {exc}"})
            notes.append('Check settings.aiida.profile, or run: '
                         'FrictionSim2D aiida setup')

    return {'ok': True, 'kind': 'check', 'source': source, 'checks': checks,
            'notes': notes, 'ready': all(c['ok'] for c in checks)}


#: What a request for the removed 'central' source answers.
CENTRAL_REMOVED = 'central database removed; use the local AiiDA store'

HANDLERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    'list': _list,
    'detail': _detail,
    'export': _export,
    'import': _import,
    'check': _check,
}


def _error(kind: str, exc: BaseException) -> Dict[str, Any]:
    """The one failure shape, whatever failed."""
    return {'ok': False, 'kind': kind, 'error': f"{type(exc).__name__}: {exc}",
            'traceback': traceback.format_exc()}


def dispatch(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run one request and return its result -- never raise.

    Handlers raise freely; this is where that becomes the error shape, so
    calling `dispatch` directly, from a test or from `main`, behaves
    identically.
    """
    kind = request.get('kind') or ''
    if request.get('source') == 'central':
        return {'ok': False, 'kind': kind, 'error': CENTRAL_REMOVED,
                'traceback': ''}
    handler = HANDLERS.get(kind)
    if handler is None:
        return {'ok': False, 'kind': kind,
                'error': f"unknown request kind: {kind!r}", 'traceback': ''}
    try:
        return handler(request)
    except Exception as exc:  # pylint: disable=broad-except
        return _error(kind, exc)


def _read(path: str) -> str:
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    return Path(path).read_text(encoding='utf-8')


def _write(path: str, text: str) -> None:
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    Path(path).write_text(text, encoding='utf-8')


def main(argv=None) -> int:
    """Read the request, dispatch it, write the result. No try/except of its
    own -- `dispatch` already turns every handler exception into the error
    shape, so there is nothing left here for one to escape into."""
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        _emit('error', message='no request file given')
        return 2

    request = json.loads(_read(argv[0]))
    result = dispatch(request)
    _write(request['result_file'], json.dumps(result, indent=2, default=str))
    _emit('done', ok=result.get('ok', False))
    return 0 if result.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
