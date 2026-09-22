"""Subprocess entry point for database queries and downloads.

Everything the Database pane asks for runs here, for the reason the preview
worker exists: ``import aiida`` plus ``load_profile`` is slow and
process-global, and psycopg2/httpx can hang. A hung or crashed query kills
this process, not the window.

Invoked as ``python -m src.gui.db_worker <request.json>``. Progress lines go to
stdout as JSON; the result is written to ``request['result_file']``.

Every heavy import (aiida, httpx, pandas) is inside a function: ``src.gui``
must stay importable on a machine that has none of them.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any, Callable, Dict, List

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


def _add_imported_at(records: List[Dict[str, Any]]) -> None:
    """Normalise the Added column's "never stored" state to ``None``.

    `QueryResult.to_dataframe()` already carries `imported_at` on every row --
    it comes from `FrictionSimulationData.to_dict()`, which reads the run
    node's `imported_at` extra and defaults to the empty string if `status`
    was never set. This only turns that empty-string default into `None`, so
    the GUI's Added column can tell "not stored" from a real timestamp with
    `is None` rather than a string comparison; it does not fetch or add
    anything that was not already on the row.
    """
    for record in records:
        if record.get('imported_at') == '':
            record['imported_at'] = None


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
#: axis is `pot_token` in the local store and `potential_type` in the central
#: one, so a facet may read more than one field.
_FACET_FIELDS = (('families', ('family',)), ('models', ('model',)),
                 ('potentials', ('pot_token', 'potential_type')),
                 ('set_labels', ('set_label',)), ('layers', ('layers',)),
                 ('temperatures', ('temperature',)))


def _facets_from_records(records: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Choices read off the rows just fetched.

    The whole answer for the central source -- its `/conditions` endpoint
    replies with ranges and count maps rather than choice lists, and knows
    nothing of potentials or sets -- and the fallback for any facet the local
    store does not offer. Only as complete as the last query, which is why it
    is a fallback and not the primary source.
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

    db = _local_db(request)
    _emit('progress', message='querying the local store')
    result = db.query(**filters)
    records = _records(result.to_dataframe())
    _add_imported_at(records)
    facets = {'materials': list(db.get_available_materials()),
              **_facet_lists(db.get_available_conditions())}
    total = result.total_count

    filled = _backfill_facets(facets, records)
    return {
        'ok': True, 'kind': 'list', 'source': request.get('source', 'local'),
        'records': records,
        'columns': list(records[0]) if records else [],
        'facets': facets,
        'set_sizes': _set_sizes(db.list_sets()),
        'total_count': total, 'returned': len(records),
        'elapsed_s': round(time.monotonic() - started, 3),
        'warnings': _facet_warnings(filled, len(records)),
    }


HANDLERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    'list': _list,
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
