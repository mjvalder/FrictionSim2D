"""The query layer over the CalcJob graph.

Every point is a ``FolderData`` output (``points__<label>``) of one deck
CalcJob; the CalcJob holds the run's parameter blocks (``blocks__<name>``)
and the family's ``schema``. Everything here is read through bulk
projections and reduced in pandas, never node by node:

* :meth:`Friction2DDB.query` -- one row per live point: the run tokens (the
  CalcJob extra ``friction2d_tokens``), the point's own attributes with
  units, the verdict and the summary the :class:`~frictionsim2d.aiida.rulings.Policy`
  selects.
* :func:`point_table` -- one row per point, one column per requested field
  (``<block name>.<field path>``, elided fields refilled from the run's own
  schema) plus the swept axes ``load``, ``angle``, ``speed``.
* :func:`related` -- the points that differ from one point only along
  ``vary``: point-level and field-level, whatever deck shape (array or
  combined mode) carried them. Joined on block hash values, never on node
  identity.
* :func:`latest` -- the verdict or summary the policy selects for a point.
* :func:`dataset_rows` -- a dataset's rows joined with their points and the
  summaries it fixed, for the plot source ``aiida://dataset/<name>``.
* :func:`list_simulation_sets` -- one row per set (``aiida list-sets``).

Superseded nodes (extra ``superseded=True``, on the point or its CalcJob)
are left out everywhere unless asked for.
"""
from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence,
                    Set, Tuple, Union)

from ..core.compare import canonical_text, refill
from .block_schema import BLOCK_KINDS, FIELD_MAP, REALIZATION_FIELDS, unindexed
from .blocks import NAME_SEP, block_kind
from .compare import block_name, schema_tables
from .layout import (
    DATASET_PREFIX, GROUP_PREFIX, SET_GROUP_FILTER, is_set_label, is_superseded,
)
from .rulings import Policy

try:
    from aiida import orm
    AIIDA_AVAILABLE = True
except ImportError:  # pragma: no cover - the aiida tier skips without it
    orm = None  # type: ignore[assignment]
    AIIDA_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ['AXES', 'Friction2DDB', 'QueryResult', 'dataset_rows', 'latest',
           'list_simulation_sets', 'point_table', 'related']

#: The swept axes: point attributes, never block content (``SWEPT``).
AXES = ('load', 'angle', 'speed')

#: The block kind ``related`` ignores unless it is named (seeds differ
#: between every pair of runs).
REALIZATION = 'realization'

_POINTS_LINK = 'points__%'
_BLOCKS_LINK = 'blocks__'
_SCHEMA_LINK = 'schema'

#: Batch size of an ``IN`` filter (SQLite caps the variables of a statement).
_CHUNK = 500

#: Swept-axis values equal to this many decimals are one value (calcjob._TOL).
_AXIS_DECIMALS = 9

#: RunTokens fields, in column order (``family`` comes from the point).
_TOKEN_FIELDS = ('model', 'materials', 'flake_material', 'size_x', 'size_y',
                 'substrate', 'substrate_amorphous', 'tip_material', 'tip_radius',
                 'temperature', 'pot_token', 'interlayer_token')

#: Point attributes copied into a ``query`` row (``status`` as ``point_status``).
_POINT_FIELDS = ('family', 'layers', 'rel_path', 'deck_name', 'point_uuid', 'load_kind',
                 'load', 'angle', 'speed', 'status', 'status_detail', 'load_achieved',
                 'ramp_converged', 'has_snapshot', 'has_peratom')

#: Summary keys that are metadata, not scalars.
_SUMMARY_META = ('units', 'options', 'options_hash', 'code_version', 'point_uuid')

#: ``query`` keyword -> the row column it filters on by equality.
_EQ_FILTERS = {
    'family': 'family', 'model': 'model', 'potential': 'pot_token',
    'substrate': 'substrate', 'substrate_amorphous': 'substrate_amorphous',
    'tip_material': 'tip_material', 'tip_radius': 'tip_radius',
    'flake_material': 'flake_material', 'size_x': 'size_x', 'size_y': 'size_y',
    'rel_path': 'rel_path',
}

Range = Tuple[Optional[float], Optional[float]]


def _require_aiida() -> None:
    if not AIIDA_AVAILABLE:
        raise RuntimeError(
            "AiiDA is not available. Install with: pip install 'FrictionSim2D[aiida]'")


def _chunks(values: Sequence[Any]) -> Iterable[List[Any]]:
    for start in range(0, len(values), _CHUNK):
        yield list(values[start:start + _CHUNK])


def _process_type() -> str:
    from .store import CALCJOB_PROCESS_TYPE  # pylint: disable=import-outside-toplevel
    return CALCJOB_PROCESS_TYPE


@lru_cache(maxsize=None)
def _parsed_tokens(rel_path: str, family: Optional[str]) -> Optional[str]:
    from .store import run_tokens  # pylint: disable=import-outside-toplevel
    tokens = run_tokens(rel_path, family)
    return None if tokens is None else json.dumps(tokens)


def _tokens(extras: Mapping[str, Any], rel_path: Optional[str],
            family: Optional[str]) -> Dict[str, Any]:
    """The CalcJob's ``friction2d_tokens``; parsed from ``rel_path`` for a
    CalcJob imported before the extra existed (``aiida reindex`` writes it)."""
    from .store import TOKENS_EXTRA  # pylint: disable=import-outside-toplevel
    tokens = extras.get(TOKENS_EXTRA)
    if tokens is None and rel_path:
        text = _parsed_tokens(rel_path, family)
        tokens = json.loads(text) if text else None
    return dict(tokens or {})


def _set_group_rows() -> List[Tuple[str, str, int, Dict[str, Any]]]:
    """``(label, uuid, pk, extras)`` of every ``friction2d/<label>`` set Group."""
    qb = orm.QueryBuilder().append(
        orm.Group, filters=SET_GROUP_FILTER, project=['label', 'uuid', 'id', 'extras'])
    return [(label, uuid, pk, extras or {}) for label, uuid, pk, extras in qb.all()
            if is_set_label(label)]


def _labels_of_calcs() -> Dict[str, List[str]]:
    """CalcJob uuid -> the set labels (``friction2d/`` stripped) holding it."""
    qb = orm.QueryBuilder().append(orm.Group, tag='g', filters=SET_GROUP_FILTER,
                                   project='label')
    qb.append(orm.CalcJobNode, with_group='g', project='uuid')
    out: Dict[str, Set[str]] = {}
    for label, uuid in qb.all():
        if is_set_label(label):
            out.setdefault(str(uuid), set()).add(label[len(GROUP_PREFIX):])
    return {uuid: sorted(labels) for uuid, labels in out.items()}


def _deck_points(calc_project: List[str], point_project: List[str]):
    """A query of every friction2d CalcJob (tag ``c``) and its point FolderData
    (tag ``p``), projecting the given columns."""
    qb = orm.QueryBuilder()
    qb.append(orm.CalcJobNode, tag='c', filters={'process_type': _process_type()},
              project=calc_project)
    return _with_points(qb, point_project)


def _with_points(qb, project: List[str]):
    """Append the point FolderData of the CalcJob tagged ``c``, as tag ``p``."""
    return qb.append(orm.FolderData, tag='p', with_incoming='c',
                     edge_filters={'label': {'like': _POINTS_LINK}}, project=project)


# ---------------------------------------------------------------------------
# Points with their identity: one projection
# ---------------------------------------------------------------------------

@dataclass
class _Points:
    """Live points (rows) and their nodes, from one projection."""
    rows: List[Dict[str, Any]]
    nodes: Dict[str, Any]


def _points(include_superseded: bool = False) -> _Points:
    """Every point of a friction2d CalcJob with its identity columns."""
    from .block_schema import POINT_UNITS  # pylint: disable=import-outside-toplevel
    from .store import RUN_UUID_EXTRA  # pylint: disable=import-outside-toplevel

    qb = _deck_points(['uuid', 'extras'], ['*'])
    labels = _labels_of_calcs()
    rows: List[Dict[str, Any]] = []
    nodes: Dict[str, Any] = {}
    for calc_uuid, calc_extras, point in qb.all():
        calc_extras = calc_extras or {}
        superseded = bool(calc_extras.get('superseded')) or is_superseded(point)
        if superseded and not include_superseded:
            continue
        attrs = point.base.attributes.all
        units = attrs.get('units') or POINT_UNITS.get(attrs.get('load_kind'), {})
        sets = labels.get(str(calc_uuid), [])
        row: Dict[str, Any] = {'set_label': sets[0] if sets else None, 'set_labels': sets}
        tokens = _tokens(calc_extras, attrs.get('rel_path'), attrs.get('family'))
        row.update({name: tokens.get(name) for name in _TOKEN_FIELDS})
        row.update({('point_status' if name == 'status' else name): attrs.get(name)
                    for name in _POINT_FIELDS})
        if row['layers'] is None:
            row['layers'] = tokens.get('layers')
        for axis in AXES:
            row[f'{axis}_unit'] = units.get(axis)
        row.update({'run_uuid': calc_extras.get(RUN_UUID_EXTRA), 'calc_uuid': str(calc_uuid),
                    'node_uuid': point.uuid, 'superseded': superseded})
        rows.append(row)
        nodes[point.uuid] = point
    return _Points(rows, nodes)


def _in_range(value: Any, span: Range) -> bool:
    if value is None:
        return False
    lo, hi = span
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def _matches(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    """Whether one ``query`` row passes every given filter (None: not given)."""
    # pylint: disable=too-many-return-statements
    materials = filters.get('materials')
    if materials and not set(materials) <= set(row.get('materials') or ()):
        return False
    for name, column in _EQ_FILTERS.items():
        wanted = filters.get(name)
        if wanted is not None and row.get(column) != wanted:
            return False
    if filters.get('set_label') is not None and \
            filters['set_label'] not in row.get('set_labels', ()):
        return False
    layers = filters.get('layers')
    if layers is not None:
        allowed = [layers] if isinstance(layers, int) else list(layers)
        if row.get('layers') not in allowed:
            return False
    if filters.get('temperature_range') is not None and \
            not _in_range(row.get('temperature'), filters['temperature_range']):
        return False
    for kind in ('force', 'pressure'):
        span = filters.get(f'{kind}_range')
        if span is not None and not (row.get('load_kind') == kind
                                     and _in_range(row.get('load'), span)):
            return False
    for axis in ('angle', 'speed'):
        span = filters.get(f'{axis}_range')
        if span is not None and not _in_range(row.get(axis), span):
            return False
    return True


# ---------------------------------------------------------------------------
# Verdicts and summaries, in bulk
# ---------------------------------------------------------------------------

def _summary_digest(options: Optional[Mapping[str, Any]]) -> str:
    # pylint: disable=import-outside-toplevel
    from .steps import normalise_options, options_hash, summary_options
    return options_hash(normalise_options(dict(options) if options is not None
                                          else summary_options()))


def _selected(points: List[Any], policy: Policy, digest: str
              ) -> Dict[int, Tuple[Optional[Any], Optional[Any]]]:
    """point pk -> (selected verdict, selected summary), a few queries per
    batch of points (:func:`frictionsim2d.aiida.steps.latest_verdicts` and
    :func:`~frictionsim2d.aiida.steps.latest_summaries`)."""
    from .steps import latest_summaries, latest_verdicts  # pylint: disable=import-outside-toplevel
    out: Dict[int, Tuple[Optional[Any], Optional[Any]]] = {}
    for batch in _chunks(points):
        verdicts = latest_verdicts(batch, policy)
        summaries = latest_summaries(batch, policy, digest=digest)
        out.update({p.pk: (verdicts[p.pk], summaries[p.pk]) for p in batch})
    return out


def _verdict_columns(verdict: Optional[Any]) -> Dict[str, Any]:
    from .steps import KIND_EXTRA, is_provisional  # pylint: disable=import-outside-toplevel
    if verdict is None:
        return {'status': None, 'reasons': None, 'verdict_uuid': None,
                'verdict_kind': None, 'thresholds_version': None, 'provisional': None}
    return {'status': verdict.get('status'),
            'reasons': '; '.join(verdict.get('reasons') or []),
            'verdict_uuid': verdict.uuid,
            'verdict_kind': verdict.base.extras.get(KIND_EXTRA, 'verdict'),
            'thresholds_version': verdict.get('thresholds_version'),
            'provisional': is_provisional(verdict)}


def _summary_columns(summary: Optional[Any], taken: Iterable[str]) -> Dict[str, Any]:
    if summary is None:
        return {'summary_uuid': None}
    taken = set(taken)
    out: Dict[str, Any] = {'summary_uuid': summary.uuid,
                           'summary_code_version': summary.get('code_version')}
    for key, value in summary.get_dict().items():
        if key in _SUMMARY_META or isinstance(value, (dict, list)):
            continue
        out[f'summary_{key}' if key in taken else key] = value
    return out


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """The points a query matched: one row per point.

    Attributes:
        rows: One dict per point (see :meth:`Friction2DDB.query`).
        total_count: Matching points before ``limit``.
        query_params: The arguments the query was called with.
    """
    rows: List[Dict[str, Any]]
    total_count: int
    query_params: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self):
        """The rows as a ``pandas.DataFrame``."""
        import pandas as pd  # pylint: disable=import-outside-toplevel
        return pd.DataFrame(self.rows)

    def export_json(self, filepath: Path) -> Path:
        """The query parameters, the count and the rows as JSON."""
        filepath = Path(filepath)
        filepath.write_text(json.dumps({
            'query_params': self.query_params, 'total_count': self.total_count,
            'rows': self.rows}, indent=2, default=str), encoding='utf-8')
        return filepath


def _sort_key(column: str):
    def key(row: Mapping[str, Any]):
        value = row.get(column)
        return (value is None, value if value is not None else 0)
    return key


def _default_order(row: Mapping[str, Any]):
    return tuple(_sort_key(c)(row) for c in ('rel_path', 'load', 'angle', 'speed',
                                             'point_uuid'))


# ---------------------------------------------------------------------------
# The point table: block fields, flat
# ---------------------------------------------------------------------------

def _block_links(include_superseded: bool):
    """Query 1: (point identity, block link label, block hash, schema hash)
    rows, one per point and block (or schema) link of its CalcJob."""
    from .store import BLOCK_HASH_EXTRA, RUN_UUID_EXTRA, SCHEMA_HASH_EXTRA  # pylint: disable=import-outside-toplevel
    point_attrs = ('point_uuid', 'family', 'load_kind', 'load', 'angle', 'speed',
                   'rel_path', 'deck_name', 'units')
    qb = _deck_points(['uuid', 'extras.superseded', f'extras.{RUN_UUID_EXTRA}'],
                      ['uuid', 'extras.superseded']
                      + [f'attributes.{name}' for name in point_attrs])
    qb.append(orm.Dict, tag='b', with_outgoing='c', edge_tag='link',
              edge_filters={'or': [{'label': {'like': f'{_BLOCKS_LINK}%'}},
                                   {'label': _SCHEMA_LINK}]},
              edge_project=['label'],
              project=[f'extras.{BLOCK_HASH_EXTRA}', f'extras.{SCHEMA_HASH_EXTRA}'])
    points: Dict[str, Dict[str, Any]] = {}
    for row in qb.all():
        calc_uuid, calc_dead, run_uuid, node_uuid, point_dead = row[:5]
        attrs = dict(zip(point_attrs, row[5:5 + len(point_attrs)]))
        block, schema, label = row[5 + len(point_attrs):]
        superseded = bool(calc_dead) or bool(point_dead)
        if superseded and not include_superseded:
            continue
        units = attrs.pop('units') or {}
        entry = points.setdefault(str(node_uuid), {
            'meta': {**attrs, **{f'{axis}_unit': units.get(axis) for axis in AXES},
                     'node_uuid': str(node_uuid), 'calc_uuid': str(calc_uuid),
                     'run_uuid': run_uuid, 'superseded': superseded},
            'blocks': [], 'schema': None})
        if label == _SCHEMA_LINK:
            entry['schema'] = schema
        else:
            entry['blocks'].append((label[len(_BLOCKS_LINK):], block))
    return points


def _contents(block_hashes: Set[str], schema_hashes: Set[str]
              ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Query 2: the block and schema contents, by hash (one query per
    :data:`_CHUNK` hashes; twins of one hash hold one content)."""
    from .store import BLOCK_HASH_EXTRA, SCHEMA_HASH_EXTRA  # pylint: disable=import-outside-toplevel
    wanted = [(BLOCK_HASH_EXTRA, h) for h in sorted(block_hashes)] + \
             [(SCHEMA_HASH_EXTRA, h) for h in sorted(schema_hashes)]
    blocks: Dict[str, dict] = {}
    schemas: Dict[str, dict] = {}
    for batch in _chunks(wanted):
        ors = []
        for extra in (BLOCK_HASH_EXTRA, SCHEMA_HASH_EXTRA):
            values = [h for e, h in batch if e == extra]
            if values:
                ors.append({f'extras.{extra}': {'in': values}})
        qb = orm.QueryBuilder().append(
            orm.Dict, filters={'or': ors},
            project=[f'extras.{BLOCK_HASH_EXTRA}', f'extras.{SCHEMA_HASH_EXTRA}',
                     'attributes'])
        for block, schema, content in qb.all():
            if block is not None:
                blocks.setdefault(block, content)
            if schema is not None:
                schemas.setdefault(schema, content)
    return blocks, schemas


class _Refiller:  # pylint: disable=too-few-public-methods
    """Flat ``<block name>.<path>`` columns of one block, refilled from its
    run's schema defaults (:func:`frictionsim2d.core.compare.refill`), memoised."""

    def __init__(self, schemas: Mapping[str, dict]) -> None:
        self.tables = {h: schema_tables(doc) for h, doc in schemas.items()}
        self._memo: Dict[Tuple[str, str, Optional[str]], Dict[str, Any]] = {}

    def columns(self, label: str, digest: str, content: Mapping[str, Any],
                schema: Optional[str]) -> Dict[str, Any]:
        """The columns of block ``label`` (hash ``digest``) under ``schema``."""
        key = (label, digest, schema)
        if key not in self._memo:
            name = block_name(label, str(content.get('kind')))
            _units, defaults, kinds = self.tables.get(schema, ({}, {}, {}))
            fields = refill(name, content.get('fields') or {}, defaults, kinds)
            self._memo[key] = {f'{name}.{path}': value for path, value in fields.items()}
        return self._memo[key]


def point_table(*, include_superseded: bool = False):
    """One row per point, one column per requested field.

    Columns: ``<block name>.<field path>`` (``integration.settings.simulation.
    timestep``, ``material__sheet_0.name``, ``realization.seed_thermal_tip``),
    each elided field refilled from the default of the run's own stored
    schema; the swept axes ``load``, ``angle``, ``speed``; and ``point_uuid``,
    ``node_uuid``, ``calc_uuid``, ``run_uuid``, ``family``, ``load_kind``,
    ``load_unit``, ``angle_unit``, ``speed_unit``, ``rel_path``, ``deck_name``,
    ``superseded``, ``schema_hash``.
    ``table.attrs['units']`` maps each schema hash to its ``{schema path:
    unit}``.

    Built from one projection of (point, CalcJob, block link label, block
    hash) and one fetch of the distinct block contents by hash, so the cost
    does not grow with a query per point or per block. The
    points of a CalcJob whose blocks carry no hash extra, or that has no
    schema to refill from, are left out, with one warning per run
    (``FrictionSim2D aiida reindex`` restores the extras).
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    _require_aiida()
    points = _block_links(include_superseded)
    block_hashes = {h for e in points.values() for _l, h in e['blocks'] if h}
    schema_hashes = {e['schema'] for e in points.values() if e['schema']}
    blocks, schemas = (_contents(block_hashes, schema_hashes) if points else ({}, {}))
    refiller = _Refiller(schemas)
    rows = []
    left_out: Dict[Tuple[Optional[str], str], Set[str]] = {}
    for entry in points.values():
        meta = entry['meta']
        reasons = {f'block {label} has no stored content by hash'
                   for label, h in entry['blocks'] if h not in blocks}
        if entry['schema'] not in schemas:
            reasons.add('no schema to refill defaults from')
        if reasons:
            left_out.setdefault((meta['run_uuid'], meta['rel_path']), set()).update(reasons)
            continue
        row = {**meta, 'schema_hash': entry['schema']}
        for label, digest in entry['blocks']:
            row.update(refiller.columns(label, digest, blocks[digest], entry['schema']))
        rows.append(row)
    for (run_uuid, rel_path), reasons in sorted(left_out.items(), key=str):
        logger.warning('run %s (%s): points left out of the point table: %s; run '
                       '`FrictionSim2D aiida reindex`', run_uuid, rel_path,
                       '; '.join(sorted(reasons)))
    rows.sort(key=_default_order)
    # object dtype: a column with an absent cell must not turn its ints into
    # floats (a float is keyed to rtol, an int exactly).
    table = pd.DataFrame(rows, dtype=object)
    table.attrs['units'] = {h: tables[0] for h, tables in refiller.tables.items()}
    return table


# ---------------------------------------------------------------------------
# related()
# ---------------------------------------------------------------------------

def _is_field_column(column: str) -> bool:
    return '.' in column


def _column_kind(column: str) -> str:
    return block_kind(column.split('.', 1)[0])


def _column_path(column: str) -> str:
    """``integration.settings.simulation.timestep`` -> ``settings.simulation.timestep``."""
    return column.split('.', 1)[1]


#: Non-field columns of :func:`point_table`: identity and bookkeeping, never
#: something ``related`` can vary.
_META_COLUMNS = ('point_uuid', 'node_uuid', 'calc_uuid', 'run_uuid', 'family',
                 'load_kind', 'rel_path', 'deck_name', 'superseded', 'schema_hash',
                 'load_unit', 'angle_unit', 'speed_unit')

#: Axis units, compared whether or not the axis varies: a load in nN and one
#: in GPa are not two values of one quantity.
_UNIT_COLUMNS = tuple(f'{axis}_unit' for axis in AXES)


def _vary_columns(vary: Sequence[str], columns: Sequence[str]) -> Set[str]:
    """The table columns ``vary`` names: a swept axis, a block kind
    (``realization`` among them), a block name, or a field path (bare, or
    with its block). ``ValueError`` for anything else, the identity and
    bookkeeping columns (``family``, ``rel_path``, ...) included."""
    fields = [c for c in columns if _is_field_column(c)]
    out: Set[str] = set()
    for name in vary:
        if name in AXES:
            out.add(name)
        elif name in BLOCK_KINDS:
            out |= {c for c in fields if _column_kind(c) == name}
        elif name in _META_COLUMNS:
            raise ValueError(f'cannot vary {name!r}: it is point identity, not a '
                             f'parameter; vary a swept axis {AXES}, a block kind, a '
                             f'block name or a field path')
        elif NAME_SEP in name and '.' not in name:
            block_kind(name)
            out |= {c for c in fields if c.split('.', 1)[0] == name}
        elif name in fields:
            out.add(name)
        else:
            found = {c for c in fields
                     if _column_path(c) == name or unindexed(_column_path(c)) == name}
            if not found and unindexed(name) not in FIELD_MAP \
                    and name not in REALIZATION_FIELDS:
                raise ValueError(f'cannot vary {name!r}: not a swept axis {AXES}, a block '
                                 f'kind {BLOCK_KINDS}, a block name or a field path')
            out |= found
    return out


_ABSENT = '\0absent'


def _absent(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _key(value: Any) -> str:
    """One canonical string per value (:func:`frictionsim2d.core.compare.canonical_text`:
    floats to the significant figures ``compare`` resolves, ints exact),
    absent alike. numpy scalars (pandas hands them back) count as their
    Python value."""
    if hasattr(value, 'item') and not isinstance(value, (list, dict)):
        value = value.item()
    if _absent(value):
        return _ABSENT
    return canonical_text(value)


def _axis_key(value: Any) -> str:
    if _absent(value):
        return _ABSENT
    return repr(round(float(value), _AXIS_DECIMALS))


def _reference_row(table, point):
    """The table row of ``point`` (a point node, a ``point_uuid`` or a node uuid)."""
    if isinstance(point, str):
        rows = table[table['point_uuid'] == point]
        if rows.empty:
            rows = table[table['node_uuid'] == point]
    else:
        rows = table[table['node_uuid'] == point.uuid]
    if rows.empty:
        raise LookupError(f'{point!r} is not a current point of this profile')
    if len(rows) > 1:
        raise LookupError(f'point uuid {point!r} names {len(rows)} current nodes; '
                          'pass the point node')
    return rows.iloc[0]


def _narrow(table, mask, columns: Sequence[str], reference):
    """``mask`` narrowed, column by column, to the rows equal to
    ``reference`` there: each column is keyed on the surviving rows only."""
    mask = mask.copy()
    for column in columns:
        if not mask.any():
            break
        if column not in table.columns:
            continue
        to_key = _axis_key if column in AXES else _key
        rows = mask[mask].index
        mask.loc[rows] = (table.loc[rows, column].map(to_key)
                          == to_key(reference.get(column))).to_numpy()
    return mask


def _unit_clash(table, mask, compared: Sequence[str], reference):
    """The rows of ``mask`` whose schema states another unit than the
    reference's for a compared field the reference carries (rows that
    survived :func:`_narrow` carry it too); a field both leave absent never
    clashes. Those values cannot be equal."""
    units = table.attrs.get('units', {})
    ref_schema = reference.get('schema_hash')
    ref = units.get(ref_schema, {})
    paths = {unindexed(_column_path(c)) for c in compared
             if _is_field_column(c) and not _absent(reference.get(c))}
    schemas = set(table.loc[mask, 'schema_hash']) - {ref_schema}
    clashing = [schema for schema in schemas
                if any(ref.get(p) != units.get(schema, {}).get(p) for p in paths)]
    return mask & table['schema_hash'].isin(clashing)


def related(point: Union[str, Any], vary: Union[str, Sequence[str]], *,
            table=None) -> List[Any]:
    """The current points that differ from ``point`` only along ``vary``.

    A point relates when it equals ``point`` on every requested field column
    of :func:`point_table` outside ``vary`` (elided fields refilled from each
    run's own schema; reals equal to the significant figures
    :func:`frictionsim2d.core.compare.real_key` resolves, as ``compare`` judges them;
    a field stored in another unit is never equal), on its family, on every
    swept axis outside ``vary``, and on the axes' units. Deck shape is not
    identity, so an array-mode point and a combined-mode point of the same
    physics relate; groups do not matter either.

    ``vary`` is one name or several: a block kind (``'thermostat'``), a block
    name (``'material__sheet_0'``), a dotted field path
    (``'settings.simulation.timestep'``, or with its block,
    ``'integration.settings.simulation.timestep'``), a swept axis
    (``'load'``, ``'angle'``, ``'speed'``), or ``'realization'``.

    The ``realization`` block (seeds) is ignored: seeds differ between every
    pair of runs, so comparing them would relate nothing. A *replicate* -- a
    point equal to ``point`` on every field but the seeds, ``vary`` included
    -- is returned only when ``'realization'`` is in ``vary``.

    Args:
        point: The point ``FolderData``, or its ``point_uuid``.
        vary: What may differ.
        table: A :func:`point_table` to reuse (several ``related`` calls over
            one table cost one projection).

    Returns:
        The related point nodes (``point`` and its own ``point_uuid`` never),
        ordered by ``rel_path``, ``load``, ``angle``, ``speed``.

    Raises:
        LookupError: ``point`` is not a current point.
        ValueError: ``vary`` names nothing ``related`` can vary.
    """
    _require_aiida()
    table = point_table() if table is None else table
    reference = _reference_row(table, point)
    names = [vary] if isinstance(vary, str) else list(vary)
    varied = _vary_columns(names, list(table.columns))
    fields = [c for c in table.columns if _is_field_column(c)]
    realization = {c for c in fields if _column_kind(c) == REALIZATION}
    # Cheap, selective columns first: each later one is keyed on fewer rows.
    identity = list(AXES) + ['family', 'load_kind'] + list(_UNIT_COLUMNS) + fields
    if 'load' in varied:
        identity.remove('load_kind')
    compared = [c for c in identity if c not in varied and c not in realization]
    free = [c for c in identity if c in varied and c not in realization]

    start = (table['point_uuid'] != reference['point_uuid']) & \
            (table['node_uuid'] != reference['node_uuid'])
    same = _narrow(table, start, compared, reference)
    keep = same & ~_unit_clash(table, same, compared, reference)
    if REALIZATION not in names:
        keep &= ~_narrow(table, keep, free, reference)
    return _nodes_by_uuid(list(table.loc[keep, 'node_uuid']))


def _nodes_by_uuid(uuids: Sequence[str]) -> List[Any]:
    """The nodes of ``uuids``, in that order, in one query per batch."""
    found: Dict[str, Any] = {}
    for batch in _chunks(list(uuids)):
        qb = orm.QueryBuilder().append(orm.Node, filters={'uuid': {'in': batch}},
                                       project=['*'])
        found.update({n.uuid: n for n in qb.all(flat=True)})
    return [found[u] for u in uuids if u in found]


# ---------------------------------------------------------------------------
# latest()
# ---------------------------------------------------------------------------

def _point_node(point: Union[str, Any]):
    if not isinstance(point, str):
        return point
    qb = orm.QueryBuilder().append(
        orm.FolderData, filters={'attributes.point_uuid': point}, project=['*'])
    found = [n for n in qb.all(flat=True) if not is_superseded(n)]
    if not found:
        found = _nodes_by_uuid([point])
    if len(found) != 1:
        raise LookupError(f'{point!r} names {len(found)} current points')
    return found[0]


def latest(point: Union[str, Any], kind: Literal['verdict', 'summary'],
           policy: Policy = Policy(), *,
           options: Optional[Mapping[str, Any]] = None) -> Optional[Any]:
    """The verdict or summary ``policy`` selects for ``point``, or None.

    ``kind='verdict'``: :func:`frictionsim2d.aiida.steps.latest_verdict` (generation
    first, then ruled > withdrawn > checker, then the uuid). ``kind=
    'summary'``: among the summaries made with ``options`` (default: the
    settings in effect, :func:`frictionsim2d.aiida.steps.summary_options`), the highest
    ``code_version``, then the uuid. Never by creation time.
    """
    from .steps import latest_summaries, latest_verdict  # pylint: disable=import-outside-toplevel
    if kind not in ('verdict', 'summary'):
        raise ValueError(f'unknown kind {kind!r} (verdict or summary)')
    _require_aiida()
    node = _point_node(point)
    if kind == 'verdict':
        return latest_verdict(node, policy)
    return latest_summaries([node], policy, digest=_summary_digest(options))[node.pk]


# ---------------------------------------------------------------------------
# dataset_rows()
# ---------------------------------------------------------------------------

#: Dataset row keys that are not a scalar of the joined row.
_DATASET_ONLY = ('block_hashes', 'family', 'point_uuid', 'calc_uuid', 'code_version',
                 'options_hash', 'skip_fraction')


def dataset_rows(name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    """``(rows, units)`` of the dataset ``friction2d/datasets/<name>``.

    One row per dataset row: its point's :meth:`Friction2DDB.query` identity
    columns (run tokens, ``rel_path``, ``family``, ``layers``, ``load``,
    ``angle``, ``speed`` and their units), the dataset row's own bookkeeping
    (``set``, ``summary_uuid``, ``verdict_uuid``, ``verdict_status``,
    ``thresholds_version``, ``provisional``) and every scalar of the summary
    the dataset fixed -- never the one the settings in effect would select
    now. ``units`` is the dataset's own table: family -> key -> unit.

    A point is found by its ``calc_uuid`` and ``point_uuid``, superseded or
    not: a dataset keeps the points it was built from. A row whose point or
    summary is gone is left out, with a warning.

    Raises:
        LookupError: no dataset of that name.
    """
    from .steps import DATASET_EXTRA  # pylint: disable=import-outside-toplevel
    _require_aiida()
    qb = orm.QueryBuilder().append(
        orm.Group, tag='g', filters={'label': f'{DATASET_PREFIX}{name}'})
    qb.append(orm.Dict, with_group='g', filters={f'extras.{DATASET_EXTRA}': name},
              project=['*'])
    tables = qb.all(flat=True)
    if not tables:
        raise LookupError(f'no dataset {name!r} ({DATASET_PREFIX}{name})')
    table = tables[0]
    entries = table.get('rows') or []
    points = {(r['calc_uuid'], r['point_uuid']): r
              for r in _points(include_superseded=True).rows}
    summaries = {n.uuid: n for n in _nodes_by_uuid([e['summary_uuid'] for e in entries])}
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        point = points.get((entry.get('calc_uuid'), entry.get('point_uuid')))
        summary = summaries.get(entry.get('summary_uuid'))
        if point is None or summary is None:
            logger.warning('dataset %r: point %s left out (its %s is gone)', name,
                           entry.get('point_uuid'), 'point' if point is None else 'summary')
            continue
        row = {k: v for k, v in point.items() if k not in ('superseded', 'set_labels')}
        row.update({k: v for k, v in entry.items() if k not in _DATASET_ONLY})
        row.update(_summary_columns(summary, row))
        rows.append(row)
    rows.sort(key=_default_order)
    return rows, dict(table.get('units') or {})


def point_frames(calc_uuid: str, node_uuid: str) -> Dict[str, Any]:
    """The stored tiers of one point as frames (``tier -> DataFrame``), read
    the way the DataReader and the store's summaries read them.

    Nothing is exported: the point's files go through a temporary directory
    and are gone when this returns.

    ``node_uuid`` is the point's FolderData (the ``node_uuid`` column of
    :func:`dataset_rows`), not its ``point_uuid`` identity.

    Raises:
        LookupError: no such point of that CalcJob.
    """
    from tempfile import TemporaryDirectory  # pylint: disable=import-outside-toplevel

    from .steps import point_frames as _stored_frames  # pylint: disable=import-outside-toplevel
    _require_aiida()
    nodes = {n.uuid: n for n in _nodes_by_uuid([calc_uuid, node_uuid])}
    calc, point = nodes.get(calc_uuid), nodes.get(node_uuid)
    if calc is None or point is None:
        raise LookupError(f'no point node {node_uuid} of CalcJob {calc_uuid}')
    family = point.base.attributes.get('family')
    n_layers = int(calc.inputs.parameters.get_dict()['n_layers'])
    with TemporaryDirectory() as tmp:
        return _stored_frames(point, Path(tmp) / 'results', family, n_layers)


# ---------------------------------------------------------------------------
# Friction2DDB
# ---------------------------------------------------------------------------

class Friction2DDB:
    """Query interface for the friction2d store (a loaded AiiDA profile).

    Example::

        db = Friction2DDB()
        rows = db.query(materials=['h-MoS2'], force_range=(5.0, 15.0)).to_dataframe()
        siblings = db.related(rows.point_uuid[0], 'settings.simulation.timestep')

    Raises:
        RuntimeError: If AiiDA is not available.
    """

    def __init__(self):
        _require_aiida()
        self._snapshot: Optional[_Points] = None

    @contextmanager
    def snapshot(self):
        """Within it, :meth:`query`, :meth:`get_available_conditions` and
        :meth:`get_available_materials` share one
        projection of the live points (one refresh of the GUI list costs one)."""
        self._snapshot = _points()
        try:
            yield self
        finally:
            self._snapshot = None

    def _live(self) -> _Points:
        return self._snapshot if self._snapshot is not None else _points()

    # -- sets ------------------------------------------------------------------

    def list_sets(self) -> List[Dict[str, Any]]:
        """One dict per set Group: ``label``, ``uuid``, ``pk``, the Group
        extras (``model``, ``materials``, ``created``, ``git_commit``,
        ``source_host``, ``source_root``), ``n_decks`` (live CalcJobs),
        ``n_runs`` (their distinct ``run_uuid``), ``n_points`` and
        ``n_results`` (live points). Sorted by label."""
        from .store import RUN_UUID_EXTRA  # pylint: disable=import-outside-toplevel
        decks: Dict[str, Dict[str, Optional[str]]] = {}
        qb = orm.QueryBuilder().append(
            orm.Group, tag='g', filters=SET_GROUP_FILTER, project='label')
        qb.append(orm.CalcJobNode, with_group='g', tag='c',
                  project=['uuid', 'extras.superseded', f'extras.{RUN_UUID_EXTRA}'])
        for label, uuid, dead, run_uuid in qb.all():
            if not dead:
                decks.setdefault(label, {})[str(uuid)] = run_uuid
        _with_points(qb, ['uuid', 'extras.superseded'])
        points: Dict[str, Set[str]] = {}
        for label, _uuid, calc_dead, _run, point_uuid, dead in qb.all():
            if not calc_dead and not dead:
                points.setdefault(label, set()).add(str(point_uuid))
        rows = []
        for label, uuid, pk, extras in _set_group_rows():
            live = decks.get(label, {})
            n_points = len(points.get(label, ()))
            rows.append({
                'label': label[len(GROUP_PREFIX):], 'uuid': uuid, 'pk': pk,
                **{key: extras.get(key) for key in ('model', 'materials', 'created',
                                                    'git_commit', 'source_host',
                                                    'source_root')},
                'n_decks': len(live), 'n_runs': len(set(live.values())),
                'n_points': n_points, 'n_results': n_points})
        return sorted(rows, key=lambda r: r['label'])

    # -- the query -------------------------------------------------------------

    def query(  # pylint: disable=too-many-arguments,too-many-locals,unused-argument
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
        temperature_range: Optional[Range] = None,
        force_range: Optional[Range] = None,
        pressure_range: Optional[Range] = None,
        angle_range: Optional[Range] = None,
        speed_range: Optional[Range] = None,
        limit: Optional[int] = None, order_by: Optional[str] = None,
        policy: Policy = Policy(),
        summary_options: Optional[Mapping[str, Any]] = None,
    ) -> QueryResult:
        """One row per live point matching every given filter.

        Every filter defaults to None, "not filtered". ``materials`` needs all
        of them in the run's materials; ``layers`` is a count or a list of
        counts; the ``*_range`` filters are ``(lo, hi)``, either side open
        (None). ``force_range``/``pressure_range`` also keep only points of
        that load kind. ``set_label`` names a ``friction2d/<label>`` Group;
        ``potential`` is the run's ``pot_token``.

        A row holds the run tokens (``model``, ``materials``, ``size_x``,
        ``temperature``, ``pot_token``, ...), ``set_label`` (the first set
        holding the deck) and ``set_labels``, the point's identity
        (``point_uuid``, ``node_uuid``, ``calc_uuid``, ``run_uuid``,
        ``rel_path``, ``deck_name``, ``family``, ``layers``), ``load``,
        ``angle``, ``speed`` with ``load_unit``, ``angle_unit``,
        ``speed_unit``, the parser's ``point_status``, the verdict ``policy``
        selects (``status``, ``reasons``, ``verdict_uuid``, ``verdict_kind``,
        ``thresholds_version``, ``provisional``) and the summary it selects
        among those made with ``summary_options`` (default: the settings in
        effect) -- ``summary_uuid`` and every summary scalar (``mean_cof``,
        ...). A point with no verdict or summary has None there.

        ``order_by`` names a column (``-`` prefix: descending; default
        ``rel_path``, ``load``, ``angle``, ``speed``); ``limit`` keeps the
        first rows. ``total_count`` counts the matches before ``limit``.
        Superseded points never match.
        """
        params = {k: v for k, v in locals().items()
                  if k not in ('self', 'policy', 'summary_options')}
        found = self._live()
        rows = [dict(row) for row in found.rows if _matches(row, params)]
        params.update(policy=policy.as_dict(), summary_options=summary_options)
        selected = _selected([found.nodes[r['node_uuid']] for r in rows], policy,
                             _summary_digest(summary_options)) if rows else {}
        for row in rows:
            verdict, summary = selected[found.nodes[row['node_uuid']].pk]
            row.update(_verdict_columns(verdict))
            row.update(_summary_columns(summary, row))
            row.pop('superseded', None)
        if order_by:
            column = order_by.lstrip('-')
            rows.sort(key=_sort_key(column), reverse=order_by.startswith('-'))
        else:
            rows.sort(key=_default_order)
        total = len(rows)
        if limit:
            rows = rows[:limit]
        return QueryResult(rows=rows, total_count=total, query_params=params)

    # -- parameters ------------------------------------------------------------

    @staticmethod
    def point_table(*, include_superseded: bool = False):
        """See :func:`point_table`."""
        return point_table(include_superseded=include_superseded)

    @staticmethod
    def related(point: Union[str, Any], vary: Union[str, Sequence[str]], *,
                table=None) -> List[Any]:
        """See :func:`related`."""
        return related(point, vary, table=table)

    @staticmethod
    def latest(point: Union[str, Any], kind: Literal['verdict', 'summary'],
               policy: Policy = Policy(), *,
               options: Optional[Mapping[str, Any]] = None) -> Optional[Any]:
        """See :func:`latest`."""
        return latest(point, kind, policy, options=options)

    # -- aggregation -----------------------------------------------------------

    def get_available_conditions(self) -> Dict[str, List[Any]]:
        """Distinct values behind each GUI filter dropdown, over live points:
        ``forces``, ``pressures``, ``angles``, ``speeds`` (a PES grid's
        placeholder zeros left out), ``temperatures``, ``layers``,
        ``families``, ``models``, ``potentials`` (``pot_token``) and
        ``set_labels`` (every set Group, empty ones included)."""
        values: Dict[str, Set[Any]] = {k: set() for k in (
            'forces', 'pressures', 'angles', 'speeds', 'temperatures', 'layers',
            'families', 'models', 'potentials')}
        for row in self._live().rows:
            kind = row.get('load_kind')
            if kind in ('force', 'pressure'):
                values['forces' if kind == 'force' else 'pressures'].add(row['load'])
                values['angles'].add(row['angle'])
                values['speeds'].add(row['speed'])
            for key, column in (('temperatures', 'temperature'), ('layers', 'layers'),
                                ('families', 'family'), ('models', 'model'),
                                ('potentials', 'pot_token')):
                if row.get(column) is not None:
                    values[key].add(row[column])
        out = {key: sorted(found) for key, found in values.items()}
        out['set_labels'] = sorted(label[len(GROUP_PREFIX):]
                                   for label, *_rest in _set_group_rows())
        return out

    def get_available_materials(self) -> List[str]:
        """Every material named by a live point's run, sorted."""
        return sorted({m for row in self._live().rows for m in (row.get('materials') or ())})


def list_simulation_sets(profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """:meth:`Friction2DDB.list_sets` on ``profile`` (loaded first; None: the
    configured one) -- one row per set, for ``aiida list-sets``."""
    _require_aiida()
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel
    load_aiida_profile(profile)
    return Friction2DDB().list_sets()
