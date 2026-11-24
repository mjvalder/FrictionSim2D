"""Analysis results in the local AiiDA store (the profile must be loaded by the caller).

A result is the ``SinglefileData`` (a CSV) that the calcfunction :func:`record_analysis`
returns, in the Group ``friction2d/analyses/<dataset>``, with extras ``analysis``,
``version``, ``params_hash``, ``dataset``, ``n_rows``, ``n_dropped`` and, when it read
other stored results, ``depends``. Its attributes carry the ``dropped``
``(point_uuid, reason)`` pairs, the ``notes`` and those ``depends``.
"""
from __future__ import annotations

import hashlib
import io
import json
from contextvars import ContextVar
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import pandas as pd
from pydantic import BaseModel

from .base import Analysis, check_identity, get_analysis

try:
    from aiida import orm
    from aiida.engine import calcfunction
except ImportError:  # pragma: no cover - the aiida tier skips without it
    orm = None  # type: ignore[assignment]

    def calcfunction(func):  # type: ignore[no-redef]
        """Stand-in so the module imports without aiida-core."""
        return func

__all__ = ['ANALYSIS_GROUP_PREFIX', 'find_result_node', 'params_hash', 'record_analysis',
           'result_node', 'select_result_node', 'store_result', 'table_from_csv']

#: The Group of a dataset's results is this plus the dataset name.
ANALYSIS_GROUP_PREFIX = 'friction2d/analyses/'
_CSV_NAME = 'table.csv'

#: The table and drops :func:`record_analysis` stores. Not calcfunction inputs:
#: the table is what the analysis computed from them, not something it read.
_PENDING: ContextVar[Tuple[bytes, List[List[str]], List[str], Optional[Dict[str, Any]]]] = \
    ContextVar('_PENDING')

AnalysisLike = Union[Type[Analysis], Analysis]


def params_hash(params: Mapping[str, Any]) -> str:
    """sha256 of the params as sorted-key JSON."""
    text = json.dumps(dict(params), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _float(value: float) -> str:
    """The shortest text that reads back as exactly ``value``."""
    return repr(float(value))


def _to_csv(table: pd.DataFrame) -> bytes:
    return table.to_csv(index=False, float_format=_float).encode('utf-8')


def table_from_csv(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), float_precision='round_trip')


@calcfunction
def record_analysis(dataset, params, **depends):  # pylint: disable=unused-argument
    """The stored table of one analysis of ``dataset`` (the dataset's table
    ``Dict``) under ``params`` (``analysis``, ``version``, ``params``), from
    the other stored results in ``depends``."""
    content, dropped, notes, meta = _PENDING.get()
    node = orm.SinglefileData(io.BytesIO(content), filename=_CSV_NAME)
    node.base.attributes.set('dropped', dropped)
    node.base.attributes.set('notes', notes)
    if meta is not None:
        node.base.attributes.set('depends', meta)
    return node


def _params_dict(params: Union[BaseModel, Mapping[str, Any], None]) -> Dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, BaseModel):
        return params.model_dump(mode='json')
    return dict(params)


def _depends_key(depends: Mapping[str, Any]) -> str:
    """The extra ``depends`` of a result that read ``depends`` (name ->
    node or None): the uuids as sorted-key JSON."""
    return json.dumps({name: None if node is None else node.uuid
                       for name, node in depends.items()}, sort_keys=True)


def _depends_meta(node: Any) -> Optional[Dict[str, Any]]:
    if node is None:
        return None
    return {'uuid': node.uuid, 'version': node.base.extras.get('version', None),
            'params_hash': node.base.extras.get('params_hash', None)}


def store_result(analysis: AnalysisLike, dataset: str,
                 params: Union[BaseModel, Mapping[str, Any], None],
                 table: pd.DataFrame, *,
                 dropped: Sequence[Tuple[str, str]],
                 notes: Sequence[str] = (),
                 depends: Optional[Mapping[str, Any]] = None) -> str:
    """Store ``table`` as the result of ``analysis`` over ``dataset`` under
    ``params``, with its ``dropped`` pairs and ``notes``, read from the stored
    results ``depends`` (name -> node or None); return the result node's uuid.

    Raises:
        LookupError: no dataset of that name.
    """
    # pylint: disable=import-outside-toplevel
    from .source import dataset_table
    values = _params_dict(analysis.Params(**_params_dict(params)))
    digest = params_hash(values)
    pairs = [[str(uuid), str(reason)] for uuid, reason in dropped]
    meta = ({name: _depends_meta(n) for name, n in depends.items()} if depends else None)
    token = _PENDING.set((_to_csv(table), pairs, [str(n) for n in notes], meta))
    try:
        node = record_analysis(
            dataset_table(dataset),
            orm.Dict({'analysis': analysis.name, 'version': analysis.version,
                      'params': values}),
            # Never a cache hit: the table is not a function of the inputs.
            metadata={'label': f'{analysis.name}:{dataset}', 'disable_cache': True},
            **{name: n for name, n in (depends or {}).items() if n is not None})
    finally:
        _PENDING.reset(token)
    extras = {'analysis': analysis.name, 'version': analysis.version, 'params_hash': digest,
              'dataset': dataset, 'n_rows': int(len(table)), 'n_dropped': len(pairs)}
    if depends:
        extras['depends'] = _depends_key(depends)
    node.base.extras.set_many(extras)
    group, _ = orm.Group.collection.get_or_create(f'{ANALYSIS_GROUP_PREFIX}{dataset}')
    group.add_nodes([node])
    return node.uuid


def _current_version(name: str, version: Optional[str] = None) -> Optional[str]:
    """``version``, else the registered analysis' current one (None: unregistered)."""
    if version is not None:
        return version
    try:
        return get_analysis(name).version
    except KeyError:
        return None


def _results(name: str, dataset: str, **extras: Any):
    """The query of the result nodes of ``name`` over ``dataset``, narrowed by
    the given extras (those that are not None)."""
    filters: Dict[str, Any] = {'extras.analysis': name, 'extras.dataset': dataset}
    filters.update({f'extras.{key}': value for key, value in extras.items()
                    if value is not None})
    qb = orm.QueryBuilder().append(
        orm.Group, tag='g', filters={'label': f'{ANALYSIS_GROUP_PREFIX}{dataset}'})
    qb.append(orm.SinglefileData, with_group='g', filters=filters, project=['*'])
    return qb


def _find_node(name: str, dataset: str, digest: Optional[str], version: Optional[str],
               depends: Optional[str] = None):
    qb = _results(name, dataset, depends=depends, params_hash=digest, version=version)
    qb.order_by({orm.SinglefileData: {'id': 'desc'}})
    return qb.first(flat=True)


def find_result_node(name: str, dataset: str, params_hash: Optional[str] = None,  # pylint: disable=redefined-outer-name
                     version: Optional[str] = None):
    """The newest stored result node (its ``uuid`` and extras carry the
    metadata) of analysis ``name`` over ``dataset`` (with that ``params_hash``,
    when given), or None. ``version`` defaults to the registered analysis'
    current one; an unregistered name needs it given."""
    return _find_node(name, dataset, params_hash, _current_version(name, version))


def _stored_params(name: str, dataset: str, version: Optional[str] = None
                  ) -> Dict[str, Dict[str, Any]]:
    """``params_hash -> params`` of every stored result of ``name`` over
    ``dataset`` (of ``version``, when given): the params its
    ``record_analysis`` read."""
    out: Dict[str, Dict[str, Any]] = {}
    for node in _results(name, dataset, version=version).all(flat=True):
        digest = node.base.extras.get('params_hash', None)
        if digest not in out:
            out[digest] = node.creator.inputs.params.get_dict().get('params', {})
    return out


def _normalised_params(name: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """``params`` as :func:`result_node` hashes them: validated by the
    registered analysis' ``Params`` model, defaults filled in. An
    unregistered name keeps them as given.

    Raises:
        ValueError: a name the analysis has no parameter of, or a bad value.
    """
    params = dict(params or {})
    try:
        cls = get_analysis(name)
    except KeyError:
        return params
    unknown = sorted(set(params) - set(cls.Params.model_fields))
    if unknown:
        raise ValueError(f'{name} has no parameter {", ".join(unknown)} (known: '
                         f'{", ".join(sorted(cls.Params.model_fields)) or "none"})')
    return _params_dict(cls.Params(**params))


def select_result_node(name: str, dataset: str, params: Optional[Mapping[str, Any]] = None,
                       version: Optional[str] = None):
    """The newest stored result of ``name`` over ``dataset`` under exactly
    ``params`` (none: the defaults), of ``version`` (default: the registered
    one), or None when nothing of that analysis is stored for the dataset.

    Raises:
        ValueError: an unknown parameter or a bad value.
        LookupError: none under these params, but results under others are
            stored; the message names them (the parameters that differ).
    """
    values = _normalised_params(name, params)
    version = _current_version(name, version)
    node = _find_node(name, dataset, params_hash(values), version)
    if node is not None:
        return node
    others = _stored_params(name, dataset, version)
    if not others:
        return None
    sets = ['{' + ', '.join(f'{k}={v!r}' for k, v in sorted(p.items())
                             if values.get(k, object()) != v) + '}'
            for p in others.values()]
    asked = ', '.join(f'{k}={v!r}' for k, v in sorted(dict(params or {}).items())) or 'defaults'
    raise LookupError(f'no stored {name} over {dataset} under {asked}; stored under: '
                      f'{"; ".join(sorted(sets))} -- name those params, or run: '
                      f'FrictionSim2D analyse run {name} {dataset}')


def result_node(name: str, dataset: str, params: Optional[dict] = None, *,
                force: bool = False):
    """The stored result node of analysis ``name`` over ``dataset``: the one
    with the same name, version, params, dataset and read results
    (:meth:`~frictionsim2d.analysis.base.Analysis.depends_on`), else (or with
    ``force``) computed from the dataset and stored.

    Raises:
        KeyError: no analysis of that name.
        ValueError: the dataset is not a family the analysis reads.
    """
    # pylint: disable=import-outside-toplevel
    from .source import RunSource
    cls = get_analysis(name)
    model = cls.Params(**(params or {}))
    values = _params_dict(model)
    digest = params_hash(values)
    depends = cls.depends_on(dataset, model)
    if not force:
        node = _find_node(name, dataset, digest, cls.version,
                          _depends_key(depends) if depends else None)
        if node is not None:
            return node
    with RunSource.from_dataset(dataset) as source:
        if source.family not in cls.families:
            raise ValueError(f'{name} reads families {sorted(cls.families)}, '
                             f'not {source.family!r} (dataset {dataset!r})')
        source.depends = depends
        table = cls().compute(source, model)
        check_identity(cls, table)
        dropped = source.dropped
        notes = list(getattr(source, 'notes', []))
    uuid = store_result(cls, dataset, values, table, dropped=dropped, notes=notes,
                        depends=depends)
    return orm.load_node(uuid)
