"""Loading parameter blocks for :mod:`src.core.compare`.

A reference is one of: a CalcJob pk or uuid, a ``point_uuid``, a group label
(with or without the ``friction2d/`` prefix), or a ``<archive>.params/``
sidecar directory. A group gives the fields shared by every deck in it; a field
that varies is :data:`src.core.compare.VARIES`.

Units and defaults come from the CalcJob's ``schema`` input (or the sidecar's
``schema-<version>.json``), so a comparison reads the units the run was stored
with, not today's.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..core.compare import DEFAULT_ATOL, DEFAULT_RTOL, merge_varying
from .data_keys import is_superseded
from .block_schema import version_key

__all__ = ['LoadedBlocks', 'load_blocks', 'load_blocks_from_sidecar']

_NAMED_PER_ENTRY = ('material', 'potential')


@dataclass
class LoadedBlocks:
    """Blocks of one run or campaign, ready for ``compare_blocks``.

    Attributes:
        blocks: Block name (``thermostat``, ``potential__sheet_0``) -> flat fields.
        derived: Derived values, flat (``geometry.lat_c``).
        units: Schema path -> unit.
        defaults: Schema path -> default.
        kinds: Schema path -> block kind.
    """
    blocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    derived: Dict[str, Any] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    kinds: Dict[str, str] = field(default_factory=dict)


def _from_schema(document: Mapping[str, Any]):
    fields = document.get('fields') or {}
    units = {p: f['unit'] for p, f in fields.items() if f.get('unit')}
    defaults = {p: f['default'] for p, f in fields.items() if f.get('default') is not None}
    kinds = {p: f['kind'] for p, f in fields.items() if f.get('kind')}
    return units, defaults, kinds


def _block_name(label: str, kind: str) -> str:
    """``material_sheet_0`` (a link label) -> ``material__sheet_0``."""
    if kind in _NAMED_PER_ENTRY and label.startswith(f'{kind}_'):
        return f'{kind}__{label[len(kind) + 1:]}'
    return label


def _loaded_from_calcjob(node) -> LoadedBlocks:
    inputs = node.inputs
    if 'blocks' not in inputs:
        raise ValueError(f'CalcJob {node.pk} has no parameter blocks')
    blocks = {}
    for label, block in inputs.blocks.items():
        content = block.get_dict()
        blocks[_block_name(label, str(content.get('kind')))] = dict(content['fields'])
    derived = dict(inputs.derived.get_dict().get('fields') or {}) \
        if 'derived' in inputs else {}
    units, defaults, kinds = (_from_schema(inputs.schema.get_dict()) if 'schema' in inputs
                              else ({}, {}, {}))
    return LoadedBlocks(blocks, derived, units, defaults, kinds)


def _merge(loaded: List[LoadedBlocks], *, rtol: float = DEFAULT_RTOL,
           atol: float = DEFAULT_ATOL) -> LoadedBlocks:
    """The fields every member shares. Each member is read against its own
    defaults; members that store a field in different units (a different
    schema version) cannot be merged and raise ``ValueError``."""
    units = dict(loaded[0].units)
    for item in loaded[1:]:
        clash = sorted(p for p in set(units) | set(item.units)
                       if units.get(p) != item.units.get(p))
        if clash:
            raise ValueError('group mixes schema versions with different units for '
                             + ', '.join(clash[:5]))
        units.update(item.units)
    kinds: Dict[str, str] = {}
    for item in loaded:
        kinds.update(item.kinds)
    blocks, derived = merge_varying([(i.blocks, i.derived, i.defaults) for i in loaded],
                                    rtol=rtol, atol=atol, kinds=kinds)
    same = all(i.defaults == loaded[0].defaults for i in loaded)
    return LoadedBlocks(blocks, derived, units, dict(loaded[0].defaults) if same else {},
                        kinds)


def _calcjob_of(node):
    from aiida import orm  # pylint: disable=import-outside-toplevel

    if isinstance(node, orm.CalcJobNode):
        return node
    creator = getattr(node, 'creator', None)
    return creator if isinstance(creator, orm.CalcJobNode) else None


def _node_for(ref: str, kinds=('pk', 'uuid', 'point')):
    """The node a pk, uuid or ``point_uuid`` names, or None.

    A ``point_uuid`` can sit on a superseded node and its replacement; the
    current one is taken, and two current ones raise ``LookupError``.
    """
    from aiida import orm  # pylint: disable=import-outside-toplevel
    from aiida.common.exceptions import (  # pylint: disable=import-outside-toplevel
        MultipleObjectsError, NotExistent)

    if 'pk' in kinds and ref.isdigit():
        try:
            return orm.load_node(int(ref))
        except NotExistent:
            pass
    if 'uuid' in kinds and not ref.isdigit():
        try:
            return orm.load_node(ref)
        except (NotExistent, MultipleObjectsError, ValueError):
            pass
    if 'point' in kinds:
        found = [n for (n,) in orm.QueryBuilder().append(
            orm.Node, filters={'attributes.point_uuid': ref}, project=['*']).all()
            if not is_superseded(n)]
        if len(found) > 1:
            raise LookupError(f'point uuid {ref!r} names {len(found)} current nodes')
        return found[0] if found else None
    return None


def _group_for(ref: str):
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .ingest import group_label  # pylint: disable=import-outside-toplevel

    for label in (ref, group_label(ref)):
        group = orm.Group.collection.find(filters={'label': label})
        if group:
            return group[0]
    return None


_PREFIXES = ('pk', 'uuid', 'point', 'group')


def _from_group(group, rtol: float, atol: float) -> LoadedBlocks:
    jobs = {}
    for member in group.nodes:
        job = _calcjob_of(member)
        if (job is not None and 'blocks' in job.inputs and not is_superseded(job)
                and not is_superseded(member)):
            jobs[job.pk] = job
    if not jobs:
        raise LookupError(f'group {group.label!r} holds no current CalcJob with '
                          'parameter blocks')
    return _merge([_loaded_from_calcjob(j) for _, j in sorted(jobs.items())],
                  rtol=rtol, atol=atol)


def load_blocks(ref: str, *, rtol: float = DEFAULT_RTOL,
                atol: float = DEFAULT_ATOL) -> LoadedBlocks:
    """The blocks a CalcJob, point or group reference names.

    ``ref`` may be prefixed ``pk:``, ``uuid:``, ``point:`` or ``group:`` to say
    what it is. Unprefixed, the order is: an all-digit ref as a pk (falling
    through to a group of that label), anything else as a group label, then a
    uuid, then a ``point_uuid``. A group gives the fields shared by its current
    (non-superseded) CalcJobs; ``rtol``/``atol`` decide when derived values
    agree.

    Raises:
        LookupError: ``ref`` names nothing, or a group with no stored blocks.
    """
    prefix, sep, rest = ref.partition(':')
    if sep and prefix in _PREFIXES:
        ref, order = rest, (prefix,)
    elif ref.isdigit():
        order = ('pk', 'group')
    else:
        order = ('group', 'uuid', 'point')
    for step in order:
        if step == 'group':
            group = _group_for(ref)
            if group is not None:
                return _from_group(group, rtol, atol)
            continue
        node = _node_for(ref, (step,))
        if node is not None:
            job = _calcjob_of(node)
            if job is None:
                raise LookupError(f'{ref!r} is not a CalcJob or one of its outputs')
            return _loaded_from_calcjob(job)
    raise LookupError(f'{ref!r} is not a CalcJob, point, or group of this profile')


def _schema_for(root: Path, versions: List[str]) -> Optional[Path]:
    """The ``schema-<version>.json`` a deck's blocks were written at (their
    ``block_schema``), else the newest one present."""
    found = {p.stem[len('schema-'):]: p for p in root.glob('schema-*.json')}
    for version in versions:
        if version in found:
            return found[version]
    return found[max(found, key=version_key)] if found else None


def _loaded_from_sidecar_file(path: Path, root: Path) -> LoadedBlocks:
    document = json.loads(path.read_text(encoding='utf-8'))
    derived = document.get('derived') or {}
    derived = derived.get('fields', derived)
    contents = {name: content for name, content in document.items() if name != 'derived'}
    schema = _schema_for(root, sorted({str(c.get('block_schema')) for c in contents.values()}))
    units, defaults, kinds = (_from_schema(json.loads(schema.read_text(encoding='utf-8')))
                              if schema else ({}, {}, {}))
    return LoadedBlocks({n: dict(c['fields']) for n, c in contents.items()},
                        dict(derived), units, defaults, kinds)


def load_blocks_from_sidecar(path: Path, *, rtol: float = DEFAULT_RTOL,
                             atol: float = DEFAULT_ATOL) -> LoadedBlocks:
    """Blocks from an archive's ``<archive>.params/`` directory (or one deck's JSON
    in it): ``<rel_path>/<deck>.json`` = ``{block name: content, 'derived': ...}``
    and ``schema-<version>.json`` for units and defaults. Each deck is read
    against the schema of its own ``block_schema`` version (the newest present
    when that one is missing). Needs no aiida-core."""
    path = Path(path)
    root = path if path.is_dir() else next(
        (d for d in path.resolve().parents if any(d.glob('schema-*.json'))), path.parent)
    files = (sorted(p for p in path.rglob('*.json') if not p.name.startswith('schema-'))
             if path.is_dir() else [path])
    if not files:
        raise LookupError(f'no deck sidecars under {path}')
    loaded = [_loaded_from_sidecar_file(f, root) for f in files]
    return loaded[0] if len(loaded) == 1 else _merge(loaded, rtol=rtol, atol=atol)
