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

from ..core.compare import merge_varying
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
    """
    blocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    derived: Dict[str, Any] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)


def _from_schema(document: Mapping[str, Any]):
    fields = document.get('fields') or {}
    units = {p: f['unit'] for p, f in fields.items() if f.get('unit')}
    defaults = {p: f['default'] for p, f in fields.items() if f.get('default') is not None}
    return units, defaults


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
    units, defaults = _from_schema(inputs.schema.get_dict()) if 'schema' in inputs else ({}, {})
    return LoadedBlocks(blocks, derived, units, defaults)


def _merge(loaded: List[LoadedBlocks]) -> LoadedBlocks:
    units: Dict[str, str] = {}
    defaults: Dict[str, Any] = {}
    for item in loaded:
        units.update(item.units)
        defaults.update(item.defaults)
    blocks, derived = merge_varying([(i.blocks, i.derived) for i in loaded],
                                    defaults=defaults)
    return LoadedBlocks(blocks, derived, units, defaults)


def _calcjob_of(node):
    from aiida import orm  # pylint: disable=import-outside-toplevel

    if isinstance(node, orm.CalcJobNode):
        return node
    creator = getattr(node, 'creator', None)
    return creator if isinstance(creator, orm.CalcJobNode) else None


def _node_for(ref: str):
    """The node a pk, uuid or ``point_uuid`` names, or None."""
    from aiida import orm  # pylint: disable=import-outside-toplevel
    from aiida.common.exceptions import (  # pylint: disable=import-outside-toplevel
        MultipleObjectsError, NotExistent)

    try:
        return orm.load_node(int(ref) if ref.isdigit() else ref)
    except (NotExistent, MultipleObjectsError, ValueError):
        pass
    query = orm.QueryBuilder().append(
        orm.Node, filters={'attributes.point_uuid': ref}, project=['id'])
    found = query.first()
    return orm.load_node(found[0]) if found else None


def _group_for(ref: str):
    from aiida import orm  # pylint: disable=import-outside-toplevel

    from .ingest import group_label  # pylint: disable=import-outside-toplevel

    for label in (ref, group_label(ref)):
        group = orm.Group.collection.find(filters={'label': label})
        if group:
            return group[0]
    return None


def load_blocks(ref: str) -> LoadedBlocks:
    """The blocks a CalcJob, point or group reference names.

    Raises:
        LookupError: ``ref`` names nothing, or a group with no stored blocks.
    """
    group = _group_for(ref)
    if group is not None:
        jobs = {}
        for member in group.nodes:
            job = _calcjob_of(member)
            if job is not None and 'blocks' in job.inputs:
                jobs[job.pk] = job
        if not jobs:
            raise LookupError(f'group {group.label!r} holds no CalcJob with parameter blocks')
        return _merge([_loaded_from_calcjob(j) for _, j in sorted(jobs.items())])
    node = _node_for(ref)
    job = _calcjob_of(node) if node is not None else None
    if job is None:
        raise LookupError(f'{ref!r} is not a CalcJob, point, or group of this profile')
    return _loaded_from_calcjob(job)


def _newest_schema(root: Path) -> Optional[Path]:
    found = list(root.glob('schema-*.json'))
    return max(found, key=lambda p: version_key(p.stem[len('schema-'):])) if found else None


def _loaded_from_sidecar_file(path: Path) -> LoadedBlocks:
    document = json.loads(path.read_text(encoding='utf-8'))
    derived = document.get('derived') or {}
    derived = derived.get('fields', derived)
    blocks = {name: dict(content['fields']) for name, content in document.items()
              if name != 'derived'}
    return LoadedBlocks(blocks, dict(derived))


def load_blocks_from_sidecar(path: Path) -> LoadedBlocks:
    """Blocks from an archive's ``<archive>.params/`` directory (or one deck's JSON
    in it): ``<rel_path>/<deck>.json`` = ``{block name: content, 'derived': ...}``
    and ``schema-<version>.json`` for units and defaults. Needs no aiida-core."""
    path = Path(path)
    root = path if path.is_dir() else path.parent
    files = (sorted(p for p in path.rglob('*.json') if not p.name.startswith('schema-'))
             if path.is_dir() else [path])
    if not files:
        raise LookupError(f'no deck sidecars under {path}')
    units: Dict[str, str] = {}
    defaults: Dict[str, Any] = {}
    schema = _newest_schema(root)
    if schema is not None:
        units, defaults = _from_schema(json.loads(schema.read_text(encoding='utf-8')))
    loaded = [_loaded_from_sidecar_file(f) for f in files]
    for item in loaded:
        item.units, item.defaults = units, defaults
    return loaded[0] if len(loaded) == 1 else _merge(loaded)
