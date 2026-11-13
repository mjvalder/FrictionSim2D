"""Content-addressed nodes in the AiiDA store (spec §5.2).

A parameter block (:mod:`src.aiida.blocks`) is stored once as an ``orm.Dict``
and found again by its hash, so every run with an identical block links to
the same node. A file (a potential, a CIF) is stored once as an
``orm.SinglefileData`` and found again by its sha256. Lookups go through node
extras: ``friction2d_block_hash`` / ``friction2d_block_kind`` on blocks,
``friction2d_sha256`` on files.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Mapping, Optional, Type

from aiida import orm
from aiida.common.exceptions import NotExistent

from ..data.provenance import sha256_file
from .blocks import block_hash, block_kind
from .data.redact import stage_file

BLOCK_HASH_EXTRA = 'friction2d_block_hash'
BLOCK_KIND_EXTRA = 'friction2d_block_kind'
FILE_SHA_EXTRA = 'friction2d_sha256'


def _find(node_class: Type[orm.Node], extra: str, value: str) -> Optional[orm.Node]:
    """The oldest stored ``node_class`` whose extra ``extra`` equals ``value``."""
    qb = orm.QueryBuilder().append(
        node_class, filters={f'extras.{extra}': value}, project='*', tag='n')
    qb.order_by({'n': {'id': 'asc'}}).limit(1)
    found = qb.first()
    return found[0] if found else None


def get_or_create_block(kind: str, content: Mapping[str, Any]) -> orm.Dict:
    """The stored ``Dict`` for this block, created on first sight.

    ``kind`` may be a block name (``material__sheet_0``): the node is keyed by
    the kind alone, so one material in two roles is one node.
    """
    kind = block_kind(kind)
    digest = block_hash(kind, content)
    found = _find(orm.Dict, BLOCK_HASH_EXTRA, digest)
    if found is not None:
        return found
    node = orm.Dict(dict(content)).store()
    node.base.extras.set_many({BLOCK_HASH_EXTRA: digest, BLOCK_KIND_EXTRA: kind})
    return node


def get_or_create_file(path: Path) -> orm.SinglefileData:
    """The stored ``SinglefileData`` for this file's content, created on first sight.

    Keyed by the sha256 of the file on disk. The content goes in through
    :func:`~src.aiida.data.redact.stage_file`, so a settings/config document
    is sanitised before it is stored.
    """
    path = Path(path)
    digest = sha256_file(path)
    found = _find(orm.SinglefileData, FILE_SHA_EXTRA, digest)
    if found is not None:
        return found
    node = orm.SinglefileData(io.BytesIO(b''), filename=path.name)
    stage_file(node.base.repository, path, path.name)
    node.store()
    node.base.extras.set(FILE_SHA_EXTRA, digest)
    return node


def machine_computer(name: str) -> orm.Computer:
    """The ``Computer`` labelled ``name``, created and configured on first use.

    Imported runs are never executed through it: it records where a run
    came from. Hence ``core.local``/``core.direct`` for every machine.
    """
    try:
        computer = orm.Computer.collection.get(label=name)
    except NotExistent:
        computer = orm.Computer(
            label=name,
            hostname='localhost',
            description=f'Machine {name!r} that produced imported FrictionSim2D runs',
            transport_type='core.local',
            scheduler_type='core.direct',
            workdir=str(Path('~/aiida_workdir').expanduser() / name),
        ).store()
    if not computer.is_configured:
        computer.configure()
    return computer
