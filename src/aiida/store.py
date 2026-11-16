"""Content-addressed nodes in the AiiDA store (spec §5.2).

A parameter block (:mod:`src.aiida.blocks`) is stored once as an ``orm.Dict``
and found again by its hash, so every run with an identical block links to
the same node. A file (a potential, a CIF) is stored once as an
``orm.SinglefileData`` and found again by its sha256. Lookups go through node
extras: ``friction2d_block_hash`` / ``friction2d_block_kind`` on blocks,
``friction2d_sha256`` on files, ``friction2d_folder_sha256`` on a run's
provenance folder.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path, PurePosixPath
from typing import Any, List, Mapping, Optional, Tuple, Type

from aiida import orm
from aiida.common.exceptions import NotExistent

from ..data.provenance import sha256_file
from .blocks import block_hash, block_kind
from .data.redact import stage_file

BLOCK_HASH_EXTRA = 'friction2d_block_hash'
BLOCK_KIND_EXTRA = 'friction2d_block_kind'
FILE_SHA_EXTRA = 'friction2d_sha256'
FOLDER_SHA_EXTRA = 'friction2d_folder_sha256'

_CONFLICT = '.sync-conflict-'


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
    node = orm.Dict(dict(content))
    node.base.extras.set_many({BLOCK_HASH_EXTRA: digest, BLOCK_KIND_EXTRA: kind})
    return node.store()


class _AsBasename:
    """A repository view that stores every object under its basename.

    ``stage_file`` decides whether to sanitise from the full name it is given
    (a ``provenance/`` component counts), while a ``SinglefileData`` holds
    exactly one top-level object.
    """

    def __init__(self, repository: Any) -> None:
        self._repository = repository

    def put_object_from_file(self, source: str, name: str) -> None:
        """Store the file at ``source`` under ``name``'s basename."""
        self._repository.put_object_from_file(source, PurePosixPath(name).name)

    def put_object_from_bytes(self, content: bytes, name: str) -> None:
        """Store ``content`` under ``name``'s basename."""
        self._repository.put_object_from_bytes(content, PurePosixPath(name).name)


def _staging_name(path: Path) -> str:
    """``path`` from its last ``provenance`` component on, else its basename."""
    parts = path.parts
    lowered = [p.lower() for p in parts[:-1]]
    if 'provenance' in lowered:
        start = len(lowered) - 1 - lowered[::-1].index('provenance')
        return PurePosixPath(*parts[start:]).as_posix()
    return path.name


def get_or_create_file(path: Path, name: Optional[str] = None) -> orm.SinglefileData:
    """The stored ``SinglefileData`` for this file's content, created on first sight.

    Keyed by the sha256 of the file on disk. The content goes in through
    :func:`~src.aiida.data.redact.stage_file` under ``name`` -- by default the
    path from its last ``provenance/`` component on (``provenance/cif/x.cif``),
    else the basename -- so redaction's name- and location-based detection
    both apply. The node's single file is the basename of ``name``.
    """
    path = Path(path)
    name = name or _staging_name(path)
    digest = sha256_file(path)
    found = _find(orm.SinglefileData, FILE_SHA_EXTRA, digest)
    if found is not None:
        return found
    node = orm.SinglefileData(io.BytesIO(b''), filename=PurePosixPath(name).name)
    stage_file(_AsBasename(node.base.repository), path, name)
    node.base.extras.set(FILE_SHA_EXTRA, digest)
    return node.store()


def new_file(path: Path, name: Optional[str] = None) -> orm.SinglefileData:
    """An UNSTORED ``SinglefileData`` of ``path``, staged through redaction.

    For process outputs, which must be new nodes. ``name`` (default: the
    basename) is what redaction judges the file by; the node's file is its
    basename.
    """
    path = Path(path)
    name = name or path.name
    node = orm.SinglefileData(io.BytesIO(b''), filename=PurePosixPath(name).name)
    stage_file(_AsBasename(node.base.repository), path, name)
    return node


def new_folder(files: Mapping[str, Path]) -> orm.FolderData:
    """An UNSTORED ``FolderData`` of ``{stored name: path}``, staged through
    redaction."""
    node = orm.FolderData()
    for name, path in files.items():
        stage_file(node.base.repository, Path(path), name)
    return node


def _provenance_members(provenance_file: Optional[Path],
                         provenance_dir: Optional[Path]) -> List[Tuple[str, Path]]:
    """``(stored name, path)`` of every provenance file, sorted by name.

    Syncthing ``*.sync-conflict-*`` copies are left out: the original is the
    record.
    """
    members: List[Tuple[str, Path]] = []
    if provenance_file is not None and Path(provenance_file).is_file():
        members.append(('provenance.json', Path(provenance_file)))
    if provenance_dir is not None and Path(provenance_dir).is_dir():
        root = Path(provenance_dir)
        for path in root.rglob('*'):
            if path.is_file() and _CONFLICT not in path.name:
                rel = path.relative_to(root).as_posix()
                members.append((f'provenance/{rel}', path))
    return sorted(members)


def get_or_create_provenance(provenance_file: Optional[Path],
                             provenance_dir: Optional[Path]) -> orm.FolderData:
    """The stored ``FolderData`` of a run's ``provenance.json`` + ``provenance/``.

    Keyed by a sha256 over every member's stored name and file sha256, so two
    runs with byte-identical provenance share one node. Every member goes in
    through :func:`~src.aiida.data.redact.stage_file` under its stored name
    (``provenance/settings.yaml``), so a credential never reaches the store.
    """
    members = _provenance_members(provenance_file, provenance_dir)
    digest = hashlib.sha256()
    for name, path in members:
        digest.update(f'{name}\0{sha256_file(path)}\n'.encode('utf-8'))
    key = digest.hexdigest()
    found = _find(orm.FolderData, FOLDER_SHA_EXTRA, key)
    if found is not None:
        return found
    node = orm.FolderData()
    for name, path in members:
        stage_file(node.base.repository, path, name)
    node.base.extras.set(FOLDER_SHA_EXTRA, key)
    return node.store()


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
