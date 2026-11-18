"""Content-addressed nodes in the AiiDA store (spec §5.2).

A parameter block (:mod:`src.aiida.blocks`) is stored once as an ``orm.Dict``
and found again by its hash, so every run with an identical block links to
the same node. A file (a potential, a CIF) is stored once as an
``orm.SinglefileData`` and found again by the sha256 of its STORED bytes --
what went in through :func:`~src.aiida.data.redact.stage_file` -- so the
published hash can be checked against the published content (critique M2).
A schema document is stored once, like a block.

Lookups go through node extras, which are an index only:
``friction2d_block_hash`` / ``friction2d_block_kind`` on blocks,
``friction2d_sha256`` on files, ``friction2d_folder_sha256`` on a run's
provenance folder, ``friction2d_schema_hash`` on a schema. Each is
recomputable from the node itself (:func:`reindex`).

Every content-addressed node gets the deterministic UUID :func:`node_uuid` of
its slot (block, file, folder, schema) and hash (critique C2).
``import_archive`` matches nodes by UUID
(``aiida/tools/archive/imports.py``, ``_import_nodes``: archive nodes whose
uuid already exists in the profile are not added, only their extras merged),
so the same block built by two scientists in two profiles is ONE node after an
archive import, and the UUID unique constraint (``db_dbnode.uuid``) is the
database-level guarantee. Setting the UUID uses private API
(:func:`_set_uuid`); when it is unavailable the node keeps a random UUID and
joins on the hash extras still work (:func:`verify` reports the duplicates).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union
from uuid import UUID, uuid5

from aiida import orm
from aiida.common.exceptions import NotExistent

from ..data.provenance import sha256_file
from .blocks import BLOCK_KINDS, block_hash, block_kind
from .data.redact import stage_file

logger = logging.getLogger(__name__)

BLOCK_HASH_EXTRA = 'friction2d_block_hash'
BLOCK_KIND_EXTRA = 'friction2d_block_kind'
FILE_SHA_EXTRA = 'friction2d_sha256'
FOLDER_SHA_EXTRA = 'friction2d_folder_sha256'
SCHEMA_HASH_EXTRA = 'friction2d_schema_hash'
#: The CalcJob extra that carries the run's identity (``provenance.json``
#: ``run_uuid``), so every deck of one run is found by one value.
RUN_UUID_EXTRA = 'friction2d_run_uuid'
#: The CalcJob extra that carries the deck's identity,
#: ``f'{run_uuid}|{deck_name}'`` -- recomputable from ``parameters``.
DECK_KEY_EXTRA = 'friction2d_deck_key'
#: The CalcJob extra that carries the deck's fingerprint
#: (:func:`src.aiida.ingest.deck_fingerprint`): what was imported.
DECK_SHA_EXTRA = 'friction2d_deck_sha256'


#: The CalcJob extra that carries the run's directory tokens
#: (:class:`src.run.naming.RunTokens` as a dict, ``family`` from
#: ``parameters``) -- what :mod:`src.aiida.query` filters on.
TOKENS_EXTRA = 'friction2d_tokens'


def deck_key(run_uuid: str, deck_name: str) -> str:
    """A deck's identity: ``f'{run_uuid}|{deck_name}'``."""
    return f'{run_uuid}|{deck_name}'


def run_tokens(rel_path: Optional[str], family: Optional[str] = None
               ) -> Optional[Dict[str, Any]]:
    """``RunTokens`` of ``rel_path`` as a JSON dict, ``family`` replacing the
    path's guess; None for a path :func:`src.run.naming.parse_run_path`
    cannot read."""
    from dataclasses import asdict  # pylint: disable=import-outside-toplevel

    from ..run.naming import parse_run_path  # pylint: disable=import-outside-toplevel

    if not rel_path:
        return None
    try:
        tokens = asdict(parse_run_path(rel_path))
    except ValueError:
        return None
    tokens['materials'] = list(tokens['materials'])
    if family:
        tokens['family'] = family
    return tokens


def calcjob_extras(params: Mapping[str, Any]) -> Dict[str, Any]:
    """The run, deck and token extras of a friction2d CalcJob, recomputed from
    its ``parameters``: :data:`RUN_UUID_EXTRA`, :data:`DECK_KEY_EXTRA`,
    :data:`TOKENS_EXTRA` (each only when its source is there)."""
    out: Dict[str, Any] = {}
    if params.get('run_uuid'):
        out[RUN_UUID_EXTRA] = params['run_uuid']
        if params.get('deck_name'):
            out[DECK_KEY_EXTRA] = deck_key(params['run_uuid'], params['deck_name'])
    tokens = run_tokens(params.get('rel_path'), params.get('family'))
    if tokens is not None:
        out[TOKENS_EXTRA] = tokens
    return out

#: The uuid5 namespace of every content-addressed FrictionSim2D node. Fixed
#: once (``uuid5(NAMESPACE_URL, 'https://frictionsim2d/aiida/node')``) and
#: NEVER to be changed -- nor are the slot strings of :data:`UUID_SLOTS`, nor
#: the ``f'{slot}:{digest}'`` name format of :func:`node_uuid`: changing any
#: of them splits every stored node from its twin in every other profile.
FRICTION2D_NAMESPACE = UUID('e1ff2212-fbb6-5980-a513-7e1d9d76aa12')

#: The process type of the import-only CalcJob (entry point ``friction2d.lammps``).
CALCJOB_PROCESS_TYPE = 'aiida.calculations:friction2d.lammps'

_CONFLICT = '.sync-conflict-'
_CHUNK = 1 << 20

_warned_uuid = False   # pylint: disable=invalid-name


#: The kinds of content-addressed node, each its own uuid name space, so a
#: file whose bytes are a block's canonical JSON is not that block's node.
#: Frozen with :data:`FRICTION2D_NAMESPACE`.
UUID_SLOTS = ('block', 'file', 'folder', 'schema')


def node_uuid(slot: str, digest: str) -> str:
    """The deterministic UUID of a ``slot`` node with content hash ``digest``:
    ``uuid5(FRICTION2D_NAMESPACE, f'{slot}:{digest}')``."""
    if slot not in UUID_SLOTS:
        raise ValueError(f'unknown node slot {slot!r}; expected one of {UUID_SLOTS}')
    return str(uuid5(FRICTION2D_NAMESPACE, f'{slot}:{digest}'))


def _set_uuid(node: orm.Node, value: str) -> None:
    """Give the unstored ``node`` the UUID ``value``.

    Private API: ``node.backend_entity.bare_model`` is the SQLAlchemy
    ``DbNode`` (``aiida/storage/psql_dos/orm/entities.py``, ``bare_model``,
    shared by the sqlite backends). Its ``uuid`` is populated at construction
    by ``instant_defaults_listener`` (``storage/psql_dos/models/base.py``)
    with ``get_new_uuid()`` -- a ``str`` -- and is only written on ``store()``,
    so assigning it first stores the node under ``value``. When the
    attribute is missing (a backend or version without it), one warning is
    logged and the random UUID is kept; hash-value joins stay correct.
    """
    global _warned_uuid  # pylint: disable=global-statement
    try:
        model = node.backend_entity.bare_model
        if not hasattr(model, 'uuid'):
            raise AttributeError('uuid')
        model.uuid = value
    except AttributeError:
        if not _warned_uuid:
            logger.warning('Cannot set a deterministic node UUID on this AiiDA '
                           'backend; content-addressed nodes keep random UUIDs '
                           '(joins on the hash extras still work).')
            _warned_uuid = True


# --- the lookup memo -------------------------------------------------------------

@dataclass
class StoreIndex:
    """Hash -> pk of every content-addressed node: one lookup memo per import.

    Attributes:
        blocks: ``friction2d_block_hash`` -> ``Dict`` pk.
        files: ``friction2d_sha256`` -> ``SinglefileData`` pk.
        folders: ``friction2d_folder_sha256`` -> ``FolderData`` pk.
        schemas: ``friction2d_schema_hash`` -> ``Dict`` pk.
    """
    blocks: Dict[str, int] = field(default_factory=dict)
    files: Dict[str, int] = field(default_factory=dict)
    folders: Dict[str, int] = field(default_factory=dict)
    schemas: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls) -> 'StoreIndex':
        """The index of the loaded profile: one projection query per extra.

        Where two nodes share a hash (see :func:`verify`), the one under the
        deterministic uuid wins, else the oldest -- as in an unindexed lookup.
        """
        index = cls()
        for attr, node_class, extra, slot in _SLOTS:
            qb = orm.QueryBuilder().append(
                node_class, tag='n', filters={'extras': {'has_key': extra}},
                project=[f'extras.{extra}', 'id', 'uuid'])
            qb.order_by({'n': {'id': 'asc'}})
            mapping = getattr(index, attr)
            for digest, pk, uuid in qb.iterall():
                if str(uuid) == node_uuid(slot, digest):
                    mapping[digest] = pk
                else:
                    mapping.setdefault(digest, pk)
        return index


#: (StoreIndex attribute, node class, hash extra, uuid slot).
_SLOTS: Tuple[Tuple[str, Type[orm.Node], str, str], ...] = (
    ('blocks', orm.Dict, BLOCK_HASH_EXTRA, 'block'),
    ('files', orm.SinglefileData, FILE_SHA_EXTRA, 'file'),
    ('folders', orm.FolderData, FOLDER_SHA_EXTRA, 'folder'),
    ('schemas', orm.Dict, SCHEMA_HASH_EXTRA, 'schema'),
)
_SLOT_OF = {attr: (node_class, extra, slot) for attr, node_class, extra, slot in _SLOTS}


def _find(node_class: Type[orm.Node], extra: str, value: str) -> Optional[orm.Node]:
    """The oldest stored ``node_class`` whose extra ``extra`` equals ``value``."""
    qb = orm.QueryBuilder().append(
        node_class, filters={f'extras.{extra}': value}, project='*', tag='n')
    qb.order_by({'n': {'id': 'asc'}}).limit(1)
    found = qb.first()
    return found[0] if found else None


def _find_by_uuid(value: str) -> Optional[orm.Node]:
    found = orm.QueryBuilder().append(
        orm.Node, filters={'uuid': value}, project='*').first()
    return found[0] if found else None


def _content_digest(slot: str, node: orm.Node) -> Optional[str]:
    """``node``'s hash recomputed from its stored content (None if unreadable)."""
    try:
        if slot == 'block':
            return block_hash(node.get_dict())
        if slot == 'schema':
            return schema_hash(node.get_dict())
        if slot == 'file':
            return _file_sha(node)
        return _stored_folder_key(node)
    except (TypeError, ValueError, OSError, AttributeError):
        return None


def _get_or_create(attr: str, digest: str, extras: Mapping[str, str],
                   build: Callable[[], orm.Node],
                   index: Optional[StoreIndex]) -> orm.Node:
    """The stored node for ``digest``, else ``build()`` stored under :func:`node_uuid`.

    Lookup order:

    1. the memo ``index``, whose miss is trusted (it was loaded for this import);
    2. without a memo, the node under the deterministic uuid if it carries the
       hash extra, else the oldest node carrying it;
    3. the node under the deterministic uuid whose extras were dropped
       (``import_new_extras=False``): its content is checked against
       ``digest`` before the extras are written back. A uuid held by any
       other node -- other class, or content that does not hash to
       ``digest`` -- is left alone, and the new node gets a random uuid.
    """
    node_class, extra, slot = _SLOT_OF[attr]
    mapping = getattr(index, attr) if index is not None else None
    if mapping is not None and digest in mapping:
        return orm.load_node(mapping[digest])
    wanted = node_uuid(slot, digest)
    twin = _find_by_uuid(wanted)
    found = None
    if (mapping is None and isinstance(twin, node_class)
            and twin.base.extras.get(extra, None) == digest):
        found = twin
    elif mapping is None:
        found = _find(node_class, extra, digest)
    if found is None and twin is not None:
        if isinstance(twin, node_class) and _content_digest(slot, twin) == digest:
            twin.base.extras.set_many(dict(extras))
            found = twin
        else:
            logger.warning('Node %s holds the uuid of %s %s but its content does '
                           'not match; storing the new node under a random uuid',
                           twin.pk, extra, digest)
            wanted = ''
    if found is None:
        found = build()
        found.base.extras.set_many(dict(extras))
        if wanted:
            _set_uuid(found, wanted)
        found.store()
    if mapping is not None:
        mapping[digest] = found.pk
    return found


# --- blocks and schemas ----------------------------------------------------------

def get_or_create_block(content: Mapping[str, Any], *,
                        index: Optional[StoreIndex] = None) -> orm.Dict:
    """The stored ``Dict`` for this block, created on first sight.

    ``content`` is the canonical block (:func:`src.aiida.blocks.split_blocks`);
    its ``kind`` is part of the content, so the hash is recomputable from the
    node alone. Keyed by the content hash, so one material in two roles is
    one node.
    """
    kind = block_kind(str(content.get('kind')))
    digest = block_hash(content)
    return _get_or_create('blocks', digest,
                          {BLOCK_HASH_EXTRA: digest, BLOCK_KIND_EXTRA: kind},
                          lambda: orm.Dict(dict(content)), index)


def schema_hash(document: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of a schema document."""
    payload = json.dumps(document, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def get_or_create_schema(document: Mapping[str, Any], *,
                         index: Optional[StoreIndex] = None) -> orm.Dict:
    """The stored ``Dict`` of a schema document
    (:func:`src.aiida.block_schema.schema_document`), content-addressed like a block."""
    digest = schema_hash(document)
    return _get_or_create('schemas', digest, {SCHEMA_HASH_EXTRA: digest},
                          lambda: orm.Dict(dict(document)), index)


# --- files -------------------------------------------------------------------------

class _Capture:
    """A repository stand-in that records what :func:`stage_file` would store.

    A verbatim file stays a path (hashed and stored by streaming it); a
    sanitised document is held as its redacted bytes.
    """

    def __init__(self) -> None:
        self.objects: List[Tuple[str, Union[Path, bytes]]] = []

    def put_object_from_file(self, source: str, name: str) -> None:
        """Record ``source`` (unchanged) under ``name``."""
        self.objects.append((name, Path(source)))

    def put_object_from_bytes(self, content: bytes, name: str) -> None:
        """Record ``content`` under ``name``."""
        self.objects.append((name, content))


def _staged(path: Path, name: str) -> Union[Path, bytes]:
    """What :func:`stage_file` stores for ``path`` under ``name``."""
    capture = _Capture()
    stage_file(capture, path, name)
    return capture.objects[0][1]


def _sha(content: Union[Path, bytes]) -> str:
    if isinstance(content, Path):
        return sha256_file(content)
    return hashlib.sha256(content).hexdigest()


def staged_sha256(path: Path, name: Optional[str] = None) -> str:
    """The sha256 the store records for ``path`` staged under ``name``
    (default: as :func:`get_or_create_file` names it), without storing anything."""
    path = Path(path)
    return _sha(_staged(path, name or _staging_name(path)))


def _put(repository: Any, content: Union[Path, bytes], name: str) -> None:
    if isinstance(content, Path):
        repository.put_object_from_file(str(content), name)
    else:
        repository.put_object_from_bytes(content, name)


def _staging_name(path: Path) -> str:
    """``path`` from its last ``provenance`` component on, else its basename."""
    parts = path.parts
    lowered = [p.lower() for p in parts[:-1]]
    if 'provenance' in lowered:
        start = len(lowered) - 1 - lowered[::-1].index('provenance')
        return PurePosixPath(*parts[start:]).as_posix()
    return path.name


def get_or_create_file(path: Path, name: Optional[str] = None, *,
                       index: Optional[StoreIndex] = None) -> orm.SinglefileData:
    """The stored ``SinglefileData`` for this file's content, created on first sight.

    The file is staged through :func:`~src.aiida.data.redact.stage_file`
    under ``name`` -- by default the path from its last ``provenance/``
    component on (``provenance/cif/x.cif``), else the basename -- so
    redaction's name- and location-based detection both apply. It is keyed
    by the sha256 of the STAGED bytes (``friction2d_sha256``), which are
    exactly the stored bytes. The node's single file is the basename of ``name``.
    """
    path = Path(path)
    name = name or _staging_name(path)
    filename = PurePosixPath(name).name
    content = _staged(path, name)
    digest = _sha(content)

    def build() -> orm.SinglefileData:
        node = orm.SinglefileData(io.BytesIO(b''), filename=filename)
        _put(node.base.repository, content, filename)
        return node

    return _get_or_create('files', digest,
                          {FILE_SHA_EXTRA: digest}, build, index)


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
                         provenance_dir: Optional[Path],
                         config_file: Optional[Path] = None) -> List[Tuple[str, Path]]:
    """``(stored name, path)`` of every provenance file, sorted by name.

    Syncthing ``*.sync-conflict-*`` copies are left out: the original is the
    record. ``config_file``, when given, is stored as
    ``provenance/config.json`` in place of the file of that name: the config
    the run was built from (``calcjob.run_config_file``), when
    ``provenance/config.json`` was overwritten after the build.
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
    if config_file is not None:
        members = [m for m in members if m[0] != 'provenance/config.json']
        members.append(('provenance/config.json', Path(config_file)))
    return sorted(members)


def _folder_key(members: List[Tuple[str, str]]) -> str:
    """sha256 over sorted ``(stored name, sha256 of stored bytes)`` pairs."""
    digest = hashlib.sha256()
    for name, sha in sorted(members):
        digest.update(f'{name}\0{sha}\n'.encode('utf-8'))
    return digest.hexdigest()


def provenance_key(provenance_file: Optional[Path],
                   provenance_dir: Optional[Path],
                   config_file: Optional[Path] = None) -> str:
    """The key :func:`get_or_create_provenance` stores this provenance
    under, computed without storing anything."""
    return _folder_key([(name, _sha(_staged(path, name)))
                        for name, path in _provenance_members(provenance_file,
                                                              provenance_dir, config_file)])


def get_or_create_provenance(provenance_file: Optional[Path],
                             provenance_dir: Optional[Path], *,
                             config_file: Optional[Path] = None,
                             index: Optional[StoreIndex] = None) -> orm.FolderData:
    """The stored ``FolderData`` of a run's ``provenance.json`` + ``provenance/``.

    Every member is staged through :func:`~src.aiida.data.redact.stage_file`
    under its stored name (``provenance/settings.yaml``), so a credential
    never reaches the store. Keyed by a sha256 over every member's stored
    name and the sha256 of its STAGED bytes, so two runs with byte-identical
    provenance share one node and the key is recomputable from the node.
    ``config_file`` replaces ``provenance/config.json``
    (:func:`_provenance_members`).
    """
    staged = [(name, _staged(path, name))
              for name, path in _provenance_members(provenance_file, provenance_dir,
                                                    config_file)]
    key = _folder_key([(name, _sha(content)) for name, content in staged])

    def build() -> orm.FolderData:
        node = orm.FolderData()
        for name, content in staged:
            _put(node.base.repository, content, name)
        return node

    return _get_or_create('folders', key, {FOLDER_SHA_EXTRA: key},
                          build, index)


# --- recomputation from the nodes themselves ----------------------------------------

def _stream_sha(repository: Any, name: str) -> str:
    digest = hashlib.sha256()
    with repository.open(name, mode='rb') as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha(node: orm.SinglefileData) -> str:
    return _stream_sha(node.base.repository, node.filename)


def _stored_folder_key(node: orm.FolderData) -> str:
    repository = node.base.repository
    members = []
    for root, _dirs, files in repository.walk():
        for name in files:
            path = (root / name).as_posix()
            members.append((path, _stream_sha(repository, path)))
    return _folder_key(members)


def _is_block(content: Mapping[str, Any]) -> bool:
    return content.get('kind') in BLOCK_KINDS and 'fields' in content


def _is_schema(content: Mapping[str, Any]) -> bool:
    return ('kind' not in content and
            {'block_schema', 'family', 'fields'} <= set(content))


def _dict_extras(node: orm.Dict) -> Dict[str, str]:
    """The index extras ``node``'s content implies (empty for other Dicts)."""
    content = node.get_dict()
    try:
        if _is_block(content):
            return {BLOCK_HASH_EXTRA: block_hash(content),
                    BLOCK_KIND_EXTRA: str(content['kind'])}
        if _is_schema(content):
            return {SCHEMA_HASH_EXTRA: schema_hash(content)}
    except (TypeError, ValueError):
        pass
    return {}


def _calcjob_inputs(node_class: Type[orm.Node]) -> List[orm.Node]:
    """``node_class`` inputs of every friction2d CalcJob that no process created."""
    qb = orm.QueryBuilder().append(
        orm.CalcJobNode, tag='c', filters={'process_type': CALCJOB_PROCESS_TYPE})
    qb.append(node_class, with_outgoing='c', project='*')
    seen: Dict[int, orm.Node] = {}
    for (node,) in qb.iterall():
        if node.pk not in seen and node.creator is None:
            seen[node.pk] = node
    return list(seen.values())


def _with_extra(node_class: Type[orm.Node], extra: str) -> List[orm.Node]:
    return orm.QueryBuilder().append(
        node_class, filters={'extras': {'has_key': extra}}).all(flat=True)


def _content_nodes() -> Tuple[List[orm.Node], List[orm.Node]]:
    """friction2d ``SinglefileData`` and provenance ``FolderData`` nodes: those
    carrying the extra, plus the non-output inputs of friction2d CalcJobs."""
    files = {n.pk: n for n in _with_extra(orm.SinglefileData, FILE_SHA_EXTRA)}
    files.update({n.pk: n for n in _calcjob_inputs(orm.SinglefileData)})
    folders = {n.pk: n for n in _with_extra(orm.FolderData, FOLDER_SHA_EXTRA)}
    folders.update({n.pk: n for n in _calcjob_inputs(orm.FolderData)})
    return list(files.values()), list(folders.values())


def _expected_extras() -> List[Tuple[orm.Node, Dict[str, str]]]:
    """Every content-addressed node with the extras its content implies."""
    out: List[Tuple[orm.Node, Dict[str, str]]] = []
    for node in orm.QueryBuilder().append(orm.Dict).all(flat=True):
        wanted = _dict_extras(node)
        if wanted:
            out.append((node, wanted))
    files, folders = _content_nodes()
    out += [(node, {FILE_SHA_EXTRA: _file_sha(node)}) for node in files]
    out += [(node, {FOLDER_SHA_EXTRA: _stored_folder_key(node)}) for node in folders]
    return out


def reindex() -> int:
    """Recompute every friction2d index extra from the stored nodes themselves.

    * ``friction2d_block_hash`` / ``_kind`` from each block ``Dict``'s content
      (``'kind' in content``), ``friction2d_schema_hash`` from each schema;
    * ``friction2d_sha256`` from each friction2d ``SinglefileData``'s stored
      bytes, ``friction2d_folder_sha256`` from each provenance folder's;
    * the run, deck and token extras (``friction2d_run_uuid``,
      ``friction2d_deck_key``, ``friction2d_tokens``) of each friction2d
      CalcJob from its ``parameters``.

    Run after an archive import that dropped or edited extras. Returns the
    number of nodes whose extras changed.
    """
    touched = 0
    for node, wanted in _expected_extras():
        current = node.base.extras.all
        if any(current.get(k) != v for k, v in wanted.items()):
            node.base.extras.set_many(wanted)
            touched += 1
    qb = orm.QueryBuilder().append(
        orm.CalcJobNode, filters={'process_type': CALCJOB_PROCESS_TYPE})
    for calc in qb.all(flat=True):
        params = calc.inputs.parameters.get_dict() if 'parameters' in calc.inputs else {}
        wanted = calcjob_extras(params)
        current = calc.base.extras.all
        if any(current.get(k) != v for k, v in wanted.items()):
            calc.base.extras.set_many(wanted)
            touched += 1
    return touched


@dataclass
class VerifyReport:
    """What :func:`verify` found.

    Attributes:
        duplicate_hashes: hash -> uuids of the nodes sharing it (more than one).
        mismatches: ``'<uuid> <extra>: stored <x> != recomputed <y>'`` per
            node whose stored hash extra disagrees with its content.
    """
    duplicate_hashes: Dict[str, List[str]] = field(default_factory=dict)
    mismatches: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No duplicates and no mismatches."""
        return not self.duplicate_hashes and not self.mismatches

    def render(self) -> str:
        """Both findings as text, one line each."""
        lines = [f'{len(self.duplicate_hashes)} duplicate hash(es), '
                 f'{len(self.mismatches)} mismatch(es)']
        for digest, uuids in sorted(self.duplicate_hashes.items()):
            lines.append(f'duplicate {digest}: {", ".join(uuids)}')
        lines += [f'mismatch {line}' for line in self.mismatches]
        return '\n'.join(lines)


def verify() -> VerifyReport:
    """Report hashes held by more than one node, and stored hash extras that
    disagree with the node's content."""
    report = VerifyReport()
    for _attr, node_class, extra, _slot in _SLOTS:
        qb = orm.QueryBuilder().append(
            node_class, tag='n', filters={'extras': {'has_key': extra}},
            project=[f'extras.{extra}', 'uuid'])
        qb.order_by({'n': {'id': 'asc'}})
        groups: Dict[str, List[str]] = defaultdict(list)
        for digest, uuid in qb.iterall():
            groups[digest].append(str(uuid))
        report.duplicate_hashes.update(
            {d: uuids for d, uuids in groups.items() if len(uuids) > 1})
    for node, wanted in _expected_extras():
        current = node.base.extras.all
        for key, value in wanted.items():
            if key in current and current[key] != value:
                report.mismatches.append(
                    f'{node.uuid} {key}: stored {current[key]} != recomputed {value}')
    return report


_SCHEDULERS = {'pbs': 'core.pbspro', 'slurm': 'core.slurm'}
_IMPORT_WORKDIR = '/unused/friction2d-import'


def machine_computer(name: str, hpc: Mapping[str, Any]) -> orm.Computer:
    """The ``Computer`` labelled ``name``, created on first use and left
    **unconfigured**: it records where a run was produced, never executes.

    ``hostname`` and ``scheduler_type`` come from the run's ``settings.hpc``
    (``hpc_host``, ``scheduler_type``); ``workdir`` is a literal that is never
    expanded. An existing Computer is returned untouched.
    """
    try:
        return orm.Computer.collection.get(label=name)
    except NotExistent:
        pass
    hostname = hpc.get('hpc_host') or name
    return orm.Computer(
        label=name,
        hostname=hostname,
        description=f'Machine {name!r} that produced imported FrictionSim2D runs',
        transport_type='core.local' if hostname == 'localhost' else 'core.ssh',
        scheduler_type=_SCHEDULERS.get(hpc.get('scheduler_type'), 'core.direct'),
        workdir=_IMPORT_WORKDIR,
    ).store()


def machine_mismatch(name: str, hpc: Mapping[str, Any]) -> Optional[str]:
    """Why the existing Computer ``name`` disagrees with ``hpc``, or None.

    Only what ``hpc`` states is compared (``hpc_host``, ``scheduler_type``):
    :func:`machine_computer` reuses an existing label untouched, so a
    different host or scheduler under the same label would go unnoticed.
    """
    try:
        computer = orm.Computer.collection.get(label=name)
    except NotExistent:
        return None
    problems = []
    host = hpc.get('hpc_host')
    if host and computer.hostname != host:
        problems.append(f'hostname {computer.hostname!r} != {host!r}')
    scheduler = hpc.get('scheduler_type')
    if scheduler and computer.scheduler_type != _SCHEDULERS.get(scheduler, 'core.direct'):
        problems.append(f'scheduler {computer.scheduler_type!r} != '
                        f'{_SCHEDULERS.get(scheduler, "core.direct")!r}')
    if not problems:
        return None
    return (f'machine {name!r} is already registered with {", ".join(problems)}; '
            f'reusing it. To record these runs under another label, set '
            f'settings.aiida.machine (it overrides each run\'s hpc_host)')


VERSION_EXTRA = 'friction2d_lammps_version'
EXECUTABLE_SOURCE_EXTRA = 'friction2d_executable_source'


def _version_slug(version: str) -> str:
    """``29 Aug 2024 - Update 1`` -> ``29Aug2024_Update1``."""
    slug = re.sub(r'\s+-\s+', '_', version.strip())
    return re.sub(r'[^A-Za-z0-9._-]', '', slug)


def lammps_code(computer: orm.Computer, version: str,
                executable: Optional[str]) -> orm.InstalledCode:
    """The ``InstalledCode`` of LAMMPS ``version`` on ``computer``, one per
    (machine, version).

    Found by label ``lammps-<slug>@<computer>`` on that computer, else
    created. A second ``executable`` for an existing key is logged, not
    recorded. Not content-addressed (no uuid slot): the label is the key.
    """
    label = f'lammps-{_version_slug(version)}@{computer.label}'
    found = orm.QueryBuilder().append(
        orm.InstalledCode, filters={'label': label}, tag='c', project='*'
    ).append(orm.Computer, with_node='c', filters={'id': computer.pk}).first()
    if found:
        code = found[0]
        if executable and executable != code.filepath_executable.as_posix():
            logger.info('LAMMPS %s on %s: ignoring executable %r (recorded %r)',
                        version, computer.label, executable,
                        code.filepath_executable.as_posix())
        return code
    code = orm.InstalledCode(
        computer=computer, filepath_executable=executable or 'lmp', label=label,
        description=f'LAMMPS {version} on {computer.label}, created from a run log')
    code.base.extras.set_many({
        VERSION_EXTRA: version,
        EXECUTABLE_SOURCE_EXTRA: 'job_script' if executable else 'assumed'})
    return code.store()
