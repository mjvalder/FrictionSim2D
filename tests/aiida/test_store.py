"""Content-addressed parameter blocks and files in the AiiDA store (spec §5.2)."""
import hashlib
import io
from pathlib import Path

import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from frictionsim2d.aiida import store  # noqa: E402
from frictionsim2d.aiida.blocks import block_hash  # noqa: E402
from frictionsim2d.data.provenance import sha256_file  # noqa: E402
from frictionsim2d.aiida.store import (  # noqa: E402
    FRICTION2D_NAMESPACE, StoreIndex, get_or_create_block, get_or_create_file,
    get_or_create_provenance, get_or_create_schema, machine_computer, node_uuid,
    reindex, schema_hash, verify,
)

THERMOSTAT = {'kind': 'thermostat', 'block_schema': '1.0.0',
              'fields': {'settings.thermostat.type': 'nose-hoover',
                         'general.temp': 300.0}}


def test_same_block_reused(aiida_profile_clean):            # Review Focus 2
    t = THERMOSTAT
    first = get_or_create_block(t)
    assert first.is_stored
    assert first.pk == get_or_create_block(dict(t)).pk
    assert first.base.extras.get('friction2d_block_hash') == block_hash(t)
    assert first.base.extras.get('friction2d_block_kind') == 'thermostat'
    assert first.get_dict() == t


def test_different_content_or_kind_is_a_new_block(aiida_profile_clean):
    base = get_or_create_block(THERMOSTAT)
    warmer = get_or_create_block({**THERMOSTAT, 'fields': {'general.temp': 400.0}})
    other_kind = get_or_create_block({**THERMOSTAT, 'kind': 'drive'})
    assert len({base.pk, warmer.pk, other_kind.pk}) == 3


def test_an_unknown_kind_is_refused(aiida_profile_clean):
    with pytest.raises(ValueError, match='kind'):
        get_or_create_block({**THERMOSTAT, 'kind': 'bogus'})


def test_one_material_in_two_roles_is_one_node(aiida_profile_clean):
    """The role is the link, not the content."""
    content = {'kind': 'material', 'block_schema': '1.0.0',
               'fields': {'name': 'Si', 'cif_sha256': 'ab' * 32}}
    tip, sub = get_or_create_block(content), get_or_create_block(dict(content))
    assert tip.pk == sub.pk
    assert tip.base.extras.get('friction2d_block_kind') == 'material'


def test_potential_file_deduplicated(aiida_profile_clean, tmp_path):
    a = tmp_path / 'a' / 'MoS.sw'
    b = tmp_path / 'b' / 'copy_of_MoS.sw'
    for path in (a, b):
        path.parent.mkdir()
        path.write_text('Mo S S 1.0 2.0\n', encoding='utf-8')
    first = get_or_create_file(a)
    assert isinstance(first, orm.SinglefileData) and first.is_stored
    assert first.pk == get_or_create_file(b).pk
    assert first.get_content() == 'Mo S S 1.0 2.0\n'
    assert first.base.extras.get('friction2d_sha256') == sha256_file(a)

    c = tmp_path / 'other.sw'
    c.write_text('W S S 1.0 2.0\n', encoding='utf-8')
    assert get_or_create_file(c).pk != first.pk


def test_stored_file_goes_through_redaction(aiida_profile_clean, tmp_path):
    settings = tmp_path / 'settings.yaml'
    settings.write_text('database:\n  password: hunter2\n', encoding='utf-8')
    node = get_or_create_file(settings)
    assert 'hunter2' not in node.get_content()


def test_file_hash_covers_the_stored_redacted_bytes(aiida_profile_clean, tmp_path):
    """M2: the published hash is checkable against the published content."""
    settings = tmp_path / 'settings.yaml'
    settings.write_text('password: x\n', encoding='utf-8')
    node = get_or_create_file(settings)
    stored = node.get_content(mode='rb')
    assert b'x' not in stored
    digest = node.base.extras.get('friction2d_sha256')
    assert digest == hashlib.sha256(stored).hexdigest()
    assert digest != sha256_file(settings)
    assert node.uuid == node_uuid('file', digest)
    # a second file redacting to the same bytes is the same node
    other = tmp_path / 'b' / 'settings.yaml'
    other.parent.mkdir()
    other.write_text('password: y\n', encoding='utf-8')
    assert get_or_create_file(other).pk == node.pk


def test_provenance_key_covers_the_staged_bytes(aiida_profile_clean, tmp_path):
    """Two runs whose provenance differs only by a redacted secret share a node."""
    def run(secret):
        root = tmp_path / secret
        prov = root / 'provenance'
        prov.mkdir(parents=True)
        (prov / 'settings.yaml').write_text(f'database:\n  password: {secret}\n',
                                            encoding='utf-8')
        (root / 'provenance.json').write_text('{"run_uuid": "x"}', encoding='utf-8')
        return get_or_create_provenance(root / 'provenance.json', prov)

    first, second = run('hunter2'), run('swordfish')
    assert first.pk == second.pk
    assert first.uuid == node_uuid('folder', first.base.extras.get('friction2d_folder_sha256'))
    assert 'hunter2' not in first.base.repository.get_object_content(
        'provenance/settings.yaml')


def test_provenance_location_triggers_redaction(aiida_profile_clean, tmp_path):
    """A file under provenance/ is checked whatever its name (redact's location rule)."""
    prov = tmp_path / 'run' / 'provenance'
    prov.mkdir(parents=True)
    notes = prov / 'notes.yaml'
    notes.write_text('central:\n  password: hunter2\n', encoding='utf-8')
    node = get_or_create_file(notes)
    assert node.filename == 'notes.yaml'
    assert 'hunter2' not in node.get_content()


def test_explicit_name_and_cif_exemption(aiida_profile_clean, tmp_path):
    cif = tmp_path / 'run' / 'provenance' / 'cif' / 'MoS2.cif'
    cif.parent.mkdir(parents=True)
    cif.write_text('data_MoS2\n_cell_length_a 3.16\n', encoding='utf-8')
    node = get_or_create_file(cif)
    assert node.get_content() == 'data_MoS2\n_cell_length_a 3.16\n'
    other = tmp_path / 'loose.txt'
    other.write_text('password: hunter2\n', encoding='utf-8')
    named = get_or_create_file(other, name='provenance/loose.txt')
    assert named.filename == 'loose.txt'
    assert 'hunter2' not in named.get_content()


def test_machine_computer_get_or_create(aiida_profile_clean):
    hpc = {'hpc_host': 'login.archer2.ac.uk', 'scheduler_type': 'slurm'}
    comp = machine_computer('archer2', hpc)
    assert comp.label == 'archer2'
    assert comp.hostname == 'login.archer2.ac.uk'
    assert comp.transport_type == 'core.ssh'
    assert comp.scheduler_type == 'core.slurm'
    assert comp.get_workdir() == '/unused/friction2d-import'
    assert not comp.is_configured
    assert machine_computer('archer2', {}).pk == comp.pk


def test_machine_mismatch_names_a_host_or_scheduler_that_disagrees(aiida_profile_clean):
    from frictionsim2d.aiida.store import machine_mismatch

    assert machine_mismatch('nowhere', {'hpc_host': 'a'}) is None     # no such Computer
    machine_computer('archer2', {'hpc_host': 'login.archer2.ac.uk', 'scheduler_type': 'slurm'})

    assert machine_mismatch('archer2', {}) is None                    # states nothing
    assert machine_mismatch('archer2', {'hpc_host': 'login.archer2.ac.uk',
                                        'scheduler_type': 'slurm'}) is None
    message = machine_mismatch('archer2', {'hpc_host': 'other.host', 'scheduler_type': 'pbs'})
    assert "hostname 'login.archer2.ac.uk' != 'other.host'" in message
    assert "scheduler 'core.slurm' != 'core.pbspro'" in message
    assert 'settings.aiida.machine' in message


def test_machine_computer_defaults(aiida_profile_clean):
    comp = machine_computer('localhost', {})
    assert (comp.hostname, comp.transport_type, comp.scheduler_type) == (
        'localhost', 'core.local', 'core.direct')
    assert machine_computer('cx3', {'scheduler_type': 'pbs'}).scheduler_type == 'core.pbspro'


def test_lammps_code_is_keyed_by_machine_and_version(aiida_profile_clean):
    from frictionsim2d.aiida.store import lammps_code
    comp = machine_computer('archer2', {})
    code = lammps_code(comp, '29 Aug 2024 - Update 1', '/opt/bin/lmp_mpi')
    assert code.label == 'lammps-29Aug2024_Update1@archer2'
    assert lammps_code(comp, '29 Aug 2024 - Update 1', 'lmp_other').pk == code.pk
    assert str(code.filepath_executable) == '/opt/bin/lmp_mpi'
    assert lammps_code(comp, '2 Aug 2023', None).pk != code.pk
    assert lammps_code(machine_computer('cx3', {}), '2 Aug 2023', None).label == \
        'lammps-2Aug2023@cx3'


# --- deterministic uuids (critique C2) ------------------------------------------

def test_node_uuid_is_uuid5_of_slot_and_hash():
    """Frozen: the namespace, the slot strings and the name format."""
    from uuid import UUID, uuid5
    digest = block_hash(THERMOSTAT)
    assert node_uuid('block', digest) == str(uuid5(FRICTION2D_NAMESPACE, f'block:{digest}'))
    assert FRICTION2D_NAMESPACE == UUID('e1ff2212-fbb6-5980-a513-7e1d9d76aa12')
    assert store.UUID_SLOTS == ('block', 'file', 'folder', 'schema')
    assert node_uuid('file', '0' * 64) == 'f15ec894-3e6e-5f5b-bc38-bb58c1d206af'
    assert len({node_uuid(s, digest) for s in store.UUID_SLOTS}) == 4
    with pytest.raises(ValueError, match='slot'):
        node_uuid('blocks', digest)


def test_a_file_holding_a_blocks_json_is_not_that_block(aiida_profile_clean, tmp_path):
    import json
    from frictionsim2d.aiida.blocks import _normalise
    payload = json.dumps(_normalise(THERMOSTAT), sort_keys=True, separators=(',', ':'))
    path = tmp_path / 'block.json'
    path.write_text(payload, encoding='utf-8')
    block, file = get_or_create_block(THERMOSTAT), get_or_create_file(path)
    assert file.base.extras.get('friction2d_sha256') == block_hash(THERMOSTAT)
    assert block.uuid == node_uuid('block', block_hash(THERMOSTAT))
    assert file.uuid == node_uuid('file', block_hash(THERMOSTAT)) != block.uuid


def test_block_uuid_is_the_same_in_two_fresh_profiles(aiida_profile_clean):
    first = get_or_create_block(THERMOSTAT).uuid
    assert first == node_uuid('block', block_hash(THERMOSTAT))
    aiida_profile_clean.reset_storage()
    assert orm.QueryBuilder().append(orm.Dict).count() == 0
    second = get_or_create_block(dict(THERMOSTAT))
    assert second.uuid == first


def test_schema_node_is_content_addressed(aiida_profile_clean):
    from frictionsim2d.aiida.block_schema import schema_document
    doc = schema_document('afm')
    node = get_or_create_schema(doc)
    assert node.get_dict() == doc
    assert node.base.extras.get('friction2d_schema_hash') == schema_hash(doc)
    assert node.uuid == node_uuid('schema', schema_hash(doc))
    assert get_or_create_schema(schema_document('afm')).pk == node.pk
    assert get_or_create_schema(schema_document('sheetonsheet')).pk != node.pk


def _export(node, path):
    from aiida.tools.archive import create_archive
    create_archive([node], filename=path)


def _same_hash(digest):
    return orm.QueryBuilder().append(
        orm.Dict, filters={'extras.friction2d_block_hash': digest}).all(flat=True)


#: aiida-core's own warning, not ours: ``_make_import_group``
#: (aiida/tools/archive/imports.py:1143) builds an unstored
#: ``orm.ImportGroup(label, backend=backend_to)`` only to read its type string
#: and user; the user relationship's ``DbUser.dbgroups`` backref attaches that
#: unsaved DbGroup, and the ``bulk_insert`` on the next line
#: (storage/psql_dos/backend.py:338) autoflushes and warns that it will not
#: add it. Raised by every ``import_archive`` that creates an import group.
_IMPORT_GROUP_WARNING = pytest.mark.filterwarnings(
    'ignore:Object of type <DbGroup> not in session:sqlalchemy.exc.SAWarning')


@_IMPORT_GROUP_WARNING
def test_an_imported_block_is_the_local_node(aiida_profile_clean, tmp_path):
    """Review Focus 3: A's block imported into B, where it already exists, is one node."""
    from aiida.tools.archive import import_archive
    digest = block_hash(THERMOSTAT)
    _export(get_or_create_block(THERMOSTAT), tmp_path / 'a.aiida')     # profile A
    aiida_profile_clean.reset_storage()                                 # profile B
    local = get_or_create_block(dict(THERMOSTAT))
    import_archive(tmp_path / 'a.aiida')
    assert [n.pk for n in _same_hash(digest)] == [local.pk]
    assert verify().duplicate_hashes == {}


@_IMPORT_GROUP_WARNING
def test_random_uuids_duplicate_and_verify_says_so(aiida_profile_clean, tmp_path,
                                                   monkeypatch):
    """Without _set_uuid, B ends with two nodes; the hash-value join still finds both."""
    from aiida.tools.archive import import_archive
    monkeypatch.setattr(store, '_set_uuid', lambda node, value: None)
    digest = block_hash(THERMOSTAT)
    _export(get_or_create_block(THERMOSTAT), tmp_path / 'a.aiida')
    aiida_profile_clean.reset_storage()
    local = get_or_create_block(dict(THERMOSTAT))
    import_archive(tmp_path / 'a.aiida')
    both = _same_hash(digest)
    assert len(both) == 2 and local.pk in {n.pk for n in both}
    assert sorted(verify().duplicate_hashes[digest]) == sorted(n.uuid for n in both)
    assert 'duplicate' in verify().render()


def test_set_uuid_without_the_private_api_warns_once(aiida_profile_clean, caplog,
                                                    monkeypatch):
    class Bare:
        backend_entity = object()
    monkeypatch.setattr(store, '_warned_uuid', False)
    with caplog.at_level('WARNING', logger='frictionsim2d.aiida.store'):
        store._set_uuid(Bare(), node_uuid('block', 'a'))
        store._set_uuid(Bare(), node_uuid('block', 'b'))
    assert len([r for r in caplog.records if 'deterministic' in r.message]) == 1


def test_a_block_whose_extras_were_dropped_is_found_by_uuid(aiida_profile_clean):
    node = get_or_create_block(THERMOSTAT)
    node.base.extras.clear()
    again = get_or_create_block(dict(THERMOSTAT))
    assert again.pk == node.pk
    assert again.base.extras.get('friction2d_block_hash') == block_hash(THERMOSTAT)


def test_a_file_and_a_schema_whose_extras_were_dropped_are_found_by_uuid(
        aiida_profile_clean, tmp_path):
    pot = tmp_path / 'MoS.sw'
    pot.write_text('# potential\n', encoding='utf-8')
    file = get_or_create_file(pot)
    schema = get_or_create_schema({'block_schema': '1', 'family': 'afm', 'fields': {}})
    file_sha = file.base.extras.get('friction2d_sha256')
    schema_sha = schema.base.extras.get('friction2d_schema_hash')
    for node in (file, schema):
        node.base.extras.clear()

    assert get_or_create_file(pot).pk == file.pk
    assert get_or_create_schema({'block_schema': '1', 'family': 'afm', 'fields': {}}).pk == schema.pk
    assert file.base.extras.get('friction2d_sha256') == file_sha
    assert schema.base.extras.get('friction2d_schema_hash') == schema_sha


def _stored_under(node, uuid, extras=None):
    if extras:
        node.base.extras.set_many(extras)
    store._set_uuid(node, uuid)
    return node.store()


def test_a_uuid_twin_with_other_content_is_not_labelled(aiida_profile_clean, tmp_path,
                                                        caplog):
    digest = block_hash(THERMOSTAT)
    impostor = _stored_under(orm.Dict({'kind': 'thermostat', 'fields': {}}),
                             node_uuid('block', digest))
    with caplog.at_level('WARNING', logger='frictionsim2d.aiida.store'):
        node = get_or_create_block(THERMOSTAT)
    assert node.pk != impostor.pk and node.get_dict() == THERMOSTAT
    assert node.uuid != node_uuid('block', digest)               # random fallback
    assert 'friction2d_block_hash' not in impostor.base.extras.all
    assert any('does not match' in r.message for r in caplog.records)

    pot = tmp_path / 'MoS.sw'
    pot.write_text('sw\n', encoding='utf-8')
    fake = orm.SinglefileData(io.BytesIO(b'other\n'), filename='MoS.sw')
    fake = _stored_under(fake, node_uuid('file', sha256_file(pot)))
    real = get_or_create_file(pot)
    assert real.pk != fake.pk and real.get_content() == 'sw\n'
    assert 'friction2d_sha256' not in fake.base.extras.all


def test_the_deterministic_node_wins_over_an_older_duplicate(aiida_profile_clean,
                                                             monkeypatch):
    digest = block_hash(THERMOSTAT)
    extras = {'friction2d_block_hash': digest, 'friction2d_block_kind': 'thermostat'}
    older = orm.Dict(dict(THERMOSTAT))
    older.base.extras.set_many(extras)
    older.store()                                                 # random uuid
    canonical = _stored_under(orm.Dict(dict(THERMOSTAT)), node_uuid('block', digest),
                              extras)
    assert older.pk < canonical.pk
    assert get_or_create_block(dict(THERMOSTAT)).pk == canonical.pk
    assert StoreIndex.load().blocks[digest] == canonical.pk
    assert get_or_create_block(dict(THERMOSTAT), index=StoreIndex.load()).pk == canonical.pk


# --- the memo, reindex and verify ------------------------------------------------

def test_store_index_memoises_lookups(aiida_profile_clean, tmp_path, monkeypatch):
    pot = tmp_path / 'MoS.sw'
    pot.write_text('sw\n', encoding='utf-8')
    block, file = get_or_create_block(THERMOSTAT), get_or_create_file(pot)

    calls = []
    real = orm.QueryBuilder

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(orm, 'QueryBuilder', counting)
    index = StoreIndex.load()
    assert len(calls) <= 4
    assert index.blocks == {block_hash(THERMOSTAT): block.pk}
    assert index.files == {file.base.extras.get('friction2d_sha256'): file.pk}
    calls.clear()
    assert get_or_create_block(dict(THERMOSTAT), index=index).pk == block.pk
    assert get_or_create_file(pot, index=index).pk == file.pk
    assert calls == []                          # a memo hit runs no hash lookup
    new = get_or_create_block({**THERMOSTAT, 'kind': 'drive'}, index=index)
    assert index.blocks[new.base.extras.get('friction2d_block_hash')] == new.pk


def test_reindex_restores_cleared_extras(aiida_profile_clean, tmp_path):
    node = get_or_create_block(THERMOSTAT)
    settings = tmp_path / 'settings.yaml'
    settings.write_text('password: x\n', encoding='utf-8')
    file = get_or_create_file(settings)
    file_sha = file.base.extras.get('friction2d_sha256')
    schema = get_or_create_schema({'block_schema': '1', 'family': 'afm', 'fields': {}})
    for n in (node, schema):
        n.base.extras.clear()
    # a file is known as friction2d's by its extra (or a CalcJob link): edit it
    file.base.extras.set('friction2d_sha256', '0' * 64)
    assert reindex() == 3
    assert node.base.extras.get('friction2d_block_hash') == block_hash(node.get_dict())
    assert node.base.extras.get('friction2d_block_kind') == 'thermostat'
    assert file.base.extras.get('friction2d_sha256') == file_sha
    assert schema.base.extras.get('friction2d_schema_hash') == schema_hash(schema.get_dict())
    assert reindex() == 0


def test_verify_names_a_mismatched_hash(aiida_profile_clean):
    node = get_or_create_block(THERMOSTAT)
    assert verify().ok
    node.base.extras.set('friction2d_block_hash', '0' * 64)
    report = verify()
    assert any(node.uuid in line for line in report.mismatches)
    assert node.uuid in report.render()
    reindex()
    assert verify().ok
