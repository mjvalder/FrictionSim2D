"""Content-addressed parameter blocks and files in the AiiDA store (spec §5.2)."""
import hashlib
from pathlib import Path

import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from src.aiida import store  # noqa: E402
from src.aiida.blocks import block_hash  # noqa: E402
from src.data.provenance import sha256_file  # noqa: E402
from src.aiida.store import (  # noqa: E402
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
    assert node.uuid == node_uuid(digest)
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
    assert first.uuid == node_uuid(first.base.extras.get('friction2d_folder_sha256'))
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
    comp = machine_computer('archer2')
    assert comp.label == 'archer2'
    assert comp.transport_type == 'core.local'
    assert comp.scheduler_type == 'core.direct'
    assert comp.get_workdir() == str(Path('~/aiida_workdir/archer2').expanduser())
    assert comp.is_configured
    assert machine_computer('archer2').pk == comp.pk


# --- deterministic uuids (critique C2) ------------------------------------------

def test_node_uuid_is_uuid5_of_the_hash():
    from uuid import UUID, uuid5
    digest = block_hash(THERMOSTAT)
    assert node_uuid(digest) == str(uuid5(FRICTION2D_NAMESPACE, digest))
    assert FRICTION2D_NAMESPACE == UUID('e1ff2212-fbb6-5980-a513-7e1d9d76aa12')


def test_block_uuid_is_the_same_in_two_fresh_profiles(aiida_profile_clean):
    first = get_or_create_block(THERMOSTAT).uuid
    assert first == node_uuid(block_hash(THERMOSTAT))
    aiida_profile_clean.reset_storage()
    assert orm.QueryBuilder().append(orm.Dict).count() == 0
    second = get_or_create_block(dict(THERMOSTAT))
    assert second.uuid == first


def test_schema_node_is_content_addressed(aiida_profile_clean):
    from src.aiida.block_schema import schema_document
    doc = schema_document('afm')
    node = get_or_create_schema(doc)
    assert node.get_dict() == doc
    assert node.base.extras.get('friction2d_schema_hash') == schema_hash(doc)
    assert node.uuid == node_uuid(schema_hash(doc))
    assert get_or_create_schema(schema_document('afm')).pk == node.pk
    assert get_or_create_schema(schema_document('sheetonsheet')).pk != node.pk


def _export(node, path):
    from aiida.tools.archive import create_archive
    create_archive([node], filename=path)


def _same_hash(digest):
    return orm.QueryBuilder().append(
        orm.Dict, filters={'extras.friction2d_block_hash': digest}).all(flat=True)


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
    with caplog.at_level('WARNING', logger='src.aiida.store'):
        store._set_uuid(Bare(), node_uuid('a'))
        store._set_uuid(Bare(), node_uuid('b'))
    assert len([r for r in caplog.records if 'deterministic' in r.message]) == 1


def test_a_block_whose_extras_were_dropped_is_found_by_uuid(aiida_profile_clean):
    node = get_or_create_block(THERMOSTAT)
    node.base.extras.clear()
    again = get_or_create_block(dict(THERMOSTAT))
    assert again.pk == node.pk
    assert again.base.extras.get('friction2d_block_hash') == block_hash(THERMOSTAT)


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
