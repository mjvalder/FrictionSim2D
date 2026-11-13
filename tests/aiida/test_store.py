"""Content-addressed parameter blocks and files in the AiiDA store (spec §5.2)."""
from pathlib import Path

import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from src.aiida.blocks import block_hash  # noqa: E402
from src.aiida.store import (  # noqa: E402
    get_or_create_block, get_or_create_file, machine_computer,
)

THERMOSTAT = {'settings': {'thermostat': {'type': 'langevin'}},
              'general': {'temp': 300.0}}


def test_same_block_reused(aiida_profile_clean):            # Review Focus 2
    t = THERMOSTAT
    first = get_or_create_block('thermostat', t)
    assert first.is_stored
    assert first.pk == get_or_create_block('thermostat', dict(t)).pk
    assert first.base.extras.get('friction2d_block_hash') == block_hash('thermostat', t)
    assert first.base.extras.get('friction2d_block_kind') == 'thermostat'
    assert first.get_dict() == t


def test_different_content_or_kind_is_a_new_block(aiida_profile_clean):
    base = get_or_create_block('thermostat', THERMOSTAT)
    warmer = get_or_create_block(
        'thermostat', {**THERMOSTAT, 'general': {'temp': 400.0}})
    other_kind = get_or_create_block('drive', THERMOSTAT)
    assert len({base.pk, warmer.pk, other_kind.pk}) == 3


def test_block_name_resolves_to_its_kind(aiida_profile_clean):
    """Two roles of one material share the node: the role is the link, not the content."""
    content = {'name': 'Si', 'cif_sha256': 'ab' * 32}
    tip = get_or_create_block('material__tip_0', content)
    sub = get_or_create_block('material__substrate_0', content)
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
    assert len(first.base.extras.get('friction2d_sha256')) == 64

    c = tmp_path / 'other.sw'
    c.write_text('W S S 1.0 2.0\n', encoding='utf-8')
    assert get_or_create_file(c).pk != first.pk


def test_stored_file_goes_through_redaction(aiida_profile_clean, tmp_path):
    settings = tmp_path / 'settings.yaml'
    settings.write_text('database:\n  password: hunter2\n', encoding='utf-8')
    node = get_or_create_file(settings)
    assert 'hunter2' not in node.get_content()


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
