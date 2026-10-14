"""Provenance record tests (spec §5 F5, Task 8)."""
import hashlib
import json

import pydantic
import pytest

from src.data.provenance import (
    BuiltGeometry, MATERIAL_ROLES, MaterialProvenance, PROVENANCE_SCHEMA_VERSION,
    ProvenanceRecord, parse_log_extras, read_provenance, sha256_file,
    write_provenance,
)


def _record(**kw):
    d = dict(family='afm', model='afm',
             materials=[MaterialProvenance(
                 name='h-MoS2', role='sheet', pot_path='/p/MoS.sw', pot_type='sw',
                 pot_resolution_route='explicit', pot_file_sha256='ab' * 32)],
             seed_thermal_tip=1, seed_thermal_sub=2, seed_velocity=3,
             seed_sheet_center=None,
             substrate_realization_id='abc', damping_ratio=0.0072, damp_ev=None,
             git_commit='deadbeef', git_dirty=False, config_hash='ff' * 32,
             layer_type_map={'layer_1': [7, 8, 9], 'layer_2': [10, 11, 12]},
             unloaded_reference_z=None, contact_cutoff=4.8,
             cutoff_source='uff_sigma_k1.5',
             thermostat_scheme='langevin_y_tip_sub_nve_sheet',
             tip_group_sizes={'tip_all': 495, 'tip_fix': 191,
                              'tip_thermo': 150},
             run_role='label', load_requested=30.0)
    d.update(kw)
    return ProvenanceRecord(**d)


def test_roundtrip(tmp_path):
    p = write_provenance(tmp_path, _record())
    data = json.loads(p.read_text())
    assert data['seed_thermal_tip'] == 1
    assert data['schema_version']


def test_missing_required_field_fails_build():
    with pytest.raises(pydantic.ValidationError):
        ProvenanceRecord(seed_thermal_tip=1)  # everything else missing


def test_run_role_enum():
    with pytest.raises(pydantic.ValidationError):
        _record(run_role='banana')


def test_checksums(tmp_path):
    f = tmp_path / 'x.data'
    f.write_text('hello')
    assert sha256_file(f) == hashlib.sha256(b'hello').hexdigest()


def test_parse_log_extras(tmp_path):
    log = tmp_path / 'log.lammps'
    log.write_text(
        "LAMMPS (29 Aug 2024)\n"
        "# UNLOADED_REF layer=1 z=16.02\n"
        "# UNLOADED_REF layer=2 z=22.31\n"
        "# RAMP_STATUS converged=0 r=0.31 iterations=40\n"
        "# RAMP_STATUS converged=1 r=0.084 iterations=7\n"
    )
    up = parse_log_extras(log)
    assert up['ramp_converged'] is True        # last RAMP_STATUS wins
    assert up['ramp_r'] == pytest.approx(0.084)
    assert up['ramp_iterations'] == 7
    assert up['unloaded_reference_z'] == {'layer_1': 16.02, 'layer_2': 22.31}
    assert up['lammps_version'] == '29 Aug 2024'


def test_schema_is_two_and_materials_replace_the_single_triple():
    assert PROVENANCE_SCHEMA_VERSION == '2.1.0'
    rec = _record()
    assert rec.schema_version == '2.1.0'
    assert not hasattr(rec, 'pot_path')
    assert [m.role for m in rec.materials] == ['sheet']


def test_every_component_gets_its_own_entry():
    rec = _record(materials=[
        MaterialProvenance(name='h-MoS2', role='sheet', pot_path='/p/MoS.sw',
                           pot_type='sw', pot_resolution_route='explicit',
                           pot_file_sha256='ab' * 32),
        MaterialProvenance(name='h-WS2', role='sheet', pot_path='/p/WS.sw',
                           pot_type='sw'),
        MaterialProvenance(name='h-WSe2', role='flake'),
        MaterialProvenance(name='Si', role='tip'),
        MaterialProvenance(name='aSi', role='substrate'),
    ])

    assert {m.role for m in rec.materials} <= set(MATERIAL_ROLES)
    assert [m.name for m in rec.materials if m.role == 'sheet'] == ['h-MoS2', 'h-WS2']
    assert rec.materials[0].pot_file_sha256 == 'ab' * 32


def test_reader_upgrades_a_one_x_file(tmp_path):
    legacy = {
        'schema_version': '1.0.0',
        'seed_thermal_tip': 1, 'seed_thermal_sub': 2, 'seed_velocity': 3,
        'substrate_realization_id': 'abc', 'damping_ratio': 0.0072, 'damp_ev': None,
        'git_commit': 'deadbeef', 'git_dirty': False, 'config_hash': 'ff' * 32,
        'layer_type_map': {'layer_1': [7, 8, 9]}, 'contact_cutoff': 4.8,
        'cutoff_source': 'uff_sigma_k1.5',
        'thermostat_scheme': 'langevin_y_tip_sub_nve_sheet',
        'run_role': 'label', 'load_requested': 30.0,
        'pot_path': '/p/MoS.sw', 'pot_type': 'sw',
        'pot_resolution_route': 'explicit', 'pot_file_sha256': 'cd' * 32,
    }
    path = tmp_path / 'provenance.json'
    path.write_text(json.dumps(legacy), encoding='utf-8')

    rec = read_provenance(path, family='hetero', model='sheetonsheet',
                          material_name='h-MoS2')

    assert rec.schema_version == '2.1.0'
    assert rec.family == 'hetero'
    assert rec.model == 'sheetonsheet'
    assert len(rec.materials) == 1
    assert rec.materials[0] == MaterialProvenance(
        name='h-MoS2', role='sheet', pot_path='/p/MoS.sw', pot_type='sw',
        pot_resolution_route='explicit', pot_file_sha256='cd' * 32)
    assert rec.contact_cutoff == 4.8


def test_reader_leaves_a_two_x_file_alone(tmp_path):
    path = write_provenance(tmp_path, _record())

    rec = read_provenance(path, family='afm', model='afm')

    assert rec == _record()


def test_builder_writes_provenance(tmp_path):
    from src.builders.afm import AFMSimulation
    from tests.helpers.afm import (_make_afm_config,
                                    _prepare_builder_for_slide_write)
    config = _make_afm_config(tmp_path, layers=[1])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)
    prov = tmp_path / "out" / "L1" / "provenance.json"
    assert prov.is_file()
    data = json.loads(prov.read_text())
    assert data['contact_cutoff'] > 0
    assert data['run_role'] == 'label'
    assert data['thermostat_scheme'] == 'langevin_y_tip_sub_nve_sheet'


def test_new_fields_default_to_none_and_round_trip(tmp_path):
    bare = _record()
    assert bare.drive_k is None and bare.geometry is None
    assert bare.ramp_by_load is None and bare.load_achieved_by_load is None

    rec = _record(
        geometry=BuiltGeometry(lat_c=6.15, lat_c_source='minimised',
                               cell_a=3.19),
        drive_k=0.4994, drive_rest_length=37.5,
        ramp_by_load={'10': {'converged': True, 'r': 0.01, 'iterations': 4}},
        load_achieved_by_load={'10': 9.8})
    path = write_provenance(tmp_path, rec)

    assert read_provenance(path) == rec


def test_a_2_0_record_without_the_new_fields_still_reads(tmp_path):
    path = write_provenance(tmp_path, _record())
    data = json.loads(path.read_text(encoding='utf-8'))
    data['schema_version'] = '2.0.0'
    for key in ('drive_k', 'drive_rest_length', 'geometry', 'ramp_by_load',
                'load_achieved_by_load'):
        data.pop(key)
    path.write_text(json.dumps(data), encoding='utf-8')

    rec = read_provenance(path)

    assert rec.drive_k is None and rec.drive_rest_length is None
    assert rec.geometry is None and rec.ramp_by_load is None
    assert rec.load_achieved_by_load is None


def test_lat_c_source_vocabulary_is_closed():
    with pytest.raises(pydantic.ValidationError):
        BuiltGeometry(lat_c_source='guess')
