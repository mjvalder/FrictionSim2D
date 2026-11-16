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
    assert PROVENANCE_SCHEMA_VERSION == '2.2.0'
    rec = _record()
    assert rec.schema_version == '2.2.0'
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

    assert rec.schema_version == '2.2.0'
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


_RAMP_PRINT = ('print           "# RAMP_STATUS converged=1 r=${r} '
               'iterations=${ramp_iter} load=$(v_find)"\n')


def test_three_ramp_lines_give_three_entries(tmp_path):
    log = tmp_path / 'log.lammps'
    body = ''.join(
        _RAMP_PRINT + f"# RAMP_STATUS converged=1 r=0.0{i} iterations={i} load={ld}\n"
        for i, ld in ((1, 10), (2, 20), (3, 30)))
    log.write_text("LAMMPS (29 Aug 2024)\n" + body)
    up = parse_log_extras(log)
    assert up['ramp_by_load'] == {
        '10': {'converged': True, 'r': 0.01, 'iterations': 1},
        '20': {'converged': True, 'r': 0.02, 'iterations': 2},
        '30': {'converged': True, 'r': 0.03, 'iterations': 3}}
    assert up['ramp_iterations'] == 3
    assert up['ramp_r'] == pytest.approx(0.03)


def _old_log(tmp_path):
    log = tmp_path / 'old.lammps'
    log.write_text(''.join(
        f"# RAMP_STATUS converged=1 r=0.1 iterations={i}\n" for i in (1, 2, 3)))
    return log


def test_old_lines_pair_by_position(tmp_path):
    up = parse_log_extras(_old_log(tmp_path), loads=[10, 20, 30.0])
    assert list(up['ramp_by_load']) == ['10', '20', '30']
    assert up['ramp_by_load']['20']['iterations'] == 2


def test_old_lines_with_mismatched_load_count_are_not_paired(tmp_path):
    up = parse_log_extras(_old_log(tmp_path), loads=[10, 20])
    assert 'ramp_by_load' not in up
    assert up['ramp_iterations'] == 3


def test_parse_hold_files(tmp_path):
    from src.data.provenance import parse_hold_files
    hdr = ("# Time-averaged data for fix hold_ave\n"
           "# TimeStep v_hold_fz v_hold_tipz v_hold_shz\n")
    (tmp_path / 'hold_10N.txt').write_text(hdr + "1000 1.0 5 6\n2000 3.0 5 6\n")
    (tmp_path / 'hold_2.5N.txt').write_text(hdr + "1000 0.5 5 6\n")
    (tmp_path / 'hold_7N.txt').write_text("garbage\n")
    assert parse_hold_files(tmp_path) == {'10': 2.0, '2.5': 0.5}


def test_parse_hold_files_skips_a_partial_last_line(tmp_path):
    """A job killed mid-write leaves "2000" alone on the last line: that row
    is skipped, the complete rows still give the mean, nothing raises."""
    from src.data.provenance import parse_hold_files
    hdr = "# TimeStep v_hold_fz v_hold_tipz v_hold_shz\n"
    (tmp_path / 'hold_10N.txt').write_text(hdr + "1000 1.0 5 6\n2000")
    (tmp_path / 'hold_20N.txt').write_text(hdr + "2000\n")
    assert parse_hold_files(tmp_path) == {'10': 1.0}


# --- run identity (Task 5b, critique C3) ------------------------------------

def test_schema_version_is_2_2_0():
    assert PROVENANCE_SCHEMA_VERSION == '2.2.0'


def test_new_run_uuid_is_a_fresh_uuid4():
    from uuid import UUID
    from src.data.provenance import new_run_uuid
    first, second = new_run_uuid(), new_run_uuid()
    assert UUID(first).version == 4 and str(UUID(first)) == first
    assert first != second


def test_run_uuid_round_trips(tmp_path):
    rec = _record(run_uuid='3f0c2a7e-1b7d-4c4e-9a51-6d2f0e8b9c11')
    assert read_provenance(write_provenance(tmp_path, rec)).run_uuid == rec.run_uuid


def test_a_2_1_0_file_without_run_uuid_reads_as_none(tmp_path):
    path = write_provenance(tmp_path, _record())
    data = json.loads(path.read_text(encoding='utf-8'))
    data['schema_version'] = '2.1.0'
    data.pop('run_uuid', None)
    path.write_text(json.dumps(data), encoding='utf-8')

    rec = read_provenance(path)

    assert rec.run_uuid is None and rec.run_uuid_stamped_at is None


def test_an_unset_run_uuid_is_not_written(tmp_path):
    """No `"run_uuid": null` key: a legacy record re-written by enrichment
    stays one the stamper recognises by the key's absence."""
    data = json.loads(write_provenance(tmp_path, _record()).read_text(encoding='utf-8'))
    assert 'run_uuid' not in data and 'run_uuid_stamped_at' not in data


# --- legacy stamping -----------------------------------------------------------

def _legacy_file(path, **extra):
    """A 2.1.0 provenance.json as a pre-5b build wrote it (no run_uuid)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(_record().model_dump_json(exclude={'interlayer'}))
    data['schema_version'] = '2.1.0'
    data.pop('run_uuid', None)
    data.pop('run_uuid_stamped_at', None)
    data['drive_k'] = 8.0 / 16.02          # a float that must survive unchanged
    data.update(extra)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return path


def _afm_tree(root):
    base = root / 'afm' / 'h-MoS2' / '100x_100y' / 'sub_aSi_tip_Si_r25' / 'K300'
    return [_legacy_file(base / f'L{n}' / 'provenance.json') for n in (1, 2)]


def test_stamp_gives_each_legacy_record_its_own_uuid(tmp_path):
    import datetime
    from uuid import UUID
    from src.data.provenance import stamp_run_uuids
    files = _afm_tree(tmp_path)
    before = [json.loads(p.read_text(encoding='utf-8')) for p in files]

    report = stamp_run_uuids(tmp_path)

    assert sorted(report.stamped) == sorted(files)
    assert report.already == [] and report.unreadable == []
    after = [json.loads(p.read_text(encoding='utf-8')) for p in files]
    uuids = [a['run_uuid'] for a in after]
    assert all(UUID(u).version == 4 for u in uuids) and uuids[0] != uuids[1]
    for old, new in zip(before, after):
        assert new['run_uuid_stamped_at'] == datetime.date.today().isoformat()
        assert {k: v for k, v in new.items()
                if k not in ('run_uuid', 'run_uuid_stamped_at')} == old
        assert new['schema_version'] == '2.1.0'
    assert read_provenance(files[0]).run_uuid == uuids[0]


def test_stamp_is_idempotent(tmp_path):
    from src.data.provenance import stamp_run_uuids
    files = _afm_tree(tmp_path)
    stamp_run_uuids(tmp_path)
    texts = [p.read_bytes() for p in files]

    again = stamp_run_uuids(tmp_path)

    assert again.stamped == [] and sorted(again.already) == sorted(files)
    assert [p.read_bytes() for p in files] == texts


def test_stamp_never_overwrites_an_existing_uuid(tmp_path):
    from src.data.provenance import stamp_run_uuids
    path = write_provenance(tmp_path / 'run',
                            _record(run_uuid='3f0c2a7e-1b7d-4c4e-9a51-6d2f0e8b9c11'))
    text = path.read_bytes()

    report = stamp_run_uuids(tmp_path)

    assert report.already == [path] and report.stamped == []
    assert path.read_bytes() == text


def test_stamp_dry_run_writes_nothing(tmp_path):
    from src.data.provenance import stamp_run_uuids
    files = _afm_tree(tmp_path)
    texts = [p.read_bytes() for p in files]

    report = stamp_run_uuids(tmp_path, dry_run=True)

    assert sorted(report.stamped) == sorted(files)
    assert [p.read_bytes() for p in files] == texts
    assert sorted(p.name for p in files[0].parent.iterdir()) == ['provenance.json']


def test_stamp_reports_an_unreadable_file(tmp_path):
    from src.data.provenance import stamp_run_uuids
    good = _legacy_file(tmp_path / 'a' / 'provenance.json')
    bad = tmp_path / 'b' / 'provenance.json'
    bad.parent.mkdir()
    bad.write_text('{"truncated": ', encoding='utf-8')
    listed = tmp_path / 'c' / 'provenance.json'
    listed.parent.mkdir()
    listed.write_text('[1, 2]', encoding='utf-8')

    report = stamp_run_uuids(tmp_path)

    assert report.stamped == [good]
    assert sorted(report.unreadable) == sorted([bad, listed])
    assert bad.read_text(encoding='utf-8') == '{"truncated": '


def test_stamp_keeps_the_file_mode(tmp_path):
    from src.data.provenance import stamp_run_uuids
    path = _legacy_file(tmp_path / 'provenance.json')
    path.chmod(0o644)
    stamp_run_uuids(tmp_path)
    assert path.stat().st_mode & 0o777 == 0o644


def test_stamp_lists_a_non_uuid_run_uuid_as_invalid(tmp_path):
    from src.data.provenance import stamp_run_uuids
    path = write_provenance(tmp_path / 'run', _record(run_uuid='keep-me'))
    text = path.read_bytes()

    report = stamp_run_uuids(tmp_path)

    assert report.invalid == [path]
    assert report.stamped == [] and report.already == []
    assert path.read_bytes() == text
    assert 'invalid run_uuid: ' + str(path) in report.render()


def test_stamp_treats_a_null_run_uuid_as_missing(tmp_path):
    from uuid import UUID
    from src.data.provenance import stamp_run_uuids
    path = _legacy_file(tmp_path / 'provenance.json', run_uuid=None)

    report = stamp_run_uuids(tmp_path)

    assert report.stamped == [path]
    assert UUID(json.loads(path.read_text(encoding='utf-8'))['run_uuid']).version == 4


def test_a_failed_rename_leaves_the_original_and_no_temp_file(tmp_path, monkeypatch):
    import os
    from src.data.provenance import stamp_run_uuids
    path = _legacy_file(tmp_path / 'provenance.json')
    text = path.read_bytes()

    def boom(*_a, **_k):
        raise OSError('disk full')
    monkeypatch.setattr(os, 'replace', boom)

    report = stamp_run_uuids(tmp_path)

    assert report.unreadable == [path] and report.stamped == []
    assert path.read_bytes() == text
    assert sorted(p.name for p in tmp_path.iterdir()) == ['provenance.json']


def test_render_says_would_stamp_on_a_dry_run():
    from pathlib import Path
    from src.data.provenance import StampReport
    report = StampReport(stamped=[Path('a')])
    assert report.render(dry_run=True).startswith('would stamp 1,')
    assert report.render().startswith('stamped 1,')
