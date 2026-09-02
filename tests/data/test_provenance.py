"""Provenance record tests (spec §5 F5, Task 8)."""
import hashlib
import json

import pydantic
import pytest

from src.data.provenance import (
    ProvenanceRecord, parse_log_extras, sha256_file, write_provenance,
)


def _record(**kw):
    d = dict(seed_thermal_tip=1, seed_thermal_sub=2, seed_velocity=3,
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


def test_builder_writes_provenance(tmp_path):
    from src.builders.afm import AFMSimulation
    from tests.builders.test_afm_builder import (_make_afm_config,
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
