"""The per-run `provenance` block of every `runs[]` entry."""

from __future__ import annotations

import json

import pytest

from src.data.provenance import (BuiltGeometry, read_provenance,
                                 sha256_file, write_provenance)
from src.postprocessing.read_data import DataReader
from src.postprocessing.run_provenance import config_settings
from tests.helpers.results_tree import make_family_run, slide_config


def _export(root):
    written, = DataReader(results_dir=str(root)).export_full_data_to_json()
    return json.loads(written.read_text(encoding='utf-8'))['runs']


def _rewrite(run_dir, **update):
    record = read_provenance(run_dir / 'provenance.json')
    write_provenance(run_dir, record.model_copy(update=update))


def test_block_has_record_config_hash_and_flags(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm', config=slide_config(
        'afm', loads=(30,)))

    block = _export(tmp_path)[0]['provenance']

    assert block['family'] == 'afm'
    assert block['git_commit'] == 'deadbeef'
    assert block['provenance_sha256'] == sha256_file(run_dir / 'provenance.json')
    assert block['enriched'] is False
    assert block['config']['output']['tier_frequency'] == {
        'window': 1000, 'raw': 100, 'peratom': 1000}
    assert block['config']['simulation']['slide_run_steps'] == 50_000
    assert block['config']['simulation']['afm_hold_steps'] == 10000
    assert block['config']['tip'] is None


def test_pot_path_is_reduced_to_its_basename(tmp_path):
    make_family_run(tmp_path, 'afm')

    mat, = _export(tmp_path)[0]['provenance']['materials']

    assert mat['pot_path'] == 'MoS.sw'
    assert mat['pot_file_sha256'] == 'ab' * 32


def test_execution_fields_are_nested(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm')
    _rewrite(run_dir, wall_time_s=12.0, n_procs=32)

    block = _export(tmp_path)[0]['provenance']

    assert block['execution'] == {'wall_time_s': 12.0, 'n_procs': 32,
                                  'atoms_per_proc': None}
    assert 'wall_time_s' not in block


def test_enriched_vs_unenriched(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm')
    before = _export(tmp_path)[0]['provenance']
    _rewrite(run_dir, lammps_version='29 Aug 2024', ramp_converged=True)
    after = _export(tmp_path)[0]['provenance']

    assert before['enriched'] is False
    assert after['enriched'] is True
    assert before['provenance_sha256'] != after['provenance_sha256']


def test_per_load_ramp_and_load_achieved_reach_each_entry(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm', loads=(10, 20))
    _rewrite(run_dir,
             ramp_by_load={'10': {'converged': True, 'r': 0.05, 'iterations': 3},
                           '20': {'converged': False, 'r': 0.5,
                                  'iterations': 40}},
             load_achieved_by_load={'10': 9.8, '20': 19.5})

    ten, twenty = _export(tmp_path)

    assert ten['provenance']['ramp']['converged'] is True
    assert ten['provenance']['load_achieved'] == 9.8
    assert twenty['provenance']['ramp']['iterations'] == 40
    assert twenty['provenance']['load_achieved'] == 19.5


def test_geometry_is_copied_verbatim(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm')
    _rewrite(run_dir, geometry=BuiltGeometry(lat_c=6.15,
                                             lat_c_source='minimised'))

    assert _export(tmp_path)[0]['provenance']['geometry']['lat_c'] == 6.15


def test_no_output_points_carry_the_block_too(tmp_path):
    make_family_run(tmp_path, 'afm', loads=(10,), config=slide_config(
        'afm', loads=(10, 20)))

    runs = _export(tmp_path)

    assert [r['status'] for r in runs] == ['complete', 'no_output']
    assert all(r['provenance']['git_commit'] == 'deadbeef' for r in runs)


@pytest.mark.parametrize('text', ['{"family": "afm"}', '{'])
def test_unreadable_provenance_does_not_raise(tmp_path, text):
    run_dir = make_family_run(tmp_path, 'afm')
    (run_dir / 'provenance.json').write_text(text, encoding='utf-8')

    run, = _export(tmp_path)

    assert 'error' in run['provenance']
    assert run['provenance']['provenance_sha256']
    assert run['status_detail'].endswith('provenance unreadable')


def test_config_settings_tolerates_missing_keys():
    assert config_settings(None) is None
    out = config_settings({})
    assert out['simulation']['timestep'] is None
    assert out['output']['tier_frequency'] is None
    assert out['thermostat'] is None
    assert 'afm_hold_steps' not in out['simulation']
