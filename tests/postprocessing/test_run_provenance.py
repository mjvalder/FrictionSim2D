"""The per-run `provenance` block of every `runs[]` entry."""

from __future__ import annotations

import json

import pytest

from frictionsim2d.data.provenance import (BuiltGeometry, read_provenance,
                                 sha256_file, write_provenance)
from frictionsim2d.postprocessing.read_data import DataReader
from frictionsim2d.postprocessing.run_provenance import config_settings
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


# --- pooled metadata ---------------------------------------------------------

def _pooled(root):
    reader = DataReader(results_dir=str(root))
    written, = reader.export_full_data_to_json()
    return json.loads(written.read_text(encoding='utf-8'))['metadata']


def _two_runs(root, **update_b):
    dir_a = make_family_run(root, 'afm', pot='a', loads=(10, 20, 30))
    dir_b = make_family_run(root, 'afm', pot='b')
    return dir_a, dir_b


def test_identical_fields_pool_to_a_value(tmp_path):
    for pot in ('a', 'b'):
        make_family_run(tmp_path, 'afm', pot=pot,
                        config=slide_config('afm', loads=(30,)))

    meta = _pooled(tmp_path)

    assert meta['provenance']['config.simulation.slide_run_steps'] == {
        'value': 50_000}


def test_varying_fields_list_counts(tmp_path):
    dir_a, dir_b = _two_runs(tmp_path)
    _rewrite(dir_a, drive_k=0.4994)
    _rewrite(dir_b, drive_k=3.12)

    meta = _pooled(tmp_path)

    # three load points of run A still count once
    assert meta['provenance']['drive_k'] == {
        'values': {'0.4994': 1, '3.12': 1}}


def test_config_hashes_count_runs(tmp_path):
    dir_a, dir_b = _two_runs(tmp_path)
    dir_c = make_family_run(tmp_path, 'afm', pot='c')
    _rewrite(dir_a, config_hash='aa' * 32)
    _rewrite(dir_b, config_hash='aa' * 32)
    _rewrite(dir_c, config_hash='bb' * 32)

    assert _pooled(tmp_path)['provenance']['config_hashes'] == {
        'aa' * 32: 2, 'bb' * 32: 1}


def test_seeds_are_not_pooled(tmp_path):
    dir_a, dir_b = _two_runs(tmp_path)
    _rewrite(dir_a, seed_velocity=1)
    _rewrite(dir_b, seed_velocity=2)

    prov = _pooled(tmp_path)['provenance']

    assert not any(key.startswith('seed_') for key in prov)
    assert prov['seeds_unique'] is True
    assert prov['seeds_shared_across_runs'] == 0

    _rewrite(dir_a, seed_velocity=7)
    _rewrite(dir_b, seed_velocity=7)
    prov = _pooled(tmp_path)['provenance']
    assert prov['seeds_unique'] is False
    assert prov['seeds_shared_across_runs'] == 1


def test_pooling_tolerates_an_unreadable_record(tmp_path):
    dir_a, dir_b = _two_runs(tmp_path)
    (dir_b / 'provenance.json').write_text('{not json', encoding='utf-8')

    prov = _pooled(tmp_path)['provenance']

    assert prov['config_hashes'] == {'ff' * 32: 1}


def test_afm_hold_steps_for_every_family_that_runs_the_afm_hold():
    """`indent` renders `afm/system_init.lmp` too, so it holds like `afm`."""
    assert config_settings({}, 'afm')['simulation']['afm_hold_steps'] == 10000
    assert config_settings({}, 'indent')['simulation']['afm_hold_steps'] == 10000
    assert 'afm_hold_steps' not in config_settings(
        {}, 'sheetonsheet')['simulation']


def test_afm_hold_steps_lives_in_core_observables():
    """One definition, importable without the AFM builder stack; the builder
    re-exports the same object for the template render."""
    from frictionsim2d.builders.afm.base import AFM_HOLD_STEPS as builder_value
    from frictionsim2d.core.observables import AFM_HOLD_STEPS
    assert AFM_HOLD_STEPS == builder_value == 10000
