"""DataReader walks with discover_runs and emits flat RunFrames (P1, P2).

The two-arm sweep case is the one that fails on the old reader: the nested
setdefault chain had no pot_/K{T} dimension, so the last directory walked won
and one of the two arms vanished without a warning.
"""

from __future__ import annotations

import json

import pandas as pd

from src.postprocessing.read_data import (DataReader, LoadPoint, RunFrames,
                                          parse_results_name)
from tests.helpers.results_tree import make_family_run


def test_parse_results_name_reads_a_window_file():
    point = parse_results_name('friction_f30_a-15_s2_layer3.txt')

    assert point == LoadPoint(load_kind='force', load=30.0, angle=-15.0,
                              speed=2.0, tier='window', layer=3)


def test_parse_results_name_reads_pressure_raw_and_legacy():
    assert parse_results_name('friction_p1.5_a0_s2.txt') == LoadPoint(
        load_kind='pressure', load=1.5, angle=0.0, speed=2.0,
        tier='window', layer=None)
    assert parse_results_name('raw_f30_a0_s2_layer2.txt').tier == 'raw'
    assert parse_results_name('peratom_f30_a0_s2_layer2.txt').tier == 'peratom'
    assert parse_results_name('fc_ave_slide_30nN_0angle_2ms_l2.txt') == LoadPoint(
        load_kind='force', load=30.0, angle=0.0, speed=2.0,
        tier='window', layer=2)
    assert parse_results_name('fc_ave_slide_1.5GPa_0angle_2ms.txt') == LoadPoint(
        load_kind='pressure', load=1.5, angle=0.0, speed=2.0,
        tier='window', layer=None)


def test_parse_results_name_refuses_anything_else():
    assert parse_results_name('log.lammps') is None
    assert parse_results_name('final_f30N_a0.data') is None


def test_a_potential_sweep_keeps_both_arms(tmp_path):
    make_family_run(tmp_path, 'afm', pot='sw_a')
    make_family_run(tmp_path, 'afm', pot='sw_b')

    reader = DataReader(results_dir=str(tmp_path))

    assert len(reader.runs) == 2
    assert sorted(r.tokens.pot_token for r in reader.runs) == [
        'pot_sw_a', 'pot_sw_b']


def test_a_temperature_sweep_keeps_both_arms(tmp_path):
    make_family_run(tmp_path, 'afm', temp=77)
    make_family_run(tmp_path, 'afm', temp=300)

    reader = DataReader(results_dir=str(tmp_path))

    assert sorted(r.tokens.temperature for r in reader.runs) == [77.0, 300.0]


def test_a_sheet_run_reports_its_substrate_as_none_not_k300(tmp_path):
    make_family_run(tmp_path, 'sheetonsheet')

    reader = DataReader(results_dir=str(tmp_path))

    run, = reader.runs
    assert run.tokens.substrate is None
    assert run.tokens.temperature == 300.0
    assert run.load_kind == 'pressure'


def test_one_run_frames_per_load_point(tmp_path):
    make_family_run(tmp_path, 'afm', loads=(10.0, 30.0), angles=(0.0, 90.0),
                    speeds=(2.0,))

    reader = DataReader(results_dir=str(tmp_path))

    assert len(reader.runs) == 4
    assert sorted((r.load, r.angle) for r in reader.runs) == [
        (10.0, 0.0), (10.0, 90.0), (30.0, 0.0), (30.0, 90.0)]
    assert all(isinstance(r, RunFrames) for r in reader.runs)


def test_run_frames_properties(tmp_path):
    make_family_run(tmp_path, 'hetero', material='h-MoS2',
                    second_material='h-WS2')

    reader = DataReader(results_dir=str(tmp_path))

    run, = reader.runs
    assert run.family == 'hetero'
    assert run.material == 'h_MoS2+h_WS2'
    assert run.size == '100x100y'
    assert run.n_layers == 2
    assert isinstance(run.window, pd.DataFrame)
    assert 'time' in run.window.columns
    assert set(run.frames_by_tier()) == {'window'}


def test_runs_are_sorted_deterministically(tmp_path):
    make_family_run(tmp_path, 'afm', material='h-WS2')
    make_family_run(tmp_path, 'afm', material='h-MoS2',
                    loads=(10.0, 30.0))

    reader = DataReader(results_dir=str(tmp_path))

    keys = [(r.tokens.rel_path, r.load_kind, r.load, r.angle, r.speed)
            for r in reader.runs]
    assert keys == sorted(keys)


def test_a_pes_grid_is_one_run_with_no_load(tmp_path):
    """A PES scan has no friction files at all -- its whole result is
    `results/pes_scan.csv`, which IS its window tier (the same mapping
    `FrictionResultsData._TIER_PREFIX` makes). `load_kind` is 'none' rather
    than a fictitious 0.0 nN force."""
    make_family_run(tmp_path, 'pes_sheet')

    reader = DataReader(results_dir=str(tmp_path))

    run, = reader.runs
    assert run.family == 'pes_sheet'
    assert run.load_kind == 'none'
    assert run.load == 0.0
    assert set(run.tiers) == {'window'}
    assert run.source_files['window'].name == 'pes_scan.csv'
    assert {'dx', 'dy', 'energy_eV'} <= set(run.window.columns)


def test_a_pes_tip_grid_carries_the_force_columns(tmp_path):
    make_family_run(tmp_path, 'pes_tip')

    run, = DataReader(results_dir=str(tmp_path)).runs

    assert run.tokens.layers == 2          # pes_tip runs live under L{n}
    assert {'x', 'y', 'fx', 'fy', 'fz'} <= set(run.window.columns)
    assert 'dx' not in run.window.columns


def test_parse_results_name_reads_a_pes_grid():
    point = parse_results_name('pes_scan.csv')

    assert point == LoadPoint(load_kind='none', load=0.0, angle=0.0,
                              speed=0.0, tier='window', layer=None)
    assert parse_results_name('pes_scan_refine.csv') is None


def test_an_empty_root_is_not_an_exception(tmp_path):
    reader = DataReader(results_dir=str(tmp_path))

    assert reader.runs == []


def test_all_three_tiers_are_read(tmp_path):
    """The old file selection anchored on friction_/fc_ave_, so raw_ and
    peratom_ files were never opened and the tier branch was unreachable."""
    make_family_run(tmp_path, 'afm', tiers=('window', 'raw', 'peratom'))

    reader = DataReader(results_dir=str(tmp_path))

    run, = reader.runs
    assert set(run.tiers) == {'window', 'raw', 'peratom'}
    assert set(run.source_files) == {'window', 'raw', 'peratom'}
    assert run.source_files['raw'].name.startswith('raw_')
    assert 'raw_lfx' in run.tiers['raw'].columns
    assert 'lfx' not in run.tiers['raw'].columns


def test_a_tier_absent_on_disk_is_simply_absent(tmp_path):
    make_family_run(tmp_path, 'sheetonsheet', tiers=('window',))

    reader = DataReader(results_dir=str(tmp_path))

    run, = reader.runs
    assert set(run.tiers) == {'window'}
    assert run.window is run.tiers['window']


def test_time_is_kept_in_every_tier(tmp_path):
    make_family_run(tmp_path, 'afm', tiers=('window', 'raw'))

    reader = DataReader(results_dir=str(tmp_path))

    run, = reader.runs
    assert 'time' in run.tiers['window'].columns
    assert 'time' in run.tiers['raw'].columns


def test_a_sheet_run_takes_its_layer_count_from_provenance(tmp_path):
    """`output_dir_for` emits no L{n} for a sheet-like model, so
    `tokens.layers` is None and `or 2` would silently cap every sheet run at
    two layers -- the exact defect P3 exists to fix, missed in the family that
    needs it most."""
    run_dir = make_family_run(tmp_path, 'sheetonsheet')
    prov = run_dir / 'provenance.json'
    record = json.loads(prov.read_text(encoding='utf-8'))
    record['layer_type_map'] = {f'layer_{i}': [i] for i in range(1, 5)}
    prov.write_text(json.dumps(record), encoding='utf-8')

    run, = DataReader(results_dir=str(tmp_path)).runs

    assert run.tokens.layers is None      # nothing in the path says so
    assert run.n_layers == 4              # provenance does


def test_an_empty_layer_type_map_falls_back_to_two(tmp_path):
    """hetero and flake partition by z-band and honestly write `{}`."""
    run_dir = make_family_run(tmp_path, 'hetero', material='h-MoS2',
                              second_material='h-WS2')
    prov = run_dir / 'provenance.json'
    record = json.loads(prov.read_text(encoding='utf-8'))
    record['layer_type_map'] = {}
    prov.write_text(json.dumps(record), encoding='utf-8')

    run, = DataReader(results_dir=str(tmp_path)).runs

    assert run.n_layers == 2


def test_a_five_layer_run_names_its_layer_columns(tmp_path):
    """rename_map(family, tier) defaulted to the bilayer map, so a 5-layer
    run's L3..L5 columns reached the frame as v_comz_L3 and were invisible to
    every consumer that keys on channel names."""
    run_dir = make_family_run(tmp_path, 'afm', layers=5)
    natives = ['v_fz_tip', 'v_fx_spr', 'v_fy_spr', 'v_comz_L5']
    path = run_dir / 'results' / 'friction_f30_a0_s2_layer5.txt'
    path.write_text(
        '# schema=2 lf_units=nN family=afm\n'
        '# TimeStep ' + ' '.join(natives) + '\n'
        + '\n'.join(f'{(i + 1) * 1000} 30.0 3.0 0.0 {6.5 + i * 0.1:g}'
                    for i in range(10)) + '\n',
        encoding='utf-8')

    reader = DataReader(results_dir=str(tmp_path))

    run, = reader.runs
    assert run.n_layers == 5
    assert 'comz_L5' in run.window.columns
    assert 'v_comz_L5' not in run.window.columns
