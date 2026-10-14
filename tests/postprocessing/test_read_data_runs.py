"""DataReader walks with discover_runs and emits flat RunFrames (P1, P2).

The two-arm sweep case is the one that fails on the old reader: the nested
setdefault chain had no pot_/K{T} dimension, so the last directory walked won
and one of the two arms vanished without a warning.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.core.observables import FAMILIES
from src.postprocessing.read_data import (SUMMARY_IDENTITY_COLUMNS,
                                          SUMMARY_LF_COLUMNS,
                                          SUMMARY_POST_COLUMNS, DataReader,
                                          LoadPoint, RunFrames,
                                          parse_results_name)
from tests.helpers.results_tree import make_family_run

GOLDEN_DIR = Path(__file__).resolve().parents[1] / 'fixtures' / 'golden'

#: Keys schema 3 adds to each `runs[]` entry and to `metadata`. Removing them
#: from a v3 document must give back the v2 golden exactly: v3 is additive.
V3_RUN_KEYS = {'status', 'status_detail', 'n_rows', 'expected_rows',
               'provenance'}
V3_META_KEYS = {'provenance', 'status_counts'}


def as_v2(doc):
    """A v3 document minus everything v3 added; must equal the v2 golden."""
    doc = copy.deepcopy(doc)
    doc['schema'] = 2
    for key in V3_META_KEYS:
        doc['metadata'].pop(key, None)
    for run in doc['runs']:
        for key in V3_RUN_KEYS:
            run.pop(key, None)
    return doc


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


def test_metadata_carries_every_sweep_axis(tmp_path):
    make_family_run(tmp_path, 'afm', material='h-MoS2', temp=300, pot='sw_a',
                    loads=(10.0, 30.0), tiers=('window', 'raw'))
    make_family_run(tmp_path, 'afm', material='h-WS2', temp=77, pot='sw_b',
                    layers=3)

    meta = DataReader(results_dir=str(tmp_path)).metadata

    assert meta['families'] == ['afm']
    assert meta['models'] == ['afm']
    assert meta['materials'] == ['h_MoS2', 'h_WS2']
    assert meta['potentials'] == ['pot_sw_a', 'pot_sw_b']
    assert meta['temperatures'] == [77.0, 300.0]
    assert meta['layers'] == [2, 3]
    assert meta['sizes'] == ['100x100y']
    assert meta['speeds'] == [2.0]
    # `aSi` is the LEGACY spelling of an amorphous Si substrate;
    # `_split_amorph_token` separates it into the material name and the
    # flag, so the substrate axis carries materials and nothing else.
    assert meta['substrates'] == ['Si']
    assert meta['tip_materials'] == ['Si']
    assert meta['tip_radii'] == [25.0]
    assert meta['tiers'] == ['raw', 'window']
    assert meta['loads_by_kind']['force'] == {'10.0': [0.0], '30.0': [0.0]}
    assert meta['loads_by_kind']['pressure'] == {}


def test_material_types_classifies_by_prefix():
    from src.postprocessing.read_data import material_types

    types = material_types(['h_MoS2', 't_CoTe2', 'b_SiSe', 'p_CS', 'AlSe'])

    assert types == {'b_type': ['b_SiSe'], 'h_type': ['h_MoS2'],
                     't_type': ['t_CoTe2'], 'p_type': ['p_CS'],
                     'other': ['AlSe']}


def test_ntimestep_and_time_series(tmp_path):
    make_family_run(tmp_path, 'afm', n=30)

    reader = DataReader(results_dir=str(tmp_path))

    assert reader.ntimestep == 30
    assert reader.metadata['ntimestep'] == 30
    assert reader.time_series[:2] == [1000, 2000]
    assert len(reader.time_series) == 30


def test_a_short_run_is_reported_incomplete(tmp_path):
    make_family_run(tmp_path / 'r', 'afm', material='h-MoS2', n=50)
    make_family_run(tmp_path / 'r', 'afm', material='h-WS2', n=5)

    reader = DataReader(results_dir=str(tmp_path / 'r'))
    reader.export_issue_reports()

    assert reader.incomplete_materials['100x100y'] == {'h_WS2'}
    report = (tmp_path / 'r' / 'outputs' / 'incomplete_materials_100x100y.txt')
    assert report.read_text(encoding='utf-8').strip() == 'h_WS2'
    assert len(reader.runs) == 2      # incomplete runs are kept, only flagged


def test_runs_by_size_groups_and_sorts(tmp_path):
    make_family_run(tmp_path, 'afm', size=(100.0, 100.0))
    make_family_run(tmp_path, 'afm', size=(50.0, 50.0))

    grouped = DataReader(results_dir=str(tmp_path)).runs_by_size()

    assert sorted(grouped) == ['100x100y', '50x50y']
    assert len(grouped['50x50y']) == 1


def test_export_writes_one_file_per_size(tmp_path):
    make_family_run(tmp_path, 'afm', size=(100.0, 100.0))
    make_family_run(tmp_path, 'afm', size=(50.0, 50.0))

    written = DataReader(results_dir=str(tmp_path)).export_full_data_to_json()

    assert sorted(p.name for p in written) == [
        'output_full_100x100y.json', 'output_full_50x50y.json']
    assert all(p.parent.name == 'outputs' for p in written)


def test_v2_document_shape(tmp_path):
    make_family_run(tmp_path, 'afm', pot='sw_a', loads=(10.0, 30.0),
                    angles=(0.0, 90.0), tiers=('window', 'raw'),
                    final_data=True)

    written, = DataReader(results_dir=str(tmp_path)).export_full_data_to_json()
    doc = json.loads(written.read_text(encoding='utf-8'))

    assert doc['schema'] == 3
    assert doc['size'] == '100x100y'
    assert doc['generated_by'] == 'FrictionSim2D'
    assert doc['units']['time'] == 'step'
    assert doc['units']['nf'] == 'nN'
    assert doc['units']['cof'] == '1'
    assert doc['units']['lateral_force'] == 'nN'
    assert doc['metadata']['families'] == ['afm']
    assert doc['metadata']['potentials'] == ['pot_sw_a']
    assert doc['metadata']['tiers'] == ['raw', 'window']
    assert doc['metadata']['summary_definition']['skip_fraction'] == 0.2
    assert 'ratio_of_means' in doc['metadata']['summary_definition']['cof']
    assert len(doc['runs']) == 4

    run = doc['runs'][0]
    assert set(run) == {
        'model', 'family', 'material', 'materials', 'flake_material', 'size',
        'size_x', 'size_y', 'substrate', 'substrate_amorphous', 'tip_material',
        'tip_radius', 'temperature', 'layers', 'potential', 'rel_path',
        'load_kind', 'load', 'angle', 'speed', 'n_layers', 'tiers', 'post',
        'status', 'status_detail', 'n_rows', 'expected_rows', 'provenance'}
    assert run['load_kind'] == 'force'
    assert set(run['tiers']) == {'window', 'raw'}
    assert run['tiers']['window']['columns'][0] == 'time'
    assert 'cof' in run['tiers']['window']['columns']
    assert 'cof' not in run['tiers']['raw']['columns']
    assert run['tiers']['window']['attrs']['lf_schema'] == 2
    assert run['post'] == {'gap_min': pytest.approx(3.1),
                           'deflect_max': pytest.approx(0.4)}


def test_runs_are_sorted_in_the_document(tmp_path):
    make_family_run(tmp_path, 'afm', loads=(30.0, 10.0), angles=(90.0, 0.0))

    written, = DataReader(results_dir=str(tmp_path)).export_full_data_to_json()
    runs = json.loads(written.read_text(encoding='utf-8'))['runs']

    keys = [(r['rel_path'], r['load_kind'], r['load'], r['angle'], r['speed'])
            for r in runs]
    assert keys == sorted(keys)


def test_a_run_without_final_data_has_an_empty_post(tmp_path):
    make_family_run(tmp_path, 'sheetonsheet')

    written, = DataReader(results_dir=str(tmp_path)).export_full_data_to_json()

    assert json.loads(written.read_text(encoding='utf-8'))['runs'][0]['post'] == {}


@pytest.mark.parametrize('family', FAMILIES)
def test_v3_is_v2_plus_new_keys_only(tmp_path, family):
    """The whole document is pinned, byte for byte after a stable dump.

    A golden is the only test that catches a key silently disappearing --
    every targeted assertion above passes just as happily on a document that
    lost three keys nobody asserted on. The v2 golden stays the reference:
    with the keys schema 3 added stripped, the document must be unchanged,
    so no v2 key was renamed, removed or changed meaning."""
    make_family_run(tmp_path, family, final_data=True, n=4,
                    tiers=('window', 'raw'))

    written, = DataReader(results_dir=str(tmp_path)).export_full_data_to_json()
    produced = json.loads(written.read_text(encoding='utf-8'))
    golden = json.loads(
        (GOLDEN_DIR / f'output_full_v2_{family}.json').read_text(encoding='utf-8'))

    assert as_v2(produced) == golden


def test_summary_rows_come_from_summarise_run(tmp_path):
    """The one COF. `df.mean()` in plot_common, element-wise
    compute_derived_columns in read_data and ratio-of-means in the DB were
    three different numbers on one plot.

    Checked against `compute_friction_stats` on the parsed arrays -- the
    kernel `summarise_run` uses and the number the store writes into
    `FrictionResultsData`'s attributes and the central `ResultRecord` --
    rather than against `summarise_run` called again here, which would only
    prove the function is deterministic."""
    from src.data.models import compute_friction_stats

    make_family_run(tmp_path, 'afm')

    reader = DataReader(results_dir=str(tmp_path))
    row, = reader.summary_rows()

    run, = reader.runs
    window = run.window
    kernel = compute_friction_stats(window['nf'].to_numpy(float),
                                    window['lfx'].to_numpy(float),
                                    window['lfy'].to_numpy(float), 0.2)
    assert row['mean_cof'] == pytest.approx(kernel['mean_cof'])
    assert row['mean_lf'] == pytest.approx(kernel['mean_lf'])
    assert row['std_lfy'] == pytest.approx(kernel['std_lfy'])
    assert row['ntimesteps'] == len(window)
    assert row['is_complete'] is True


def test_summary_row_identity_columns(tmp_path):
    make_family_run(tmp_path, 'hetero', material='h-MoS2',
                    second_material='h-WS2', pot='sw_a',
                    tiers=('window', 'raw'))

    row, = DataReader(results_dir=str(tmp_path)).summary_rows()

    assert row['model'] == 'sheetonsheet'
    assert row['family'] == 'hetero'
    assert row['material'] == 'h_MoS2+h_WS2'
    assert row['materials'] == 'h-MoS2+h-WS2'
    assert row['flake_material'] is None
    assert row['potential'] == 'pot_sw_a'
    assert row['tiers'] == 'raw;window'
    assert row['load_kind'] == 'pressure'


def test_the_summary_row_layers_column_is_the_resolved_count(tmp_path):
    """`layers` is a headline filter in the DB, the API and the GUI. Taking it
    from the path alone made it NULL for every sheet, hetero, flake and
    pes_sheet row."""
    run_dir = make_family_run(tmp_path, 'sheetonsheet')
    prov = run_dir / 'provenance.json'
    record = json.loads(prov.read_text(encoding='utf-8'))
    record['layer_type_map'] = {f'layer_{i}': [i] for i in range(1, 5)}
    prov.write_text(json.dumps(record), encoding='utf-8')

    row, = DataReader(results_dir=str(tmp_path)).summary_rows()

    assert row['layers'] == 4


def test_post_channels_reach_the_row(tmp_path):
    make_family_run(tmp_path, 'afm', final_data=True)

    row, = DataReader(results_dir=str(tmp_path)).summary_rows()

    assert row['gap_min'] == pytest.approx(3.1)
    assert math.isnan(row['gap_min_il'])
    assert math.isnan(row['corrugation_max'])


def test_export_summary_csv_column_order(tmp_path):
    make_family_run(tmp_path, 'afm', loads=(10.0, 30.0))

    written, = DataReader(results_dir=str(tmp_path)).export_summary_csv()
    frame = pd.read_csv(written)

    assert written.name == 'summary_100x100y.csv'
    assert list(frame.columns[:len(SUMMARY_IDENTITY_COLUMNS)]) == list(
        SUMMARY_IDENTITY_COLUMNS)
    tail = list(SUMMARY_LF_COLUMNS) + list(SUMMARY_POST_COLUMNS)
    assert list(frame.columns[-len(tail):]) == tail
    assert 'mean_cof' in frame.columns
    assert len(frame) == 2


#: The fourteen RunSummary scalars, in `RunSummary.as_dict()` order. Every
#: family carries all of them; `extras` is what varies.
_SUMMARY_SCALARS = ('mean_cof', 'std_cof', 'mean_lf', 'std_lf', 'mean_nf',
                    'std_nf', 'mean_lfx', 'std_lfx', 'mean_lfy', 'std_lfy',
                    'skip_fraction', 'ntimesteps', 'time_series_hash',
                    'is_complete')


@pytest.mark.parametrize('family', FAMILIES)
def test_every_family_produces_the_same_frame(tmp_path, family):
    """Not the same HEADER -- `RunSummary.extras` is family-specific by
    design -- but the same frame: identity columns first, the fourteen
    scalars next, the lf and post columns last, extras in the middle."""
    root = tmp_path / family
    make_family_run(root, family)

    written, = DataReader(results_dir=str(root)).export_summary_csv()
    columns = list(pd.read_csv(written).columns)

    head = len(SUMMARY_IDENTITY_COLUMNS)
    tail = list(SUMMARY_LF_COLUMNS) + list(SUMMARY_POST_COLUMNS)
    assert columns[:head] == list(SUMMARY_IDENTITY_COLUMNS)
    assert columns[-len(tail):] == tail
    assert set(_SUMMARY_SCALARS) <= set(columns)


def test_two_families_concatenate_through_pandas(tmp_path):
    """The honest concatenation story: pd.concat fills the extras one family
    has and the other does not with NaN."""
    frames = []
    for family in ('afm', 'sheetonsheet'):
        root = tmp_path / family
        make_family_run(root, family)
        written, = DataReader(results_dir=str(root)).export_summary_csv()
        frames.append(pd.read_csv(written))

    combined = pd.concat(frames, ignore_index=True)

    assert len(combined) == 2
    assert set(combined['family']) == {'afm', 'sheetonsheet'}
    assert set(_SUMMARY_SCALARS) <= set(combined.columns)


def test_the_regex_walk_is_gone():
    """These six patterns were the walk. Two of them carried real defects --
    _PATH_PATTERN_SHEET group 3 captured the K{T} token as the substrate, and
    neither path pattern had a pot_ or K{T} group at all."""
    import src.postprocessing.read_data as rd

    for name in ('_FILE_PATTERN_TIP', '_FILE_PATTERN_TIP_ALT',
                 '_FILE_PATTERN_SHEET', '_FILE_PATTERN_SHEET_ALT',
                 '_PATH_PATTERN_TIP', '_PATH_PATTERN_SHEET'):
        assert not hasattr(rd.DataReader, name), name


def test_the_nested_tree_and_its_walk_are_gone(tmp_path):
    import src.postprocessing.read_data as rd

    make_family_run(tmp_path, 'afm')
    reader = rd.DataReader(results_dir=str(tmp_path))

    assert not hasattr(reader, 'full_data_nested')
    for name in ('_first_pass_ntimestep', '_resolve_path_info',
                 '_is_sheet_family', '_parse_filename_params',
                 '_finalize_metadata', 'read_data'):
        assert not hasattr(reader, name), name


def test_the_file_level_parser_survives():
    """Spec section 3: the file-level parser is untouched. 20 calls in
    test_read_data_schema.py drive it directly."""
    import src.postprocessing.read_data as rd

    assert callable(rd.DataReader._read_dataframe)
    assert callable(rd.DataReader._registry_sim_type)
    assert rd.DataReader._LF_SCHEMA1_CORRECTION == 1.602176565
    assert rd.DataReader._KNOWN_FAMILIES
