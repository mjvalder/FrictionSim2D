"""Stage 3 in DataReader: integrity statuses, and halted points keep frames."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scipy')

from frictionsim2d.core.config import IntegritySettings  # noqa: E402
from frictionsim2d.postprocessing.integrity import (  # noqa: E402
    NO_SNAPSHOT_DETAIL, check_files, resolve_layers, writes_end_snapshot)
from frictionsim2d.postprocessing.read_data import (  # noqa: E402
    SUMMARY_POST_COLUMNS, DataReader, resolve_n_layers)
from frictionsim2d.run.tree import discover_runs  # noqa: E402
from tests.helpers.results_tree import make_family_run, slide_config  # noqa: E402
from tests.postprocessing.test_integrity import (  # noqa: E402
    _set_layer_map, _stack, _write_data)

S = IntegritySettings()
DECK = 'write_data      ./data/final_f$(v_find)N_a$(v_a).data\n'


def _halt_line(**fields) -> str:
    return '# HALT ' + ' '.join(f'{k}={v}' for k, v in fields.items())


def _point(reader, load, speed=None, points=False):
    pool = reader.points if points else reader.runs
    hits = [p for p in pool if p.load == load and (speed is None or p.speed == speed)]
    assert len(hits) == 1, hits
    return hits[0]


def _afm_snapshots(run_dir, load, *, eject=False):
    """`data/load_<F>N.data` reference and `data/final_f<F>N_a0.data`."""
    stack, box = _stack(2)
    _write_data(run_dir / 'data' / f'load_{load:g}N.data', stack, box)
    final = [(t, xyz.copy()) for t, xyz in stack]
    if eject:
        final[1][1][:2, 2] += 30.0                   # two layer-2 atoms 30 A up
    _write_data(run_dir / 'data' / f'final_f{load:g}N_a0.data', final, box)


def _afm_tree(root, loads=(30.0,), eject=(), speeds=(2.0,), deck=True):
    run_dir = make_family_run(root, 'afm', loads=loads, speeds=speeds, n=50,
                              config=slide_config('afm', loads=loads,
                                                  speeds=speeds))
    if deck:
        (run_dir / 'lammps' / 'slide.in').write_text(DECK, encoding='utf-8')
    for load in loads:
        _afm_snapshots(run_dir, load, eject=load in eject)
    return run_dir


def _tree_with_halt(root):
    run_dir = _afm_tree(root, loads=(10.0, 30.0))
    (run_dir / 'results' / 'halted.txt').write_text(
        _halt_line(stage='slide', load=30, angle=0, speed=2, T=9000,
                   step=3000) + '\n', encoding='utf-8')
    return root


# --- DataReader -------------------------------------------------------------

def test_halted_point_keeps_frames(tmp_path):            # Review Focus 3
    root = _tree_with_halt(tmp_path)
    r = DataReader(str(root))
    p = _point(r, load=30.0)
    assert p.status == 'halted' and p.tiers and p in r.runs
    assert 'T=9000' in p.status_detail and len(p.window) == 50
    v = p.integrity                                  # R8: every point has a verdict
    assert (v.status, v.failed, v.observed) == ('halted', True, True)
    assert v.reasons[0] == 'halted'
    assert v.measures['n_heat'] == 512 and np.isfinite(v.measures['T_sigma'])
    assert p not in DataReader(str(root), complete_only=True).runs


def test_no_output_point_payload_is_failed_and_unobserved(tmp_path):
    """R8: a configured point with no file still carries failed/observed."""
    run_dir = _afm_tree(tmp_path, loads=(10.0, 30.0))
    for path in (run_dir / 'results').glob('friction_f30_*'):
        path.unlink()
    r = DataReader(str(tmp_path))
    p = _point(r, 30.0, points=True)
    assert p.status == 'no_output'
    pay = r.run_payload(p)['integrity']
    assert pay['status'] == 'no_output'
    assert pay['failed'] is True and pay['observed'] is False
    assert _point(r, 10.0).integrity.status == 'complete'


def test_truncated_point_gets_t_sigma(tmp_path):
    """R8 makes the reference atom-count fallback reachable for truncated."""
    from tests.helpers.synthetic_results import write_results
    run_dir = _afm_tree(tmp_path)
    write_results(run_dir, 'afm', load_kind='force', loads=(30.0,), layers=2, n=10)
    p = _point(DataReader(str(tmp_path)), 30.0)
    assert p.status == 'truncated'
    assert p.integrity.status == 'truncated' and p.integrity.failed
    assert p.integrity.measures['n_heat'] == 512


def test_final_file_without_deck_counts_as_writing_a_snapshot(tmp_path):
    """R9: a tree copied without lammps/ still runs the snapshot rules."""
    _afm_tree(tmp_path, eject=(30.0,), deck=False)
    p = _point(DataReader(str(tmp_path)), 30.0)
    assert p.status == 'failed_integrity' and 'ejected_atoms' in p.status_detail
    assert NO_SNAPSHOT_DETAIL not in p.integrity.not_applicable


def test_halted_frames_give_no_summary_stats(tmp_path):
    root = _tree_with_halt(tmp_path)
    rows = {row['load']: row for row in DataReader(str(root)).summary_rows()}
    assert rows[30.0]['status'] == 'halted' and rows[30.0]['mean_cof'] is None
    assert all(np.isnan(rows[30.0][name]) for name in SUMMARY_POST_COLUMNS)
    assert rows[10.0]['mean_cof'] is not None


def test_failed_integrity_point(tmp_path):               # final_data with an ejected pair
    _afm_tree(tmp_path, eject=(30.0,))
    p = _point(DataReader(str(tmp_path)), 30.0)
    assert p.status == 'failed_integrity' and 'ejected_atoms' in p.status_detail
    assert p.integrity.measures['ejected'] == 2


def test_clean_point_stays_complete(tmp_path):
    _afm_tree(tmp_path)
    p = _point(DataReader(str(tmp_path)), 30.0)
    assert p.status == 'complete', p.status_detail
    assert p.integrity.measures['hole'] < 1.5 and p.integrity.measures['n_heat'] == 512


def test_integrity_off(tmp_path):
    _afm_tree(tmp_path, eject=(30.0,))
    r = DataReader(str(tmp_path), integrity=False)
    p = _point(r, 30.0)
    assert p.status == 'complete' and p.integrity is None
    assert r.metadata['integrity_thresholds'] is None


def test_stage3_never_creates_not_reached(tmp_path):     # AFM chain; 30 nN fails, 40 nN complete
    _afm_tree(tmp_path, loads=(30.0, 40.0), eject=(30.0,))
    r = DataReader(str(tmp_path))
    assert _point(r, 30.0).status == 'failed_integrity'
    assert _point(r, 40.0).status == 'complete'


def test_multi_speed_skips_snapshot_rules(tmp_path):     # Review Focus 2
    _afm_tree(tmp_path, speeds=(1.0, 2.0), eject=(30.0,))
    r = DataReader(str(tmp_path))
    assert len(r.runs) == 2
    assert all(p.status == 'complete' for p in r.runs)
    assert all('multi-speed' in p.status_detail for p in r.runs)


def test_no_end_snapshot_deck_is_not_applicable(tmp_path):
    """R5: a run whose decks write no final_ snapshot is not quarantined."""
    run_dir = _afm_tree(tmp_path, deck=False)
    for path in (run_dir / 'data').glob('final_*'):
        path.unlink()
    p = _point(DataReader(str(tmp_path)), 30.0)
    assert p.status == 'complete'
    assert NO_SNAPSHOT_DETAIL in p.integrity.not_applicable
    assert NO_SNAPSHOT_DETAIL in p.status_detail


def test_deck_writes_snapshot_but_file_absent_quarantines(tmp_path):
    run_dir = _afm_tree(tmp_path)
    (run_dir / 'data' / 'final_f30N_a0.data').unlink()
    p = _point(DataReader(str(tmp_path)), 30.0)
    assert p.status == 'quarantined' and 'missing final snapshot' in p.status_detail


def test_export_carries_integrity(tmp_path):
    _afm_tree(tmp_path)
    r = DataReader(str(tmp_path))
    (size, runs), = r.points_by_size().items()
    pay = r.export_payload(size, runs)
    assert pay['metadata']['integrity_thresholds']['hole_bonds'] == 1.5
    assert pay['runs'][0]['integrity']['observed'] is True
    assert pay['metadata']['status_counts']['failed_integrity'] == 0
    json.dumps(pay['runs'][0]['integrity'], allow_nan=False)


def test_failed_integrity_excluded_from_complete_only(tmp_path):
    _afm_tree(tmp_path, loads=(30.0, 40.0), eject=(30.0,))
    r = DataReader(str(tmp_path), complete_only=True)
    kept = [p for ps in r.points_by_size().values() for p in ps]
    assert [p.load for p in kept] == [40.0]
    assert [p.load for p in r.runs] == [40.0]
    assert [row['load'] for row in r.summary_rows()] == [40.0]
    assert r.metadata['status_counts']['failed_integrity'] == 1


def test_hetero_layer_count_comes_from_config(tmp_path):
    """R6: an empty layer map falls back on the config's sheet stack, not 2."""
    config = {'general': {'temp': 300},
              'sheets': [{'layers': [2]}, {'layers': [2]}]}
    run_dir = make_family_run(tmp_path, 'hetero', material='h-MoS2',
                              second_material='h-WS2', config=config)
    _set_layer_map(run_dir, {})
    loc, = discover_runs(tmp_path)
    assert resolve_n_layers(loc, config) == 4
    assert resolve_n_layers(loc, {'general': {}, 'sheet': {'layers': [3]}}) == 3
    assert resolve_n_layers(loc, None) == 2


# --- check_files / resolve_layers (Task 3 review carry-overs) ---------------

def _temp_window(n=60):
    return pd.DataFrame({'temp_sheet_3d': 300.0 + np.linspace(0.0, 10.0, n)})


@pytest.mark.parametrize('status', ['halted', 'truncated'])
def test_t_sigma_for_non_complete_points(tmp_path, status):
    """s48 :191 / s51 :169 heat every point, so n_heat comes from the reference."""
    stack, box = _stack(2)
    ref = _write_data(tmp_path / 'load.data', stack, box)
    v = check_files(status, _temp_window(), 'afm', 2, 300.0, None, ref,
                    {'layer_1': {1}, 'layer_2': {2}}, S)
    assert v.status == status and v.measures['n_heat'] == 512
    assert np.isfinite(v.measures['T_sigma'])


def test_check_files_without_end_snapshot_is_not_quarantined(tmp_path):
    stack, box = _stack(2)
    ref = _write_data(tmp_path / 'load.data', stack, box)
    v = check_files('complete', _temp_window(), 'afm', 2, 300.0, None, ref, {}, S,
                    writes_snapshot=False)
    assert v.status == 'complete' and v.quarantine == []
    assert NO_SNAPSHOT_DETAIL in v.not_applicable
    assert {'hole', 'isolated_atoms', 'ejected_atoms'} <= set(v.not_applicable)


def test_zband_fallback_is_sheet_only(tmp_path):
    """An AFM reference holds tip and substrate atoms: no z-band guess."""
    run_dir = make_family_run(tmp_path, 'afm')
    _set_layer_map(run_dir, {})
    stack, box = _stack(2)
    ref = _write_data(run_dir / 'data' / 'load_30N.data', stack, box)
    loc, = discover_runs(tmp_path)
    assert resolve_layers(loc, {}, ref, 2) == {}
    assert resolve_layers(loc, {}, ref, 2, family='sheetonsheet') == {
        'layer_1': {1}, 'layer_2': {2}}


def test_writes_end_snapshot(tmp_path):
    lammps = tmp_path / 'lammps'
    assert writes_end_snapshot(lammps) is False      # no directory
    lammps.mkdir()
    (lammps / 'system.in').write_text('write_data ./data/load_$(v_find)N.data\n',
                                      encoding='utf-8')
    assert writes_end_snapshot(lammps) is False
    (lammps / 'slide.in').write_text(
        '# write_data ./data/final_x.data\n', encoding='utf-8')
    assert writes_end_snapshot(lammps) is False      # a comment is not a command
    (lammps / 'slide.in').write_text(
        'write_data      /abs/data/final_p$(v_pressure)GPa_a$(v_a).data\n',
        encoding='utf-8')
    assert writes_end_snapshot(lammps) is True
    (lammps / 'slide.in').unlink()
    data = tmp_path / 'data'
    assert writes_end_snapshot(lammps, data) is False
    data.mkdir()
    (data / 'final_f30N_a0.data').write_text('x', encoding='utf-8')
    assert writes_end_snapshot(lammps, data) is True   # R9: the file is enough


def test_complete_only_time_series_from_complete_runs(tmp_path):
    """ntimestep is taken after complete_only narrows runs."""
    from tests.helpers.synthetic_results import write_results
    run_dir = _afm_tree(tmp_path, loads=(30.0, 40.0), eject=(30.0,))
    write_results(run_dir, 'afm', load_kind='force', loads=(30.0,), layers=2, n=60)
    r = DataReader(str(tmp_path), complete_only=True)
    assert [p.load for p in r.runs] == [40.0]
    assert r.ntimestep == 50 and len(r.time_series) == 50
