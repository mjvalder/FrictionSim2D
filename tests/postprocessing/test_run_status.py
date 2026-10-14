"""Every expected load point gets a status tag; nothing is dropped silently.

The grid comes from each run's `provenance/config.json` (`slide_config` here),
not from what is on disk, so a load that never wrote a file is still exported
-- as `no_output`, or `not_reached` when an earlier load of the same chained
script already failed.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.postprocessing.read_data import DataReader
from src.postprocessing.run_status import (classify, expected_chains,
                                           mark_not_reached, point_key)
from tests.helpers.results_tree import make_family_run, slide_config
from tests.helpers.synthetic_results import write_results

LOADS5 = (10, 20, 30, 40, 50)


def _statuses(reader):
    return [p.status for p in reader.points]


def _cut_short_tree(root: Path) -> Path:
    """afm loads 10..50: load 30 stops at 20/50 rows, 40 and 50 never ran."""
    run_dir = make_family_run(root, 'afm', loads=LOADS5,
                              config=slide_config('afm', loads=LOADS5), n=50)
    write_results(run_dir, 'afm', load_kind='force', loads=(30,), layers=2,
                  n=20)
    for load in (40, 50):
        for path in (run_dir / 'results').glob(f'friction_f{load}_*'):
            path.unlink()
    return run_dir


# --- the pure helpers ------------------------------------------------------

def test_point_key_agrees_across_spellings():
    assert point_key('force', 10, 0, 2) == point_key('force', 10.0, 0.0, '2')


def test_classify():
    assert classify({'window': 50}, {'window': 50}) == ('complete', '')
    assert classify({'window': 49}, {'window': 50})[0] == 'complete'
    assert classify({'window': 20}, {'window': 50}) == (
        'truncated', 'window 20/50 rows')
    assert classify({'raw': 500}, {'window': 50, 'raw': 500})[0] == 'truncated'
    assert classify({'window': 3}, {}) == ('complete', '')


def test_mark_not_reached_only_after_a_failure():
    chain = [point_key('force', x, 0, 2) for x in (10, 20, 30)]
    status = dict(zip(chain, ('no_output', 'complete', 'no_output')))
    mark_not_reached(chain, status)
    assert [status[k] for k in chain] == ['no_output', 'complete', 'not_reached']


def test_expected_chains_orders_speed_then_load_then_angle():
    cfg = slide_config('afm', loads=(10, 20), angles=(0, 90), speeds=(1, 2))
    chain, = expected_chains(cfg, 'afm')
    assert chain[:3] == [point_key('force', 10, 0, 1),
                         point_key('force', 10, 90, 1),
                         point_key('force', 20, 0, 1)]
    assert len(chain) == 8


def test_expected_chains_unknown_or_unmodelled():
    assert expected_chains(None, 'afm') is None
    assert expected_chains({'general': {'temp': 300}}, 'afm') is None
    assert expected_chains(slide_config('afm', loads=(10,)), 'indent') is None
    assert expected_chains(None, 'pes_tip') == [[('none', 0.0, 0.0, 0.0)]]


# --- through the reader ------------------------------------------------------

def test_all_loads_complete(tmp_path):
    make_family_run(tmp_path, 'afm', loads=(10, 20, 30), n=50,
                    config=slide_config('afm', loads=(10, 20, 30)))

    reader = DataReader(results_dir=str(tmp_path))

    assert _statuses(reader) == ['complete'] * 3
    for point in reader.points:
        assert point.expected_rows == {'window': 50}
        assert point.n_rows == {'window': 50}


def test_a_run_cut_short_at_load_3_of_5(tmp_path):
    _cut_short_tree(tmp_path)

    reader = DataReader(results_dir=str(tmp_path))

    assert [p.load for p in reader.points] == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _statuses(reader) == ['complete', 'complete', 'truncated',
                                 'not_reached', 'not_reached']
    truncated = reader.points[2]
    assert truncated.status_detail == 'window 20/50 rows'
    assert set(truncated.tiers) == {'window'}
    assert len(truncated.window) == 20


def test_one_row_short_is_complete(tmp_path):
    make_family_run(tmp_path, 'afm', loads=(10,), n=49,
                    config=slide_config('afm', loads=(10,)))

    point, = DataReader(results_dir=str(tmp_path)).points

    assert point.status == 'complete'


def test_an_unreadable_window_file(tmp_path):
    run_dir = make_family_run(tmp_path, 'sheetonsheet', loads=(1, 2, 3),
                              config=slide_config('sheetonsheet',
                                                  loads=(1, 2, 3)))
    (run_dir / 'results' / 'friction_p2_a0_s2.txt').write_text(
        '# schema=2 lf_units=nN family=sheetonsheet\n', encoding='utf-8')

    reader = DataReader(results_dir=str(tmp_path))

    assert _statuses(reader) == ['complete', 'unreadable', 'complete']
    bad = reader.points[1]
    assert bad.tiers == {}
    assert 'no column header' in bad.status_detail


def test_empty_results_dir(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm', loads=(10, 20, 30),
                              config=slide_config('afm', loads=(10, 20, 30)))
    for path in (run_dir / 'results').iterdir():
        path.unlink()

    reader = DataReader(results_dir=str(tmp_path))

    assert _statuses(reader) == ['no_output', 'not_reached', 'not_reached']
    assert reader.runs == []


def test_outer_loop_splits_chains(tmp_path):
    make_family_run(tmp_path, 'afm', loads=(10,),
                    config=slide_config('afm', loads=(10, 20, 30),
                                        outer_loop='force'))

    reader = DataReader(results_dir=str(tmp_path))

    assert _statuses(reader) == ['complete', 'no_output', 'no_output']


def test_scan_angle_force_gating(tmp_path):
    make_family_run(tmp_path, 'afm', loads=(10, 20), angles=(0, 90),
                    config=slide_config('afm', loads=(10, 20), angles=(0, 90),
                                        scan_angle_force=[20]))

    reader = DataReader(results_dir=str(tmp_path))

    by_point = {(p.load, p.angle): p for p in reader.points}
    assert sorted(by_point) == [(10.0, 0.0), (10.0, 90.0), (20.0, 0.0),
                                (20.0, 90.0)]
    stray = by_point[(10.0, 90.0)]
    assert 'not in configured grid' in stray.status_detail
    for key in ((10.0, 0.0), (20.0, 0.0), (20.0, 90.0)):
        assert 'not in configured grid' not in by_point[key].status_detail
        assert by_point[key].status == 'complete'


def test_float_spelling_agrees(tmp_path):
    cfg = slide_config('afm', loads=(10,))
    cfg['general']['force'] = [10.0, 20]
    make_family_run(tmp_path, 'afm', loads=(10, 20), config=cfg)

    reader = DataReader(results_dir=str(tmp_path))

    assert _statuses(reader) == ['complete', 'complete']
    assert all('not in configured grid' not in p.status_detail
               for p in reader.points)


def test_grid_unknown_without_settings(tmp_path):
    make_family_run(tmp_path / 'a', 'afm', loads=(10, 20))
    reader = DataReader(results_dir=str(tmp_path / 'a'))
    assert len(reader.points) == 2
    assert all(p.status_detail.startswith('grid unknown')
               for p in reader.points)

    run_dir = make_family_run(tmp_path / 'b', 'afm', loads=(10, 20))
    (run_dir.parent / 'provenance' / 'config.json').write_text(
        '{not json', encoding='utf-8')
    reader = DataReader(results_dir=str(tmp_path / 'b'))
    assert len(reader.points) == 2
    assert all(p.status_detail.startswith('grid unknown')
               for p in reader.points)
    assert all(p.config is None for p in reader.points)


def test_pes_without_grid_file_is_no_output(tmp_path):
    run_dir = make_family_run(tmp_path, 'pes_sheet')
    (run_dir / 'results' / 'pes_scan.csv').unlink()

    point, = DataReader(results_dir=str(tmp_path)).points

    assert point.load_kind == 'none'
    assert point.status == 'no_output'


def test_indent_is_on_disk_only(tmp_path):
    make_family_run(tmp_path, 'indent', loads=(10,),
                    config=slide_config('afm', loads=(10, 20)))

    point, = DataReader(results_dir=str(tmp_path)).points

    assert 'grid not modelled for indent' in point.status_detail


def test_runs_keeps_only_points_with_frames(tmp_path):
    _cut_short_tree(tmp_path)

    reader = DataReader(results_dir=str(tmp_path))

    assert len(reader.runs) == 3
    assert len(reader.points) == 5


def test_payload_carries_status_keys(tmp_path):
    _cut_short_tree(tmp_path)

    reader = DataReader(results_dir=str(tmp_path))

    for point in reader.points:
        payload = reader.run_payload(point)
        assert {'status', 'status_detail', 'n_rows',
                'expected_rows'} <= set(payload)
    empty = reader.run_payload(reader.points[-1])
    assert empty['tiers'] == {}
    assert empty['n_rows'] == {}
    assert empty['expected_rows'] == {'window': 50}


def test_export_writes_every_point_unless_complete_only(tmp_path):
    _cut_short_tree(tmp_path)

    written, = DataReader(results_dir=str(tmp_path)).export_full_data_to_json()
    doc = json.loads(written.read_text(encoding='utf-8'))
    assert [r['status'] for r in doc['runs']] == [
        'complete', 'complete', 'truncated', 'not_reached', 'not_reached']

    written, = DataReader(results_dir=str(tmp_path),
                          complete_only=True).export_full_data_to_json()
    doc = json.loads(written.read_text(encoding='utf-8'))
    assert [r['status'] for r in doc['runs']] == ['complete', 'complete']


def test_incomplete_report_lists_every_non_complete_point(tmp_path):
    run_dir = _cut_short_tree(tmp_path)

    reader = DataReader(results_dir=str(tmp_path))
    reader.export_issue_reports()

    out = tmp_path / 'outputs'
    lines = (out / 'incomplete_files_100x100y.txt').read_text(
        encoding='utf-8').splitlines()
    assert len(lines) == 3
    assert sorted(line.rsplit(' ', 1)[1] for line in lines) == [
        '[not_reached]', '[not_reached]', '[truncated]']
    assert any(line.startswith(str(run_dir / 'results' / 'friction_f30_'))
               for line in lines)
    assert f'{run_dir} force=40 a=0 s=2 [not_reached]' in lines
    assert (out / 'incomplete_materials_100x100y.txt').read_text(
        encoding='utf-8').strip() == 'h_MoS2'
