"""`integrity_traces` and `run_status` (plan Task 8): per-sample traces and the
verdict status of every point, failed ones included."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.catalogue  # noqa: F401  (registers the analyses)
from src.analysis.base import get_analysis
from src.analysis.source import RunSource

MEASURES = {'hole': 1.5, 'isolated': 0, 'ejected': 0, 'T_sigma': 0.4, 'gap_rise': 0.1}


class _Fetch:
    def __init__(self, windows, measures=None):
        self.windows, self.measures = windows, measures or {}

    def frames(self, row):
        if row.point_uuid not in self.windows:
            return {}
        return {'window': self.windows[row.point_uuid]}

    def verdict(self, row):
        m = self.measures.get(row.point_uuid)
        return {} if m is None else {'status': row.verdict_status, 'measures': m}

    def close(self):
        pass


def _window(n=4, gap=3.0, t=300.0):
    return pd.DataFrame({'time': 1000 * np.arange(1, n + 1), 'nf': 1.0,
                         'temp_sheet_3d': t + np.arange(n, dtype=float),
                         'comz_L1': np.zeros(n), 'comz_L2': gap + np.arange(n) * 0.1})


def _row(uuid, load, status='complete', material='h-M', layers=2):
    return {'point_uuid': uuid, 'run_uuid': f'r-{uuid}', 'calc_uuid': 'c', 'node_uuid': uuid,
            'verdict_status': status, 'family': 'afm', 'materials': [material],
            'flake_material': None, 'size_x': 100.0, 'size_y': 100.0, 'layers': layers,
            'load': load, 'load_kind': 'force', 'angle': 0.0, 'speed': 2.0}


def _source():
    rows = [_row('a', 10.0), _row('b', 20.0, 'failed_integrity'), _row('c', 30.0, 'halted'),
            _row('d', 40.0, 'not_reached')]
    windows = {'a': _window(), 'b': _window(gap=5.0)}
    measures = {'a': MEASURES, 'b': {**MEASURES, 'hole': 9.0}}
    return RunSource(pd.DataFrame(rows), 'afm', fetch=_Fetch(windows, measures))


def _compute(name, source):
    cls = get_analysis(name)
    return cls().compute(source, cls.Params())


def test_integrity_traces_hold_every_sample_with_the_stored_measures():
    source = _source()
    table = _compute('integrity_traces', source)
    assert list(table.columns[:6]) == ['run_id', 'step', 'temp_sheet_3d', 'comz_L1',
                                       'comz_L2', 'gap']
    assert set(table['run_id']) == {'h_M/l2/f10/a0', 'h_M/l2/f20/a0'}
    one = table[table['run_id'] == 'h_M/l2/f10/a0']
    assert one['step'].tolist() == [0, 1, 2, 3]
    assert one['gap'].tolist() == pytest.approx([3.0, 3.1, 3.2, 3.3])
    assert one['temp_sheet_3d'].tolist() == [300.0, 301.0, 302.0, 303.0]
    assert set(one['hole']) == {1.5} and set(one['T_sigma']) == {0.4}
    bad = table[table['run_id'] == 'h_M/l2/f20/a0']
    assert set(bad['verdict_status']) == {'failed_integrity'} and set(bad['hole']) == {9.0}
    # a failed point is shown; a point with no window (halted, not reached) is dropped
    assert sorted(source.dropped) == [('c', 'no_window_tier'), ('d', 'no_window_tier')]


def test_integrity_traces_keep_only_the_materials_and_layer_asked_for():
    rows = [_row('a', 10.0), _row('b', 10.0, material='h-N'), _row('c', 10.0, layers=1),
            _row('e', 10.0, material='h-X')]
    windows = {u: _window() for u in 'abce'}
    source = RunSource(pd.DataFrame(rows), 'afm', fetch=_Fetch(windows))
    cls = get_analysis('integrity_traces')

    table = cls().compute(source, cls.Params(materials='["h_M", "h-N"]', layer=2))
    assert sorted(set(table['run_id'])) == ['h_M/l2/f10/a0', 'h_N/l2/f10/a0']
    assert source.dropped == []                 # left out, not dropped
    every = cls().compute(source, cls.Params())
    assert len(set(every['run_id'])) == 4
    assert cls.Params(materials='h_M,h_N').materials == ['h_M', 'h_N']


def test_run_status_has_one_row_per_point_and_drops_none():
    source = _source()
    assert len(source.dropped) == 3          # RunSource itself records the non-pass points
    table = _compute('run_status', source)
    assert list(table.columns) == ['run_id', 'material', 'layer', 'force', 'angle', 'status',
                                   'verdict']
    assert table['run_id'].tolist() == ['h_M/l2/f10/a0', 'h_M/l2/f20/a0', 'h_M/l2/f30/a0',
                                        'h_M/l2/f40/a0']
    # status: the run's completion (a failed-integrity run finished); verdict: the integrity one
    assert table['status'].tolist() == ['complete', 'complete', 'halted', 'not_reached']
    assert table['verdict'].tolist() == ['complete', 'failed_integrity', 'halted', 'not_reached']
    assert table['force'].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert source.dropped == [], 'run_status drops nothing: the non-pass points are its rows'


def test_completion_status_maps_integrity_verdicts_to_complete():
    from src.analysis.integrity_traces import completion_status
    assert [completion_status(v) for v in ('complete', 'failed_integrity', 'quarantined',
                                           'halted', 'no_output')] == \
        ['complete', 'complete', 'complete', 'halted', 'no_output']


def test_a_pressure_load_is_no_force():
    row = _row('p', 5.0)
    row['load_kind'] = 'pressure'
    source = RunSource(pd.DataFrame([row]), 'afm', fetch=_Fetch({}))
    table = _compute('run_status', source)
    assert table['run_id'].tolist() == ['h_M/l2/p5/a0'] and np.isnan(table['force'][0])


# --------------------------------------------------------------------------
# publication check
# --------------------------------------------------------------------------

_ANALYSIS = Path('/home/matteo/mnt/PhD/Sync/Publication-data/publication_export/data/'
                 '260909_afm_v4/analysis')

_PUB_SCRIPT = """
import sys
from src.aiida import load_aiida_profile
load_aiida_profile('friction2d_pub')
from src.analysis.base import get_analysis
from src.analysis.source import RunSource
dataset, traces, status = sys.argv[1:4]
with RunSource.from_dataset(dataset) as source:
    for name, out in (('integrity_traces', traces), ('run_status', status)):
        cls = get_analysis(name)
        cls().compute(source, cls.Params()).to_csv(out, index=False, float_format=repr)
"""


@pytest.mark.publication
def test_traces_and_status_match_the_frozen_tables(tmp_path):
    frozen_t, frozen_s = _ANALYSIS / 'integrity_traces_l2.csv', _ANALYSIS / 'run_status_l2.csv'
    if not (frozen_t.exists() and frozen_s.exists()):
        pytest.skip('frozen integrity_traces_l2.csv / run_status_l2.csv absent')
    pytest.importorskip('aiida')
    out_t, out_s = tmp_path / 't.csv', tmp_path / 's.csv'
    env = {k: v for k, v in os.environ.items() if k != 'AIIDA_PATH'}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, '-c', _PUB_SCRIPT, '260909-afm', str(out_t),
                           str(out_s)], env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        pytest.skip(f'friction2d_pub / dataset 260909-afm unavailable: {done.stderr[-300:]}')
    read = lambda p: pd.read_csv(p, float_precision='round_trip')  # noqa: E731
    ours_t, ours_s = read(out_t), read(out_s)
    want_t, want_s = read(frozen_t), read(frozen_s)
    mats = ['h_MoS2', 'h_NiSe2']

    ours_t = ours_t[ours_t['run_id'].str.split('/').str[0].isin(mats)
                    & ours_t['run_id'].str.contains('/l2/')]
    cols = ['run_id', 'step', 'temp_sheet_3d', 'comz_L1', 'comz_L2']
    merged = want_t.merge(ours_t[cols], on=['run_id', 'step'], how='outer',
                          suffixes=('_f', ''), indicator=True)
    print(f'\ntraces: frozen {len(want_t)}, ours {len(ours_t)}, '
          f'matched {(merged["_merge"] == "both").sum()}')
    assert (merged['_merge'] == 'both').all()
    for c in cols[2:]:
        dev = float(np.nanmax(np.abs(merged[c] - merged[c + '_f'])))
        print(f'  max |{c} - frozen| = {dev:.3g}')
        assert dev < 1e-3          # the frozen parquet frames held float32

    key = ['run_id']
    ours_s = ours_s.set_index(key)
    shared = want_s.set_index(key).join(ours_s, rsuffix='_ours', how='left')
    print(f'status (frozen, ours) counts over all frozen rows:\n'
          f'{shared.groupby(["status", "status_ours"], dropna=False).size()}')
    # status is the completion status of the frozen table; the store differs only where
    # it holds a crashed / unreached run the frozen file calls halted (listed, not accepted
    # as either value).
    known = {('halted', 'no_output'): 3, ('halted', 'not_reached'): 2}
    got = shared.dropna(subset=['status_ours'])
    diff = got[got['status'] != got['status_ours']]
    print(f'divergent: {diff.groupby(["status", "status_ours"]).size().to_dict()}')
    assert diff.groupby(['status', 'status_ours']).size().to_dict() == known
    mine = shared[shared.index.str.split('/').str[0].isin(mats)]
    assert (mine['status'] == mine['status_ours']).all()
    assert (mine['force'] == mine['force_ours']).all()
    assert (mine.loc[mine['material'] == 'h_MoS2', 'verdict'] == 'complete').all()
    assert (mine.loc[mine['material'] == 'h_NiSe2', 'verdict'] == 'failed_integrity').sum() == 5
