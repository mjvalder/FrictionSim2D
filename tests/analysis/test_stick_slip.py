"""`stick_slip` (plan Task 7): ring period, lateral-force period, slips and the
energy oscillation of each AFM run (s39_stickslip_v2 + taxlib plausibility)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scipy')

import src.analysis.catalogue  # noqa: E402,F401  (registers the analyses)
from src.analysis.base import get_analysis  # noqa: E402
from src.analysis.source import RunSource  # noqa: E402
from src.analysis.stick_slip import (BIN_A, on_displacement, period_from_acf,  # noqa: E402
                                     ring_period, slips, slips_per_period_taxlib)


def test_ring_period_of_a_sinusoid():
    i = np.arange(4000, dtype=float)
    x = 50.0 + 0.0025 * i + 0.5 * np.sin(2 * np.pi * i / 80.0)   # a drive ramp + a ring of 80
    assert ring_period(x) == pytest.approx(80.0)
    # outside the 30-150 sample bounds the ring is not seen: the bounded maximum is taken
    assert 30 < ring_period(50.0 + 0.5 * np.sin(2 * np.pi * i / 10.0)) < 150


def test_lf_period_of_a_sawtooth_equals_its_period():
    a = 3.0
    x = np.linspace(0.0, 10.0, 5001)                 # the tip advances steadily over 10 A
    f = (x % a) / a                                  # a sawtooth of period a in displacement
    grid, y = on_displacement(x, f)
    period, r_peak, clear = period_from_acf(grid, y)
    assert period == pytest.approx(a, abs=BIN_A)
    assert clear and r_peak > 0.9


def test_slip_count_of_a_clean_sawtooth():
    a = 3.0
    i = np.arange(400)
    fs = (i % 100) / 100.0                           # 4 teeth: 3 drops inside, the last ends high
    xs = a * (i // 100).astype(float)                # stuck, then one lattice period per drop
    drops, adv, thr = slips(fs, xs, a)
    assert len(drops) == 3
    assert drops == pytest.approx([0.99] * 3)
    assert adv == pytest.approx([a] * 3)
    assert 0 < thr < 0.99
    # a jump shorter than MIN_SLIP_A lattice periods is no slip
    assert len(slips(fs, xs * 0.2, a)[0]) == 0


def test_taxlib_slips_per_period_of_a_stepped_tip():
    a, dx = 3.0, 0.02
    rng = np.random.default_rng(0)
    steps = dx + 0.001 * rng.standard_normal(499)     # drive + thermal jitter
    steps[[100, 250, 400]] += a                        # three slips of one lattice period
    tipx = np.concatenate([[50.0], 50.0 + np.cumsum(steps)])
    L = dx * len(tipx)
    slip_len, per_period = slips_per_period_taxlib(tipx, dx, a, L)
    assert per_period == pytest.approx(3 / (L / a))
    assert slip_len == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------
# the analysis over a fake fetcher
# --------------------------------------------------------------------------

def _run_frames(a=3.0, n_raw=5001):
    """A stick-slip run: the tip sticks, then slips one lattice period every
    ``a / dx`` raw samples, ringing after each slip; the force is the spring
    stretch. Raw every 100 MD steps, window every 1000, peratom every 1000."""
    dx_raw = 0.002                                     # A per raw sample (2 m/s, 1 fs, 100 steps)
    t = np.arange(n_raw) * 100
    drive = dx_raw * np.arange(n_raw)
    tip = a * np.floor(drive / a)
    since = np.arange(n_raw) - np.searchsorted(drive, tip, side='left')
    tip = tip + 0.3 * np.exp(-since / 200.0) * np.sin(2 * np.pi * since / 80.0)
    lfx = 0.8 * (drive - tip)
    raw = pd.DataFrame({'time': t, 'raw_tipx': 50.0 + tip, 'raw_lfx': lfx})
    win = raw.iloc[10::10].reset_index(drop=True).rename(columns={'raw_tipx': 'tipx',
                                                                    'raw_lfx': 'lfx'})
    pa = pd.DataFrame({'time': t[::10], 'E_int_tip_sheet': -20.0 + np.cos(2 * np.pi
                                                                          * tip[::10] / a)})
    return {'window': win, 'raw': raw, 'peratom': pa}


class _Fetch:
    def __init__(self, frames):
        self.data = frames            # point_uuid -> frames

    def frames(self, row):
        return self.data[row.point_uuid]

    def run_fields(self, row):
        return {'settings.simulation.timestep': 0.001}

    def close(self):
        pass


def _row(uuid, load=10.0, status='complete', angle=0.0):
    return {'point_uuid': uuid, 'run_uuid': f'r-{uuid}', 'calc_uuid': 'c', 'node_uuid': uuid,
            'verdict_status': status, 'family': 'afm', 'materials': ['h-MoS2'],
            'flake_material': None, 'size_x': 100.0, 'size_y': 100.0, 'layers': 2,
            'load': load, 'load_kind': 'force', 'angle': angle, 'speed': 2.0}


def _stick_slip(rows, frames, **params):
    cls = get_analysis('stick_slip')
    source = RunSource(pd.DataFrame(rows), 'afm', fetch=_Fetch(frames))
    params.setdefault('period_x', {'h_MoS2': 3.0})
    return cls().compute(source, cls.Params(**params)), source


def test_stick_slip_of_a_lattice_stick_slip_run():
    table, source = _stick_slip([_row('p1')], {'p1': _run_frames()})
    assert source.dropped == []
    row = table.iloc[0]
    assert row['material'] == 'h_MoS2' and row['load'] == 10.0 and row['a_x_A'] == 3.0
    assert row['ring_period_samples'] == pytest.approx(80.0, rel=0.05)
    assert row['lf_period_clear'] and row['lf_period_A'] == pytest.approx(3.0, abs=2 * BIN_A)
    assert row['locked']
    assert row['n_slips'] == 3                       # 10 A of drive: slips at 3, 6 and 9 A
    assert row['slip_len_a_lf'] == pytest.approx(1.0, abs=0.15)
    assert row['E_periods'] >= 2 and row['E_amp_eV'] > 0
    assert row['slips_per_period_taxlib'] == pytest.approx(3 / (10.0 / 3.0), rel=0.05)
    expected = {'ring_period_samples', 'travel_A', 'lf_period_A', 'lf_acf_peak',
                'lf_period_clear', 'period_ratio', 'r_at_a', 'locked', 'n_slips',
                'slip_threshold_nN', 'slips_per_period_lf', 'drop_per_slip_nN',
                'slip_len_a_lf', 'E_amp_eV', 'E_periods', 'slips_per_period_taxlib'}
    assert expected <= set(table.columns)
    assert not any(c.startswith('pes_') or c == 'cof' for c in table.columns)


def test_stick_slip_drops_runs_without_raw_or_peratom_and_reads_the_window():
    full = _run_frames()
    no_raw = {k: v for k, v in full.items() if k != 'raw'}
    no_pa = {k: v for k, v in full.items() if k != 'peratom'}
    rows = [_row('p1'), _row('p2', load=20.0), _row('p3', load=30.0),
            _row('p4', load=50.0, status='ejected'), _row('p5', load=5.0),
            _row('p6', angle=90.0)]
    frames = {'p1': full, 'p2': no_raw, 'p3': no_pa, 'p4': full, 'p5': full, 'p6': full}
    table, source = _stick_slip(rows, frames)
    assert list(table['point_uuid']) == ['p1']
    assert sorted(source.dropped) == [('p2', 'no_raw_tier'), ('p3', 'no_peratom_tier'),
                                      ('p4', 'ejected')]
    table, _ = _stick_slip(rows, frames, loads='5,10')
    assert list(table['point_uuid']) == ['p1', 'p5']


def test_stick_slip_drops_a_run_without_a_lattice_period():
    table, source = _stick_slip([_row('p1')], {'p1': _run_frames()}, period_x={'h_X': 3.0})
    assert table.empty
    assert source.dropped == [('p1', 'no_lattice_period')]


def test_taxlib_lattice_overrides_a_x_for_the_taxlib_count_only():
    base, _ = _stick_slip([_row('p1')], {'p1': _run_frames()})
    other, _ = _stick_slip([_row('p1')], {'p1': _run_frames()}, taxlib_a={'h_MoS2': 6.0})
    assert other.iloc[0]['slips_per_period_taxlib'] == pytest.approx(
        2 * base.iloc[0]['slips_per_period_taxlib'])
    assert other.iloc[0]['slips_per_period_lf'] == base.iloc[0]['slips_per_period_lf']


# --------------------------------------------------------------------------
# publication check: the read-only `friction2d_pub` store against the frozen file
# --------------------------------------------------------------------------

_ANALYSIS = Path('/home/matteo/mnt/PhD/Sync/Publication-data/publication_export/data/'
                 '260909_afm_v4/analysis')
_FROZEN = _ANALYSIS / 'stickslip_v2.csv'
_LATTICE = Path('/home/matteo/mnt/PhD/Sync/ML_taxonomy/baseline_inputs/descriptors/'
                'materials_database.csv')

_PUB_SCRIPT = """
import json, sys
import numpy as np
from src.aiida import load_aiida_profile
load_aiida_profile('friction2d_pub')
from src.analysis.base import get_analysis
from src.analysis.source import RunSource, StoreFetcher
dataset, params, out32, out64 = sys.argv[1:5]

class Cached:
    '''Reads each point's tiers once; ``f32`` rounds them to float32, as the
    ML_taxonomy parquet frames (afm2_raw/_peratom/_long) held them.'''
    def __init__(self, inner):
        self.inner, self.cache, self.f32 = inner, {}, False
    def frames(self, row):
        if row.point_uuid not in self.cache:
            self.cache[row.point_uuid] = self.inner.frames(row)
        frames = self.cache[row.point_uuid]
        if not self.f32:
            return frames
        return {k: v.astype({c: np.float32 for c in v.columns if c != 'time'}).astype(
                    {c: float for c in v.columns if c != 'time'}) for k, v in frames.items()}
    def __getattr__(self, name):
        return getattr(self.inner, name)

cls = get_analysis('stick_slip')
fetch = Cached(StoreFetcher())
with RunSource.from_dataset(dataset) as base:
    for f32, out in ((True, out32), (False, out64)):
        fetch.f32 = f32
        source = RunSource(base.all_rows, None, fetch)
        cls().compute(source, cls.Params(**json.loads(params))).to_csv(
            out, index=False, float_format=repr)
"""


def _from_pub_store(tmp_path, dataset, params):
    """`stick_slip` over ``dataset`` of the read-only `friction2d_pub` store,
    in a subprocess (the aiida pytest plugin hides the real profile from this
    process): once on float32-rounded tiers, once as stored. Computed, never
    stored."""
    pytest.importorskip('aiida')
    import json  # pylint: disable=import-outside-toplevel
    out32, out64 = tmp_path / 'ss32.csv', tmp_path / 'ss64.csv'
    env = {k: v for k, v in os.environ.items() if k != 'AIIDA_PATH'}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, '-c', _PUB_SCRIPT, dataset, json.dumps(params),
                           str(out32), str(out64)],
                          env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        pytest.skip(f'friction2d_pub / dataset {dataset} unavailable: {done.stderr[-300:]}')
    return (pd.read_csv(out32, float_precision='round_trip'),
            pd.read_csv(out64, float_precision='round_trip'))


#: Compared within 1e-9 (relative to max(1, |value|)); the integer/bool ones exactly.
_FLOATS = ['a_x_A', 'ring_period_samples', 'travel_A', 'lf_period_A', 'lf_acf_peak',
           'period_ratio', 'r_at_a', 'slips_per_period_lf', 'drop_per_slip_nN', 'slip_len_a_lf',
           'slip_threshold_nN', 'E_amp_eV', 'slips_per_period_taxlib']
_EXACT = ['lf_period_clear', 'n_slips', 'E_periods', 'locked']


def _close(a, b, tol=1e-9):
    if pd.isna(a) and pd.isna(b):
        return True
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def _compare(ours, frozen, tol):
    off = {}
    for k in ours.index.intersection(frozen.index):
        got, want = ours.loc[k], frozen.loc[k]
        bad = {c: (got[c], want[c]) for c in _FLOATS if not _close(got[c], want[c], tol)}
        bad.update({c: (got[c], want[c]) for c in _EXACT if bool(got[c] != want[c])})
        if bad:
            off[k] = bad
    return off


@pytest.mark.publication
def test_stick_slip_matches_the_frozen_table_on_shared_rows(tmp_path):
    if not (_FROZEN.exists() and _LATTICE.exists()):
        pytest.skip('frozen stickslip_v2.csv / materials_database.csv absent')
    sys.path.insert(0, str(_LATTICE.parents[2]))
    from runtax.ids import json_to_cif  # pylint: disable=import-outside-toplevel
    frozen = pd.read_csv(_FROZEN, float_precision='round_trip')
    db = pd.read_csv(_LATTICE)
    lat = dict(zip(db['material_id'], db['lat_a'].astype(float)))
    # s39's inputs: a_x from the frozen ranking (s39:208) and taxlib's lattice
    # (taxlib_adapter.py:46-51), so the comparison tests the measures alone.
    a_x = frozen.drop_duplicates('material').set_index('material')['a_x_A'].to_dict()
    every = pd.read_csv(_ANALYSIS / 'cof_ranking_v2.csv', float_precision='round_trip')
    a_x.update({m: v for m, v in zip(every['material'], every['a_x_A']) if m not in a_x
                and np.isfinite(v)})
    params = {'period_x': a_x,
              'taxlib_a': {m: lat[json_to_cif(m)] for m in a_x if json_to_cif(m) in lat}}
    t32, t64 = _from_pub_store(tmp_path, '260909-afm', params)
    key = ['material', 'load']
    frozen = frozen.assign(load=frozen['load_nN'].astype(float)).set_index(key)
    ours32 = t32[t32['layer'] == 2].set_index(key)
    ours64 = t64[t64['layer'] == 2].set_index(key)
    assert not ours32.index.duplicated().any()

    shared = ours32.index.intersection(frozen.index)
    only_frozen = frozen.index.difference(ours32.index)
    only_ours = ours32.index.difference(frozen.index)
    off32 = _compare(ours32, frozen, 1e-9)
    off64 = _compare(ours64, frozen, 1e-9)
    dev = {c: float(np.nanmax(np.abs(ours64.loc[shared, c].astype(float)
                                     - frozen.loc[shared, c].astype(float))))
           for c in _FLOATS}
    mats_frozen = sorted(set(frozen.index.get_level_values(0))
                         - set(ours32.index.get_level_values(0)))
    mats_ours = sorted(set(ours32.index.get_level_values(0))
                       - set(frozen.index.get_level_values(0)))
    print(f'\nours {len(ours32)} layer-2 rows, frozen {len(frozen)}, shared {len(shared)}; '
          f'frozen-only rows {len(only_frozen)} {list(only_frozen)}; ours-only rows '
          f'{len(only_ours)}')
    print(f'materials only frozen: {mats_frozen}; only ours: {mats_ours}')
    print(f'float32 tiers off: {len(off32)} {dict(list(off32.items())[:10])}')
    flips = {k: {c: v for c, v in bad.items() if c in _EXACT} for k, bad in off64.items()
             if any(c in _EXACT for c in bad)}
    print(f'stored float64 tiers off at 1e-9: {len(off64)}; max |diff| {dev}; '
          f'integer/bool flips: {len(flips)} {flips}')
    # p_CTe's bilayer is absent from the store (Task 4 ruling).
    assert {k[0] for k in only_frozen} <= {'p_CTe'}
    assert not off32
