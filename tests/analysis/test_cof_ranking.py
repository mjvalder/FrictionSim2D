"""`period_summary` and `cof_ranking` (plan Task 4): the whole-period COF."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.catalogue  # noqa: F401  (registers the analyses)
from src.analysis.base import get_analysis
from src.analysis.periods import (NO_DRIVE_STEP, NO_PERIOD, PeriodParams, dominant_period,
                                  period_map, settle_skip, whole_periods)
from src.analysis.source import RunSource
from tests.analysis.test_sheet_geometry import SETTINGS, _bilayer, _square

#: dx = 2 m/s * 0.01 * (1000 steps * 0.001 ps) = 0.02 A per sample.
FIELDS = {'settings.simulation.timestep': 0.001, 'general.driving_spring': 8.0,
          'derived.drive_k': None}


def _window(n=500, lfx=0.0, nf=10.0, cof=None):
    time = 1000 * np.arange(1, n + 1)
    lfx = np.full(n, float(lfx)) if np.isscalar(lfx) else np.asarray(lfx, float)
    nf = np.full(n, float(nf))
    if cof is not None:
        lfx = np.asarray(cof, float) * nf
    return pd.DataFrame({'time': time, 'lfx': lfx, 'lfy': np.zeros(n), 'nf': nf})


class _Fetch:
    def __init__(self, windows, fields=None):
        self.windows = windows
        self.fields = fields or FIELDS

    raws = {}

    def frames(self, row):
        out = {'window': self.windows[row.point_uuid]}
        if row.point_uuid in self.raws:
            out['raw'] = self.raws[row.point_uuid]
        return out

    def run_fields(self, row):
        return dict(self.fields)

    def close(self):
        pass


def _row(uuid, material='h-M', load=10.0, status='complete', layers=2, angle=0.0):
    return {'point_uuid': uuid, 'run_uuid': f'r-{uuid}', 'calc_uuid': 'c', 'node_uuid': uuid,
            'verdict_status': status, 'family': 'afm', 'materials': [material],
            'flake_material': None, 'size_x': 100.0, 'size_y': 100.0, 'layers': layers,
            'load': load, 'load_kind': 'force', 'angle': angle, 'speed': 2.0}


def _source(rows, windows):
    return RunSource(pd.DataFrame(rows), 'afm', fetch=_Fetch(windows))


def _compute(name, source, **params):
    cls = get_analysis(name)
    return cls().compute(source, cls.Params(**params))


def test_n_skip_rule():
    # |mean lfx over the 2nd half| / K is the spring stretch, in drive samples.
    lfx = np.r_[np.zeros(50), np.full(50, -2.0)]

    assert settle_skip(lfx, 0.8, 0.02) == 50          # ceil(125) capped at 0.5 * N = 50
    assert settle_skip(lfx, 8.0, 0.02) == math.ceil((2.0 / 8.0) / 0.02)  # 13
    assert settle_skip(lfx, float('nan'), 0.02) == 0
    assert settle_skip(lfx, 0.8, 0.0) == 0


def test_whole_periods_counted_from_the_end():
    # 500 samples, 13 skipped, period 160.5 samples -> 3 periods, the last 482 samples.
    assert whole_periods(500, 13, 160.5) == (3, 500 - round(3 * 160.5))
    # fewer than one period after the transient: everything after n_skip.
    assert whole_periods(500, 13, 600.0) == (0, 13)
    assert whole_periods(500, 13, float('nan')) == (0, 13)


def test_period_summary_cuts_whole_periods_and_is_summary_shaped():
    cof = np.r_[np.full(100, 9.0), np.full(400, 0.1)]
    source = _source([_row('p1')], {'p1': _window(cof=cof)})

    table = _compute('period_summary', source, period_x={'h_M': 3.0})
    row = table.iloc[0]

    # K = 8 N/m -> 0.4993 eV/A^2 -> 0.8 nN/A; late lfx = 1 nN -> n_skip = ceil(1.25/0.02)
    assert row.n_skip == 63
    assert row.period_samples == pytest.approx(150.0)
    assert row.n_periods == 2 and row.start_row == 200
    assert row.cof_periods == pytest.approx(0.1)
    assert row.cof == row.cof_periods and row.lf == row.lf_periods and row.nf == row.nf_periods
    assert row.cof_full_run == pytest.approx((100 * 9.0 + 400 * 0.1) / 500)
    assert (row.id, row.material, row['size'], row.force, row.layer) == (
        'h_M', 'h_M', '100x100y', 10.0, 2)
    assert math.isnan(row.pressure)


def test_period_summary_drops_non_pass_and_off_axis_runs():
    rows = [_row('p1'), _row('p2', status='failed_integrity'), _row('p3', angle=90.0)]
    windows = {p: _window(cof=np.full(500, 0.1)) for p in ('p1', 'p2', 'p3')}
    source = _source(rows, windows)

    table = _compute('period_summary', source, period_x={'h_M': 3.0})

    assert list(table.point_uuid) == ['p1']
    assert ('p2', 'failed_integrity') in source.dropped
    assert ('p3', 'period_x_only_at_0deg') in source.dropped


def _material(name, cofs, statuses=None, missing=()):
    """Rows and windows of ``name`` at 10/20/30/50 nN with constant COF."""
    rows, windows = [], {}
    for k, (load, cof) in enumerate(zip((10.0, 20.0, 30.0, 50.0), cofs)):
        if load in missing:
            continue
        uuid = f'{name}-{load:g}'
        status = (statuses or {}).get(load, 'complete')
        rows.append(_row(uuid, material=name, load=load, status=status))
        windows[uuid] = _window(cof=np.full(500, cof))
        del k
    return rows, windows


def _ranking(*materials, **params):
    rows, windows = [], {}
    for r, w in materials:
        rows += r
        windows.update(w)
    period = {m: 3.0 for m in ('h_A', 'h_B', 'h_C', 'h_D')}
    return _compute('cof_ranking', _source(rows, windows), period_x=period, **params)


def test_missing_load_is_not_eligible():
    table = _ranking(_material('h-A', (0.1, 0.1, 0.1, 0.1)),
                     _material('h-B', (0.1, 0.1, 0.1, 0.1), missing=(50.0,)),
                     _material('h-C', (0.1, 0.1, 0.1, 0.1),
                               statuses={30.0: 'failed_integrity'}),
                     _material('h-D', (0.1, 0.1, 0.1, 0.1),
                               statuses={20.0: 'not_reached'})).set_index('material')

    assert bool(table.loc['h_A', 'eligible'])
    assert not table.loc['h_B', 'eligible']
    assert table.loc['h_B', 'excluded_because'] == 'not run at 50 nN'
    assert table.loc['h_B', 'n_loads'] == 3
    assert not table.loc['h_C', 'eligible']
    assert table.loc['h_C', 'excluded_because'] == 'integrity check: failed_integrity at 30 nN'
    assert table.loc['h_C', 'n_loads'] == 4
    assert math.isnan(table.loc['h_C', 'cof_30nN'])   # a non-pass run is never read
    assert table.loc['h_D', 'excluded_because'] == 'not run at 20 nN'


def test_rank_orders_eligible_only():
    table = _ranking(_material('h-A', (0.3, 0.3, 0.3, 0.3)),
                     _material('h-B', (0.1, 0.2, 0.3, 0.4)),
                     _material('h-C', (0.01, 0.01, 0.01, 0.01), missing=(10.0,)),
                     _material('h-D', (0.2, 0.2, 0.2, 0.2)))

    assert list(table.material) == ['h_D', 'h_B', 'h_A', 'h_C']
    assert list(table['rank'][:3]) == [1, 2, 3]
    assert pd.isna(table['rank'].iloc[3])
    b = table.set_index('material').loc['h_B']
    assert b.cof == pytest.approx(0.25)
    assert b.se == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4], ddof=1) / 2)
    assert [b[f'cof_{f}nN'] for f in (10, 20, 30, 50)] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    # 0.4 at 10 nN of drive: lfx 4 nN -> n_skip 250 (the cap) -> one 150-sample period
    assert b.a_x_A == 3.0 and b.min_periods == 1 and b.n_loads == 4
    assert math.isnan(b.trib_index)


def test_trib_index_joined_from_json_param():
    trib = {'materials': ['h_A'], 'tribIndex': [0.5], 'dev': [0.01]}
    table = _ranking(_material('h-A', (0.3,) * 4), trib_index=json.dumps(trib))

    assert (table.trib_index.iloc[0], table.trib_index_dev.iloc[0]) == (0.5, 0.01)


def test_registered_levels_and_families():
    assert get_analysis('period_summary').level == 'run'
    assert get_analysis('cof_ranking').level == 'material'
    assert get_analysis('cof_ranking').families == {'afm'}


def test_n_skip_ignores_nan_samples():
    # taxlib takes the late mean with pandas' .mean(), which skips NaN.
    lfx = np.r_[np.zeros(50), np.full(49, -1.0), np.nan]

    assert settle_skip(lfx, 8.0, 0.02) == math.ceil((1.0 / 8.0) / 0.02)
    assert settle_skip(np.full(10, np.nan), 8.0, 0.02) == 0


def test_dominant_period_finds_the_lattice_wavelength():
    drive = np.arange(2000) * 0.01                   # A
    force = 0.3 * drive + np.sin(2 * np.pi * drive / 2.5)

    assert dominant_period(drive, force) == pytest.approx(2.5, rel=0.02)
    assert math.isnan(dominant_period(drive[:63], force[:63]))


def test_period_summary_carries_the_fft_period_of_the_raw_force():
    raw_time = 100 * np.arange(5000)
    drive = raw_time * (2.0 / 100 * 0.001)          # A: 0.002 A per raw sample
    raw = pd.DataFrame({'time': raw_time, 'raw_lfx': np.sin(2 * np.pi * drive / 2.5)})
    fetch = _Fetch({'p1': _window(cof=np.full(500, 0.1))})
    fetch.raws = {'p1': raw}
    source = RunSource(pd.DataFrame([_row('p1')]), 'afm', fetch=fetch)

    row = _compute('period_summary', source, period_x={'h_M': 3.0}).iloc[0]

    assert row.fft_period_A == pytest.approx(2.5, rel=0.02)
    assert row.fft_ratio == pytest.approx(row.fft_period_A / 3.0)


def test_schema_default_none_is_missing_and_drops_the_run():
    fields = {**FIELDS, 'settings.simulation.timestep': None}
    source = RunSource(pd.DataFrame([_row('p1')]), 'afm',
                       fetch=_Fetch({'p1': _window(cof=np.full(500, 0.1))}, fields))

    table = _compute('period_summary', source, period_x={'h_M': 3.0})

    assert table.empty
    assert source.dropped == [('p1', NO_DRIVE_STEP)]


# -- period_map: one test per branch ----------------------------------------

def _geometry_fetch(windows, structures):
    """frames/run_fields as `_Fetch`, plus a built structure per material."""
    fetch = _Fetch(windows)
    fetch.structure_text = lambda row: structures.get(row.materials[0])
    fetch.settings_text = lambda row: SETTINGS
    return fetch


def test_period_map_own_source_reads_every_point():
    pts, box = _square(3.0)
    rows = [_row('p1', status='failed_integrity')]  # its structure still counts
    source = RunSource(pd.DataFrame(rows), 'afm',
                       fetch=_geometry_fetch({}, {'h-M': _bilayer(pts, box)}))

    period = period_map(source, PeriodParams())

    assert period['h_M'] == pytest.approx(3.0, abs=0.002)


def _geometry_node(table):
    """A stand-in stored result node holding ``table`` as its CSV."""
    from types import SimpleNamespace
    content = table.to_csv(index=False).encode()
    return SimpleNamespace(get_content=lambda mode='rb': content)


def test_period_map_geometry_dataset(monkeypatch):
    from src.analysis import store
    calls = []

    def fake(name, dataset, params=None, **kw):
        calls.append((name, dataset))
        return _geometry_node(pd.DataFrame({'material': ['h_M'], 'layer': [2],
                                            'period_x_top': [2.5]}))

    monkeypatch.setattr(store, 'result_node', fake)
    period = period_map(_source([_row('p0')], {}), PeriodParams(geometry_dataset='geo'))

    assert calls == [('sheet_geometry', 'geo')]
    assert period == {'h_M': 2.5}


def test_period_map_period_x_is_used_as_given():
    params = PeriodParams(period_x=json.dumps({'h_M': 3.1}), lattice={'h_N': 4.0})

    assert period_map(_source([_row('p0')], {}), params) == {'h_M': 3.1, 'h_N': 4.0}


def test_period_map_lattice_fallback(monkeypatch):
    from src.analysis import store
    monkeypatch.setattr(store, 'result_node', lambda *a, **k: _geometry_node(pd.DataFrame(
        {'material': ['h_M'], 'layer': [2], 'period_x_top': [2.5]})))

    period = period_map(_source([_row('p0')], {}), PeriodParams(geometry_dataset='geo',
                                                      lattice={'h_M': 9.0, 'h_N': 4.0}))

    assert period == {'h_M': 2.5, 'h_N': 4.0}        # the geometry wins where it has a row


def test_period_map_layer_fallback(monkeypatch):
    from src.analysis import store
    monkeypatch.setattr(store, 'result_node', lambda *a, **k: _geometry_node(pd.DataFrame(
        {'material': ['h_M', 'h_M', 'h_N', 'h_S'], 'layer': [1, 2, 1, 4],
         'period_x_top': [2.4, 2.5, 3.3, 5.0]})))

    period = period_map(_source([_row('p0')], {}), PeriodParams(geometry_dataset='geo'))

    # bilayer first, then the monolayer, then a sheet's own stack
    assert period == {'h_M': 2.5, 'h_N': 3.3, 'h_S': 5.0}


def test_material_without_a_period_is_dropped_not_averaged():
    source = _source([_row('p1')], {'p1': _window(cof=np.full(500, 0.1))})

    table = _compute('period_summary', source, period_x={'h_other': 3.0})

    assert table.empty
    assert source.dropped == [('p1', NO_PERIOD)]


def test_ranking_material_without_a_period_is_not_eligible():
    rows, windows = _material('h-A', (0.1,) * 4)
    table = _compute('cof_ranking', _source(rows, windows), period_x={'h_B': 3.0})

    assert not table.eligible.iloc[0]
    assert 'no lattice period at 10 nN' in table.excluded_because.iloc[0]


# --------------------------------------------------------------------------
# publication checks: the read-only `friction2d_pub` store against frozen files
# --------------------------------------------------------------------------

_PUB = Path('/home/matteo/mnt/PhD/Sync/Publication-data/publication_export/data')
_FROZEN_RANKING = _PUB / '260909_afm_v4/analysis/cof_ranking_v2.csv'
_LEGACY_PERIODS = _PUB / '261107_sheetonsheet/periods/outputs/output_full_100x100y.json'

_PUB_SCRIPT = """
import sys
from src.aiida import load_aiida_profile
load_aiida_profile('friction2d_pub')
from src.analysis.base import get_analysis
from src.analysis.source import RunSource
name, dataset, out = sys.argv[1:4]
# No period_x: each analysis takes a_x from the dataset's own sheet_geometry,
# as `FrictionSim2D analyse run` would (computed here, never stored).
with RunSource.from_dataset(dataset) as source:
    cls = get_analysis(name)
    cls().compute(source, cls.Params()).to_csv(out, index=False, float_format=repr)
"""


def _from_pub_store(tmp_path, name, dataset):
    """``name`` over ``dataset`` of the read-only `friction2d_pub` store, in a
    subprocess (as `test_sheet_geometry._table_from_pub_store`: the aiida
    pytest plugin hides the real profile from this process). Nothing is
    stored: the analysis is computed, never run through `run_analysis`."""
    pytest.importorskip('aiida')
    out = tmp_path / f'{name}.csv'
    env = {k: v for k, v in os.environ.items() if k != 'AIIDA_PATH'}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, '-c', _PUB_SCRIPT, name, dataset, str(out)],
                          env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        pytest.skip(f'friction2d_pub / dataset {dataset} unavailable: {done.stderr[-300:]}')
    return pd.read_csv(out, float_precision='round_trip')


@pytest.mark.publication
def test_cof_ranking_matches_the_frozen_table(tmp_path):
    if not _FROZEN_RANKING.exists():
        pytest.skip('frozen cof_ranking_v2.csv absent')
    table = _from_pub_store(tmp_path, 'cof_ranking', '260909-afm').set_index('material')
    frozen = pd.read_csv(_FROZEN_RANKING, float_precision='round_trip').set_index('material')

    absent = sorted(set(frozen.index) - set(table.index))
    common = sorted(set(frozen.index) & set(table.index))
    flips = {m: (bool(table.loc[m, 'eligible']), frozen.loc[m, 'excluded_because'])
             for m in common if bool(table.loc[m, 'eligible']) != bool(frozen.loc[m, 'eligible'])}
    both = [m for m in common if table.loc[m, 'eligible'] and frozen.loc[m, 'eligible']]
    cols = ['cof', 'se', 'cof_10nN', 'cof_20nN', 'cof_30nN', 'cof_50nN', 'cof_full_run']
    # The frozen CSV is written with %.6g (rank_cof_v2.py:114): 6 significant figures.
    off = {(m, c): (float(table.loc[m, c]), float(frozen.loc[m, c])) for m in both for c in cols
           if not math.isclose(table.loc[m, c], frozen.loc[m, c], rel_tol=5e-6, abs_tol=1e-12)}
    periods = {m: (int(table.loc[m, 'min_periods']), int(frozen.loc[m, 'min_periods']))
               for m in both if int(table.loc[m, 'min_periods']) != int(frozen.loc[m, 'min_periods'])}
    # A frozen-eligible material the store lacks shifts every later rank by
    # one: the frozen order is re-ranked over the materials both hold.
    rerank = frozen.loc[both, 'rank'].rank().astype(int)
    rank_off = {m: (int(table.loc[m, 'rank']), int(rerank[m])) for m in both
                if int(table.loc[m, 'rank']) != int(rerank[m])}
    lost = [m for m in absent if frozen.loc[m, 'eligible']]
    print(f'frozen-eligible materials absent from the store: {lost}')
    print(f'\neligible: ours {int(table.eligible.sum())} of {len(table)}, frozen '
          f'{int(frozen.eligible.sum())} of {len(frozen)}; absent from the store: {absent}')
    print(f'eligibility flips: {flips}')
    print(f'{len(both)} eligible in both; cof off: {len(off)} {off}')
    print(f'min_periods off: {periods}; rank off: {rank_off}')

    assert not flips
    assert not off
    assert not periods
    assert not rank_off


def _legacy_runs(path):
    """``(material, pressure, angle) -> data array`` of the legacy nested file."""
    results = json.loads(path.read_text())['results']
    out = {}
    for material, by_t in results.items():
        for node in _leaves(by_t):
            keys, leaf = node
            pressure = float(next(k for k in keys if k.startswith('p'))[1:])
            angle = float(next(k for k in keys if k.startswith('a'))[1:])
            out[(material, pressure, angle)] = (leaf['columns'], np.asarray(leaf['data'], float))
    return out


def _leaves(node, keys=()):
    if 'columns' in node:
        yield keys, node
        return
    for key, child in node.items():
        yield from _leaves(child, keys + (key,))


@pytest.mark.publication
def test_sheet_period_summary_matches_the_legacy_period_file(tmp_path):
    if not _LEGACY_PERIODS.exists():
        pytest.skip('legacy 261107_sheetonsheet period file absent')
    table = _from_pub_store(tmp_path, 'period_summary', '261107-sheet')
    legacy = _legacy_runs(_LEGACY_PERIODS)

    bad, missing = {}, []
    for row in table.itertuples():
        key = (row.material, float(row.pressure), float(row.angle))
        if key not in legacy:
            missing.append(key)
            continue
        columns, data = legacy[key]
        col = {c: data[:, i] for i, c in enumerate(columns)}
        lf = np.hypot(col['lfx'], col['lfy'])
        nf = np.abs(col['nf'])
        with np.errstate(divide='ignore', invalid='ignore'):
            cof = np.where(nf > 0, lf / nf, np.nan)
        expect = {'n': len(data), 'cof_periods': np.nanmean(cof), 'lf_periods': lf.mean(),
                  'nf_periods': nf.mean(),
                  'lf_sx_sy_periods': np.hypot(col['v_sx'], col['v_sy']).mean()}
        got = {'n': row.n_rows - row.start_row, 'cof_periods': row.cof_periods,
               'lf_periods': row.lf_periods, 'nf_periods': row.nf_periods,
               'lf_sx_sy_periods': row.lf_sx_sy_periods}
        diff = {k: (got[k], expect[k]) for k in expect
                if not abs(got[k] - expect[k]) <= 1e-9 * max(1.0, abs(expect[k]))}
        if diff:
            bad[key] = diff
    print(f'\n{len(table) - len(missing) - len(bad)} / {len(table)} runs within 1e-9; '
          f'not in the legacy file: {len(missing)} {missing[:10]}; off: {len(bad)} '
          f'{dict(list(bad.items())[:10])}')

    assert len(table) == 1275     # every pass 0 degree run: none dropped for its period
    assert not missing
    # The one known difference is the geometry's source, not the cut: the
    # converter took a_x from the AFM bilayers' s42 table (b_CSe 3.062 A); the
    # sheet's own built stack gives 3.064 A (one 0.002 A refinement step), which
    # moves this run's whole-period start by one row.
    known = {('b_CSe', 20.0, 0.0)}
    assert set(bad) <= known
    for key in set(bad):
        row = table[(table.material == key[0]) & (table.pressure == key[1])].iloc[0]
        assert row.period_x_A == pytest.approx(3.064, abs=1e-9)
        assert abs(bad[key]['n'][0] - bad[key]['n'][1]) == 1
