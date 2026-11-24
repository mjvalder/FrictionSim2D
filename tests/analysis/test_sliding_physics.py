"""`sliding_physics` (plan Task 10): taxlib's per-run scales, physics residual,
features and phenomena (`taxlib/scales.py`, `physics.py:54-62`,
`features.py:58-129`, `phenomena.py:18-60`), each formula on synthetic arrays
with a known answer, then the analysis over a fake fetcher."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import frictionsim2d.analysis.catalogue  # noqa: F401  (registers the analyses)
from frictionsim2d.analysis import sliding_physics as sp
from frictionsim2d.analysis.base import get_analysis
from frictionsim2d.analysis.source import RunSource

KB, EV = 8.617333262e-5, 1.602176634          # taxlib/constants.py:2-3


# --------------------------------------------------------------------------
# the formulas
# --------------------------------------------------------------------------

def test_spring_fit_recovers_the_spring():
    dx, n = 0.02, 1000
    i = np.arange(n)
    tipx = 50.0 + 0.5 * dx * i + 0.3 * np.sin(2 * np.pi * i / 150)
    lfx = 0.8 * (dx * i - (tipx - tipx[0])) + 0.1
    k, r2 = sp.spring_fit(lfx, tipx, dx)
    assert k == pytest.approx(0.8) and r2 == pytest.approx(1.0)
    # a stretch range below 0.5 A is unidentifiable (scales.py:7, :14)
    assert all(math.isnan(v) for v in sp.spring_fit(lfx[:20], tipx[:20], dx))


def test_pick_k_prefers_provenance_then_a_positive_fit():
    assert sp.pick_k(1.6, 0.9) == (1.6, 'provenance')
    assert sp.pick_k(float('nan'), 0.9) == (0.9, 'fit')
    k, src = sp.pick_k(float('nan'), -0.9)
    assert math.isnan(k) and src == 'missing'


def test_sigma_th_and_friction_work():
    assert sp.sigma_th(2.0, 300.0) == pytest.approx(math.sqrt(2.0 * KB * 300.0 * EV))
    assert math.isnan(sp.sigma_th(float('nan'), 300.0))
    assert sp.friction_work(np.full(10, 2.0), 10.0) == pytest.approx(20.0 / EV)


def test_energy_residual_of_a_balanced_run_keeps_the_first_drive_step_and_is_scaled():
    n, dx, k, f_n = 200, 0.02, 1.0, 10.0
    lfx = np.linspace(0.5, 1.5, n)
    tipz = 5.0 - 0.001 * np.arange(n)
    wth = 0.01 * np.arange(n)
    w_drive = np.cumsum(lfx) * dx / EV
    w_load = f_n * (tipz[0] - tipz) / EV
    e_spring = (lfx ** 2 - lfx[0] ** 2) / (2 * k) / EV
    etotal = 3.0 + w_drive + w_load - (wth - wth[0]) - e_spring
    w = pd.DataFrame({'etotal': etotal, 'w_thermostat_tip': wth / 2,
                      'w_thermostat_sub': wth / 2, 'lfx': lfx, 'tipz': tipz})
    # taxlib's W_drive is cumsum(lfx) dx unshifted (physics.py:35): a balanced
    # run keeps the first sample's drive work, -lfx[0] dx
    first = -lfx[0] * dx / EV
    assert sp.energy_residual(w, f_n, dx, 2.0, 0.1, k) == pytest.approx(first / 2.0, abs=1e-12)
    w.loc[n - 1, 'etotal'] += 0.5
    assert sp.energy_residual(w, f_n, dx, 2.0, 0.1, k) == pytest.approx((first + 0.5) / 2.0)
    # the thermal floor: max(|W_fric|, floor) = 0.1
    assert sp.energy_residual(w, f_n, dx, 0.01, 0.1, k) == pytest.approx((first + 0.5) / 0.1)
    assert math.isnan(sp.energy_residual(w.drop(columns='etotal'), f_n, dx, 2.0, 0.1, k))


def test_body_start_keeps_the_whole_run_when_too_short():
    assert sp.body_start(1000, 30) == 30
    assert sp.body_start(120, 30) == 0              # 90 < 100 samples left (features.py:54)


def test_block_se_of_known_block_means():
    means = np.arange(10, dtype=float)
    lf = np.repeat(means, 10)
    assert sp.block_se(lf, 2.0) == pytest.approx(means.std(ddof=1) / math.sqrt(10) / 2.0)
    assert math.isnan(sp.block_se(lf[:49], 2.0))     # fewer than 5 samples per block


def test_superlubric_margin():
    lf = np.repeat(np.arange(10, dtype=float), 10) + 1.0
    se = sp.block_se(lf, 1.0)
    assert sp.superlubric_margin(lf) == pytest.approx(abs(lf.mean()) / se)


def test_stick_frac_counts_the_stuck_samples():
    dx = 0.02
    steps = np.tile([0.0, dx], 50)
    tipx = np.concatenate([[0.0], np.cumsum(steps)])
    assert sp.stick_frac(tipx, dx) == pytest.approx(0.5)


def test_sheet_carried_and_interlayer_slip():
    tipx = np.linspace(0.0, 10.0, 50)
    comx = np.linspace(0.0, 2.0, 50)
    assert sp.sheet_carried(comx, tipx, 3.0) == pytest.approx(0.2)
    assert math.isnan(sp.sheet_carried(comx, tipx * 0.2, 3.0))     # advance below a
    l1, l2 = np.zeros(50), np.linspace(0.0, 4.0, 50)
    assert sp.il_slip_frac(l1, l2, tipx, 3.0) == pytest.approx(0.4)


def test_lattice_periodicity_of_a_periodic_and_an_aperiodic_force():
    a = 3.0
    x = np.linspace(0.0, 10 * a, 2000)
    assert sp.lattice_periodicity(x, np.sin(2 * np.pi * x / a), a) > 0.95
    rng = np.random.default_rng(1)
    assert sp.lattice_periodicity(x, rng.standard_normal(2000), a) < 0.2
    assert math.isnan(sp.lattice_periodicity(x[:100], np.ones(100), a))   # under 2 a of travel


def test_ageing_late_minus_early_over_the_mean():
    lf = np.concatenate([np.ones(100), np.full(100, 2.0), np.full(100, 3.0)])
    assert sp.ageing(lf, 0.1) == pytest.approx(1.0)
    assert math.isnan(sp.ageing(lf * 1e-3, 0.1))     # mean below sigma_th / 10


def test_e_int_rel_and_shear_strength():
    assert sp.e_int_rel(np.full(5, -2.0), 10.0, 3.0) == pytest.approx(-2.0 * EV / 30.0)
    # 1 nN over 100 A^2 = 1e-9 N / 1e-18 m^2 = 1 GPa
    assert sp.shear_strength_gpa(1.0, 100.0) == pytest.approx(1.0)
    assert math.isnan(sp.shear_strength_gpa(1.0, 0.0))


# --------------------------------------------------------------------------
# the analysis over a fake fetcher
# --------------------------------------------------------------------------

N, DX = 1000, 0.02           # samples; A per sample (2 m/s, 1 fs, 1000 steps)


def _afm_frames():
    i = np.arange(N)
    tipx = 50.0 + DX * i - 0.6 * np.abs(np.sin(np.pi * DX * i / 3.0))
    lfx = 0.8 * (DX * i - (tipx - tipx[0])) + 0.5
    return {'window': pd.DataFrame({
        'time': 1000 * i, 'lfx': lfx, 'lfy': 0.01 * np.ones(N), 'nf': np.full(N, 10.0),
        'tipx': tipx, 'tipy': np.zeros(N), 'tipz': np.full(N, 5.0),
        'comx': 0.1 * DX * i, 'comx_L1': np.zeros(N), 'comx_L2': 0.1 * DX * i,
        'etotal': np.zeros(N), 'w_thermostat_tip': np.zeros(N),
        'w_thermostat_sub': np.zeros(N)}),
        'peratom': pd.DataFrame({'time': 10000 * np.arange(N // 10),
                                 'E_int_tip_sheet': np.full(N // 10, -3.0)})}


def _sheet_frames():
    i = np.arange(N)
    return {'window': pd.DataFrame({
        'time': 1000 * i, 'lfx': 0.4 + 0.1 * np.sin(i / 10.0), 'lfy': np.zeros(N),
        'nf': np.full(N, -50.0), 'drivex': DX * i, 'comx_L1': np.zeros(N),
        'comx_L2': DX * i, 'etotal': np.zeros(N), 'w_thermostat_center': np.zeros(N)})}


class _Fetch:
    def __init__(self, frames, fields=None):
        self.data = frames
        self.fields = fields if fields is not None else {
            'settings.simulation.timestep': 0.001, 'derived.drive_k': 0.5}

    def frames(self, row):
        return self.data[row.point_uuid]

    def run_fields(self, row):
        return dict(self.fields)

    def close(self):
        pass


def _row(uuid, family='afm', load=10.0, status='complete', angle=0.0, **extra):
    return {'point_uuid': uuid, 'run_uuid': f'r-{uuid}', 'calc_uuid': 'c', 'node_uuid': uuid,
            'verdict_status': status, 'family': family, 'materials': ['h-MoS2'],
            'flake_material': None, 'size_x': 100.0, 'size_y': 100.0, 'layers': 2,
            'load': load, 'load_kind': 'force', 'angle': angle, 'speed': 2.0,
            'temperature': 300.0, 'load_achieved': None, **extra}


def _sliding(rows, frames, family='afm', fields=None, dataset=None, depends=None,
             **params):
    cls = get_analysis('sliding_physics')
    source = RunSource(pd.DataFrame(rows), family, fetch=_Fetch(frames, fields))
    source.dataset = dataset
    if depends is not None:
        source.depends = depends
    params.setdefault('period_x', {'h_MoS2': 3.0})
    return cls().compute(source, cls.Params(**params)), source


def test_sliding_physics_afm_row():
    rows = [_row('p1', load_achieved=9.5), _row('p2', status='failed'), _row('p3', angle=30.0)]
    table, source = _sliding(rows, {'p1': _afm_frames()})
    assert source.dropped == [('p2', 'failed')]
    assert len(table) == 1                                   # 30 deg is not read
    row = table.iloc[0]
    k = 0.5 * EV
    assert row['K_nN_A'] == pytest.approx(k) and row['K_source'] == 'provenance'
    assert row['K_fit'] == pytest.approx(0.8) and row['K_fit_r2'] == pytest.approx(1.0)
    assert row['spring_K_ratio'] == pytest.approx(0.8 / k)
    assert row['F_N_nN'] == 9.5 and row['F_N_source'] == 'provenance'
    assert row['dx_A'] == pytest.approx(DX) and row['L_A'] == pytest.approx(DX * N)
    assert row['a_A'] == 3.0
    assert row['sigma_th_nN'] == pytest.approx(math.sqrt(k * KB * 300.0 * EV))
    w = _afm_frames()['window']
    assert row['W_fric_eV'] == pytest.approx(w['lfx'].mean() * DX * N / EV)
    body = w['lfx'].to_numpy()[int(row['n_skip']):]
    assert row['lf_mean_nN'] == pytest.approx(body.mean())
    assert row['cof_mean'] == pytest.approx(body.mean() / 9.5)
    assert row['sheet_carried'] == pytest.approx(0.1, rel=0.05)
    assert row['il_slip_frac'] == pytest.approx(0.1, rel=0.05)
    assert row['e_int_rel'] == pytest.approx(-3.0 * EV / (9.5 * 3.0))
    for col in ('energy_resid_rel', 'cof_se', 'superlubric_margin', 'stick_frac',
                'lattice_periodicity', 'ageing'):
        assert col in table.columns
    assert row['nan_reason'] == '' or pd.isna(row['nan_reason'])
    assert 'shear_strength_GPa' not in table.columns      # no contact result
    assert any('contact' in note for note in source.notes)


def test_sliding_physics_takes_the_run_fit_without_a_recorded_spring():
    table, _ = _sliding([_row('p1')], {'p1': _afm_frames()},
                        fields={'settings.simulation.timestep': 0.001})
    row = table.iloc[0]
    assert row['K_source'] == 'fit' and row['K_nN_A'] == pytest.approx(0.8)
    assert row['F_N_nN'] == 10.0 and row['F_N_source'] == 'requested'


def test_sliding_physics_lattice_a_from_taxlib_a():
    table, _ = _sliding([_row('p1')], {'p1': _afm_frames()}, taxlib_a={'h_MoS2': 3.16})
    assert table.iloc[0]['a_A'] == 3.16


def _contact_node(rows):
    """A stand-in for a stored contact result node: uuid, extras, CSV content."""
    content = pd.DataFrame(rows).to_csv(index=False).encode('utf-8')
    extras = {'version': '1', 'params_hash': 'abc'}
    return SimpleNamespace(uuid='contact-uuid', base=SimpleNamespace(
        extras=SimpleNamespace(get=lambda key, default=None: extras.get(key, default))),
        get_content=lambda mode='rb': content)


def _contact_row(run, load, state='load', area=50.0, angle=0.0):
    return {'run_uuid': run, 'load': load, 'angle': angle, 'state': state, 'contact_area': area}


def test_sliding_physics_joins_shear_strength_from_a_stored_contact():
    node = _contact_node([_contact_row('r-p1', 10.0),
                          _contact_row('r-p1', 10.0, state='final', area=99.0)])
    table, source = _sliding([_row('p1'), _row('p2', load=20.0)],
                             {'p1': _afm_frames(), 'p2': _afm_frames()},
                             depends={'contact': node})
    by_load = table.set_index('load')
    assert by_load.loc[10.0, 'contact_area_A2'] == 50.0
    assert by_load.loc[10.0, 'shear_strength_GPa'] == pytest.approx(
        by_load.loc[10.0, 'lf_mean_nN'] / 50.0 * 100.0)
    assert 'shear_strength_GPa' not in by_load.loc[10.0, 'nan_reason']
    # no contact row for (r-p2, 20 nN, 0 deg): NaN with its reason
    assert np.isnan(by_load.loc[20.0, 'shear_strength_GPa'])
    assert 'shear_strength_GPa: no contact row' in by_load.loc[20.0, 'nan_reason']
    assert source.notes == []


def test_sliding_physics_looks_the_contact_up_when_not_handed_one(monkeypatch):
    seen = []
    node = _contact_node([_contact_row('r-p1', 10.0)])

    def fake(dataset):
        seen.append(dataset)
        return node
    monkeypatch.setattr(sp, 'contact_result', fake)
    table, _ = _sliding([_row('p1')], {'p1': _afm_frames()}, dataset='paper')
    assert seen == ['paper'] and table.iloc[0]['contact_area_A2'] == 50.0


def test_join_contact_refuses_duplicate_contact_rows():
    table = pd.DataFrame([{'run_uuid': 'r', 'load': 10.0, 'angle': 0.0, 'lf_mean_nN': 1.0,
                           'nan_reason': ''}])
    contact = pd.DataFrame([_contact_row('r', 10.0, area=50.0), _contact_row('r', 10.0, area=60.0)])
    with pytest.raises(ValueError, match='duplicate'):
        sp.join_contact(table, contact)


def test_sliding_physics_sheet_tip_values_are_nan_with_a_reason():
    rows = [_row('s1', family='sheetonsheet')]
    table, source = _sliding(rows, {'s1': _sheet_frames()}, family='sheetonsheet')
    row = table.iloc[0]
    for col in ('K_fit', 'K_fit_r2', 'spring_K_ratio', 'energy_resid_rel', 'stick_frac',
                'sheet_carried', 'il_slip_frac', 'lattice_periodicity', 'e_int_rel'):
        assert np.isnan(row[col]), col
        assert col in row['nan_reason']
    assert 'no tip' in row['nan_reason']
    for col in ('sigma_th_nN', 'W_fric_eV', 'cof_se', 'superlubric_margin', 'lf_mean_nN'):
        assert np.isfinite(row[col]), col
    assert 'shear_strength_GPa' not in table.columns
    assert len(source.notes) == 1 and 'AFM-only' in source.notes[0]


def test_sliding_physics_sheet_pressure_load_has_no_f_n():
    rows = [_row('s1', family='sheetonsheet', load_kind='pressure', load=1.0)]
    table, _ = _sliding(rows, {'s1': _sheet_frames()}, family='sheetonsheet')
    row = table.iloc[0]
    assert np.isnan(row['F_N_nN']) and np.isnan(row['cof_se']) and np.isnan(row['cof_mean'])
    assert 'cof_se' in row['nan_reason']
    assert np.isfinite(row['superlubric_margin'])


def test_sliding_physics_refuses_pes_families():
    cls = get_analysis('sliding_physics')
    source = RunSource(pd.DataFrame([_row('p1', family='pes_tip')]), 'pes_tip',
                       fetch=_Fetch({}))
    with pytest.raises(ValueError, match='pes_tip'):
        cls().compute(source, cls.Params())


def test_sliding_physics_drops_a_run_with_no_window_tier():
    rows = [_row('p1'), _row('p2')]
    frames = {'p1': {'peratom': _afm_frames()['peratom']}, 'p2': _afm_frames()}
    table, source = _sliding(rows, frames)
    assert source.dropped == [('p1', sp.NO_WINDOW)]
    assert list(table['point_uuid']) == ['p2']
