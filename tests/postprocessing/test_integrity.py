"""Integrity measurements ported from ML_taxonomy s48/s51 (frictionsim2d/postprocessing/integrity.py)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scipy')

from frictionsim2d.core.config import IntegritySettings  # noqa: E402
from frictionsim2d.postprocessing.integrity import (  # noqa: E402
    AFM_LIKE, SHEET_LIKE, layer_measures, read_snapshot, series_measures,
    snapshot_measures)

S = IntegritySettings()


def _hex_layer(z, n=20, a=1.0):
    """Periodic honeycomb sheet, bond `a`, `n` x `n` rectangular 4-atom cells."""
    w, h = np.sqrt(3.0) * a, 3.0 * a
    cell = np.array([[0.0, 0.0], [w / 2, a / 2], [w / 2, 1.5 * a], [0.0, 2.0 * a]])
    pts = [cell + [i * w, j * h] for i in range(n) for j in range(n)]
    xy = np.vstack(pts)
    xyz = np.c_[xy, np.full(len(xy), float(z))]
    box = {'x': (0.0, n * w), 'y': (0.0, n * h), 'z': (-50.0, 50.0)}
    return xyz, box


def _jittered(seed=0, sigma=0.05):
    xyz, box = _hex_layer(0.0)
    rng = np.random.default_rng(seed)
    return xyz + rng.normal(0.0, sigma, xyz.shape), box


def _centre(box):
    return np.array([box['x'][1] / 2, box['y'][1] / 2])


def _punch(xyz, box, r):
    keep = np.linalg.norm(xyz[:, :2] - _centre(box), axis=1) > r
    return xyz[keep]


def test_module_constants():
    assert AFM_LIKE == ('afm', 'indent', 'pes_tip')
    assert SHEET_LIKE == ('sheetonsheet', 'hetero', 'flake', 'pes_sheet')


def test_intact_thermal_sheet_passes():
    xyz, box = _jittered()
    ref, _ = _hex_layer(0.0)              # the loaded (reference) state
    m = layer_measures(xyz, ref, box, box, S)
    assert m['bond'] == pytest.approx(1.0, rel=0.02)
    # A perfect honeycomb's largest empty disk is one bond (hexagon centre), so
    # a thermal sheet sits just above 1.0 -- below the 1.5 threshold.
    assert m['hole'] < S.hole_bonds
    assert m['isolated'] == 0 and m['ejected'] == 0 and m['n'] == len(xyz)


def test_perfect_lattice_hole_is_one_bond():
    xyz, box = _hex_layer(0.0)
    assert layer_measures(xyz, xyz, box, box, S)['hole'] == pytest.approx(1.0, abs=1e-9)


def test_punched_hole_measured():
    xyz, box = _jittered()
    m = layer_measures(_punch(xyz, box, 2.5), xyz, box, box, S)
    assert m['hole'] >= 2.0


def test_two_isolated_atoms_counted():
    xyz, box = _jittered()
    holed = _punch(xyz, box, 6.0)
    c = _centre(box)
    strays = np.array([[c[0] - 2.0, c[1], 0.0], [c[0] + 2.0, c[1], 0.0]])
    m = layer_measures(np.vstack([holed, strays]), xyz, box, box, S)
    assert m['isolated'] == 2


def test_one_stray_atom_is_one():
    xyz, box = _jittered()
    moved = xyz.copy()
    moved[100, 2] += 3.0
    m = layer_measures(moved, xyz, box, box, S)
    assert m['isolated'] == 1 and m['ejected'] == 0


def test_ejected_pair():
    xyz, box = _jittered()
    moved = xyz.copy()
    moved[[10, 500], 2] += 10.0
    assert layer_measures(moved, xyz, box, box, S)['ejected'] == 2


def test_wrapping_is_periodic_in_xy():
    xyz, box = _jittered()
    shifted = xyz.copy()
    shifted[:, 0] += 3.0 * (box['x'][1] - box['x'][0]) + 0.37
    shifted[:, 1] -= 2.0 * (box['y'][1] - box['y'][0])
    a = layer_measures(xyz, xyz, box, box, S)
    b = layer_measures(shifted, xyz, box, box, S)
    assert a['isolated'] == b['isolated'] == 0
    assert b['hole'] < S.hole_bonds


def _write(path, style, rows, comment=True):
    head = ['LAMMPS data file', '', f'{len(rows)} atoms', '2 atom types', '',
            '0.0 10.0 xlo xhi', '0.0 12.0 ylo yhi', '-5.0 5.0 zlo zhi', '',
            'Masses', '', '1 12.0', '2 32.0', '',
            f'Atoms # {style}' if comment else 'Atoms', '']
    body = []
    for i, (t, x, y, z) in enumerate(rows, 1):
        if style == 'atomic':
            body.append(f'{i} {t} {x} {y} {z} 0 0 0')
        else:
            body.append(f'{i} 1 {t} {x} {y} {z} 0 0 0')
    tail = ['', 'Velocities', '', *[f'{i} 0 0 0' for i in range(1, len(rows) + 1)]]
    path.write_text('\n'.join(head + body + tail) + '\n')
    return path


ROWS = [(1, 0.5, 0.5, 0.0), (2, 1.5, 0.5, 1.0), (1, 2.5, 3.0, 0.0), (2, 9.5, 11.0, -1.0)]


def test_read_snapshot_atomic_and_molecular(tmp_path):
    t1, x1, b1 = read_snapshot(_write(tmp_path / 'a.data', 'atomic', ROWS))
    t2, x2, b2 = read_snapshot(_write(tmp_path / 'm.data', 'molecular', ROWS))
    assert (t1 == t2).all() and np.allclose(x1, x2) and b1 == b2
    assert t1.tolist() == [1, 2, 1, 2] and x1.shape == (4, 3)
    assert b1 == {'x': (0.0, 10.0), 'y': (0.0, 12.0), 'z': (-5.0, 5.0)}


def test_read_snapshot_atomic_without_comment(tmp_path):
    t, x, _ = read_snapshot(_write(tmp_path / 'a.data', 'atomic', ROWS, comment=False))
    assert t.tolist() == [1, 2, 1, 2] and x[1].tolist() == [1.5, 0.5, 1.0]


def test_read_snapshot_ambiguous_without_comment_raises(tmp_path):
    with pytest.raises(ValueError):
        read_snapshot(_write(tmp_path / 'm.data', 'molecular', ROWS, comment=False))


def test_read_snapshot_unknown_style_raises(tmp_path):
    with pytest.raises(ValueError):
        read_snapshot(_write(tmp_path / 'e.data', 'ellipsoid', ROWS))


def test_snapshot_measures_combines_layers(tmp_path):
    l1, box = _hex_layer(0.0)
    l2, _ = _hex_layer(6.0)
    l2 = _punch(l2, box, 2.5)
    rows_ref = [(1, *p) for p in l1] + [(2, *p) for p in _hex_layer(6.0)[0]]
    moved = l1.copy()
    moved[[3, 7], 2] += 10.0
    rows_fin = [(1, *p) for p in moved] + [(2, *p) for p in l2]

    def write(path, rows):
        body = '\n'.join(f'{i} {t} {x} {y} {z}' for i, (t, x, y, z) in enumerate(rows, 1))
        path.write_text(f"x\n\n{len(rows)} atoms\n\n{box['x'][0]} {box['x'][1]} xlo xhi\n"
                        f"{box['y'][0]} {box['y'][1]} ylo yhi\n-50 50 zlo zhi\n\n"
                        f"Atoms # atomic\n\n{body}\n")
        return path

    m = snapshot_measures(write(tmp_path / 'f.data', rows_fin), write(tmp_path / 'r.data', rows_ref),
                          {'layer_1': {1}, 'layer_2': {2}}, S)
    assert m['ejected'] == 2 and m['isolated'] == 2
    assert m['hole'] >= 2.0
    assert m['n'] == len(l1) + len(l2)
    assert m['n_by_layer'] == {'layer_1': len(l1), 'layer_2': len(l2)}


def _afm_window(n=60, **cols):
    rng = np.random.default_rng(3)
    base = {c: rng.normal(0.0, 1.0, n) for c in
            ('nf', 'lfx', 'lfy', 'comx', 'comy', 'comz', 'tipx', 'tipy', 'tipz')}
    base.update(cols)
    return pd.DataFrame(base)


def _sheet_window(n=100, n_layers=4, gap=3.0, **cols):
    rng = np.random.default_rng(4)
    base = {c: rng.normal(0.0, 1.0, n) for c in ('lfx', 'lfy', 'nf')}
    for i in range(1, n_layers + 1):
        base[f'comz_L{i}'] = (i - 1) * gap + (0.0 if i == 1 else rng.normal(0.0, 1e-3, n))
    base['temp_center_3d'] = np.full(n, 300.0)
    base.update(cols)
    return pd.DataFrame(base)


def test_heating_sigma_afm_uses_run_temperature():
    w = _afm_window(temp_sheet_3d=[400.0] * 60)
    m = series_measures(w, 'afm', 2, T_target=400.0, n_heat=1000, settings=S)
    assert m['T_sigma'] == pytest.approx(0.0)


def test_heating_sigma_value():
    w = _afm_window(temp_sheet_3d=[300.0] * 59 + [330.0])
    m = series_measures(w, 'afm', 1, T_target=300.0, n_heat=1500, settings=S)
    assert m['T_sigma'] == pytest.approx(30.0 / (300.0 * np.sqrt(2.0 / 4500.0)))


def test_heating_nan_without_channel_or_n_heat():
    w = _afm_window()
    assert np.isnan(series_measures(w, 'afm', 1, 300.0, 1000, S)['T_sigma'])
    w = _afm_window(temp_sheet_3d=[300.0] * 60)
    assert np.isnan(series_measures(w, 'afm', 1, 300.0, None, S)['T_sigma'])


def test_sheet_heating_skips_first_five():
    t = np.full(100, 300.0)
    t[2] = 900.0
    w = _sheet_window(temp_center_3d=t)
    assert series_measures(w, 'sheetonsheet', 4, 300.0, 1000, S)['T_sigma'] < 1.0
    t[5] = 900.0
    assert series_measures(w.assign(temp_center_3d=t), 'sheetonsheet', 4, 300.0, 1000, S)['T_sigma'] > 8.0


def test_sheet_gap_is_lasting_shift_either_sign():
    n, g0 = 100, 3.0
    w = _sheet_window(n=n, gap=g0)
    l3 = np.full(n, 2 * g0)
    l3[-50:] = 2 * g0 - 0.06 * g0        # shear gap L3 - L2 ends 6 % lower
    w['comz_L2'] = g0
    w['comz_L3'] = l3
    w['comz_L4'] = l3 + g0
    assert series_measures(w, 'sheetonsheet', 4, 300.0, 1000, S)['gap_rise'] == pytest.approx(0.06, rel=1e-3)


def test_sheet_gap_ignores_transient_peak():
    n, g0 = 100, 3.0
    w = _sheet_window(n=n, gap=g0)
    w['comz_L2'] = g0
    l3 = np.full(n, 2 * g0)
    l3[40] += 0.5 * g0
    w['comz_L3'] = l3
    assert series_measures(w, 'sheetonsheet', 4, 300.0, 1000, S)['gap_rise'] == pytest.approx(0.0)


def test_afm_gap_is_peak():
    n, g0 = 60, 3.0
    l2 = np.full(n, g0)
    l2[40] = 1.06 * g0                    # transient 6 % peak, then back
    w = _afm_window(n=n, comz_L1=np.zeros(n), comz_L2=l2)
    assert series_measures(w, 'afm', 2, 300.0, 1000, S)['gap_rise'] == pytest.approx(0.06, rel=1e-3)


def test_gap_nan_for_monolayer_and_two_layer_sheet():
    assert np.isnan(series_measures(_afm_window(), 'afm', 1, 300.0, 1000, S)['gap_rise'])
    w = _sheet_window(n_layers=2)
    m = series_measures(w, 'sheetonsheet', 2, 300.0, 1000, S)
    assert np.isnan(m['gap_rise']) and m['gap_min_any'] == pytest.approx(3.0, abs=0.01)


def test_collapse_negative_gap():
    w = _sheet_window()
    w.loc[30, 'comz_L3'] = w.loc[30, 'comz_L2'] - 0.5
    assert series_measures(w, 'sheetonsheet', 4, 300.0, 1000, S)['gap_min_any'] < 0


def test_gap_min_any_nan_for_afm():
    w = _afm_window(comz_L1=np.zeros(60), comz_L2=np.full(60, 3.0))
    assert np.isnan(series_measures(w, 'afm', 2, 300.0, 1000, S)['gap_min_any'])


def test_non_finite_and_frozen():
    lfx = np.random.default_rng(0).normal(size=60)
    lfx[7] = np.nan
    m = series_measures(_afm_window(lfx=lfx, tipx=np.full(60, 2.0)), 'afm', 1, 300.0, 1000, S)
    assert m['non_finite'] and m['frozen']


def test_clean_series_not_flagged():
    m = series_measures(_afm_window(), 'afm', 1, 300.0, 1000, S)
    assert not m['non_finite'] and not m['frozen']


def test_non_finite_energy():
    e = np.zeros(60)
    e[-1] = np.inf
    assert series_measures(_afm_window(etotal=e), 'afm', 1, 300.0, 1000, S)['non_finite']


def test_absent_channels_are_skipped():
    w = _afm_window().drop(columns=['tipx', 'tipy', 'tipz'])
    m = series_measures(w, 'afm', 1, 300.0, 1000, S)
    assert not m['non_finite'] and not m['frozen']


def test_sheet_frozen_ignores_fixed_layer_1():
    w = _sheet_window()
    assert not series_measures(w, 'sheetonsheet', 4, 300.0, 1000, S)['frozen']
    w['comz_L3'] = 6.0
    assert series_measures(w, 'sheetonsheet', 4, 300.0, 1000, S)['frozen']


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        series_measures(_afm_window(), 'nope', 1, 300.0, 1000, S)


# --------------------------------------------------------------------------
# s51 window guard (Task 2 review carry-over a)
# --------------------------------------------------------------------------

@pytest.mark.parametrize('rows', [0, 1])
def test_sheet_short_window_skips_gate_checks(rows):
    """s51 :89 evaluates the series only when the window has >1 row."""
    import warnings  # pylint: disable=import-outside-toplevel
    w = _sheet_window().head(rows)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        m = series_measures(w, 'sheetonsheet', 4, 300.0, 1000, S)
    assert m['frozen'] is not True and m['non_finite'] is not True
    assert all(np.isnan(m[k]) for k in ('non_finite', 'frozen', 'T_sigma',
                                         'gap_rise', 'gap_min_any'))


# --------------------------------------------------------------------------
# Task 3: verdict, quarantine, layer resolution
# --------------------------------------------------------------------------
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from frictionsim2d.core.observables import rename_map  # noqa: E402
from frictionsim2d.data.provenance import read_provenance  # noqa: E402
from frictionsim2d.postprocessing.integrity import (  # noqa: E402
    IntegrityVerdict, bond_reference, check_files, check_point,
    resolve_layers, verdict)
from frictionsim2d.postprocessing.read_data import DataReader  # noqa: E402
from frictionsim2d.run.tree import discover_runs  # noqa: E402
from tests.helpers.results_tree import make_family_run  # noqa: E402

CLEAN = {'non_finite': False, 'frozen': False, 'hole': 1.0, 'isolated': 0,
         'ejected': 0, 'T_sigma': 1.0, 'gap_rise': 0.01, 'gap_min_any': 3.0}

BAD = {'hole': 2.0, 'isolated': 5, 'ejected': 5, 'T_sigma': 10.0,
       'gap_rise': 0.1, 'gap_min_any': -0.5, 'non_finite': True, 'frozen': True}

RULE_OF = {'hole': 'hole', 'isolated': 'isolated_atoms', 'ejected': 'ejected_atoms',
           'T_sigma': 'sheet_heating', 'gap_rise': 'gap_rise',
           'gap_min_any': 'collapse', 'non_finite': 'non_finite', 'frozen': 'frozen'}


@pytest.mark.parametrize('status,failed,observed', [
    ('no_output', True, False), ('not_reached', True, False),
    ('truncated', True, True), ('halted', True, True), ('unreadable', True, True)])
def test_status_mapping(status, failed, observed):
    v = verdict(status, {}, IntegritySettings())
    assert (v.status, v.failed, v.observed) == (status, failed, observed)


def test_clean_complete_point():
    v = verdict('complete', CLEAN, IntegritySettings())
    assert (v.status, v.failed, v.observed, v.reasons) == ('complete', False, True, [])
    assert v.quarantine == [] and v.not_applicable == []
    assert v.thresholds_version == IntegritySettings().version


@pytest.mark.parametrize('key', sorted(BAD))
def test_each_rule_fires(key):
    v = verdict('complete', {**CLEAN, key: BAD[key]}, IntegritySettings())
    assert v.status == 'failed_integrity' and v.failed and v.observed
    assert len(v.reasons) == 1 and v.reasons[0].split()[0] == RULE_OF[key]


def test_reason_text_and_order():
    v = verdict('complete', {**CLEAN, **BAD, 'hole': 3.6}, IntegritySettings())
    assert v.reasons == ['non_finite', 'frozen', 'hole 3.6 bonds', 'isolated_atoms 5',
                         'ejected_atoms 5', 'sheet_heating 10 sigma_T', 'gap_rise 0.1',
                         'collapse -0.5']


def test_threshold_is_inclusive():
    """s48 fires on `>=`."""
    assert verdict('complete', {**CLEAN, 'hole': 1.5}, IntegritySettings()).failed


def test_gate_status_keeps_status_and_rules():
    v = verdict('halted', {**CLEAN, 'non_finite': True}, IntegritySettings())
    assert v.status == 'halted' and v.reasons == ['halted', 'non_finite']


def test_rule_off_never_fires():
    v = verdict('complete', {**CLEAN, 'hole': 9.0}, IntegritySettings(hole_bonds=None))
    assert v.status == 'complete' and 'hole' not in v.not_applicable
    off = IntegritySettings(collapse=False, frozen_std=None)
    assert verdict('complete', {**CLEAN, 'gap_min_any': -1.0, 'frozen': True},
                   off).status == 'complete'


def test_nan_measure_is_not_applicable_not_pass():
    v = verdict('complete', {**CLEAN, 'hole': float('nan')}, IntegritySettings())
    assert 'hole' in v.not_applicable and v.status == 'complete'
    assert 'non_finite' in verdict('complete', {}, IntegritySettings()).not_applicable


def test_margin_quarantines():
    s = IntegritySettings(margins={'hole': 0.9})
    v = verdict('complete', {**CLEAN, 'hole': 1.40}, s)
    assert v.status == 'quarantined' and not v.failed and v.observed
    assert v.quarantine and v.quarantine[0].startswith('hole')
    assert verdict('complete', {**CLEAN, 'hole': 1.20}, s).status == 'complete'
    assert verdict('complete', {**CLEAN, 'hole': 1.60}, s).status == 'failed_integrity'


def test_as_dict_is_json_safe():
    v = verdict('complete', {**CLEAN, 'hole': float('nan')}, IntegritySettings())
    d = v.as_dict()
    assert d['measures']['hole'] is None
    json.dumps(d, allow_nan=False)
    assert isinstance(v, IntegrityVerdict)


# --- trees ------------------------------------------------------------------

def _write_data(path, layers, box, style='atomic'):
    """LAMMPS data file from [(type, xyz), ...]."""
    rows, k = [], 0
    for t, xyz in layers:
        for x, y, z in xyz:
            k += 1
            rows.append(f'{k} {t} {x:.6f} {y:.6f} {z:.6f}' if style == 'atomic'
                        else f'{k} 1 {t} {x:.6f} {y:.6f} {z:.6f}')
    ntypes = max(t for t, _ in layers)
    body = ['LAMMPS data file', '', f'{k} atoms', f'{ntypes} atom types', '',
            f"{box['x'][0]} {box['x'][1]} xlo xhi", f"{box['y'][0]} {box['y'][1]} ylo yhi",
            '-10.0 60.0 zlo zhi', '', f'Atoms # {style}', '', *rows, '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(body), encoding='utf-8')
    return path


def _stack(n_layers, n=8, spacing=6.0):
    out = []
    for i in range(n_layers):
        xyz, box = _hex_layer(i * spacing, n=n)
        out.append((i + 1, xyz))
    return out, box


def _set_layer_map(run_dir, layer_map):
    prov = run_dir / 'provenance.json'
    data = json.loads(prov.read_text(encoding='utf-8'))
    data['layer_type_map'] = layer_map
    prov.write_text(json.dumps(data), encoding='utf-8')


def _sheet_results(run_dir, n_layers=4, n=120, pressure=1.0, rise=0.0, seed=0):
    """A realistic window file: noisy channels, layers 6 A apart."""
    for f in (run_dir / 'results').glob('friction_*'):
        f.unlink()
    rng = np.random.default_rng(seed)
    natives = sorted(rename_map('sheetonsheet', 'window', n_layers))
    cols = {}
    for nat in natives:
        cols[nat] = rng.normal(0.0, 0.1, n)
    for i in range(1, n_layers + 1):
        z = np.full(n, 6.0 * (i - 1)) + (rng.normal(0.0, 0.01, n) if i > 1 else 0.0)
        if i > n_layers // 2:
            z = z + np.linspace(0.0, rise, n)
        cols[f'v_comz_L{i}'] = z
    cols['v_temp_center_3d'] = 300.0 + rng.normal(0.0, 2.0, n)
    lines = ['# schema=2 lf_units=nN family=sheetonsheet',
             '# TimeStep ' + ' '.join(natives)]
    lines += [f'{(k + 1) * 1000} ' + ' '.join(f'{cols[c][k]:.6f}' for c in natives)
              for k in range(n)]
    path = run_dir / 'results' / f'friction_p{pressure:g}_a0_s2.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _sheet_tree(root, n_layers=4, final=True, ref=True, layer_map=True, **kw):
    run_dir = make_family_run(root, 'sheetonsheet', loads=(1.0,))
    _set_layer_map(run_dir, {f'layer_{i}': [i] for i in range(1, n_layers + 1)}
                   if layer_map else {})
    _sheet_results(run_dir, n_layers=n_layers, **kw)
    stack, box = _stack(n_layers)
    if ref:
        _write_data(run_dir / 'build' / 'stack_4.lmp', stack, box, style='molecular')
    if final:
        _write_data(run_dir / 'data' / 'final_p1GPa_a0.data', stack, box, style='molecular')
    return run_dir


def _point_and_record(root):
    # integrity=False: these tests run check_point themselves, on the
    # file-level status DataReader gives before stage 3.
    reader = DataReader(results_dir=str(root), integrity=False)
    point = reader.runs[0]
    record = read_provenance(point.locator.provenance_file)
    return point, record


def test_sheet_window_channels_reach_series(tmp_path):
    """Carry-over b: every channel name the checker keys on arrives from DataReader."""
    _sheet_tree(tmp_path)
    point, _ = _point_and_record(tmp_path)
    assert point.n_layers == 4
    for col in ('lfx', 'lfy', 'nf', 'temp_center_3d', 'etotal',
                *[f'comz_L{i}' for i in range(1, 5)]):
        assert col in point.window.columns, col
    m = series_measures(point.window, 'sheetonsheet', 4, 300.0, 512, S)
    assert np.isfinite(m['gap_rise']) and np.isfinite(m['gap_min_any'])
    assert np.isfinite(m['T_sigma'])
    assert m['non_finite'] is False and m['frozen'] is False


def test_afm_window_channels_reach_series(tmp_path):
    make_family_run(tmp_path, 'afm', n=50)
    point, _ = _point_and_record(tmp_path)
    for col in ('nf', 'lfx', 'lfy', 'comx', 'comy', 'comz', 'tipx', 'tipy', 'tipz',
                'temp_sheet_3d', 'comz_L1', 'comz_L2', 'etotal'):
        assert col in point.window.columns, col


def test_clean_sheet_point_is_complete(tmp_path):
    _sheet_tree(tmp_path)
    point, record = _point_and_record(tmp_path)
    v = check_point(point, record, S, multi_speed=False)
    assert v.status == 'complete', v
    assert v.measures['n_heat'] == 2 * 256          # layers 2-3 of the 4
    assert np.isfinite(v.measures['gap_rise']) and v.measures['hole'] < 1.5


def test_sheet_gap_shift_fails_through_check_point(tmp_path):
    _sheet_tree(tmp_path, rise=1.0)                  # 6 A gap opens by 1 A
    point, record = _point_and_record(tmp_path)
    v = check_point(point, record, S, multi_speed=False)
    assert v.status == 'failed_integrity' and v.reasons[0].startswith('gap_rise')


def test_missing_snapshot_quarantines(tmp_path):
    _sheet_tree(tmp_path, final=False)
    point, record = _point_and_record(tmp_path)
    v = check_point(point, record, IntegritySettings(), multi_speed=False)
    assert v.status == 'quarantined' and 'snapshot' in v.quarantine[0]


def test_missing_reference_quarantines(tmp_path):
    _sheet_tree(tmp_path, ref=False)
    point, record = _point_and_record(tmp_path)
    v = check_point(point, record, IntegritySettings(), multi_speed=False)
    assert v.status == 'quarantined' and 'reference' in v.quarantine[0]


def test_unreadable_snapshot_quarantines(tmp_path):
    run_dir = _sheet_tree(tmp_path)
    (run_dir / 'data' / 'final_p1GPa_a0.data').write_text('garbage\n', encoding='utf-8')
    point, record = _point_and_record(tmp_path)
    v = check_point(point, record, S, multi_speed=False)
    assert v.status == 'quarantined' and 'unreadable' in v.quarantine[0]


def test_multi_speed_skips_snapshot_rules(tmp_path):
    _sheet_tree(tmp_path, final=False)
    point, record = _point_and_record(tmp_path)
    v = check_point(point, record, S, multi_speed=True)
    assert v.status == 'complete'
    assert {'hole', 'isolated_atoms', 'ejected_atoms'} <= set(v.not_applicable)
    assert 'snapshot rules skipped: multi-speed run' in v.not_applicable
    assert np.isfinite(v.measures['T_sigma'])       # n_heat still counted from the reference


def test_unresolvable_layers_quarantine(tmp_path):
    """Review Focus 1: empty map, no group lines, overlapping z-band types."""
    run_dir = _sheet_tree(tmp_path, layer_map=False)
    stack, box = _stack(2)
    mixed = [(1, stack[0][1][::2]), (2, stack[0][1][1::2]),
             (1, stack[1][1][::2]), (2, stack[1][1][1::2])]
    _write_data(run_dir / 'build' / 'stack_4.lmp', mixed, box)
    _write_data(run_dir / 'data' / 'final_p1GPa_a0.data', mixed, box)
    point, record = _point_and_record(tmp_path)
    assert record.layer_type_map == {}
    v = check_point(point, record, S, multi_speed=False)
    assert v.status == 'quarantined' and 'no layer types' in v.quarantine[0]


def _hetero_ref(path):
    stack, box = _stack(2)
    a, b = stack[0][1], stack[1][1]
    layers = [(1, a[::2]), (2, a[1::2] + [0, 0, 1.5]), (3, b[::2]), (4, b[1::2] + [0, 0, 1.5])]
    return _write_data(path, layers, box)


def test_zband_fallback_for_hetero(tmp_path):
    run_dir = make_family_run(tmp_path, 'hetero', material='h-MoS2', second_material='h-WS2')
    _set_layer_map(run_dir, {})
    ref = _hetero_ref(run_dir / 'build' / 'hetero.lmp')
    loc = discover_runs(tmp_path)[0]
    rec = read_provenance(loc.provenance_file)
    assert resolve_layers(loc, rec, ref, 2) == {'layer_1': {1, 2}, 'layer_2': {3, 4}}


def test_resolve_layers_order(tmp_path):
    run_dir = make_family_run(tmp_path, 'sheetonsheet')
    loc = discover_runs(tmp_path)[0]
    rec = read_provenance(loc.provenance_file)
    ref = _hetero_ref(run_dir / 'build' / 'x.lmp')
    # 1. provenance map wins
    assert resolve_layers(loc, rec, ref, 2) == {'layer_1': {1}, 'layer_2': {2}}
    # 2. then the group lines, with s48's `a*b` expansion
    _set_layer_map(run_dir, {})
    rec = read_provenance(loc.provenance_file)
    (run_dir / 'lammps' / 'system.in.settings').write_text(
        'group layer_1 type 1*2 5\ngroup layer_2 type 3 4\n', encoding='utf-8')
    assert resolve_layers(loc, rec, ref, 2) == {'layer_1': {1, 2, 5}, 'layer_2': {3, 4}}
    # 3. no map, no groups, no reference -> {}
    (run_dir / 'lammps' / 'system.in.settings').unlink()
    assert resolve_layers(loc, rec, None, 2) == {}


def test_bond_reference_afm_prefers_load_snapshot(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm', final_data=True)
    (run_dir / 'data' / 'load_30N.data').write_text('x', encoding='utf-8')
    (run_dir / 'data' / 'load_3N.data').write_text('x', encoding='utf-8')
    (run_dir.parent / 'build').mkdir()
    (run_dir.parent / 'build' / 'sheet.lmp').write_text('x', encoding='utf-8')
    point, _ = _point_and_record(tmp_path)
    assert bond_reference(point.locator, 'afm', point) == run_dir / 'data' / 'load_30N.data'
    (run_dir / 'data' / 'load_30N.data').unlink()
    assert bond_reference(point.locator, 'afm', point) == run_dir.parent / 'build' / 'sheet.lmp'


def test_bond_reference_sheet_is_build(tmp_path):
    run_dir = _sheet_tree(tmp_path)
    point, _ = _point_and_record(tmp_path)
    assert bond_reference(point.locator, 'sheetonsheet', point) == run_dir / 'build' / 'stack_4.lmp'
    (run_dir / 'build' / 'stack_4.lmp').unlink()
    assert bond_reference(point.locator, 'sheetonsheet', point) is None


def test_check_files_without_window_or_files():
    v = check_files('no_output', None, 'afm', 2, 300.0, None, None, {}, S)
    assert (v.status, v.failed, v.observed) == ('no_output', True, False)
    assert v.quarantine == []


def test_check_files_afm_n_heat_sums_all_sheet_layers(tmp_path):
    stack, box = _stack(2)
    f = _write_data(tmp_path / 'final.data', stack + [(3, stack[1][1][:4] + [0, 0, 5])], box)
    r = _write_data(tmp_path / 'load.data', stack + [(3, stack[1][1][:4] + [0, 0, 5])], box)
    v = check_files('complete', None, 'afm', 2, 300.0, f, r,
                    {'layer_1': {1}, 'layer_2': {2}}, S)
    assert v.measures['n_heat'] == 512 and v.measures['hole'] < 1.5
