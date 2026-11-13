"""Integrity measurements ported from ML_taxonomy s48/s51 (src/postprocessing/integrity.py)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scipy')

from src.core.config import IntegritySettings  # noqa: E402
from src.postprocessing.integrity import (  # noqa: E402
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
