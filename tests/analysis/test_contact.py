"""`contact` (plan Task 6): tip-sheet contact metrics from the load and final snapshots."""
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
from src.analysis.contact import RMIN, anion  # noqa: E402
from src.analysis.source import RunSource  # noqa: E402

#: A bilayer: layer 1 = Mo (type 7), layer 2 = S (type 8); the tip is types 4-6.
SETTINGS = """\
mass 1*6 28.085000 #Si
mass 7 95.95 #Mo
mass 8 32.06 #S
group tip_all type 4 5 6
group layer_1 type 7
group layer_2 type 8
pair_coeff 4*6 7 lj/cut 0.01 3.0 # Si(tip)-Mo(sheet)
pair_coeff 4*6 8 lj/cut 0.02 3.4 # Si(tip)-S(sheet)
"""
A = 3.0          # sheet lattice spacing (A)
N = 30           # N x N atoms per layer: a 90 A box, far field beyond 25 A
GAP = 3.0        # layer 1 -> layer 2
LIFT = 3.0       # tip atoms above the top layer: inside 2^(1/6) * 3.4 = 3.816 A


def _structure():
    """Types and positions: two flat square layers and a flat 3 x 3 tip patch,
    each tip atom directly above a top-layer atom at the box centre."""
    grid = [(i * A, j * A) for i in range(N) for j in range(N)]
    atoms = [(7, x, y, 0.0) for x, y in grid] + [(8, x, y, GAP) for x, y in grid]
    centre = (N // 2) * A
    tip = [(4, centre + i * A, centre + j * A, GAP + LIFT)
           for i in (-1, 0, 1) for j in (-1, 0, 1)]
    return atoms + tip


def _write(path, atoms):
    lines = ['LAMMPS data file', '', f'{len(atoms)} atoms', '8 atom types', '',
             f'0 {N * A} xlo xhi', f'0 {N * A} ylo yhi', '-5 30 zlo zhi', '', 'Atoms # atomic', '']
    lines += [f'{i} {t} {x} {y} {z}' for i, (t, x, y, z) in enumerate(atoms, 1)]
    path.write_text('\n'.join(lines) + '\n')
    return path


class _Fetch:
    def __init__(self, snapshots, e0=None):
        self.snapshots = snapshots   # (point_uuid, kind) -> Path
        self.e0 = e0

    def snapshot(self, row, kind):
        return self.snapshots.get((row.point_uuid, kind))

    def settings_text(self, row):
        return SETTINGS

    def frames(self, row):
        out = {'window': pd.DataFrame({'time': [0]})}
        if self.e0 is not None:
            out['peratom'] = pd.DataFrame({'time': [0, 1000], 'E_int_tip_sheet': [self.e0, 0.0]})
        return out

    def close(self):
        pass


def _row(uuid, load=10.0, status='complete', angle=0.0):
    return {'point_uuid': uuid, 'run_uuid': f'r-{uuid}', 'calc_uuid': 'c', 'node_uuid': uuid,
            'verdict_status': status, 'family': 'afm', 'materials': ['h-MoS2'],
            'flake_material': None, 'size_x': 100.0, 'size_y': 100.0, 'layers': 2,
            'load': load, 'load_kind': 'force', 'angle': angle, 'speed': 2.0}


def _contact(rows, fetch, **params):
    cls = get_analysis('contact')
    source = RunSource(pd.DataFrame(rows), 'afm', fetch=fetch)
    # a_x below A: the local surface of a tip atom is the one atom beneath it.
    return cls().compute(source, cls.Params(period_x={'h_MoS2': 1.5}, **params)), source


def _lj(atoms):
    """Brute-force unshifted 12-6 LJ of every tip x sheet pair within 11 A."""
    pairs = {7: (0.01, 3.0), 8: (0.02, 3.4)}
    tip = np.array([a[1:] for a in atoms if a[0] == 4])
    total = 0.0
    for t, x, y, z in atoms:
        if t not in pairs:
            continue
        eps, sig = pairs[t]
        r = np.linalg.norm(tip - [x, y, z], axis=1)
        r = r[r <= 11.0]
        total += float(np.sum(4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)))
    return total


def test_contact_atoms_of_a_flat_tip_on_a_flat_sheet(tmp_path):
    atoms = _structure()
    snap = _write(tmp_path / 'load_10N.data', atoms)
    table, source = _contact([_row('p1')], _Fetch({('p1', 'load'): snap, ('p1', 'final'): snap},
                                                  e0=-1.5))

    assert list(table['state']) == ['load', 'final']
    row = table.iloc[0]
    assert row['material'] == 'h_MoS2' and row['layer'] == 2 and row['load'] == 10.0
    assert row['contact_atoms'] == 9                       # the 3 x 3 S atoms under the patch
    assert row['contact_area'] == pytest.approx((2 * A) ** 2)
    assert row['top_sub_density'] == pytest.approx(N * N / (N * A) ** 2)
    assert row['contact_density'] == pytest.approx((9 / (2 * A) ** 2) / (1 / A ** 2))
    assert row['n_contact_tip'] == 9
    assert row['h_over_sigma'] == pytest.approx(LIFT / 3.4)
    assert row['sink_over_sigma'] == pytest.approx(RMIN - LIFT / 3.4)
    assert row['frac_below_tops'] == 0.0
    assert row['E_lj_eV'] == pytest.approx(_lj(atoms), rel=1e-12)
    assert row['E_lj_per_contact_eV'] == pytest.approx(_lj(atoms) / 9, rel=1e-12)
    assert row['anion'] == 'S' and row['tip_x_sigma'] == 3.4
    assert row['top_element'] == 'S' and row['top_sigma'] == 3.4
    assert row['E_int_step0_eV'] == -1.5
    assert row['a_x_A'] == 1.5
    assert source.dropped == []


def test_contact_skips_runs_without_snapshots(tmp_path):
    snap = _write(tmp_path / 's.data', _structure())
    rows = [_row('p1'), _row('p2', load=20.0), _row('p3', load=30.0),
            _row('p4', load=50.0, status='ejected')]
    fetch = _Fetch({('p1', 'load'): snap, ('p1', 'final'): snap, ('p2', 'load'): snap,
                    ('p4', 'load'): snap, ('p4', 'final'): snap})
    table, source = _contact(rows, fetch)

    assert list(zip(table['point_uuid'], table['state'])) == [('p1', 'load'), ('p1', 'final'),
                                                               ('p2', 'load')]
    assert table['E_int_step0_eV'].isna().all()          # no peratom tier
    assert sorted(source.dropped) == [('p2', 'no_final_snapshot'), ('p3', 'no_final_snapshot'),
                                      ('p3', 'no_load_snapshot'), ('p4', 'ejected')]


def test_contact_reads_only_the_window_loads_at_the_angle(tmp_path):
    snap = _write(tmp_path / 's.data', _structure())
    rows = [_row('p1'), _row('p2', load=5.0), _row('p3', angle=90.0)]
    fetch = _Fetch({(u, k): snap for u in ('p1', 'p2', 'p3') for k in ('load', 'final')})
    table, _ = _contact(rows, fetch)
    assert set(table['point_uuid']) == {'p1'}
    table, _ = _contact(rows, fetch, loads='5,10')
    assert set(table['point_uuid']) == {'p1', 'p2'}


def test_contact_drops_a_state_whose_contact_atoms_are_collinear(tmp_path):
    """A tip touching one row of atoms has a flat convex hull: scipy raises
    QhullError (a RuntimeError). The state is dropped, the run goes on."""
    atoms = _structure()
    centre = (N // 2) * A
    line = [a for a in atoms if a[0] != 4] + [(4, centre + i * A, centre, GAP + LIFT)
                                              for i in (-1, 0, 1)]
    flat = _write(tmp_path / 'line.data', line)
    good = _write(tmp_path / 'good.data', atoms)
    table, source = _contact([_row('p1'), _row('p2', load=20.0)],
                             _Fetch({('p1', 'load'): flat, ('p1', 'final'): good,
                                     ('p2', 'load'): good, ('p2', 'final'): good}))

    assert list(zip(table['point_uuid'], table['state'])) == [
        ('p1', 'final'), ('p2', 'load'), ('p2', 'final')]
    assert source.dropped == [('p1', 'contact_degenerate_load')]


def test_anion_is_the_highest_group_then_the_most_electronegative():
    # Electronegativity alone picks W (2.36) over Te (2.1) (s40:154-156).
    assert anion({'W', 'Te'}) == 'Te'
    assert anion({'Mo', 'S'}) == 'S'
    assert anion({'S', 'Se'}) == 'S'          # same group: S 2.58 > Se 2.55
    assert anion({'P'}) == 'P'


# --------------------------------------------------------------------------
# publication check: the read-only `friction2d_pub` store against the frozen file
# --------------------------------------------------------------------------

_FROZEN = Path('/home/matteo/mnt/PhD/Sync/Publication-data/publication_export/data/'
               '260909_afm_v4/analysis/contact_loaded_v2.csv')

_PUB_SCRIPT = """
import sys
from src.aiida import load_aiida_profile
load_aiida_profile('friction2d_pub')
from src.analysis.base import get_analysis
from src.analysis.source import RunSource
dataset, out = sys.argv[1:3]
with RunSource.from_dataset(dataset) as source:
    cls = get_analysis('contact')
    cls().compute(source, cls.Params()).to_csv(out, index=False, float_format=repr)
"""


def _from_pub_store(tmp_path, dataset):
    """`contact` over ``dataset`` of the read-only `friction2d_pub` store, in a
    subprocess (the aiida pytest plugin hides the real profile from this
    process). Computed, never stored."""
    pytest.importorskip('aiida')
    out = tmp_path / 'contact.csv'
    env = {k: v for k, v in os.environ.items() if k != 'AIIDA_PATH'}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, '-c', _PUB_SCRIPT, dataset, str(out)],
                          env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        pytest.skip(f'friction2d_pub / dataset {dataset} unavailable: {done.stderr[-300:]}')
    return pd.read_csv(out, float_precision='round_trip')


#: Compared within 1e-9 (relative to max(1, |value|)); the integer ones exactly.
_FLOATS = ['contact_area', 'top_sub_density', 'contact_density', 'h_over_sigma',
           'sink_over_sigma', 'frac_below_tops', 'E_lj_eV', 'E_lj_per_contact_eV',
           'tip_x_sigma', 'top_sigma', 'a_x_A']
_INTS = ['contact_atoms', 'n_contact_tip']
_STRINGS = ['anion', 'top_element']


def _close(a, b, tol=1e-9):
    if pd.isna(a) and pd.isna(b):
        return True
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


@pytest.mark.publication
def test_contact_matches_the_frozen_table(tmp_path):
    if not _FROZEN.exists():
        pytest.skip('frozen contact_loaded_v2.csv absent')
    table = _from_pub_store(tmp_path, '260909-afm')
    table = table[table['layer'] == 2]
    key = ['material', 'load', 'state']
    ours = table.set_index(key)
    frozen = pd.read_csv(_FROZEN, float_precision='round_trip')
    frozen = frozen.assign(load=frozen['load_nN'].astype(float)).set_index(key)
    assert not ours.index.duplicated().any()

    shared = ours.index.intersection(frozen.index)
    absent = frozen.index.difference(ours.index)
    off = {}
    for k in shared:
        got, want = ours.loc[k], frozen.loc[k]
        bad = {c: (got[c], want[c]) for c in _FLOATS if not _close(got[c], want[c])}
        bad.update({c: (got[c], want[c]) for c in _INTS if int(got[c]) != int(want[c])})
        bad.update({c: (got[c], want[c]) for c in _STRINGS if got[c] != want[c]})
        # The ML_taxonomy parquet held E_int_tip_sheet as float32 (s40:131-133).
        if not (pd.isna(got['E_int_step0_eV']) and pd.isna(want['E_int_step0_eV'])) and \
                np.float32(got['E_int_step0_eV']) != np.float32(want['E_int_step0_eV']):
            bad['E_int_step0_eV'] = (got['E_int_step0_eV'], want['E_int_step0_eV'])
        if bad:
            off[k] = bad
    print(f'\nours {len(ours)} layer-2 rows, frozen {len(frozen)}, shared {len(shared)}; '
          f'frozen rows absent from ours: {len(absent)} {list(absent)[:12]}; '
          f'off: {len(off)} {dict(list(off.items())[:10])}')
    # p_CTe's bilayer is absent from the store (ConfigMismatch at the AFM import,
    # 2026-11-18; the Task 4 ruling): its 8 frozen rows have no run here.
    assert {k[0] for k in absent} <= {'p_CTe'}
    assert len(shared) == len(frozen) - len(absent)
    assert not off
