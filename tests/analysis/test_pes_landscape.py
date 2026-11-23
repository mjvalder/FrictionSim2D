"""`pes_landscape` (plan Task 10): the PES-grid descriptors of each PES run,
plus the Prandtl-Tomlinson eta of a given spring and period.

There is no PES set among the store's datasets; the frames here are shaped
like the ``results/pes_scan.csv`` the scans write (`afm/pes_scan.lmp:100`:
``x,y,energy_eV,fx,fy,fz``; `sheetonsheet/pes_scan.lmp:82`: ``dx,dy,energy_eV``).
"""
import math

import numpy as np
import pandas as pd
import pytest

import frictionsim2d.analysis.catalogue  # noqa: F401  (registers the analyses)
from frictionsim2d.analysis.base import get_analysis
from frictionsim2d.analysis.pes_landscape import NO_GRID, pt_eta
from frictionsim2d.analysis.source import RunSource
from frictionsim2d.postprocessing import pes_scan


def _grid(kind='sheet', n=12, amp=1.0, period=3.0, amp_y=None):
    """E = amp (1 - cos 2pi ix/n) + amp_y (1 - cos 2pi iy/n), as the scan writes it."""
    amp_y = amp if amp_y is None else amp_y
    rows = []
    for ix in range(n):
        for iy in range(n):
            e = (amp * (1 - math.cos(2 * math.pi * ix / n))
                 + amp_y * (1 - math.cos(2 * math.pi * iy / n)))
            if kind == 'tip':
                rows.append({'x': ix * period / n, 'y': iy * period / n, 'energy_eV': e,
                             'fx': 0.5 * math.sin(2 * math.pi * ix / n),
                             'fy': 0.5 * math.sin(2 * math.pi * iy / n), 'fz': -0.2})
            else:
                rows.append({'dx': ix * period / n, 'dy': iy * period / n, 'energy_eV': e})
    return pd.DataFrame(rows)


def test_pt_eta_formula():
    # eta = 2 pi^2 U / (k a^2): U = 0.1 eV, k = 0.5 eV/A^2, a = 3 A
    assert pt_eta(0.1, 0.5, 3.0) == pytest.approx(2 * math.pi ** 2 * 0.1 / (0.5 * 9.0))
    assert math.isnan(pt_eta(0.1, None, 3.0))
    assert math.isnan(pt_eta(0.1, 0.5, None))
    assert math.isnan(pt_eta(0.1, 0.0, 3.0))


class _Fetch:
    def __init__(self, frames):
        self.data = frames

    def frames(self, row):
        return self.data[row.point_uuid]

    def close(self):
        pass


def _row(uuid, family='pes_sheet', status='complete', material='h-MoS2'):
    return {'point_uuid': uuid, 'run_uuid': f'r-{uuid}', 'calc_uuid': 'c', 'node_uuid': uuid,
            'verdict_status': status, 'family': family, 'materials': [material],
            'flake_material': None, 'size_x': 30.0, 'size_y': 30.0, 'layers': 2,
            'load': 0.0, 'load_kind': 'none', 'angle': 0.0, 'speed': 0.0}


def _pes(rows, frames, family='pes_sheet', **params):
    cls = get_analysis('pes_landscape')
    source = RunSource(pd.DataFrame(rows), family, fetch=_Fetch(frames))
    return cls().compute(source, cls.Params(**params)), source


def test_pes_landscape_wraps_grid_descriptors_per_run():
    grid = _grid('sheet', amp=1.0, amp_y=0.5)
    table, source = _pes([_row('p1'), _row('p2', status='failed')], {'p1': {'window': grid}},
                         k=0.5, a=3.0)
    assert source.dropped == [('p2', 'failed')]
    assert len(table) == 1
    row = table.iloc[0]
    assert row['material'] == 'h_MoS2' and row['run_uuid'] == 'r-p1'
    for key, value in pes_scan.grid_descriptors(grid, 'sheet').items():
        assert row[key] == pytest.approx(value)
    assert row['pes_barrier_x'] == pytest.approx(2.0)      # 2 amp along x
    assert row['pes_barrier_y'] == pytest.approx(1.0)      # 2 amp_y along y
    assert row['pt_eta'] == pytest.approx(2 * math.pi ** 2 * row['pes_barrier_U'] / (0.5 * 9.0))
    assert row['pt_k_eV_A2'] == 0.5 and row['pt_a_A'] == 3.0


def test_pes_landscape_tip_carries_forces_and_eta_is_nan_without_k():
    table, _ = _pes([_row('p1', family='pes_tip')], {'p1': {'window': _grid('tip')}}, family='pes_tip')
    row = table.iloc[0]
    assert row['pes_fmax'] == pytest.approx(0.5 * math.sqrt(2), rel=1e-6)
    assert row['pes_load_mean'] == pytest.approx(-0.2)
    assert np.isnan(row['pt_eta'])


def test_pes_landscape_a_per_material():
    table, _ = _pes([_row('p1'), _row('p2', material='h-WS2')],
                    {'p1': {'window': _grid()}, 'p2': {'window': _grid()}}, k=1.0, a={'h_MoS2': 3.0})
    eta = table.set_index('material')['pt_eta']
    assert np.isfinite(eta['h_MoS2']) and np.isnan(eta['h_WS2'])


def test_pes_landscape_drops_a_run_without_a_grid():
    degenerate = pd.DataFrame({'dx': [0.0, 1.0], 'dy': [0.0, 0.0], 'energy_eV': [0.0, 1.0]})
    table, source = _pes([_row('p1'), _row('p2')], {'p1': {}, 'p2': {'window': degenerate}})
    assert table.empty
    assert source.dropped == [('p1', NO_GRID), ('p2', NO_GRID)]


@pytest.mark.parametrize('family', ['afm', 'sheetonsheet', 'indent'])
def test_pes_refuses_other_families(family):
    cls = get_analysis('pes_landscape')
    source = RunSource(pd.DataFrame([_row('p1', family=family)]), family,
                       fetch=_Fetch({'p1': {'window': _grid()}}))
    with pytest.raises(ValueError, match=family):
        cls().compute(source, cls.Params())
