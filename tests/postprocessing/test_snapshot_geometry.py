"""Snapshot geometry helpers shared by the sheet-profile and integrity plots."""

import numpy as np
import pytest

from frictionsim2d.postprocessing.snapshot_geometry import radial_profile, read_type_groups


def _grid(n=40, step=1.0):
    g = np.arange(n) * step
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_radial_profile_flat_sheet_is_zero():
    xy = _grid()
    z = np.full(len(xy), 5.0)
    mid, prof = radial_profile(xy, z, np.array([20.0, 20.0]), 5.0, np.arange(0, 15, 1.5))
    assert len(mid) == len(prof) > 0
    assert np.allclose(prof, 0.0)


def test_radial_profile_dimple():
    xy = _grid()
    r2 = ((xy - 20.0) ** 2).sum(axis=1)
    z = 5.0 - np.exp(-r2 / (2 * 3.0 ** 2))
    _, prof = radial_profile(xy, z, np.array([20.0, 20.0]), 5.0, np.arange(0, 15, 1.5))
    assert prof[0] == pytest.approx(-1.0, abs=0.1)
    assert prof[-1] == pytest.approx(0.0, abs=0.01)


def test_bins_with_fewer_than_three_atoms_are_skipped():
    xy = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0], [5.2, 0.0]])
    mid, _ = radial_profile(xy, np.zeros(5), np.array([0.0, 0.0]), 0.0, np.array([0, 1, 6]))
    assert list(mid) == [3.5]


def test_read_type_groups_expands_ranges(tmp_path):
    f = tmp_path / 'system.in.settings'
    f.write_text('group layer_1 type 1 2\ngroup tip_all type 4*6\n'
                 'group layer_1 type 9\n', encoding='utf-8')
    assert read_type_groups(f) == {'layer_1': {1, 2}, 'tip_all': {4, 5, 6}}
