"""compute_friction_stats, compute_derived_columns, compute_time_series_hash."""
import hashlib

import numpy as np
import pytest

from frictionsim2d.data.models import (
    compute_derived_columns, compute_friction_stats, compute_time_series_hash)


def test_stats_reject_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        compute_friction_stats([], [], [])


def test_stats_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_friction_stats([1.0, 2.0], [1.0], [1.0, 2.0])


@pytest.mark.parametrize("skip", [-0.1, 1.0, 1.5])
def test_stats_reject_skip_fraction_outside_unit_interval(skip):
    with pytest.raises(ValueError, match="skip_fraction must be in"):
        compute_friction_stats([1.0, 2.0], [1.0, 1.0], [0.0, 0.0], skip)


def test_stats_with_a_single_sample_survive_the_largest_skip():
    out = compute_friction_stats([2.0], [1.0], [0.0], skip_fraction=0.999)
    assert out['mean_cof'] == pytest.approx(0.5)


def test_stats_with_nonpositive_mean_normal_force_report_zero_cof():
    nf = np.array([-1.0, -2.0, 0.0, -3.0])
    out = compute_friction_stats(nf, np.ones(4), np.zeros(4), skip_fraction=0.0)
    assert out['mean_cof'] == 0.0
    assert out['std_cof'] == 0.0 and np.isfinite(out['std_cof'])


def test_stats_drop_the_leading_transient():
    """The first `skip_fraction` of the samples is discarded: a transient of
    100 nN / 1000 nN lateral in the first 20 % must not reach any mean."""
    nf = np.array([100.0] * 2 + [10.0] * 8)
    lfx = np.array([1000.0] * 2 + [5.0] * 8)
    out = compute_friction_stats(nf, lfx, np.zeros(10), skip_fraction=0.2)
    assert out['mean_nf'] == pytest.approx(10.0)
    assert out['mean_lf'] == pytest.approx(5.0)
    assert out['mean_cof'] == pytest.approx(0.5)
    kept_all = compute_friction_stats(nf, lfx, np.zeros(10), skip_fraction=0.0)
    assert kept_all['mean_nf'] == pytest.approx(28.0)


def test_stats_with_a_zero_mean_normal_force_report_zero_cof():
    out = compute_friction_stats(np.zeros(4), np.ones(4), np.zeros(4), skip_fraction=0.0)
    assert out['mean_cof'] == 0.0


def test_derived_columns_zero_normal_force_gives_zero_cof_and_signed_nf_stays_signed():
    lfx = np.array([3.0, 3.0, 3.0])
    lfy = np.array([4.0, 4.0, 4.0])
    nf = np.array([0.0, 5.0, -5.0])
    lateral, cof = compute_derived_columns(lfx, lfy, nf)
    assert np.allclose(lateral, 5.0)
    assert cof.tolist() == [0.0, 1.0, -1.0]


def test_time_series_hash_is_stable_and_order_sensitive():
    nf, lfx, lfy = np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])
    expected = hashlib.sha256(
        nf.tobytes() + lfx.tobytes() + lfy.tobytes()).hexdigest()
    assert compute_time_series_hash(nf, lfx, lfy) == expected
    assert compute_time_series_hash(lfx, nf, lfy) != expected
    assert compute_time_series_hash(list(nf), list(lfx), list(lfy)) == expected
