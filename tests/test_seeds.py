"""Tests for deterministic seed derivation (spec §5 F5, Task 2)."""
from src.core.seeds import derive_seed

BASE = 20260803


def _s(**kw):
    d = dict(material='h-MoS2', layer=2, load=30.0, angle=0.0,
             seed_index=0, channel='thermal_tip')
    d.update(kw)
    return derive_seed(BASE, **d)


def test_deterministic():
    assert _s() == _s()


def test_range_valid_for_lammps():
    assert 1 <= _s() <= 900_000_000


def test_changing_load_changes_seed():
    # spec §5 F5: the REAL hazard is sharing a seed across loads/angles
    # within one material, where the initial realization is literally shared.
    assert _s(load=30.0) != _s(load=50.0)


def test_changing_angle_changes_seed():
    assert _s(angle=0.0) != _s(angle=30.0)


def test_channels_independent():
    assert _s(channel='thermal_tip') != _s(channel='thermal_sub')


def test_seed_index_changes_seed():
    assert _s(seed_index=0) != _s(seed_index=1)


def test_none_load_distinct_from_zero():
    # None (in-file loop identity) must not collide with load=0.0
    assert _s(load=None) != _s(load=0.0)
