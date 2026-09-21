"""summarise_run is the ONLY COF computation on the store path (spec §3 F5)."""
import math

import numpy as np
import pandas as pd
import pytest

from src.core.observables import channel_by_role
from src.data.models import RunSummary, compute_friction_stats, summarise_run


def _window(n=100):
    return pd.DataFrame({
        'time': np.arange(n) * 1000,
        'nf': np.full(n, 30.0),
        'lfx': np.linspace(0.0, 6.0, n),
        'lfy': np.zeros(n),
    })


def test_slide_summary_matches_the_kernel():
    df = _window()

    out = summarise_run({'window': df}, 'afm')

    kernel = compute_friction_stats(df['nf'].to_numpy(), df['lfx'].to_numpy(),
                                    df['lfy'].to_numpy(), 0.2)
    assert isinstance(out, RunSummary)
    assert out.mean_cof == pytest.approx(kernel['mean_cof'])
    assert out.std_lfy == pytest.approx(kernel['std_lfy'])
    assert out.ntimesteps == 100
    assert out.skip_fraction == 0.2
    assert out.is_complete is True
    assert len(out.time_series_hash) == 64
    assert out.extras == {}


def test_as_dict_is_flat_and_merges_extras():
    out = summarise_run({'window': _window()}, 'afm')
    out.extras['pes_barrier_U'] = 0.5

    flat = out.as_dict()

    assert flat['mean_cof'] == out.mean_cof
    assert flat['pes_barrier_U'] == 0.5
    assert 'extras' not in flat


def test_driven_com_roles_become_extras():
    df = _window()
    df['tipx'] = np.linspace(0.0, 20.0, len(df))
    df['tipy'] = np.zeros(len(df))
    df['tipz'] = np.linspace(3.0, 3.4, len(df))

    out = summarise_run({'window': df}, 'afm')

    assert out.extras['slide_distance'] == pytest.approx(20.0)
    assert out.extras['driven_com_z_mean'] == pytest.approx(3.2)
    assert out.extras['driven_com_z_drift'] == pytest.approx(0.4)


def test_missing_driven_columns_add_no_extras():
    out = summarise_run({'window': _window()}, 'afm')

    assert 'slide_distance' not in out.extras
    assert 'driven_com_z_mean' not in out.extras


def test_sheet_family_resolves_its_own_channel_names():
    """R29: sheetonsheet resolves nf/lfx/lfy under the SAME column names as
    afm (both are literally 'nf'/'lfx'/'lfy' in the registry, only their
    registry KEYS differ), so those three columns can never show a
    family-specific resolution failure. The driven-com roles do differ:
    afm's driven body is tipx/tipy/tipz, the sheet family's is
    drivex/drivey/drivez (see observables.py's per-family `drive{x,y,z}`
    entries). Build the frame with the sheet's own names and prove
    summarise_run only picks them up under family='sheetonsheet' -- running
    the identical frame under family='afm' (which looks for tipx/tipy/tipz)
    yields no driven-com extras at all, so hard-coded afm names could not
    have passed this test.
    """
    dx_name = channel_by_role('sheetonsheet', 'driven_com_x').name
    dy_name = channel_by_role('sheetonsheet', 'driven_com_y').name
    dz_name = channel_by_role('sheetonsheet', 'driven_com_z').name
    assert {dx_name, dy_name, dz_name} == {'drivex', 'drivey', 'drivez'}

    df = _window()
    df[dx_name] = np.linspace(0.0, 10.0, len(df))
    df[dy_name] = np.zeros(len(df))
    df[dz_name] = np.linspace(2.0, 2.5, len(df))

    out = summarise_run({'window': df}, 'sheetonsheet')

    assert out.mean_nf == pytest.approx(30.0)
    assert out.extras['slide_distance'] == pytest.approx(10.0)
    assert out.extras['driven_com_z_mean'] == pytest.approx(2.25)
    assert out.extras['driven_com_z_drift'] == pytest.approx(0.5)

    # would pass under a family='afm' resolution too if summarise_run ignored
    # `family` (nf/lfx/lfy resolve identically) or hard-coded afm's tipx/y/z
    # -- but afm's roles resolve to tipx/tipy/tipz, absent from this frame,
    # so no driven-com extras appear at all.
    out_afm = summarise_run({'window': df}, 'afm')
    assert 'slide_distance' not in out_afm.extras
    assert 'driven_com_z_mean' not in out_afm.extras


def test_flake_family_has_no_normal_load_channel_but_does_not_raise():
    """R27: the registry declares no normal_load role for 'flake' (only
    sheetonsheet/hetero carry nf@sheet). summarise_run must not raise, and
    every stat that depends on nf -- including mean_cof and mean_nf -- must
    come out None, never 0.0."""
    out = summarise_run({'window': _window()}, 'flake')

    assert out.mean_cof is None
    assert out.mean_nf is None
    assert out.std_cof is None
    assert out.mean_lfx is None
    assert out.is_complete is False


def test_pes_family_gets_grid_descriptors_and_no_cof():
    grid = pd.DataFrame({'dx': [0, 1, 0, 1], 'dy': [0, 0, 1, 1],
                         'energy_eV': [-1.0, -0.5, -0.6, -0.2]})

    out = summarise_run({'window': grid}, 'pes_sheet')

    assert out.mean_cof is None          # NOT 0.0 -- a grid has no COF
    assert out.extras['pes_barrier_U'] == pytest.approx(0.8)
    assert out.ntimesteps == 4


def test_missing_cof_is_none_not_zero():
    """0.0 is indistinguishable from superlubricity and validates as in-range."""
    empty = summarise_run({}, 'afm')
    unresolved = summarise_run(
        {'window': pd.DataFrame({'time': [1, 2], 'x': [1.0, 2.0]})}, 'afm')

    for out in (empty, unresolved):
        assert out.mean_cof is None
        assert out.mean_nf is None
        assert out.std_lfy is None
        assert out.is_complete is False


def test_non_finite_extras_become_none():
    """AiiDA's clean_value raises on nan/inf, which would abort the run's store."""
    n = 40
    raw = pd.DataFrame({'time': np.arange(n) * 100,
                        'raw_nf': np.full(n, 30.0),
                        'raw_lfx': np.full(n, 1.0),      # flat -> std 0 -> nan
                        'raw_lfy': np.zeros(n)})
    retract = pd.DataFrame({'z_tip': [0.0, 1.0, 2.0],
                            'fz_tip': [-1.0, -2.0, -0.1]})   # never loaded -> nan

    out = summarise_run({'window': _window(), 'raw': raw, 'retract': retract},
                        'indent')

    assert out.extras['slip_asymmetry'] is None
    assert out.extras['contact_stiffness'] is None
    assert all(v is None or math.isfinite(v)
               for v in out.extras.values() if isinstance(v, float))


def test_indent_family_gets_retract_features():
    retract = pd.DataFrame({'z_tip': [0.0, 1.0, 2.0, 3.0],
                            'fz_tip': [5.0, -2.0, -0.1, 0.0]})

    out = summarise_run({'window': _window(), 'retract': retract}, 'indent')

    assert out.extras['pulloff_force'] == pytest.approx(2.0)
    assert out.mean_nf == pytest.approx(30.0)


def test_raw_tier_adds_slip_counts():
    n = 200
    raw = pd.DataFrame({
        'time': np.arange(n) * 100,
        'raw_nf': np.full(n, 30.0),
        'raw_lfx': np.tile(np.concatenate([np.linspace(0, 5, 19), [0.0]]), 10),
        'raw_lfy': np.zeros(n),
    })

    out = summarise_run({'window': _window(), 'raw': raw}, 'afm')

    assert out.extras['slip_n_events'] >= 1
    assert 'slip_mean_amplitude' in out.extras


def test_empty_frames_are_incomplete_not_an_exception():
    out = summarise_run({}, 'afm')

    assert out.is_complete is False
    assert out.ntimesteps == 0
    assert out.mean_cof is None
