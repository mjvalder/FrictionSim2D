"""summarise_run is the ONLY COF computation on the store path (spec §3 F5)."""
import math

import numpy as np
import pandas as pd
import pytest

from frictionsim2d.core.observables import channel_by_role
from frictionsim2d.data.models import RunSummary, compute_friction_stats, summarise_run


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


def test_nan_in_driven_com_channel_becomes_none_not_nan():
    """Fix round 1: slide_distance/driven_com_z_mean/driven_com_z_drift are
    derived with plain float() sums/means, so a single NaN sample in a
    driven-com channel (a dropped frame, a bad LAMMPS write) propagates
    through np.mean/np.sum/np.hypot into extras -- and AiiDA's clean_value
    would reject the whole run's store on it, same as the other extras this
    module already guards with _finite."""
    df = _window()
    df['tipx'] = np.linspace(0.0, 20.0, len(df))
    df['tipy'] = np.zeros(len(df))
    df.loc[5, 'tipy'] = np.nan
    df['tipz'] = np.linspace(3.0, 3.4, len(df))
    df.loc[len(df) - 1, 'tipz'] = np.nan

    out = summarise_run({'window': df}, 'afm')

    assert out.extras['slide_distance'] is None
    assert out.extras['driven_com_z_mean'] is None
    assert out.extras['driven_com_z_drift'] is None
    assert all(v is None or math.isfinite(v)
               for v in out.extras.values() if isinstance(v, float))


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


def test_flake_family_resolves_a_normal_load_and_gets_a_cof():
    """flake/slide.lmp loads the flake and emits v_fz (follow-up C,
    2026-09-22), so 'flake' resolves normal_load exactly like the sheet: its
    COF is lateral force / normal load, and the run is complete. (It used to
    be the known gap R27 -- no normal_load role, COF always None.)"""
    assert channel_by_role('flake', 'normal_load').name == 'nf'

    df = _window()
    out = summarise_run({'window': df}, 'flake')
    kernel = compute_friction_stats(df['nf'].to_numpy(), df['lfx'].to_numpy(),
                                    df['lfy'].to_numpy(), 0.2)

    assert out.mean_cof == pytest.approx(kernel['mean_cof'])
    assert out.mean_nf == pytest.approx(30.0)
    assert out.is_complete is True
    assert len(out.time_series_hash) == 64


def test_flake_results_file_round_trips_to_a_cof(tmp_path):
    """End to end over the synthetic writer: the flake window header now
    carries v_fz, the reader renames it to nf, and summarise_run computes a
    COF from it."""
    from frictionsim2d.postprocessing.read_data import read_results_file
    from tests.helpers.synthetic_results import make_run

    run = make_run(tmp_path, 'sheetonsheet/h-MoS2+flake-h-MoS2/100x_100y/K300',
                   'flake', loads=(1.0,))
    friction = next((run / 'results').glob('friction_*.txt'))
    header = friction.read_text().splitlines()[1].split()
    assert 'v_fz' in header

    frame = read_results_file(friction, 'flake')
    assert 'nf' in frame.columns
    out = summarise_run({'window': frame}, 'flake')
    assert out.mean_cof is not None
    assert out.is_complete is True


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

    assert out.extras['contact_stiffness'] is None
    assert all(v is None or math.isfinite(v)
               for v in out.extras.values() if isinstance(v, float))


def test_indent_family_gets_retract_features():
    retract = pd.DataFrame({'z_tip': [0.0, 1.0, 2.0, 3.0],
                            'fz_tip': [5.0, -2.0, -0.1, 0.0]})

    out = summarise_run({'window': _window(), 'retract': retract}, 'indent')

    assert out.extras['pulloff_force'] == pytest.approx(2.0)
    assert out.mean_nf == pytest.approx(30.0)


def test_raw_tier_writes_no_slip_extras():
    n = 200
    raw = pd.DataFrame({'time': np.arange(n) * 100, 'raw_nf': np.full(n, 30.0),
                        'raw_lfx': np.tile(np.linspace(0, 5, 20), 10)[:n],
                        'raw_lfy': np.zeros(n)})

    out = summarise_run({'window': _window(), 'raw': raw}, 'afm')

    assert not [k for k in out.extras if k.startswith('slip_')]


def test_empty_frames_are_incomplete_not_an_exception():
    out = summarise_run({}, 'afm')

    assert out.is_complete is False
    assert out.ntimesteps == 0
    assert out.mean_cof is None


def test_sheet_lateral_force_pairs_become_extras():
    """mean |(react_x, react_y)| and |(drive_fx, drive_fy)| over the whole window:
    the MoS2 figures plot the first (lf_source: sx_sy), so the store must carry it."""
    df = _window()
    df['react_x'] = np.full(len(df), 3.0)
    df['react_y'] = np.full(len(df), 4.0)
    out = summarise_run({'window': df}, 'sheetonsheet')
    assert out.extras['mean_lf_sx_sy'] == pytest.approx(5.0)
    assert 'mean_lf_fx_fy' not in out.extras  # no drive_fx/drive_fy columns


def test_afm_gets_no_lateral_force_pairs():
    df = _window()
    df['react_x'] = df['react_y'] = np.ones(len(df))
    assert 'mean_lf_sx_sy' not in summarise_run({'window': df}, 'afm').extras


def test_pes_summary_hashes_the_energy_column_alone():
    from frictionsim2d.data.models import compute_time_series_hash
    grid = pd.DataFrame({'dx': [0, 1, 0, 1], 'dy': [0, 0, 1, 1],
                         'energy_eV': [-1.0, -0.5, -0.6, -0.2]})
    out = summarise_run({'window': grid}, 'pes_tip')
    assert out.time_series_hash == compute_time_series_hash(
        grid['energy_eV'].to_numpy(dtype=float), np.zeros(1), np.zeros(1))


def test_lf_pair_means_are_recorded_for_sheet_families_only():
    n = 10
    window = _window(n).assign(react_x=3.0, react_y=4.0, drive_fx=6.0, drive_fy=8.0)
    sheet = summarise_run({'window': window}, 'sheetonsheet')
    assert sheet.extras['mean_lf_sx_sy'] == pytest.approx(5.0)
    assert sheet.extras['mean_lf_fx_fy'] == pytest.approx(10.0)
    assert 'mean_lf_sx_sy' not in summarise_run({'window': window}, 'afm').extras
