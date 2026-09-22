"""Slip analyser tests (spec §5 F4, Task 11)."""
import numpy as np
import pandas as pd
import pytest

from src.postprocessing.slip_analysis import detect_slips, window_stats


def _sawtooth(n_teeth=20, pts_per_tooth=50):
    ramp = np.linspace(0.0, 1.0, pts_per_tooth)
    f = np.tile(ramp, n_teeth)
    t = np.arange(f.size) * 0.1   # 100 fs sampling
    return pd.DataFrame({'time': t, 'raw_lfx': f})


def test_detects_all_teeth():
    res = detect_slips(_sawtooth())
    assert res.n_events in (19, 20)   # boundary-tooth tolerance


def test_1ps_blockmean_hides_teeth():
    # NB the tooth period must equal the block size: a 50-sample tooth
    # SURVIVES a 10-sample block mean (the inter-tooth drop is ~0.8 full
    # scale). With tooth period == block size the blocking flattens it.
    df = _sawtooth(n_teeth=20, pts_per_tooth=10)
    block = df.groupby(df.index // 10).mean()
    assert detect_slips(block).n_events < 5


def test_window_stats_reconciles_with_blockmean():
    df = _sawtooth()
    ws = window_stats(df, window_ps=1.0)
    assert abs(ws['raw_lfx_mean'].iloc[0] - df['raw_lfx'][:10].mean()) < 1e-12
    assert ws['raw_lfx_max'].iloc[0] == df['raw_lfx'][:10].max()


def test_flat_trace_has_no_events():
    df = pd.DataFrame({'time': np.arange(100) * 0.1,
                       'raw_lfx': np.ones(100)})
    assert detect_slips(df).n_events == 0


def test_versioned():
    assert detect_slips(_sawtooth()).version == '1.0.0'


def _sawtooth_trace(n=200, teeth=10):
    per = n // teeth
    ramp = np.concatenate([np.linspace(0, 5, per - 1), [0.0]])
    return pd.DataFrame({
        'time': np.arange(n) * 100,
        'raw_lfx': np.tile(ramp, teeth)[:n],
    })


def test_the_field_is_named_for_what_it_computes():
    """`asymmetry` was documented as a rise/fall ratio and returned the mean
    number of SAMPLES between events. The name is now the computation."""
    from src.postprocessing.slip_analysis import SlipResult

    assert 'asymmetry' not in SlipResult.__dataclass_fields__
    assert 'mean_interval_samples' in SlipResult.__dataclass_fields__


def test_mean_interval_samples_is_the_spacing_between_events():
    result = detect_slips(_sawtooth_trace(), channel='raw_lfx')

    assert result.n_events >= 2
    assert result.mean_interval_samples == pytest.approx(20.0, abs=1.0)


def test_summary_is_the_four_extras_keys():
    result = detect_slips(_sawtooth_trace(), channel='raw_lfx')

    out = result.summary()

    assert set(out) == {'slip_n_events', 'slip_amplitude_mean',
                        'slip_amplitude_std', 'slip_mean_interval_samples'}
    assert out['slip_n_events'] == float(result.n_events)
    assert out['slip_amplitude_mean'] == pytest.approx(
        float(np.mean(result.amplitudes)))


def test_summary_of_an_eventless_trace_is_all_zeros_but_the_interval():
    flat = pd.DataFrame({'time': [0, 100, 200], 'raw_lfx': [1.0, 1.0, 1.0]})

    out = detect_slips(flat, channel='raw_lfx').summary()

    assert out['slip_n_events'] == 0.0
    assert out['slip_amplitude_mean'] == 0.0
    assert out['slip_amplitude_std'] == 0.0
    assert np.isnan(out['slip_mean_interval_samples'])
