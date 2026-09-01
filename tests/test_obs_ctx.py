"""Tests for the shared observable-context builder."""
from src.builders.utils.obs_ctx import build_obs_ctx


def _ctx(**kw):
    base = dict(
        sim_type='afm', n_layers=2,
        results_file_pattern='out/results/friction_p1_a0_s1',
        use_langevin=True,
        enabled_tiers={'window': True, 'raw': True, 'peratom': True},
        contact_cutoff=3.5,
        results_freq=1000,
    )
    base.update(kw)
    return build_obs_ctx(**base)


def test_returns_the_documented_keys():
    ctx = _ctx()
    for key in ('compute_lines', 'variable_lines', 'emit_window', 'emit_raw',
                'emit_peratom', 'window_varlist', 'raw_varlist',
                'peratom_varlist', 'results_file_pattern',
                'results_file_raw', 'results_file_peratom'):
        assert key in ctx, key


def test_disabled_peratom_tier_drops_its_computes():
    # the peratom computes are the expensive ones; emitting them for a fix
    # that is never written wastes the run
    ctx = _ctx(enabled_tiers={'window': True, 'raw': True, 'peratom': False})
    assert ctx['emit_peratom'] is False
    assert ctx['peratom_varlist'] == ''


def test_non_langevin_drops_thermostat_work_channels():
    # w_thermostat_* reference f_lang_*, which does not exist under nose-hoover
    ctx = _ctx(use_langevin=False)
    assert 'w_thermostat' not in ctx['window_varlist']


def test_computes_are_deduped():
    # comx/comy/comz_L1 share one `compute COM_L1`; emitting it three times
    # would redefine it
    lines = _ctx()['compute_lines']
    assert len(lines) == len(set(lines))


def test_window_args_carry_results_freq():
    # Nevery/Nrepeat come from TIER_ARGS; Nfreq is the user's setting.
    # AFM silently lost this when it converted to the macro.
    assert _ctx(results_freq=2000)['window_args'] == '1 1000 2000'


def test_window_args_default_matches_tier_args():
    from src.core.observables import TIER_ARGS
    every, repeat, freq = TIER_ARGS['window']
    assert _ctx(results_freq=freq)['window_args'] == f'{every} {repeat} {freq}'


def test_title1_carries_schema_units_and_family():
    from src.postprocessing.read_data import DataReader
    t = _ctx(sim_type='afm')['title1']
    assert t == '# schema=2 lf_units=nN family=afm'
    # the reader must agree with what the writer stamps
    assert DataReader._parse_lf_schema(t) == DataReader._LF_SCHEMA_CURRENT


def test_empty_varlist_suppresses_the_fix():
    # `fix fc_pa all ave/time 1000 1 1000  file out` is an Illegal fix
    # ave/time command. A many-body sheet filters out its only peratom
    # channel, so this is reachable, not hypothetical.
    ctx = _ctx(extra_filter=lambda ob: ob.tier != 'peratom')
    assert ctx['peratom_varlist'] == ''
    assert ctx['emit_peratom'] is False


def test_unknown_optional_channel_fails_the_build():
    import pytest
    # a silently-ignored typo means a pilot arm runs without the
    # instrumentation it was commissioned to measure
    with pytest.raises(ValueError, match='no such observable'):
        _ctx(optional_channels=['not_a_channel'])


def test_default_build_emits_no_opt_in_channel():
    from src.core.observables import REGISTRY
    ctx = _ctx()
    emitted = set(ctx['window_varlist'].split()) | set(
        ctx['peratom_varlist'].split())
    for ob in REGISTRY.values():
        if ob.opt_in and ob.rename_to:
            assert ob.rename_to not in emitted, ob.registry_key
