"""Tests for the shared observable-context builder."""
from src.builders.utils.obs_ctx import build_obs_ctx


def _ctx(**kw):
    base = dict(
        sim_type='afm', n_layers=2,
        results_file_pattern='out/results/friction_p1_a0_s1',
        use_langevin=True,
        enabled_tiers={'window': True, 'raw': True, 'peratom': True},
        contact_cutoff=3.5,
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
