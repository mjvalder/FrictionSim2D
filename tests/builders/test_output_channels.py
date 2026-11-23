"""settings.output.channels gates which observable tiers get a fix ave/time.

The raw and peratom tiers cost runtime and disk on every run, and nothing in
postprocessing ingests them today (read_data's filename patterns are anchored
on the ``friction_`` prefix). They stay on by default so existing campaigns are
unaffected; these tests pin the opt-out.
"""
from jinja2 import Environment

from frictionsim2d.core.config import OutputSettings
from frictionsim2d.interfaces.jinja import PackageLoader


def _render_fixes(**flags) -> str:
    """Render emit_output_fixes with the given tier flags."""
    env = Environment(loader=PackageLoader('frictionsim2d.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    macros = env.get_template('common/observables.lmp').module
    ctx = {
        'emit_window': True, 'emit_raw': True, 'emit_peratom': True,
        'window_varlist': 'v_fz_tip', 'raw_varlist': 'v_fx_spr',
        'peratom_varlist': 'v_n_contact',
        'results_file_pattern': 'friction_x.txt',
        'results_file_raw': 'raw_x.txt',
        'results_file_peratom': 'peratom_x.txt',
    }
    ctx.update(flags)
    return macros.emit_output_fixes(ctx)


def test_channels_default_to_all_on():
    """Default must preserve pre-setting behaviour: all three tiers written."""
    assert OutputSettings().channels == {
        'window': True, 'raw': True, 'peratom': True,
    }


def test_all_tiers_emitted_by_default():
    out = _render_fixes()
    assert 'fc_win' in out
    assert 'fc_raw' in out
    assert 'fc_pa' in out


def test_disabled_tiers_are_not_emitted():
    out = _render_fixes(emit_raw=False, emit_peratom=False)
    assert 'fc_win' in out, "the window tier is the COF source; it must survive"
    assert 'fc_raw' not in out
    assert 'fc_pa' not in out


def test_partial_disable_keeps_the_other():
    """Tiers are independent -- disabling raw must not disturb peratom."""
    out = _render_fixes(emit_raw=False)
    assert 'fc_raw' not in out
    assert 'fc_pa' in out
