"""`settings.output.channels` must gate the sheet family's tiers too.

The original channels work wired the gate into `common/observables.lmp` and
the AFM builder only. The sheet templates hand-write their `fix ave/time`
lines rather than taking them from the observables registry, so the setting
silently did nothing for `sheetonsheet`, `hetero` and `flake` runs -- half the
simulation types. `tests/test_output_channels.py` did not catch it because it
renders the *macro*, proving the macro works rather than that anyone uses it.

These tests render the real templates instead, which is the only way this class
of gap shows up.

Note the sheet varlists are still hand-written: routing them through the
registry would change which columns are emitted (the registry declares 10
window observables for sheetonsheet; the template emits 19 + 3n), which is a
separate change with its own risk. Only the gate is shared.
"""
import pytest
from jinja2 import Environment, meta

from src.interfaces.jinja import PackageLoader


def _fix_emitted(rendered: str, name: str) -> bool:
    """Whether an actual `fix <name> ...` directive was emitted.

    Substring matching is not enough: the tier names also appear in comments
    ("fc_win keeps the legacy 16-column prefix"), so a naive `in` check passes
    even when the gate worked.
    """
    return any(l.startswith(f"fix ") and f" {name} " in l
               for l in rendered.split("\n"))


def _render(template: str, **channels) -> str:
    """Render a template with every variable stubbed except the channel gate.

    Supplying 0 for the whole undeclared-variable set keeps the test about the
    gate, not about producing a runnable script.
    """
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    source = env.loader.get_source(env, template)[0]
    ctx = {name: 0 for name in meta.find_undeclared_variables(env.parse(source))}
    ctx['n_layers'] = 3
    # Names the templates iterate over rather than interpolate. A scalar stub
    # raises "'int' object is not iterable" before reaching the gate.
    bands = [{'idx': i, 'zlo': 0.0, 'zhi': 1.0} for i in (1, 2, 3)]
    for key in ('layers', 'sheet_layers', 'layer_groups'):
        if key in ctx:
            ctx[key] = bands
    ctx['channels'] = {'window': True, 'raw': True, 'peratom': True, **channels}
    return env.get_template(template).render(ctx)


ALL_TIERS = "sheetonsheet/slide.lmp"
WINDOW_ONLY = ["hetero/slide.lmp", "flake/slide.lmp"]


def test_sheet_emits_every_tier_by_default():
    out = _render(ALL_TIERS)
    for fix in ("fc_win", "fc_raw", "fc_pa"):
        assert _fix_emitted(out, fix), f"{fix} missing with all channels on"


@pytest.mark.parametrize("off,gone,kept", [
    ("raw",     "fc_raw", "fc_pa"),
    ("peratom", "fc_pa",  "fc_raw"),
])
def test_sheet_tiers_are_independently_disablable(off, gone, kept):
    out = _render(ALL_TIERS, **{off: False})
    assert not _fix_emitted(out, gone), f"channels.{off}=False did not drop {gone}"
    assert _fix_emitted(out, kept), f"disabling {off} also dropped {kept}"
    assert _fix_emitted(out, "fc_win"), "the window tier is the COF source; it must survive"


def test_sheet_window_tier_can_be_disabled():
    out = _render(ALL_TIERS, window=False)
    assert not _fix_emitted(out, "fc_win")


@pytest.mark.parametrize("template", WINDOW_ONLY)
def test_hetero_and_flake_window_tier_is_gated(template):
    """Both carry only the window tier, under the name fc_ave."""
    assert _fix_emitted(_render(template), "fc_ave")
    assert not _fix_emitted(_render(template, window=False), "fc_ave")


def test_the_shared_group_group_compute_survives_disabling_peratom():
    """Unlike AFM, the sheet's group/group compute is NOT peratom-exclusive.

    `compute frict` feeds v_xfrict (window) as well as v_e_int_il (peratom), so
    dropping it with the peratom tier would break the COF source. AFM can drop
    its peratom computes; the sheet family cannot.
    """
    out = _render(ALL_TIERS, peratom=False)
    assert "compute         frict" in out
    assert not _fix_emitted(out, "fc_pa")
