"""`settings.output.channels` must gate the sheet family's tiers too.

The original channels work wired the gate into `common/observables.lmp` and
the AFM builder only. The sheet templates hand-write their `fix ave/time`
lines rather than taking them from the observables registry, so the setting
silently did nothing for `sheetonsheet`, `hetero` and `flake` runs -- half the
simulation types. `tests/test_output_channels.py` did not catch it because it
renders the *macro*, proving the macro works rather than that anyone uses it.

These tests render the real templates instead, which is the only way this class
of gap shows up.

Task 14 routed `sheetonsheet/slide.lmp` through the observables registry,
Task 15 did the same for `hetero/slide.lmp`, and Task 16 finished the family
with `flake/slide.lmp`: all three now have their gate live in a real
`obs_ctx` (built by `build_obs_ctx`), not a bare `channels` dict -- a stubbed
`obs_ctx = 0` renders as Jinja Undefined, which is falsy, so every tier would
silently vanish regardless of a `channels` dict passed alongside it. The
stub-based `_render`/`channels`-dict path this module used for the
not-yet-converted templates is gone along with the last template that needed
it; see `_render_flake` below for the real-`obs_ctx` replacement.
"""
import pytest
from jinja2 import Environment, meta

from src.builders.utils.obs_ctx import build_obs_ctx
from src.interfaces.jinja import PackageLoader


def _fix_emitted(rendered: str, name: str) -> bool:
    """Whether an actual `fix <name> ...` directive was emitted.

    Substring matching is not enough: the tier names also appear in comments
    ("fc_win keeps the legacy 16-column prefix"), so a naive `in` check passes
    even when the gate worked.
    """
    return any(l.startswith(f"fix ") and f" {name} " in l
               for l in rendered.split("\n"))


def _render_sheetonsheet(**channels) -> str:
    """Render sheetonsheet/slide.lmp with a real obs_ctx driving the gate.

    Everything else is stubbed as 0, same as `_render`; only `obs_ctx` needs
    to be real, since the template now reads `obs_ctx.emit_window` etc.
    rather than a bare `channels` dict.
    """
    template = 'sheetonsheet/slide.lmp'
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    source = env.loader.get_source(env, template)[0]
    ctx = {name: 0 for name in meta.find_undeclared_variables(env.parse(source))}
    ctx['n_layers'] = 3
    enabled_tiers = {'window': True, 'raw': True, 'peratom': True, **channels}
    ctx['obs_ctx'] = build_obs_ctx(
        sim_type='sheetonsheet',
        n_layers=3,
        results_file_pattern='results/friction.txt',
        use_langevin=False,
        enabled_tiers=enabled_tiers,
        contact_cutoff=5.0,
        results_freq=1000,
        optional_channels=(),
    )
    return env.get_template(template).render(ctx)


def _render_hetero(**channels) -> str:
    """Render hetero/slide.lmp with a real obs_ctx driving the gate.

    Mirrors `_render_sheetonsheet` (Task 15 routed hetero/slide.lmp through
    the observables registry the same way Task 14 did for
    sheetonsheet/slide.lmp): the template now reads
    obs_ctx.emit_window/emit_raw/emit_peratom, not a bare channels dict.
    `has_internal_lj` is pinned True, mirroring
    HeteroSheetOnSheetSimulation.write_run_scripts (hardcoded there, never a
    parameter) -- so `extra_filter` always drops the interlayer channels,
    same as the real builder.
    """
    template = 'hetero/slide.lmp'
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    source = env.loader.get_source(env, template)[0]
    ctx = {name: 0 for name in meta.find_undeclared_variables(env.parse(source))}
    ctx['n_layers'] = 4
    ctx['has_internal_lj'] = True
    bands = [{'idx': i, 'zlo': 0.0, 'zhi': 1.0} for i in range(1, 5)]
    for key in ('layers', 'sheet_layers', 'layer_groups'):
        if key in ctx:
            ctx[key] = bands
    enabled_tiers = {'window': True, 'raw': True, 'peratom': True, **channels}
    interlayer = {'E_int_interlayer', 'n_contact_il'}
    ctx['obs_ctx'] = build_obs_ctx(
        sim_type='hetero',
        n_layers=4,
        results_file_pattern='results/friction.txt',
        use_langevin=False,
        enabled_tiers=enabled_tiers,
        contact_cutoff=5.0,
        results_freq=1000,
        optional_channels=(),
        extra_filter=lambda ob: ob.name not in interlayer,
    )
    return env.get_template(template).render(ctx)


def _render_flake(**channels) -> str:
    """Render flake/slide.lmp with a real obs_ctx driving the gate.

    Mirrors `_render_hetero` (Task 16 routed flake/slide.lmp through the
    observables registry the same way Tasks 14/15 did for
    sheetonsheet/hetero): the template now reads
    obs_ctx.emit_window/emit_raw/emit_peratom, not a bare channels dict.
    Unlike hetero, no `extra_filter` is needed: flake declares no
    interlayer/contact channel at all, so `channels_for('flake', 'peratom')`
    is already empty -- there is nothing for a filter to drop.
    """
    template = 'flake/slide.lmp'
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    source = env.loader.get_source(env, template)[0]
    ctx = {name: 0 for name in meta.find_undeclared_variables(env.parse(source))}
    ctx['n_layers'] = 3
    ctx['has_internal_lj'] = True
    bands = [{'idx': i, 'zlo': 0.0, 'zhi': 1.0} for i in range(1, 4)]
    for key in ('layers', 'sheet_layers', 'layer_groups'):
        if key in ctx:
            ctx[key] = bands
    enabled_tiers = {'window': True, 'raw': True, 'peratom': True, **channels}
    ctx['obs_ctx'] = build_obs_ctx(
        sim_type='flake',
        n_layers=3,
        results_file_pattern='results/friction.txt',
        use_langevin=False,
        enabled_tiers=enabled_tiers,
        contact_cutoff=5.0,
        results_freq=1000,
        optional_channels=(),
    )
    return env.get_template(template).render(ctx)


ALL_TIERS = "sheetonsheet/slide.lmp"


def test_sheet_emits_every_tier_by_default():
    out = _render_sheetonsheet()
    for fix in ("fc_win", "fc_raw", "fc_pa"):
        assert _fix_emitted(out, fix), f"{fix} missing with all channels on"


@pytest.mark.parametrize("off,gone,kept", [
    ("raw",     "fc_raw", "fc_pa"),
    ("peratom", "fc_pa",  "fc_raw"),
])
def test_sheet_tiers_are_independently_disablable(off, gone, kept):
    out = _render_sheetonsheet(**{off: False})
    assert not _fix_emitted(out, gone), f"channels.{off}=False did not drop {gone}"
    assert _fix_emitted(out, kept), f"disabling {off} also dropped {kept}"
    assert _fix_emitted(out, "fc_win"), "the window tier is the COF source; it must survive"


def test_sheet_window_tier_can_be_disabled():
    out = _render_sheetonsheet(window=False)
    assert not _fix_emitted(out, "fc_win")


def test_hetero_emits_window_and_raw_by_default():
    """Task 15: hetero/slide.lmp gained the raw tier (it never had one before
    routing through the registry)."""
    out = _render_hetero()
    assert _fix_emitted(out, "fc_win")
    assert _fix_emitted(out, "fc_raw")


def test_hetero_window_tier_can_be_disabled():
    out = _render_hetero(window=False)
    assert not _fix_emitted(out, "fc_win")
    assert _fix_emitted(out, "fc_raw")


def test_hetero_raw_tier_can_be_disabled():
    out = _render_hetero(raw=False)
    assert not _fix_emitted(out, "fc_raw")
    assert _fix_emitted(out, "fc_win")


def test_hetero_peratom_stays_absent_regardless_of_the_channel_setting():
    """hetero's has_internal_lj is hardcoded True
    (HeteroSheetOnSheetSimulation.write_run_scripts), so extra_filter always
    drops its one peratom channel (E_int_interlayer): the peratom varlist is
    empty and build_obs_ctx's emptiness gate suppresses fc_pa no matter what
    channels.peratom says. Not a bug -- task-15 brief Ruling C. A real hetero
    peratom channel is a later opt-in addition, not something this task
    forces into existence."""
    assert not _fix_emitted(_render_hetero(), "fc_pa")
    assert not _fix_emitted(_render_hetero(peratom=True), "fc_pa")


def test_flake_emits_only_the_window_tier_by_default():
    """flake declares no raw- or peratom-tier channel at all (unlike hetero,
    which gained the raw tier in Task 15): both varlists are empty regardless
    of the channel setting, so build_obs_ctx's emptiness gate suppresses
    fc_raw/fc_pa unconditionally. Not a bug -- task-16 brief: 'that is the
    correct outcome; do not manufacture a tier to make something appear.'"""
    out = _render_flake()
    assert _fix_emitted(out, "fc_win")
    assert not _fix_emitted(out, "fc_raw")
    assert not _fix_emitted(out, "fc_pa")


def test_flake_window_tier_can_be_disabled():
    out = _render_flake(window=False)
    assert not _fix_emitted(out, "fc_win")


def test_flake_raw_and_peratom_stay_absent_regardless_of_the_channel_setting():
    """Turning raw/peratom ON in settings cannot summon a tier flake never
    declared a channel for."""
    out = _render_flake(raw=True, peratom=True)
    assert not _fix_emitted(out, "fc_raw")
    assert not _fix_emitted(out, "fc_pa")


def test_the_shared_group_group_compute_survives_disabling_peratom():
    """Unlike AFM, the sheet's group/group compute is NOT peratom-exclusive.

    `compute frict` feeds v_xfrict (window) as well as v_e_int_il (peratom), so
    dropping it with the peratom tier would break the COF source. AFM can drop
    its peratom computes; the sheet family cannot.
    """
    out = _render_sheetonsheet(peratom=False)
    assert "compute         frict" in out
    assert not _fix_emitted(out, "fc_pa")
