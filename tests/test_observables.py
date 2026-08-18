"""Tests for the central observable registry (spec §3, Task 1)."""
from src.core.observables import (
    REGISTRY, TIERS, STREAMS, ROLES,
    channels_for, expected_columns, rename_map, layer_channels,
    validate_registry,
)


def test_registry_validates():
    validate_registry()  # raises on any malformed entry


def test_every_channel_has_tier_stream_role_group():
    for name, ob in REGISTRY.items():
        assert ob.tier in TIERS, name
        assert ob.stream in STREAMS, name
        assert ob.channel_role in ROLES, name
        # spec §1.2: no bare 'tip' group anywhere
        assert ob.group != 'tip', name


def test_eint_is_peratom_tier():
    # spec §4: group/group rebuilds a neighbour list; must not run at Nevery=1
    assert REGISTRY['E_int_tip_sheet'].tier == 'peratom'


def test_eint_interlayer_is_conditional():
    assert REGISTRY['E_int_interlayer'].condition == 'is_sheet_lj'
    # template-owned: the registry must not carry a compute with per-material
    # group names it cannot know
    assert REGISTRY['E_int_interlayer'].lammps_compute is None


def test_window_columns_afm_start_with_legacy_order():
    # first 9 channels preserve the legacy semantic order (spec §10 back-compat)
    cols = expected_columns('afm', 'window')
    assert cols[:9] == ['nf', 'lfx', 'lfy', 'comx', 'comy', 'comz',
                        'tipx', 'tipy', 'tipz']


def test_rename_map_afm_window():
    m = rename_map('afm', 'window')
    assert m['v_fz_tip'] == 'nf'
    assert m['v_fx_spr'] == 'lfx'
    assert m['v_comz'] == 'comz'   # legacy headers carry v_comx/v_comy/v_comz


def test_rename_map_is_per_tier():
    # the same native name renames differently per tier (raw_ vs friction_ file)
    assert rename_map('afm', 'raw')['v_fx_spr'] == 'raw_lfx'
    assert rename_map('afm', 'window')['v_fx_spr'] == 'lfx'


def test_hold_channels_are_diagnostic_and_hold_phase():
    # spec §6: production-run hold channels are forbidden as ML features,
    # and they exist only in system_init's hold — never in slide.lmp's fixes
    for name, ob in REGISTRY.items():
        if name.startswith('hold_'):
            assert ob.channel_role == 'diagnostic', name
            assert ob.phase == 'hold', name
    win = expected_columns('afm', 'window')          # slide-phase default
    assert not any(c.startswith('hold_') for c in win)


def test_layer_channels_dynamic():
    chans = layer_channels(4)
    names = [ob.name for ob in chans]
    assert 'comz_L4' in names and 'comz_L3' in names
    assert all(ob.rename_to for ob in chans)


def test_tip_min_z_present():
    # spec §7: punch-through needs min tip z in the SLIDING script
    assert REGISTRY['tip_min_z'].tier == 'window'
    assert REGISTRY['tip_min_z'].channel_role == 'diagnostic'


def test_post_stream_channels_have_no_compute():
    # post-stream channels are derived from retained artefacts, never emitted
    for ob in REGISTRY.values():
        if ob.stream == 'post':
            assert ob.lammps_compute is None, ob.name
            assert ob.rename_to is None, ob.name
