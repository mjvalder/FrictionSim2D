"""Tests for the central observable registry."""
from src.core.observables import (
    REGISTRY, TIERS, STREAMS,
    channels_for, expected_columns, rename_map, layer_channels,
    validate_registry,
)


def test_registry_validates():
    validate_registry()  # raises on any malformed entry


def test_every_channel_has_tier_stream_group():
    for name, ob in REGISTRY.items():
        assert ob.tier in TIERS, name
        assert ob.stream in STREAMS, name
        # no bare 'tip' group anywhere: it is not a LAMMPS group in these scripts
        assert ob.group != 'tip', name


def test_eint_is_peratom_tier():
    # group/group rebuilds a neighbour list; must not run at Nevery=1
    assert REGISTRY['E_int_tip_sheet'].tier == 'peratom'


def test_eint_interlayer_is_template_owned():
    # emitted conditionally by sheetonsheet/slide.lmp: the registry must not
    # carry a compute with per-material group names it cannot know, but it
    # must still know how to parse the column back
    assert REGISTRY['E_int_interlayer'].lammps_compute is None
    assert REGISTRY['E_int_interlayer'].rename_to == 'v_e_int_il'


def test_window_columns_afm_start_with_legacy_order():
    # first 9 channels preserve the legacy 2026 column order
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


def test_hold_channels_are_not_slide_columns():
    # the constant-load hold is written by system_init to its own hold_*.txt,
    # never into slide.lmp's fixes
    win = expected_columns('afm', 'window')
    assert not any(c.startswith('hold_') for c in win)


def test_layer_channels_dynamic():
    chans = layer_channels(4)
    names = [ob.name for ob in chans]
    assert 'comz_L4' in names and 'comz_L3' in names
    assert all(ob.rename_to for ob in chans)


def test_tip_min_z_present():
    # punch-through detection needs min tip z in the SLIDING script
    assert REGISTRY['tip_min_z'].tier == 'window'


def test_post_stream_channels_have_no_compute():
    # post-stream channels are derived in postprocessing, never a LAMMPS column
    for ob in REGISTRY.values():
        if ob.stream == 'post':
            assert ob.lammps_compute is None, ob.name
            assert ob.rename_to is None, ob.name
