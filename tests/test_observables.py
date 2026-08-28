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


def test_every_results_channel_has_units():
    for name, ob in REGISTRY.items():
        if ob.stream == 'results':
            assert ob.units, f"{name} has no declared units"


def test_afm_cof_inputs_share_force_units():
    # COF = lateral force / normal load: the two must carry the same unit or
    # the ratio is meaningless.
    #
    # AFM ONLY, deliberately. Both entries are sim_types=('afm','indent');
    # the sheet family has no 'nf' channel in REGISTRY and its lateral
    # channel is 'xfrict', so this cannot be widened to the sheet without the
    # registry work in the parity spec. It therefore could NOT have caught
    # the sheet eV/A-over-nN defect -- tests/test_template_contract.py's
    # test_lj_branch_lateral_force_scaled_to_nN is the one that does.
    assert REGISTRY['lfx'].units == REGISTRY['nf'].units == 'nN'
    assert REGISTRY['lfy'].units == 'nN'


def test_sheet_lateral_channel_declares_nN():
    # the sheet's lateral force is 'xfrict', not 'lfx', and there is no sheet
    # 'nf' to pair it with here. Pinned so the asymmetry is visible rather
    # than silently assumed away.
    assert REGISTRY['xfrict'].units == 'nN'
    assert 'sheetonsheet' in REGISTRY['xfrict'].sim_types
    channel_names = {o.name for o in channels_for('sheetonsheet', 'window')}
    assert channel_names
    assert 'nf' not in channel_names, (
        'a sheet nf channel now exists -- widen '
        'test_afm_cof_inputs_share_force_units to cover it'
    )


def test_raw_legacy_block_is_afm_only():
    # a sheet raw file has no v_fx_spr column; applying the AFM legacy block
    # to it was inert but wrong
    assert 'v_fx_spr' not in rename_map('sheetonsheet', 'raw')
    assert rename_map('afm', 'raw')['v_fx_spr'] == 'raw_lfx'
