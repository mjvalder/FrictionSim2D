"""Tests for the central observable registry."""
import pytest

from frictionsim2d.core.observables import (
    REGISTRY, TIERS, STREAMS,
    channels_for, rename_map, layer_channels,
    validate_registry, expected_tier_rows, tier_frequencies,
)


def expected_columns(sim_type, tier):
    return [ob.name for ob in channels_for(sim_type, tier)]


def test_registry_validates():
    validate_registry()  # raises on any malformed entry


def test_every_channel_has_tier_and_stream():
    for name, ob in REGISTRY.items():
        assert ob.tier in TIERS, name
        assert ob.stream in STREAMS, name


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
    # AFM ONLY, deliberately. Both entries are sim_types=('afm','indent').
    # The sheet family now has its own 'lfx@sheet'/'nf@sheet' pair, checked
    # by test_sheet_cof_inputs_share_force_units below -- this test still
    # could NOT have caught the sheet eV/A-over-nN defect, since that lived
    # in the TEMPLATE, not the registry -- tests/builders/test_template_contract.py's
    # test_lj_branch_lateral_force_scaled_to_nN is the one that does.
    assert REGISTRY['lfx'].units == REGISTRY['nf'].units == 'nN'
    assert REGISTRY['lfy'].units == 'nN'


def test_sheet_cof_inputs_share_force_units():
    # the widening test_afm_cof_inputs_share_force_units asked for. The sheet
    # family now HAS an nf channel, so COF's two inputs can be compared for
    # dimensional coherence here rather than only in the template tests.
    assert REGISTRY['lfx@sheet'].units == REGISTRY['nf@sheet'].units == 'nN'
    assert REGISTRY['lfy@sheet'].units == 'nN'


def test_raw_legacy_block_is_afm_only():
    # a sheet raw file has no v_fx_spr column; applying the AFM legacy block
    # to it was inert but wrong
    assert 'v_fx_spr' not in rename_map('sheetonsheet', 'raw')
    assert rename_map('afm', 'raw')['v_fx_spr'] == 'raw_lfx'


def test_sheet_window_map_covers_the_emitted_natives():
    m = rename_map('sheetonsheet', 'window')
    assert m['v_xfrict'] == 'lfx'
    assert m['v_yfrict'] == 'lfy'
    assert m['v_fz'] == 'nf'
    assert m['v_sx'] == 'react_x'
    assert m['v_comz_top'] == 'drivez'
    assert m['v_fx'] == 'drive_fx'


def test_sheet_raw_map_covers_the_emitted_natives():
    m = rename_map('sheetonsheet', 'raw')
    assert m['v_xfrict'] == 'raw_lfx'
    assert m['v_comx_top'] == 'raw_drivex'
    assert 'v_fx_spr' not in m       # the AFM legacy block stays AFM-only


def test_afm_and_sheet_both_have_lfx_under_their_own_key():
    assert REGISTRY['lfx'].rename_to == 'v_fx_spr'
    assert REGISTRY['lfx@sheet'].rename_to == 'v_xfrict'
    assert REGISTRY['lfx@sheet'].name == 'lfx'


def test_registry_key_defaults_to_name():
    from frictionsim2d.core.observables import Observable
    ob = Observable(name='foo', sim_types=('afm',),
                    tier='window', stream='results', units='A')
    assert ob.registry_key == 'foo'
    assert Observable(name='foo', key='foo@sheet', sim_types=('sheetonsheet',),
                      tier='window', stream='results',
                      units='A').registry_key == 'foo@sheet'


def test_existing_lookups_unchanged():
    # every entry that existed before keeps its key
    assert REGISTRY['E_int_tip_sheet'].name == 'E_int_tip_sheet'
    assert REGISTRY['tip_min_z'].tier == 'window'


def test_shared_name_requires_disjoint_sim_types():
    import pytest
    from frictionsim2d.core.observables import Observable, _add
    with pytest.raises(ValueError, match='disjoint'):
        # 'lfx' already exists for ('afm', 'indent'); overlapping sim_types
        # would emit two identical column headers in one file. role_key is
        # set so the disjoint-name check under test is what fires, not the
        # unrelated Task 6 role_key requirement.
        _add(Observable(name='lfx', key='lfx@bogus', sim_types=('afm',),
                        tier='window', stream='results',
                        units='nN', role_key='lateral_force_x'))


def test_every_results_channel_has_a_role_key():
    from frictionsim2d.core.observables import ROLE_EXEMPT
    for key, ob in REGISTRY.items():
        if ob.stream != 'results':
            continue
        assert ob.role_key or ob.name in ROLE_EXEMPT, (
            f"{key}: results channels need a role_key so the parity contract "
            f"can see them, or an explicit ROLE_EXEMPT entry"
        )


def test_role_exempt_entries_carry_a_reason():
    from frictionsim2d.core.observables import ROLE_EXEMPT
    for name, reason in ROLE_EXEMPT.items():
        assert reason.strip(), name
        assert len(reason.split()) >= 4, f"{name}: reason is not prose"


def test_role_key_unique_within_a_sim_type():
    from collections import defaultdict
    seen = defaultdict(dict)
    for ob in REGISTRY.values():
        if not ob.role_key:
            continue
        for st in ob.sim_types:
            prev = seen[st].get(ob.role_key)
            assert prev is None, (
                f"{st}: {ob.registry_key} and {prev} both claim role "
                f"{ob.role_key}")
            seen[st][ob.role_key] = ob.registry_key


def test_add_rejects_a_results_channel_with_no_role():
    import pytest
    from frictionsim2d.core.observables import Observable, _add
    with pytest.raises(ValueError, match='role_key'):
        _add(Observable(name='rolelesss', sim_types=('afm',),
                        tier='window', stream='results', units='A'))


def test_channel_by_role_finds_the_family_specific_channel():
    from frictionsim2d.core.observables import channel_by_role
    assert channel_by_role('afm', 'normal_load').name == 'nf'
    assert channel_by_role('afm', 'no_such_role') is None


def test_role_parity_between_families():
    from frictionsim2d.core.observables import channel_by_role
    from tests.core.parity_tables import PARITY_PAIRS, PARITY_EXEMPT
    for source, target in PARITY_PAIRS:
        for ob in REGISTRY.values():
            if source not in ob.sim_types or ob.stream != 'results':
                continue
            if not ob.role_key:
                continue
            if (target, ob.role_key) in PARITY_EXEMPT:
                continue
            assert channel_by_role(target, ob.role_key) is not None, (
                f"{ob.registry_key} carries role {ob.role_key} for {source} "
                f"but {target} has no channel for it and no PARITY_EXEMPT "
                f"entry"
            )


def test_every_parity_exemption_carries_a_reason():
    from tests.core.parity_tables import PARITY_EXEMPT
    for (sim_type, role), reason in PARITY_EXEMPT.items():
        assert reason.strip(), (sim_type, role)
        assert len(reason.split()) >= 4, f"{sim_type}/{role}: not prose"


def test_every_parity_exemption_names_a_live_role():
    """An exemption must die with the role it excused. A stale entry whose
    role no longer exists anywhere is noise that outlives its subject."""
    from tests.core.parity_tables import PARITY_EXEMPT
    live = {ob.role_key for ob in REGISTRY.values() if ob.role_key}
    for (sim_type, role) in PARITY_EXEMPT:
        assert role in live, (
            f"PARITY_EXEMPT[{sim_type!r}, {role!r}] excuses a role no channel "
            f"claims any more -- delete the exemption")


def test_opt_in_channels_are_peratom_or_post():
    # opt_in exists to gate runtime cost; a window-tier channel is cheap by
    # construction, so marking one opt_in would be meaningless
    for key, ob in REGISTRY.items():
        if ob.opt_in:
            assert ob.tier in ('peratom',) or ob.stream == 'post', key


import pytest

from frictionsim2d.core import observables as obs


def _ob(**kw):
    base = dict(name='t_x', sim_types=('afm',), tier='window', stream='results',
                role_key='rk', key='t_x')
    base.update(kw)
    return obs.Observable(**base)


def test_add_rejects_a_results_channel_without_role_key(monkeypatch):
    monkeypatch.setattr(obs, 'REGISTRY', dict(obs.REGISTRY))
    with pytest.raises(ValueError, match='role_key'):
        obs._add(_ob(role_key=None, name='t_norole', key='t_norole'))


def test_add_rejects_a_duplicate_key(monkeypatch):
    monkeypatch.setattr(obs, 'REGISTRY', dict(obs.REGISTRY))
    existing = next(iter(obs.REGISTRY))
    with pytest.raises(ValueError, match='duplicate observable key'):
        obs._add(_ob(key=existing, name=existing))


def test_validate_registry_rejects_group_group_at_window_tier(monkeypatch):
    reg = dict(obs.REGISTRY)
    reg['t_gg'] = _ob(name='t_gg', key='t_gg', lammps_compute='group/group sheet tip')
    monkeypatch.setattr(obs, 'REGISTRY', reg)
    with pytest.raises(ValueError, match='group/group'):
        obs.validate_registry()


def test_rename_map_covers_layers_beyond_the_registry_pair():
    """REGISTRY holds com{x,y,z}_L1/L2 only, so a 5-layer run's L3..L5 columns
    reached the JSON under their native v_comz_L3 names. layer_channels(n) has
    always known them; rename_map just never asked."""
    five = rename_map('sheetonsheet', 'window', 5)

    assert five['v_comz_L5'] == 'comz_L5'
    assert five['v_comx_L3'] == 'comx_L3'
    assert 'v_comz_L6' not in five


def test_rename_map_default_is_the_bilayer_map():
    assert rename_map('afm', 'window') == rename_map('afm', 'window', 2)
    assert rename_map('afm', 'window')['v_fz_tip'] == 'nf'


def test_rename_map_layer_expansion_respects_the_tier():
    """pe_sum_L{n}/sigma_*_L{n} are peratom-tier; they must not leak into the
    window map just because layer_channels emits them."""
    window = rename_map('sheetonsheet', 'window', 4)
    peratom = rename_map('sheetonsheet', 'peratom', 4)

    assert 'v_pe_sum_L4' not in window
    assert peratom['v_pe_sum_L4'] == 'pe_sum_L4'
    assert peratom['v_sigma_xx_2d_L4'] == 'sigma_xx_2d_L4'


def test_rename_map_layer_expansion_respects_the_sim_type():
    """layer_channels' com{x,y,z} entries declare afm/sheetonsheet/indent/
    hetero/flake; pes_tip is not among them and must get no layer columns."""
    assert 'v_comz_L3' not in rename_map('pes_tip', 'window', 3)


CFG = {'settings': {'simulation': {'slide_run_steps': 100000},
                    'output': {'results_frequency': 1000}}}


def test_expected_rows_per_tier_for_a_slide_family():
    assert expected_tier_rows(CFG, 'afm') == {'window': 100, 'raw': 1000, 'peratom': 100}


def test_disabled_channels_have_no_expectation():
    cfg = {'settings': {**CFG['settings'], 'output': {
        'results_frequency': 1000, 'channels': {'window': True, 'raw': False, 'peratom': False}}}}
    assert expected_tier_rows(cfg, 'sheetonsheet') == {'window': 100}


@pytest.mark.parametrize('family', ['indent', 'pes_tip', 'pes_sheet'])
def test_non_slide_families_expect_nothing(family):
    assert expected_tier_rows(CFG, family) == {}


@pytest.mark.parametrize('cfg', [{}, {'settings': {}}, {'general': {'temp': 300}},
                                 {'settings': {'simulation': {'slide_run_steps': 'x'},
                                               'output': {'results_frequency': 0}}}])
def test_missing_or_bad_settings_expect_nothing(cfg):
    assert expected_tier_rows(cfg, 'afm') == {}


def test_tier_frequencies_come_from_tier_args():
    assert tier_frequencies(CFG) == {'window': 1000, 'raw': 100, 'peratom': 1000}


@pytest.mark.parametrize('freq', [0, -1000])
def test_a_non_positive_results_frequency_has_no_tier_frequencies(freq):
    cfg = {'settings': {'output': {'results_frequency': freq}}}
    assert tier_frequencies(cfg) == {}


def test_per_layer_channels_render_the_pinned_statements():
    """The shared per-layer builders keep the exact LAMMPS strings decks depend on."""
    by_name = {ob.name: ob for ob in layer_channels(3)}
    comz = by_name['comz_L3']
    assert comz.lammps_compute == 'compute COM_L3 layer_3 com'
    assert comz.lammps_variable == 'variable comz_L3 equal c_COM_L3[3]'
    assert comz.rename_to == 'v_comz_L3' and comz.role_key == 'layer_com_z_L3'
    assert comz is not REGISTRY['comz_L2']
    assert REGISTRY['comz_L2'] == by_name['comz_L2']
    sxx, syy = by_name['sigma_xx_2d_L2'], by_name['sigma_yy_2d_L2']
    assert sxx.lammps_compute == ('compute satom_L2 layer_2 stress/atom NULL\n'
                                  'compute sxx_L2 layer_2 reduce sum c_satom_L2[1]')
    assert syy.lammps_compute == ('compute satom_L2 layer_2 stress/atom NULL\n'
                                  'compute syy_L2 layer_2 reduce sum c_satom_L2[2]')
    assert sxx.lammps_variable == 'variable sigma_xx_2d_L2 equal c_sxx_L2'
    assert syy.lammps_variable == 'variable sigma_yy_2d_L2 equal c_syy_L2'
    assert syy.role_key == 'membrane_stress_yy_L2' and sxx.opt_in and syy.opt_in
    assert REGISTRY['sigma_yy_2d_L1'] == by_name['sigma_yy_2d_L1']


def test_raw_afm_channels_keep_their_natives_and_order():
    raw = channels_for('afm', 'raw')
    assert [(o.name, o.rename_to, o.units, o.role_key) for o in raw] == [
        ('raw_lfx', 'v_fx_spr', 'nN', 'raw_lateral_force_x'),
        ('raw_lfy', 'v_fy_spr', 'nN', 'raw_lateral_force_y'),
        ('raw_nf', 'v_fz_tip', 'nN', 'raw_normal_load'),
        ('raw_tipx', 'v_comx_tip', 'A', 'raw_driven_com_x'),
        ('raw_tipy', 'v_comy_tip', 'A', 'raw_driven_com_y'),
        ('raw_tipz', 'v_comz_tip', 'A', 'raw_driven_com_z'),
    ]
