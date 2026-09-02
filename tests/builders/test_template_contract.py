"""Template/parser contract tests (spec §3.4, §4 — Task 4/5)."""
import re

import pytest

from src.builders.afm import AFMSimulation
from src.core.observables import expected_columns
from tests.builders.test_afm_builder import (
    _make_afm_config,
    _prepare_builder_for_slide_write,
)

FIX_RE = re.compile(
    r'fix\s+(fc_\w+)\s+all\s+ave/time\s+(\d+)\s+(\d+)\s+(\d+)\s+(.*?)\s+file\s+(\S+)'
)


@pytest.fixture()
def slide_text(tmp_path):
    config = _make_afm_config(tmp_path, layers=[1])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    slide_path = _prepare_builder_for_slide_write(builder, tmp_path)
    return slide_path.read_text(encoding='utf-8')


def test_three_tiers_present_with_correct_args(slide_text):
    fixes = {m.group(1): m for m in FIX_RE.finditer(slide_text)}
    assert set(fixes) == {'fc_win', 'fc_raw', 'fc_pa'}
    assert (fixes['fc_win'].group(2), fixes['fc_win'].group(3),
            fixes['fc_win'].group(4)) == ('1', '1000', '1000')
    assert (fixes['fc_raw'].group(2), fixes['fc_raw'].group(3),
            fixes['fc_raw'].group(4)) == ('100', '1', '100')
    assert (fixes['fc_pa'].group(2), fixes['fc_pa'].group(3),
            fixes['fc_pa'].group(4)) == ('1000', '1', '1000')


def test_window_varlist_starts_with_legacy_layout(slide_text):
    m = [m for m in FIX_RE.finditer(slide_text) if m.group(1) == 'fc_win'][0]
    vars_in_fix = m.group(5).split()
    assert vars_in_fix[:9] == [
        'v_fz_tip', 'v_fx_spr', 'v_fy_spr', 'v_comx', 'v_comy', 'v_comz',
        'v_comx_tip', 'v_comy_tip', 'v_comz_tip',
    ]


def test_window_columns_match_registry_monolayer(slide_text):
    # contract: rendered variable count == registry expectation for L1.
    # Static registry carries bilayer entries; the builder swaps in
    # layer_channels(1), so expected = window set - L2 triplet.
    m = [m for m in FIX_RE.finditer(slide_text) if m.group(1) == 'fc_win'][0]
    vars_in_fix = m.group(5).split()
    expected = [c for c in expected_columns('afm', 'window')
                if not c.endswith('_L2')]
    assert len(vars_in_fix) == len(expected)


def test_no_group_group_in_window_tier(slide_text):
    win_line = [l for l in slide_text.splitlines() if 'fc_win' in l][0]
    assert 'eint' not in win_line          # spec §4: E_int is peratom-tier
    pa_line = [l for l in slide_text.splitlines() if 'fc_pa' in l][0]
    assert 'v_e_int' in pa_line


def test_each_compute_emitted_exactly_once(slide_text):
    # The clear+jump loop re-executes the whole setup block from the top of
    # the script — textual re-emission after each clear would define computes
    # against wiped groups. Correct structure: each compute appears once.
    assert slide_text.count('compute COM_L1') == 1
    assert slide_text.count('compute T_sheet') == 1


def test_every_var_in_fixes_is_defined(slide_text):
    # Every v_<name> referenced by an output fix must have a matching
    # `variable <name>` definition (or come from a fix reference f_*).
    defined = set(re.findall(r'^variable\s+(\w+)', slide_text, re.M))
    for m in FIX_RE.finditer(slide_text):
        for token in m.group(5).split():
            assert token.startswith('v_'), token
            assert token[2:] in defined, f"{token} referenced but never defined"


def test_contact_cutoff_is_numeric_in_compute(slide_text):
    m = re.search(r'coord/atom cutoff ([\d.]+) group tip_all', slide_text)
    assert m, "n_contact compute missing"
    assert 0.5 < float(m.group(1)) < 12.0


def test_langevin_tally_present_nonvacuous(slide_text):
    lang_lines = [l for l in slide_text.splitlines()
                  if 'langevin' in l and l.strip().startswith('fix')]
    assert lang_lines, 'no langevin fix rendered — vacuous-pass guard'
    for line in lang_lines:
        assert 'tally yes' in line, line


def test_sheet_temp_diagnostic_present(slide_text):
    assert 'compute T_sheet 2D_all temp' in slide_text


def _sheet_obs_ctx(has_internal_lj: bool, n_layers: int = 4,
                   use_langevin: bool = True, optional_channels=()) -> dict:
    """Call build_obs_ctx directly, with the same defaults _render_sheet_slide
    uses (all tiers on, the documented contact-cutoff fallback).

    interface_layers is always the (n_layers // 2, n_layers // 2 + 1) pair --
    the same rule sheetonsheet/slide.lmp's own {% set il_bot %}/{% set
    il_top %} use (Ruling 3) -- so n_contact_il's placeholders resolve
    correctly whenever a test opts it in.
    """
    from src.builders.utils.obs_ctx import build_obs_ctx
    interlayer = {'E_int_interlayer', 'n_contact_il'}
    return build_obs_ctx(
        sim_type='sheetonsheet',
        n_layers=n_layers,
        results_file_pattern='results/friction.txt',
        use_langevin=use_langevin,
        enabled_tiers={'window': True, 'raw': True, 'peratom': True},
        contact_cutoff=5.0,
        results_freq=1000,
        optional_channels=optional_channels,
        interface_layers=(n_layers // 2, n_layers // 2 + 1),
        extra_filter=(
            None if not has_internal_lj
            else lambda ob: ob.name not in interlayer),
    )


def _render_sheet_slide(has_internal_lj: bool, n_layers: int = 4,
                        constraint_mode: str = 'none') -> str:
    """Render sheetonsheet/slide.lmp directly for the conditional-channel test.

    Only the has_internal_lj branch is under test; the non-strict env renders
    missing context keys empty, which is fine here.
    """
    from jinja2 import Environment
    from src.interfaces.jinja import PackageLoader
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    return env.get_template('sheetonsheet/slide.lmp').render(
        n_layers=n_layers, has_internal_lj=has_internal_lj,
        constraint_mode=constraint_mode, pot_type='sw' if not has_internal_lj else 'rebomos',
        scan_speed_config=1.0, pressures=1.0, scan_angle_config=0.0,
        scan_angle_force=None, temp=300, thermostat_type='langevin',
        drive_method='fix_move', dump_enabled=False,
        seed_sheet_center=12345,
        # All tiers on: these tests are about the has_internal_lj branch,
        # not the channel gate. The tier gate now lives in obs_ctx (built by
        # build_obs_ctx), not a bare channels dict -- see
        # tests/test_output_channels_sheet.py for the gate itself.
        obs_ctx=_sheet_obs_ctx(has_internal_lj, n_layers),
        xlo=0.0, xhi=100.0, ylo=0.0, yhi=100.0, zhi=15.0,  # zhi feeds arithmetic
        num_atom_types=6, data_file='build/sheet.lmp',
        potential_file='system.in.settings', neighbor_list=2.0,
        neigh_modify_command='neigh_modify delay 0', atom_style='atomic',
        min_style='cg', minimization_command='minimize 1e-6 1e-8 1000 10000',
        timestep=0.001, thermo=1000, results_freq=1000, run_steps=1000,
        results_file_pattern='results/friction.txt',
        ev_a3_to_gpa=160.2176, ev_a_to_nn=1.602176565,
        driving_spring_ev=3.0,
    )


def test_conditional_interlayer_channel():
    # spec §10 conditional-channel row: E_int_interlayer present for an
    # is_sheet_lj potential, ABSENT (cleanly, not silently zero) otherwise.
    sw_text = _render_sheet_slide(has_internal_lj=False)
    assert 'variable        e_int_il equal c_frict' in sw_text
    rb_text = _render_sheet_slide(has_internal_lj=True)
    assert 'e_int_il' not in rb_text


def test_sheet_slide_takes_its_varlist_from_the_macro():
    src = _template_source('sheetonsheet/slide.lmp')
    assert 'emit_output_fixes(obs_ctx)' in src
    # hold_ave is exempt: it is NOT a results-tier file (no title1, and it
    # keeps its own hand-written varlist by design -- see
    # test_hold_fix_never_carries_schema_marker). Only fc_win/fc_raw/fc_pa
    # must have vanished from the source in favour of the macro call.
    results_fix_lines = [ln for ln in src.splitlines()
                         if ln.startswith('fix') and 'ave/time' in ln
                         and 'hold_ave' not in ln]
    assert not results_fix_lines, (
        f'no hand-written results-tier fix ave/time may remain: {results_fix_lines}')


def test_sheet_window_varlist_matches_the_registry():
    from src.core.observables import channels_for, layer_channels
    ctx = _sheet_obs_ctx(has_internal_lj=False, n_layers=4)
    emitted = ctx['window_varlist'].split()
    for ob in channels_for('sheetonsheet', 'window'):
        if ob.name.startswith('com') and '_L' in ob.name:
            continue          # the static bilayer pair is replaced per-run
        assert (ob.rename_to or f'v_{ob.name}') in emitted, ob.registry_key
    # layer_channels(4) also carries the opt-in peratom-tier per-layer
    # channels (Task 18) -- only the window-tier com{x,y,z}_L{i} entries
    # belong in the (always-on) window varlist.
    for ob in layer_channels(4):
        if ob.tier != 'window':
            continue
        assert ob.rename_to in emitted, ob.name


def test_many_body_sheet_drops_the_interlayer_channels():
    # the template's fallback branch never defines v_e_int_il, so emitting it
    # would reference an undefined variable
    ctx = _sheet_obs_ctx(has_internal_lj=True, n_layers=4)
    assert 'v_e_int_il' not in ctx['peratom_varlist']
    assert ctx['emit_peratom'] is False      # it was the only peratom channel


def test_sheet_results_fix_carries_the_family_marker():
    ctx = _sheet_obs_ctx(has_internal_lj=False, n_layers=4)
    assert ctx['title1'] == '# schema=2 lf_units=nN family=sheetonsheet'


def test_unloaded_reference_lines_in_system_init(tmp_path):
    from src.builders.afm import AFMSimulation
    config = _make_afm_config(tmp_path, layers=[1])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)
    text = (tmp_path / "out" / "L1" / "lammps" / "system.in").read_text()
    assert 'variable        zref_L1 equal xcm(layer_1,z)' in text
    assert 'UNLOADED_REF layer=1' in text


def test_settle_loop_is_capped(tmp_path):
    from src.builders.afm import AFMSimulation
    config = _make_afm_config(tmp_path, layers=[1])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)
    text = (tmp_path / "out" / "L1" / "lammps" / "system.in").read_text()
    # guards the silent-empty-render failure (env has no StrictUndefined)
    assert 'variable        ramp_iter loop 40' in text
    assert 'RAMP_STATUS' in text
    assert 'variable        ramp_iter delete' in text   # cap resets per load


def test_hold_channels_emitted_selfcontained(tmp_path):
    from src.builders.afm import AFMSimulation
    config = _make_afm_config(tmp_path, layers=[1])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)
    text = (tmp_path / "out" / "L1" / "lammps" / "system.in").read_text()
    assert 'fix             hold_ave' in text
    assert 'v_hold_fz v_hold_tipz v_hold_shz' in text   # NOT v_fz_tip - undefined here
    assert 'hold_$(v_find)N.txt' in text


def test_sheet_hold_logs_shearing_interface():
    text = _render_sheet_slide(has_internal_lj=False)   # n_layers=4
    # shearing interface for n=4 is L3/L2 (spec §6.1)
    assert 'variable        d_shear equal v_comz_L3-v_comz_L2' in text
    assert 'hold_p$(v_pressure)GPa.txt' in text
    assert text.index('fix             hold_ave') > text.index('run             10000')


@pytest.mark.parametrize('n_layers', [4, 6])
def test_atom_bonds_r0_top_names_the_bonded_pair(n_layers):
    """r0_top must read the pair the harmonic bonds actually connect
    (layer_{n-1}/layer_n, from `create_bonds many layer_{{ n_layers - 1 }}
    layer_{{ n_layers }}`), NOT the middle sliding interface (il_top/il_bot).

    The two pairs coincide only at n_layers == 4 (il_top == n_layers - 1
    there); n_layers == 6 is where a wrong repoint would show up as a
    silently-different r0_top.
    """
    text = _render_sheet_slide(has_internal_lj=False, n_layers=n_layers,
                               constraint_mode='atom_bonds')
    assert (f'variable        r0_top equal '
           f'v_comz_top-v_comz_L{n_layers - 1}') in text
    assert 'v_comz_ctop' not in text


def test_sheet_three_tiers():
    # Column order is registry-driven (Task 14); the exact varlist is
    # checked by test_sheet_window_varlist_matches_the_registry instead.
    text = _render_sheet_slide(has_internal_lj=False)
    assert 'fc_win' in text and 'fc_raw' in text and 'fc_pa' in text


def test_sheet_peratom_absent_for_internal_lj():
    text = _render_sheet_slide(has_internal_lj=True)
    # an empty-varlist fix would be invalid LAMMPS: fc_pa must vanish entirely
    assert 'fc_pa' not in text
    assert 'fc_win' in text and 'fc_raw' in text


# --- seeded lines must carry a real integer ----------------------------------
# The templates used to fall back to a Jinja-side `range(1, 999999) | random`
# when the builder supplied no seed. That fallback is gone (a template-drawn
# seed is invisible to provenance.json), which means a builder that forgets to
# supply one would now render an EMPTY seed field -- valid Jinja, invalid
# LAMMPS. These tests are the tripwire.

_SEEDED_RE = re.compile(
    r'^\s*(?:fix\s+\S+\s+\S+\s+langevin\s+\S+\s+\S+\s+\S+\s+(?P<lang>\S*)'
    r'|velocity\s+\S+\s+create\s+\S+\s+(?P<vel>\S*))',
    re.M)


def _assert_seeds_are_integers(text, where):
    hits = list(_SEEDED_RE.finditer(text))
    assert hits, f'{where}: no seeded lines found -- regex drifted?'
    for m in hits:
        seed = m.group('lang') or m.group('vel')
        assert seed.isdigit(), f'{where}: non-integer seed {seed!r} in {m.group(0)!r}'


def test_afm_scripts_have_integer_seeds(tmp_path):
    config = _make_afm_config(tmp_path, layers=[1])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    _prepare_builder_for_slide_write(builder, tmp_path)
    lammps_dir = tmp_path / "out" / "L1" / "lammps"
    for script in sorted(lammps_dir.glob("*.in")):
        _assert_seeds_are_integers(script.read_text(encoding='utf-8'), script.name)


def test_sheet_slide_has_integer_seeds():
    _assert_seeds_are_integers(_render_sheet_slide(has_internal_lj=False),
                               'sheetonsheet/slide.lmp')


def test_seed_channels_are_distinct_and_recorded(tmp_path):
    # tip and substrate thermostats must not share an RNG stream, and both
    # must reach provenance
    config = _make_afm_config(tmp_path, layers=[1])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    seeds = {c: builder._run_seed(c)
             for c in ('velocity', 'thermal_tip', 'thermal_sub')}
    assert len(set(seeds.values())) == 3, seeds
    # stable within one build (system.in and slide.in must agree)
    assert builder._run_seed('thermal_tip') == seeds['thermal_tip']


def test_sheet_lj_branch_lateral_force_is_scaled_to_nN():
    # compute group/group's VECTOR is a force in eV/A; nf is nN. Dividing one
    # by the other gave a COF 1.602176565x too SMALL for the whole SW set.
    text = _render_sheet_slide(has_internal_lj=False)
    assert 'variable        xfrict equal c_frict[1]*1.602176565' in text
    assert 'variable        yfrict equal c_frict[2]*1.602176565' in text


def test_sheet_many_body_branch_stays_unscaled_at_source():
    # v_sx is ALREADY nN (f_fsbot[1]*ev_a_to_nn); scaling it again would be a
    # second bug in the opposite direction
    text = _render_sheet_slide(has_internal_lj=True)
    assert 'variable        xfrict equal v_sx\n' in text
    assert 'variable        yfrict equal v_sy\n' in text


def test_interlayer_energy_is_not_scaled():
    # c_frict (SCALAR) is an energy in eV and is already correct
    text = _render_sheet_slide(has_internal_lj=False)
    assert 'variable        e_int_il equal c_frict\n' in text


def test_sheet_fixes_carry_the_schema_marker():
    # line 1 distinguishes post-fix (nN) output from pre-fix (eV/A) output.
    # LAMMPS title1 replaces ONLY line 1; the column header stays on line 2
    # where read_data reads it.
    text = _render_sheet_slide(has_internal_lj=False)
    assert text.count(
        'title1 "# schema=2 lf_units=nN family=sheetonsheet"') == 3  # win, raw, pa


def test_hold_fix_has_no_schema_marker():
    # the hold writes its own hold_*.txt and is not a results-tier file
    text = _render_sheet_slide(has_internal_lj=False)
    hold = [ln for ln in text.splitlines() if 'hold_ave' in ln and 'ave/time' in ln]
    assert hold and 'title1' not in hold[0]


def _hetero_obs_ctx(n_layers: int) -> dict:
    """Call build_obs_ctx directly, mirroring
    HeteroSheetOnSheetSimulation._build_obs_ctx (src/builders/sheetonsheet/hetero.py).

    Unlike `_sheet_obs_ctx`, this takes no `has_internal_lj` parameter: the
    hetero builder hardcodes `has_internal_lj: True` in its render context
    (the "hybrid sw -> reaction-force friction proxy" comment on
    `write_run_scripts`), so `extra_filter` always drops the interlayer
    channels here too. That leaves the peratom tier empty -- E_int_interlayer
    is hetero's only peratom channel -- and build_obs_ctx's emptiness gate
    correctly suppresses fc_pa. That is intentional (task-15 Ruling C): hetero
    gains the raw tier here, not a populated peratom tier.
    """
    from src.builders.utils.obs_ctx import build_obs_ctx
    interlayer = {'E_int_interlayer', 'n_contact_il'}
    return build_obs_ctx(
        sim_type='hetero',
        n_layers=n_layers,
        results_file_pattern='results/friction.txt',
        use_langevin=True,
        enabled_tiers={'window': True, 'raw': True, 'peratom': True},
        contact_cutoff=5.0,
        results_freq=1000,
        optional_channels=(),
        extra_filter=lambda ob: ob.name not in interlayer,
    )


def _render_hetero_slide(n_layers: int) -> str:
    """Render hetero/slide.lmp, whose layer groups come from z-bands.

    has_internal_lj=False exercises the template's non-many-body `compute
    frict` branch (needed by the COM-probe/frict-interface tests below,
    which pin the interface `compute frict` names). `obs_ctx` still comes
    from `_hetero_obs_ctx`, which always filters the interlayer channels the
    same way the real builder does (Ruling C) -- so the peratom tier stays
    empty here regardless of which template branch is under test, sidestepping
    the fact that hetero's has_internal_lj=False branch (pre-existing,
    untouched by Task 15, never exercised by the real builder) does not
    itself define `v_e_int_il` the way sheetonsheet's does.
    """
    from jinja2 import Environment
    from src.interfaces.jinja import PackageLoader
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    return env.get_template('hetero/slide.lmp').render(
        n_layers=n_layers, has_internal_lj=False,
        layers=[{'idx': i, 'zlo': 3.0 * (i - 1), 'zhi': 3.0 * i}
                for i in range(1, n_layers + 1)],
        constraint_mode='none', pot_type='sw',
        scan_speed_config=1.0, pressures=1.0, scan_angle_config=0.0,
        scan_angle_force=None, temp=300, thermostat_type='langevin',
        drive_method='fix_move', dump_enabled=False, seed_sheet_center=12345,
        obs_ctx=_hetero_obs_ctx(n_layers),
        xlo=0.0, xhi=100.0, ylo=0.0, yhi=100.0, zhi=15.0,
        num_atom_types=6, data_file='build/hetero.lmp',
        potential_file='system.in.settings', neighbor_list=2.0,
        neigh_modify_command='neigh_modify delay 0', atom_style='atomic',
        min_style='cg', minimization_command='minimize 1e-6 1e-8 1000 10000',
        timestep=0.001, thermo=1000, results_freq=1000, run_steps=1000,
        results_file_pattern='results/friction.txt',
        ev_a3_to_gpa=160.2176, ev_a_to_nn=1.602176565, driving_spring_ev=3.0,
    )


def test_hetero_langevin_fix_tallies():
    """w_thermostat_center reads f_lang, which only exists with `tally yes`.
    hetero/slide.lmp had `zero yes` and no tally, so the channel the spec
    called free is a LAMMPS error here without this one word."""
    src = _template_source('hetero/slide.lmp')
    line = [l for l in src.split('\n')
           if 'fix' in l and 'lang' in l and 'langevin' in l][0]
    assert 'tally yes' in line


def test_hetero_slide_takes_its_varlist_from_the_macro():
    src = _template_source('hetero/slide.lmp')
    assert 'emit_output_fixes(obs_ctx)' in src
    # Unlike sheetonsheet/slide.lmp, hetero has no hold_ave fix to exempt --
    # every hand-written results-tier fix ave/time is gone.
    assert 'ave/time' not in src


def test_hetero_gains_the_layer_coms_and_energies():
    ctx = _hetero_obs_ctx(n_layers=4)
    emitted = set(ctx['window_varlist'].split())
    assert {'v_comz_L1', 'v_comz_L4', 'v_etot', 'v_pe_now'} <= emitted


def test_hetero_results_fix_carries_its_own_family_marker():
    assert _hetero_obs_ctx(n_layers=4)['title1'].endswith('family=hetero')


def _com_probe_groups(text: str):
    """(upper, lower) groups straddling the sliding interface.

    Read from `compute frict`, which is now the single definition of the
    interface -- COM_ctop/COM_cbot were deleted as exact aliases of
    com{x,y,z}_L{il_top}/L{il_bot} (spec 4.2).
    """
    return _frict_groups(text)


def _frict_groups(text: str):
    """(upper group, lower group) of `compute frict ... group/group ...`."""
    m = re.search(r'compute\s+frict\s+(\S+)\s+group/group\s+(\S+)', text)
    return m.group(1), m.group(2)


# (n_layers, expected cbot, expected ctop): the stack splits into two sheets of
# equal thickness, thinner half below for odd n. COM_cbot used to be pinned to
# layer_2, which only lands on the middle interface at n_layers == 4 -- at the
# 3-layer minimum it aliased COM_ctop, so the extracted interlayer gap
# (comz_ctop - comz_cbot, the only z pair that reached the results file) was
# identically zero.
MIDDLE_INTERFACE = [(3, 1, 2), (4, 2, 3), (5, 2, 3), (6, 3, 4), (7, 3, 4)]


@pytest.mark.parametrize('n_layers,cbot,ctop', MIDDLE_INTERFACE)
def test_sheet_com_probes_straddle_the_middle_interface(n_layers, cbot, ctop):
    got = _com_probe_groups(
        _render_sheet_slide(has_internal_lj=False, n_layers=n_layers))
    assert got == (f'layer_{ctop}', f'layer_{cbot}')


@pytest.mark.parametrize('n_layers,cbot,ctop',
                         [r for r in MIDDLE_INTERFACE if r[0] >= 4])
def test_hetero_com_probes_straddle_the_middle_interface(n_layers, cbot, ctop):
    """Same rule for hetero/slide.lmp (MIN_LAYERS is 4 there)."""
    got = _com_probe_groups(_render_hetero_slide(n_layers))
    assert got == (f'layer_{ctop}', f'layer_{cbot}')


def test_sheet_results_file_carries_all_four_com_z_probes():
    """comz_top and comz_bot were computed and never emitted. After the
    alias removal the same four numbers arrive as drivez plus the layer
    COMs at il_top, il_bot and layer 1."""
    ctx = _sheet_obs_ctx(has_internal_lj=False, n_layers=4)
    emitted = set(ctx['window_varlist'].split())
    assert {'v_comz_top', 'v_comz_L3', 'v_comz_L2', 'v_comz_L1'} <= emitted


def test_hetero_results_file_carries_all_four_com_z_probes():
    """comz_top and comz_bot were computed and never emitted (and hetero never
    emitted comz_ctop/comz_cbot into the results file either, despite
    computing them). After the alias removal the same four numbers arrive as
    drivez plus the layer COMs at il_top, il_bot and layer 1 (n_layers=4:
    il_top=3, il_bot=2) -- same shape as
    test_sheet_results_file_carries_all_four_com_z_probes."""
    ctx = _hetero_obs_ctx(n_layers=4)
    emitted = set(ctx['window_varlist'].split())
    assert {'v_comz_top', 'v_comz_L3', 'v_comz_L2', 'v_comz_L1'} <= emitted


# --- the test that would actually have caught the eV/A-vs-nN defect ---------
# A registry-level units assertion alone would not have: REGISTRY['nf@sheet']
# and REGISTRY['lfx@sheet'] now exist and agree on 'nN' (see
# test_sheet_cof_inputs_share_force_units), but the original defect lived in
# the TEMPLATE emitting an unscaled eV/A value, not in a registry mismatch.
# So the template is where it has to be pinned -- for every sheet template,
# not just the one that was looked at.
SHEET_SLIDE_TEMPLATES = ('sheetonsheet/slide.lmp',
                         'hetero/slide.lmp',
                         'flake/slide.lmp')


def _template_source(name: str) -> str:
    from jinja2 import Environment
    from src.interfaces.jinja import PackageLoader
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    return env.loader.get_source(env, name)[0]


@pytest.mark.parametrize('template', SHEET_SLIDE_TEMPLATES)
def test_lj_branch_lateral_force_scaled_to_nN(template):
    # compute group/group's VECTOR is a force in eV/A. nf is nN. Unscaled,
    # COF came out a factor 1.602176565 too SMALL for the entire SW set.
    src = _template_source(template)
    assert 'c_frict[1]*{{ ev_a_to_nn }}' in src, template
    assert 'c_frict[2]*{{ ev_a_to_nn }}' in src, template
    # and never the bare form
    assert 'equal c_frict[1]\n' not in src, template
    assert 'equal c_frict[2]\n' not in src, template


@pytest.mark.parametrize('template', SHEET_SLIDE_TEMPLATES)
def test_interlayer_energy_stays_unscaled(template):
    # c_frict's SCALAR is an energy in eV and was always correct; scaling it
    # would be a new bug in the opposite direction.
    src = _template_source(template)
    if 'e_int_il' in src:
        assert 'variable        e_int_il equal c_frict\n' in src, template


#: All three sheet-family templates are now routed through the observables
#: registry (Task 14 sheetonsheet, Task 15 hetero, Task 16 flake): none of
#: them has a hand-written `fix ... ave/time` line left to grep for -- only
#: the macro call -- so the marker must be checked on the RENDERED output
#: instead. This used to be split into a CONVERTED/UNCONVERTED pair of tuples
#: (one checked by rendering, the other by scanning source) so that moving a
#: template between them was a one-line diff; now that every member has
#: moved, the UNCONVERTED list and its source-scanning test would be
#: permanently empty and would never run -- dead code asserting nothing, not
#: a real regression guard -- so both are deleted rather than kept as inert
#: scaffolding. `SHEET_SLIDE_TEMPLATES_CONVERTED` is kept as an alias of
#: `SHEET_SLIDE_TEMPLATES` (defined above) so the render-based test below
#: reads the same regardless of which name the templates are declared under.
SHEET_SLIDE_TEMPLATES_CONVERTED = SHEET_SLIDE_TEMPLATES

#: One render call per template, keyed by name -- each family's obs_ctx is
#: built differently (has_internal_lj is a per-family fixed value for hetero
#: and flake, a parameter for sheetonsheet), so a single shared render call
#: does not cover every member.
_CONVERTED_RENDER = {
    'sheetonsheet/slide.lmp': lambda: _render_sheet_slide(has_internal_lj=False),
    'hetero/slide.lmp': lambda: _render_hetero_slide(4),
    'flake/slide.lmp': lambda: _render_flake_slide(4),
}


@pytest.mark.parametrize('template', SHEET_SLIDE_TEMPLATES_CONVERTED)
def test_results_fixes_carry_schema_marker_converted(template):
    family = template.split('/')[0]
    text = _CONVERTED_RENDER[template]()
    results_fixes = [ln for ln in text.splitlines()
                     if ln.startswith('fix') and 'ave/time' in ln
                     and 'hold_ave' not in ln]
    assert results_fixes, template
    for ln in results_fixes:
        assert (f'title1 "# schema=2 lf_units=nN family={family}"' in ln
               ), f"{template}: {ln[:60]}"


@pytest.mark.parametrize('template', SHEET_SLIDE_TEMPLATES)
def test_hold_fix_never_carries_schema_marker(template):
    # the hold writes its own hold_*.txt and is not read by read_data
    src = _template_source(template)
    for ln in src.splitlines():
        if 'hold_ave' in ln and 'ave/time' in ln:
            assert 'title1' not in ln, template


@pytest.mark.parametrize('n_layers,cbot,ctop', MIDDLE_INTERFACE)
def test_sheet_frict_measures_the_same_interface_the_com_probes_do(
        n_layers, cbot, ctop):
    """One interface definition, used by every observable that names it.

    `compute frict` and v_d_shear used layer_{n-1}/layer_{n-2} while the COM
    probes used the middle pair; they agree only at n_layers 3 and 4, so a
    5-layer run reported an interlayer gap at one interface and a lateral
    force at another.
    """
    text = _render_sheet_slide(has_internal_lj=False, n_layers=n_layers)
    assert _frict_groups(text) == (f'layer_{ctop}', f'layer_{cbot}')
    assert _com_probe_groups(text) == _frict_groups(text)
    assert (f'variable        d_shear equal '
            f'v_comz_L{ctop}-v_comz_L{cbot}') in text


@pytest.mark.parametrize('n_layers,cbot,ctop',
                         [r for r in MIDDLE_INTERFACE if r[0] >= 4])
def test_hetero_frict_measures_the_same_interface_the_com_probes_do(
        n_layers, cbot, ctop):
    text = _render_hetero_slide(n_layers)
    assert _frict_groups(text) == (f'layer_{ctop}', f'layer_{cbot}')
    assert _com_probe_groups(text) == _frict_groups(text)


def _flake_obs_ctx(n_layers: int) -> dict:
    """Call build_obs_ctx directly, mirroring
    FlakeSlideSimulation._build_obs_ctx (src/builders/sheetonsheet/flake_slide.py).

    Unlike `_hetero_obs_ctx`, no `extra_filter` is needed: the flake family
    declares no interlayer/contact channel at all (its friction proxy is the
    bottom-sheet reaction force `react_{x,y,z}`), so `channels_for('flake',
    'peratom')` is already empty -- there is nothing for a filter to drop.
    """
    from src.builders.utils.obs_ctx import build_obs_ctx
    return build_obs_ctx(
        sim_type='flake',
        n_layers=n_layers,
        results_file_pattern='results/friction.txt',
        use_langevin=True,
        enabled_tiers={'window': True, 'raw': True, 'peratom': True},
        contact_cutoff=5.0,
        results_freq=1000,
        optional_channels=(),
    )


def _render_flake_slide(n_layers: int) -> str:
    """Render flake/slide.lmp, whose substrate layer groups come from z-bands
    (like hetero) and whose driven body is the flake, not a sheet layer_N."""
    from jinja2 import Environment
    from src.interfaces.jinja import PackageLoader
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    return env.get_template('flake/slide.lmp').render(
        n_layers=n_layers, has_internal_lj=True,
        sheet_layers=[{'idx': i, 'zlo': 3.0 * (i - 1), 'zhi': 3.0 * i}
                     for i in range(1, n_layers + 1)],
        constraint_mode='none', pot_type='sw',
        scan_speed_config=1.0, pressures=1.0, scan_angle_config=0.0,
        scan_angle_force=None, temp=300, thermostat_type='langevin',
        drive_method='fix_move', dump_enabled=False, seed_sheet_center=12345,
        obs_ctx=_flake_obs_ctx(n_layers),
        xlo=0.0, xhi=100.0, ylo=0.0, yhi=100.0, zhi=15.0,
        num_atom_types=6, data_file='build/flake_slide.data',
        potential_file='system.in.settings', neighbor_list=2.0,
        neigh_modify_command='neigh_modify delay 0', atom_style='atomic',
        min_style='cg', minimization_command='minimize 1e-6 1e-8 1000 10000',
        timestep=0.001, thermo=1000, results_freq=1000, run_steps=1000,
        results_file_pattern='results/friction.txt',
        ev_a3_to_gpa=160.2176, ev_a_to_nn=1.602176565, driving_spring_ev=3.0,
        bond_spring_ev=3.0, flake_group_types='5 6', flake_corner_types='6',
        flake_corner_k=10.0, corner_hold=True, virtual_offset=3.16,
        reserve_virtual_type=False,
    )


def test_flake_langevin_fix_tallies():
    """w_thermostat_center would read f_lang, which only exists with `tally
    yes`. flake/slide.lmp had `zero yes` and no tally, same gap Task 15 fixed
    for hetero."""
    src = _template_source('flake/slide.lmp')
    line = [l for l in src.split('\n')
           if 'fix' in l and 'lang' in l and 'langevin' in l][0]
    assert 'tally yes' in line


def test_flake_slide_takes_its_varlist_from_the_macro():
    src = _template_source('flake/slide.lmp')
    assert 'emit_output_fixes(obs_ctx)' in src
    # flake/slide.lmp has no hold_ave fix to exempt -- every hand-written
    # results-tier fix ave/time is gone.
    assert 'ave/time' not in src


def test_flake_drive_com_maps_to_the_flake_body():
    from src.core.observables import rename_map
    m = rename_map('flake', 'window')
    assert m['v_comx_flake'] == 'drivex'
    assert m['v_comz_flake'] == 'drivez'
    # the sheet's own drive channels must not leak into the flake map: the
    # flake model has one mobile body, not a driven top layer
    assert 'v_comx_top' not in m


def test_flake_gains_no_energy_or_normal_load_channels():
    """flake/slide.lmp defines no v_etot/v_pe_now/v_ke_now, no v_fx/v_fy/v_fz
    (no fix aveforce on a driven layer -- the flake is driven by a spring),
    and no interlayer/contact channel. Emitting any of these would reference
    an undefined LAMMPS variable at setup."""
    ctx = _flake_obs_ctx(n_layers=4)
    emitted = set(ctx['window_varlist'].split())
    forbidden = {'v_etot', 'v_pe_now', 'v_ke_now', 'v_fx', 'v_fy', 'v_fz'}
    assert not (forbidden & emitted)
    assert ctx['raw_varlist'] == ''
    assert ctx['peratom_varlist'] == ''
    assert ctx['emit_raw'] is False
    assert ctx['emit_peratom'] is False


def test_flake_results_fix_carries_its_own_family_marker():
    assert _flake_obs_ctx(n_layers=4)['title1'].endswith('family=flake')


def test_afm_window_fix_honours_results_frequency():
    """Regression: the macro hardcoded `1 1000 1000`, so
    settings.output.results_frequency was computed and discarded for every
    AFM slide build (afm/base.py passes it; the macro ignored it)."""
    from jinja2 import Environment
    from src.interfaces.jinja import PackageLoader
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    src = env.loader.get_source(env, 'common/observables.lmp')[0]
    assert 'ave/time 1 1000 1000' not in src
    assert 'obs_ctx.window_args' in src


def test_macro_emits_the_schema_marker_on_every_results_fix():
    """Interlock: a results file with no marker parses as schema 1, which
    triggers the frozen legacy correction and multiplies correctly-scaled
    nN data by 1.602176565. The macro had no title1 at all."""
    from jinja2 import Environment
    from src.interfaces.jinja import PackageLoader
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    src = env.loader.get_source(env, 'common/observables.lmp')[0]
    fixes = [l for l in src.split('\n')
             if l.startswith('fix') and 'ave/time' in l]
    assert len(fixes) == 3, fixes
    for line in fixes:
        assert 'title1 "{{ obs_ctx.title1 }}"' in line, line


# --- Task 17: thermostat work and group temperatures ------------------------

def test_thermostat_work_present_under_langevin_absent_otherwise():
    assert 'v_w_lang_center' in _sheet_obs_ctx(
        has_internal_lj=False, n_layers=4, use_langevin=True)['window_varlist']
    assert 'v_w_lang_center' not in _sheet_obs_ctx(
        has_internal_lj=False, n_layers=4, use_langevin=False)['window_varlist']


def test_f_lang_never_referenced_in_the_nvt_branch():
    # the nose-hoover branch emits `fix nvt_center` with no tally, so f_lang
    # would not resolve and LAMMPS would error
    ctx = _sheet_obs_ctx(has_internal_lj=False, n_layers=4, use_langevin=False)
    assert 'f_lang' not in ' '.join(ctx['variable_lines'])


#: temp_Ltop_3d's compute carries a `{n_layers}` placeholder (like n_contact's
#: `{contact_cutoff}`) because `layer_n` is a registry LABEL for "the top
#: layer", not a real LAMMPS group -- the real group is layer_4, layer_6,
#: etc. Parametrized over two layer counts so layer_4 vs layer_6 is visible,
#: not just "some digit got substituted".
@pytest.mark.parametrize('n_layers', [4, 6])
def test_temp_ltop_names_the_real_top_layer_sheetonsheet(n_layers):
    text = _render_sheet_slide(has_internal_lj=False, n_layers=n_layers)
    assert f'compute T_Ltop layer_{n_layers} temp' in text
    assert 'layer_n' not in text


@pytest.mark.parametrize('n_layers', [4, 6])
def test_temp_ltop_names_the_real_top_layer_hetero(n_layers):
    text = _render_hetero_slide(n_layers)
    assert f'compute T_Ltop layer_{n_layers} temp' in text
    assert 'layer_n' not in text


@pytest.mark.parametrize('n_layers', [4, 6])
def test_temp_ltop_names_the_real_top_layer_flake(n_layers):
    text = _render_flake_slide(n_layers)
    assert f'compute T_Ltop layer_{n_layers} temp' in text
    assert 'layer_n' not in text


def test_sheet_group_temperatures_present_regardless_of_thermostat():
    # compute ... temp works under either langevin or nose-hoover; only
    # w_thermostat_center is conditional on use_langevin
    for use_langevin in (True, False):
        ctx = _sheet_obs_ctx(has_internal_lj=False, n_layers=4,
                             use_langevin=use_langevin)
        emitted = ctx['window_varlist'].split()
        assert 'v_temp_center_3d' in emitted
        assert 'v_temp_L1_3d' in emitted
        assert 'v_temp_Ltop_3d' in emitted


# --- Task 18: opt-in per-atom and post channels ------------------------------

def test_opt_in_sheet_channels_are_off_by_default():
    ctx = _sheet_obs_ctx(has_internal_lj=False, n_layers=4)
    assert 'v_n_contact_il' not in ctx['peratom_varlist']


def test_naming_an_opt_in_channel_emits_it():
    ctx = _sheet_obs_ctx(has_internal_lj=False, n_layers=4,
                         optional_channels=['n_contact_il'])
    assert 'v_n_contact_il' in ctx['peratom_varlist']


def test_n_contact_il_names_the_real_sliding_interface():
    # 4f0ed92 moved the interface to the middle of the stack; layer_{n-1}
    # against layer_{n-2} coincides with it only at n_layers == 4
    from src.core.observables import REGISTRY
    compute = REGISTRY['n_contact_il'].lammps_compute
    assert '{il_top}' in compute and '{il_bot}' in compute


def test_opt_in_channels_default_build_leaves_render_unchanged():
    """A default build (no optional_channels) must be a no-op: the rendered
    script is byte-identical to what it emitted before Task 18."""
    before = _render_sheet_slide(has_internal_lj=False, n_layers=4)
    ctx = _sheet_obs_ctx(has_internal_lj=False, n_layers=4)
    assert 'v_n_contact_il' not in ctx['window_varlist']
    assert 'v_n_contact_il' not in ctx['peratom_varlist']
    assert 'v_pe_sum_L' not in ctx['window_varlist']
    assert 'v_pe_sum_L' not in ctx['peratom_varlist']
    assert 'v_sigma_xx_2d_L' not in ctx['peratom_varlist']
    assert 'v_sigma_yy_2d_L' not in ctx['peratom_varlist']
    # rendering twice with identical inputs must be deterministic
    after = _render_sheet_slide(has_internal_lj=False, n_layers=4)
    assert before == after


@pytest.mark.parametrize('n_layers,lyr', [(4, 1), (4, 2), (4, 3), (4, 4), (6, 5)])
def test_pe_sum_and_sigma_layer_channels_named_in_turn(n_layers, lyr):
    """Each per-layer opt-in channel is independently requestable -- neither
    sigma_xx_2d_L{i} nor sigma_yy_2d_L{i} may depend on the other having
    been requested, unlike AFM's always-both-present sigma_xx_2d/sigma_yy_2d
    pair. lyr=3,4,5 exercise layer_channels(n_layers)'s real-count generation
    beyond the static REGISTRY L1/L2 pair."""
    for name in (f'pe_sum_L{lyr}', f'sigma_xx_2d_L{lyr}',
                f'sigma_yy_2d_L{lyr}'):
        ctx = _sheet_obs_ctx(has_internal_lj=False, n_layers=n_layers,
                             optional_channels=[name])
        assert f'v_{name}' in ctx['peratom_varlist'], name


def test_sigma_xx_and_sigma_yy_layer_share_the_stress_atom_compute_once():
    # both requested together must not redefine `compute satom_L{i}`
    ctx = _sheet_obs_ctx(has_internal_lj=False, n_layers=4,
                         optional_channels=['sigma_xx_2d_L2', 'sigma_yy_2d_L2'])
    satom_defs = [l for l in ctx['compute_lines']
                 if l.startswith('compute satom_L2 ')]
    assert len(satom_defs) == 1, ctx['compute_lines']


@pytest.mark.parametrize('n_layers,cbot,ctop', MIDDLE_INTERFACE)
def test_interface_layers_matches_the_template_rule(n_layers, cbot, ctop):
    """Python-side interface_layers=(n_layers // 2, n_layers // 2 + 1)
    (build_obs_ctx, used by SheetOnSheetSimulation._build_obs_ctx for
    n_contact_il) must agree with the Jinja-side il_bot/il_top the template
    computes for itself. Two definitions of one interface that can drift is
    exactly the defect this branch exists to remove."""
    text = _render_sheet_slide(has_internal_lj=False, n_layers=n_layers)
    top, bot = _frict_groups(text)
    rendered_pair = (int(bot.split('_')[1]), int(top.split('_')[1]))
    assert rendered_pair == (n_layers // 2, n_layers // 2 + 1)
    assert rendered_pair == (cbot, ctop)


# --- Fix round 1: hetero/flake defensive interface_layers wiring ------------
#
# Neither hetero nor flake can reach n_contact_il through their real,
# unmodified code paths: hetero.py's extra_filter drops it UNCONDITIONALLY
# (not gated on has_internal_lj, unlike sheetonsheet's), and flake's
# sim_types never include 'flake' at all (see n_contact_il's REGISTRY
# declaration). So no render of the real, untouched builder classes can be
# made to leak an unsubstituted {il_bot}/{il_top} today -- that route is
# foreclosed by a different mechanism than the placeholder-substitution gap
# this fix closes. The two tests below pin the fix a different way: a
# source check (fails against the pre-fix source, where interface_layers
# was passed nowhere in either file, and passes once every build_obs_ctx
# call site passes it), and a direct build_obs_ctx behavioural check proving
# the substitution itself is correct for hetero/flake's exact n_layers // 2
# formula -- so if a future change ever lifts either foreclosure (e.g. a
# later task makes hetero's extra_filter conditional on has_internal_lj, the
# way sheetonsheet's already is), the placeholder plumbing this fix added is
# already correct rather than something that finds out the hard way at
# LAMMPS parse time.

def test_hetero_and_flake_pass_interface_layers_to_build_obs_ctx():
    """Every build_obs_ctx call site in hetero.py and flake_slide.py (the
    module-level _with_obs_ctx helper and the builder class's own
    _build_obs_ctx) must pass interface_layers=(n_layers // 2, n_layers //
    2 + 1) -- the same formula sheetonsheet/slide.lmp's own {% set il_bot %}
    / {% set il_top %} and SheetOnSheetSimulation._build_obs_ctx use. This
    is a source check rather than a render check because, per the module
    docstring above, no render of either family's real, unmodified code path
    can be forced to expose an unsubstituted placeholder -- the exposure
    this fix forecloses only exists hypothetically, if the *other*, unrelated
    foreclosure (hetero's unconditional extra_filter / flake's sim_types)
    is ever lifted by later work.
    """
    import inspect
    from src.builders.sheetonsheet import hetero as hetero_mod
    from src.builders.sheetonsheet import flake_slide as flake_mod

    needle = 'interface_layers=(n_layers // 2, n_layers // 2 + 1)'
    for mod in (hetero_mod, flake_mod):
        src = inspect.getsource(mod)
        count = src.count(needle)
        assert count == 2, (
            f'{mod.__name__}: expected 2 build_obs_ctx call sites passing '
            f'interface_layers (the module-level _with_obs_ctx helper and '
            f'the builder class _build_obs_ctx method), found {count}')


def test_hetero_and_flake_interface_layers_substitution_is_correct():
    """Direct build_obs_ctx behavioural proof, bypassing the two independent
    foreclosures described above (temporarily lifted here) so the
    placeholder substitution hetero.py/flake_slide.py's fix wires up is
    actually exercised end to end: call build_obs_ctx with hetero's and
    flake's exact sim_type/n_layers/interface_layers/optional_channels
    arguments, with n_contact_il's sim_types monkeypatched to include the
    family under test and, for hetero, with no extra_filter (mirroring what
    a future has_internal_lj-conditional filter would allow through). No
    unsubstituted {il_bot}/{il_top}/{n_layers}/{contact_cutoff} placeholder
    may appear, and the real layer groups the n_layers // 2 formula names
    must be the ones substituted in.
    """
    import dataclasses
    from src.builders.utils.obs_ctx import build_obs_ctx
    from src.core.observables import REGISTRY

    original = REGISTRY['n_contact_il']
    try:
        for sim_type in ('hetero', 'flake'):
            REGISTRY['n_contact_il'] = dataclasses.replace(
                original, sim_types=(sim_type,))
            for n_layers in (4, 5, 6):
                il_bot, il_top = n_layers // 2, n_layers // 2 + 1
                ctx = build_obs_ctx(
                    sim_type=sim_type,
                    n_layers=n_layers,
                    results_file_pattern='results/friction.txt',
                    use_langevin=False,
                    enabled_tiers={'window': True, 'raw': True, 'peratom': True},
                    contact_cutoff=5.0,
                    results_freq=1000,
                    optional_channels=['n_contact_il'],
                    interface_layers=(il_bot, il_top),
                )
                joined = '\n'.join(ctx['compute_lines'])
                assert 'v_n_contact_il' in ctx['peratom_varlist'], (
                    sim_type, n_layers)
                for placeholder in ('{il_bot}', '{il_top}', '{n_layers}',
                                    '{contact_cutoff}'):
                    assert placeholder not in joined, (
                        sim_type, n_layers, placeholder)
                assert f'layer_{il_bot}' in joined, (sim_type, n_layers)
                assert f'layer_{il_top}' in joined, (sim_type, n_layers)
    finally:
        REGISTRY['n_contact_il'] = original
