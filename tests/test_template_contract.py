"""Template/parser contract tests (spec §3.4, §4 — Task 4/5)."""
import re

import pytest

from src.builders.afm import AFMSimulation
from src.core.observables import expected_columns
from tests.test_afm_builder import (
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


def _render_sheet_slide(has_internal_lj: bool) -> str:
    """Render sheetonsheet/slide.lmp directly for the conditional-channel test.

    Only the has_internal_lj branch is under test; the non-strict env renders
    missing context keys empty, which is fine here.
    """
    from jinja2 import Environment
    from src.interfaces.jinja import PackageLoader
    env = Environment(loader=PackageLoader('src.templates'),
                      trim_blocks=True, lstrip_blocks=True)
    return env.get_template('sheetonsheet/slide.lmp').render(
        n_layers=4, has_internal_lj=has_internal_lj,
        constraint_mode='none', pot_type='sw' if not has_internal_lj else 'rebomos',
        scan_speed_config=1.0, pressures=1.0, scan_angle_config=0.0,
        scan_angle_force=None, temp=300, thermostat_type='langevin',
        drive_method='fix_move', dump_enabled=False,
        seed_sheet_center=12345,
        # All tiers on: these tests are about the has_internal_lj branch,
        # not the channel gate. Passed explicitly because the template
        # requires it -- a missing gate must error, not silently emit
        # everything, which is how the sheet family went ungated for so long.
        channels={'window': True, 'raw': True, 'peratom': True},
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


def test_sheet_three_tiers():
    text = _render_sheet_slide(has_internal_lj=False)
    assert 'fc_win' in text and 'fc_raw' in text and 'fc_pa' in text
    # fc_win keeps the legacy 16-column prefix
    win = [l for l in text.splitlines()
           if 'fc_win' in l and l.strip().startswith('fix')][0]
    assert win.split('ave/time 1 1000')[1].split()[1:17] == [
        'v_xfrict', 'v_yfrict', 'v_sx', 'v_sy', 'v_sz', 'v_fx', 'v_fy',
        'v_fz', 'v_comx_top', 'v_comy_top', 'v_comx_ctop', 'v_comy_ctop',
        'v_comz_ctop', 'v_comx_cbot', 'v_comy_cbot', 'v_comz_cbot']


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
    assert text.count('title1 "# schema=2 lf_units=nN"') == 3  # win, raw, pa


def test_hold_fix_has_no_schema_marker():
    # the hold writes its own hold_*.txt and is not a results-tier file
    text = _render_sheet_slide(has_internal_lj=False)
    hold = [ln for ln in text.splitlines() if 'hold_ave' in ln and 'ave/time' in ln]
    assert hold and 'title1' not in hold[0]
