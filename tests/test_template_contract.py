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


def _render_sheet_slide(has_internal_lj: bool, n_layers: int = 4) -> str:
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


def _render_hetero_slide(n_layers: int) -> str:
    """Render hetero/slide.lmp, whose layer groups come from z-bands."""
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
        channels={'window': True, 'raw': True, 'peratom': True},
        xlo=0.0, xhi=100.0, ylo=0.0, yhi=100.0, zhi=15.0,
        num_atom_types=6, data_file='build/hetero.lmp',
        potential_file='system.in.settings', neighbor_list=2.0,
        neigh_modify_command='neigh_modify delay 0', atom_style='atomic',
        min_style='cg', minimization_command='minimize 1e-6 1e-8 1000 10000',
        timestep=0.001, thermo=1000, results_freq=1000, run_steps=1000,
        results_file_pattern='results/friction.txt',
        ev_a3_to_gpa=160.2176, ev_a_to_nn=1.602176565, driving_spring_ev=3.0,
    )


def _com_probe_groups(text: str):
    """(ctop group, cbot group)."""
    ctop = re.search(r'compute\s+COM_ctop\s+(\S+)\s+com', text).group(1)
    cbot = re.search(r'compute\s+COM_cbot\s+(\S+)\s+com', text).group(1)
    return ctop, cbot


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


ALL_FOUR_COM_Z = ['v_comz_bot', 'v_comz_cbot', 'v_comz_ctop', 'v_comz_top']


def test_sheet_results_file_carries_all_four_com_probes():
    """All four COM probes are computed; all four must also be written.

    comz_top and comz_bot were computed and never emitted, so downstream had
    only the ctop/cbot pair to work with.
    """
    text = _render_sheet_slide(has_internal_lj=False, n_layers=3)
    m = [m for m in FIX_RE.finditer(text) if m.group(1) == 'fc_win'][0]
    emitted = set(m.group(5).split())
    assert set(ALL_FOUR_COM_Z) <= emitted


def test_hetero_results_file_carries_all_four_com_probes():
    text = _render_hetero_slide(4)
    line = [l for l in text.split('\n') if l.startswith('fix') and ' fc_ave ' in l][0]
    assert set(ALL_FOUR_COM_Z) <= set(line.split())


def test_sheet_window_keeps_legacy_sixteen_variable_prefix():
    """The new columns must be appended, not inserted: the header-less
    positional fallback in read_data keys off this exact prefix."""
    from src.postprocessing.read_data import DataReader
    m = [m for m in FIX_RE.finditer(
        _render_sheet_slide(has_internal_lj=False, n_layers=3))
        if m.group(1) == 'fc_win'][0]
    legacy = DataReader._SHEET_FILE_COLUMNS[1:]  # drop 'time'
    assert m.group(5).split()[:len(legacy)] == legacy


# --- the test that would actually have caught the eV/A-vs-nN defect ---------
# A registry-level units assertion cannot: the sheet family has no `nf`
# channel in REGISTRY at all, and its lateral channel is `xfrict`, so a
# REGISTRY['lfx'].units == REGISTRY['nf'].units check only ever compares two
# AFM entries. The defect lived in the TEMPLATE, so the template is where it
# has to be pinned -- for every sheet template, not just the one that was
# looked at.
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


@pytest.mark.parametrize('template', SHEET_SLIDE_TEMPLATES)
def test_results_fixes_carry_schema_marker(template):
    src = _template_source(template)
    results_fixes = [ln for ln in src.splitlines()
                     if ln.startswith('fix') and 'ave/time' in ln
                     and 'hold_ave' not in ln]
    assert results_fixes, template
    for ln in results_fixes:
        assert 'title1 "# schema=2 lf_units=nN"' in ln, f"{template}: {ln[:60]}"


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
