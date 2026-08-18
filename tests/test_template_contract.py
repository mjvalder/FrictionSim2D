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
