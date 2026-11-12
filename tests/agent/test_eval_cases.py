"""The eval cases themselves, without a model: coverage, no vacuous passes, and
that a correct turn (built by calling the tools directly) does pass."""
from types import SimpleNamespace

import pytest

pytest.importorskip('pydantic_ai')

from pydantic_ai.models.function import FunctionModel  # noqa: E402

from src.agent.changes import build_changeset  # noqa: E402
from src.agent.events import TurnEvent  # noqa: E402
from src.agent.runtime import AgentSession  # noqa: E402
from src.agent.settings import AgentSettings  # noqa: E402
from src.agent.tools.knowledge import search_knowledge  # noqa: E402
from src.agent.tools.materials import lattice_mismatch  # noqa: E402
from src.agent.tools.materials_db import fetch_2dmatpedia_structure  # noqa: E402
from src.agent.tools.plots import PlotField, PlotProposalRequest, propose_plot  # noqa: E402
from src.agent.tools.setup import (AFMSetupRequest, PESSheetSetupRequest,  # noqa: E402
                                   SheetSetupRequest, make_sweep, setup_afm,
                                   setup_pes_sheet, setup_sheet)
from tests.agent.evals.cases import (ANGLES, CASES, DATASET, DECLINES, DM_ROWS,  # noqa: E402
                                     GROUPS, LAYERS, MP_ROWS, MUST_ASK, band, grounded,
                                     prepare_deps, verdict)
from tests.helpers.agent import make_deps  # noqa: E402
from tests.helpers.agent_eval import FUNCTION_REPLY, EvalResult  # noqa: E402


def _deps(name, tmp_path, mock_settings):
    case = CASES[name]
    deps = make_deps(tmp_path, mock_settings, model=case.model)
    prepare_deps(case, deps)
    return deps


def _result(deps, text='', tools=()):
    events = [TurnEvent('tool_call', f'{t}({{}})') for t in tools]
    return EvalResult(deps=deps, events=events + [TurnEvent('text', text)])


def test_enough_cases_in_every_group():
    assert len(CASES) >= 40
    assert all(GROUPS.values())
    assert len(CASES) == sum(len(g) for g in GROUPS.values())  # no name in two groups
    for name in ('ambiguous_unit', 'unknown_material', 'unknown_potential',
                 'ambiguous_family', 'missing_dataset', 'sheet_force_unit',
                 'sheet_low_pressure'):
        assert MUST_ASK[name].kind == 'ask'
    for name in ('executed_setting_walltime', 'executed_setting_minimize', 'negative_load'):
        assert DECLINES[name].kind == 'declines'


def test_every_registered_tool_is_exercised_by_a_case():
    session = AgentSession(AgentSettings(), model=FunctionModel(lambda m, i: None))
    covered = {tool for case in CASES.values() for tool in case.tools}
    assert set(session.tool_names()) <= covered
    assert covered <= set(session.tool_names())  # no typo in a case's tools


@pytest.mark.parametrize('name', sorted(CASES))
def test_no_case_passes_on_a_turn_that_only_asked_back(name, tmp_path, mock_settings):
    case = CASES[name]
    deps = _deps(name, tmp_path, mock_settings)
    if case.kind == 'ask':
        assert not verdict(case, _result(deps, 'Done.'))
        deps.pending.append(build_changeset(deps, 'x', []))
        assert not verdict(case, _result(deps, FUNCTION_REPLY))
    elif case.kind == 'declines':
        assert not verdict(case, _result(deps, 'Done.'))
    else:
        assert not verdict(case, _result(deps, FUNCTION_REPLY))
        assert not verdict(case, _result(deps, ''))


@pytest.mark.parametrize('name', sorted(MUST_ASK))
def test_must_ask_cases_pass_on_a_question(name, tmp_path, mock_settings):
    deps = _deps(name, tmp_path, mock_settings)
    assert verdict(CASES[name], _result(deps, FUNCTION_REPLY))


def test_setup_from_the_tool_passes(tmp_path, mock_settings):
    deps = _deps('afm_single', tmp_path, mock_settings)
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw',
                                    loads_nN=[10, 20, 30]))
    assert verdict(CASES['afm_single'], _result(deps, 'Proposed.'))


def test_sweep_values_as_the_tool_prints_them_pass(tmp_path, mock_settings):
    deps = _deps('log_load_sweep', tmp_path, mock_settings)
    printed = [float(f'{v:g}') for v in make_sweep(1, 100, 5, 'log')]  # 6 significant figures
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw', loads_nN=printed))
    assert verdict(CASES['log_load_sweep'], _result(deps, 'Proposed.'))


def test_low_loads_pass_as_a_question_quoting_the_band(tmp_path, mock_settings):
    deps = _deps('low_loads', tmp_path, mock_settings)
    values = ', '.join(f'{v:g}' for v in band('afm', 'low', 3))
    assert verdict(CASES['low_loads'], _result(deps, f'Use {values} nN?'))
    assert not verdict(CASES['low_loads'], _result(deps, 'Use 1, 2, 3 nN?'))


def test_plot_and_fetch_from_the_tools_pass(tmp_path, mock_settings):
    deps = _deps('plot_cof_vs_load', tmp_path, mock_settings)
    propose_plot(deps, PlotProposalRequest(
        plot_type='summary', datasets=[f'workspace:/{DATASET}'],
        fields=[PlotField(key='x_axis', value='force'), PlotField(key='y_axis', value='cof')]))
    assert verdict(CASES['plot_cof_vs_load'], _result(deps, 'Proposed.'))
    deps = _deps('dm_fetch', tmp_path / 'b', mock_settings)
    fetch_2dmatpedia_structure(deps, DM_ROWS[1]['dm_id'])
    assert verdict(CASES['dm_fetch'], _result(deps, 'Proposed.'))


def test_lookup_cases_pass_on_the_fixture_answer(tmp_path, mock_settings):
    deps = _deps('summarise_dataset', tmp_path, mock_settings)
    assert verdict(CASES['summarise_dataset'],
                   _result(deps, 'It holds h-MoS2 and h-WS2.', tools=['summarise_results']))
    assert not verdict(CASES['summarise_dataset'],
                       _result(deps, 'It holds h-MoS2.', tools=['summarise_results']))
    deps = _deps('count_complete_runs', tmp_path / 'b', mock_settings)
    assert verdict(CASES['count_complete_runs'],
                   _result(deps, '2 of them completed.', tools=['summarise_results']))


# ---- fix round 1: loosened checks pass a correct answer, still fail a wrong one ----

DECLINE_REPLIES = {
    'executed_setting_walltime': 'The walltime is used by the HPC scheduler; change it in '
                                 'the Settings tab yourself.',
    'executed_setting_minimize': 'The minimization command is executed by LAMMPS, so I '
                                 'cannot change it; edit it in Settings.',
    'negative_load': 'Normal loads must be positive in this setup.',
}


@pytest.mark.parametrize('name', sorted(DECLINE_REPLIES))
def test_declines_pass_on_a_statement_but_not_with_a_proposal(name, tmp_path, mock_settings):
    deps = _deps(name, tmp_path, mock_settings)
    assert verdict(CASES[name], _result(deps, DECLINE_REPLIES[name]))
    deps.pending.append(build_changeset(deps, 'x', []))
    assert not verdict(CASES[name], _result(deps, DECLINE_REPLIES[name]))


def test_negative_load_also_passes_on_a_question(tmp_path, mock_settings):
    deps = _deps('negative_load', tmp_path, mock_settings)
    assert verdict(CASES['negative_load'], _result(deps, 'Did you mean 5 nN?'))
    assert not verdict(CASES['negative_load'], _result(deps, 'Sure.'))


def test_numbers_are_parsed_not_matched_as_substrings(tmp_path, mock_settings):
    deps = _deps('low_loads', tmp_path, mock_settings)
    low, mid, high = band('afm', 'low', 3)
    assert verdict(CASES['low_loads'], _result(deps, f'Use {low:.1f}, {mid:.1f} and {high:.1f} nN?'))
    # '20' inside '2026' is not the number 20
    text = f'In {high:g}{high:g} we used {low:g} and {mid:g} nN?'
    assert not verdict(CASES['low_loads'], _result(deps, text))


def test_sweep_values_to_three_figures_pass_and_a_linear_sweep_does_not(tmp_path,
                                                                         mock_settings):
    deps = _deps('log_load_sweep', tmp_path, mock_settings)
    printed = [float(f'{v:.3g}') for v in make_sweep(1, 100, 5, 'log')]
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw', loads_nN=printed))
    assert verdict(CASES['log_load_sweep'], _result(deps, 'Proposed.'))
    deps = _deps('log_load_sweep', tmp_path / 'b', mock_settings)
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw',
                                    loads_nN=make_sweep(1, 100, 5)))
    assert not verdict(CASES['log_load_sweep'], _result(deps, 'Proposed.'))


def test_band_gap_is_matched_as_a_number(tmp_path, mock_settings):
    deps = _deps('mp_band_gap', tmp_path, mock_settings)
    tools = ['query_materials_project']
    assert verdict(CASES['mp_band_gap'], _result(deps, 'About 1.230 eV.', tools=tools))
    assert not verdict(CASES['mp_band_gap'], _result(deps, 'About 11.23 eV.', tools=tools))


def test_dataset_materials_may_drop_the_structure_prefix(tmp_path, mock_settings):
    deps = _deps('summarise_dataset', tmp_path, mock_settings)
    tools = ['summarise_results']
    assert verdict(CASES['summarise_dataset'], _result(deps, 'MoS2 and WS2.', tools=tools))
    assert not verdict(CASES['summarise_dataset'], _result(deps, 'MoS2 and WSe2.', tools=tools))


def test_search_cases_still_need_the_exact_file_name(tmp_path, mock_settings):
    deps = _deps('search_structures', tmp_path, mock_settings)
    tools = ['search_structures']
    assert verdict(CASES['search_structures'], _result(deps, 'h-MoS2.', tools=tools))
    assert not verdict(CASES['search_structures'], _result(deps, 'MoS2.', tools=tools))


def test_completed_count_must_sit_by_the_word_completed(tmp_path, mock_settings):
    deps = _deps('count_complete_runs', tmp_path, mock_settings)
    tools = ['summarise_results']
    assert verdict(CASES['count_complete_runs'],
                   _result(deps, 'h-WS2 has 2 completed runs.', tools=tools))
    assert verdict(CASES['count_complete_runs'],
                   _result(deps, 'The h-WS2 runs are **2**, and both completed.', tools=tools))
    assert not verdict(CASES['count_complete_runs'],
                       _result(deps, 'Across 2 materials, 4 runs completed.', tools=tools))


def test_a_plain_statement_answers_the_yes_or_no_concept(tmp_path, mock_settings):
    deps = _deps('concept_tmd', tmp_path, mock_settings)
    assert verdict(CASES['concept_tmd'],
                   _result(deps, 'h-WS2 is a transition-metal dichalcogenide.'))
    assert not verdict(CASES['concept_tmd'],
                       _result(deps, 'h-WS2 is not a transition-metal dichalcogenide.'))


def test_a_number_past_the_cut_shown_in_the_chat_is_still_grounded():
    long_result = TurnEvent('tool_result', 'lattice_mismatch: ' + 'x' * 182,
                            payload='x' * 200 + ' area 8.79 A^2')
    reply = SimpleNamespace(events=[long_result], text='The area is 8.79 A^2.')
    assert grounded(reply)


def test_cof_plot_needs_cof_on_the_y_axis(tmp_path, mock_settings):
    deps = _deps('plot_cof_vs_load', tmp_path, mock_settings)
    propose_plot(deps, PlotProposalRequest(
        plot_type='summary', datasets=[f'workspace:/{DATASET}'],
        fields=[PlotField(key='x_axis', value='force'), PlotField(key='y_axis', value='lf')]))
    assert not verdict(CASES['plot_cof_vs_load'], _result(deps, 'Proposed.'))


def test_slope_bar_chart_prompt_names_what_cof_histogram_draws():
    assert 'slope' in CASES['plot_slope_bars'].prompt
    assert 'plot_histogram' not in CASES


# ---- domain group: concepts, goal -> simulation, grounded numbers ----------------

def _tool_result(text):
    return TurnEvent('tool_result', text)


def test_grounded_ignores_integers_and_formula_tokens(tmp_path, mock_settings):
    deps = _deps('concept_tmd', tmp_path, mock_settings)
    assert grounded(_result(deps, 'Yes: 2H MoS2 and 1T phases, 3 layers, 2D sheets.'))


def test_grounded_accepts_tool_and_fixture_values_rounded(tmp_path, mock_settings):
    deps = _deps('gap_needs_lookup', tmp_path, mock_settings)
    gap = MP_ROWS[0]['band_gap']
    assert grounded(_result(deps, f'The gap is {gap} eV.'))
    assert grounded(_result(deps, f'The gap is about {gap:.1f} eV.'))
    result = _result(deps, 'The mismatch is 4.27 %.')
    result.events.insert(0, _tool_result('lattice: mismatch 4.271 %'))
    assert grounded(result)
    assert not grounded(_result(deps, 'The gap is 1.8 eV.'))


@pytest.mark.parametrize('name, good, bad', [
    ('concept_superlubricity',
     'Their lattices are incommensurate, so the lateral forces from the atoms cancel out.',
     'Because the sheets are smooth.'),
    ('concept_prefix_t', 'It means the T-phase (octahedral, 1T) polytype.',
     'It is a hexagonal structure.'),
    ('concept_tmd', 'Yes, WS2 is a transition-metal dichalcogenide.',
     'No, it is an oxide.'),
    ('concept_cof', 'The COF is the mean lateral force divided by the normal load.',
     'It is the friction.'),
    ('concept_stick_slip', 'The tip sticks in a minimum of the energy landscape, the '
                           'spring loads, then it slips by one lattice period.',
     'The tip moves at constant speed.'),
])
def test_concept_cases_judge_the_explanation(name, good, bad, tmp_path, mock_settings):
    deps = _deps(name, tmp_path, mock_settings)
    assert verdict(CASES[name], _result(deps, good))
    assert not verdict(CASES[name], _result(deps, bad))
    assert not verdict(CASES[name], _result(deps, good + ' Typically 0.37 nN.'))
    deps.pending.append(build_changeset(deps, 'x', []))
    assert not verdict(CASES[name], _result(deps, good))


def test_cheap_corrugation_wants_a_pes_scan_of_both(tmp_path, mock_settings):
    deps = _deps('cheap_corrugation', tmp_path, mock_settings)
    setup_pes_sheet(deps, PESSheetSetupRequest(materials=['h-MoS2', 'h-WS2'],
                                               potential_type='sw'))
    assert verdict(CASES['cheap_corrugation'], _result(deps, 'Proposed a PES scan.'))
    deps = _deps('cheap_corrugation', tmp_path / 'b', mock_settings)
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2', 'h-WS2'], potential_type='sw',
                                    loads_nN=[10]))
    assert not verdict(CASES['cheap_corrugation'], _result(deps, 'Proposed AFM.'))


def test_layer_dependence_wants_a_layer_sweep(tmp_path, mock_settings):
    deps = _deps('layer_dependence', tmp_path, mock_settings)
    setup_afm(deps, AFMSetupRequest(materials=['h-MoS2'], potential_type='sw',
                                    loads_nN=[20], layers=list(LAYERS)))
    assert verdict(CASES['layer_dependence'], _result(deps, 'Proposed.'))


def test_direction_dependence_wants_scan_angles_on_sheets(tmp_path, mock_settings):
    deps = _deps('direction_dependence', tmp_path, mock_settings)
    setup_sheet(deps, SheetSetupRequest(materials=['h-MoS2'], potential_type='sw',
                                        pressures_GPa=[1], scan_angles_deg=list(ANGLES)))
    assert verdict(CASES['direction_dependence'], _result(deps, 'Proposed.'))


def test_gap_lookup_needs_the_tool_and_the_fixture_value(tmp_path, mock_settings):
    deps = _deps('gap_needs_lookup', tmp_path, mock_settings)
    tools = ['query_materials_project']
    gap = min(MP_ROWS, key=lambda row: row['e_above_hull'])['band_gap']
    assert verdict(CASES['gap_needs_lookup'], _result(deps, f'{gap} eV.', tools=tools))
    assert not verdict(CASES['gap_needs_lookup'], _result(deps, f'{gap} eV.'))
    assert not verdict(CASES['gap_needs_lookup'],
                       _result(deps, f'{gap} eV, bulk 1.29 eV.', tools=tools))


def test_stable_polytype_names_the_entry(tmp_path, mock_settings):
    deps = _deps('stable_polytype', tmp_path, mock_settings)
    tools = ['query_materials_project']
    assert verdict(CASES['stable_polytype'],
                   _result(deps, '2H, P6_3/mmc (mp-2815), 0.000 eV above the hull.',
                           tools=tools))
    assert not verdict(CASES['stable_polytype'], _result(deps, 'The R3m one.', tools=tools))


def test_cof_guess_declines_to_invent_a_value(tmp_path, mock_settings):
    deps = _deps('cof_guess', tmp_path, mock_settings)
    assert verdict(CASES['cof_guess'],
                   _result(deps, "I can't give a value without running a simulation; "
                                 'shall I set one up?'))
    assert not verdict(CASES['cof_guess'], _result(deps, 'Around 0.01 to 0.05.'))


def test_mismatch_case_needs_the_tool_and_the_cif_value(tmp_path, mock_settings):
    deps = _deps('mismatch_pair', tmp_path, mock_settings)
    tool_text = lattice_mismatch(deps, 'h-MoS2', 'h-WS2')
    value = tool_text.split('mismatch a ', 1)[1].split(' %', 1)[0]
    good = _result(deps, f'The mismatch is {value} %.', tools=['lattice_mismatch'])
    good.events.insert(0, TurnEvent('tool_result', f'lattice_mismatch: {tool_text[:200]}'))
    assert verdict(CASES['mismatch_pair'], good)
    assert not verdict(CASES['mismatch_pair'], _result(deps, f'The mismatch is {value} %.'))
    bad = _result(deps, 'The mismatch is 4.10 %.', tools=['lattice_mismatch'])
    bad.events.insert(0, TurnEvent('tool_result', f'lattice_mismatch: {tool_text[:200]}'))
    assert not verdict(CASES['mismatch_pair'], bad)


def test_lattice_constant_case_accepts_the_rounded_cif_value(tmp_path, mock_settings):
    deps = _deps('lattice_constant', tmp_path, mock_settings)
    tool_text = lattice_mismatch(deps, 'h-WS2')
    a_value = float(tool_text.split(': a ', 1)[1].split(',', 1)[0])
    for printed in (f'{a_value:.3f}', f'{a_value:.2f}'):
        result = _result(deps, f'a = {printed} Angstrom.', tools=['lattice_mismatch'])
        result.events.insert(0, TurnEvent('tool_result', f'lattice_mismatch: {tool_text}'))
        assert verdict(CASES['lattice_constant'], result)
    assert not verdict(CASES['lattice_constant'],
                       _result(deps, 'a = 3.32 Angstrom.', tools=['lattice_mismatch']))


def test_cite_knowledge_needs_the_passage_title(tmp_path, mock_settings):
    deps = _deps('cite_knowledge', tmp_path, mock_settings)
    hit = search_knowledge(deps, 'friction thin sheets layers added')
    assert 'Puckering of thin sheets' in hit.splitlines()[0]
    tools = ['search_knowledge']
    assert verdict(CASES['cite_knowledge'],
                   _result(deps, 'Thin sheets pucker (source: Puckering of thin sheets).',
                           tools=tools))
    assert not verdict(CASES['cite_knowledge'], _result(deps, 'Thin sheets pucker.',
                                                        tools=tools))
