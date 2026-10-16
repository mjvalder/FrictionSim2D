"""The eval cases themselves, without a model: coverage, no vacuous passes, and
that a correct turn (built by calling the tools directly) does pass."""
import pytest

pytest.importorskip('pydantic_ai')

from pydantic_ai.models.function import FunctionModel  # noqa: E402

from src.agent.changes import build_changeset  # noqa: E402
from src.agent.events import TurnEvent  # noqa: E402
from src.agent.runtime import AgentSession  # noqa: E402
from src.agent.settings import AgentSettings  # noqa: E402
from src.agent.tools.materials_db import fetch_2dmatpedia_structure  # noqa: E402
from src.agent.tools.plots import PlotField, PlotProposalRequest, propose_plot  # noqa: E402
from src.agent.tools.setup import AFMSetupRequest, make_sweep, setup_afm  # noqa: E402
from tests.agent.evals.cases import (CASES, DATASET, DECLINES, DM_ROWS, GROUPS,  # noqa: E402
                                     MUST_ASK, band, prepare_deps, verdict)
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
    assert not verdict(CASES['count_complete_runs'],
                       _result(deps, 'Across 2 materials, 4 runs completed.', tools=tools))


def test_cof_plot_needs_cof_on_the_y_axis(tmp_path, mock_settings):
    deps = _deps('plot_cof_vs_load', tmp_path, mock_settings)
    propose_plot(deps, PlotProposalRequest(
        plot_type='summary', datasets=[f'workspace:/{DATASET}'],
        fields=[PlotField(key='x_axis', value='force'), PlotField(key='y_axis', value='lf')]))
    assert not verdict(CASES['plot_cof_vs_load'], _result(deps, 'Proposed.'))


def test_slope_bar_chart_prompt_names_what_cof_histogram_draws():
    assert 'slope' in CASES['plot_slope_bars'].prompt
    assert 'plot_histogram' not in CASES
