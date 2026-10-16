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
from tests.agent.evals.cases import (CASES, DATASET, DM_ROWS, GROUPS, MUST_ASK,  # noqa: E402
                                     band, prepare_deps, verdict)
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
    for name in ('ambiguous_unit', 'unknown_material', 'executed_setting_walltime',
                 'ambiguous_family'):
        assert MUST_ASK[name].ask


def test_every_registered_tool_is_exercised_by_a_case():
    session = AgentSession(AgentSettings(), model=FunctionModel(lambda m, i: None))
    covered = {tool for case in CASES.values() for tool in case.tools}
    assert set(session.tool_names()) <= covered
    assert covered <= set(session.tool_names())  # no typo in a case's tools


@pytest.mark.parametrize('name', sorted(CASES))
def test_no_case_passes_on_a_turn_that_only_asked_back(name, tmp_path, mock_settings):
    case = CASES[name]
    deps = _deps(name, tmp_path, mock_settings)
    if case.ask:
        assert not verdict(case, _result(deps, 'Done.'))
        deps.pending.append(build_changeset(deps, 'x', []))
        assert not verdict(case, _result(deps, FUNCTION_REPLY))
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
