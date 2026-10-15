"""Live-LLM evaluation of the Phase 1 assistant. Deselected by default.

    pytest tests/agent/evals -m agent_eval                       # Gemini (GEMINI_API_KEY)
    AGENT_EVAL_PROVIDER=ollama pytest tests/agent/evals -m agent_eval
    AGENT_EVAL_REPEATS=3 AGENT_EVAL_PAUSE=20 ...                 # free-tier pacing

Expected values come from code (make_sweep, resolve_material), never typed by hand.
"""
import asyncio
import os
import time

import pytest

pytest.importorskip('pydantic_ai')

from src.agent.providers import ProviderUnavailable, build_model, ollama_status  # noqa: E402
from src.agent.runtime import AgentSession  # noqa: E402
from src.agent.settings import AgentSettings  # noqa: E402
from src.agent.tools.setup import make_sweep  # noqa: E402
from tests.helpers.agent import make_deps  # noqa: E402

pytestmark = pytest.mark.agent_eval
REPEATS = int(os.environ.get('AGENT_EVAL_REPEATS', '3'))
PAUSE = float(os.environ.get('AGENT_EVAL_PAUSE', '0'))


def _settings():
    settings = AgentSettings(provider=os.environ.get('AGENT_EVAL_PROVIDER', 'gemini'))
    if settings.provider == 'ollama':
        ok, message = ollama_status(settings)
        if not ok:
            pytest.skip(message)
    else:
        try:
            build_model(settings)
        except ProviderUnavailable as exc:
            pytest.skip(str(exc))
    return settings


def _sets(proposals):
    return {(op.section, op.key): op.value for p in proposals
            for op in getattr(p, 'ops', []) if getattr(op, 'kind', '') == 'set'}


def _materials(proposals):
    for p in proposals:
        for op in getattr(p, 'ops', []):
            if getattr(op, 'kind', '') == 'materials' and op.value:
                return set(op.value)
            if getattr(op, 'kind', '') == 'set' and op.key == 'mat':
                return {op.value}
    return set()


def _models(proposals):
    return {op.value for p in proposals for op in getattr(p, 'ops', [])
            if getattr(op, 'kind', '') == 'set_model'}


CASES = {
    'afm_two_materials': (
        'Set up an AFM simulation on h-MoS2 and h-WS2 with sw potentials at 5, 10 and 20 nN.',
        lambda p, e: _materials(p) == {'h-MoS2', 'h-WS2'} and _sets(p).get(('general', 'force')) == [5, 10, 20]),
    'sheet_single': (
        'Set up a sheet-on-sheet simulation of h-MoS2 with sw potentials at 1 GPa.',
        lambda p, e: 'sheetonsheet' in _models(p) and _sets(p).get(('general', 'pressure')) == [1]),
    'pes_tip_not_sheet': (
        'Run a PES scan of the AFM tip over h-WS2 with sw potentials.',
        lambda p, e: _models(p) == {'pes_tip'}),
    'even_sweep': (
        'For the current AFM config use 5 loads evenly spaced from 2 to 10 nN on h-MoS2 with sw.',
        lambda p, e: _sets(p).get(('general', 'force')) == make_sweep(2, 10, 5)),
    'speed': (
        'Change the sliding speed of the current AFM config to 2 m/s.',
        lambda p, e: _sets(p).get(('general', 'scan_speed')) in (2, 2.0)),
    'ambiguous_unit': (
        'Set up AFM on h-MoS2 with loads from 10 to 50 nm.',
        lambda p, e: not p),
    'unknown_material': (
        'Set up an AFM simulation on unobtainium.',
        lambda p, e: not p),
    'explain': (
        "What's wrong with my current config?",
        lambda p, e: any(ev.kind == 'tool_call' and ev.text.startswith('explain_current') for ev in e)),
}


@pytest.mark.parametrize('repeat', range(REPEATS))
@pytest.mark.parametrize('case', sorted(CASES))
def test_case(case, repeat, tmp_path, mock_settings):  # pylint: disable=unused-argument
    prompt, check = CASES[case]
    session = AgentSession(_settings())
    deps = make_deps(tmp_path, mock_settings)
    events = []
    time.sleep(PAUSE)
    asyncio.run(session.run_turn(prompt, deps, events.append, session.new_token()))
    errors = [e for e in events if e.kind == 'error']
    assert not any(e.payload == 'rate_limited' for e in errors), 'rate limited: raise AGENT_EVAL_PAUSE'
    assert check(deps.pending, events), (
        f"pending={[getattr(p, 'title', p) for p in deps.pending]} "
        f"text={''.join(e.text for e in events if e.kind == 'text')[:300]}")
