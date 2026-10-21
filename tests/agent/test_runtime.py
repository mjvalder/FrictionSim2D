"""The agent loop, driven by pydantic-ai's FunctionModel (no network)."""
import asyncio
import json

import pytest

pytest.importorskip('pydantic_ai')

from pydantic_ai import CancellationToken  # noqa: E402
from pydantic_ai.exceptions import ModelHTTPError  # noqa: E402
from pydantic_ai.messages import ToolReturnPart, RetryPromptPart  # noqa: E402
from pydantic_ai.models.function import DeltaToolCall, FunctionModel  # noqa: E402

from src.agent.runtime import AgentSession  # noqa: E402
from src.agent.settings import AgentSettings  # noqa: E402
from tests.helpers.agent import make_deps  # noqa: E402


def _last_parts(messages):
    return messages[-1].parts if messages else []


def scripted(*steps):
    """A stream function: step i runs on the i-th model request of the turn.

    Each step is either text (str) or a (tool_name, args_dict) call.
    """
    async def stream(messages, info):
        requests = sum(1 for m in messages if m.kind == 'request')
        step = steps[min(requests, len(steps)) - 1]
        if isinstance(step, str):
            yield step
        else:
            yield {0: DeltaToolCall(name=step[0], json_args=json.dumps(step[1]))}
    return stream


def run(session, deps, text, token=None):
    events = []
    asyncio.run(session.run_turn(text, deps, events.append, token or session.new_token()))
    return events


@pytest.fixture(name='deps')
def _deps(tmp_path, mock_settings):
    return make_deps(tmp_path, mock_settings)


def test_tool_call_streams_proposal_and_text(deps):
    model = FunctionModel(stream_function=scripted(
        ('set_fields', {'changes': [{'section': 'general', 'key': 'scan_speed', 'value': 2}]}),
        'Proposed: speed 2 m/s.'))
    session = AgentSession(AgentSettings(), model=model)
    events = run(session, deps, 'speed 2')
    kinds = [e.kind for e in events]
    assert 'tool_call' in kinds and 'proposal' in kinds and kinds[-1] == 'done'
    assert ''.join(e.text for e in events if e.kind == 'text').endswith('speed 2 m/s.')
    assert session.history
    results = [e.text for e in events if e.kind == 'tool_result']
    assert results and 'proposed' in results[0].lower()


def test_bad_arguments_retry_then_succeed(deps):
    model = FunctionModel(stream_function=scripted(
        ('set_fields', {'changes': [{'section': 'general', 'key': 'scan_sped', 'value': 2}]}),
        ('set_fields', {'changes': [{'section': 'general', 'key': 'scan_speed', 'value': 2}]}),
        'done'))
    events = run(AgentSession(AgentSettings(), model=model), deps, 'speed 2')
    assert sum(e.kind == 'proposal' for e in events) == 1
    assert events[-1].kind == 'done'


def test_gives_up_after_retries(deps):
    bad = ('set_fields', {'changes': [{'section': 'general', 'key': 'nope', 'value': 2}]})
    model = FunctionModel(stream_function=scripted(bad, bad, bad, bad, bad))
    events = run(AgentSession(AgentSettings(), model=model), deps, 'x')
    assert events[-1].kind == 'error' and events[-1].payload == 'gave_up'
    assert 'Builder form' in events[-1].text


def test_cancel_before_start(deps):
    model = FunctionModel(stream_function=scripted('hello'))
    session = AgentSession(AgentSettings(), model=model)
    token = session.new_token()
    token.cancel()
    events = run(session, deps, 'x', token)
    assert events[-1].kind == 'error' and events[-1].payload == 'cancelled'


def test_rate_limit_event(deps):
    async def limited(messages, info):
        raise ModelHTTPError(429, 'gemini', body={'error': {'message': 'Please retry in 7.5s.'}})
        yield ''  # pragma: no cover  (makes this an async generator)
    events = run(AgentSession(AgentSettings(), model=FunctionModel(stream_function=limited)), deps, 'x')
    assert events[-1].payload == 'rate_limited'
    assert events[-1].retry_after == pytest.approx(7.5)


def test_switch_keeps_text_only(deps):
    model = FunctionModel(stream_function=scripted(
        ('explain_current', {}), 'All good.'))
    session = AgentSession(AgentSettings(), model=model)
    run(session, deps, 'check')
    session.switch(AgentSettings(provider='ollama'))
    parts = [type(p).__name__ for m in session.history for p in m.parts]
    assert 'ToolCallPart' not in parts and 'ToolReturnPart' not in parts
    assert 'UserPromptPart' in parts and 'TextPart' in parts


def test_new_chat_clears(deps):
    session = AgentSession(AgentSettings(), model=FunctionModel(stream_function=scripted('hi')))
    run(session, deps, 'x')
    session.new_chat()
    assert session.history == []


def test_every_tool_registered():
    session = AgentSession(AgentSettings(), model=FunctionModel(stream_function=scripted('hi')))
    names = set(session.tool_names())
    assert names == {'list_modes', 'explain_current', 'find_parameter', 'set_fields',
                     'make_sweep', 'suggest_values', 'setup_afm', 'setup_sheet',
                     'setup_indent', 'setup_pes_tip', 'setup_pes_sheet',
                     'search_structures', 'search_potentials', 'lattice_mismatch',
                     'workspace_overview',
                     'describe_run', 'propose_mkdir', 'propose_copy_in',
                     'find_result_sets', 'summarise_results', 'plot_types',
                     'plot_schema', 'propose_plot', 'query_materials_project',
                     'query_2dmatpedia', 'fetch_2dmatpedia_structure'}


def test_suggest_values_tool_has_no_spacing(deps):
    schemas = {}

    async def spy(messages, info):
        schemas.update({t.name: t.parameters_json_schema for t in info.function_tools})
        yield 'ok'
    session = AgentSession(AgentSettings(), model=FunctionModel(stream_function=spy))
    run(session, deps, 'x')
    properties = schemas['suggest_values']['properties']
    assert 'spacing' not in properties and 'descriptor' in properties


def test_instructions_survive_switch(deps):
    seen = []

    async def spy(messages, info):
        seen.append(info.instructions)
        yield 'ok'
    session = AgentSession(AgentSettings(), model=FunctionModel(stream_function=spy))
    run(session, deps, 'one')
    session.switch(AgentSettings(provider='ollama'))
    session._model = FunctionModel(stream_function=spy)  # pylint: disable=protected-access
    run(session, deps, 'two')
    assert len(seen) == 2 and '/no_think' in seen[1]


def test_unavailable_provider_is_an_event(deps, tmp_path, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'empty'))
    events = run(AgentSession(AgentSettings()), deps, 'x')
    assert events[-1].kind == 'error' and events[-1].payload == 'unavailable'


def test_tool_crash_hides_paths(deps, monkeypatch):
    from src.agent.tools import setup

    def boom(_deps):
        raise RuntimeError('/home/x/secret')
    monkeypatch.setattr(setup, 'list_modes', boom)
    events = run(AgentSession(AgentSettings(), model=FunctionModel(
        stream_function=scripted(('list_modes', {}), 'ok'))), deps, 'x')
    results = [e.text for e in events if e.kind == 'tool_result']
    assert results and '/home/' not in results[0]


def test_tool_results_carry_no_absolute_paths(deps, tmp_path):
    from types import SimpleNamespace
    from src.agent.changes import apply_changeset
    from src.agent.runtime import _call
    from src.agent.tools import setup
    deps.document.workspace_dir = None  # an unsaved config keeps absolute paths
    setup.setup_afm(deps, setup.AFMSetupRequest(materials=['h-MoS2'], potential_type='sw'))
    apply_changeset(deps.pending[-1], deps.document, deps.settings_document, force=True)
    deps.document.set('2D', 'cif_path', str(deps.roots.workspace / 'missing.cif'))
    ctx = SimpleNamespace(deps=deps)
    for text in (_call(ctx, setup.find_parameter, 'cif path'),
                 _call(ctx, setup.explain_current)):
        assert '/home' not in text and str(tmp_path) not in text, text
    assert 'examples:/potentials/' in _call(ctx, setup.find_parameter, 'pot_path')


def test_retry_messages_carry_no_absolute_paths(deps, tmp_path):
    from types import SimpleNamespace
    from pydantic_ai import ModelRetry
    from src.agent.runtime import _call
    from src.agent.sandbox import SandboxError

    def refuse(d):
        raise SandboxError(f'{d.roots.workspace}/x: outside')
    with pytest.raises(ModelRetry) as info:
        _call(SimpleNamespace(deps=deps), refuse)
    assert str(tmp_path) not in str(info.value) and 'workspace:/x' in str(info.value)


def test_new_chat_mid_turn_keeps_history_empty(deps):
    session = AgentSession(AgentSettings())

    async def stream(messages, info):
        session.new_chat()  # e.g. the workspace changed while this turn ran
        yield 'late reply'
    session._model = FunctionModel(stream_function=stream)  # pylint: disable=protected-access
    run(session, deps, 'x')
    assert session.history == []


def test_unexpected_exception_becomes_an_internal_error_event(deps):
    async def stream(messages, info):
        raise RuntimeError('boom')
        yield 'never'  # pylint: disable=unreachable
    session = AgentSession(AgentSettings(), model=FunctionModel(stream_function=stream))
    events = run(session, deps, 'hi')
    assert [(e.kind, e.text, e.payload) for e in events] == [
        ('error', 'RuntimeError: boom', 'internal')]


def _looping_model():
    async def stream(messages, info):
        yield {0: DeltaToolCall(name='list_modes', json_args='{}')}
    return FunctionModel(stream_function=stream)


def test_step_cap_stops_with_a_continue_hint(deps):
    settings = AgentSettings(max_requests_per_turn=3)
    events = run(AgentSession(settings, model=_looping_model()), deps, 'loop')
    assert sum(e.kind == 'tool_call' for e in events) == 3
    last = events[-1]
    assert (last.kind, last.payload) == ('error', 'step_limit')
    assert last.text == "Stopped after 3 steps to save quota — type 'continue' to carry on."


def test_continue_after_the_step_cap_keeps_the_work_so_far(deps):
    seen = []

    async def stream(messages, info):
        seen.append(messages)
        if len(seen) <= 2:
            yield {0: DeltaToolCall(name='list_modes', json_args='{}')}
        else:
            yield 'carried on'
    session = AgentSession(AgentSettings(max_requests_per_turn=2),
                           model=FunctionModel(stream_function=stream))
    assert run(session, deps, 'loop')[-1].payload == 'step_limit'
    events = run(session, deps, 'continue')
    assert events[-1].kind == 'done'
    kinds = [type(p).__name__ for m in seen[-1] for p in m.parts]
    assert 'ToolReturnPart' in kinds  # the capped turn's tool results are remembered


def test_default_step_cap_is_fifty(deps):
    events = run(AgentSession(AgentSettings(), model=_looping_model()), deps, 'loop')
    assert sum(e.kind == 'tool_call' for e in events) == 50
    assert events[-1].payload == 'step_limit'


def test_connection_refused_is_an_unreachable_event(deps):
    import httpx
    from pydantic_ai.exceptions import ModelAPIError

    async def refused(messages, info):
        raise ModelAPIError('qwen3:4b', 'Connection error.') from httpx.ConnectError('refused')
        yield ''  # pragma: no cover
    settings = AgentSettings(provider='ollama')
    events = run(AgentSession(settings, model=FunctionModel(stream_function=refused)), deps, 'x')
    assert (events[-1].kind, events[-1].payload) == ('error', 'unreachable')
    assert 'could not reach' in events[-1].text.lower()


def test_continue_after_the_cap_survives_a_history_collapse(deps):
    seen = []

    async def stream(messages, info):
        seen.append(messages)
        last = messages[-1].parts[-1]
        if type(last).__name__ == 'UserPromptPart' and last.content == 'continue':
            yield 'carried on'
        elif len(seen) <= 3 or type(last).__name__ != 'UserPromptPart':
            yield {0: DeltaToolCall(name='list_modes', json_args='{}')}
        else:
            yield 'ok'
    session = AgentSession(AgentSettings(max_requests_per_turn=2),
                           model=FunctionModel(stream_function=stream), history_budget=50)
    run(session, deps, 'earlier question')  # capped too: history is long now
    assert run(session, deps, 'set up a sweep')[-1].payload == 'step_limit'
    assert run(session, deps, 'continue')[-1].kind == 'done'
    sent = seen[-1]
    kinds = [type(p).__name__ for m in sent for p in m.parts]
    assert 'ToolReturnPart' in kinds  # the capped turn reached the model intact
    text = repr(sent)
    assert 'Summary of the earlier conversation' in text  # earlier turns collapsed
    assert 'set up a sweep' in text


def test_history_collapse_keeps_the_latest_turn_intact():
    from pydantic_ai.messages import (ModelRequest, ModelResponse, TextPart,
                                      ToolCallPart, ToolReturnPart, UserPromptPart)
    session = AgentSession(AgentSettings(), model=FunctionModel(stream_function=scripted('x')),
                           history_budget=10)
    old = [ModelRequest(parts=[UserPromptPart(content='old ' * 50)]),
           ModelResponse(parts=[TextPart(content='old answer ' * 50)])]
    latest = [ModelRequest(parts=[UserPromptPart(content='latest')]),
              ModelResponse(parts=[ToolCallPart(tool_name='list_modes', args={},
                                                tool_call_id='c1')]),
              ModelRequest(parts=[ToolReturnPart(tool_name='list_modes', content='r',
                                                 tool_call_id='c1')])]
    session.history = old + latest
    trimmed = session._trimmed_history()  # pylint: disable=protected-access
    assert trimmed[-3:] == latest
    assert 'Summary of the earlier conversation' in repr(trimmed[0])


def test_new_chat_bumps_the_generation_before_clearing():
    seen = []

    class Watched(AgentSession):
        @property
        def history(self):
            return self.__dict__.get('_hist', [])

        @history.setter
        def history(self, value):
            seen.append(getattr(self, '_generation', None))
            self.__dict__['_hist'] = value

    session = Watched(AgentSettings())
    seen.clear()
    session.new_chat()
    assert seen == [1]


def test_turn_cancellation_reaches_the_materials_client(deps):
    seen = []

    class Client:  # pylint: disable=too-few-public-methods
        downloaded_listing = False

        def twodmatpedia(self, formula):  # pylint: disable=unused-argument
            seen.append(seen[0]())
            return []

    def factory(should_stop=None):
        seen.append(should_stop)
        return Client()
    deps.materials_db = factory
    session = AgentSession(AgentSettings(), model=FunctionModel(stream_function=scripted(
        ('query_2dmatpedia', {'formula': 'MoS2'}), 'none')))
    token = session.new_token()
    run(session, deps, 'MoS2?', token)
    assert seen[0] is deps.cancelled and seen[1] is False
    token.cancel()
    assert deps.cancelled() is True
