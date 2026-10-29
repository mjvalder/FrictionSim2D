"""Live-model evaluation of the assistant: judges a provider/model. Deselected by default.

    pytest tests/agent/evals -m agent_eval                          # Gemini (GEMINI_API_KEY)
    AGENT_EVAL_PROVIDER=ollama AGENT_EVAL_MODEL=qwen3:8b pytest tests/agent/evals -m agent_eval
    AGENT_EVAL_PROVIDER=anthropic pytest tests/agent/evals -m agent_eval  # ANTHROPIC_API_KEY
    AGENT_EVAL_REPEATS=3 AGENT_EVAL_PAUSE=20 ...                    # free-tier pacing

Cases and their code-computed expectations: ``cases.py``. Results:
``tests/agent/evals/.results/<provider>-<model>-<timestamp>.json`` and a table
at the end (``conftest.py``). Materials Project is always a fake client.
"""
import time

import pytest

pytest.importorskip('pydantic_ai')

from src.agent.providers import ProviderUnavailable  # noqa: E402
from tests.agent.evals.cases import CASES, GROUP_OF, prepare_deps, verdict  # noqa: E402
from tests.helpers.agent import make_deps  # noqa: E402
from tests.helpers.agent_eval import build_session, eval_env, run_eval_turn  # noqa: E402

pytestmark = pytest.mark.agent_eval
ENV = eval_env()
#: Turns recorded so far in this process: the first is marked cold.
_RECORDED = []


@pytest.mark.parametrize('repeat', range(ENV.repeats))
@pytest.mark.parametrize('case', sorted(CASES))
def test_case(case, repeat, tmp_path, mock_settings, request):
    spec = CASES[case]
    deps = make_deps(tmp_path, mock_settings, model=spec.model)
    try:
        session = build_session(ENV, deps)
    except ProviderUnavailable as exc:
        pytest.skip(str(exc))
    prepare_deps(spec, deps)
    time.sleep(ENV.pause)
    result = run_eval_turn(session, spec.prompt, deps)
    detail = (f"errors={[(e.payload, e.text[:120]) for e in result.errors]} "
              f"tools={result.tool_calls} "
              f"pending={[getattr(p, 'title', p) for p in result.pending]} "
              f"text={result.text[:300]!r}")
    request.node.user_properties.append(('agent_eval', {
        'case': case, 'group': GROUP_OF[case], 'repeat': repeat,
        'latency_s': round(result.latency_s, 3), 'cold': not _RECORDED,
        'requests': result.requests, 'tokens': result.tokens,
        'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens,
        'cache_read_tokens': result.cache_read_tokens,
        'cache_write_tokens': result.cache_write_tokens,
        'detail': detail}))
    _RECORDED.append(case)
    if any(e.payload == 'rate_limited' for e in result.errors):
        pytest.skip(f'rate limited ({result.errors[0].text}); raise AGENT_EVAL_PAUSE')
    assert not result.errors, detail
    assert verdict(spec, result), detail
