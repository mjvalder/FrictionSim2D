"""The eval tier's machinery: env switches, sessions, one measured turn, results.

Used by ``tests/agent/evals`` (the live-model cases and their conftest plugin)
and by ``tests/agent/test_eval_plugin.py``. pydantic-ai is imported inside the
functions that need it, so the aggregation helpers stay importable without it.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO / 'tests' / 'agent' / 'evals' / '.results'
#: The scripted provider the plugin's own test uses; never a real model.
FUNCTION_PROVIDER = 'function'
FUNCTION_REPLY = 'Could you tell me which one you mean?'


@dataclass(frozen=True)
class EvalEnv:
    """The env switches: AGENT_EVAL_PROVIDER, _MODEL, _REPEATS, _PAUSE, _RESULTS_DIR;
    for Ollama also _PREFETCH (passages per request)."""

    provider: str
    model: Optional[str]
    repeats: int
    pause: float
    results_dir: Path


def eval_env() -> EvalEnv:
    """The switches from the environment, with their defaults."""
    return EvalEnv(provider=os.environ.get('AGENT_EVAL_PROVIDER', 'gemini'),
                   model=os.environ.get('AGENT_EVAL_MODEL') or None,
                   repeats=int(os.environ.get('AGENT_EVAL_REPEATS', '3')),
                   pause=float(os.environ.get('AGENT_EVAL_PAUSE', '0')),
                   results_dir=Path(os.environ.get('AGENT_EVAL_RESULTS_DIR')
                                    or DEFAULT_RESULTS_DIR))


# ---- sessions ----------------------------------------------------------------

async def _ask_back(messages, info):  # pylint: disable=unused-argument
    """The function provider: one streamed question, no tool call."""
    yield FUNCTION_REPLY


def model_name(env: EvalEnv) -> str:
    """The model the run judges, as the results file names it."""
    if env.provider == FUNCTION_PROVIDER:
        return FUNCTION_PROVIDER
    from src.agent.settings import AgentSettings  # pylint: disable=import-outside-toplevel
    defaults = AgentSettings()
    if env.model:
        return env.model
    return getattr(defaults, env.provider).model


#: Models already warmed up in this process (a cold Ollama load is not a turn).
_WARMED: set = set()
#: One event loop for every turn of the process: the model's HTTP client is cached
#: on the agent, and a turn on a fresh ``asyncio.run`` loop finds its pool closed.
_RUNNER = None
WARM_UP_PROMPT = 'Reply with the single word OK.'


def _run(coro):
    """Run ``coro`` on the process's one event loop (see `src.agent.loop`)."""
    global _RUNNER  # pylint: disable=global-statement
    if _RUNNER is None:
        from src.agent.loop import TurnRunner  # pylint: disable=import-outside-toplevel
        _RUNNER = TurnRunner()
    return _RUNNER.run(coro)


def warm_up(session, deps) -> bool:
    """Run one unrecorded turn, once per process and model, then forget it.

    Returns whether it ran. ``deps`` is left as it was found (no pending changes).
    """
    key = (session.settings.provider, getattr(session.agent.model, 'model_name', ''))
    if key in _WARMED:
        return False
    _WARMED.add(key)
    _run(session.run_turn(WARM_UP_PROMPT, deps, lambda event: None, session.new_token()))
    session.new_chat()
    deps.pending.clear()
    return True


def build_session(env: EvalEnv, deps=None):
    """An `AgentSession` for ``env``; raises `ProviderUnavailable` when it cannot run.

    For Ollama with ``deps`` given, the first session of the process runs one
    unrecorded warm-up turn (`warm_up`), so the model load is not timed.
    """
    # pylint: disable=import-outside-toplevel
    from src.agent.providers import ProviderUnavailable, build_model, ollama_status
    from src.agent.runtime import AgentSession
    from src.agent.settings import AgentSettings
    if env.provider == FUNCTION_PROVIDER:
        from pydantic_ai.models.function import FunctionModel
        return AgentSession(AgentSettings(), model=FunctionModel(
            stream_function=_ask_back, model_name=FUNCTION_PROVIDER))
    if env.provider not in ('gemini', 'ollama', 'anthropic'):
        raise ProviderUnavailable(f'unknown AGENT_EVAL_PROVIDER {env.provider!r}')
    settings = AgentSettings(provider=env.provider)
    if env.model:
        getattr(settings, env.provider).model = env.model
    if env.provider == 'ollama':
        settings.ollama.prefetch_passages = int(os.environ.get('AGENT_EVAL_PREFETCH', '0'))
        ok, message = ollama_status(settings)
        if not ok:
            raise ProviderUnavailable(message)
        session = AgentSession(settings)
        if deps is not None:
            warm_up(session, deps)
        return session
    return AgentSession(settings, model=build_model(settings))


# ---- one measured turn -------------------------------------------------------

@dataclass
class EvalResult:
    """What one turn produced, for a case's check."""

    deps: Any
    events: List[Any] = field(default_factory=list)
    latency_s: float = 0.0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    #: Of ``input_tokens``: read from / written to the provider's prompt cache.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def pending(self) -> List[Any]:
        """The proposals the turn made."""
        return list(self.deps.pending)

    @property
    def text(self) -> str:
        """The assistant's reply, all text events joined."""
        return ''.join(e.text for e in self.events if e.kind == 'text')

    @property
    def tool_calls(self) -> List[str]:
        """Names of the tools the model called, in order."""
        return [e.text.split('(', 1)[0] for e in self.events if e.kind == 'tool_call']

    @property
    def errors(self) -> List[Any]:
        """The turn's error events."""
        return [e for e in self.events if e.kind == 'error']


def _turn_usages(history: Sequence[Any]) -> List[Any]:
    """Usage of each model response in the latest turn."""
    # pylint: disable=import-outside-toplevel
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
    start = 0
    for index, message in enumerate(history):
        if isinstance(message, ModelRequest) and any(
                isinstance(p, UserPromptPart) for p in message.parts):
            start = index
    return [m.usage for m in history[start:]
            if isinstance(m, ModelResponse) and m.usage is not None]


def turn_tokens(history: Sequence[Any]) -> Tuple[int, int]:
    """``(input, output)`` tokens of the model responses in the latest turn."""
    usages = _turn_usages(history)
    return (sum(u.input_tokens or 0 for u in usages),
            sum(u.output_tokens or 0 for u in usages))


def turn_cache_tokens(history: Sequence[Any]) -> Tuple[int, int]:
    """``(cache read, cache write)`` input tokens in the latest turn."""
    usages = _turn_usages(history)
    return (sum(getattr(u, 'cache_read_tokens', 0) or 0 for u in usages),
            sum(getattr(u, 'cache_write_tokens', 0) or 0 for u in usages))


def run_eval_turn(session, prompt: str, deps) -> EvalResult:
    """Run one turn of ``session`` on ``deps``, timing it and counting its usage."""
    result = EvalResult(deps=deps)
    start = time.perf_counter()
    _run(session.run_turn(prompt, deps, result.events.append, session.new_token()))
    result.latency_s = time.perf_counter() - start
    done = [e for e in result.events if e.kind == 'done']
    result.requests = int(done[-1].payload or 0) if done else 0
    if done:
        result.input_tokens, result.output_tokens = turn_tokens(session.history)
        result.cache_read_tokens, result.cache_write_tokens = turn_cache_tokens(
            session.history)
    return result


# ---- aggregation and reporting -------------------------------------------------

def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile ``q`` (0-100) of ``values``; None when empty."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def _rate(passed: int, failed: int) -> Optional[float]:
    return passed / (passed + failed) if passed + failed else None


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _tally(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for record in records:
        row = rows.setdefault(record.get(key) or '-', {'runs': 0, 'passed': 0, 'failed': 0,
                                                       'skipped': 0})
        row['runs'] += 1
        row[record['outcome']] += 1
    for row in rows.values():
        row['pass_rate'] = _rate(row['passed'], row['failed'])
    return dict(sorted(rows.items()))


def summarise(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass rate per case, per group and overall (skips excluded), latency
    (all turns, and warm: without each session's cold first turn) and usage."""
    cases = _tally(records, 'case')
    passed = sum(r['passed'] for r in cases.values())
    failed = sum(r['failed'] for r in cases.values())
    ran = [r for r in records if r['outcome'] != 'skipped']
    latencies = [r['latency_s'] for r in ran]
    warm = [r['latency_s'] for r in ran if not r.get('cold')]
    return {'cases': cases, 'groups': _tally(records, 'group'),
            'overall': {'runs': len(records), 'passed': passed, 'failed': failed,
                        'skipped': len(records) - len(ran), 'pass_rate': _rate(passed, failed)},
            'latency_p50_s': percentile(latencies, 50),
            'latency_p95_s': percentile(latencies, 95),
            'latency_p50_warm_s': percentile(warm, 50),
            'latency_p95_warm_s': percentile(warm, 95),
            'mean_requests': _mean([r['requests'] for r in ran]),
            'mean_tokens': _mean([r['tokens'] for r in ran]),
            'mean_input_tokens': _mean([r.get('input_tokens', 0) for r in ran]),
            'mean_output_tokens': _mean([r.get('output_tokens', 0) for r in ran])}


def _pct(rate: Optional[float]) -> str:
    return '-' if rate is None else f'{rate:.0%}'


def _num(value: Optional[float], spec: str) -> str:
    return '-' if value is None else format(value, spec)


def format_table(summary: Dict[str, Any], provider: str, model: str) -> str:
    """The session-end table: cases, groups, overall, then latency and usage."""
    groups = summary.get('groups', {})
    width = max([len('overall')] + [len(name) for name in summary['cases']]
                + [len(f'[{name}]') for name in groups])
    lines = [f'agent eval: {provider} / {model}',
             f"{'case':<{width}}  {'pass':>5}  {'runs':>4}  {'skip':>4}"]
    rows = (list(summary['cases'].items())
            + [(f'[{name}]', row) for name, row in groups.items()]
            + [('overall', summary['overall'])])
    for name, row in rows:
        lines.append(f"{name:<{width}}  {_pct(row['pass_rate']):>5}  {row['runs']:>4}  "
                     f"{row['skipped']:>4}")
    skipped = summary['overall']['skipped']
    if skipped:
        lines.append(f'pass rate not comparable: {skipped} skipped')
    lines.append(f"latency warm p50 {_num(summary.get('latency_p50_warm_s'), '.1f')} s, "
                 f"warm p95 {_num(summary.get('latency_p95_warm_s'), '.1f')} s "
                 f"(all turns p95 {_num(summary['latency_p95_s'], '.1f')} s)")
    lines.append(f"per turn: {_num(summary['mean_requests'], '.1f')} requests, "
                 f"{_num(summary.get('mean_input_tokens'), '.0f')} input tokens, "
                 f"{_num(summary.get('mean_output_tokens'), '.0f')} output tokens")
    return '\n'.join(lines)


def _safe(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', text)


def results_path(directory: Path, provider: str, model: str, now: datetime) -> Path:
    """``<provider>-<model>-<timestamp>.json`` in ``directory``."""
    return Path(directory) / f'{_safe(provider)}-{_safe(model)}-{now:%Y%m%dT%H%M%S}.json'


def write_results(path: Path, provider: str, model: str,
                  records: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Write the run's records and summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'provider': provider, 'model': model,
                                'records': list(records), 'summary': summary},
                               indent=1), encoding='utf-8')
