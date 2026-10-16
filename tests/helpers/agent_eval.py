"""The eval tier's machinery: env switches, sessions, one measured turn, results.

Used by ``tests/agent/evals`` (the live-model cases and their conftest plugin)
and by ``tests/agent/test_eval_plugin.py``. pydantic-ai is imported inside the
functions that need it, so the aggregation helpers stay importable without it.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO / 'tests' / 'agent' / 'evals' / '.results'
#: The scripted provider the plugin's own test uses; never a real model.
FUNCTION_PROVIDER = 'function'
FUNCTION_REPLY = 'Could you tell me which one you mean?'


@dataclass(frozen=True)
class EvalEnv:
    """The env switches: AGENT_EVAL_PROVIDER, _MODEL, _REPEATS, _PAUSE, _RESULTS_DIR."""

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
    return defaults.ollama.model if env.provider == 'ollama' else defaults.gemini.model


def build_session(env: EvalEnv):
    """An `AgentSession` for ``env``; raises `ProviderUnavailable` when it cannot run."""
    # pylint: disable=import-outside-toplevel
    from src.agent.providers import ProviderUnavailable, build_model, ollama_status
    from src.agent.runtime import AgentSession
    from src.agent.settings import AgentSettings
    if env.provider == FUNCTION_PROVIDER:
        from pydantic_ai.models.function import FunctionModel
        return AgentSession(AgentSettings(), model=FunctionModel(
            stream_function=_ask_back, model_name=FUNCTION_PROVIDER))
    if env.provider not in ('gemini', 'ollama'):
        raise ProviderUnavailable(f'unknown AGENT_EVAL_PROVIDER {env.provider!r}')
    settings = AgentSettings(provider=env.provider)
    if env.model:
        if env.provider == 'ollama':
            settings.ollama.model = env.model
        else:
            settings.gemini.model = env.model
    if env.provider == 'ollama':
        ok, message = ollama_status(settings)
        if not ok:
            raise ProviderUnavailable(message)
        return AgentSession(settings)
    return AgentSession(settings, model=build_model(settings))


# ---- one measured turn -------------------------------------------------------

@dataclass
class EvalResult:
    """What one turn produced, for a case's check."""

    deps: Any
    events: List[Any] = field(default_factory=list)
    latency_s: float = 0.0
    requests: int = 0
    tokens: int = 0

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


def turn_tokens(history: Sequence[Any]) -> int:
    """Input + output tokens of the model responses in the latest turn."""
    # pylint: disable=import-outside-toplevel
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
    start = 0
    for index, message in enumerate(history):
        if isinstance(message, ModelRequest) and any(
                isinstance(p, UserPromptPart) for p in message.parts):
            start = index
    total = 0
    for message in history[start:]:
        if isinstance(message, ModelResponse) and message.usage is not None:
            total += (message.usage.input_tokens or 0) + (message.usage.output_tokens or 0)
    return total


def run_eval_turn(session, prompt: str, deps) -> EvalResult:
    """Run one turn of ``session`` on ``deps``, timing it and counting its usage."""
    result = EvalResult(deps=deps)
    start = time.perf_counter()
    asyncio.run(session.run_turn(prompt, deps, result.events.append, session.new_token()))
    result.latency_s = time.perf_counter() - start
    done = [e for e in result.events if e.kind == 'done']
    result.requests = int(done[-1].payload or 0) if done else 0
    result.tokens = turn_tokens(session.history) if done else 0
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


def summarise(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass rate per case and overall (skips excluded), latency and usage."""
    cases: Dict[str, Dict[str, Any]] = {}
    for record in records:
        row = cases.setdefault(record['case'], {'runs': 0, 'passed': 0, 'failed': 0,
                                                'skipped': 0})
        row['runs'] += 1
        row[record['outcome']] += 1
    for row in cases.values():
        row['pass_rate'] = _rate(row['passed'], row['failed'])
    passed = sum(r['passed'] for r in cases.values())
    failed = sum(r['failed'] for r in cases.values())
    ran = [r for r in records if r['outcome'] != 'skipped']
    latencies = [r['latency_s'] for r in ran]
    return {'cases': dict(sorted(cases.items())),
            'overall': {'runs': len(records), 'passed': passed, 'failed': failed,
                        'skipped': len(records) - len(ran), 'pass_rate': _rate(passed, failed)},
            'latency_p50_s': percentile(latencies, 50),
            'latency_p95_s': percentile(latencies, 95),
            'mean_requests': _mean([r['requests'] for r in ran]),
            'mean_tokens': _mean([r['tokens'] for r in ran])}


def _pct(rate: Optional[float]) -> str:
    return '-' if rate is None else f'{rate:.0%}'


def _num(value: Optional[float], spec: str) -> str:
    return '-' if value is None else format(value, spec)


def format_table(summary: Dict[str, Any], provider: str, model: str) -> str:
    """The session-end table: one line per case, then the totals."""
    width = max([len('overall')] + [len(name) for name in summary['cases']])
    lines = [f'agent eval: {provider} / {model}',
             f"{'case':<{width}}  {'pass':>5}  {'runs':>4}  {'skip':>4}"]
    rows = list(summary['cases'].items()) + [('overall', summary['overall'])]
    for name, row in rows:
        lines.append(f"{name:<{width}}  {_pct(row['pass_rate']):>5}  {row['runs']:>4}  "
                     f"{row['skipped']:>4}")
    lines.append(f"latency p50 {_num(summary['latency_p50_s'], '.1f')} s, "
                 f"p95 {_num(summary['latency_p95_s'], '.1f')} s; per turn: "
                 f"{_num(summary['mean_requests'], '.1f')} requests, "
                 f"{_num(summary['mean_tokens'], '.0f')} tokens")
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
