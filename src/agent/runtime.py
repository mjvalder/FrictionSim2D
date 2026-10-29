"""The agent loop: pydantic-ai Agent, tool wrappers, events and history.

All pydantic-ai *run* APIs live here (spec §3); `providers.py` builds models.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, List, Optional

import httpx
from pydantic_ai import (Agent, AgentRunResultEvent, CancellationToken,
                         ModelRetry, RunContext, capture_run_messages)
from pydantic_ai.exceptions import (ModelAPIError, ModelHTTPError, RunCancelled,
                                    UnexpectedModelBehavior, UsageLimitExceeded)
from pydantic_ai.messages import (FunctionToolCallEvent, FunctionToolResultEvent,
                                  ModelRequest, ModelResponse, PartDeltaEvent,
                                  PartStartEvent, TextPart, TextPartDelta,
                                  UserPromptPart)
from pydantic_ai.usage import UsageLimits

from .changes import ToolInputError
from .context import AgentDeps
from .events import TurnEvent
from .prompts import build_system_prompt, estimate_tokens
from .providers import ProviderUnavailable, build_model, run_settings
from .sandbox import SandboxError, redact
from .settings import AgentSettings
from .tools import analysis, compact, materials, plots, setup, workspace
from .tools import knowledge, materials_db

logger = logging.getLogger(__name__)

HISTORY_BUDGET = {'ollama': 6000, 'gemini': 32000, 'anthropic': 32000}
_RETRY_IN = re.compile(r'retry in ([\d.]+)\s*s', re.IGNORECASE)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s")


def _call(ctx: RunContext[AgentDeps], func: Callable, *args, **kwargs) -> str:
    """Run a plain tool: bad input becomes a retry hint, a crash a short error.

    Results and retry hints are redacted: no absolute path reaches the model.
    """
    try:
        return compact(redact(func(ctx.deps, *args, **kwargs), ctx.deps.roots))
    except (ToolInputError, SandboxError) as exc:
        raise ModelRetry(redact(str(exc), ctx.deps.roots)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception('assistant tool %s failed', func.__name__)
        return f'error in {func.__name__}: {type(exc).__name__}'


def _register_tools(agent: Agent) -> None:  # pylint: disable=too-many-locals
    @agent.tool
    def list_modes(ctx: RunContext[AgentDeps]) -> str:
        """Simulation families and modes, with the model each builds."""
        return _call(ctx, setup.list_modes)

    @agent.tool
    def explain_current(ctx: RunContext[AgentDeps]) -> str:
        """Summarise the config open in the Builder and its validation issues."""
        return _call(ctx, setup.explain_current)

    @agent.tool
    def find_parameter(ctx: RunContext[AgentDeps], query: str) -> str:
        """Search config and settings parameters by name or description."""
        return _call(ctx, setup.find_parameter, query)

    @agent.tool
    def set_fields(ctx: RunContext[AgentDeps], changes: List[setup.FieldChange]) -> str:
        """Propose values for parameters found with find_parameter."""
        return _call(ctx, setup.set_fields, changes)

    @agent.tool
    def make_sweep(ctx: RunContext[AgentDeps], start: float, stop: float, n: int,
                   spacing: str = 'linear') -> str:
        """n evenly spaced values from start to stop ('linear' or 'log')."""
        return _call(ctx, setup.make_sweep_tool, start, stop, n, spacing)

    @agent.tool
    def suggest_values(ctx: RunContext[AgentDeps], quantity: str, descriptor: str,
                       n: int) -> str:
        """Turn a descriptor such as 'low load' into values, when a rule exists."""
        return _call(ctx, setup.suggest_values, quantity, descriptor, n)

    @agent.tool
    def setup_afm(ctx: RunContext[AgentDeps], request: setup.AFMSetupRequest) -> str:
        """Propose an AFM tip sliding simulation."""
        return _call(ctx, setup.setup_afm, request)

    @agent.tool
    def setup_sheet(ctx: RunContext[AgentDeps], request: setup.SheetSetupRequest) -> str:
        """Propose a sheet-on-sheet sliding simulation."""
        return _call(ctx, setup.setup_sheet, request)

    @agent.tool
    def setup_indent(ctx: RunContext[AgentDeps], request: setup.IndentSetupRequest) -> str:
        """Propose an indentation (hold and retract) simulation."""
        return _call(ctx, setup.setup_indent, request)

    @agent.tool
    def setup_pes_tip(ctx: RunContext[AgentDeps], request: setup.PESTipSetupRequest) -> str:
        """Propose a static PES scan of the AFM tip over a sheet."""
        return _call(ctx, setup.setup_pes_tip, request)

    @agent.tool
    def setup_pes_sheet(ctx: RunContext[AgentDeps],
                        request: setup.PESSheetSetupRequest) -> str:
        """Propose a static PES scan of one sheet over another."""
        return _call(ctx, setup.setup_pes_sheet, request)

    @agent.tool
    def search_structures(ctx: RunContext[AgentDeps], query: str) -> str:
        """Find structure (CIF) files by material name or formula."""
        return _call(ctx, materials.search_structures, query)

    @agent.tool
    def search_potentials(ctx: RunContext[AgentDeps], material: str,
                          potential_type: Optional[str] = None) -> str:
        """Find interatomic potential files for a material."""
        return _call(ctx, materials.search_potentials, material, potential_type)

    @agent.tool
    def search_knowledge(ctx: RunContext[AgentDeps], query: str) -> str:
        """Search the local 2D-materials and tribology corpus; cite what it returns."""
        return _call(ctx, knowledge.search_knowledge, query)

    @agent.tool
    def lattice_mismatch(ctx: RunContext[AgentDeps], material_a: str,
                         material_b: Optional[str] = None) -> str:
        """In-plane lattice constants of a material; for two, their mismatch and
        smallest coincidence cell."""
        return _call(ctx, materials.lattice_mismatch, material_a, material_b)

    @agent.tool
    def workspace_overview(ctx: RunContext[AgentDeps], depth: int = 2) -> str:
        """List the workspace's folders."""
        return _call(ctx, workspace.workspace_overview, depth)

    @agent.tool
    def describe_run(ctx: RunContext[AgentDeps], path: str) -> str:
        """Describe what a workspace folder holds."""
        return _call(ctx, workspace.describe_run, path)

    @agent.tool
    def propose_mkdir(ctx: RunContext[AgentDeps], path: str) -> str:
        """Propose creating a folder in the workspace."""
        return _call(ctx, workspace.propose_mkdir, path)

    @agent.tool
    def propose_copy_in(ctx: RunContext[AgentDeps], source: str,
                        dest_dir: str = 'materials') -> str:
        """Propose copying a found structure or potential file into the workspace."""
        return _call(ctx, workspace.propose_copy_in, source, dest_dir)

    @agent.tool
    def find_result_sets(ctx: RunContext[AgentDeps]) -> str:
        """List result datasets (outputs/output_full_*.json) and simulation roots."""
        return _call(ctx, analysis.find_result_sets)

    @agent.tool
    def summarise_results(ctx: RunContext[AgentDeps], path: str,
                          material: Optional[str] = None, size: Optional[str] = None,
                          max_rows: int = 15) -> str:
        """Status counts and a run table for a dataset or simulation root."""
        return _call(ctx, analysis.summarise_results, path, material, size, max_rows)

    @agent.tool
    def plot_types(ctx: RunContext[AgentDeps]) -> str:
        """The plot types, one line each."""
        return _call(ctx, analysis.plot_types)

    @agent.tool
    def plot_schema(ctx: RunContext[AgentDeps], plot_type: str) -> str:
        """The fields one plot type accepts."""
        return _call(ctx, analysis.plot_schema, plot_type)

    @agent.tool
    def propose_plot(ctx: RunContext[AgentDeps], request: plots.PlotProposalRequest) -> str:
        """Propose a plot for the Plots tab on datasets from find_result_sets; fields
        are key/value pairs from plot_schema."""
        return _call(ctx, plots.propose_plot, request)

    @agent.tool
    def query_materials_project(ctx: RunContext[AgentDeps], formula: str) -> str:
        """Materials Project entries (stability, band gap, moduli) for a formula."""
        return _call(ctx, materials_db.query_materials_project, formula)

    @agent.tool
    def query_2dmatpedia(ctx: RunContext[AgentDeps], formula: str) -> str:
        """2DMatPedia monolayers for a formula, with their dm_id."""
        return _call(ctx, materials_db.query_2dmatpedia, formula)

    @agent.tool
    def fetch_2dmatpedia_structure(ctx: RunContext[AgentDeps], dm_id: str,
                                   dest_dir: str = 'materials') -> str:
        """Propose saving a 2DMatPedia structure (dm_id) as a CIF in the workspace."""
        return _call(ctx, materials_db.fetch_2dmatpedia_structure, dm_id, dest_dir)


def _retry_after(exc: ModelHTTPError) -> float:
    text = str(exc.body)
    for pattern in (_RETRY_IN, _RETRY_DELAY):
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return 30.0


def _text_only(messages: List[Any]) -> List[Any]:
    """User prompts and assistant text, without tool calls (provider-neutral)."""
    out = []
    for message in messages:
        if isinstance(message, ModelRequest):
            parts = [p for p in message.parts if isinstance(p, UserPromptPart)]
            if parts:
                out.append(ModelRequest(parts=parts))
        elif isinstance(message, ModelResponse):
            parts = [p for p in message.parts if isinstance(p, TextPart) and p.content]
            if parts:
                out.append(ModelResponse(parts=parts))
    return out


def _latest_turn_start(messages: List[Any]) -> int:
    """Index of the request carrying the latest user prompt (0 if none)."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelRequest) and any(
                isinstance(p, UserPromptPart) for p in message.parts):
            return index
    return 0


def _unreachable(exc: BaseException) -> bool:
    """Whether ``exc`` (or what caused it) is a refused or failed connection."""
    seen = 0
    while exc is not None and seen < 5:
        if isinstance(exc, (httpx.ConnectError, ConnectionError)) \
                or type(exc).__name__ == 'APIConnectionError':
            return True
        exc, seen = exc.__cause__ or exc.__context__, seen + 1
    return False


class AgentSession:
    """One chat: the agent, its history, and how a turn is run."""

    def __init__(self, settings: AgentSettings, model: Any = None,
                 history_budget: Optional[int] = None) -> None:
        self.settings = settings
        self._model = model
        self._history_budget = history_budget  # tokens; None: per provider
        self._agent: Optional[Agent] = None
        self.history: List[Any] = []
        self._generation = 0  # bumped by new_chat; a turn from before it keeps out

    def _build_agent(self) -> Agent:
        model = self._model if self._model is not None else build_model(self.settings)
        agent = Agent(model, deps_type=AgentDeps, retries=2,
                      instructions=build_system_prompt(self.settings.provider,
                                                       self.settings.ollama.model))
        _register_tools(agent)
        return agent

    @property
    def agent(self) -> Agent:
        """The pydantic-ai agent, built on first use (so a missing key surfaces then)."""
        if self._agent is None:
            self._agent = self._build_agent()
        return self._agent

    def tool_names(self) -> List[str]:
        """Names of the registered tools."""
        return list(self.agent._function_toolset.tools)  # pylint: disable=protected-access

    @staticmethod
    def new_token() -> CancellationToken:
        """A fresh token for one turn; ``cancel()`` is thread-safe."""
        return CancellationToken()

    def new_chat(self) -> None:
        """Forget the conversation; a turn still running will not write it back."""
        self._generation += 1  # first: a turn finishing in between must not write back
        self.history = []

    def switch(self, settings: AgentSettings) -> None:
        """Change provider/model; keep only text so formats cannot clash."""
        self.settings = settings
        self._agent = None
        self._model = None
        self.history = _text_only(self.history)

    def _trimmed_history(self) -> List[Any]:
        """Append-only history; once over budget, collapse the earlier turns.

        The latest turn stays intact (tool calls and results too), so a turn
        stopped at the step cap can be carried on with 'continue'.
        """
        budget = self._history_budget or HISTORY_BUDGET.get(self.settings.provider, 32000)
        if estimate_tokens(repr(self.history)) <= budget:
            return self.history
        start = _latest_turn_start(self.history)
        if start == 0:
            return self.history
        earlier, latest = self.history[:start], self.history[start:]
        lines = []
        for message in _text_only(earlier):
            for part in message.parts:
                who = 'User' if isinstance(part, UserPromptPart) else 'Assistant'
                lines.append(f'{who}: {str(part.content)[:200]}')
        self.history = [ModelRequest(parts=[UserPromptPart(
                            content='Summary of the earlier conversation:\n'
                            + '\n'.join(lines[-30:]))]),
                        ModelResponse(parts=[TextPart(content='Understood.')])] + latest
        return self.history

    def _dispatch(self, event: Any, emit: Callable[[TurnEvent], None],
                  generation: int) -> None:
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            if event.part.content:
                emit(TurnEvent('text', event.part.content))
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            emit(TurnEvent('text', event.delta.content_delta))
        elif isinstance(event, FunctionToolCallEvent):
            emit(TurnEvent('tool_call', f'{event.part.tool_name}({str(event.part.args)[:120]})'))
        elif isinstance(event, FunctionToolResultEvent):
            name = getattr(event.part, 'tool_name', 'tool')
            emit(TurnEvent('tool_result', f'{name}: {str(event.part.content)[:200]}'))
        elif isinstance(event, AgentRunResultEvent):
            if generation == self._generation:
                self.history = event.result.all_messages()
            emit(TurnEvent('done', payload=event.result.usage.requests))

    async def _stream(self, text: str, deps: AgentDeps, emit: Callable[[TurnEvent], None],
                      token: CancellationToken, generation: int, limit: int) -> None:
        async with self.agent.run_stream_events(
                text, deps=deps, message_history=self._trimmed_history(),
                cancellation_token=token, model_settings=run_settings(self.settings),
                usage_limits=UsageLimits(request_limit=limit)) as events:
            async for event in events:
                self._dispatch(event, emit, generation)

    async def run_turn(self, text: str, deps: AgentDeps,
                       emit: Callable[[TurnEvent], None], token: CancellationToken) -> None:
        """Run one user turn, emitting events; never raises for provider trouble."""
        deps.on_proposal = lambda change: emit(TurnEvent('proposal', payload=change))
        deps.cancelled = lambda: token.cancelled
        generation = self._generation
        limit = self.settings.max_requests_per_turn
        messages: List[Any] = []
        try:
            with capture_run_messages() as messages:
                await self._stream(text, deps, emit, token, generation, limit)
        except UsageLimitExceeded:
            # Keep the capped turn so 'continue' picks up where it stopped.
            if generation == self._generation and messages:
                self.history = list(messages)
            emit(TurnEvent('error', f'Stopped after {limit} steps to save quota — type '
                                    "'continue' to carry on.", payload='step_limit'))
        except ProviderUnavailable as exc:
            emit(TurnEvent('error', str(exc), payload='unavailable'))
        except RunCancelled:
            emit(TurnEvent('error', 'Stopped.', payload='cancelled'))
        except ModelHTTPError as exc:
            if exc.status_code in (429, 503):
                delay = _retry_after(exc)
                emit(TurnEvent('error', f'The provider is rate-limited or busy; retry in '
                                        f'{delay:.0f} s.', payload='rate_limited',
                               retry_after=delay))
            else:
                emit(TurnEvent('error', f'Provider error {exc.status_code}: {exc}',
                               payload='provider'))
        except UnexpectedModelBehavior as exc:
            emit(TurnEvent('error', f'The assistant could not finish this ({exc}). Any '
                                    'proposal above still stands; finish the rest in the '
                                    'Builder form.', payload='gave_up'))
        except ModelAPIError as exc:
            if _unreachable(exc):
                emit(TurnEvent('error', f'Could not reach the {self.settings.provider} '
                                        f'server ({exc}).', payload='unreachable'))
            else:
                emit(TurnEvent('error', f'Provider error: {exc}', payload='provider'))
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception('assistant turn failed unexpectedly')
            emit(TurnEvent('error', f'{type(exc).__name__}: {exc}', payload='internal'))
