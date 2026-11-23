"""Events a turn emits to the GUI (no pydantic-ai, so fakes and Qt can use them)."""
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class TurnEvent:
    """One thing that happened during a turn.

    A ``tool_result``'s ``text`` is cut for display; its ``payload`` is the whole result.
    """

    kind: Literal['text', 'tool_call', 'tool_result', 'proposal', 'done', 'error']
    text: str = ''
    payload: Any = None
    retry_after: float = 0.0
