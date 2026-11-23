"""One asyncio event loop per worker, reused for every turn.

The model's async HTTP client is cached on the agent, so its pooled
connections belong to the loop that opened them. ``asyncio.run`` per turn
closed that loop after turn 1, and turn 2 failed on the stale pool with
'Event loop is closed'. A `TurnRunner` keeps one loop for its lifetime.
Qt-free, so it is testable without a GUI.
"""
from __future__ import annotations

import asyncio
from typing import Any, Coroutine, Optional


class TurnRunner:
    """Runs coroutines on one event loop, made lazily in the calling thread.

    Use it from a single thread (the worker's): the loop is created there on
    the first `run` and should be closed there with `close`.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run ``coro`` to completion on the runner's loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(coro)

    def close(self) -> None:
        """Cancel leftover tasks and close the loop; safe to call twice."""
        loop, self._loop = self._loop, None
        if loop is None or loop.is_closed():
            return
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
