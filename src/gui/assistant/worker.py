"""Runs agent turns on a QThread with its own asyncio loop."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal, Slot

from ...agent.events import TurnEvent
from ...agent.loop import TurnRunner

logger = logging.getLogger(__name__)


class AgentWorker(QObject):
    """Lives on a worker thread; emits `TurnEvent`s back to the GUI thread.

    Every turn runs on one event loop (`TurnRunner`), made on the first turn
    and closed by `close` when the thread finishes: the model's HTTP client
    keeps connections bound to it between turns.
    """

    event = Signal(int, object)  # (turn id, TurnEvent)
    finished = Signal(int)

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self._token = None
        self._runner = TurnRunner()

    def arm(self, token) -> None:
        """Store the turn's token (GUI thread, before the turn is queued)."""
        self._token = token

    @property
    def busy(self) -> bool:
        """True from `arm` until the turn ends."""
        return self._token is not None

    @Slot(str, object, object, int)
    def run_turn(self, text: str, deps, token, turn_id: int) -> None:
        """One turn, start to finish; always emits `finished`. Every signal
        carries ``turn_id`` so the pane can drop a turn it has moved past."""
        def emit(event) -> None:
            self.event.emit(turn_id, event)
        try:
            self._runner.run(self.session.run_turn(text, deps, emit, token))
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception('assistant turn failed')
            emit(TurnEvent('error', f'Internal error: {exc}', payload='internal'))
        finally:
            self._token = None
            self.finished.emit(turn_id)

    def cancel(self) -> None:
        """Called from the GUI thread; the token is thread-safe."""
        token = self._token
        if token is not None:
            token.cancel()

    @Slot()
    def close(self) -> None:
        """Close the turn loop; connect directly to the thread's ``finished``."""
        try:
            self._runner.close()
        except Exception:  # pylint: disable=broad-except
            logger.exception('closing the assistant event loop failed')
