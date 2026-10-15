"""Runs agent turns on a QThread with its own asyncio loop."""
from __future__ import annotations

import asyncio
import logging

from PySide6.QtCore import QObject, Signal, Slot

from ...agent.events import TurnEvent

logger = logging.getLogger(__name__)


class AgentWorker(QObject):
    """Lives on a worker thread; emits `TurnEvent`s back to the GUI thread."""

    event = Signal(object)
    finished = Signal()

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self._token = None

    def arm(self, token) -> None:
        """Store the turn's token (GUI thread, before the turn is queued)."""
        self._token = token

    @property
    def busy(self) -> bool:
        """True from `arm` until the turn ends."""
        return self._token is not None

    @Slot(str, object, object)
    def run_turn(self, text: str, deps, token) -> None:
        """One turn, start to finish; always emits `finished`."""
        try:
            asyncio.run(self.session.run_turn(text, deps, self.event.emit, token))
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception('assistant turn failed')
            self.event.emit(TurnEvent('error', f'Internal error: {exc}', payload='internal'))
        finally:
            self._token = None
            self.finished.emit()

    def cancel(self) -> None:
        """Called from the GUI thread; the token is thread-safe."""
        token = self._token
        if token is not None:
            token.cancel()
