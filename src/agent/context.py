"""What every tool receives: snapshots of the Builder's documents plus roots."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..core.config import GlobalSettings
from ..gui.controllers.document import ConfigDocument
from ..gui.controllers.settings_document import SettingsDocument
from .sandbox import SearchRoots


@dataclass
class AgentDeps:
    """Per-turn context. ``document``/``settings_document`` are deep copies taken
    on the GUI thread, so the worker thread never races the form; tools never
    mutate them (they build change sets on further copies)."""

    document: ConfigDocument
    settings_document: SettingsDocument
    settings: GlobalSettings
    roots: SearchRoots
    pending: List[Any] = field(default_factory=list)
    on_proposal: Optional[Callable[[Any], None]] = None
    material_index: Any = None

    def propose(self, change: Any) -> str:
        """Record a change set for the user, notify the pane, return its handle."""
        self.pending.append(change)
        if self.on_proposal is not None:
            self.on_proposal(change)
        return change.handle()


def snapshot_deps(document: ConfigDocument, settings_document: SettingsDocument,
                  settings: GlobalSettings, roots: SearchRoots) -> AgentDeps:
    """Deps for one turn, holding copies of the live documents."""
    return AgentDeps(document=copy.deepcopy(document),
                     settings_document=copy.deepcopy(settings_document),
                     settings=settings, roots=roots)
