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
    #: Whether the GUI applies config/settings change sets as they arrive
    #: (the default) or shows them as cards first; it changes what the model
    #: is told about a proposal.
    direct_apply: bool = False
    #: The Plots tab's document as a dict (a copy taken on the GUI thread),
    #: so plot tools know which datasets are loaded; None without a Plots tab.
    plot_document: Optional[dict] = None
    #: Builds the Materials Project / 2DMatPedia client from a ``should_stop``
    #: keyword (raising
    #: `MaterialsDBError` when off or without a key). The GUI passes
    #: `materials_db.default_materials_db`; tests a fake; None: unavailable.
    materials_db: Optional[Callable[..., Any]] = None
    #: Whether the user pressed Stop on this turn; `AgentSession.run_turn`
    #: points it at the turn's token. Long downloads check it between requests.
    cancelled: Callable[[], bool] = field(default=lambda: False)

    def propose(self, change: Any) -> str:
        """Record a change set for the user, notify the pane, return its handle.

        In direct-apply mode the snapshot is moved on to the change's preview,
        so later tools in the same turn read -- and later change sets are
        fingerprinted against -- what the GUI is about to hold. If the GUI
        refuses it after all, its stale re-check catches the mismatch.
        """
        self.pending.append(change)
        preview = getattr(change, 'preview', None)
        if self.direct_apply and preview is not None:
            self.document, self.settings_document = copy.deepcopy(preview)
        plot_preview = getattr(change, 'plot_preview', None)
        if self.direct_apply and plot_preview is not None:
            self.plot_document = copy.deepcopy(plot_preview)
        if self.on_proposal is not None:
            self.on_proposal(change)
        return change.handle(direct=self.direct_apply)


def snapshot_deps(document: ConfigDocument, settings_document: SettingsDocument,
                  settings: GlobalSettings, roots: SearchRoots,
                  plot_document: Optional[dict] = None,
                  materials_db: Optional[Callable[..., Any]] = None) -> AgentDeps:
    """Deps for one turn, holding copies of the live documents."""
    return AgentDeps(document=copy.deepcopy(document),
                     settings_document=copy.deepcopy(settings_document),
                     settings=settings, roots=roots,
                     plot_document=copy.deepcopy(plot_document),
                     materials_db=materials_db)
