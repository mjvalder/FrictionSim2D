"""The working directory the application operates in.

A workspace is an output root: generated packages, saved configs and the
disposable preview area live under it. Configs elsewhere on disk can still be
opened -- the workspace is where new work lands, not a container you must move
things into.

Qt-free.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = ['Workspace', 'RecentWorkspaces']

#: Preview builds are disposable; keep only the most recent few.
PREVIEW_KEEP = 8

_STATE_FILE = '.frictionsim2d-workspace.json'


@dataclass
class Workspace:
    """An output root plus its scratch areas."""

    root: Path
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()

    # ------------------------------------------------------------ layout

    @property
    def runs_dir(self) -> Path:
        """Where generated simulation packages go."""
        return self.root

    @property
    def configs_dir(self) -> Path:
        """Where configs authored here are saved by default."""
        return self.root / 'configs'

    @property
    def preview_dir(self) -> Path:
        """Disposable preview builds, keyed by config hash."""
        return self.root / '.preview'

    @property
    def state_path(self) -> Path:
        return self.root / _STATE_FILE

    @property
    def settings_path(self) -> Path:
        """Workspace-local settings.yaml.

        Always written complete: ``load_settings`` replaces rather than merges,
        so a partial file would silently reset every value it omitted to a
        built-in default.
        """
        return self.root / 'settings.yaml'

    # ------------------------------------------------------------ lifecycle

    @classmethod
    def open(cls, root: Union[str, Path]) -> 'Workspace':
        """Open (creating if needed) a workspace and load its state."""
        workspace = cls(Path(root))
        workspace.root.mkdir(parents=True, exist_ok=True)
        workspace.configs_dir.mkdir(exist_ok=True)
        workspace.preview_dir.mkdir(exist_ok=True)
        workspace._load_state()
        workspace.prune_previews()
        return workspace

    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                self.state = {}

    def save_state(self) -> None:
        """Persist workspace state; never fatal if it cannot be written."""
        try:
            self.state_path.write_text(
                json.dumps(self.state, indent=2), encoding='utf-8')
        except OSError:
            pass

    def remember(self, key: str, value: Any) -> None:
        self.state[key] = value
        self.save_state()

    # -------------------------------------------------------------- preview

    def preview_path(self, key: str) -> Path:
        """Directory for a preview build with the given cache key."""
        return self.preview_dir / key

    def prune_previews(self, keep: int = PREVIEW_KEEP) -> List[Path]:
        """Delete all but the newest preview builds.

        Returns:
            The directories removed.
        """
        if not self.preview_dir.exists():
            return []

        # `_db` (the database pane's query cache) and `_plots` (the Plots
        # tab's rendered figures) live beside the preview builds and are not
        # builds: pruning to the newest few would throw them away after eight
        # renders. Underscore-prefixed entries are scratch areas owned by a
        # controller, and are never pruned.
        entries = sorted(
            (p for p in self.preview_dir.iterdir()
             if p.is_dir() and not p.name.startswith('_')),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        removed = []
        for stale in entries[keep:]:
            shutil.rmtree(stale, ignore_errors=True)
            removed.append(stale)
        return removed

    # ----------------------------------------------------------------- misc

    def default_config_path(self, model: str) -> Path:
        """A non-colliding default filename for a new config."""
        candidate = self.configs_dir / f"{model}.ini"
        index = 2
        while candidate.exists():
            candidate = self.configs_dir / f"{model}_{index}.ini"
            index += 1
        return candidate

    def __str__(self) -> str:
        return str(self.root)


class RecentWorkspaces:
    """The list of recently opened workspaces, stored in the user's config dir."""

    def __init__(self, path: Optional[Path] = None, limit: int = 10) -> None:
        self.path = path or (
            Path.home() / '.config' / 'FrictionSim2D' / 'recent_workspaces.json'
        )
        self.limit = limit

    def load(self) -> List[Path]:
        """Recent workspaces, newest first, excluding any that have vanished."""
        try:
            entries = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return []
        return [Path(e) for e in entries if Path(e).is_dir()]

    def add(self, root: Union[str, Path]) -> List[Path]:
        """Record a workspace as most recent."""
        root = Path(root).expanduser().resolve()
        entries = [p for p in self.load() if p != root]
        entries.insert(0, root)
        entries = entries[:self.limit]

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([str(p) for p in entries], indent=2), encoding='utf-8')
        except OSError:
            pass
        return entries

    def most_recent(self) -> Optional[Path]:
        entries = self.load()
        return entries[0] if entries else None
