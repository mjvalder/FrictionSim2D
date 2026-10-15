"""Where the assistant may read, and where an applied change may write.

Containment is checked *lexically* (normalised absolute paths, symlinks not
followed): a workspace is the user's own choice, and results symlinked from it
to scratch or a mount are legitimate, while `..` and absolute paths elsewhere
are not.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

PathLike = Union[str, Path]


class SandboxError(ValueError):
    """A path the assistant may not read or write."""


@dataclass(frozen=True)
class SearchRoots:
    """The folders the assistant may read, in search order."""

    workspace: Path
    examples: Optional[Path] = None

    def tiers(self) -> List[Tuple[str, Path]]:
        """``(label, root)`` pairs: the workspace first, then bundled examples."""
        tiers = [('workspace', Path(self.workspace))]
        if self.examples is not None:
            tiers.append(('examples', Path(self.examples)))
        return tiers


def find_examples_dir() -> Optional[Path]:
    """The repository's examples/ folder, when running from a checkout."""
    candidate = Path(__file__).resolve().parents[2] / 'examples'
    return candidate if candidate.is_dir() else None


def _lexical(path: PathLike) -> Path:
    return Path(os.path.normpath(os.path.abspath(str(path))))


def _within(path: PathLike, root: PathLike) -> bool:
    try:
        _lexical(path).relative_to(_lexical(root))
        return True
    except ValueError:
        return False


def within_roots(path: PathLike, roots: SearchRoots) -> bool:
    """Whether ``path`` lies inside any readable root."""
    return any(_within(path, root) for _, root in roots.tiers())


def _from_display(path: str, roots: SearchRoots) -> Optional[Path]:
    """Turn ``workspace:/a/b`` (what tools show) back into a real path."""
    if ':/' not in path:
        return None
    label, rest = path.split(':/', 1)
    base = dict(roots.tiers()).get(label)
    return base / rest if base is not None else None


def resolve_readable(path: PathLike, roots: SearchRoots) -> Path:
    """An existing file or folder inside the roots; relative paths try each tier."""
    text = str(path)
    raw = _from_display(text, roots) or Path(text).expanduser()
    candidates = [raw] if raw.is_absolute() else [root / raw for _, root in roots.tiers()]
    for candidate in candidates:
        if within_roots(candidate, roots) and candidate.exists():
            return _lexical(candidate)
    raise SandboxError(f'{text}: not found inside the workspace or the bundled examples')


def check_write_target(path: PathLike, workspace: PathLike) -> Path:
    """A new path inside the workspace, outside dot-folders, not yet existing."""
    workspace = _lexical(workspace)
    target = Path(str(path)).expanduser()
    target = _lexical(target if target.is_absolute() else workspace / target)
    if not _within(target, workspace) or target == workspace:
        raise SandboxError(f'{path}: outside the workspace')
    relative = target.relative_to(workspace)
    if any(part.startswith('.') for part in relative.parts):
        raise SandboxError(f'{path}: dot-folders such as .preview are off limits')
    if target.exists():
        raise SandboxError(f'{path}: already exists (nothing is overwritten)')
    return target


def display_path(path: PathLike, roots: SearchRoots) -> str:
    """``workspace:/rel`` or ``examples:/rel`` -- never an absolute path."""
    for label, root in roots.tiers():
        if _within(path, root):
            relative = _lexical(path).relative_to(_lexical(root)).as_posix()
            return f'{label}:/' if relative == '.' else f'{label}:/{relative}'
    return '(outside the allowed folders)'
