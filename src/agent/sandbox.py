"""Where the assistant may read, and where an applied change may write.

Containment is checked *lexically* (normalised absolute paths, symlinks not
followed): a workspace is the user's own choice, and results symlinked from it
to scratch or a mount are legitimate, while `..` and absolute paths elsewhere
are not.
"""
from __future__ import annotations

import os
import re
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


_DISPLAY = re.compile(r'^([A-Za-z]\w*):/(.*)$', re.DOTALL)


def from_display(path: str, roots: SearchRoots) -> Optional[Path]:
    """Turn ``workspace:/a/b`` (what tools show) back into a real path."""
    match = _DISPLAY.match(path)
    if match is None:
        return None
    base = dict(roots.tiers()).get(match.group(1))
    return base / match.group(2) if base is not None else None


def write_path(path: PathLike, roots: SearchRoots) -> Path:
    """A write destination as a real path: ``workspace:/x`` maps back, and any
    other ``label:/`` (the read-only examples) is refused. Containment is left
    to `check_write_target`."""
    text = str(path)
    match = _DISPLAY.match(text)
    if match is None:
        return Path(text)
    if match.group(1) != 'workspace':
        raise SandboxError(f'{text}: only workspace:/ paths can be written; '
                           f'{match.group(1)}:/ is read-only')
    return Path(roots.workspace) / match.group(2)


def resolve_readable(path: PathLike, roots: SearchRoots) -> Path:
    """An existing file or folder inside the roots; relative paths try each tier."""
    text = str(path)
    raw = from_display(text, roots) or Path(text).expanduser()
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
    if os.path.lexists(target):
        raise SandboxError(f'{path}: already exists (nothing is overwritten)')
    _check_real_parent(path, target, workspace)
    return target


def _check_real_parent(path: PathLike, target: Path, workspace: Path) -> None:
    """Refuse writes that reach outside the workspace or a dot-folder via symlinks."""
    ancestor = target.parent
    while not os.path.lexists(ancestor):
        ancestor = ancestor.parent
    real_ws = Path(os.path.realpath(workspace))
    real_ancestor = Path(os.path.realpath(ancestor))
    try:
        real_rel = real_ancestor.relative_to(real_ws)
    except ValueError:
        raise SandboxError(f'{path}: outside the workspace') from None
    rest = target.relative_to(ancestor).parts
    if any(part.startswith('.') for part in (*real_rel.parts, *rest)):
        raise SandboxError(f'{path}: dot-folders such as .preview are off limits')


def display_path(path: PathLike, roots: SearchRoots) -> str:
    """``workspace:/rel`` or ``examples:/rel`` -- never an absolute path."""
    for label, root in roots.tiers():
        if _within(path, root):
            relative = _lexical(path).relative_to(_lexical(root)).as_posix()
            return f'{label}:/' if relative == '.' else f'{label}:/{relative}'
    return '(outside the allowed folders)'


def _path_pattern(root: str) -> re.Pattern:
    """``root`` as a whole path component: followed by ``/`` or not by a name character."""
    return re.compile(re.escape(root.rstrip('/')) + r'(?:/|(?![\w.\-]))')


def redact(text: str, roots: SearchRoots, home: Optional[PathLike] = None) -> str:
    """``text`` with every root shown as ``label:/`` and the home folder as ``~``.

    The longest root goes first, so a root nested in another (or in home) keeps
    its own label. Both the lexical and the symlink-resolved spelling count.
    """
    home = Path.home() if home is None else Path(home)
    replacements = []
    for label, root in roots.tiers():
        for spelling in (str(_lexical(root)), os.path.realpath(root)):
            replacements.append((spelling, f'{label}:/'))
    for spelling in (str(_lexical(home)), os.path.realpath(home)):
        if spelling != '/':
            replacements.append((spelling, '~/'))
    for spelling, label in sorted(replacements, key=lambda r: len(r[0]), reverse=True):
        text = _path_pattern(spelling).sub(label, text)
    return text
