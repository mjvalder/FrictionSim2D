"""Workspace tools: look around, and propose new folders or copied-in files.

No move, rename or delete exists here or anywhere in the assistant.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..changes import build_fs_changeset
from ..sandbox import SandboxError, display_path, resolve_readable
from . import compact

MAX_LINES = 60
MAX_DESCRIBE_FILES = 5000


def _walk_capped(folder: Path) -> tuple[list[Path], bool]:
    """Files under ``folder``, stopping at MAX_DESCRIBE_FILES; flag if capped."""
    files: list[Path] = []
    for base, _dirs, names in os.walk(folder):
        for name in sorted(names):
            if len(files) >= MAX_DESCRIBE_FILES:
                return files, True
            files.append(Path(base) / name)
    return files, False


def workspace_overview(deps, depth: int = 2) -> str:
    """Folder tree of the workspace (no dot-folders), with entry counts."""
    root = Path(deps.roots.workspace)
    lines = ['workspace:/']

    def walk(folder: Path, level: int) -> None:
        if level > depth or len(lines) >= MAX_LINES:
            return
        for child in sorted(p for p in folder.iterdir()
                            if p.is_dir() and not p.name.startswith('.')):
            if len(lines) >= MAX_LINES:
                return
            count = sum(1 for _ in child.iterdir())
            lines.append(f"{'  ' * level}{child.name}/ ({count} entries)")
            walk(child, level + 1)

    walk(root, 1)
    return compact('\n'.join(lines))


def describe_run(deps, path: str) -> str:
    """What a folder holds: provenance keys, results files, inputs and logs."""
    folder = resolve_readable(path, deps.roots)
    if not folder.is_dir():
        raise SandboxError(f'{path}: not a folder')
    files, capped = _walk_capped(folder)
    if capped:
        summary = f'{len(files)}+ files'
    else:
        size = sum(p.stat().st_size for p in files if p.is_file()) / 1e6
        summary = f'{len(files)} files, {size:.1f} MB'
    lines = [f'{display_path(folder, deps.roots)}: {summary}']
    provenance = folder / 'provenance.json'
    if provenance.is_file():
        try:
            keys = list(json.loads(provenance.read_text(encoding='utf-8')))[:10]
            lines.append('provenance.json keys: ' + ', '.join(keys))
        except (ValueError, OSError):
            lines.append('provenance.json: unreadable')
    for label, pattern in (('results', 'output_full_*.json'), ('inputs', '*.in'),
                           ('logs', 'log*.lammps')):
        found = [p.name for p in files if p.match(pattern)][:8]
        if found:
            lines.append(f'{label}: ' + ', '.join(found))
    return compact('\n'.join(lines))


def propose_mkdir(deps, path: str) -> str:
    """Propose creating a folder inside the workspace."""
    return deps.propose(build_fs_changeset(deps, f'Create folder {path}', [('mkdir', path)]))


def propose_copy_in(deps, source: str, dest_dir: str = 'materials') -> str:
    """Propose copying a found structure or potential file into the workspace."""
    name = resolve_readable(source, deps.roots).name
    ops = [('mkdir', dest_dir), ('copy', source, f'{dest_dir}/{name}')]
    return deps.propose(build_fs_changeset(deps, f'Copy {name} into {dest_dir}/', ops))
