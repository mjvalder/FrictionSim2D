"""What counts as a plottable dataset, and where the Plots tab looks for one.

Qt-free, so the assistant's analysis tools share the rule instead of keeping a
copy of it.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .workspace import Workspace

__all__ = ['DATASETS_DIRNAME', 'dataset_files', 'is_dataset', 'discover_datasets']

#: Where `aiida export-results --output-dir ...` is expected to drop a set.
DATASETS_DIRNAME = '_datasets'


def dataset_files(folder: Path) -> List[Path]:
    """The `outputs/output_full_*.json` files of ``folder``, sorted (may be empty)."""
    outputs = Path(folder) / 'outputs'
    if not outputs.is_dir():
        return []
    return sorted(outputs.glob('output_full_*.json'))


def is_dataset(folder: Path) -> bool:
    """A dataset is a directory with an `outputs/` holding at least one
    `output_full_*.json` -- the same thing `Plotter` accepts as a `data_dir`."""
    outputs = Path(folder) / 'outputs'
    return outputs.is_dir() and any(outputs.glob('output_full_*.json'))


def discover_datasets(workspace: Workspace) -> List[Path]:
    """Every directory that looks like a plottable dataset.

    Two places are searched: the workspace root (where `run` writes its
    simulation packages) and `<workspace>/.preview/_datasets/`, which is where
    `FrictionSim2D aiida export-results --output-dir ...` puts a set pulled
    back out of the local store.
    """
    roots = [workspace.root, workspace.preview_dir / DATASETS_DIRNAME]
    found: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in sorted(root.iterdir()):
            if candidate.is_dir() and is_dataset(candidate) and candidate not in found:
                found.append(candidate)
    return found
