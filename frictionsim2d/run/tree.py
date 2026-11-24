"""Locating runs in a built simulation tree: one walk, one parser.

``hpc_dispatch.collect_hpc_simulation_paths`` keeps its own walk because it
collects directories holding a lammps script, which exist before a run has
produced a ``provenance.json``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, List, Tuple

from .naming import MODELS, RunTokens, parse_run_path

logger = logging.getLogger(__name__)

__all__ = ['RunLocator', 'discover_runs', 'unparseable_runs', 'provenance_dir_for']


@dataclass(frozen=True)
class RunLocator:
    """One run: its tokens and the absolute directories that belong to it."""
    tokens: RunTokens
    run_dir: Path
    provenance_file: Path
    provenance_dir: Path
    results_dir: Path
    data_dir: Path
    lammps_dir: Path


def provenance_dir_for(run_dir: Path) -> Path:
    """Where `provenance/` lives for a run.

    The AFM family writes `provenance.json` into the per-layer `L{n}` level while
    `provenance/` stays one level up; sheet-like runs have both in one place.
    """
    here = run_dir / 'provenance'
    if here.is_dir():
        return here
    return run_dir.parent / 'provenance'


def _walk_provenance_files(root: Path) -> Iterator[Tuple[str, Path]]:
    """Every `provenance.json` under `<root>/<model>/`, as `(rel_path, file)`."""
    root = Path(root)
    for model in MODELS:
        model_dir = root / model
        if not model_dir.is_dir():
            continue
        for prov_file in sorted(model_dir.rglob('provenance.json')):
            run_dir = prov_file.parent
            rel = run_dir.relative_to(root).as_posix()
            yield rel, prov_file


def discover_runs(root: Path) -> List[RunLocator]:
    """Every run under ``<root>/<model>/``, sorted by ``rel_path``.

    A run is one ``provenance.json``. Paths that do not parse are logged and
    skipped, so one malformed directory cannot hide a whole campaign -- see
    :func:`unparseable_runs` for the list of what got skipped.
    """
    root = Path(root)
    found: List[RunLocator] = []
    for rel, prov_file in _walk_provenance_files(root):
        run_dir = prov_file.parent
        try:
            tokens = parse_run_path(rel)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", rel, exc)
            continue
        family = tokens.model
        try:
            record = json.loads(prov_file.read_text(encoding='utf-8'))
            family = record.get('family') or tokens.model
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Unreadable provenance.json at %s (%s); "
                           "falling back to family=%s", rel, exc, family)
        found.append(RunLocator(
            tokens=replace(tokens, family=family),
            run_dir=run_dir,
            provenance_file=prov_file,
            provenance_dir=provenance_dir_for(run_dir),
            results_dir=run_dir / 'results',
            data_dir=run_dir / 'data',
            lammps_dir=run_dir / 'lammps',
        ))
    return sorted(found, key=lambda loc: loc.tokens.rel_path)


def unparseable_runs(root: Path) -> List[str]:
    """``rel_path`` of every run under ``root`` that :func:`discover_runs`
    had to skip -- a ``provenance.json`` whose directory :func:`parse_run_path`
    rejects with ``ValueError``.

    Walks the same model top-level directories as :func:`discover_runs`
    (via :func:`_walk_provenance_files`); a path parses in exactly one of the
    two functions, never both.
    """
    root = Path(root)
    skipped: List[str] = []
    for rel, _prov_file in _walk_provenance_files(root):
        try:
            parse_run_path(rel)
        except ValueError:
            skipped.append(rel)
    return sorted(skipped)
