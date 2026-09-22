"""Getting data back out of the store.

`export_results` writes a set's stored files back into an ordinary simulation
tree, so `postprocess read`, the GUI download and `rebuild` all work on the
same shape the runner produces. It replaces `integration.dump_results_to_json`'s
second, nested-dict writer.

Only `export_results` (and the `resolve_set` lookup it shares with later
tasks) is implemented here -- `export_provenance_bundle`, `rebuild_set`,
`export_archive` and `import_archive` belong to later tasks in this plan and
are added to this module then, not pre-declared as empty names now.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

__all__ = ['export_results', 'resolve_set']


def _load_profile(profile: Optional[str]) -> None:
    """Load the AiiDA profile for every public function in this module.

    Deferred, so `import src.aiida.export` stays cheap and works without
    aiida-core installed (the headless-safety rule); every entry point calls
    it first.
    """
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel

    load_aiida_profile(profile)


def resolve_set(label: str):
    """The `FrictionSimulationSetData` for `label`, by label then UUID prefix."""
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    from .data import FrictionSimulationSetData  # pylint: disable=import-outside-toplevel

    qb = QueryBuilder()
    qb.append(FrictionSimulationSetData, filters={'attributes.label': label})
    node = qb.first(flat=True)
    if node is not None:
        return node
    qb = QueryBuilder()
    qb.append(FrictionSimulationSetData)
    for candidate in qb.all(flat=True):
        if str(candidate.uuid).startswith(label):
            return candidate
    raise LookupError(f"no simulation set for label or UUID {label!r}")


def _runs_of(set_node):
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    from .data import FrictionSimulationData  # pylint: disable=import-outside-toplevel

    qb = QueryBuilder()
    qb.append(FrictionSimulationData,
              filters={'attributes.set_uuid': str(set_node.uuid)})
    return sorted(qb.all(flat=True), key=lambda n: n.rel_path)


def _results_of(run_node):
    """Every CURRENT (non-superseded) results node of `run_node`.

    A superseded node and its replacement store the same run-relative file
    names (same load point, different bytes): exporting both would make
    whichever one the QueryBuilder happens to return last win the write --
    not necessarily the replacement. Excluded with the same `is_superseded`
    check `ingest.find_results` uses, so the two can never disagree about
    which node is current.
    """
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    from .data import FrictionResultsData  # pylint: disable=import-outside-toplevel
    from .data_keys import is_superseded  # pylint: disable=import-outside-toplevel

    qb = QueryBuilder()
    qb.append(FrictionResultsData,
              filters={'attributes.run_uuid': str(run_node.uuid)})
    return [node for node in qb.all(flat=True) if not is_superseded(node)]


def _warn_oversize(node, rel_path: str) -> None:
    """Log every file `node` recorded as too large to store (R11c/Task 16).

    `FrictionSimulationData.add_run_files`/`FrictionResultsData.
    add_result_files` skip a file over `settings.aiida.max_stored_file_mb` and
    record its run-relative path, size and sha256 in `oversize_files` instead
    of storing it -- there is nothing in the repository for `_dump_repository`
    to write back for it, so it is silently absent from the exported tree
    unless something says so. This is that something: a caller (a human
    reading the log, or a future `aiida export-results` CLI command) can see
    exactly which file is missing and how to find it again (the recorded
    `sha256`), rather than trusting an export that quietly dropped a
    multi-GB `final_*.data`.
    """
    for entry in node.oversize_files:
        logger.warning(
            "%s: %s was not stored (%s bytes, sha256=%s) and is missing "
            "from the export.", rel_path, entry.get('path'), entry.get('size'),
            entry.get('sha256'))


def _dump_repository(node, mapper: Callable[[str], Path], *,
                     skip_existing: bool = False) -> List[Path]:
    """Write every file in a node's repository to `mapper(rel_posix_path)`.

    `mapper` decides the on-disk destination for each stored, run-relative
    path -- a plain `dest / rel` join for a set node or a results node, but
    something that can redirect a run node's `provenance/**` prefix one level
    up for the AFM-like models (see `_run_mapper`).

    `skip_existing`, used only for a run node's dump, skips a destination
    that already has a file on it: the AFM-like models store an independent
    copy of the SAME shared `provenance/**` subtree on every layer's run node
    (Task 16's `add_run_files` walks `provenance_dir` once per run, not once
    per campaign), so the second and third layer of one campaign would
    otherwise re-read and rewrite files the first layer already wrote here,
    byte-for-byte identical every time. A run's own `provenance.json` and
    `lammps/**` are never shared this way (their destination is unique to
    this run), so this never skips a file that should legitimately differ.
    """
    written = []
    for root, _dirs, files in node.base.repository.walk():
        for name in files:
            rel = (root / name).as_posix()
            out = mapper(rel)
            if skip_existing and out.is_file():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with node.base.repository.open(rel, 'rb') as src:
                out.write_bytes(src.read())
            written.append(out)
    return written


def _run_mapper(run_dest: Path, model: str) -> Callable[[str], Path]:
    """Where one run node's stored files land under `run_dest` (R18).

    `provenance.json`, `lammps/**` and every aux/tier file a
    `FrictionResultsData` carries (`results/**`, `data/**`, a bare
    `log_*.lammps`) are all stored relative to the run's own directory
    (`FrictionSimulationData.add_run_files`, `FrictionResultsData.
    add_result_files`) and go straight to `run_dest`.

    `provenance/**` is the one exception: `src.run.tree.provenance_dir_for`
    puts it at the run directory's own parent for the AFM-like models --
    `AFMSimulation.write_inputs` always nests the run under a per-layer
    `L{n}` directory but writes `config.json`/`manifest.json`/`cif/` etc. one
    level up, shared by every layer -- while the sheet-like models keep both
    in the same place. `add_run_files` stores that subtree under the bare
    `provenance/` prefix regardless of which case produced it, so this is the
    one place that has to tell them apart again, from the run's own model.
    """
    from ..run.naming import AFM_LIKE_MODELS  # pylint: disable=import-outside-toplevel

    provenance_dest = run_dest.parent if model in AFM_LIKE_MODELS else run_dest

    def mapper(rel: str) -> Path:
        if rel == 'provenance' or rel.startswith('provenance/'):
            return provenance_dest / rel
        return run_dest / rel

    return mapper


def export_results(label: str, dest: Path, *,
                   profile: Optional[str] = None) -> List[Path]:
    """Write a set's stored files back into a simulation tree at `dest`.

    Writes `<dest>/<rel_path>/{provenance.json, provenance/**, lammps/**,
    results/**, data/**}` for every run in the set (`provenance/**` landing
    one level up for the AFM-like models, see `_run_mapper`), plus
    `<dest>/settings.yaml`. Returns every file written, sorted.
    """
    _load_profile(profile)
    set_node = resolve_set(label)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    written = _dump_repository(set_node, lambda rel: dest / rel)
    for run_node in _runs_of(set_node):
        run_dest = dest / run_node.rel_path
        written += _dump_repository(run_node, _run_mapper(run_dest, run_node.model),
                                    skip_existing=True)
        _warn_oversize(run_node, run_node.rel_path)
        for result in _results_of(run_node):
            written += _dump_repository(result, lambda rel, d=run_dest: d / rel)
            _warn_oversize(result, run_node.rel_path)
    logger.info("Exported %d files for set %r to %s", len(written), label, dest)
    return sorted(written)
