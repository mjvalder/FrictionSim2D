"""Getting data back out of the store.

`export_results` writes a set's stored files back into an ordinary simulation
tree, so `postprocess read`, the GUI download and `rebuild` all work on the
same shape the runner produces. It replaces `integration.dump_results_to_json`'s
second, nested-dict writer.

`rebuild_set` goes one step further: it hands each run's stored `config.json`
to `run.artifacts.generate_package` -- the same router `FrictionSim2D run`
uses -- so a rebuild reaches the real builder instead of a drifted copy of
`output_dir_for`. `generate_package` is imported at module level (not deferred
into the function like the `aiida.orm` imports below) so tests can patch
`src.aiida.export.generate_package` directly: it never touches `aiida-core`,
so importing it costs nothing on a machine without AiiDA and does not weaken
the "import this module without aiida-core installed" contract the deferred
imports below exist for.

`export_archive` hands `aiida.tools.archive.create_archive` every node of
`Group('friction2d/<label>')` -- the group itself included, so the archive's
import re-creates it and the set stays visible to `resolve_set`/`delete`/
`export`/`rebuild` on the far side (spec §4.2, R19). `import_archive` is its
inverse.

`export_provenance_bundle` writes ONE run's provenance files only -- the
GUI's "download provenance" action (database-pane spec D6) -- as opposed to
`export_results`, which writes a whole set's tree including every results
file.
"""
from __future__ import annotations

import inspect
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, get_args, get_origin

from pydantic import BaseModel

from ..core.config import AFMSimulationConfig, SheetOnSheetSimulationConfig
from ..run.artifacts import generate_package
from ..run.naming import AFM_LIKE_MODELS
from ..run.tree import provenance_dir_for

logger = logging.getLogger(__name__)

__all__ = ['export_results', 'resolve_set', 'rebuild_set', 'export_archive',
          'import_archive', 'export_provenance_bundle']


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
                     seen: Optional[set] = None) -> List[Path]:
    """Write every file in a node's repository to `mapper(rel_posix_path)`.

    `mapper` decides the on-disk destination for each stored, run-relative
    path -- a plain `dest / rel` join for a set node or a results node, but
    something that can redirect a run node's `provenance/**` prefix one level
    up for the AFM-like models (see `_run_mapper`).

    `seen`, given only for a run node's dump, is a `set[Path]` the WHOLE
    `export_results` call shares across every run: a destination already
    added to it this call is skipped rather than rewritten. This is
    deliberately about THIS call's own writes, not the filesystem -- an
    earlier version checked `out.is_file()` instead, which meant re-exporting
    into a non-empty `dest` silently kept whatever bytes were already sitting
    there (a stale `provenance.json`, a hand-edited `lammps/*.in`) instead of
    overwriting them from the store, exactly backwards from what "export"
    means. Checking `seen` instead only ever short-circuits the ONE case this
    exists for: the AFM-like models store an independent copy of the SAME
    shared `provenance/**` subtree on every layer's run node (Task 16's
    `add_run_files` walks `provenance_dir` once per run, not once per
    campaign), so the second and third layer of one export would otherwise
    re-read and rewrite files the first layer already wrote earlier in this
    same call, byte-for-byte identical every time. A run's own
    `provenance.json` and `lammps/**` are never shared this way (their
    destination is unique to this run), so this never skips a file that
    should legitimately differ.
    """
    written = []
    for root, _dirs, files in node.base.repository.walk():
        for name in files:
            rel = (root / name).as_posix()
            out = mapper(rel)
            if seen is not None:
                if out in seen:
                    continue
                seen.add(out)
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
    provenance_dest = run_dest.parent if model in AFM_LIKE_MODELS else run_dest

    def mapper(rel: str) -> Path:
        if rel == 'provenance' or rel.startswith('provenance/'):
            return provenance_dest / rel
        return run_dest / rel

    return mapper


def _write_updated_provenance(run_node, run_dest: Path,
                              already: List[Path]) -> List[Path]:
    """Overwrite the exported `provenance.json` with the post-run update the
    run node carries in its extras, if any (see
    `FrictionSimulationData.refresh_provenance`). Returns the path when it
    was not already in `already`, so the caller's list stays duplicate-free.
    """
    text = run_node.provenance_file_text()
    if text is None:
        return []
    out = run_dest / 'provenance.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    return [] if out in already else [out]


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
    seen: set = set()
    for run_node in _runs_of(set_node):
        run_dest = dest / run_node.rel_path
        written += _dump_repository(run_node, _run_mapper(run_dest, run_node.model),
                                    seen=seen)
        written += _write_updated_provenance(run_node, run_dest, written)
        _warn_oversize(run_node, run_node.rel_path)
        for result in _results_of(run_node):
            written += _dump_repository(result, lambda rel, d=run_dest: d / rel)
            _warn_oversize(result, run_node.rel_path)
    logger.info("Exported %d files for set %r to %s", len(written), label, dest)
    return sorted(written)


def _nested_model_cls(annotation: Any) -> Optional[Type[BaseModel]]:
    """The `BaseModel` subclass `annotation` names, through `Optional`/`List`.

    Handles a bare model (`SheetConfig`), `Optional[SheetConfig]`
    (`Union[SheetConfig, None]`) and `List[SheetConfig]`; returns `None` for
    anything else (`str`, `float`, `Dict[str, Any]`, ...), which is most
    fields -- `_normalize_config_aliases` only recurses where this returns
    something.
    """
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        return _nested_model_cls(args[0]) if args else None
    if origin is not None:  # Union (incl. Optional) and any other generic
        none_type = type(None)
        for arg in get_args(annotation):
            if arg is none_type:  # the `None` half of `Optional[X]`, not a model
                continue
            found = _nested_model_cls(arg)
            if found is not None:
                return found
        return None
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _normalize_config_aliases(data: Dict[str, Any], model_cls: Type[BaseModel]) -> None:
    """Rename `model_cls`'s fields back to the aliases their stored JSON dropped.

    `run_simulations` stores `config_obj.model_dump_json()` with no
    `by_alias=True` (a deliberately unchanged choice -- every `config.json`
    already on disk was written this way), so a field declared
    `Field(..., alias=X)` round-trips under its PYTHON field name instead --
    `AFMSimulationConfig.sheet`/`SheetOnSheetSimulationConfig.sheet` (alias
    `'2D'`) and `SheetOnSheetSimulationConfig.flake` (alias `'Flake'`) today,
    whatever the model gains an alias for tomorrow. Two independent readers
    care: `src.run.expand.expand_config_sweeps` and `src.run.naming.
    primary_sheet`/`output_dir_for` run on the raw dict BEFORE any Pydantic
    model exists and look up the alias literally (`'2D'`), and Pydantic
    itself only accepts the alias for a field whose class has neither
    `populate_by_name` nor `validate_by_name` set (`AFMSimulationConfig`/
    `SheetOnSheetSimulationConfig` do not) -- so a field-name key such as
    `'flake'` silently validates to its default (`None`) instead of raising,
    routing `run/simulate.py`'s `_select_sheet_builder_cls` to the wrong
    builder with no error at all.

    Derives the rename table from `model_cls.model_fields` instead of
    hard-coding `'sheet'`/`'flake'`, and recurses into nested model fields
    (plain, `Optional`, or `List[...]`) -- but only where the nested class
    itself needs the same treatment, so a class that already tolerates both
    forms (`ThermostatSettings`/`QuenchSettings`/`PotentialSettings`, all
    `validate_by_name=True`) is left alone.

    Mutates `data` in place; idempotent -- a key already in alias form is
    never touched, so re-running this over an already-fixed dict is a no-op.
    """
    tolerant = model_cls.model_config.get('populate_by_name', False) or \
        model_cls.model_config.get('validate_by_name', False)
    for name, info in model_cls.model_fields.items():
        alias = info.alias
        key = name
        if alias and alias != name and not tolerant:
            if name in data and alias not in data:
                data[alias] = data.pop(name)
            key = alias
        nested_cls = _nested_model_cls(info.annotation)
        if nested_cls is None or key not in data or data[key] is None:
            continue
        if isinstance(data[key], list):
            for item in data[key]:
                if isinstance(item, dict):
                    _normalize_config_aliases(item, nested_cls)
        elif isinstance(data[key], dict):
            _normalize_config_aliases(data[key], nested_cls)


def _fix_config_aliases(config_json: Path, model: str) -> None:
    """Normalize one staged `config.json` back to alias form (see above).

    `model` picks the SAME config class `run_simulations` would build for it
    (`AFMSimulationConfig` for `AFM_LIKE_MODELS`, `SheetOnSheetSimulationConfig`
    otherwise) -- the class the stored JSON was originally validated from and
    the one whose `Field(alias=...)` declarations define what "alias form"
    means. Rewrites the staged copy in place -- never the stored node or the
    original tree.
    """
    config_cls = AFMSimulationConfig if model in AFM_LIKE_MODELS else SheetOnSheetSimulationConfig
    data = json.loads(config_json.read_text(encoding='utf-8'))
    _normalize_config_aliases(data, config_cls)
    config_json.write_text(json.dumps(data, indent=2), encoding='utf-8')


def rebuild_set(label: str, dest: Path, *, profile: Optional[str] = None,
                generate_hpc: bool = False) -> Tuple[List[Path], Path]:
    """Regenerate a set's LAMMPS inputs from the store, through the real router.

    Exports the set's stored `settings.yaml` and every run's stored
    `provenance/config.json` into a staging directory OUTSIDE `dest` -- a
    `tempfile.TemporaryDirectory()`, never a subdirectory of `dest` itself.
    A staging directory nested inside `dest` (e.g. `dest/.rebuild_inputs`)
    would sit right next to the real `lammps/` directories `generate_package`
    is about to write into `dest`, and `JobManifest.from_simulation_directory`
    finds jobs with `rglob('lammps')` under `generate_hpc=True` -- it would
    find the staged inputs' own `lammps/` copies too and duplicate every job.

    Each DISTINCT stored `config.json` is then handed once to
    `run.artifacts.generate_package` -- the SAME router `FrictionSim2D run`
    uses, so every family (hetero, flake, freestanding, finite) reaches the
    builder it actually needs. Nothing here duplicates `output_dir_for` and
    nothing instantiates a builder directly. `use_aiida=False` and
    `allow_generate_amorphous=False` are passed explicitly (a rebuild must not
    re-register nodes, and must not spend a melt-quench regenerating a
    substrate whose realization id is already recorded); `ensure_hpc_settings`
    is left at `None`. `simulation_root_name='.'` makes `generate_package`
    build straight into `dest` rather than a fresh timestamped root, which is
    what keeps every run's `rel_path` identical to the original.

    "Distinct" is by the resolved `provenance/config.json` PATH in the
    staging export, not by the file's own bytes: an AFM-like run nests every
    `L{n}` layer under one shared `provenance/`, one level above the run
    directory (`run.tree.provenance_dir_for`) -- so `L1..Ln` of a single
    layer sweep all resolve to the SAME stored file (one `config.json` was
    already flattened into one member by `run_simulations`, and that one
    member built every layer). Rebuilding `L1` already regenerates `L2`;
    calling `generate_package` again for `L2` would redo that whole layer
    sweep a second time. Two runs that merely happen to carry byte-identical
    config content -- different materials or models with an equal payload --
    are a coincidence, not a shared file, and are never deduplicated: only
    file identity (the same stored path) collapses two run nodes into one
    call.
    """
    _load_profile(profile)
    set_node = resolve_set(label)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    created: List[Path] = []
    seen_configs: set = set()
    with tempfile.TemporaryDirectory(prefix='frictionsim2d-rebuild-') as tmp_dir:
        staging = Path(tmp_dir)
        export_results(label, staging, profile=profile)
        settings_file = staging / 'settings.yaml'
        settings_file = settings_file if settings_file.is_file() else None

        for run_node in _runs_of(set_node):
            run_dir = staging / run_node.rel_path
            config_json = provenance_dir_for(run_dir) / 'config.json'
            if not config_json.is_file():
                logger.warning("No config.json stored for %s -- skipping",
                               run_node.rel_path)
                continue

            resolved = config_json.resolve()
            if resolved in seen_configs:
                continue
            seen_configs.add(resolved)
            _fix_config_aliases(config_json, run_node.model)

            result = generate_package(
                config_json,
                run_node.model,
                output_root=dest,
                settings_file=settings_file,
                generate_hpc=generate_hpc,
                use_aiida=False,
                simulation_root_name='.',
                allow_generate_amorphous=False,
            )
            for failure in result.failures:
                logger.error("Rebuild failed: %s", failure.render())
            created.extend(result.created)

    return created, dest


def export_archive(label: str, output_path: Path, *,
                   profile: Optional[str] = None) -> Path:
    """Archive every node of `friction2d/<label>`, the group itself included.

    An archive of the set's nodes alone (`FrictionSimulationSetData` + its
    runs + their results) creates no `friction2d/<label>` group on import
    (R19): the group has to be one of the archived entities for
    `aiida.tools.archive.import_archive` to re-create it, and without that
    group the imported set would be invisible to `resolve_set`/`delete`/
    `export`/`rebuild` -- breaking the hpc-direct mode (spec §4.3) an
    export/import round trip exists to support. Archiving the GROUP (not
    just its nodes) is exactly what closes that gap.
    """
    from aiida.orm import Group  # pylint: disable=import-outside-toplevel
    from aiida.tools.archive import create_archive  # pylint: disable=import-outside-toplevel

    from .ingest import group_label  # pylint: disable=import-outside-toplevel

    _load_profile(profile)
    resolve_set(label)                       # raises LookupError on an unknown label
    group = Group.collection.get(label=group_label(label))
    output_path = Path(output_path)
    create_archive([group, *group.nodes], filename=output_path)
    logger.info("Exported group %r (%d nodes) to %s", group.label,
               len(group.nodes), output_path)
    return output_path


#: The provenance bundle: exactly these five categories of a run node's
#: repository. NOT `lammps/*.in` (that is `rebuild_set`'s job) and NOT any
#: results file (that is `export_results`'s).
_BUNDLE_EXACT = ('provenance.json', 'provenance/config.json',
                 'provenance/settings.yaml')
_BUNDLE_PREFIXES = ('provenance/cif/', 'provenance/potentials/')


def export_provenance_bundle(run, dest: Path) -> List[Path]:
    """Write ONE run's provenance files into `dest`, and nothing else.

    Exactly `provenance.json`, `provenance/config.json`,
    `provenance/settings.yaml`, `provenance/cif/*` and
    `provenance/potentials/*` -- what the GUI's "download provenance" action
    (database-pane spec D6) hands the user. `export_results` writes a whole
    set's tree including every results file, which is the wrong shape here.

    Files keep their repository-relative layout, so the bundle is a strict
    subset of what `export_results` writes for the same run and the same
    reader works on both. `dest` is created if needed. Returns the paths
    written, sorted.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for root, _dirs, files in run.base.repository.walk():
        for name in files:
            rel = (root / name).as_posix()
            if rel not in _BUNDLE_EXACT and not rel.startswith(_BUNDLE_PREFIXES):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with run.base.repository.open(rel, 'rb') as src:
                out.write_bytes(src.read())
            written.append(out)
    written += _write_updated_provenance(run, dest, written)
    logger.info("Wrote %d provenance file(s) for %s to %s",
               len(written), run.rel_path, dest)
    return sorted(written)


def import_archive(archive_path: Path, *, profile: Optional[str] = None) -> int:
    """Import an archive into the current profile; returns the new-node count.

    `aiida.tools.archive.import_archive` returns the primary key of the
    import group it creates (or `None`), not a summary object with a
    `new_nodes` count -- reading a `new_nodes` attribute off that return
    value is always 0. The profile's node count is compared across the call
    instead.

    `create_group=False`: the archive already carries its own
    `friction2d/<label>` group as one of its entities (`export_archive`
    above), so aiida-core's default `create_group=True` would additionally
    create its own timestamp-labelled "import group" containing the same
    nodes on every single import -- nodes and the `friction2d` group dedup
    correctly on a re-import (`n_new=0`), but that extra group does not and
    is left behind every time.
    """
    from aiida.orm import Node, QueryBuilder  # pylint: disable=import-outside-toplevel
    from aiida.tools.archive import import_archive as _import  # pylint: disable=import-outside-toplevel

    _load_profile(profile)
    before = QueryBuilder().append(Node).count()
    _import(Path(archive_path), create_group=False)
    after = QueryBuilder().append(Node).count()
    n_new = after - before
    logger.info("Imported %d new node(s) from %s", n_new, archive_path)
    return n_new
