"""AiiDA CalcJob and Parser for LAMMPS friction simulations.

Provides :class:`LammpsFrictionCalcJob`, which submits pre-built
FrictionSim2D simulation directories to an HPC cluster via AiiDA's
job scheduling infrastructure, and :class:`LammpsFrictionParser`,
which extracts results from the retrieved outputs.
"""

from __future__ import annotations

import io
import logging
import re
import shlex
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser

from .retention import retrieve_patterns

logger = logging.getLogger(__name__)

_SIM_SUBDIRS = (
    'lammps', 'data', 'potentials', 'results', 'visuals', 'build', 'provenance',
)

#: Attribute key on a staged `simulation_dir` FolderData listing which of
#: `_SIM_SUBDIRS` were sourced from the run directory's PARENT rather than the
#: run directory itself -- see `_stage_sim_dir`'s docstring. `None`, never
#: `()`, is the "not present" default wherever this is read back:
#: `NodeAttributes.get`'s own "no default given" sentinel IS the empty tuple
#: (CPython interns it), so passing `()` there is silently indistinguishable
#: from omitting the default and raises instead of falling back.
_SHARED_SUBDIRS_ATTR = '_frictionsim2d_shared_subdirs'

#: Only these two `_SIM_SUBDIRS` are ever looked for at `sim_dir.parent` when
#: missing at `sim_dir` -- `_stage_sim_dir`'s docstring has the AFM-nesting
#: reason. Restricted (fix round 1, finding 2) so an unrelated directory that
#: happens to sit next to the run (e.g. a stray `results/` from a differently
#: named sibling run) is never picked up as this run's own.
_SHAREABLE_SUBDIRS = ('build', 'provenance')

#: Attribute key recording the run directory's own basename (`sim_dir.name`,
#: e.g. `'L2'`) on a `simulation_dir` FolderData built by
#: `prepare_simulation_folder`. Fix round 1, finding 2: without a
#: `simulation_node` input, `_sim_prefix_for` had no way to recover the run's
#: own level -- only `_detect_sim_prefix`'s shared-ancestor guess -- so a
#: `smart_submit`/`aiida submit` call (which never sets `simulation_node`,
#: see `smart_submit`/`submit_batch(config_path=None)`) staged both the run's
#: own files AND the shared `build`/`provenance` ones a level off. Combined
#: with `_SHARED_SUBDIRS_ATTR` in `_sim_prefix_for`, this reconstructs the
#: run's full prefix (`<shared ancestor>/<leaf name>`) without needing
#: `simulation_node` at all. Not written by `prepare_simulation_root` (the
#: array path, out of scope for this fix -- see its own module docstring
#: note): that stages several runs already relative to the true sim ROOT, so
#: `_detect_sim_prefix` alone is already correct there, and this attribute
#: would be meaningless (one leaf name for many staged runs).
_LEAF_NAME_ATTR = '_frictionsim2d_leaf_name'


# ---------------------------------------------------------------------------
# CalcJob
# ---------------------------------------------------------------------------

class LammpsFrictionCalcJob(CalcJob):
    """AiiDA CalcJob that runs a FrictionSim2D LAMMPS simulation."""

    _DEFAULT_LAMMPS_SCRIPTS = ['system.in', 'slide.in']

    @classmethod
    def define(cls, spec):
        """Define the CalcJob input/output specification."""
        super().define(spec)

        spec.input('code', valid_type=orm.AbstractCode,
                   help='LAMMPS executable code node.')
        spec.input('simulation_dir', valid_type=orm.FolderData,
                   help='FolderData with the simulation directory contents.')
        spec.input('simulation_node', valid_type=orm.Data, required=False,
                   help='FrictionSimulationData node for status tracking.')
        spec.input('parameters', valid_type=orm.Dict, required=False,
                   default=lambda: orm.Dict({}),
                   help='Optional runtime parameters.')

        spec.input('metadata.options.resources', valid_type=dict,
                   default={'num_machines': 1, 'num_mpiprocs_per_machine': 32})
        spec.input('metadata.options.max_wallclock_seconds', valid_type=int,
                   default=72000)
        spec.input('metadata.options.parser_name', valid_type=str,
                   default='friction2d.lammps')

        spec.output_namespace('results', valid_type=orm.Data, dynamic=True,
                              required=False,
                              help='One FrictionResultsData per load point.')
        spec.output('log_file', valid_type=orm.SinglefileData, required=False,
                    help='LAMMPS log file.')
        spec.output('stdout_folder', valid_type=orm.FolderData, required=False,
                    help='FolderData with LAMMPS stdout files (*.out).')

        spec.exit_code(300, 'ERROR_NO_LAMMPS_SCRIPTS',
                       message='No LAMMPS input scripts found in simulation directory.')
        spec.exit_code(310, 'ERROR_LAMMPS_FAILED',
                       message='LAMMPS returned a non-zero exit code.')
        spec.exit_code(320, 'ERROR_NO_RESULTS',
                       message='No results files were produced.')
        spec.exit_code(321, 'ERROR_NO_SIMULATION_NODE',
                       message='No simulation_node input was given, so the '
                               'parser has no rel_path to locate this run\'s '
                               'results in the retrieved folder.')
        spec.exit_code(330, 'ERROR_NO_LOAD_POINTS',
                       message='Results were retrieved but no load point could '
                               'be formed from them.')

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Prepare the calculation for submission."""
        inputs = self.inputs  # type: ignore[attr-defined]
        sim_folder = inputs.simulation_dir
        repo = sim_folder.base.repository

        params = (inputs.get('parameters', None) or orm.Dict({})).get_dict()
        scripts_override = params.get('lammps_scripts')
        lammps_flags = params.get('lammps_flags', '-l none')
        lammps_flag_list = (
            shlex.split(lammps_flags) if isinstance(lammps_flags, str)
            else list(lammps_flags)
        )

        lammps_scripts = (
            _collect_lammps_scripts(sim_folder, scripts_override)
            or self._DEFAULT_LAMMPS_SCRIPTS
        )
        sim_prefix = _sim_prefix_for(inputs, repo, lammps_scripts)
        shared_subdirs = frozenset(
            sim_folder.base.attributes.get(_SHARED_SUBDIRS_ATTR, None) or ())

        _stage_files_into_sandbox(folder, repo, sim_prefix, shared_subdirs)
        _handle_array_mode(folder, sim_folder, params, scripts_override, lammps_scripts)

        code_infos = [
            _make_code_info(inputs.code.uuid, lammps_flag_list, script)
            for script in lammps_scripts
        ]

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = code_infos
        calcinfo.retrieve_list = _retrieve_list_for(sim_prefix)
        return calcinfo

    def on_terminated(self) -> None:
        """Once this CalcJob reaches a terminal state, file its parser's
        `results.*` outputs into the run's set group.

        The engine attaches `Parser.out()`'s nodes to this calculation node
        with real provenance links and stores them, but it never adds
        anything to an AiiDA `Group` on its own -- `create_results` does that
        itself for the import path; this is the CalcJob-native counterpart,
        run once there is a finished node with real output links to read.
        """
        super().on_terminated()
        _group_results_from_node_safely(self.node)
        _supersede_older_results_safely(self.node)


def _sim_prefix_for(inputs, repo, lammps_scripts) -> str | None:
    """Where this run's own directory sits under the remote working directory.

    `simulation_node.rel_path`, when the input is given, IS that directory --
    the CalcJob stages `simulation_dir` rooted at exactly the run's own
    directory (`prepare_simulation_folder(sim_dir=run_dir)`). It is
    authoritative and preferred over `_detect_sim_prefix`'s heuristic: that
    heuristic reads a `read_data <path>` line containing `/build/` and returns
    everything before it, which for an AFM-family run recovers only the
    shared K-level ancestor of the per-layer `build/` directory
    (`self.relative_run_dir` in `builders/afm/base.py`) -- ONE level shallower
    than the run's actual directory (`.../K300/L{n}`, where `results/`,
    `data/` and `provenance.json` actually live).

    Without `simulation_node` (fix round 1, finding 2: `smart_submit`'s
    `config_path=None` path never sets one), that shallower guess used to be
    returned as-is, landing every staged path -- the run's own AND the shared
    ones -- one level off from what the rendered scripts actually read. If
    `simulation_dir` carries `_LEAF_NAME_ATTR`/`_SHARED_SUBDIRS_ATTR` (set by
    `prepare_simulation_folder`, never by `prepare_simulation_root`'s
    multi-run array path), the shallower guess is the shared ancestor and the
    leaf name is the missing last segment, so joining them recovers the same
    prefix `simulation_node.rel_path` would have given. `_detect_sim_prefix`
    alone is still the answer for a sheet-like family (no such nesting, so
    the guess is already the run's own directory) and for the array path.
    """
    sim_node = inputs.get('simulation_node', None)
    if sim_node is not None:
        return sim_node.rel_path

    ancestor = _detect_sim_prefix(repo, lammps_scripts)
    sim_folder = inputs.get('simulation_dir', None)
    if sim_folder is not None and ancestor:
        shared = sim_folder.base.attributes.get(_SHARED_SUBDIRS_ATTR, None)
        leaf_name = sim_folder.base.attributes.get(_LEAF_NAME_ATTR, None)
        if shared and leaf_name:
            return f"{ancestor}/{leaf_name}"
    return ancestor


def _retrieve_list_for(sim_prefix: str | None) -> list:
    """The CalcJob's `retrieve_list`: the retention rule, unmodified, plus
    `array_status/*`.

    `retrieve_patterns` (Task 13, verified against the real aiida-core
    execmanager; widened for a prefix by the Task 22 fix round so every entry
    nests consistently under it -- see `retention.py`) IS the retention
    guarantee for everything `ingest` reads -- it is used here as-is, never
    re-derived. Array-mode status files are a different concern (job progress
    polling, not something the parser ingests) and are always written to the
    sim root by `_build_array_wrapper` regardless of `sim_prefix`, so they are
    appended separately rather than folded into the retention list.
    """
    patterns = retrieve_patterns(sim_prefix) if sim_prefix else retrieve_patterns()
    return list(patterns) + ['array_status/*']


def _run_dir_in_retrieved(tmp_root: Path, run_node) -> Path:
    """Where the run's own directory landed inside the rebuilt retrieved tree.

    `prepare_for_submission` nests the ENTIRE retrieval under
    `run_node.rel_path` whenever `simulation_node` is given (`_sim_prefix_for`
    -- which is every real submission that can be ingested at all, since the
    parser itself requires `simulation_node`) -- `results/`, `data/` and
    everything else then sit at `tmp_root/<rel_path>/`, not at `tmp_root`
    itself. Falls back to `tmp_root` only for a run retrieved unprefixed
    (nothing to nest under -- a caller that never gave `simulation_node` at
    submission time either, so `_sim_prefix_for` fell back to
    `_detect_sim_prefix` and found nothing).

    NOT keyed on `provenance.json`: `prepare_simulation_folder`/`_stage_sim_dir`
    only walks `_SIM_SUBDIRS` (`lammps`, `data`, `results`, `provenance`, ...)
    under the run directory, and `provenance.json` is a bare FILE sitting
    directly in the run directory, not inside any of those -- it is never
    staged into `simulation_dir` at all, so a real prefixed retrieval never
    has one to check for. Checking for the directory itself is enough: the
    only thing that could put `tmp_root/<rel_path>` there is
    `prepare_for_submission` having prefixed everything under it.
    """
    prefixed = tmp_root / run_node.rel_path
    if prefixed.is_dir():
        return prefixed
    return tmp_root


def _group_results_from_node_safely(node) -> None:
    """`_group_results_from_node`, but never lets a failure escape.

    Filing a finished run's results into its set `Group` is bookkeeping on
    top of an already-successful calculation -- plumpy's `transition_failed`
    turns an exception raised from `on_terminated` into an EXCEPTED state for
    a calculation that otherwise finished (and parsed) correctly, which would
    misrepresent good results as failed over what is, at worst, a `Group`
    write hiccup. The results stay fully queryable by `run_uuid`/`set_uuid`
    either way (`create_results`'s own import path already tolerates the
    equivalent per-point failure, R3 of this task's first fix round); only
    `export_archive`'s group-based listing would miss this run, which is a
    much smaller problem than relabelling a finished calculation as failed.
    """
    try:
        _group_results_from_node(node)
    except Exception:  # pylint: disable=broad-except
        logger.exception(
            "Could not file %s's results into its set group; the "
            "calculation itself finished successfully.", node.pk)


def _supersede_older_results_safely(node) -> None:
    """`_supersede_older_results`, but never lets a failure escape -- the same
    reasoning as `_group_results_from_node_safely`: marking an older node
    stale is bookkeeping on top of a calculation that already finished."""
    try:
        _supersede_older_results(node)
    except Exception:  # pylint: disable=broad-except
        logger.exception(
            "Could not supersede older results for %s; the calculation "
            "itself finished successfully.", node.pk)


def _calc_results(node):
    """`(simulation_node, [results outputs])` of a finished CalcJob node."""
    from aiida.common.links import LinkType  # pylint: disable=import-outside-toplevel

    sim_node = getattr(node.inputs, 'simulation_node', None)
    if sim_node is None:
        return None, []
    outgoing = node.base.links.get_outgoing(link_type=LinkType.CREATE)
    return sim_node, [triple.node for triple in outgoing.all()
                      if triple.link_label.startswith('results__')]


def _supersede_older_results(node) -> None:
    """Mark every older live results node at the conditions this CalcJob just
    produced as superseded, so a resubmitted run keeps ONE live node per load
    point -- exactly what `ingest_run` does on the import path."""
    sim_node, results = _calc_results(node)
    if sim_node is None or not results:
        return
    from .ingest import supersede_older  # pylint: disable=import-outside-toplevel
    supersede_older(sim_node, results)


def _group_results_from_node(node) -> None:
    """File a finished CalcJob node's `results.*` outputs into its set group.

    Free function, not a method: it only reads `node`'s own inputs/outgoing
    links, so it can be exercised directly against a plain, manually-built
    `CalcJobNode` in a test, with no live `Process` (engine, runner, broker)
    required -- exactly the granularity `LammpsFrictionCalcJob.on_terminated`
    needs to delegate to.

    The link label to match is `results__point_0`, NOT `results.point_0`: a
    dotted string is only the developer-facing spelling accepted by
    `self.out()` (`Process.out` splits it on `spec().namespace_separator`,
    `'.'`, to nest it under the `results` output namespace); AiiDA link labels
    themselves allow only alphanumerics and underscores, so once the engine
    actually stores the output (`Process.update_outputs` -> `_flat_outputs`,
    `PORT_NAMESPACE_SEPARATOR = '__'`) the real link label is joined with a
    double underscore instead.
    """
    sim_node, results = _calc_results(node)
    if sim_node is None or not results:
        return
    from .ingest import add_results_to_group  # pylint: disable=import-outside-toplevel
    add_results_to_group(sim_node.set_label or sim_node.set_uuid, results)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class LammpsFrictionParser(Parser):
    """Parse retrieved outputs for FrictionSim2D LAMMPS runs."""

    def parse(self, **kwargs):  # type: ignore[override]
        """Parse retrieved files: rebuild the run tree, then ingest it.

        The retrieved `FolderData` reproduces the run directory's own shape
        (that IS the retention rule -- see `src.aiida.retention`), so it is
        written to a scratch directory and handed to `ingest_retrieved`
        exactly as a directory on disk would be. `Parser.out()` requires
        UNSTORED nodes so the engine can attach them to this calculation with
        real provenance links -- `ingest_retrieved` returns exactly those.

        The retrieved files are copied back into the local run directory
        (`parameters.local_sim_dir`) and the log/stdout outputs attached
        BEFORE `simulation_node` is checked: a run submitted untracked still
        ran, and its results must land in the tree for a later `aiida import`
        instead of being stranded in the retrieved folder. Only the ingest
        itself needs `simulation_node` -- without it there is no `rel_path` to
        locate the run by and no run node to attach results to.
        """
        retrieved = self.retrieved
        if retrieved is None:
            return self.exit_codes.ERROR_NO_RESULTS

        repo = retrieved.base.repository
        top_names = repo.list_object_names()
        all_names = [(root / name).as_posix()
                     for root, _dirs, files in repo.walk() for name in files]

        self._store_log(repo, 'log.lammps' in top_names)
        self._store_stdout(repo, [n for n in all_names if n.endswith('.out')])
        self._copy_local(repo, all_names)

        run_node = getattr(self.node.inputs, 'simulation_node', None)
        if run_node is None:
            return self.exit_codes.ERROR_NO_SIMULATION_NODE

        with TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            for rel in all_names:
                out = tmp_root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with repo.open(rel, 'rb') as src:
                    out.write_bytes(src.read())

            run_dir = _run_dir_in_retrieved(tmp_root, run_node)
            from .ingest import ingest_retrieved  # pylint: disable=import-outside-toplevel
            nodes = ingest_retrieved(run_dir, run_node.rel_path, run_node)

        if not nodes:
            return self.exit_codes.ERROR_NO_LOAD_POINTS
        for index, node in enumerate(nodes):
            self.out(f'results.point_{index}', node)
        return None

    def _store_log(self, repo, has_log):
        """Store LAMMPS log file as SinglefileData."""
        if not has_log:
            return
        with repo.open('log.lammps', 'rb') as handle:
            with NamedTemporaryFile(prefix='lammps_log_', suffix='.log',
                                    delete=True) as tmp:
                tmp.write(handle.read())
                tmp.flush()
                log_file = orm.SinglefileData(file=tmp.name)
        log_file.store()
        self.out('log_file', log_file)

    def _store_stdout(self, repo, stdout_names):
        """Store stdout files as FolderData."""
        if not stdout_names:
            return
        stdout_folder = orm.FolderData()
        for name in stdout_names:
            with repo.open(name, 'rb') as handle:
                stdout_folder.base.repository.put_object_from_filelike(handle, name)
        stdout_folder.store()
        self.out('stdout_folder', stdout_folder)

    def _copy_local(self, repo, names):
        """Copy outputs to local simulation directory if configured."""
        params = getattr(self.node.inputs, 'parameters', None)
        if params is None:
            return
        local_sim_dir = params.get_dict().get('local_sim_dir')
        if not local_sim_dir:
            return
        _copy_outputs_to_local(repo, names, Path(local_sim_dir))


# ---------------------------------------------------------------------------
# Preparation helpers (public API)
# ---------------------------------------------------------------------------

def prepare_simulation_folder(sim_dir: Path) -> orm.FolderData:
    """Create an AiiDA ``FolderData`` from a simulation directory.

    Args:
        sim_dir: Path to the simulation directory (must contain ``lammps/``).

    Returns:
        A new (unstored) ``FolderData`` node.
    """
    sim_dir = Path(sim_dir)
    if not (sim_dir / 'lammps').exists():
        raise FileNotFoundError(f"No lammps/ directory in {sim_dir}")

    folder_data = orm.FolderData()
    shared_subdirs = _stage_sim_dir(folder_data, sim_dir, relative_to=sim_dir)
    # Always recorded, even when nothing was shared (sheet-like families):
    # `_sim_prefix_for` only acts on it once `_SHARED_SUBDIRS_ATTR` is ALSO
    # set, so a harmless leaf name sitting unused on that path costs nothing.
    folder_data.base.attributes.set(_LEAF_NAME_ATTR, sim_dir.name)
    if shared_subdirs:
        folder_data.base.attributes.set(_SHARED_SUBDIRS_ATTR, sorted(shared_subdirs))
    return folder_data


def prepare_simulation_root(root_dir: Path, simulation_dirs: list) -> orm.FolderData:
    """Create a FolderData containing multiple simulation directories.

    Args:
        root_dir: Root directory containing simulation subdirectories.
        simulation_dirs: List of simulation directories under root_dir.

    Returns:
        A new (unstored) ``FolderData`` node with staged inputs.
    """
    root_dir = Path(root_dir)
    folder_data = orm.FolderData()
    for sim_dir in simulation_dirs:
        # Return value (which subdirs came from a parent fallback) is
        # per-sim_dir and meaningless once several runs share one FolderData
        # -- `relative_to=root_dir` already keys every file at its true
        # sim-root-relative path, so no later re-prefixing is needed here at
        # all (see `_sim_prefix_for`'s docstring on why this differs from
        # `prepare_simulation_folder`).
        _stage_sim_dir(folder_data, Path(sim_dir), relative_to=root_dir)
    return folder_data


def apply_options(builder, options: dict[str, Any] | None) -> None:
    """Apply metadata options to a CalcJob builder."""
    if not options:
        return
    for key in ('resources', 'max_wallclock_seconds', 'queue_name',
                'account', 'custom_scheduler_commands', 'prepend_text'):
        if key in options:
            setattr(builder.metadata.options, key, options[key])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _stage_sim_dir(folder_data: orm.FolderData, sim_dir: Path,
                   relative_to: Path) -> set[str]:
    """Stage simulation subdirectories into a FolderData node.

    Most `_SIM_SUBDIRS` live directly under `sim_dir`. AFM-family runs are
    nested one level deeper (`.../K<temp>/L<layers>`, see
    `AFMSimulation.build`/`write_inputs`): the run's own `lammps/`, `data/`,
    `results/` and `visuals/` sit at that leaf, but `build/` and
    `provenance/` (the built tip/sheet geometry and the potential files
    `pair_coeff` reads) are shared across every layer count and sit one level
    up, at `sim_dir.parent` -- exactly what `run/tree.py`'s
    `provenance_dir_for` already falls back to for the same reason. Only
    those two (`_SHAREABLE_SUBDIRS`, fix round 1 finding 2) are ever looked
    for at `sim_dir.parent` when missing at `sim_dir` -- restricted so an
    unrelated directory that happens to sit beside the run is never picked up
    as this run's own. Sheet-like families have no such nesting: `build`/
    `provenance` sit beside `lammps` at `sim_dir` itself, taken by the first
    branch below like everything else, so the parent fallback never triggers
    for them.

    Returns the names of the subdirs actually sourced from `sim_dir.parent`,
    for the caller to persist as `folder_data`'s `_SHARED_SUBDIRS_ATTR`
    (`prepare_simulation_folder`, one `sim_dir`) or to discard
    (`prepare_simulation_root`, several `sim_dir`s already keyed relative to
    the true sim root -- see that function's own comment).
    """
    shared_subdirs = set()
    for subdir in _SIM_SUBDIRS:
        sub_path = sim_dir / subdir
        if not sub_path.exists():
            if subdir not in _SHAREABLE_SUBDIRS or sim_dir.parent == sim_dir:
                continue
            sub_path = sim_dir.parent / subdir
            if not sub_path.exists():
                continue
        for file_path in sub_path.rglob('*'):
            if not file_path.is_file():
                continue
            try:
                rel = file_path.relative_to(relative_to)
            except ValueError:
                rel = file_path.relative_to(sim_dir.parent)
                shared_subdirs.add(subdir)
            folder_data.base.repository.put_object_from_file(
                str(file_path), str(rel),
            )
    return shared_subdirs


def _stage_files_into_sandbox(folder: Folder, repo, sim_prefix: str | None,
                              shared_subdirs: frozenset[str] = frozenset()) -> None:
    """Copy all repository files into the AiiDA sandbox folder.

    Every file is written twice: once at its own repository-relative path (so
    the fixed `-in lammps/{script}` entry point in `_make_code_info` always
    resolves), and once nested under a prefix (so the LAMMPS scripts' own
    sim-root-relative `read_data`/`include`/`pair_coeff` paths resolve too --
    see `_sim_prefix_for`). `shared_subdirs` are the top-level directories
    `_stage_sim_dir` sourced from the run directory's PARENT rather than the
    run directory itself (AFM's `build/` and `provenance/`): those nest under
    the PARENT of `sim_prefix` instead of `sim_prefix` itself, matching
    `relative_run_dir` rather than `relative_run_dir_layer` in the rendered
    scripts.
    """
    def _prefix_root(top_dir: str | None) -> str | None:
        if not sim_prefix:
            return None
        if top_dir in shared_subdirs:
            parent = Path(sim_prefix).parent
            return None if parent == Path('.') else str(parent)
        return sim_prefix

    for subdir in _SIM_SUBDIRS:
        folder.get_subfolder(subdir, create=True)
        prefix_root = _prefix_root(subdir)
        if prefix_root:
            folder.get_subfolder(f"{prefix_root}/{subdir}", create=True)

    for root, _dirs, files in repo.walk():
        top_dir = root.parts[0] if root.parts else None
        prefix_root = _prefix_root(top_dir)
        for file_name in files:
            rel_path = (root / file_name).as_posix()
            with repo.open(rel_path, 'rb') as src:
                payload = src.read()

            if root.as_posix() == '.':
                folder.create_file_from_filelike(io.BytesIO(payload), file_name)
            else:
                sub = folder.get_subfolder(root.as_posix(), create=True)
                sub.create_file_from_filelike(io.BytesIO(payload), file_name)

            if prefix_root:
                pref_root = (Path(prefix_root) / root).as_posix()
                pref = folder.get_subfolder(pref_root, create=True)
                pref.create_file_from_filelike(io.BytesIO(payload), file_name)


def _handle_array_mode(folder, sim_folder, params, scripts_override,
                       lammps_scripts) -> None:
    """Write array-mode helper files into the sandbox if needed."""
    if not params.get('array_mode'):
        return

    if 'array_map.txt' not in sim_folder.base.repository.list_object_names():
        raise ValueError("array_map.txt missing for array mode")

    if scripts_override:
        scripts_payload = "\n".join(lammps_scripts).encode('utf-8')
        folder.create_file_from_filelike(
            io.BytesIO(scripts_payload), 'array_scripts.txt',
        )

    wrapper = _build_array_wrapper(
        launcher_cmd=params.get('launcher_cmd', 'mpirun'),
        lammps_exec=params.get('lammps_executable', 'lmp'),
        lammps_flags=params.get('lammps_flags', '-l none'),
    )
    folder.create_file_from_filelike(
        io.BytesIO(wrapper.encode('utf-8')), 'run_array.sh',
    )


def _make_code_info(code_uuid: str, flag_list: list, script: str) -> CodeInfo:
    """Build a CodeInfo for a single LAMMPS script."""
    codeinfo = CodeInfo()
    codeinfo.code_uuid = code_uuid
    codeinfo.cmdline_params = flag_list + ['-in', f'lammps/{script}']
    codeinfo.stdout_name = f'{script}.out'
    codeinfo.withmpi = True
    return codeinfo


def _collect_lammps_scripts(sim_folder, scripts_override):
    """Return ordered LAMMPS scripts, or override list if provided."""
    if scripts_override:
        if isinstance(scripts_override, str):
            return [scripts_override]
        return [str(script) for script in scripts_override]

    repo = sim_folder.base.repository
    discovered = [
        name for name in repo.list_object_names('lammps')
        if name.endswith('.in')
    ]
    system = [s for s in discovered if s.lower() == 'system.in']
    slides = sorted(s for s in discovered
                    if s.lower() != 'system.in' and 'slide' in s.lower())
    others = sorted(s for s in discovered
                    if s.lower() != 'system.in' and 'slide' not in s.lower())
    return system + slides + others

def _detect_sim_prefix(repo, lammps_scripts):
    """Infer the simulation path prefix from LAMMPS scripts, if any."""
    pattern = re.compile(r"\bread_data\s+(\S+)")
    for script in lammps_scripts:
        path = f"lammps/{script}"
        try:
            with repo.open(path, 'r') as handle:
                for line in handle:
                    match = pattern.search(line)
                    if match and '/build/' in match.group(1):
                        return match.group(1).split('/build/', 1)[0]
        except FileNotFoundError:
            continue
    return None


def _build_array_wrapper(launcher_cmd: str, lammps_exec: str,
                         lammps_flags: str) -> str:
    """Build a TMPDIR staging wrapper for array jobs."""
    return f"""#!/usr/bin/env bash
set -euo pipefail

IDX="${{PBS_ARRAY_INDEX:-${{SLURM_ARRAY_TASK_ID:-1}}}}"

mapfile -t SIMS < array_map.txt
SIM_REL="${{SIMS[$((IDX-1))]:-}}"
if [[ -z "${{SIM_REL}}" ]]; then
    echo "No simulation for index ${{IDX}}" >&2
    exit 2
fi

SIM_ROOT="$PWD"
RUN_ROOT="${{TMPDIR:-/tmp}}/friction2d_${{IDX}}"
LMP_FLAGS="{lammps_flags}"

mkdir -p "$RUN_ROOT"
mkdir -p "$SIM_ROOT/array_status"

trap 'echo "{{\\\\\"index\\\\\": $IDX, \\\\\"simulation\\\\\": \\\\\"$SIM_REL\\\\\", \\\\\"status\\\\\": \\\\\"failed\\\\\"}}\" \\
    > "$SIM_ROOT/array_status/${{IDX}}.json"' ERR

echo "{{\\\\\"index\\\\\": $IDX, \\\\\"simulation\\\\\": \\\\\"$SIM_REL\\\\\", \\\\\"status\\\\\": \\\\\"running\\\\\"}}\" \\
    > "$SIM_ROOT/array_status/${{IDX}}.json"

rsync -a --exclude='*.lammpstrj' "$SIM_ROOT/$SIM_REL/" "$RUN_ROOT/$SIM_REL/"

cd "$RUN_ROOT/$SIM_REL"

if [[ -f "$SIM_ROOT/array_scripts.txt" ]]; then
    mapfile -t SCRIPTS < "$SIM_ROOT/array_scripts.txt"
else
    mapfile -t ALLSCRIPTS < <(find lammps -maxdepth 1 -type f -name '*.in' -printf '%f\\\\n' | sort)
    SYSTEM=()
    SLIDES=()
    OTHERS=()
    for s in "${{ALLSCRIPTS[@]}}"; do
        low="${{s,,}}"
        if [[ "$low" == "system.in" ]]; then
            SYSTEM+=("$s")
        elif [[ "$low" == *slide* ]]; then
            SLIDES+=("$s")
        else
            OTHERS+=("$s")
        fi
    done
    SCRIPTS=("${{SYSTEM[@]}}" "${{SLIDES[@]}}" "${{OTHERS[@]}}")
fi

for script in "${{SCRIPTS[@]}}"; do
    {launcher_cmd} {lammps_exec} $LMP_FLAGS -in "lammps/$script"
done

for sub in results visuals; do
    if [[ -d "$RUN_ROOT/$SIM_REL/$sub" ]]; then
        rsync -a "$RUN_ROOT/$SIM_REL/$sub" "$SIM_ROOT/$SIM_REL/"
    fi
done

echo "{{\\\\\"index\\\\\": $IDX, \\\\\"simulation\\\\\": \\\\\"$SIM_REL\\\\\", \\\\\"status\\\\\": \\\\\"completed\\\\\"}}\" \\
    > "$SIM_ROOT/array_status/${{IDX}}.json"
"""


def _copy_outputs_to_local(repo, names, local_sim_dir: Path) -> None:
    """Copy results and visuals into the local simulation directory."""
    results_dir = local_sim_dir / 'results'
    visuals_dir = local_sim_dir / 'visuals'
    results_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        kind, rel_name = _categorize_output(name)
        if kind is None:
            continue
        target_dir = results_dir if kind == 'results' else visuals_dir
        target_path = target_dir / rel_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with repo.open(name, 'rb') as handle:
            target_path.write_bytes(handle.read())


def _categorize_output(name: str) -> tuple[str | None, str]:
    """Categorize a retrieved file as 'results', 'visuals', or None."""
    for kind, sep in (('results', '/results/'), ('visuals', '/visuals/')):
        if sep in name:
            return kind, name.split(sep, 1)[1]
        if name.startswith(f'{kind}/'):
            return kind, name.split(f'{kind}/', 1)[1]
    if name.startswith('friction_'):
        return 'results', name
    if name.endswith('.lammpstrj'):
        return 'visuals', name
    return None, name
