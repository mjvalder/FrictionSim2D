"""AiiDA submission logic for FrictionSim2D simulations.

Provides both a high-level Python API (:func:`run_with_aiida`) and
an interactive CLI helper (:func:`smart_submit`) for submitting
LAMMPS friction simulations through AiiDA.

Submission functions (:func:`submit_simulation`, :func:`submit_batch`)
handle the mechanics of building CalcJob inputs and dispatching them to
the AiiDA daemon. Array jobs on an HPC are done with `FrictionSim2D hpc
generate` (PBS/SLURM array scripts, see `src/hpc/`) + qsub + rsync back +
`aiida import` -- AiiDA-managed array CalcJobs (`aiida submit --array`)
were removed; `aiida submit` submits one CalcJob per run.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from aiida import orm
from aiida.common.exceptions import NotExistent
from aiida.orm import QueryBuilder
from aiida.engine import run_get_node as _run_get_node
from aiida.engine import submit as aiida_submit

from ..core.config import GlobalSettings, load_settings
from ..run.simulate import run_simulations
from .calcjob import (
    LammpsFrictionCalcJob,
    prepare_simulation_folder,
    apply_options,
)

logger = logging.getLogger(__name__)

_SUBMISSION_EXCEPTIONS = (OSError, ValueError, KeyError, TypeError, RuntimeError)


def _dispatch(builder, daemon: bool):
    """Launch one CalcJob.

    `aiida.engine.run` needs no message broker, which is the difference that
    matters here: RabbitMQ is not part of the local or hpc-direct workflow.
    `--daemon` opts into the broker when one is actually running.
    """
    if daemon:
        return aiida_submit(builder)
    _result, node = _run_get_node(builder)
    return node


# ---------------------------------------------------------------------------
# High-level Python API
# ---------------------------------------------------------------------------

def run_with_aiida(
    config_file: Union[str, Path],
    model: str = 'afm',
    settings_file: Optional[Union[str, Path]] = None,
    output_root: Optional[Union[str, Path]] = None,
    auto_submit: bool = True,
    code_label: Optional[str] = None,
    auto_setup: bool = True,
    profile: Optional[str] = None,
    daemon: bool = False,
    label: Optional[str] = None,
    **resource_overrides,
) -> Tuple[List[Path], Path, Optional[List]]:
    """Run FrictionSim2D simulation with automatic AiiDA setup and submission.

    Handles generating simulation files via :func:`run_simulations`,
    auto-detecting the AiiDA code, building resource options, and
    submitting to the AiiDA daemon.

    Args:
        config_file: Path to simulation config (.ini, .yaml, or .json).
        model: Simulation model type (``'afm'`` or ``'sheetonsheet'``).
        output_root: Output directory (default: auto-generated).
        auto_submit: If ``True``, submit to AiiDA immediately.
        code_label: AiiDA code label (e.g. ``'lammps@hpc'``). Auto-detects if ``None``.
        auto_setup: Auto-load AiiDA profile if not loaded.
        profile: AiiDA profile name to load (default: current profile).
        daemon: If ``True``, submit through the AiiDA daemon (needs RabbitMQ).
            Default runs in-process via ``aiida.engine.run``, no broker needed.
        label: Set label for the fresh tree (default: the root directory name).
        **resource_overrides: Manual resource overrides (machines, mpiprocs,
            walltime_hours, queue, account, memory_mb, etc.).

    Returns:
        Tuple of ``(simulation_dirs, root_dir, process_nodes)``.
    """
    if auto_setup:
        _ensure_profile_loaded(profile)

    settings: GlobalSettings = load_settings(settings_file=settings_file)

    logger.info("Generating simulation files from %s", config_file)
    simulation_dirs, root_dir, _, _ = run_simulations(
        config_file=str(config_file),
        model=model,
        settings_file=settings_file,
        output_root=output_root,
        use_aiida=True,
    )
    logger.info("Generated %d simulation(s) in %s", len(simulation_dirs), root_dir)

    # Same as `run --aiida`: register the fresh tree (set + run nodes, no
    # results yet) so every CalcJob below finds its `simulation_node`.
    # Without it the parser has no run to attach results to (exit 321).
    from .ingest import ingest_root  # pylint: disable=import-outside-toplevel
    report = ingest_root(root_dir, label=label, profile=profile)
    if not report.ok():
        raise RuntimeError(
            f"Registering {root_dir} in AiiDA failed:\n{report.render()}")

    if not auto_submit:
        logger.info("auto_submit=False, skipping submission")
        return simulation_dirs, root_dir, None

    run_dirs = _run_dirs_under(root_dir)

    code = _get_code(code_label, interactive=False, settings_file=settings_file)
    options, parameters = _build_resources(settings=settings, **resource_overrides)

    logger.info("Submitting %d run(s) to AiiDA", len(run_dirs))
    processes = submit_batch(
        simulation_dirs=run_dirs,
        code_label=code,
        root_dir=root_dir,
        options=options,
        parameters=parameters,
        daemon=daemon,
    )
    pks = [p.pk for p in processes]
    logger.info("Submitted %d jobs: PKs=%s", len(processes), pks)

    return simulation_dirs, root_dir, processes


# ---------------------------------------------------------------------------
# CLI interactive submission
# ---------------------------------------------------------------------------

def smart_submit(
    simulation_dir: Path,
    settings_file: Optional[Union[str, Path]] = None,
    code_label: Optional[str] = None,
    scripts: Optional[str] = None,
    dry_run: bool = False,
    daemon: bool = False,
    profile: Optional[str] = None,
    **manual_overrides,
) -> List[orm.ProcessNode]:
    """Smart CLI submission with auto-detection and interactive prompting.

    `simulation_dir` is a `simulation_*` root, or any directory inside one (a
    model directory, a single run): the root is found by walking up to the
    nearest directory that holds `<model>/.../provenance.json`, and only the
    runs at or below `simulation_dir` are submitted. Each run is submitted
    with that root as `root_dir`, so its `FrictionSimulationData` (created by
    `run --aiida` or `aiida import`) is found and passed as
    `simulation_node` -- without it the parser cannot store results.

    Args:
        simulation_dir: A simulation root, or a directory inside one.
        code_label: AiiDA code label (auto-detects if ``None``).
        scripts: Comma-separated LAMMPS script names.
        dry_run: Preview only, don't submit.
        daemon: If ``True``, submit through the AiiDA daemon (needs RabbitMQ).
            Default runs in-process via ``aiida.engine.run``, no broker needed.
        profile: AiiDA profile to load (default: the configured one).
        **manual_overrides: Resource overrides (machines, mpiprocs, walltime, etc.).

    Returns:
        List of ``ProcessNode`` instances (empty if dry_run or cancelled).
    """
    import click  # pylint: disable=import-outside-toplevel

    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel
    from .ingest import find_run_by_rel_path  # pylint: disable=import-outside-toplevel

    try:
        load_aiida_profile(profile)
    except Exception as exc:  # pylint: disable=broad-except
        raise click.ClickException(
            f"Could not load AiiDA profile {profile or '(default)'}: {exc}") from exc

    settings: GlobalSettings = load_settings(settings_file=settings_file)

    try:
        root = find_simulation_root(simulation_dir)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    simulation_dirs = _run_dirs_under(root, within=simulation_dir)
    if not simulation_dirs:
        raise click.ClickException(
            f"No simulation runs found in {simulation_dir}"
        )
    click.echo(f"Found {len(simulation_dirs)} run(s) under {root}")

    untracked = [d for d in simulation_dirs if find_run_by_rel_path(d, root) is None]
    if untracked:
        click.echo(
            f"WARNING: {len(untracked)} run(s) are not registered in AiiDA; their "
            f"results will be copied back into the tree but NOT stored. Register "
            f"them first with: FrictionSim2D aiida import {root}")

    code = _get_code(code_label, interactive=True, settings_file=settings_file)
    click.echo(f"Using code: {code.full_label}")

    options, parameters = _build_resources(
        settings=settings, scripts=scripts, **manual_overrides,
    )

    _preview_submission(
        simulation_dirs=simulation_dirs,
        code=code,
        options=options,
        parameters=parameters,
        settings=settings,
    )

    if dry_run:
        click.echo("\n[DRY RUN] No jobs submitted")
        return []

    if not click.confirm("\nProceed with submission?", default=True):
        click.echo("Submission cancelled")
        return []

    click.echo("\nSubmitting jobs..." if daemon else "\nRunning jobs in-process...")
    processes = submit_batch(
        simulation_dirs=simulation_dirs,
        code_label=code,
        root_dir=root,
        options=options,
        parameters=parameters,
        daemon=daemon,
    )

    if not processes:
        raise click.ClickException(
            f"None of the {len(simulation_dirs)} run(s) could be submitted; "
            "see the log for each failure.")
    click.echo(describe_processes(processes, daemon=daemon,
                                  n_requested=len(simulation_dirs)))
    return processes


def describe_processes(processes: list, daemon: bool, n_requested: int) -> str:
    """One summary line per submission outcome.

    With the daemon the processes are only queued, so "Submitted" is true.
    In-process (`aiida.engine.run`, the default) they have already FINISHED
    by the time this is called, so saying "submitted" would hide a failed
    calculation behind a success message: report each exit status instead.
    """
    pks = [p.pk for p in processes]
    skipped = n_requested - len(processes) if n_requested > len(processes) else 0
    tail = f" ({skipped} could not be submitted)" if skipped else ''
    if daemon:
        return f"Submitted {len(processes)} job(s) (PKs={pks}){tail}"
    failed = [p for p in processes if not getattr(p, 'is_finished_ok', False)]
    lines = [f"Ran {len(processes)} job(s): {len(processes) - len(failed)} "
             f"finished ok, {len(failed)} failed{tail}"]
    for proc in failed:
        lines.append(f"  PK {proc.pk}: exit status {getattr(proc, 'exit_status', None)} "
                     f"{getattr(proc, 'exit_message', '') or ''}".rstrip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Submission functions
# ---------------------------------------------------------------------------

def submit_simulation(
    sim_dir: Path,
    code_label: Any,
    sim_node: Any = None,
    options: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    daemon: bool = False,
) -> orm.ProcessNode:
    """Submit a single simulation via AiiDA.

    Args:
        sim_dir: Path to the simulation directory.
        code_label: AiiDA code label (e.g. ``'lammps@hpc'``) or Code node.
        sim_node: Optional ``FrictionSimulationData`` for tracking.
        options: Optional dict overriding resource/walltime settings.
        parameters: Optional dict with LAMMPS runtime parameters.
        daemon: If ``True``, submit through the AiiDA daemon (needs RabbitMQ).
            Default runs in-process via ``aiida.engine.run``, no broker needed.

    Returns:
        The submitted (or, without ``daemon``, already-finished) ``ProcessNode``.
    """
    code = orm.load_code(code_label) if isinstance(code_label, str) else code_label
    folder = prepare_simulation_folder(sim_dir)
    folder.store()

    builder = LammpsFrictionCalcJob.get_builder()  # type: ignore[attr-defined]
    builder.code = code
    builder.simulation_dir = folder

    if sim_node:
        builder.simulation_node = sim_node

    merged_params = dict(parameters or {})
    merged_params.setdefault('local_sim_dir', str(sim_dir))
    builder.parameters = orm.Dict(merged_params)

    if options:
        apply_options(builder, options)

    process = _dispatch(builder, daemon)

    if sim_node and not sim_node.is_stored:
        sim_node.store()
    if sim_node:
        sim_node.job_id = str(process.pk)
        sim_node.status = 'submitted'

    logger.info("Submitted CalcJob PK=%d for %s", process.pk, sim_dir.name)
    return process


def submit_batch(
    simulation_dirs: list,
    code_label: Any,
    root_dir: Optional[Path] = None,
    options: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    daemon: bool = False,
) -> list:
    """Submit a batch of simulations via AiiDA.

    Args:
        simulation_dirs: List of simulation directories.
        code_label: AiiDA code label or Code node.
        root_dir: The simulation root `simulation_dirs` are relative to. When
            given, each directory's already-stored `FrictionSimulationData`
            (from `run --aiida` -> `ingest_root`) is looked up and passed
            through for tracking. `None` means submit untracked -- a
            directory built without `--aiida` simply has no node to find.
        options: Optional dict overriding resource/walltime settings.
        parameters: Optional dict with LAMMPS runtime parameters.
        daemon: If ``True``, submit through the AiiDA daemon (needs RabbitMQ).
            Default runs in-process via ``aiida.engine.run``, no broker needed.

    Returns:
        List of submitted ``ProcessNode`` instances.
    """
    from .ingest import find_run_by_rel_path  # pylint: disable=import-outside-toplevel

    processes = []
    for sim_dir in simulation_dirs:
        sim_dir = Path(sim_dir)

        sim_node = None
        if root_dir:
            sim_node = find_run_by_rel_path(sim_dir, root_dir)

        try:
            proc = submit_simulation(
                sim_dir, code_label,
                sim_node=sim_node,
                options=options,
                parameters=parameters,
                daemon=daemon,
            )
            processes.append(proc)
        except _SUBMISSION_EXCEPTIONS:
            logger.warning("Failed to submit %s", sim_dir, exc_info=True)

    return processes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_profile_loaded(profile: Optional[str] = None) -> None:
    """Load `profile` the same way every other command does.

    Delegates to `load_aiida_profile` so an omitted `profile` resolves to
    `settings.aiida.profile` here too -- the `ingest_root` call that follows
    in `run_with_aiida` resolves it that way, and two different resolutions
    would try to switch profiles mid-call.
    """
    from . import load_aiida_profile  # pylint: disable=import-outside-toplevel

    try:
        load_aiida_profile(profile)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load AiiDA profile. Run 'FrictionSim2D aiida setup' first."
        ) from exc


def _get_code(code_label: Optional[str], interactive: bool = False,
              settings_file=None) -> orm.Code:
    """Resolve the LAMMPS code: explicit label, then settings, then discovery.

    An EXPLICITLY passed `code_label` is authoritative: if it isn't
    registered, that's an error, full stop. A label that came from
    `settings.aiida.lammps_code_label` instead is only a best-effort
    default -- `AiidaSettings.lammps_code_label` always has a value (it is
    not `Optional`), so treating it the same as an explicit label would make
    the `%lammps%` discovery scan below unreachable for anyone who has not
    edited settings.yaml to match whatever `aiida setup` actually registered.
    A settings-provided label that isn't registered falls through to
    discovery instead of raising.

    Args:
        code_label: Code label (e.g. ``'lammps@hpc'``). If ``None``, falls
            back to ``settings.aiida.lammps_code_label`` before scanning.
        interactive: If ``True``, prompt user to select when ambiguous (CLI mode).
            If ``False``, raise ``ValueError`` on ambiguity (API mode).
        settings_file: Explicit settings.yaml path for the fallback lookup.

    Returns:
        AiiDA Code node.
    """
    explicit = bool(code_label)
    if not code_label:
        # `load_settings` is already imported at module scope (line 23);
        # re-importing it locally here would just shadow it.
        try:
            code_label = load_settings(settings_file=settings_file).aiida.lammps_code_label
        except (OSError, ValueError, AttributeError):
            code_label = None

    if code_label:
        try:
            return orm.load_code(code_label)
        except NotExistent as exc:
            if not explicit:
                logger.debug(
                    "Settings code label '%s' is not registered; falling "
                    "back to discovery.", code_label)
            else:
                msg = (
                    f"Code '{code_label}' not found. "
                    "Run 'FrictionSim2D aiida setup' to configure codes."
                )
                if interactive:
                    import click  # pylint: disable=import-outside-toplevel
                    raise click.ClickException(msg) from exc
                raise ValueError(msg) from exc

    qb = QueryBuilder()
    qb.append(orm.Code, filters={'label': {'like': '%lammps%'}})
    codes = [code for [code] in qb.all()]

    if not codes:
        msg = (
            "No LAMMPS codes found in AiiDA. "
            "Run 'FrictionSim2D aiida setup' to configure codes."
        )
        if interactive:
            import click  # pylint: disable=import-outside-toplevel
            raise click.ClickException(msg)
        raise ValueError(msg)

    if len(codes) == 1:
        logger.info("Auto-detected code: %s", codes[0].full_label)
        return codes[0]

    if interactive:
        import click  # pylint: disable=import-outside-toplevel
        click.echo("\nAvailable LAMMPS codes:")
        for idx, code in enumerate(codes, 1):
            click.echo(f"  {idx}. {code.full_label}")
        choice = click.prompt(
            "Select code",
            type=click.IntRange(1, len(codes)),
            default=1,
        )
        return codes[choice - 1]

    code_labels = [c.full_label for c in codes]
    raise ValueError(
        f"Multiple LAMMPS codes found: {code_labels}. "
        "Please specify code_label explicitly."
    )


def _build_resources(
    settings: GlobalSettings,
    scripts: Optional[str] = None,
    **overrides,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build resource options and parameters for AiiDA submission.

    Args:
        settings: ``GlobalSettings`` instance.
        scripts: Comma-separated script names (CLI only).
        **overrides: Manual resource overrides.

    Returns:
        Tuple of ``(options, parameters)`` dicts for CalcJob.
    """
    hpc = settings.hpc
    num_machines = overrides.get('machines', hpc.num_nodes)
    num_mpiprocs = overrides.get('mpiprocs', hpc.num_cpus)

    walltime = overrides.get('walltime') or overrides.get('walltime_hours')
    if walltime is None:
        max_wallclock_seconds = hpc.walltime_hours * 3600
    elif isinstance(walltime, str) and ':' in walltime:
        max_wallclock_seconds = _parse_walltime(walltime)
    else:
        max_wallclock_seconds = int(float(walltime) * 3600)

    queue_name = overrides.get('queue', hpc.queue or hpc.partition)
    account = overrides.get('account') or overrides.get('project') or hpc.account or None
    memory_mb = overrides.get(
        'memory_mb',
        overrides.get('memory', hpc.memory_gb * 1024 if hpc.memory_gb else None),
    )

    prepend_text = overrides.get('prepend_text')
    if prepend_text is None and hpc.modules:
        prepend_text = '\n'.join(f"module load {m}" for m in hpc.modules)

    custom_commands = overrides.get('custom_scheduler_commands')
    if memory_mb:
        mem_gb = memory_mb // 1024
        if hpc.scheduler_type == 'slurm':
            mem_cmd = f"#SBATCH --mem={mem_gb}gb"
        else:
            mem_cmd = f"#PBS -l mem={mem_gb}gb"
        custom_commands = f"{custom_commands}\n{mem_cmd}" if custom_commands else mem_cmd

    options: Dict[str, Any] = {
        'resources': {
            'num_machines': num_machines,
            'num_mpiprocs_per_machine': num_mpiprocs,
        },
        'max_wallclock_seconds': max_wallclock_seconds,
    }
    if queue_name:
        options['queue_name'] = queue_name
    if account:
        options['account'] = account
    if prepend_text:
        options['prepend_text'] = prepend_text
    if custom_commands:
        options['custom_scheduler_commands'] = custom_commands

    parameters: Dict[str, Any] = {}
    if scripts:
        script_list = [s.strip() for s in scripts.split(',')]
        parameters['lammps_scripts'] = script_list
    else:
        lammps_scripts = overrides.get('lammps_scripts', hpc.lammps_scripts)
        if lammps_scripts:
            parameters['lammps_scripts'] = lammps_scripts
    if 'lammps_flags' in overrides:
        parameters['lammps_flags'] = overrides['lammps_flags']

    return options, parameters


def find_simulation_root(path: Union[str, Path]) -> Path:
    """The `simulation_*` root `path` is, or lies inside.

    Walks up from `path` (inclusive) to the nearest directory that holds a
    `<model>/` child with at least one discoverable run -- the same
    `discover_runs` walk `aiida import` uses, so submission and ingest agree
    on what a run is and on the root its `rel_path` is relative to.

    Raises:
        ValueError: When no such directory exists at or above `path`.
    """
    from ..run.naming import MODELS  # pylint: disable=import-outside-toplevel
    from ..run.tree import discover_runs  # pylint: disable=import-outside-toplevel

    path = Path(path).resolve()
    for candidate in (path, *path.parents):
        if any((candidate / model).is_dir() for model in MODELS) \
                and discover_runs(candidate):
            return candidate
    raise ValueError(
        f"{path} is not inside a simulation tree: no <model>/.../provenance.json "
        "found at or above it.")


def _run_dirs_under(root: Union[str, Path],
                    within: Optional[Union[str, Path]] = None) -> List[Path]:
    """Every run directory `discover_runs(root)` finds, optionally restricted
    to those at or below `within`, in `rel_path` order."""
    from ..run.tree import discover_runs  # pylint: disable=import-outside-toplevel

    root = Path(root).resolve()
    limit = Path(within).resolve() if within is not None else root
    return [loc.run_dir for loc in discover_runs(root)
            if loc.run_dir == limit or limit in loc.run_dir.parents]


def _parse_walltime(walltime_str: str) -> int:
    """Parse walltime string to seconds.

    Accepts ``'HH:MM:SS'``, ``'HH:MM'``, or hours as a number.

    Args:
        walltime_str: Walltime string.

    Returns:
        Walltime in seconds.
    """
    if ':' in walltime_str:
        parts = walltime_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            hours, minutes = map(int, parts)
            return hours * 3600 + minutes * 60
        raise ValueError(f"Invalid walltime format: {walltime_str}")
    return int(float(walltime_str) * 3600)


def _preview_submission(
    simulation_dirs: List[Path],
    code: orm.Code,
    options: Dict[str, Any],
    parameters: Dict[str, Any],
    settings: GlobalSettings,
) -> None:
    """Display submission configuration preview (CLI only)."""
    import click  # pylint: disable=import-outside-toplevel

    click.echo("\n" + "=" * 60)
    click.echo("SUBMISSION PREVIEW")
    click.echo("=" * 60)

    click.echo(f"Code:        {code.full_label}")
    click.echo(f"Simulations: {len(simulation_dirs)}")

    resources = options.get('resources', {})
    click.echo("\nResources:")
    click.echo(f"  Nodes:     {resources.get('num_machines', 1)}")
    click.echo(f"  CPUs/node: {resources.get('num_mpiprocs_per_machine', 32)}")

    walltime_sec = options.get('max_wallclock_seconds', 0)
    walltime_hours = walltime_sec / 3600
    click.echo(f"  Walltime:  {walltime_hours:.1f}h ({walltime_sec}s)")

    if options.get('queue_name'):
        click.echo(f"  Queue:     {options['queue_name']}")
    if options.get('account'):
        click.echo(f"  Account:   {options['account']}")

    lammps_scripts = parameters.get('lammps_scripts', settings.hpc.lammps_scripts)
    click.echo(f"\nLAMMPS scripts: {', '.join(lammps_scripts)}")

    click.echo("\nSimulations:")
    for idx, sim_dir in enumerate(simulation_dirs[:5], 1):
        click.echo(f"  {idx}. {sim_dir.name}")
    if len(simulation_dirs) > 5:
        click.echo(f"  ... and {len(simulation_dirs) - 5} more")

    click.echo("=" * 60)
