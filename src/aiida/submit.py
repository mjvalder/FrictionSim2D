"""AiiDA submission logic for FrictionSim2D simulations.

Provides both a high-level Python API (:func:`run_with_aiida`) and
an interactive CLI helper (:func:`smart_submit`) for submitting
LAMMPS friction simulations through AiiDA.

Submission functions (:func:`submit_simulation`, :func:`submit_batch`,
:func:`submit_array`) handle the mechanics of building CalcJob inputs
and dispatching them to the AiiDA daemon.
"""

import logging
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from aiida import orm
from aiida.common.exceptions import NotExistent
from aiida.orm import QueryBuilder
from aiida.engine import submit as aiida_submit

from ..core.config import GlobalSettings, load_settings
from ..core.run import run_simulations, layer_aware_path_sort_key
from .calcjob import (
    LammpsFrictionCalcJob,
    prepare_simulation_folder,
    prepare_simulation_root,
    apply_options,
)

logger = logging.getLogger(__name__)


def _build_array_directive(scheduler: str, array_size: int) -> str:
    """Build scheduler-specific array directive."""
    if scheduler == 'slurm':
        return f"#SBATCH --array=1-{array_size}"
    return f"#PBS -J 1-{array_size}"

_SUBMISSION_EXCEPTIONS = (OSError, ValueError, KeyError, TypeError, RuntimeError)


# ---------------------------------------------------------------------------
# High-level Python API
# ---------------------------------------------------------------------------

def run_with_aiida(
    config_file: Union[str, Path],
    model: str = 'afm',
    output_root: Optional[Union[str, Path]] = None,
    auto_submit: bool = True,
    code_label: Optional[str] = None,
    use_array: bool = False,
    auto_setup: bool = True,
    profile_name: Optional[str] = None,
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
        use_array: Submit as a single array job instead of batch.
        auto_setup: Auto-load AiiDA profile if not loaded.
        profile_name: AiiDA profile name to load (default: current profile).
        **resource_overrides: Manual resource overrides (machines, mpiprocs,
            walltime_hours, queue, account, memory_mb, etc.).

    Returns:
        Tuple of ``(simulation_dirs, root_dir, process_nodes)``.
    """
    if auto_setup:
        _ensure_profile_loaded(profile_name)

    settings: GlobalSettings = load_settings()

    logger.info("Generating simulation files from %s", config_file)
    simulation_dirs, root_dir, _, _ = run_simulations(
        config_file=str(config_file),
        model=model,
        output_root=output_root,
        use_aiida=True,
    )
    logger.info("Generated %d simulation(s) in %s", len(simulation_dirs), root_dir)

    if not auto_submit:
        logger.info("auto_submit=False, skipping submission")
        return simulation_dirs, root_dir, None

    code = _get_code(code_label, interactive=False)
    options, parameters = _build_resources(settings=settings, **resource_overrides)

    logger.info("Submitting %d simulation(s) to AiiDA", len(simulation_dirs))
    if use_array:
        process = submit_array(
            simulation_root=root_dir,
            simulation_dirs=simulation_dirs,
            code_label=code,
            options=options,
            parameters=parameters,
            scheduler=settings.hpc.scheduler_type,
        )
        processes = [process]
        logger.info("Submitted array job with PK=%d", process.pk)
    else:
        processes = submit_batch(
            simulation_dirs=simulation_dirs,
            code_label=code,
            config_path=Path(config_file),
            options=options,
            parameters=parameters,
        )
        pks = [p.pk for p in processes]
        logger.info("Submitted %d jobs: PKs=%s", len(processes), pks)

    return simulation_dirs, root_dir, processes


# ---------------------------------------------------------------------------
# CLI interactive submission
# ---------------------------------------------------------------------------

def smart_submit(
    simulation_dir: Path,
    code_label: Optional[str] = None,
    use_array: bool = False,
    scripts: Optional[str] = None,
    dry_run: bool = False,
    **manual_overrides,
) -> List[orm.ProcessNode]:
    """Smart CLI submission with auto-detection and interactive prompting.

    Args:
        simulation_dir: Path to simulation directory or root.
        code_label: AiiDA code label (auto-detects if ``None``).
        use_array: Submit as array job.
        scripts: Comma-separated LAMMPS script names.
        dry_run: Preview only, don't submit.
        **manual_overrides: Resource overrides (machines, mpiprocs, walltime, etc.).

    Returns:
        List of submitted ``ProcessNode`` instances (empty if dry_run).
    """
    import click  # pylint: disable=import-outside-toplevel

    settings: GlobalSettings = load_settings()

    code = _get_code(code_label, interactive=True)
    click.echo(f"Using code: {code.full_label}")

    simulation_dirs = _find_lammps_dirs(simulation_dir)
    if not simulation_dirs:
        raise click.ClickException(
            f"No simulation directories found in {simulation_dir}"
        )
    click.echo(f"Found {len(simulation_dirs)} simulation(s)")

    options, parameters = _build_resources(
        settings=settings, scripts=scripts, **manual_overrides,
    )

    _preview_submission(
        simulation_dirs=simulation_dirs,
        code=code,
        use_array=use_array,
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

    click.echo("\nSubmitting jobs...")
    if use_array:
        process = submit_array(
            simulation_root=simulation_dir,
            simulation_dirs=simulation_dirs,
            code_label=code,
            options=options,
            parameters=parameters,
            scheduler=settings.hpc.scheduler_type,
        )
        processes = [process]
        click.echo(f"Submitted array job (PK={process.pk})")
    else:
        processes = submit_batch(
            simulation_dirs=simulation_dirs,
            code_label=code,
            config_path=None,
            options=options,
            parameters=parameters,
        )
        pks = [p.pk for p in processes]
        click.echo(f"Submitted {len(processes)} jobs (PKs={pks[0]}-{pks[-1]})")

    return processes


# ---------------------------------------------------------------------------
# Submission functions
# ---------------------------------------------------------------------------

def submit_simulation(
    sim_dir: Path,
    code_label: Any,
    sim_node: Any = None,
    prov_node: Any = None,
    options: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> orm.ProcessNode:
    """Submit a single simulation via AiiDA.

    Args:
        sim_dir: Path to the simulation directory.
        code_label: AiiDA code label (e.g. ``'lammps@hpc'``) or Code node.
        sim_node: Optional ``FrictionSimulationData`` for tracking.
        prov_node: Optional ``FrictionProvenanceData`` for provenance.
        options: Optional dict overriding resource/walltime settings.
        parameters: Optional dict with LAMMPS runtime parameters.

    Returns:
        The submitted ``ProcessNode``.
    """
    code = orm.load_code(code_label) if isinstance(code_label, str) else code_label
    folder = prepare_simulation_folder(sim_dir)
    folder.store()

    builder = LammpsFrictionCalcJob.get_builder()  # type: ignore[attr-defined]
    builder.code = code
    builder.simulation_dir = folder

    if sim_node:
        builder.simulation_node = sim_node
    if prov_node:
        builder.provenance = prov_node

    merged_params = dict(parameters or {})
    merged_params.setdefault('local_sim_dir', str(sim_dir))
    builder.parameters = orm.Dict(merged_params)

    if options:
        apply_options(builder, options)

    process = aiida_submit(builder)

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
    config_path: Optional[Path] = None,
    options: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> list:
    """Submit a batch of simulations via AiiDA.

    Args:
        simulation_dirs: List of simulation directories.
        code_label: AiiDA code label or Code node.
        config_path: Original config file path (for registration).
        options: Optional dict overriding resource/walltime settings.
        parameters: Optional dict with LAMMPS runtime parameters.

    Returns:
        List of submitted ``ProcessNode`` instances.
    """
    from .integration import register_single_simulation  # pylint: disable=import-outside-toplevel

    processes = []
    for sim_dir in simulation_dirs:
        sim_dir = Path(sim_dir)

        uuid = None
        if config_path:
            uuid = register_single_simulation(sim_dir, config_path)

        sim_node = None
        prov_node = None
        if uuid:
            sim_node = orm.load_node(uuid)
            if sim_node.provenance_uuid:
                prov_node = orm.load_node(sim_node.provenance_uuid)

        try:
            proc = submit_simulation(
                sim_dir, code_label,
                sim_node=sim_node, prov_node=prov_node,
                options=options,
                parameters=parameters,
            )
            processes.append(proc)
        except _SUBMISSION_EXCEPTIONS:
            logger.warning("Failed to submit %s", sim_dir, exc_info=True)

    return processes


def submit_array(
    simulation_root: Path,
    simulation_dirs: list,
    code_label: Any,
    options: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    scheduler: str = 'pbs',
) -> orm.ProcessNode:
    """Submit a single array job for multiple simulations.

    Args:
        simulation_root: Root directory containing simulation subdirectories.
        simulation_dirs: List of simulation directories.
        code_label: AiiDA code label or Code node.
        options: Optional dict overriding resource/walltime settings.
        parameters: Optional dict with runtime parameters.
        scheduler: Scheduler type (``'pbs'`` or ``'slurm'``).

    Returns:
        The submitted ``ProcessNode``.
    """
    code = orm.load_code(code_label) if isinstance(code_label, str) else code_label
    sim_root = Path(simulation_root)

    ordered_dirs = sorted(
        [Path(p) for p in simulation_dirs],
        key=lambda p: layer_aware_path_sort_key(str(p.relative_to(sim_root))),
    )
    rel_paths = [str(p.relative_to(sim_root)) for p in ordered_dirs]

    folder = prepare_simulation_root(sim_root, ordered_dirs)
    map_payload = "\n".join(rel_paths).encode('utf-8')
    folder.base.repository.put_object_from_filelike(
        cast(Any, io.BytesIO(map_payload)),
        'array_map.txt',
    )
    folder.store()

    builder = LammpsFrictionCalcJob.get_builder()  # type: ignore[attr-defined]
    builder.code = code
    builder.simulation_dir = folder

    merged_params = dict(parameters or {})
    merged_params['array_mode'] = True
    builder.parameters = orm.Dict(merged_params)

    if options:
        apply_options(builder, options)

    array_size = len(rel_paths)
    if array_size > 0:
        array_cmd = _build_array_directive(scheduler, array_size)
        metadata = getattr(builder, 'metadata')
        existing = getattr(metadata.options, 'custom_scheduler_commands', None)
        combined = "\n".join([c for c in [existing, array_cmd] if c])
        metadata.options.custom_scheduler_commands = combined

    metadata = getattr(builder, 'metadata')
    prepend = getattr(metadata.options, 'prepend_text', '') or ''
    metadata.options.prepend_text = "\n".join(
        line for line in [prepend, 'bash run_array.sh', 'exit $?'] if line
    )

    return aiida_submit(builder)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_profile_loaded(profile_name: Optional[str] = None) -> None:
    """Ensure an AiiDA profile is loaded."""
    from aiida.manage.configuration import get_profile, load_profile  # pylint: disable=import-outside-toplevel

    try:
        current = get_profile()
        if current:
            logger.debug("AiiDA profile '%s' already loaded", current.name)
            return
    except Exception:  # pylint: disable=broad-except
        pass

    try:
        if profile_name:
            load_profile(profile_name)
            logger.info("Loaded AiiDA profile '%s'", profile_name)
        else:
            load_profile()
            logger.info("Loaded default AiiDA profile")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load AiiDA profile. Run 'FrictionSim2D aiida setup' first."
        ) from exc


def _get_code(code_label: Optional[str], interactive: bool = False) -> orm.Code:
    """Get AiiDA code by label or auto-detect.

    Args:
        code_label: Code label (e.g. ``'lammps@hpc'``). If ``None``, auto-detects.
        interactive: If ``True``, prompt user to select when ambiguous (CLI mode).
            If ``False``, raise ``ValueError`` on ambiguity (API mode).

    Returns:
        AiiDA Code node.
    """
    if code_label:
        try:
            return orm.load_code(code_label)
        except NotExistent as exc:
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


def _find_lammps_dirs(simulation_dir: Path) -> List[Path]:
    """Find all simulation directories (containing ``lammps/`` subdirectory).

    Args:
        simulation_dir: Root directory to search.

    Returns:
        Sorted list of simulation directories.
    """
    simulation_dir = Path(simulation_dir)

    if (simulation_dir / 'lammps').exists():
        return [simulation_dir]

    sim_dirs = []
    for subdir in simulation_dir.iterdir():
        if subdir.is_dir() and (subdir / 'lammps').exists():
            sim_dirs.append(subdir)

    return sorted(sim_dirs, key=lambda p: layer_aware_path_sort_key(p.name))


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
        elif len(parts) == 2:
            hours, minutes = map(int, parts)
            return hours * 3600 + minutes * 60
        else:
            raise ValueError(f"Invalid walltime format: {walltime_str}")
    return int(float(walltime_str) * 3600)


def _preview_submission(
    simulation_dirs: List[Path],
    code: orm.Code,
    use_array: bool,
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
    click.echo(f"Mode:        {'Array job' if use_array else 'Batch jobs'}")

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
