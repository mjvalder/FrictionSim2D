"""Command Line Interface for FrictionSim2D.

Unified CLI for simulation execution, HPC script generation, and AiiDA integration.
"""

import logging
import sys
import shutil
import tarfile
from pathlib import Path
from typing import List, Dict, Any, Optional, cast, NoReturn
from importlib import resources as pkg_resources
import yaml
import click

from src.core.config import load_settings
from src.core.run import run_simulations, _build_hpc_manifest_entries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _ensure_hpc_settings(hpc_settings) -> None:
    """Ensure required HPC settings are set, prompting when missing."""
    if not getattr(hpc_settings, 'log_dir', None):
        hpc_settings.log_dir = click.prompt(
            "Absolute path for HPC log files",
            type=str
        )

    if not getattr(hpc_settings, 'modules', None):
        modules_raw = click.prompt(
            "Modules to load (comma-separated)",
            type=str
        )
        hpc_settings.modules = [
            m.strip() for m in modules_raw.split(',') if m.strip()
        ]

    if getattr(hpc_settings, 'use_tmpdir', False) and not getattr(hpc_settings, 'scratch_dir', None):
        hpc_settings.scratch_dir = click.prompt(
            "Scratch directory (e.g. $EPHEMERAL or $TMPDIR)",
            type=str,
            default="$TMPDIR"
        )


def _run_simulation(model: str, config_file: str, output_dir: str,
                    use_aiida: bool, generate_hpc: bool,
                    hpc_name: Optional[str], run_local: bool):
    _ = (hpc_name, run_local)

    if use_aiida:
        from src.aiida import AIIDA_AVAILABLE
        if not AIIDA_AVAILABLE:
            click.echo("⚠️  AiiDA not available. Install with:", err=True)
            click.echo("   conda install -c conda-forge aiida-core", err=True)
            raise click.Abort()

    created_simulations, simulation_root, configs_to_run, defaults = run_simulations(
        config_file=config_file,
        model=model,
        output_root=Path(output_dir),
        ensure_hpc_settings=_ensure_hpc_settings,
        use_aiida=use_aiida,
        generate_hpc=generate_hpc,
    )

    click.echo(f"📋 Found {len(configs_to_run)} simulation configurations")

    if created_simulations and use_aiida:
        click.echo(f"\n📦 Registering {len(created_simulations)} simulations in AiiDA...")
        _register_simulations_aiida(created_simulations, Path(config_file))

    click.echo(f"\n✅ Generation complete: {len(created_simulations)}/{len(configs_to_run)} successful")
    click.echo(f"📂 Output directory: {simulation_root.absolute()}")

# =============================================================================
# MAIN CLI GROUP
# =============================================================================

@click.group()
@click.version_option(version='0.1.0', prog_name='FrictionSim2D')
def cli():
    """FrictionSim2D - Generate LAMMPS input files for 2D material friction simulations."""


# =============================================================================
# RUN COMMANDS
# =============================================================================

@cli.group('run')
def run_group():
    """Run friction simulations."""


@run_group.command('afm')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
                            help='Generate HPC scripts for the simulation root')
@click.option('--hpc', 'hpc_name', type=str, default=None,
              help='HPC configuration name (overrides settings)')
@click.option('--local', 'run_local', is_flag=True,
              help='Mark as local run (no HPC submission)')
def run_afm(config_file: str, output_dir: str, use_aiida: bool, 
                        generate_hpc: bool, hpc_name: Optional[str], run_local: bool):
    """Generate AFM simulation files.
    
    Creates all necessary LAMMPS input files, structures, and potentials
    for tip-on-substrate friction simulations.
    
    Example:
        FrictionSim2D run.afm afm_config.ini -o ./afm_output --aiida
    """
    _run_simulation(
        'afm', config_file, output_dir, use_aiida, generate_hpc, hpc_name, run_local
    )

@run_group.command('sheetonsheet')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
        help='Generate HPC scripts for the simulation root')
@click.option('--hpc', 'hpc_name', type=str, default=None,
              help='HPC configuration name (overrides settings)')
@click.option('--local', 'run_local', is_flag=True,
              help='Mark as local run (no HPC submission)')
def run_sheetonsheet(config_file: str, output_dir: str, use_aiida: bool,
            generate_hpc: bool, hpc_name: Optional[str], run_local: bool):
    """Generate sheet-on-sheet simulation files.
    
    Creates all necessary LAMMPS input files for 4-layer sheet-on-sheet
    friction simulations.
    
    Example:
        FrictionSim2D run.sheetonsheet sheet_config.ini -o ./sheet_output
    """
    _run_simulation(
        'sheetonsheet', config_file, output_dir, use_aiida, generate_hpc, hpc_name, run_local
    )

def _register_simulations_aiida(simulation_dirs: List[Path], config_path: Path):
    """Register generated simulations with AiiDA."""
    try:
        from src.aiida.integration import register_simulation_batch
        registered = register_simulation_batch(simulation_dirs, config_path)
        click.echo(f"   ✅ Registered {len(registered)} simulations")
    except ImportError:
        click.echo("   ⚠️  AiiDA integration module not found", err=True)
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"   ⚠️  Registration failed: {exc}", err=True)
        logger.exception("AiiDA registration failed")


def _parse_walltime(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    if value.isdigit():
        return int(value)
    parts = value.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = [int(p) for p in parts]
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError("Walltime must be seconds or HH:MM:SS")


def _raise_abort(message: str, exc: Exception) -> NoReturn:
    click.echo(message, err=True)
    raise click.Abort() from exc


def _build_aiida_submit_options(
    machines: Optional[int],
    mpiprocs: Optional[int],
    walltime: Optional[str],
    queue: Optional[str],
    project: Optional[str],
    mem: Optional[str],
    scheduler: str,
    prepend_text: tuple,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    resource_options: Dict[str, Any] = {}
    settings = cast(Any, load_settings())
    hpc_defaults = settings.hpc

    if machines is None:
        machines = getattr(hpc_defaults, 'num_nodes', None)
    if mpiprocs is None:
        mpiprocs = getattr(hpc_defaults, 'num_cpus', None)
    if machines:
        resource_options['num_machines'] = machines
    if mpiprocs:
        resource_options['num_mpiprocs_per_machine'] = mpiprocs
    if resource_options:
        options['resources'] = resource_options

    if walltime is None:
        default_hours = getattr(hpc_defaults, 'walltime_hours', None)
        if default_hours:
            walltime = f"{default_hours}:00:00"
    walltime_seconds = _parse_walltime(walltime)
    if walltime_seconds is not None:
        options['max_wallclock_seconds'] = walltime_seconds

    if queue is None:
        queue = getattr(hpc_defaults, 'queue', None)
    if queue:
        options['queue_name'] = queue
    if project is None:
        project = getattr(hpc_defaults, 'account', None)
    if project:
        options['account'] = project

    scheduler_cmds = []
    if mem is None:
        default_mem = getattr(hpc_defaults, 'memory_gb', None)
        if default_mem:
            mem = f"{default_mem}gb"
    if mem:
        if scheduler == 'slurm':
            scheduler_cmds.append(f"#SBATCH --mem={mem}")
        else:
            scheduler_cmds.append(f"#PBS -l mem={mem}")
    if scheduler_cmds:
        options['custom_scheduler_commands'] = "\n".join(scheduler_cmds)

    if not prepend_text:
        modules_list = cast(List[str], getattr(hpc_defaults, 'modules', []))
        if modules_list:
            prepend_text = tuple(f"module load {m}" for m in modules_list)
    if prepend_text:
        options['prepend_text'] = "\n".join(prepend_text)

    return options

# =============================================================================
# SETTINGS COMMANDS
# =============================================================================

@cli.group('settings')
def settings_group():
    """Manage simulation settings."""


@settings_group.command('show')
def settings_show():
    """Display current settings."""
    defaults = load_settings()
    click.echo(yaml.dump(defaults.dict(), default_flow_style=False))

@settings_group.command('init')
def settings_init():
    """Create a local settings.yaml file for customization."""
    with pkg_resources.as_file(pkg_resources.files('src.data.settings') / 'settings.yaml') as p:
        shutil.copy(p, "settings.yaml")
    click.echo("✅ Created 'settings.yaml' in current directory")

@settings_group.command('reset')
def settings_reset():
    """Remove local settings.yaml and use package defaults."""
    local_settings = Path("settings.yaml")
    if local_settings.exists():
        local_settings.unlink()
        click.echo("✅ Removed local settings.yaml")
    else:
        click.echo("ℹ️  No local settings found")

# =============================================================================
# HPC COMMANDS
# =============================================================================

@cli.group('hpc')
def hpc_group():
    """Generate HPC submission scripts."""


@hpc_group.command('generate')
@click.argument('simulation_dir', type=click.Path(exists=True))
@click.option('--scheduler', '-s', type=click.Choice(['pbs', 'slurm']), default='pbs',
              help='HPC scheduler type')
@click.option('--output-dir', '-o', default=None,
              help='Output directory for scripts (default: simulation_dir/hpc)')
def hpc_generate(simulation_dir: str, scheduler: str, output_dir: Optional[str]):
    """Generate HPC submission scripts for existing simulations.
    
    Scans simulation directory and creates PBS or SLURM job scripts.
    
    Example:
        FrictionSim2D hpc generate ./afm_output --scheduler pbs
    """
    from src.hpc import HPCScriptGenerator, HPCConfig

    sim_dir = Path(simulation_dir)
    out_dir = Path(output_dir) if output_dir else sim_dir / 'hpc'
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"🖥️  Generating {scheduler.upper()} scripts for: {sim_dir}")

    settings = load_settings()
    _ensure_hpc_settings(settings.hpc)
    hpc_config = HPCConfig.from_settings(settings.hpc)
    hpc_config.scheduler_type = scheduler  # type: ignore[assignment]

    simulation_paths = []
    script_names = hpc_config.lammps_scripts or ['system.in', 'slide.in']

    def script_exists(lammps_dir: Path, script_name: str) -> bool:
        if '*' in script_name or '?' in script_name:
            return any(lammps_dir.glob(script_name))
        if script_name == 'slide.in':
            return (
                (lammps_dir / 'slide.in').exists()
                or any(lammps_dir.glob('slide_*.in'))
            )
        return (lammps_dir / script_name).exists()

    for lammps_dir in sim_dir.rglob('lammps'):
        if not lammps_dir.is_dir():
            continue
        if not any(script_exists(lammps_dir, script) for script in script_names):
            continue
        rel_path = lammps_dir.parent.relative_to(sim_dir)
        simulation_paths.append(str(rel_path))

    if not simulation_paths:
        click.echo("❌ No simulation directories found", err=True)
        raise click.Abort()

    click.echo(f"📋 Found {len(simulation_paths)} simulations")

    manifest_entries, resolved_scripts = _build_hpc_manifest_entries(
        sim_dir,
        simulation_paths,
        hpc_config.lammps_scripts,
    )
    hpc_config.lammps_scripts = resolved_scripts

    generator = HPCScriptGenerator(hpc_config)
    base_dir = '$PBS_O_WORKDIR' if scheduler == 'pbs' else '$SLURM_SUBMIT_DIR'
    scripts = generator.generate_scripts(
        simulation_paths=manifest_entries,
        output_dir=out_dir,
        scheduler=scheduler,  # type: ignore[arg-type]
        base_dir=base_dir
    )

    click.echo(f"✅ Generated {len(scripts)} script(s) in {out_dir}")
    click.echo("\n📝 Next steps:")
    click.echo(f"   1. Review scripts in {out_dir}")
    click.echo("   2. Transfer to HPC cluster")
    click.echo("   3. Follow submit_all.txt")

# =============================================================================
# AIIDA COMMANDS (Optional)
# =============================================================================

@cli.group('aiida')
def aiida_group():
    """AiiDA workflow management (requires aiida-core)."""
    from src.aiida import AIIDA_AVAILABLE
    if not AIIDA_AVAILABLE:
        click.echo("⚠️  AiiDA not available. Install with:", err=True)
        click.echo("   conda install -c conda-forge aiida-core", err=True)
        raise click.Abort()

@aiida_group.command('status')
def aiida_status():
    """Check AiiDA installation and profile status."""
    from src.aiida import AIIDA_AVAILABLE
    
    if not AIIDA_AVAILABLE:
        click.echo("❌ AiiDA not installed")
        return
    
    click.echo("✅ AiiDA is installed")
    
    try:
        from aiida.manage.configuration import load_profile
        profile = load_profile()
        click.echo(f"✅ Active profile: {profile.name}")
        click.echo(f"   Storage: {profile.storage_backend}")
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"⚠️  No active profile: {exc}")
        click.echo("\n📝 Setup AiiDA with: verdi presto --use-postgres")

@aiida_group.command('import')
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--process/--no-process', default=True,
              help='Run postprocessing on results')
def aiida_import(results_dir: str, process: bool):
    """Import completed simulation results into AiiDA database.
    
    Example:
        FrictionSim2D aiida import ./returned_results
    """
    from src.aiida.integration import import_results_to_aiida
    
    results_path = Path(results_dir)
    click.echo(f"📥 Importing results from: {results_path}")
    
    try:
        if process:
            click.echo("🔄 Running postprocessing...")
            from src.postprocessing.read_data import DataReader
            _ = DataReader(results_dir=str(results_path))
            click.echo("   ✅ Postprocessing complete")
        
        imported = import_results_to_aiida(results_path)
        click.echo(f"✅ Imported {len(imported)} simulations to AiiDA")
        
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Import failed")
        _raise_abort(f"❌ Import failed: {exc}", exc)

@aiida_group.command('query')
@click.option('--material', '-m', help='Filter by material')
@click.option('--layers', '-l', type=int, help='Filter by layer count')
@click.option('--force', '-f', type=float, help='Filter by applied force')
@click.option('--format', 'output_format', type=click.Choice(['table', 'csv', 'json']),
              default='table', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Save to file')
def aiida_query(material: Optional[str], layers: Optional[int], force: Optional[float],
                output_format: str, output: Optional[str]):
    """Query simulation database.
    
    Example:
        FrictionSim2D aiida query --material h-MoS2 --layers 2 --format csv
    """
    from src.aiida.query import Friction2DDB
    
    db = Friction2DDB()
    
    filters = {}
    if material:
        filters['material'] = material
    if layers:
        filters['layers'] = layers
    if force:
        filters['force'] = force
    
    click.echo(f"🔍 Querying database with filters: {filters}")
    
    try:
        results = db.query(**filters)
        click.echo(f"📊 Found {results.total_count} results")
        
        if output_format == 'table':
            df = results.to_dataframe()
            click.echo("\n" + df.to_string(index=False))
        elif output_format == 'csv':
            df = results.to_dataframe()
            if output:
                df.to_csv(output, index=False)
                click.echo(f"✅ Saved to {output}")
            else:
                click.echo(df.to_csv(index=False))
        elif output_format == 'json':
            if output:
                results.export_json(Path(output))
                click.echo(f"✅ Saved to {output}")
            else:
                import json
                click.echo(json.dumps(results.query_params, indent=2))
                
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA query failed")
        _raise_abort(f"❌ Query failed: {exc}", exc)


@aiida_group.command('setup')
@click.option('--profile', '-p', default=None,
              help='AiiDA profile name (default: auto-created)')
@click.option('--lammps-path', type=click.Path(),
              help='Explicit path to LAMMPS executable')
@click.option('--hpc-config', type=click.Path(exists=True),
              help='(Deprecated) Path to hpc.yaml for PBS/SLURM computer setup')
@click.option('--use-remote', is_flag=True,
              help='Configure remote HPC computer using settings from settings.yaml')
def aiida_setup(profile: Optional[str], lammps_path: Optional[str],
                hpc_config: Optional[str], use_remote: bool):
    """Set up AiiDA profile, computer, and LAMMPS code.

    Performs first-time AiiDA configuration:
    - Creates a profile (via verdi presto)
    - Registers localhost or remote HPC as a computer
    - Configures the LAMMPS code

    For remote HPC setup, configure the AiiDA section in settings.yaml with
    hostname, username, workdir, etc., then use --use-remote flag.

    Example::

        FrictionSim2D aiida setup
        FrictionSim2D aiida setup --lammps-path /usr/local/bin/lmp_mpi
        FrictionSim2D aiida setup --use-remote  # Uses settings.yaml AiiDA config
    """
    from src.aiida.setup import full_setup
    from src.core.config import load_settings

    click.echo("🔧 Running AiiDA first-time setup ...")

    # Load HPC settings if using remote
    hpc_settings = None
    aiida_settings = None
    if use_remote:
        settings = load_settings()
        hpc_settings = settings.hpc
        aiida_settings = settings.aiida
        click.echo(f"📡 Configuring remote HPC: {aiida_settings.hostname or 'localhost'}")

    if hpc_config:
        click.echo("⚠️  --hpc-config is deprecated. Use --use-remote with settings.yaml instead.")

    try:
        full_setup(
            profile_name=profile or 'friction2d',
            lammps_executable=lammps_path,
            hpc_settings=hpc_settings,
            aiida_settings=aiida_settings,
        )
        click.echo("✅ AiiDA setup complete")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA setup failed")
        _raise_abort(f"❌ Setup failed: {exc}", exc)


@aiida_group.command('export')
@click.option('--output', '-o', type=click.Path(), default='friction2d.aiida',
              help='Output archive path')
@click.option('--material', '-m', default=None,
              help='Export only simulations for a given material')
def aiida_export(output: str, material: Optional[str]):
    """Export AiiDA database to a portable archive file.

    The archive can be transferred to another machine and imported
    with ``FrictionSim2D aiida import-archive``.

    Example::

        FrictionSim2D aiida export -o results.aiida
        FrictionSim2D aiida export -m h-MoS2
    """
    from src.aiida.integration import export_archive

    click.echo(f"📦 Exporting archive to {output} ...")
    try:
        export_archive(Path(output), materials=[material] if material else None)
        click.echo(f"✅ Archive saved: {output}")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA export failed")
        _raise_abort(f"❌ Export failed: {exc}", exc)


@aiida_group.command('import-archive')
@click.argument('archive_path', type=click.Path(exists=True))
def aiida_import_archive(archive_path: str):
    """Import an AiiDA archive into the current profile.

    Use this to set up a mirror database on a local workstation from
    results exported on an HPC cluster.

    Example::

        FrictionSim2D aiida import-archive results.aiida
    """
    from src.aiida.integration import import_archive

    click.echo(f"📥 Importing archive: {archive_path} ...")
    try:
        import_archive(Path(archive_path))
        click.echo("✅ Archive imported successfully")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA import-archive failed")
        _raise_abort(f"❌ Import failed: {exc}", exc)


@aiida_group.command('package')
@click.argument('simulation_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Output archive path (.tar.gz)')
def aiida_package(simulation_dir: str, output: Optional[str]):
    """Create a tar.gz archive of simulation inputs for transfer."""
    sim_path = Path(simulation_dir)
    out_path = Path(output) if output else sim_path.with_suffix('.tar.gz')

    def _tar_filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        if info.name.endswith('.lammpstrj'):
            return None
        return info

    click.echo(f"📦 Packing {sim_path} → {out_path}")
    with tarfile.open(out_path, 'w:gz') as tar:
        tar.add(sim_path, arcname=sim_path.name, filter=_tar_filter)
    click.echo(f"✅ Archive created: {out_path}")


@aiida_group.command('submit')
@click.argument('simulation_dir', type=click.Path(exists=True))
@click.option('--code', '-c', default=None,
              help='AiiDA code label (auto-detects if not specified)')
@click.option('--scripts', default=None,
              help='Comma-separated list of LAMMPS scripts to run in order')
@click.option('--array', 'use_array', is_flag=True,
              help='Submit a single array job for all simulations')
@click.option('--machines', type=int, default=None,
              help='Override: Number of machines per job')
@click.option('--mpiprocs', type=int, default=None,
              help='Override: MPI processes per machine')
@click.option('--walltime', type=str, default=None,
              help='Override: Walltime (HH:MM:SS or hours)')
@click.option('--queue', type=str, default=None,
              help='Override: Scheduler queue name')
@click.option('--project', type=str, default=None,
              help='Override: Scheduler project/account')
@click.option('--dry-run', is_flag=True,
              help='Preview configuration without submitting')
def aiida_submit(
    simulation_dir: str,
    code: Optional[str],
    scripts: Optional[str],
    use_array: bool,
    machines: Optional[int],
    mpiprocs: Optional[int],
    walltime: Optional[str],
    queue: Optional[str],
    project: Optional[str],
    dry_run: bool,
):
    """Submit simulations to AiiDA with smart defaults and prompting.

    Automatically detects codes, uses settings.yaml defaults,
    and prompts for essential missing values. Manual overrides are supported.

    Examples:

    # Minimal usage (auto-detect code, use defaults)
      FrictionSim2D aiida submit ./afm_output

      # With manual overrides
      FrictionSim2D aiida submit ./output --machines 4 --walltime 24:00:00

      # Preview before submitting
      FrictionSim2D aiida submit ./output --dry-run
    """
    from src.aiida.submit import smart_submit

    sim_path = Path(simulation_dir)
    click.echo(f"🚀 Preparing submission from {sim_path}\n")

    try:
        # Collect manual overrides
        overrides = {}
        if machines is not None:
            overrides['machines'] = machines
        if mpiprocs is not None:
            overrides['mpiprocs'] = mpiprocs
        if walltime is not None:
            overrides['walltime'] = walltime
        if queue is not None:
            overrides['queue'] = queue
        if project is not None:
            overrides['project'] = project

        # Call smart submit helper
        processes = smart_submit(
            simulation_dir=sim_path,
            code_label=code,
            use_array=use_array,
            scripts=scripts,
            dry_run=dry_run,
            **overrides,
        )

        if not dry_run and processes:
            click.echo(f"\n✅ Successfully submitted {len(processes)} job(s)")

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA submit failed")
        _raise_abort(f"❌ Submission failed: {exc}", exc)

# =============================================================================
# POSTPROCESSING COMMANDS
# =============================================================================

@cli.group('postprocess')
def postprocess_group():
    """Read, analyse and plot simulation results."""


@postprocess_group.command('read')
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--export', 'do_export', is_flag=True,
              help='Export full time-series data to JSON.')
def postprocess_read(results_dir: str, do_export: bool):
    """Read and process simulation result data."""
    from src.postprocessing import DataReader  # noqa: PLC0415

    reader = DataReader(results_dir=results_dir)
    reader.export_issue_reports()
    if do_export:
        reader.export_full_data_to_json()
    click.echo("Postprocessing complete.")


@postprocess_group.command('plot')
@click.argument('plot_config', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='plots',
              help='Output directory for plots.')
@click.option('--settings', type=click.Path(exists=True),
              help='Plot settings JSON file.')
def postprocess_plot(plot_config: str, output_dir: str,
                     settings: Optional[str]):
    """Generate plots from processed simulation data."""
    import json  # noqa: PLC0415
    from src.postprocessing import Plotter  # noqa: PLC0415

    with open(plot_config, 'r') as f:
        config = json.load(f)

    plot_settings = None
    if settings:
        with open(settings, 'r') as f:
            plot_settings = json.load(f)

    data_dirs = config.get('data_dirs', [])
    labels = config.get('labels', [])
    plots = config.get('plots', [])

    if not data_dirs or not labels or not plots:
        raise click.Abort(
            "'data_dirs', 'labels', and 'plots' must be defined in config."
        )

    if len(data_dirs) != len(labels):
        raise click.Abort(
            "Number of 'data_dirs' must match number of 'labels'."
        )

    plotter = Plotter(data_dirs, labels, output_dir, plot_settings)
    for plot_cfg in plots:
        plotter.generate_plot(plot_cfg)
    click.echo("All plots generated.")


# =============================================================================
# DATABASE COMMANDS
# =============================================================================

@cli.group('db')
def db_group():
    """Interact with the shared online FrictionSim2D PostgreSQL database.

    Upload your simulation results to the community database and query
    results contributed by other users.

    Connection settings can be provided via environment variables
    (FRICTION_DB_HOST, FRICTION_DB_PORT, FRICTION_DB_NAME,
    FRICTION_DB_USER, FRICTION_DB_PASSWORD) or via the --host / --user /
    --password options.
    """


def _make_db(
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Instantiate a :class:`~src.data.database.FrictionDB`."""
    from src.data.database import FrictionDB  # noqa: PLC0415
    return FrictionDB(host=host, port=port, dbname=dbname, user=user, password=password)


_DB_OPTIONS = [
    click.option('--host', default=None, envvar='FRICTION_DB_HOST',
                 help='Database hostname (default: $FRICTION_DB_HOST or localhost)'),
    click.option('--port', default=None, type=int, envvar='FRICTION_DB_PORT',
                 help='Database port (default: $FRICTION_DB_PORT or 5432)'),
    click.option('--dbname', default=None, envvar='FRICTION_DB_NAME',
                 help='Database name (default: $FRICTION_DB_NAME or frictionsim2d)'),
    click.option('--user', '-u', default=None, envvar='FRICTION_DB_USER',
                 help='Database username (default: $FRICTION_DB_USER)'),
    click.option('--password', default=None, envvar='FRICTION_DB_PASSWORD',
                 help='Database password (default: $FRICTION_DB_PASSWORD)'),
]


def _add_db_options(func):
    for opt in reversed(_DB_OPTIONS):
        func = opt(func)
    return func


@db_group.command('upload')
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--uploader', '-n', default=None,
              help='Your name or identifier (stored with each row).')
@_add_db_options
def db_upload(
    results_dir: str,
    uploader: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Upload simulation results from RESULTS_DIR to the shared database.

    Scans RESULTS_DIR for completed simulation outputs and uploads each
    result (mean COF, forces, conditions) to the community PostgreSQL
    database.

    Example:
        FrictionSim2D db upload ./simulation_output/afm_run_1/results \\
            --uploader alice
    """
    from src.postprocessing.read_data import DataReader  # noqa: PLC0415
    from src.data.database import FrictionDB  # noqa: PLC0415

    click.echo(f"📂 Reading results from {results_dir} ...")
    reader = DataReader(results_dir=str(results_dir))

    db = _make_db(host, port, dbname, user, password)

    n_uploaded = 0
    for material, size_data in reader.full_data_nested.items():
        for _size_key, substrate_data in size_data.items():
            for _sub, tip_data in substrate_data.items():
                for _tip_mat, radius_data in tip_data.items():
                    for _radius, layer_data in radius_data.items():
                        for layer_key, speed_data in layer_data.items():
                            layers = int(layer_key.replace('l', ''))
                            for speed_key, force_data in speed_data.items():
                                speed = int(speed_key.replace('s', ''))
                                for load_key, angle_data in force_data.items():
                                    is_pressure = load_key.startswith('p')
                                    load_val = float(load_key[1:])
                                    for angle_key, df in angle_data.items():
                                        angle = int(angle_key.replace('a', ''))
                                        try:
                                            import numpy as np  # noqa: PLC0415
                                            cof_col = 'cof' if 'cof' in df.columns else None
                                            mean_cof = float(df[cof_col].mean()) if cof_col else None
                                            std_cof = float(df[cof_col].std()) if cof_col else None
                                            lf_col = 'lfx' if 'lfx' in df.columns else None
                                            mean_lf = float(df[lf_col].mean()) if lf_col else None
                                            nf_col = 'nf' if 'nf' in df.columns else None
                                            mean_nf = float(df[nf_col].mean()) if nf_col else None

                                            db.upload_result(
                                                material=material.replace('_', '-'),
                                                simulation_type='afm',
                                                layers=layers,
                                                force_nN=None if is_pressure else load_val,
                                                pressure_gpa=load_val if is_pressure else None,
                                                scan_angle=float(angle),
                                                scan_speed=float(speed),
                                                mean_cof=mean_cof,
                                                std_cof=std_cof,
                                                mean_lf=mean_lf,
                                                mean_nf=mean_nf,
                                                uploader=uploader,
                                            )
                                            n_uploaded += 1
                                        except Exception as exc:  # pylint: disable=broad-except
                                            logger.warning(
                                                "Skipped row (%s, l%d, a%d): %s",
                                                material, layers, angle, exc,
                                            )

    click.echo(f"✅ Uploaded {n_uploaded} result(s) to the database.")


@db_group.command('query')
@click.option('--material', '-m', default=None, help='Filter by material name.')
@click.option('--type', 'sim_type', default=None,
              help='Filter by simulation type (afm or sheetonsheet).')
@click.option('--layers', '-l', type=int, default=None,
              help='Filter by number of layers.')
@click.option('--uploader', '-n', default=None, help='Filter by uploader name.')
@click.option('--limit', type=int, default=50, show_default=True,
              help='Maximum number of rows to return.')
@click.option('--csv', 'output_csv', default=None,
              help='Save results to a CSV file.')
@_add_db_options
def db_query(
    material: Optional[str],
    sim_type: Optional[str],
    layers: Optional[int],
    uploader: Optional[str],
    limit: int,
    output_csv: Optional[str],
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Query the shared database and print matching results.

    Example:
        FrictionSim2D db query --material h-MoS2 --layers 1 --limit 20

        FrictionSim2D db query --material h-WS2 --csv results.csv
    """
    db = _make_db(host, port, dbname, user, password)
    df = db.query(
        material=material,
        simulation_type=sim_type,
        layers=layers,
        uploader=uploader,
        limit=limit,
    )

    if df.empty:
        click.echo("No results found.")
        return

    click.echo(df.to_string(index=False))
    click.echo(f"\n{len(df)} row(s) returned.")

    if output_csv:
        df.to_csv(output_csv, index=False)
        click.echo(f"💾 Saved to {output_csv}")


@db_group.command('stats')
@_add_db_options
def db_stats(
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Show aggregate statistics for the shared database."""
    db = _make_db(host, port, dbname, user, password)
    stats = db.get_statistics()

    click.echo(f"Total simulations : {stats['total_rows']}")
    click.echo(f"Global mean COF   : {stats['cof_global_mean']:.4f}"
               if stats['cof_global_mean'] is not None else "Global mean COF   : n/a")
    click.echo("\nBy material:")
    for mat, count in stats['by_material'].items():
        click.echo(f"  {mat:<20} {count}")
    click.echo("\nBy simulation type:")
    for stype, count in stats['by_type'].items():
        click.echo(f"  {stype:<20} {count}")


@db_group.command('delete')
@click.option('--uploader', '-n', required=True,
              help='Delete all rows belonging to this uploader.')
@click.confirmation_option(prompt='Are you sure you want to delete these rows?')
@_add_db_options
def db_delete(
    uploader: str,
    host: Optional[str],
    port: Optional[int],
    dbname: Optional[str],
    user: Optional[str],
    password: Optional[str],
):
    """Delete your uploaded results from the shared database.

    Only rows with a matching uploader name are deleted.

    Example:
        FrictionSim2D db delete --uploader alice
    """
    db = _make_db(host, port, dbname, user, password)
    count = db.delete_own_results(uploader)
    click.echo(f"🗑  Deleted {count} row(s) for uploader '{uploader}'.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main CLI entry point."""
    try:
        cli()
    except Exception:  # pylint: disable=broad-except
        logger.exception("Command failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
