"""Command Line Interface for FrictionSim2D.

Click glue only: each command parses its options, calls its package, prints.
"""
# pylint: disable=import-outside-toplevel
# Deliberate: every subcommand imports its dependencies lazily so `FrictionSim2D --help`
# and compute-node use never pay for ase/aiida/Qt. See CLAUDE.md "Import weight".

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, NoReturn, Tuple
import click

from .core.config import load_settings, settings_origin, _global_settings_path
from .run.hpc_dispatch import (
    ensure_hpc_settings,
    generate_scripts_for_tree,
    generation_report,
    hpc_generation_blocker,
    plan_tree_generation,
)
from .run.simulate import run_simulations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _raise_abort(message: str, exc: Exception) -> NoReturn:
    click.echo(message, err=True)
    raise click.Abort() from exc


@contextmanager
def _failing(log: str, message: str, plain: tuple = ()):
    """Turn any error inside into ``❌ {message}: {exc}`` (logged) and an abort.

    An exception of ``plain`` prints as ``❌ {exc}``, unlogged; a
    ``click.ClickException`` passes through.
    """
    try:
        yield
    except click.ClickException:
        raise
    except plain as exc:
        _raise_abort(f"❌ {exc}", exc)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception(log)
        _raise_abort(f"❌ {message}: {exc}", exc)


def _echo_lines(lines) -> None:
    """Echo ``(text, to_stderr)`` pairs."""
    for text, err in lines:
        click.echo(text, err=err)


def _run_simulation(model: str, config_file: str, output_dir: str,
                    generate_hpc: bool, settings_file: Optional[Path] = None,
                    **extra) -> Optional[Path]:
    """Run ``run_simulations`` for a ``run`` subcommand; return the simulation root.

    ``extra`` goes to ``run_simulations`` as is (``quench_for``, ``force``).
    """
    from .builders.construction.amorphous import AmorphousMissing
    try:
        created_simulations, simulation_root, configs_to_run, _ = run_simulations(
            config_file=config_file, model=model, settings_file=settings_file,
            output_root=Path(output_dir), ensure_hpc_settings=ensure_hpc_settings,
            generate_hpc=generate_hpc, **extra)
    except AmorphousMissing as exc:
        # Names the missing block and the `run quench --for` line that makes it.
        click.echo(str(exc), err=True)
        raise SystemExit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        # A rejected config (expansion refusals, sweep ordering, pydantic's
        # ValidationError) is a user error whose message names the fix; an
        # unexpected error still surfaces its traceback.
        logger.debug("Simulation generation rejected the config", exc_info=True)
        click.echo(f"❌ {exc}", err=True)
        raise click.Abort() from exc

    if simulation_root is None:
        click.echo("✅ all amorphous blocks present: nothing to quench.")
        return None
    click.echo(f"📋 Found {len(configs_to_run)} simulation configurations")
    click.echo(f"\n✅ Generation complete: {len(created_simulations)}/"
               f"{len(configs_to_run)} successful")
    click.echo(f"📂 Output directory: {simulation_root.absolute()}")
    if len(created_simulations) < len(configs_to_run):
        # A failed build fails the command, or `run && hpc generate` carries on.
        raise SystemExit(1)
    return simulation_root


def _options(*options):
    """One decorator applying ``options`` in the order listed."""
    def decorate(func):
        for option in reversed(options):
            func = option(func)
        return func
    return decorate


#: ``--settings-file``, ``--output-dir`` and ``--hpc-scripts`` of every ``run`` command.
_run_options = _options(
    click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
                 default=None, help='Path to a settings.yaml file for this run only.'),
    click.option('--output-dir', '-o', default='simulation_output',
                 help='Output directory for generated files'),
    click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
                 help='Generate HPC scripts for the simulation root'))

#: ``--profile`` of the commands that load an existing AiiDA profile.
_profile_option = click.option('--profile', default=None,
                               help='AiiDA profile name (default: configured default profile)')

# =============================================================================
# MAIN CLI GROUP
# =============================================================================

@click.group()
@click.version_option(prog_name='FrictionSim2D')
def cli():
    """FrictionSim2D - Generate LAMMPS input files for 2D material friction simulations."""


# =============================================================================
# RUN COMMANDS
# =============================================================================

@cli.group('run')
def run_group():
    """Run friction simulations."""


#: (command, model, help) of the ``run`` commands that differ only in their model.
_RUN_COMMANDS = (
    ('afm', 'afm', """Generate AFM simulation files.

    Creates all necessary LAMMPS input files, structures, and potentials
    for tip-on-substrate friction simulations.

    Example:
        FrictionSim2D run afm afm_config.ini -o ./afm_output
    """),
    ('sheetonsheet', 'sheetonsheet', """Generate sheet-on-sheet simulation files.

    Creates all necessary LAMMPS input files for 4-layer sheet-on-sheet
    friction simulations.

    Example:
        FrictionSim2D run sheetonsheet sheet_config.ini -o ./sheet_output
    """),
    ('pes-sheet', 'pes_sheet', """Generate sheet-on-sheet interlayer PES scan files.

    Builds a frozen bilayer and writes a cheap, static lateral grid-scan of the
    interlayer potential-energy surface (corrugation barrier) instead of a
    sliding run. Reuses the sheet-on-sheet geometry; needs [2D] layers = [2].

    Example:
        FrictionSim2D run pes-sheet examples/pes_sheet_config.ini -o ./pes_output
    """),
    ('pes-tip', 'pes_tip', """Generate tip-on-sheet (AFM) surface PES scan files.

    Builds and indents the AFM tip + sheet + substrate stack to a low load
    exactly as the sliding sim, then writes a static lateral grid-scan of the
    tip-felt surface PES (energy + lateral force on the tip) instead of sliding.

    Example:
        FrictionSim2D run pes-tip examples/pes_tip_config.ini -o ./pes_output
    """),
    ('indent', 'indent',
     """Generate AFM indentation ("indent") simulation files: hold + pull-off retract.

    The standard AFM build + indentation ramp (system.in) to each load in
    [general] force, then per load a damped finite-T hold followed by a
    displacement-controlled quasi-static retract of the rigid tip to full
    detachment — the sliding phase is never invoked. Records both the penetration
    channels (tip_z − sheet_com_z, vertical compliance) and the force–distance
    retract curve (pull-off force, work of adhesion, contact stiffness). Uses a
    fresh, recorded velocity seed per (material, layer). Subsumes the former
    separate poke (hold-only) and adhesion (retract-only) commands.

    Example:
        FrictionSim2D run indent examples/indent_config.ini -o ./indent_output --hpc-scripts
    """),
)


def _add_run_command(name: str, model: str, doc: str) -> None:
    def command(config_file: str, settings_file: Optional[Path], output_dir: str,
                generate_hpc: bool):
        _run_simulation(model, config_file, output_dir, generate_hpc,
                        settings_file=settings_file)
    command.__doc__ = doc
    command.__name__ = f"run_{model}"
    run_group.command(name)(click.argument('config_file', type=click.Path(exists=True))(
        _run_options(command)))


for _name, _model, _doc in _RUN_COMMANDS:
    _add_run_command(_name, _model, _doc)


@run_group.command('quench')
@click.argument('config_file', required=False, type=click.Path(exists=True))
@click.option('--for', 'for_config', type=click.Path(exists=True, dir_okay=False),
              default=None,
              help='A friction config: make one member per amorphous block it is missing.')
@_run_options
@click.option('--local', 'run_local', is_flag=True,
              help='Then run every member here with serial lmp (hours) and import the blocks.')
@click.option('--force', is_flag=True,
              help='Quench members a library block already satisfies.')
def run_quench(config_file: Optional[str], for_config: Optional[str],  # pylint: disable=too-many-arguments
               settings_file: Optional[Path], output_dir: str, generate_hpc: bool,
               run_local: bool, force: bool):
    """Generate melt-quench jobs that make amorphous blocks.

    CONFIG has a [quench] section (see examples/quench_config.ini); with
    --for, the members are the blocks a friction config is missing. Only
    decks are written: run them on the HPC (`hpc generate`, or --hpc-scripts)
    and `FrictionSim2D quench import ROOT`, or add --local to run and import
    them here.

    Examples:
        FrictionSim2D run quench examples/quench_config.ini -o ./quench
        FrictionSim2D run quench --for examples/afm_config.ini -o ./quench --hpc-scripts
    """
    if bool(config_file) == bool(for_config):
        raise click.UsageError('give either CONFIG (a [quench] config) or --for FRICTION_CONFIG')
    root = _run_simulation('quench', config_file or for_config, output_dir, generate_hpc,
                           settings_file=settings_file, quench_for=bool(for_config),
                           force=force)
    if root is None:
        return
    if run_local:
        from .builders.construction.amorphous import run_quench_root_local
        click.echo("⏳ Running the melt-quench members serially (this takes hours)...")
        _done, failed = run_quench_root_local(root)
        for member, error in failed:
            click.echo(f"❌ melt-quench failed in {member}: {error}", err=True)
        _report_quench_import(root, replace=False, source='local')
    else:
        click.echo(f"\n📝 Next: run it (FrictionSim2D hpc generate {root} --scheduler pbs), "
                   f"then FrictionSim2D quench import {root}")


def _report_quench_import(root: Path, replace: bool, source: str) -> None:
    """Import a quench root, echo the outcome, exit 1 on anything left out."""
    from .builders.construction.amorphous import import_quench_root
    report = import_quench_root(root, replace=replace, source=source)
    _echo_lines(report.lines())
    if not report.ok:
        raise SystemExit(1)


# =============================================================================
# SETTINGS COMMANDS
# =============================================================================

@cli.group('settings')
def settings_group():
    """Manage simulation settings."""


@settings_group.command('show')
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Explicit settings.yaml path to inspect (run-specific).')
@click.option('--origin', is_flag=True,
              help='Show which settings file is currently in effect.')
def settings_show(settings_file: Optional[Path], origin: bool):
    """Display current effective settings.

    With --origin, also prints the path of the settings file that was loaded
    (or 'hardcoded defaults' if no file is found).

    Example:
        FrictionSim2D settings show
        FrictionSim2D settings show --origin
    """
    from .core.config import effective_settings_yaml
    if origin:
        src = settings_origin(settings_file=settings_file)
        click.echo(f"# Settings loaded from: {src}" if src else
                   "# Settings loaded from: hardcoded defaults (no settings.yaml found)")
    click.echo(effective_settings_yaml(settings_file))


@settings_group.command('init')
@click.option('--global', 'global_', is_flag=True,
              help='Write to ~/.config/FrictionSim2D/settings.yaml (semi-permanent).')
@click.option('--force', is_flag=True,
              help='Overwrite an existing file without prompting.')
def settings_init(global_: bool, force: bool):
    """Create a settings.yaml file pre-populated with current effective settings.

    Without --global, writes settings.yaml to the current directory for
    per-simulation customisation. This file is only used when explicitly
    supplied to commands (e.g. run --settings-file). With --global, writes to
    ~/.config/FrictionSim2D/settings.yaml for machine-level configuration
    such as HPC host, queue names, and database credentials.

    Example:
        FrictionSim2D settings init                # create per-run file
        FrictionSim2D settings init --global       # first-time machine setup
    """
    from .core.config import write_effective_settings
    target = _global_settings_path() if global_ else Path('settings.yaml')
    if target.exists() and not force:
        click.confirm(f"'{target}' already exists. Overwrite?", abort=True)
    write_effective_settings(target)
    click.echo(f"✅ Created '{target}'")
    if global_:
        click.echo("   Edit this file to set HPC, database, and other machine-level defaults.")
    else:
        click.echo("   Edit this file and pass it via --settings-file for per-simulation settings.")


@settings_group.command('reset')
@click.option('--global', 'global_', is_flag=True,
              help='Remove ~/.config/FrictionSim2D/settings.yaml instead.')
@click.confirmation_option(prompt='This will delete the settings file. Continue?')
def settings_reset(global_: bool):
    """Delete a settings.yaml file and revert to defaults.

    Without --global, removes settings.yaml from the current directory.
    With --global, removes ~/.config/FrictionSim2D/settings.yaml.

    Example:
        FrictionSim2D settings reset
        FrictionSim2D settings reset --global
    """
    target = _global_settings_path() if global_ else Path('settings.yaml')
    if target.exists():
        target.unlink()
        click.echo(f"✅ Removed '{target}'")
    else:
        click.echo(f"ℹ️  No settings file found at '{target}'")

# =============================================================================
# GUI AND HPC COMMANDS
# =============================================================================

@cli.group('gui', invoke_without_command=True)
@click.option('--check', 'run_check', is_flag=True,
              help='Diagnose the GUI environment and exit without launching.')
@click.option('--verbose', is_flag=True,
              help='Show all log output and library warnings in the terminal '
                   '(also FRICTIONSIM2D_GUI_VERBOSE=1).')
@click.pass_context
def gui_group(ctx, run_check: bool, verbose: bool):
    """Launch the desktop simulation builder.

    With no subcommand this starts the GUI. Use --check to verify that Qt,
    ovito and the external build tools are usable here first.
    """
    if ctx.invoked_subcommand is not None:
        return
    from .gui import GUI_AVAILABLE, INSTALL_HINT  # noqa: PLC0415
    if not GUI_AVAILABLE:
        click.echo("GUI dependencies are not available.", err=True)
        click.echo(INSTALL_HINT, err=True)
        raise click.Abort()
    if run_check:
        from .gui.diagnostics import run_checks  # noqa: PLC0415
        report = run_checks()
        click.echo(report.render())
        ctx.exit(0 if report.ok else 1)
    from .gui import main as gui_main  # noqa: PLC0415
    ctx.exit(gui_main(tuple(), verbose=verbose))


@cli.group('hpc')
def hpc_group():
    """Generate HPC submission scripts."""


@hpc_group.command('generate')
@click.argument('simulation_dir', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this command.')
@click.option('--scheduler', '-s', type=click.Choice(['pbs', 'slurm']), default='pbs',
              help='HPC scheduler type')
@click.option('--output-dir', '-o', default=None,
              help='Output directory for scripts (default: simulation_dir/hpc)')
@click.option('--preflight', is_flag=True, default=False,
              help='Also write preflight.sh + its submit wrapper '
                   '(or set hpc.preflight in settings)')
def hpc_generate(simulation_dir: str, settings_file: Optional[Path],
                 scheduler: str, output_dir: Optional[str], preflight: bool):
    """Generate HPC submission scripts for existing simulations.

    Scans simulation directory and creates PBS or SLURM job scripts.

    Example:
        FrictionSim2D hpc generate ./afm_output --scheduler pbs
    """
    sim_dir = Path(simulation_dir)
    out_dir = Path(output_dir) if output_dir else sim_dir / 'hpc'
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"🖥️  Generating {scheduler.upper()} scripts for: {sim_dir}")
    settings = load_settings(settings_file=settings_file)
    hpc_config, simulation_paths = plan_tree_generation(sim_dir, settings, scheduler)
    if not simulation_paths:
        click.echo("❌ No simulation directories found", err=True)
        raise click.Abort()
    blocker = hpc_generation_blocker(settings)
    if blocker is not None:
        click.echo(f"❌ {blocker}", err=True)
        raise click.Abort()
    click.echo(f"📋 Found {len(simulation_paths)} simulations")
    preflight = preflight or hpc_config.preflight
    try:
        result = generate_scripts_for_tree(
            sim_dir, hpc_config, out_dir, simulation_paths,
            hpc_config.base_dir(sim_dir.name), preflight=preflight)
    except ValueError as exc:
        # A refused setting (e.g. a device MACE group without hpc.gpu) names its fix.
        logger.debug("HPC script generation refused the settings", exc_info=True)
        click.echo(f"❌ {exc}", err=True)
        raise click.Abort() from exc
    for line in generation_report(result, out_dir, scheduler, preflight):
        click.echo(line)


@hpc_group.command('enrich-provenance')
@click.argument('sim_dir', type=click.Path(path_type=Path))
@click.argument('log_path', type=click.Path(path_type=Path), required=False)
@click.option('--recursive', '-r', is_flag=True,
              help='Treat SIM_DIR as a tree root and enrich every run beneath it.')
def hpc_enrich_provenance(sim_dir: Path, log_path: Optional[Path],
                          recursive: bool):
    """Fold finished runs' LAMMPS logs into their provenance.json.

    The AFM settle loop prints `# RAMP_STATUS converged=... r=... iterations=...`
    and `# UNLOADED_REF layer=... z=...`; this reads them back into the record
    the build wrote, along with the LAMMPS version.

    \b
    Two modes:
      single    SIM_DIR is one run directory (the parent of lammps/). LOG_PATH
                is optional -- omitted, it defaults to
                SIM_DIR/lammps/run_system.log, or to the sole run*.log if that
                name is absent. Several logs with no run_system.log among them
                are reported and skipped rather than guessed at.
      -r        SIM_DIR is a tree root; every run directory beneath it is
                enriched, one line each plus a final tally. This is the normal
                way to use it: the simulations are built locally, LAMMPS runs
                on the cluster, and the results (with the run logs) are rsynced
                back here, where the whole tree is swept in one go.

    This lives in the `hpc` group, not `postprocess`, on purpose: a cluster-side
    install would call it from the job script on a compute node, so its import
    path must not reach the optional plotting/DB extras `postprocess` pulls in.

    Always exits 0 on missing data. A missing log, a missing or unreadable
    provenance.json, a log with no settle-loop lines, and a tree with no runs
    in it are all reported and shrugged off -- a completed simulation must never
    be reported as failed over a provenance nicety.

    \b
    Examples:
      FrictionSim2D hpc enrich-provenance -r ./simulation_20260908_101500
      FrictionSim2D hpc enrich-provenance ./afm/h_MoS2/L1
    """
    from .builders.utils.post_run import enrich_one_run, enrich_tree
    if recursive and log_path is not None:
        raise click.UsageError(
            "--recursive sweeps a whole tree and resolves each run's log "
            "itself; it cannot also take a single LOG_PATH.")
    if not recursive:
        outcome, detail = enrich_one_run(sim_dir, log_path)
        click.echo(f"{outcome} {sim_dir}: {detail}")
        return
    for line in enrich_tree(sim_dir):
        click.echo(line)


# =============================================================================
# QUENCH COMMANDS (the amorphous block library)
# =============================================================================

@cli.group('quench')
def quench_group():
    """The amorphous block library: import finished quench runs, list blocks.

    Blocks are made with `FrictionSim2D run quench`.
    """


@quench_group.command('import')
@click.argument('root', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--replace', is_flag=True,
              help='Replace a same-key library block even if the new one is not larger.')
def quench_import(root: Path, replace: bool):
    """Copy every finished member of a quench ROOT into the user library.

    Incomplete members are listed and the command exits 1; a member is never
    half-imported. Writes only under $XDG_DATA_HOME/FrictionSim2D/amorphous.

    Example:
        FrictionSim2D quench import ./quench/quench_20261124_101500
    """
    _report_quench_import(root, replace=replace, source='hpc')


@quench_group.command('list')
@click.argument('material', required=False)
@click.option('--json', 'as_json', is_flag=True, help='Print JSON rows instead of a table.')
def quench_list(material: Optional[str], as_json: bool):
    """List the amorphous blocks of every library root (read-only).

    Example:
        FrictionSim2D quench list Si
    """
    import json
    from .builders.construction.amorphous import block_line, block_rows
    rows = block_rows(material)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    for line in [block_line(row) for row in rows] or ["No amorphous blocks found."]:
        click.echo(line)


# =============================================================================
# AIIDA COMMANDS (Optional)
# =============================================================================

@cli.group('aiida')
def aiida_group():
    """AiiDA workflow management (requires aiida-core)."""
    from .aiida import AIIDA_AVAILABLE
    if not AIIDA_AVAILABLE:
        click.echo("⚠️  AiiDA not available. Install with:", err=True)
        click.echo("   conda install -c conda-forge aiida-core", err=True)
        raise click.Abort()


@aiida_group.command('preflight')
def aiida_preflight():
    """Verify profile, database and entry points before generating a campaign.

    Distinguishes "no profile initialised here" (fine on a generation-only
    machine) from "profile broken" (spec §8). Exit code 0 only when a healthy
    profile and the friction2d entry points are both present.
    """
    from .aiida.profile import main as preflight_main
    raise SystemExit(preflight_main())


@aiida_group.command('status')
def aiida_status():
    """Check AiiDA installation and profile status."""
    click.echo("✅ AiiDA is installed")
    try:
        from aiida.manage.configuration import load_profile  # type: ignore[import-untyped]
        profile = load_profile()
        click.echo(f"✅ Active profile: {profile.name}")
        click.echo(f"   Storage: {profile.storage_backend}")
    except Exception as exc:  # pylint: disable=broad-except
        click.echo(f"⚠️  No active profile: {exc}")
        click.echo("\n📝 Setup AiiDA with: verdi presto")

#: ``--settings-file`` of the commands that read the store's settings
#: (``aiida.machine``, ``aiida.skip_fraction``, ``integrity``, ``aiida.profile``).
_store_settings_file_option = click.option(
    '--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None, help='settings.yaml to read instead of the per-user one '
                       '(~/.config/FrictionSim2D/settings.yaml).')


def _parse_integrity(_ctx, _param, pairs) -> Optional[dict]:
    """``--integrity`` callback: the pairs as validated overrides, None when none."""
    from .core.config import checked_integrity_overrides  # noqa: PLC0415
    try:
        return checked_integrity_overrides(pairs)
    except ValueError as exc:       # pydantic's ValidationError is one
        raise click.BadParameter(str(exc), param_hint='--integrity') from exc


#: Repeatable ``--integrity KEY=VALUE`` (``margins.RULE=VALUE`` for a margin).
_integrity_option = click.option(
    '--integrity', 'integrity', multiple=True, metavar='KEY=VALUE', callback=_parse_integrity,
    help='Override one integrity threshold of the settings, e.g. '
         '--integrity hole_bonds=2 --integrity margins.hole=0.1 (none turns '
         'a threshold off). Repeatable.')


def _store_profile(profile: Optional[str], settings_file: Optional[Path]) -> Optional[str]:
    """``profile``, else ``aiida.profile`` of an explicit ``settings_file``
    (``None`` leaves it to ``load_aiida_profile``: the per-user settings)."""
    if profile is None and settings_file is not None:
        return load_settings(settings_file=settings_file).aiida.profile or None
    return profile


@aiida_group.command('import')
@click.argument('simulation_folder', type=click.Path(exists=True, path_type=Path))
@click.option('--label', '-l', default=None,
              help='Label for the simulation set (default: the folder name).')
@click.option('--machine', default=None,
              help='Machine (Computer) the runs were produced on (default: '
                   "settings.aiida.machine, else each run's settings.hpc.hpc_host, "
                   "else 'localhost').")
@_profile_option
@_store_settings_file_option
@_integrity_option
@click.option('--dry-run', is_flag=True,
              help='Print what would be imported; create nothing.')
def aiida_import(simulation_folder: Path, label: Optional[str],  # pylint: disable=too-many-arguments
                 machine: Optional[str], profile: Optional[str],
                 settings_file: Optional[Path], integrity: Optional[dict], dry_run: bool):
    """Import a simulation tree into AiiDA: one CalcJob per LAMMPS deck.

    SIMULATION_FOLDER is the ``simulation_XXXXXXXX`` directory produced by one
    ``FrictionSim2D run``. Idempotent: re-importing skips every unchanged
    deck and supersedes a changed one. Every new point gets an integrity
    verdict, then the set's rulings file is applied. Exits 1 if any run
    failed; a real (non-dry-run) import also exits 1 if any path could not
    be parsed.

    Example:
        FrictionSim2D aiida import ./simulation_20260421_173546 -l 260421-afm --machine archer2
        FrictionSim2D aiida import ./simulation_20260421_173546 --integrity ejected_atoms=3
    """
    from .aiida.ingest import SetRootMismatch, import_root  # noqa: PLC0415
    with _failing("Import failed", "Import failed", plain=SetRootMismatch):
        report = import_root(simulation_folder, label=label, machine=machine,
                             profile=_store_profile(profile, settings_file),
                             dry_run=dry_run, settings_file=settings_file,
                             integrity=integrity)
    click.echo(report.render())
    if not report.clean():
        raise SystemExit(1)


@aiida_group.command('rule')
@click.argument('label')
@click.option('--rulings', 'rulings_path',
              type=click.Path(exists=True, path_type=Path), default=None,
              help='Rulings file, or its directory (default: integrity_rulings.yaml '
                   'in the root the set was imported from).')
@click.option('--withdraw-pending', 'withdraw_pending',
              type=click.Path(path_type=Path), default=None,
              help='Drop every ruling of this rulings file (or directory) from the '
                   "set's ruling ledger, e.g. a pending one whose file is gone for "
                   'good. The file need not exist.')
@_profile_option
def aiida_rule(label: str, rulings_path: Optional[Path],
               withdraw_pending: Optional[Path], profile: Optional[str]):
    """Apply a rulings file to the set LABEL: a pass or fail ruling per
    quarantined point (or, with ``override: true`` and a note, per any point).

    Every pass reads the set's own integrity_rulings.yaml, the --rulings
    file, and every other file the set's ruling ledger already holds. The
    own file's ruling wins; else the most recently added or changed one.
    A file that is gone keeps its rulings (listed as pending). Idempotent.
    Prints what was applied, replaced, skipped and withdrawn, every live
    override, every conflict and pending ruling, and the rulings that name
    no point of the set.

    Example:
        FrictionSim2D aiida rule 260421-afm --rulings ~/reviews/260421.yaml
    """
    from .aiida.steps import apply_rulings  # noqa: PLC0415
    with _failing("Rulings failed", f"Rulings for set {label!r} failed"):
        report = apply_rulings(label, rulings_path,
                               withdraw_pending=withdraw_pending, profile=profile)
    click.echo(report.render())


@aiida_group.command('recheck')
@click.argument('label')
@_profile_option
@_store_settings_file_option
@_integrity_option
def aiida_recheck(label: str, profile: Optional[str], settings_file: Optional[Path],
                  integrity: Optional[dict]):
    """Re-check the set LABEL under the current integrity settings.

    Every live point whose verdict was checked with another integrity
    ``version``, or other integrity settings, gets a new verdict; the old one
    is superseded with its rulings, then the set's own rulings file is applied
    again. Prints the points whose status changed. A second run checks
    nothing. Exits 1 if any check failed.

    Example:
        FrictionSim2D aiida recheck 260421-afm --settings-file ./settings.yaml
        FrictionSim2D aiida recheck 260421-afm --integrity hole_bonds=2.5
    """
    from .aiida.steps import recheck_set  # noqa: PLC0415
    with _failing("Recheck failed", f"Recheck of set {label!r} failed"):
        report = recheck_set(label, settings_file=settings_file, integrity=integrity,
                             profile=_store_profile(profile, settings_file))
    click.echo(report.render())
    if report.failures:
        raise SystemExit(1)


@aiida_group.command('reindex')
@_profile_option
def aiida_reindex(profile: Optional[str]):
    """Recompute every friction2d index extra (block, schema, file and
    folder hashes; run, deck and token extras) from the stored nodes.

    Run it after an archive import that dropped or edited extras.
    """
    from .aiida import load_aiida_profile  # noqa: PLC0415
    from .aiida.store import reindex  # noqa: PLC0415
    with _failing("Reindex failed", "Reindex failed"):
        load_aiida_profile(profile)
        touched = reindex()
    click.echo(f"reindexed {touched} node(s)")


@aiida_group.command('verify')
@_profile_option
def aiida_verify(profile: Optional[str]):
    """Check the store's content hashes: a hash held by more than one node,
    or a hash extra that disagrees with its node's content.

    Exits 1 when it finds either (``aiida reindex`` repairs a mismatch).
    """
    from .aiida import load_aiida_profile  # noqa: PLC0415
    from .aiida.store import verify  # noqa: PLC0415
    with _failing("Verify failed", "Verify failed"):
        load_aiida_profile(profile)
        report = verify()
    click.echo(report.render())
    if not report.ok:
        raise SystemExit(1)


@aiida_group.command('postprocess')
@click.argument('label')
@_profile_option
@_store_settings_file_option
def aiida_postprocess(label: str, profile: Optional[str], settings_file: Optional[Path]):
    """Summarise every point of the set LABEL whose latest verdict is complete.

    One tracked summary per point and summary options (settings
    ``aiida.skip_fraction``) and code version; a second pass makes none.

    Example:
        FrictionSim2D aiida postprocess 260421-afm
    """
    from .aiida.steps import postprocess_group as summarise_set  # noqa: PLC0415
    warnings: list = []
    with _failing("Postprocess failed", f"Postprocess of set {label!r} failed"):
        made = summarise_set(label, warnings=warnings, settings_file=settings_file,
                             profile=_store_profile(profile, settings_file))
    click.echo(f"set {label!r}: {made} new summar{'y' if made == 1 else 'ies'}")
    for line in warnings:
        click.echo(f"  WARNING {line}")


@aiida_group.command('dataset')
@click.argument('name')
@click.argument('labels', nargs=-1, required=True)
@_profile_option
@_store_settings_file_option
def aiida_dataset(name: str, labels: Tuple[str, ...], profile: Optional[str],
                  settings_file: Optional[Path]):
    """Make the dataset NAME from the sets LABELS: one row per point whose
    latest verdict is complete, with the summary the policy selects.

    Datasets are never rewritten: a NAME that exists is refused.

    Example:
        FrictionSim2D aiida dataset mos2-loads 260421-afm 260512-afm
    """
    from .aiida.steps import build_dataset  # noqa: PLC0415
    warnings: list = []
    with _failing("Dataset failed", f"Dataset {name!r} failed"):
        table = build_dataset(name, list(labels), warnings=warnings, settings_file=settings_file,
                              profile=_store_profile(profile, settings_file))
    click.echo(f"dataset {name!r} ({table.uuid}): {len(table['rows'])} row(s) "
               f"from {', '.join(labels)}")
    for line in warnings:
        click.echo(f"  WARNING {line}")


@aiida_group.command('list-sets')
@_profile_option
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
def aiida_list_sets(profile: Optional[str], output_format: str):
    """List imported simulation sets stored in AiiDA.

    Example:
        FrictionSim2D aiida list-sets
        FrictionSim2D aiida list-sets --format json
    """
    from .aiida.query import list_simulation_sets, render_sets  # noqa: PLC0415
    with _failing("Failed to list simulation sets", "List failed"):
        rows = list_simulation_sets(profile=profile)
    click.echo(render_sets(rows, output_format) if rows else
               "No simulation sets found in the current AiiDA profile.")


@aiida_group.command('export-results')
@click.argument('set_label')
@click.argument('output_dir', type=click.Path(path_type=Path))
@_profile_option
def aiida_export_results(set_label: str, output_dir: Path, profile: Optional[str]):
    """Write a stored set back out as an ordinary simulation tree.

    The result works unchanged with ``postprocess read`` and the GUI download.

    Example:
        FrictionSim2D aiida export-results 260421-afm ~/exported/260421-afm
    """
    from .aiida.export import export_results  # noqa: PLC0415
    with _failing("Export failed", "Export failed"):
        written = export_results(set_label, output_dir, profile=profile)
    click.echo(f"✅ Wrote {len(written)} file(s) to {Path(output_dir).absolute()}")


@aiida_group.command('rebuild')
@click.argument('set_label')
@click.option('--output-dir', '-o', default='.', type=click.Path(path_type=Path),
              help='Root directory to write the simulation tree into (default: cwd)')
@click.option('--hpc-scripts', is_flag=True, default=False,
              help='Generate HPC job-submission scripts after rebuilding')
@_profile_option
def aiida_rebuild(set_label: str, output_dir: Path, hpc_scripts: bool,
                  profile: Optional[str]):
    """Regenerate a set's LAMMPS inputs from the store.

    Exports each run's stored config.json and the set's settings.yaml, then
    reruns the real builder router -- the same one ``FrictionSim2D run`` uses.

    Example:
        FrictionSim2D aiida rebuild 260421-afm -o ~/rebuild_test --hpc-scripts
    """
    from .aiida.export import rebuild_set  # noqa: PLC0415
    failures: list = []
    with _failing("Rebuild failed", "Rebuild failed"):
        created, root = rebuild_set(set_label, output_dir, profile=profile,
                                    generate_hpc=hpc_scripts, failures=failures)
    if failures:
        click.echo(f"❌ Rebuilt {len(created)} run(s) → {Path(root).absolute()}; "
                   f"{len(failures)} failed:", err=True)
        for line in failures:
            click.echo(f"  {line}", err=True)
        raise SystemExit(1)
    click.echo(f"✅ Rebuilt {len(created)} run(s) → {Path(root).absolute()}")


@aiida_group.command('delete')
@click.argument('set_label')
@_profile_option
@click.confirmation_option(
    prompt='This will permanently delete the simulation set and all linked nodes. Continue?'
)
def aiida_delete(set_label: str, profile: Optional[str]):
    """Delete a single simulation set and all its linked nodes.

    SET_LABEL is the label of the set to delete. Its CalcJobs, their outputs
    and steps, the inputs nothing else uses, and the set's own Group are
    removed.

    Example:
        FrictionSim2D aiida delete 260312-force_rebomos
    """
    from .aiida.steps import delete_simulation_set  # noqa: PLC0415
    click.echo(f"🗑️  Deleting simulation set '{set_label}'…")
    with _failing("Delete failed", "Delete failed"):
        n = delete_simulation_set(set_label, profile=profile)
        click.echo(f"✅ Deleted {n} nodes for set '{set_label}'.")


@aiida_group.command('clear')
@_profile_option
@click.confirmation_option(
    prompt='This will permanently delete all FrictionSim2D nodes. Continue?'
)
def aiida_clear(profile: Optional[str]):
    """Delete all FrictionSim2D nodes from the AiiDA database.

    This removes every friction2d CalcJob with its outputs, steps, summaries
    and datasets, the inputs they leave unused, and every ``friction2d/*``
    Group in the profile. Computers are kept. The action is irreversible.

    Example:
        FrictionSim2D aiida clear
    """
    from .aiida.steps import clear_all_nodes  # noqa: PLC0415
    click.echo("🗑️  Deleting all FrictionSim2D nodes…")
    with _failing("Clear failed", "Clear failed"):
        n = clear_all_nodes(profile=profile)
        click.echo(f"✅ Deleted {n} nodes.")


@aiida_group.command('query')
@click.option('--material', '-m', 'materials', multiple=True,
              help='Filter by material; repeatable, and a matching run must '
                   'contain ALL given materials')
@click.option('--family', default=None, help='Filter by family (e.g. afm, hetero)')
@click.option('--model', default=None, help='Filter by model (e.g. afm, sheetonsheet)')
@click.option('--set-label', default=None,
              help='Filter by simulation set label (e.g. "251113-afm")')
@click.option('--potential', default=None, help='Filter by potential token')
@click.option('--layers', '-l', type=int, help='Filter by layer count')
@click.option('--force-min', type=float, help='Minimum applied force (nN)')
@click.option('--force-max', type=float, help='Maximum applied force (nN)')
@click.option('--pressure-min', type=float, help='Minimum applied pressure (GPa)')
@click.option('--pressure-max', type=float, help='Maximum applied pressure (GPa)')
@click.option('--angle-min', type=float, help='Minimum scan angle (degrees)')
@click.option('--angle-max', type=float, help='Maximum scan angle (degrees)')
@click.option('--speed-min', type=float, help='Minimum scan speed')
@click.option('--speed-max', type=float, help='Maximum scan speed')
@click.option('--temperature-min', type=float, help='Minimum temperature (K)')
@click.option('--temperature-max', type=float, help='Maximum temperature (K)')
@click.option('--limit', type=int, help='Maximum number of results')
@click.option('--order-by', default=None,
              help="Result attribute to sort by (prefix '-' for descending)")
@click.option('--format', 'output_format', type=click.Choice(['table', 'csv', 'json']),
              default='table', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Save to file')
@_profile_option
def aiida_query(output_format: str, output: Optional[str], profile: Optional[str], **filters):
    """Query the store's points: one row per live point, its verdict first.

    Each row leads with the point (set, run, deck, load, angle, speed), then
    the verdict the policy selects (``status``, ``reasons``,
    ``verdict_kind``, ``provisional``, ``thresholds_version``), then the
    point's tokens, identity and summary.

    Example:
        FrictionSim2D aiida query --material h-MoS2 --layers 2 --format csv
        FrictionSim2D aiida query --set-label 251113-afm --format table
    """
    from .aiida.query import open_query, query_filters, query_lines
    set_label = filters['set_label']
    with _failing("AiiDA query setup failed", "Query failed"):
        db, available = open_query(profile, set_label)
    if set_label and set_label not in available:
        raise click.ClickException(
            f"No set found with label '{set_label}'. Available: {available}")
    query_kwargs = query_filters(**filters)
    shown = {k: v for k, v in query_kwargs.items() if v is not None}
    click.echo(f"🔍 Querying database with filters: {shown}")
    with _failing("AiiDA query failed", "Query failed"):
        for line in query_lines(db.query(**query_kwargs), output_format, output):
            click.echo(line)


@aiida_group.command('setup')
@click.option('--create-profile', '-p', 'create_profile', default=None,
              help='Name of the AiiDA profile to CREATE (default: friction2d). '
                   'Every other command spells --profile, which LOADS an existing one.')
@click.option('--postgres', is_flag=True,
              help='Try a PostgreSQL profile first (needs a running server); falls '
                   'back to SQLite. Default: SQLite, which the import-only store needs.')
def aiida_setup(create_profile: Optional[str], postgres: bool):
    """Create (or load) the AiiDA profile: SQLite unless ``--postgres``.

    Only the profile: the machine Computer and the LAMMPS Code of each run
    are created when it is imported (``aiida import``).

    Example:
        FrictionSim2D aiida setup --create-profile friction2d
    """
    from .aiida.profile import full_setup  # noqa: PLC0415
    click.echo("🔧 Running AiiDA first-time setup ...")
    with _failing("AiiDA setup failed", "Setup failed"):
        full_setup(profile_name=create_profile or 'friction2d', use_postgres=postgres)
        click.echo("✅ AiiDA setup complete")


@aiida_group.command('export')
@click.argument('set_label')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              default='friction2d.aiida', help='Output archive path')
@_profile_option
@click.option('--force', is_flag=True,
              help='Write even when live points have no integrity verdict')
@click.option('--include-peratom', is_flag=True,
              help='Keep the per-atom nodes (left out by default)')
@click.option('--keep-hpc-paths', is_flag=True,
              help='Do not anonymise HPC hostnames, executables and paths')
def aiida_export(set_label: str, output: Path, profile: Optional[str],
                 force: bool, include_peratom: bool, keep_hpc_paths: bool):
    """Export one simulation set (its CalcJobs, verdicts and summaries) to a
    .aiida archive, with ``<output>.params/`` block sidecars and a README.

    The archive can be transferred to another machine and imported with
    ``FrictionSim2D aiida import-archive``. HPC details are anonymised unless
    ``--keep-hpc-paths``; per-atom data is left out unless ``--include-peratom``.

    Example:
        FrictionSim2D aiida export 260421-afm -o ~/transfer/260421-afm.aiida
    """
    from .aiida.export import export_archive  # noqa: PLC0415
    click.echo(f"📦 Exporting set {set_label!r} to {output} ...")
    with _failing("AiiDA export failed", "Export failed"):
        path = export_archive(set_label, output, profile=profile, force=force,
                              include_peratom=include_peratom, keep_hpc_paths=keep_hpc_paths)
    click.echo(f"✅ Archive written: {path.absolute()}")


@aiida_group.command('import-archive')
@click.argument('archive_path', type=click.Path(exists=True, path_type=Path))
@click.option('--profile', default=None,
              help='AiiDA profile name (default: settings.aiida.profile)')
def aiida_import_archive(archive_path: Path, profile: Optional[str]):
    """Import a .aiida archive into the current profile.

    Use this to mirror an hpc-direct store on a local workstation.

    Example:
        FrictionSim2D aiida import-archive ~/transfer/260421-afm.aiida
    """
    from .aiida.export import import_archive  # noqa: PLC0415
    click.echo(f"📥 Importing archive: {archive_path} ...")
    with _failing("AiiDA import-archive failed", "Import failed"):
        n_new = import_archive(archive_path, profile=profile)
    click.echo(f"✅ Imported {n_new} new node(s)")


# =============================================================================
# POSTPROCESSING COMMANDS
# =============================================================================

@cli.group('postprocess')
def postprocess_group():
    """Read, analyse and plot simulation results."""


@postprocess_group.command('read')
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--export', 'do_export', is_flag=True,
              help='Write outputs/output_full_<size>.json (schema 3) and '
                   'outputs/summary_<size>.csv.')
@click.option('--skip-fraction', default=0.2, show_default=True, type=float,
              help='Transient fraction dropped before the summary statistics.')
@click.option('--complete-only', is_flag=True,
              help='Export only load points whose status is complete '
                   '(the pre-schema-3 behaviour).')
@click.option('--integrity-settings', 'integrity_settings_path',
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='YAML file with an `integrity:` mapping; overrides the '
                   'thresholds of the user settings.')
@click.option('--no-integrity', is_flag=True,
              help='Skip the stage-3 integrity check.')
def postprocess_read(results_dir: str, do_export: bool, skip_fraction: float,
                     complete_only: bool, integrity_settings_path,
                     no_integrity: bool):
    """Read and process simulation result data.

    RESULTS_DIR is a simulation root -- the directory holding afm/,
    sheetonsheet/, indent/, pes_tip/, pes_sheet/.
    """
    from .postprocessing import DataReader  # noqa: PLC0415
    from .postprocessing.integrity_rulings import (  # noqa: PLC0415
        integrity_report, load_integrity_settings)
    thresholds = None
    if not no_integrity:
        try:
            thresholds = load_integrity_settings(integrity_settings_path)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    reader = DataReader(results_dir=results_dir, skip_fraction=skip_fraction,
                        complete_only=complete_only, integrity=not no_integrity,
                        integrity_settings=thresholds)
    reader.export_issue_reports()
    if not reader.points:
        click.echo(f"No runs found under {results_dir}.", err=True)
        return
    click.echo(reader.found_line())
    for line in integrity_report(reader) if reader.integrity_settings is not None else []:
        click.echo(line)
    if do_export:
        for export in (reader.export_full_data_to_json, reader.export_summary_csv):
            for path in export():
                click.echo(f"✅ {path}")
    click.echo("Postprocessing complete.")


@postprocess_group.command('plot')
@click.argument('plot_config', required=False, type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='plots',
              help='Output directory for plots.')
@click.option('--settings', type=click.Path(exists=True),
              help='Plot settings JSON, merged over --preset.')
@click.option('--preset', default=None,
              help="Shipped settings preset (currently: 'publication').")
@click.option('--validate', is_flag=True,
              help='Validate the document against the plot schema and exit.')
@click.option('--schema', 'show_schema', is_flag=True,
              help='Print the plot schema as Markdown and exit.')
@click.option('--init', 'init_path', type=click.Path(), default=None,
              help='Write a valid starter plot document to this path and exit.')
@click.option('--data-dir', 'init_data_dirs', multiple=True,
              type=click.Path(exists=True),
              help='With --init: a dataset directory to reference (repeatable).')
def postprocess_plot(plot_config, output_dir, settings, preset, validate,
                     show_schema, init_path, init_data_dirs):
    """Generate plots from processed simulation data.

    PLOT_CONFIG is a plot document (see documentation/postprocessing.md).
    """
    from .postprocessing.plot_config import (  # noqa: PLC0415
        load_plot_document, schema_markdown, write_starter_document)
    if show_schema:
        click.echo(schema_markdown())
        return
    if init_path:
        target = write_starter_document(Path(init_path), list(init_data_dirs))
        click.echo(f"✅ {target}")
        click.echo(f"Edit it, then: FrictionSim2D postprocess plot {target} "
                   f"--preset publication")
        return
    if not plot_config:
        raise click.UsageError(
            "PLOT_CONFIG is required. Start one with "
            "`postprocess plot --init plots.json --data-dir <run>`, "
            "or print the schema with --schema.")
    try:
        document = load_plot_document(Path(plot_config))
    except Exception as exc:  # pylint: disable=broad-except
        raise click.ClickException(str(exc)) from exc
    from .postprocessing.plot_style import plot_settings  # noqa: PLC0415
    try:
        style = plot_settings(preset, settings)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    if validate:
        click.echo(f"OK: {len(document.plots)} plot(s), "
                   f"{len(document.data_dirs)} dataset(s).")
        return
    from .postprocessing import Plotter  # noqa: PLC0415
    plotter = Plotter(document.data_dirs, document.labels, output_dir, style,
                      dataset_display_labels=document.dataset_display_labels,
                      series_color_map=document.series_color_map)
    for spec in document.plots:
        plotter.generate_plot(spec)
    for path in plotter.written:
        click.echo(f"✅ {path}")
    click.echo(f"{len(plotter.written)} file(s) written.")


@postprocess_group.command('pes-scan')
@click.argument('run_root', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--data-dir', '-d', type=click.Path(path_type=Path), default='data',
              help='Directory for the per-material descriptor CSVs (pes_scan_{sheet,tip}.csv).')
@click.option('--grid-dir', type=click.Path(path_type=Path), default=None,
              help='Optional directory for tidied per-material grids '
                   '(<grid-dir>/{sheet,tip}/<material>.csv).')
@click.option('--kind', type=click.Choice(['sheet', 'tip', 'both']), default='both',
              help='Which PES scans to reduce.')
def postprocess_pes_scan(run_root: Path, data_dir: Path, grid_dir: Optional[Path], kind: str):
    """Reduce PES-scan energy/force grids to per-material descriptor CSVs.

    Scans RUN_ROOT for pes_sheet/ and pes_tip/ results and writes
    data/pes_scan_sheet.csv and data/pes_scan_tip.csv (merge key 'material').

    Example:
        FrictionSim2D postprocess pes-scan ./simulation_output/simulation_YYYYMMDD_HHMMSS \\
            --data-dir data --grid-dir outputs/pes_scan
    """
    from .postprocessing import pes_scan as pes  # noqa: PLC0415
    kinds = ['sheet', 'tip'] if kind == 'both' else [kind]
    written = pes.run(run_root, data_dir, grid_dir=grid_dir, kinds=kinds)
    if not written:
        click.echo(f"No PES-scan grids found under {run_root}.", err=True)
        return
    for k, path in written.items():
        click.echo(f"✅ {k}: wrote descriptors -> {path.resolve()}")
    if grid_dir is not None:
        click.echo(f"📂 Grids copied under {Path(grid_dir).resolve()}/{{sheet,tip}}/")


@postprocess_group.command('indent')
@click.argument('run_root', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--out-dir', '-o', type=click.Path(path_type=Path), default='outputs/poke_sims',
              help='Destination for output_indent_f*_s*.json, indent/<material>_l<layer>.csv '
                   'and indent_manifest.csv.')
@click.option('--data-dir', '-d', type=click.Path(path_type=Path), default=None,
              help='Destination for indent_features.csv (default: same as --out-dir).')
def postprocess_indent(run_root: Path, out_dir: Path, data_dir: Optional[Path]):
    """Reduce an indent run to penetration JSON + adhesion features + a manifest.

    Delivers both descriptor sets from the one hold+retract run:
      - penetration: output_indent_f<load>_s<seed>.json per load (nested campaign
        schema, read by the ML side exactly like the sliding campaign);
      - adhesion: indent_features.csv (pull-off force, work of adhesion, contact
        stiffness, keyed material/layer/load) + tidied per-condition retract
        curves indent/<material>_l<layer>.csv;
      - indent_manifest.csv (material, layer, load, seed, hold_steps, z_step,
        n_steps, wall_time).

    Example:
        FrictionSim2D postprocess indent ./indent_output/simulation_YYYYMMDD_HHMMSS \\
            --out-dir outputs/poke_sims --data-dir data
    """
    from .postprocessing import indent as indent_pp  # noqa: PLC0415
    written = indent_pp.run(run_root, out_dir, data_dir=data_dir)
    if not written:
        click.echo(f"No indent results found under {run_root}.", err=True)
        return
    for name, path in written.items():
        click.echo(f"✅ {name} -> {path.resolve()}")


# =============================================================================
# COMPARE
# =============================================================================

@cli.command('compare')
@click.argument('a')
@click.argument('b')
@_profile_option
@click.option('--rtol', type=float, default=1e-6, show_default=True,
              help='Relative tolerance for derived values.')
@click.option('--atol', type=float, default=1e-9, show_default=True,
              help='Absolute tolerance floor for derived values.')
def compare_cmd(a: str, b: str, profile: Optional[str], rtol: float, atol: float):
    """Compare the parameter blocks of two runs or campaigns, field by field.

    A and B are each a CalcJob pk or uuid, a point uuid, a group label, or an
    archive's ``.params`` sidecar. Prefix a reference with ``pk:``, ``uuid:``,
    ``point:``, ``group:`` or ``sidecar:`` to say which. Unprefixed, a
    directory that is (or holds) a ``.params`` sidecar written by ``aiida
    export`` is read as one; otherwise the store is asked: an all-digit
    reference is a pk (else a group label), anything else a group, uuid or point.
    Each side is read in its own schema's units and defaults. A group gives the fields shared
    by all its decks; a field that varies inside it shows as '<varies>'.

    Example:
        FrictionSim2D compare 260421-afm 260512-afm
    """
    from .aiida.compare import compare_loaded, load_side  # noqa: PLC0415
    with _failing("Compare failed", "Compare failed"):
        left = load_side(a, profile, rtol, atol)
        right = load_side(b, profile, rtol, atol)
    click.echo(compare_loaded(left, right, rtol, atol).render())


# =============================================================================
# ANALYSE COMMANDS
# =============================================================================

@cli.group('analyse')
def analyse_group():
    """Run and read the on-demand analyses of a dataset in the AiiDA store."""


def _params(name: str, pairs: Tuple[str, ...]) -> dict:
    """``--param`` pairs of analysis ``name``, validated; a usage error if not."""
    from .analysis.base import parse_params  # noqa: PLC0415
    try:
        return parse_params(name, pairs)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _stored(name: str, dataset: str, profile: Optional[str], pairs: Tuple[str, ...]):
    """The stored result node of ``name`` over ``dataset`` under ``pairs``."""
    from .aiida import load_aiida_profile  # noqa: PLC0415
    from .analysis.store import stored_result  # noqa: PLC0415
    params = _params(name, pairs)
    load_aiida_profile(profile)
    try:
        return stored_result(name, dataset, params)
    except LookupError as exc:
        raise click.ClickException(str(exc)) from exc


#: ``--param`` of ``analyse show`` and ``analyse export``.
_stored_param_option = click.option(
    '--param', 'params', multiple=True, metavar='NAME=VALUE',
    help='Parameter the result was run with (repeatable; default: the defaults).')


@analyse_group.command('list')
def analyse_list():
    """List the registered analyses."""
    from .analysis.base import catalogue_lines  # noqa: PLC0415
    for line in catalogue_lines() or ["no analyses registered"]:
        click.echo(line)


@analyse_group.command('run')
@click.argument('name')
@click.argument('dataset')
@click.option('--param', 'params', multiple=True, metavar='NAME=VALUE',
              help='Analysis parameter (repeatable).')
@click.option('--force', is_flag=True, help='Recompute even if a result is stored.')
@_profile_option
def analyse_run(name: str, dataset: str, params: Tuple[str, ...], force: bool,
                profile: Optional[str]):
    """Run analysis NAME over DATASET (or reuse its stored result)."""
    from .aiida import load_aiida_profile  # noqa: PLC0415
    from .analysis import store  # noqa: PLC0415
    values = _params(name, params)
    with _failing("Analysis failed", "Analysis failed"):
        load_aiida_profile(profile)
        node = store.result_node(name, dataset, values, force=force)
    extras = node.base.extras.all
    click.echo(f"rows: {extras.get('n_rows', 0)}")
    click.echo(f"dropped: {extras.get('n_dropped', 0)}")
    click.echo(f"uuid: {node.uuid}")


@analyse_group.command('show')
@click.argument('name')
@click.argument('dataset')
@click.option('--rows', default=10, show_default=True, help='Rows of the table to print.')
@_stored_param_option
@_profile_option
def analyse_show(name: str, dataset: str, rows: int, params: Tuple[str, ...],
                 profile: Optional[str]):
    """Print the stored result of NAME over DATASET: metadata and head."""
    import io  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    node = _stored(name, dataset, profile, params)
    click.echo(f"uuid: {node.uuid}")
    for key, value in sorted(node.base.extras.all.items()):
        click.echo(f"{key}: {value}")
    table = pd.read_csv(io.BytesIO(node.get_content(mode='rb')))
    click.echo(table.head(rows).to_string(index=False))


@analyse_group.command('export')
@click.argument('name')
@click.argument('dataset')
@click.argument('out', type=click.Path(dir_okay=False, path_type=Path))
@_stored_param_option
@_profile_option
def analyse_export(name: str, dataset: str, out: Path, params: Tuple[str, ...],
                   profile: Optional[str]):
    """Write the stored result of NAME over DATASET to OUT (CSV)."""
    node = _stored(name, dataset, profile, params)
    out.write_bytes(node.get_content(mode='rb'))
    click.echo(f"wrote {out}")


# =============================================================================
# FETCH COMMANDS
# =============================================================================

@cli.group('fetch')
def fetch_group():
    """Download published add-ons: the assistant's fine-tuned models and the materials pack."""


@fetch_group.command('materials')
@click.option('--remove', is_flag=True, help='Delete the installed materials pack.')
def fetch_materials_cmd(remove: bool):
    """Install the materials pack (pre-quenched amorphous blocks) from huggingface.co.

    Downloads every file the pack's manifest lists into
    $XDG_DATA_HOME/FrictionSim2D/materials-pack and checks each sha256 before
    the pack replaces what is installed; a failed fetch changes nothing.
    Material lookups then find its blocks after the packaged ones.

    \b
    Examples:
        FrictionSim2D fetch materials
        FrictionSim2D fetch materials --remove
    """
    from .data.fetch import FetchError, fetch_materials, remove_materials  # noqa: PLC0415
    if remove:
        click.echo("Removed the materials pack" if remove_materials()
                   else "No materials pack is installed")
        return
    try:
        folder = fetch_materials(click.echo)
    except FetchError as exc:
        _raise_abort(f"❌ {exc}", exc)
    click.echo(f"Installed the materials pack in {folder}")


@fetch_group.command('model')
@click.argument('tag', required=False)
@click.option('--remove', is_flag=True,
              help='Uninstall TAG from Ollama and delete its downloaded bundle.')
def fetch_model_cmd(tag: Optional[str], remove: bool):
    """Install the published assistant model TAG into Ollama as fs2d-assistant:TAG.

    Without TAG, lists the published tags and whether each is installed.
    Ollama is started first when it does not answer, as the Assistant pane
    does; a server started here stops again when the command ends.

    \b
    Examples:
        FrictionSim2D fetch model
        FrictionSim2D fetch model qwen3.5-4b
        FrictionSim2D fetch model qwen3.5-4b --remove
    """
    from .agent import _AGENT_AVAILABLE  # noqa: PLC0415
    if not _AGENT_AVAILABLE:
        click.echo("The assistant extra is not installed. Install with:", err=True)
        click.echo('   pip install -e ".[agent]"', err=True)
        raise click.Abort()
    if remove and tag is None:
        raise click.UsageError('--remove needs a TAG')
    from .agent.fetch import FetchError, fetch_command  # noqa: PLC0415
    try:
        fetch_command(tag, remove, click.echo)
    except FetchError as exc:
        _raise_abort(f"❌ {exc}", exc)


def main():
    try:
        cli()
    except Exception:  # pylint: disable=broad-except
        logger.exception("Command failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
