"""Command Line Interface for FrictionSim2D.

Unified CLI for simulation execution, HPC script generation, and AiiDA integration.
"""
# pylint: disable=import-outside-toplevel
# Deliberate: every subcommand imports its dependencies lazily so `FrictionSim2D --help`
# and compute-node use never pay for ase/aiida/Qt. See CLAUDE.md "Import weight".

import logging
import sys
from pathlib import Path
from typing import Optional, NoReturn, Tuple
import click

from .core.config import load_settings, settings_origin, _global_settings_path
from .run.hpc_dispatch import (
    _build_hpc_manifest_entries,
    collect_hpc_simulation_paths,
    dispatch_hpc_generation,
    hpc_generation_blocker,
)
from .run.simulate import run_simulations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _ensure_hpc_settings(hpc_settings) -> None:
    """Ensure minimal HPC settings are set without interactive prompts."""
    # Scheduler type and scratch directory are the only required values.
    if not getattr(hpc_settings, 'scheduler_type', None):
        hpc_settings.scheduler_type = 'pbs'

    if (getattr(hpc_settings, 'use_tmpdir', False)
            and not getattr(hpc_settings, 'scratch_dir', None)):
        hpc_settings.scratch_dir = "$TMPDIR"


def _run_simulation(model: str, config_file: str, output_dir: str,
                    use_aiida: bool, generate_hpc: bool,
                    settings_file: Optional[Path] = None,
                    aiida_label: Optional[str] = None):
    if use_aiida:
        from .aiida import AIIDA_AVAILABLE
        if not AIIDA_AVAILABLE:
            click.echo("⚠️  AiiDA not available. Install with:", err=True)
            click.echo("   conda install -c conda-forge aiida-core", err=True)
            raise click.Abort()

    try:
        created_simulations, simulation_root, configs_to_run, _ = run_simulations(
            config_file=config_file,
            model=model,
            settings_file=settings_file,
            output_root=Path(output_dir),
            ensure_hpc_settings=_ensure_hpc_settings,
            use_aiida=use_aiida,
            generate_hpc=generate_hpc,
        )
    except (ValueError, FileNotFoundError) as exc:
        # A rejected config is a user error, not a crash: expansion refusals
        # (`run/expand.py`), the runtime sweep-ordering check, and pydantic's
        # ValidationError (a ValueError subclass) all land here, and every one
        # of them already carries a message naming what to change. Without this
        # the user got a bare traceback and no CLI output at all, while
        # `hpc generate` right below reports its own refusals properly. The
        # catch stays narrow -- an unexpected error still surfaces its
        # traceback -- and the exception is logged so nothing is lost.
        # debug, not exception: the message is echoed below, and logging the
        # traceback at ERROR would print the very thing this replaces.
        logger.debug("Simulation generation rejected the config", exc_info=True)
        click.echo(f"❌ {exc}", err=True)
        raise click.Abort() from exc

    click.echo(f"📋 Found {len(configs_to_run)} simulation configurations")

    if created_simulations and use_aiida:
        from .aiida.ingest import SetRootMismatch, ingest_root  # noqa: PLC0415
        click.echo(f"\n📦 Registering {simulation_root.name} in AiiDA...")
        import_hint = (
            f"   The simulation tree was generated at {simulation_root.absolute()} "
            "and is intact; register it later with:\n"
            f"   FrictionSim2D aiida import {simulation_root} "
            f"--label {aiida_label or simulation_root.name}"
        )
        try:
            report = ingest_root(simulation_root, label=aiida_label)
        except SetRootMismatch as exc:
            logger.exception("AiiDA set/root mismatch")
            _raise_abort(f"❌ {exc}\n{import_hint}", exc)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("AiiDA registration failed")
            _raise_abort(f"❌ Registration failed: {exc}\n{import_hint}", exc)
        click.echo(report.render())
        if not report.ok():
            raise click.Abort()

    click.echo(
        f"\n✅ Generation complete: {len(created_simulations)}/"
        f"{len(configs_to_run)} successful"
    )
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
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this run only.')
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--label', '-l', 'aiida_label', default=None,
              help='Label for the AiiDA simulation set it creates '
                   '(default: the simulation root name).')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
                            help='Generate HPC scripts for the simulation root')
def run_afm(config_file: str, settings_file: Optional[Path], output_dir: str,
            use_aiida: bool, generate_hpc: bool, aiida_label: Optional[str]):
    """Generate AFM simulation files.

    Creates all necessary LAMMPS input files, structures, and potentials
    for tip-on-substrate friction simulations.

    Example:
        FrictionSim2D run.afm afm_config.ini -o ./afm_output --aiida
    """
    _run_simulation(
        'afm', config_file, output_dir, use_aiida, generate_hpc,
        settings_file=settings_file, aiida_label=aiida_label
    )

@run_group.command('sheetonsheet')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
          default=None,
          help='Path to a settings.yaml file for this run only.')
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--label', '-l', 'aiida_label', default=None,
              help='Label for the AiiDA simulation set it creates '
                   '(default: the simulation root name).')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
        help='Generate HPC scripts for the simulation root')
def run_sheetonsheet(config_file: str, settings_file: Optional[Path], output_dir: str,
             use_aiida: bool, generate_hpc: bool, aiida_label: Optional[str]):
    """Generate sheet-on-sheet simulation files.

    Creates all necessary LAMMPS input files for 4-layer sheet-on-sheet
    friction simulations.

    Example:
        FrictionSim2D run.sheetonsheet sheet_config.ini -o ./sheet_output
    """
    _run_simulation(
        'sheetonsheet', config_file, output_dir, use_aiida, generate_hpc,
        settings_file=settings_file, aiida_label=aiida_label
    )


@run_group.command('pes-sheet')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this run only.')
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--label', '-l', 'aiida_label', default=None,
              help='Label for the AiiDA simulation set it creates '
                   '(default: the simulation root name).')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
              help='Generate HPC scripts for the simulation root')
def run_pes_sheet(config_file: str, settings_file: Optional[Path], output_dir: str,
                  use_aiida: bool, generate_hpc: bool, aiida_label: Optional[str]):
    """Generate sheet-on-sheet interlayer PES scan files.

    Builds a frozen bilayer and writes a cheap, static lateral grid-scan of the
    interlayer potential-energy surface (corrugation barrier) instead of a
    sliding run. Reuses the sheet-on-sheet geometry; needs [2D] layers = [2].

    Example:
        FrictionSim2D run pes-sheet examples/pes_sheet_config.ini -o ./pes_output
    """
    _run_simulation(
        'pes_sheet', config_file, output_dir, use_aiida, generate_hpc,
        settings_file=settings_file, aiida_label=aiida_label
    )


@run_group.command('pes-tip')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this run only.')
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--label', '-l', 'aiida_label', default=None,
              help='Label for the AiiDA simulation set it creates '
                   '(default: the simulation root name).')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
              help='Generate HPC scripts for the simulation root')
def run_pes_tip(config_file: str, settings_file: Optional[Path], output_dir: str,
                use_aiida: bool, generate_hpc: bool, aiida_label: Optional[str]):
    """Generate tip-on-sheet (AFM) surface PES scan files.

    Builds and indents the AFM tip + sheet + substrate stack to a low load
    exactly as the sliding sim, then writes a static lateral grid-scan of the
    tip-felt surface PES (energy + lateral force on the tip) instead of sliding.

    Example:
        FrictionSim2D run pes-tip examples/pes_tip_config.ini -o ./pes_output
    """
    _run_simulation(
        'pes_tip', config_file, output_dir, use_aiida, generate_hpc,
        settings_file=settings_file, aiida_label=aiida_label
    )


@run_group.command('indent')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this run only.')
@click.option('--output-dir', '-o', default='simulation_output',
              help='Output directory for generated files')
@click.option('--aiida', 'use_aiida', is_flag=True,
              help='Enable AiiDA provenance tracking')
@click.option('--label', '-l', 'aiida_label', default=None,
              help='Label for the AiiDA simulation set it creates '
                   '(default: the simulation root name).')
@click.option('--hpc-scripts', 'generate_hpc', is_flag=True,
              help='Generate HPC scripts for the simulation root')
def run_indent(config_file: str, settings_file: Optional[Path], output_dir: str,
               use_aiida: bool, generate_hpc: bool, aiida_label: Optional[str]):
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
    """
    _run_simulation(
        'indent', config_file, output_dir, use_aiida, generate_hpc,
        settings_file=settings_file, aiida_label=aiida_label
    )


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
    import yaml  # noqa: PLC0415
    if origin:
        src = settings_origin(settings_file=settings_file)
        if src:
            click.echo(f"# Settings loaded from: {src}")
        else:
            click.echo("# Settings loaded from: hardcoded defaults (no settings.yaml found)")
    effective = load_settings(settings_file=settings_file)
    click.echo(yaml.dump(effective.model_dump(), default_flow_style=False))


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
    import yaml  # noqa: PLC0415

    target = _global_settings_path() if global_ else Path('settings.yaml')

    if target.exists() and not force:
        click.confirm(
            f"'{target}' already exists. Overwrite?",
            abort=True,
        )

    effective = load_settings()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(effective.model_dump(), sort_keys=False),
        encoding='utf-8',
    )
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
# HPC COMMANDS
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
    from .hpc import HPCScriptGenerator, HPCConfig

    sim_dir = Path(simulation_dir)
    out_dir = Path(output_dir) if output_dir else sim_dir / 'hpc'
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"🖥️  Generating {scheduler.upper()} scripts for: {sim_dir}")

    settings = load_settings(settings_file=settings_file)
    _ensure_hpc_settings(settings.hpc)
    settings.hpc.scheduler_type = scheduler  # CLI --scheduler overrides the settings default
    hpc_config = HPCConfig.from_settings(settings.hpc)

    script_names = hpc_config.lammps_scripts or ['system.in', 'slide.in']
    simulation_paths = collect_hpc_simulation_paths(sim_dir, lammps_scripts=script_names)

    if not simulation_paths:
        click.echo("❌ No simulation directories found", err=True)
        raise click.Abort()

    blocker = hpc_generation_blocker(settings)
    if blocker is not None:
        click.echo(f"❌ {blocker}", err=True)
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
    preflight = preflight or hpc_config.preflight
    result = dispatch_hpc_generation(
        generator,
        sim_dir,
        out_dir,
        scheduler,
        base_dir,
        manifest_entries,
        preflight=preflight,
    )

    if isinstance(result, dict):
        n_scripts = sum(len(v) for v in result.values())
        submit_file = 'submit_jobs.sh'
        click.echo(f"✅ Generated {n_scripts} two-phase script(s) in {out_dir}")
        click.echo("   (system → slide; slide jobs wait for system jobs to finish)")
    else:
        n_scripts = len(result)
        submit_file = 'submit_all.txt'
        click.echo(f"✅ Generated {n_scripts} script(s) in {out_dir}")
    click.echo("\n📝 Next steps:")
    click.echo(f"   1. Review scripts in {out_dir}")
    click.echo("   2. Transfer to HPC cluster")
    if preflight:
        submit_cmd = 'qsub' if scheduler == 'pbs' else 'sbatch'
        wrapper = f"preflight.{'pbs' if scheduler == 'pbs' else 'slurm'}"
        click.echo(f"   3. From the simulation root: "
                   f"{submit_cmd} {out_dir.name}/{wrapper}")
        click.echo(f"   4. Follow {submit_file}")
    else:
        click.echo(f"   3. Follow {submit_file}")


def _resolve_run_log(sim_dir: Path) -> Tuple[Optional[Path], str]:
    """Pick the log to enrich from when the caller named only a run directory.

    Prefers ``lammps/run_system.log``: ``system*.in`` is the only phase that
    prints RAMP_STATUS, and that is the name the job scripts derive for it.
    Falls back to a lone ``run*.log`` whatever it is called, because an
    unambiguous directory needs no naming convention to be usable -- a
    slide-only run, a log saved by hand, a future template that names its
    phases differently. It is a tolerance, not a compatibility shim: no
    released layout uses another name. Several logs with no ``run_system.log``
    among them is the one case worth refusing, since picking the wrong one
    would quietly record the wrong run's settle result.

    Returns ``(log_path, '')`` or ``(None, reason)``.
    """
    lammps_dir = Path(sim_dir) / 'lammps'
    preferred = lammps_dir / 'run_system.log'
    if preferred.is_file():
        return preferred, ''
    candidates = sorted(p for p in lammps_dir.glob('run*.log') if p.is_file())
    if len(candidates) == 1:
        return candidates[0], ''
    if not candidates:
        return None, f"no run log in {lammps_dir}"
    names = ', '.join(p.name for p in candidates)
    return None, (f"{len(candidates)} run logs and none named run_system.log "
                  f"({names}) -- name one explicitly")


def _enrich_one_run(sim_dir: Path, log_path: Optional[Path],
                    enrich) -> Tuple[str, str]:
    """Enrich one run directory. Returns (outcome, detail).

    ``outcome`` is one of 'enriched', 'unchanged', 'skipped'. Nothing raises:
    every failure becomes a 'skipped' with a reason, because the caller must
    never report a completed simulation as failed over a provenance nicety.
    """
    if log_path is None:
        log_path, why = _resolve_run_log(sim_dir)
        if log_path is None:
            return 'skipped', why

    record_path = Path(sim_dir) / 'provenance.json'
    # Snapshot rather than trust the return value: the function returns the
    # record path both when it merged fields and when the log held none, and
    # only the on-disk bytes distinguish those.
    before = (record_path.read_text(encoding='utf-8', errors='ignore')
              if record_path.is_file() else None)

    result = enrich(sim_dir, log_path)
    if result is None:
        return 'skipped', (f"no readable provenance.json, or no log at "
                           f"{log_path}")
    if result.read_text(encoding='utf-8', errors='ignore') == before:
        # Byte-identity, not "the log was empty": a real LAMMPS log always
        # carries at least the version banner, so the usual way to land here
        # is a repeat pass over a tree already swept.
        return 'unchanged', f"already up to date with {log_path.name}"
    return 'enriched', f"from {log_path.name}"


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
    # Lazy, like every other subcommand's imports here: keeps `--help` and the
    # unrelated groups off this module's (ASE-bearing) import path.
    from .builders.afm.base import enrich_provenance_post_run

    if recursive and log_path is not None:
        raise click.UsageError(
            "--recursive sweeps a whole tree and resolves each run's log "
            "itself; it cannot also take a single LOG_PATH.")

    if not recursive:
        outcome, detail = _enrich_one_run(sim_dir, log_path,
                                          enrich_provenance_post_run)
        click.echo(f"{outcome} {sim_dir}: {detail}")
        return

    # Reuse the shared discovery helper (the same one `hpc generate` and
    # `run --hpc-scripts` use) rather than a second bespoke walk -- a
    # divergent copy is exactly what the C2-1 refactor removed. Passing an
    # empty script list takes its "any directory with a lammps/*.in" branch,
    # which is what a post-run sweep wants: every run directory, whatever its
    # scripts were named.
    rel_paths = (collect_hpc_simulation_paths(sim_dir, lammps_scripts=[])
                 if sim_dir.is_dir() else [])
    if not rel_paths:
        click.echo(f"No simulation directories under {sim_dir}; "
                   f"nothing to enrich.")
        return

    tally = {'enriched': 0, 'unchanged': 0, 'skipped': 0}
    for rel in rel_paths:
        outcome, detail = _enrich_one_run(sim_dir / rel, None,
                                          enrich_provenance_post_run)
        tally[outcome] += 1
        click.echo(f"{outcome} {rel}: {detail}")
    click.echo(f"{tally['enriched']} enriched, {tally['unchanged']} unchanged, "
               f"{tally['skipped']} skipped.")


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
    from .aiida.preflight import main as preflight_main
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
        click.echo("\n📝 Setup AiiDA with: verdi presto --use-postgres")

@aiida_group.command('import')
@click.argument('simulation_folder', type=click.Path(exists=True, path_type=Path))
@click.option('--label', '-l', default=None,
              help='Label for the simulation set (default: the folder name).')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
@click.option('--dry-run', is_flag=True,
              help='Print what would be ingested; create nothing.')
def aiida_import(simulation_folder: Path, label: Optional[str],
                 profile: Optional[str], dry_run: bool):
    """Ingest a simulation tree into AiiDA.

    SIMULATION_FOLDER is the ``simulation_XXXXXXXX`` directory produced by one
    ``FrictionSim2D run``. Idempotent: re-importing adds only new or changed
    load points. Exits 1 if any run failed; a real (non-dry-run) import also
    exits 1 if any run was left empty or could not be parsed -- a dry-run
    preview of a fresh, pre-HPC tree is legitimately all-empty.

    Example:
        FrictionSim2D aiida import ./simulation_20260421_173546 -l 260421-afm
    """
    from .aiida.ingest import SetRootMismatch, ingest_root  # noqa: PLC0415

    try:
        report = ingest_root(simulation_folder, label=label, profile=profile,
                             dry_run=dry_run)
    except SetRootMismatch as exc:
        _raise_abort(f"❌ {exc}", exc)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Import failed")
        _raise_abort(f"❌ Import failed: {exc}", exc)

    click.echo(report.render())
    failed = not report.ok()
    if not dry_run:
        failed = failed or bool(report.empty_runs) or bool(report.unparseable)
    if failed:
        raise SystemExit(1)


@aiida_group.command('list-sets')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
def aiida_list_sets(profile: Optional[str], output_format: str):
    """List imported simulation sets stored in AiiDA.

    Example:
        FrictionSim2D aiida list-sets
        FrictionSim2D aiida list-sets --format json
    """
    from .aiida.integration import list_simulation_sets  # noqa: PLC0415

    try:
        rows = list_simulation_sets(profile=profile)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to list simulation sets")
        _raise_abort(f"❌ List failed: {exc}", exc)

    if not rows:
        click.echo("No simulation sets found in the current AiiDA profile.")
        return

    if output_format == 'json':
        import json
        click.echo(json.dumps(rows, indent=2))
        return

    if output_format == 'csv':
        import csv  # noqa: PLC0415
        import io  # noqa: PLC0415
        keys = ['label', 'model', 'created', 'n_runs', 'n_results',
                'materials', 'git_commit', 'source_host', 'uuid', 'pk']
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        click.echo(buffer.getvalue().rstrip())
        return

    headers = [
        ('Label', 'label'),
        ('Model', 'model'),
        ('Runs', 'n_runs'),
        ('Results', 'n_results'),
        ('Materials', 'materials'),
        ('Host', 'source_host'),
        ('UUID', 'uuid'),
    ]
    widths = []
    for title, key in headers:
        max_value_width = max(len(str(row.get(key, ''))) for row in rows)
        widths.append(max(len(title), max_value_width))

    header_line = '  '.join(title.ljust(widths[idx]) for idx, (title, _) in enumerate(headers))
    sep_line = '  '.join('-' * widths[idx] for idx in range(len(headers)))
    click.echo(header_line)
    click.echo(sep_line)
    for row in rows:
        click.echo(
            '  '.join(
                str(row.get(key, '')).ljust(widths[idx])
                for idx, (_, key) in enumerate(headers)
            )
        )
    click.echo(sep_line)
    click.echo(f"\nTotal sets: {len(rows)}")


@aiida_group.command('export-results')
@click.argument('set_label')
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
def aiida_export_results(set_label: str, output_dir: Path, profile: Optional[str]):
    """Write a stored set back out as an ordinary simulation tree.

    The result works unchanged with ``postprocess read`` and the GUI download.

    Example:
        FrictionSim2D aiida export-results 260421-afm ~/exported/260421-afm
    """
    from .aiida.export import export_results  # noqa: PLC0415

    try:
        written = export_results(set_label, output_dir, profile=profile)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Export failed")
        _raise_abort(f"❌ Export failed: {exc}", exc)

    click.echo(f"✅ Wrote {len(written)} file(s) to {Path(output_dir).absolute()}")


@aiida_group.command('rebuild')
@click.argument('set_label')
@click.option('--output-dir', '-o', default='.', type=click.Path(path_type=Path),
              help='Root directory to write the simulation tree into (default: cwd)')
@click.option('--hpc-scripts', is_flag=True, default=False,
              help='Generate HPC job-submission scripts after rebuilding')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
def aiida_rebuild(set_label: str, output_dir: Path, hpc_scripts: bool,
                   profile: Optional[str]):
    """Regenerate a set's LAMMPS inputs from the store.

    Exports each run's stored config.json and the set's settings.yaml, then
    reruns the real builder router -- the same one ``FrictionSim2D run`` uses.

    Example:
        FrictionSim2D aiida rebuild 260421-afm -o ~/rebuild_test --hpc-scripts
    """
    from .aiida.export import rebuild_set  # noqa: PLC0415

    try:
        created, root = rebuild_set(set_label, output_dir, profile=profile,
                                    generate_hpc=hpc_scripts)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Rebuild failed")
        _raise_abort(f"❌ Rebuild failed: {exc}", exc)

    click.echo(f"✅ Rebuilt {len(created)} run(s) → {Path(root).absolute()}")


@aiida_group.command('delete')
@click.argument('set_label')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
@click.confirmation_option(
    prompt='This will permanently delete the simulation set and all linked nodes. Continue?'
)
def aiida_delete(set_label: str, profile: Optional[str]):
    """Delete a single simulation set and all its linked nodes.

    SET_LABEL is the label of the set to delete. All linked SimulationData
    and ResultsData nodes, and the set's own Group, are also removed.

    Example:
        FrictionSim2D aiida delete 260312-force_rebomos
    """
    from .aiida.integration import delete_simulation_set  # noqa: PLC0415

    click.echo(f"🗑️  Deleting simulation set '{set_label}'…")
    try:
        n = delete_simulation_set(set_label, profile=profile)
        click.echo(f"✅ Deleted {n} nodes for set '{set_label}'.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Delete failed")
        _raise_abort(f"❌ Delete failed: {exc}", exc)


@aiida_group.command('clear')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
@click.confirmation_option(
    prompt='This will permanently delete all FrictionSim2D nodes. Continue?'
)
def aiida_clear(profile: Optional[str]):
    """Delete all FrictionSim2D nodes from the AiiDA database.

    This removes every SimulationSet, Simulation, and Results node in the
    profile. The action is irreversible.

    Example:
        FrictionSim2D aiida clear
    """
    from .aiida.integration import clear_all_nodes  # noqa: PLC0415

    click.echo("🗑️  Deleting all FrictionSim2D nodes…")
    try:
        n = clear_all_nodes(profile=profile)
        click.echo(f"✅ Deleted {n} nodes.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Clear failed")
        _raise_abort(f"❌ Clear failed: {exc}", exc)


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
@click.option('--profile', default=None,
                            help='AiiDA profile name (default: configured default profile)')
def aiida_query(  # pylint: disable=too-many-arguments,too-many-locals
        materials: Tuple[str, ...], family: Optional[str], model: Optional[str],
        set_label: Optional[str], potential: Optional[str], layers: Optional[int],
        force_min: Optional[float], force_max: Optional[float],
        pressure_min: Optional[float], pressure_max: Optional[float],
        angle_min: Optional[float], angle_max: Optional[float],
        speed_min: Optional[float], speed_max: Optional[float],
        temperature_min: Optional[float], temperature_max: Optional[float],
        limit: Optional[int], order_by: Optional[str], output_format: str,
        output: Optional[str], profile: Optional[str]):
    """Query simulation runs and their results.

    Example:
        FrictionSim2D aiida query --material h-MoS2 --layers 2 --format csv
        FrictionSim2D aiida query --set-label 251113-afm --format table
    """
    from .aiida import load_aiida_profile
    from .aiida.query import Friction2DDB

    def _range(lo, hi):
        return (lo, hi) if lo is not None or hi is not None else None

    try:
        load_aiida_profile(profile)
        db = Friction2DDB()
        available = db.get_available_conditions()['set_labels'] if set_label else []
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA query setup failed")
        _raise_abort(f"❌ Query failed: {exc}", exc)

    if set_label and set_label not in available:
        raise click.ClickException(
            f"No set found with label '{set_label}'. Available: {available}")

    query_kwargs = {
        'materials': list(materials) if materials else None,
        'family': family, 'model': model, 'set_label': set_label,
        'potential': potential, 'layers': layers,
        'force_range': _range(force_min, force_max),
        'pressure_range': _range(pressure_min, pressure_max),
        'angle_range': _range(angle_min, angle_max),
        'speed_range': _range(speed_min, speed_max),
        'temperature_range': _range(temperature_min, temperature_max),
        'limit': limit, 'order_by': order_by,
    }
    shown = {k: v for k, v in query_kwargs.items() if v is not None}
    click.echo(f"🔍 Querying database with filters: {shown}")
    try:
        results = db.query(**query_kwargs)
        click.echo(f"📊 Found {results.total_count} results")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA query failed")
        _raise_abort(f"❌ Query failed: {exc}", exc)

    try:
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
                # The rows, like `--format table/csv` print -- not just the
                # query parameters.
                click.echo(results.to_dataframe().to_json(
                    orient='records', indent=2, default_handler=str))

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA query failed")
        _raise_abort(f"❌ Query failed: {exc}", exc)


@aiida_group.command('setup')
@click.option('--create-profile', '-p', 'create_profile', default=None,
              help='Name of the AiiDA profile to CREATE (default: friction2d). '
                   'Every other command spells --profile, which LOADS an existing one.')
@click.option('--lammps-path', type=click.Path(),
              help='Explicit path to the LAMMPS executable')
def aiida_setup(create_profile: Optional[str], lammps_path: Optional[str]):
    """Set up the AiiDA profile, the localhost computer and the LAMMPS code.

    localhost only: on the HPC, run this inside the HPC conda environment
    (hpc-direct mode) and bring the archive home with `aiida export`.

    Example:
        FrictionSim2D aiida setup --lammps-path /usr/local/bin/lmp
    """
    from .aiida.setup import full_setup  # noqa: PLC0415

    click.echo("🔧 Running AiiDA first-time setup ...")
    try:
        full_setup(profile_name=create_profile or 'friction2d',
                   lammps_executable=lammps_path)
        click.echo("✅ AiiDA setup complete")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA setup failed")
        _raise_abort(f"❌ Setup failed: {exc}", exc)


@aiida_group.command('export')
@click.argument('set_label')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              default='friction2d.aiida', help='Output archive path')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
def aiida_export(set_label: str, output: Path, profile: Optional[str]):
    """Export one simulation set (set + runs + results) to a .aiida archive.

    The archive can be transferred to another machine and imported with
    ``FrictionSim2D aiida import-archive``.

    Example:
        FrictionSim2D aiida export 260421-afm -o ~/transfer/260421-afm.aiida
    """
    from .aiida.export import export_archive  # noqa: PLC0415

    click.echo(f"📦 Exporting set {set_label!r} to {output} ...")
    try:
        path = export_archive(set_label, output, profile=profile)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA export failed")
        _raise_abort(f"❌ Export failed: {exc}", exc)
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
    try:
        n_new = import_archive(archive_path, profile=profile)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA import-archive failed")
        _raise_abort(f"❌ Import failed: {exc}", exc)
    click.echo(f"✅ Imported {n_new} new node(s)")


@aiida_group.command('package')
@click.argument('simulation_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Output archive path (.tar.gz)')
def aiida_package(simulation_dir: str, output: Optional[str]):
    """Create a tar.gz archive of simulation inputs for transfer."""
    import tarfile  # noqa: PLC0415
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
@click.option('--settings-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help='Path to a settings.yaml file for this submission.')
@click.option('--code', '-c', default=None,
              help='AiiDA code label (auto-detects if not specified)')
@click.option('--scripts', default=None,
              help='Comma-separated list of LAMMPS scripts to run in order')
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
@click.option('--daemon', is_flag=True, default=False,
              help='Submit to the AiiDA daemon (needs RabbitMQ). '
                   'Default: run in-process, which needs no broker.')
@click.option('--profile', default=None,
              help='AiiDA profile name (default: configured default profile)')
def aiida_submit(  # pylint: disable=too-many-arguments
    simulation_dir: str,
    settings_file: Optional[Path],
    code: Optional[str],
    scripts: Optional[str],
    machines: Optional[int],
    mpiprocs: Optional[int],
    walltime: Optional[str],
    queue: Optional[str],
    project: Optional[str],
    dry_run: bool,
    daemon: bool,
    profile: Optional[str],
):
    """Run (or, with --daemon, submit) simulations as AiiDA CalcJobs.

    SIMULATION_DIR is a simulation_* root or any directory inside one; every
    run at or below it is submitted. Register the tree first (``run --aiida``
    or ``aiida import``) so results are stored against its run nodes.
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
    from .aiida.submit import smart_submit

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
            settings_file=settings_file,
            code_label=code,
            scripts=scripts,
            dry_run=dry_run,
            daemon=daemon,
            profile=profile,
            **overrides,
        )

        if not dry_run and processes and not daemon and not all(
                p.is_finished_ok for p in processes):
            raise SystemExit(1)

    except click.ClickException:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AiiDA submit failed")
        _raise_abort(f"❌ Submission failed: {exc}", exc)

# =============================================================================
# POSTPROCESSING COMMANDS
# =============================================================================

@cli.group('postprocess')
def postprocess_group():
    """Read, analyse and plot simulation results."""


def _merge_settings(base: dict, overlay: dict) -> None:
    """Merge `overlay` into `base`; a list REPLACES, never extends.

    The same rule as `PlotCommonMixin._deep_merge_dict`, repeated here rather
    than imported so `--validate` never has to import the plot stack (and
    matplotlib with it).
    """
    for key, value in overlay.items():
        if (key in base and isinstance(base[key], dict)
                and isinstance(value, dict)):
            _merge_settings(base[key], value)
        else:
            base[key] = value


def _integrity_thresholds(path):
    """`IntegritySettings` from `path`, else from the user's settings."""
    import yaml  # noqa: PLC0415
    from .core.config import IntegritySettings  # noqa: PLC0415
    if path is None:
        return load_settings().integrity
    try:
        with open(path, encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        block = data.get('integrity', {}) if isinstance(data, dict) else None
        if not isinstance(block, dict):
            raise ValueError('expected an `integrity:` mapping')
        return IntegritySettings(**block)
    except (OSError, yaml.YAMLError, ValueError, TypeError) as exc:
        raise click.ClickException(
            f'cannot read integrity settings {path}: {exc}') from exc


def _echo_integrity(reader) -> None:
    """The integrity tally, a ruling stub per quarantined point, warnings."""
    from .postprocessing.integrity_rulings import point_id  # noqa: PLC0415
    failed = [p for p in reader.points if p.status == 'failed_integrity']
    held = [p for p in reader.points if p.status == 'quarantined']
    click.echo(f"Integrity: {len(failed)} failed, {len(held)} quarantined "
               f"(thresholds v{reader.integrity_settings.version})")
    for point in held:
        reason = ' '.join(str(point.status_detail or '').split())
        click.echo(f'- point: "{point_id(point, reader.results_dir)}"\n'
                   f'  ruling:   # pass or fail -- {reason}')
    overridden = [p for p in reader.points
                  if p.integrity is not None and p.integrity.overridden]
    click.echo(f"Overrides: {len(overridden)}")
    for point in overridden:
        note = ' '.join(str(point.status_detail or '').split())
        note = note.rsplit('overridden by user', 1)[-1].strip()
        click.echo(f"- {point_id(point, reader.results_dir)}: "
                   f"{point.integrity.original_status} -> {point.status}"
                   f" ({note})")
    for warning in reader.ruling_warnings:
        click.echo(f"Warning: {warning}")


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

    thresholds = None
    if not no_integrity:
        thresholds = _integrity_thresholds(integrity_settings_path)
    reader = DataReader(results_dir=results_dir, skip_fraction=skip_fraction,
                        complete_only=complete_only,
                        integrity=not no_integrity,
                        integrity_settings=thresholds)
    reader.export_issue_reports()
    if not reader.points:
        click.echo(f"No runs found under {results_dir}.", err=True)
        return
    counts: dict = {}
    for point in reader.points:
        counts[point.status] = counts.get(point.status, 0) + 1
    summary = ', '.join(f'{k}={v}' for k, v in sorted(counts.items()) if v)
    click.echo(f"Found {len(reader.points)} load point(s) ({summary}) "
               f"across {len({p.size for p in reader.points})} size(s).")
    if reader.integrity_settings is not None:
        _echo_integrity(reader)
    if do_export:
        for path in reader.export_full_data_to_json():
            click.echo(f"✅ {path}")
        for path in reader.export_summary_csv():
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
    import copy  # noqa: PLC0415
    import json  # noqa: PLC0415

    from .postprocessing.plot_config import (load_plot_document,  # noqa: PLC0415
                                             schema_markdown)

    if show_schema:
        click.echo(schema_markdown())
        return

    if init_path:
        from .postprocessing.plot_config import (PlotDocument,  # noqa: PLC0415
                                                 SummaryPlot)

        dirs = [str(Path(d).resolve()) for d in init_data_dirs] or ['./my_run']
        document = PlotDocument(
            data_dirs=dirs,
            labels=[Path(d).name for d in dirs],
            plots=[
                # plot_type is passed explicitly so exclude_unset keeps it:
                # the Literal default would be dropped from the dump, and the
                # written document would then fail its own discriminator.
                SummaryPlot(plot_type='summary', x_axis='force', y_axis='lf',
                            filename='lf_vs_load.png', add_linear_fit=True),
                SummaryPlot(plot_type='summary', x_axis='force', y_axis='cof',
                            filename='cof_vs_load.png'),
            ])
        target = Path(init_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            document.model_dump_json(indent=2, exclude_unset=True),
            encoding='utf-8')
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

    from .postprocessing.plot_style import DEFAULT_SETTINGS  # noqa: PLC0415
    from .postprocessing.presets import load_preset  # noqa: PLC0415

    plot_settings = copy.deepcopy(DEFAULT_SETTINGS)
    if preset:
        try:
            _merge_settings(plot_settings, load_preset(preset))
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc
    if settings:
        with open(settings, 'r', encoding='utf-8') as fh:
            _merge_settings(plot_settings, json.load(fh))

    if validate:
        click.echo(f"OK: {len(document.plots)} plot(s), "
                   f"{len(document.data_dirs)} dataset(s).")
        return

    from .postprocessing import Plotter  # noqa: PLC0415

    plotter = Plotter(
        document.data_dirs, document.labels, output_dir, plot_settings,
        dataset_display_labels=document.dataset_display_labels,
        series_color_map=document.series_color_map,
    )
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


def main():
    try:
        cli()
    except Exception:  # pylint: disable=broad-except
        logger.exception("Command failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
