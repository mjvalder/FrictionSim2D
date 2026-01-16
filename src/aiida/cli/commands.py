"""CLI commands for FrictionSim2D AiiDA plugin.

This module provides the command-line interface for managing friction
simulations through the offline HPC workflow.
"""

import json
import sys
from pathlib import Path
from typing import Optional, List
import click


@click.group()
@click.version_option()
def friction2d():
    """FrictionSim2D AiiDA plugin - Manage friction simulations."""
    pass


# =============================================================================
# PREPARATION COMMANDS
# =============================================================================

@friction2d.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default=None,
              help='Output directory for generated files')
@click.option('--simulation-type', '-t', type=click.Choice(['afm', 'sheetonsheet']),
              default='afm', help='Type of simulation')
@click.option('--register/--no-register', default=True,
              help='Register simulations in AiiDA database')
def prepare(config_file: str, 
            output_dir: Optional[str],
            simulation_type: str,
            register: bool):
    """Generate simulation files from a config file.
    
    This command uses FrictionSim2D to generate LAMMPS input files
    and optionally registers them in the AiiDA database.
    
    Example:
        friction2d prepare afm_config.ini -o ./output
    """
    from src.core.config import AFMSimulationConfig
    from src.core.utils import read_config
    from src.builders.afm import AFMSimulation
    from src.aiida.hpc import JobManifest
    
    config_path = Path(config_file)
    
    if output_dir is None:
        output_dir = Path.cwd() / f"{simulation_type}_output"
    else:
        output_dir = Path(output_dir)
    
    click.echo(f"📁 Preparing simulations from: {config_path}")
    click.echo(f"📂 Output directory: {output_dir}")
    
    try:
        # Read and validate config
        config_dict = read_config(config_path)
        
        # Determine materials to process
        materials = []
        if '2D' in config_dict and 'materials_list' in config_dict['2D']:
            materials_list_path = config_path.parent / config_dict['2D']['materials_list']
            if materials_list_path.exists():
                materials = [
                    line.strip() for line in materials_list_path.read_text().splitlines()
                    if line.strip() and not line.startswith('#')
                ]
                click.echo(f"📋 Found {len(materials)} materials in list")
        
        if not materials and '2D' in config_dict:
            mat = config_dict['2D'].get('mat', '')
            if mat and '{mat}' not in mat:
                materials = [mat]
        
        # Generate simulations
        n_generated = 0
        
        if simulation_type == 'afm':
            from src import afm
            
            with click.progressbar(length=1, label='Generating simulations') as bar:
                afm(str(config_path), output_dir=str(output_dir))
                bar.update(1)
                n_generated = 1  # Will be updated by manifest
        
        # Create manifest
        manifest = JobManifest.from_simulation_directory(output_dir)
        manifest.config_file = str(config_path)
        manifest_path = output_dir / 'manifest.json'
        manifest.save(manifest_path)
        
        click.echo(f"\n✅ Generated {manifest.n_jobs} simulations")
        click.echo(f"📄 Manifest saved to: {manifest_path}")
        
        # Register in AiiDA if requested
        if register:
            try:
                from src.aiida.data import (
                    FrictionSimulationData,
                    FrictionConfigData,
                    FrictionProvenanceData,
                )
                
                # Create config node
                config_node = FrictionConfigData.from_files(
                    config_path,
                    settings_path=config_path.parent / 'settings.yaml'
                )
                config_node.store()
                manifest.config_node_uuid = str(config_node.uuid)
                
                # Create provenance node
                prov_node = FrictionProvenanceData.from_simulation_config(
                    config_dict, config_path.parent
                )
                prov_node.store()
                manifest.provenance_node_uuid = str(prov_node.uuid)
                
                click.echo(f"📦 Registered config in AiiDA: {config_node.uuid}")
                
                # Update manifest with UUIDs
                manifest.save(manifest_path)
                
            except ImportError:
                click.echo("⚠️  AiiDA not available, skipping registration")
            except Exception as e:
                click.echo(f"⚠️  Failed to register in AiiDA: {e}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@friction2d.command('export')
@click.argument('simulation_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), required=True,
              help='Output directory for the HPC package')
@click.option('--scheduler', '-s', type=click.Choice(['pbs', 'slurm']),
              default='pbs', help='HPC scheduler type')
@click.option('--walltime', '-w', type=int, default=20,
              help='Walltime in hours')
@click.option('--cpus', '-c', type=int, default=32,
              help='CPUs per node')
@click.option('--memory', '-m', type=int, default=62,
              help='Memory in GB')
@click.option('--max-array', type=int, default=300,
              help='Maximum jobs per array')
@click.option('--modules', multiple=True, default=None,
              help='Modules to load (can specify multiple)')
def export_package(simulation_dir: str,
                   output_dir: str,
                   scheduler: str,
                   walltime: int,
                   cpus: int,
                   memory: int,
                   max_array: int,
                   modules: tuple):
    """Export simulations as an HPC-ready package.
    
    Creates a self-contained package with all simulation files and
    HPC submission scripts ready for transfer to a cluster.
    
    Example:
        friction2d export ./afm_output -o ./hpc_package -s pbs
    """
    from src.aiida.hpc import HPCScriptGenerator, HPCConfig, JobManifest
    from src.aiida.hpc.scripts import create_hpc_package
    
    sim_dir = Path(simulation_dir)
    out_dir = Path(output_dir)
    
    click.echo(f"📦 Creating HPC package from: {sim_dir}")
    click.echo(f"📂 Output directory: {out_dir}")
    click.echo(f"🖥️  Scheduler: {scheduler.upper()}")
    
    # Configure HPC settings
    config = HPCConfig(
        walltime_hours=walltime,
        cpus_per_node=cpus,
        memory_gb=memory,
        max_array_size=max_array,
    )
    
    if modules:
        config.modules = list(modules)
    
    try:
        package_dir = create_hpc_package(
            sim_dir, out_dir, scheduler=scheduler, config=config
        )
        
        # Load or create manifest
        manifest_path = sim_dir / 'manifest.json'
        if manifest_path.exists():
            manifest = JobManifest.load(manifest_path)
        else:
            manifest = JobManifest.from_simulation_directory(sim_dir)
        
        manifest.scheduler = scheduler
        manifest.package_directory = str(package_dir)
        
        # Mark all jobs as exported
        for job in manifest.jobs:
            from src.aiida.hpc import JobStatus
            job.update_status(JobStatus.EXPORTED)
        
        # Save manifest in package
        manifest.save(package_dir / 'manifest.json')
        
        click.echo(f"\n✅ Package created: {package_dir}")
        click.echo(f"📊 Contains {manifest.n_jobs} simulations")
        click.echo(f"\n📝 Next steps:")
        click.echo(f"   1. Transfer {package_dir} to your HPC cluster")
        click.echo(f"   2. cd to scripts/ directory")
        click.echo(f"   3. Run ./submit_all.sh")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


# =============================================================================
# IMPORT COMMANDS
# =============================================================================

@friction2d.command('import')
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--manifest', '-m', type=click.Path(exists=True), default=None,
              help='Path to manifest file')
@click.option('--process/--no-process', default=True,
              help='Run postprocessing on results')
@click.option('--store/--no-store', default=True,
              help='Store results in AiiDA database')
def import_results(results_dir: str,
                   manifest: Optional[str],
                   process: bool,
                   store: bool):
    """Import completed simulation results.
    
    Imports results from HPC, runs postprocessing, and stores
    data in the AiiDA database.
    
    Example:
        friction2d import ./returned_results -m manifest.json
    """
    from src.aiida.hpc import JobManifest, JobStatus
    
    results_path = Path(results_dir)
    
    click.echo(f"📥 Importing results from: {results_path}")
    
    # Load manifest
    if manifest:
        manifest_path = Path(manifest)
    else:
        # Try to find manifest in results directory
        manifest_path = results_path / 'manifest.json'
        if not manifest_path.exists():
            manifest_path = results_path.parent / 'manifest.json'
    
    if manifest_path.exists():
        job_manifest = JobManifest.load(manifest_path)
        click.echo(f"📄 Loaded manifest: {job_manifest.name}")
    else:
        click.echo("⚠️  No manifest found, creating from directory structure")
        job_manifest = JobManifest.from_simulation_directory(results_path)
    
    # Check for completed simulations
    completed = job_manifest.mark_completed_from_results(results_path)
    click.echo(f"✅ Found {len(completed)} completed simulations")
    
    if process:
        click.echo("\n🔄 Running postprocessing...")
        try:
            from src.postprocessing.read_data import DataReader
            
            reader = DataReader(results_dir=str(results_path))
            reader.export_full_data_to_json()
            reader.export_issue_reports()
            
            click.echo("✅ Postprocessing complete")
            
        except Exception as e:
            click.echo(f"⚠️  Postprocessing failed: {e}")
    
    if store:
        click.echo("\n📦 Storing results in AiiDA...")
        try:
            from src.aiida.data import (
                FrictionSimulationData,
                FrictionResultsData,
            )
            
            stored_count = 0
            
            for job in job_manifest.jobs:
                if job.status != JobStatus.COMPLETED.value:
                    continue
                
                try:
                    # Create simulation node
                    sim_node = FrictionSimulationData()
                    sim_node.material = job.material
                    sim_node.layers = job.layers
                    sim_node.force = job.force
                    sim_node.scan_angle = job.angle
                    sim_node.scan_speed = job.speed
                    sim_node.simulation_path = job.simulation_path
                    sim_node.status = 'imported'
                    
                    # Link to config if available
                    if job_manifest.config_node_uuid:
                        sim_node.config_uuid = job_manifest.config_node_uuid
                    if job_manifest.provenance_node_uuid:
                        sim_node.provenance_uuid = job_manifest.provenance_node_uuid
                    
                    sim_node.store()
                    job.simulation_node_uuid = str(sim_node.uuid)
                    job.update_status(JobStatus.IMPORTED)
                    stored_count += 1
                    
                except Exception as e:
                    click.echo(f"⚠️  Failed to store {job.job_id}: {e}")
            
            click.echo(f"✅ Stored {stored_count} simulations in AiiDA")
            
        except ImportError:
            click.echo("⚠️  AiiDA not available, skipping storage")
        except Exception as e:
            click.echo(f"⚠️  Failed to store in AiiDA: {e}")
    
    # Save updated manifest
    job_manifest.save(results_path / 'manifest_imported.json')
    click.echo(f"\n📄 Updated manifest saved")
    
    # Summary
    summary = job_manifest.get_summary()
    click.echo(f"\n📊 Summary:")
    click.echo(f"   Total: {summary['total_jobs']}")
    click.echo(f"   Completed: {summary['completed']}")
    click.echo(f"   Failed: {summary['failed']}")
    click.echo(f"   Imported: {summary['imported']}")


# =============================================================================
# STATUS COMMANDS
# =============================================================================

@friction2d.command()
@click.argument('manifest_file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Show detailed status')
def status(manifest_file: str, verbose: bool):
    """Check status of simulations from manifest.
    
    Example:
        friction2d status manifest.json
    """
    from src.aiida.hpc import JobManifest
    
    manifest = JobManifest.load(Path(manifest_file))
    summary = manifest.get_summary()
    
    click.echo(f"\n📊 Manifest: {summary['name']}")
    click.echo(f"   Created: {summary['created_at'][:19]}")
    click.echo(f"   Updated: {summary['last_updated'][:19]}")
    click.echo(f"   Scheduler: {summary['scheduler'].upper()}")
    click.echo()
    click.echo(f"   Total jobs: {summary['total_jobs']}")
    click.echo(f"   ├─ Prepared:  {summary['prepared']}")
    click.echo(f"   ├─ Submitted: {summary['submitted']}")
    click.echo(f"   ├─ Completed: {summary['completed']}")
    click.echo(f"   ├─ Failed:    {summary['failed']}")
    click.echo(f"   └─ Imported:  {summary['imported']}")
    
    if verbose:
        click.echo(f"\n📋 Job Details:")
        for job in manifest.jobs:
            status_icon = {
                'prepared': '⏳',
                'exported': '📦',
                'submitted': '🚀',
                'running': '🔄',
                'completed': '✅',
                'failed': '❌',
                'imported': '📥',
            }.get(job.status, '❓')
            
            click.echo(f"   {status_icon} {job.job_id}: {job.status}")
            if job.error_message:
                click.echo(f"      Error: {job.error_message}")


@friction2d.command('mark')
@click.argument('manifest_file', type=click.Path(exists=True))
@click.option('--submitted', is_flag=True, help='Mark all as submitted')
@click.option('--job-prefix', type=str, default=None, help='HPC job ID prefix')
def mark(manifest_file: str, submitted: bool, job_prefix: Optional[str]):
    """Manually update job status in manifest.
    
    Example:
        friction2d mark manifest.json --submitted --job-prefix 12345
    """
    from src.aiida.hpc import JobManifest
    
    manifest_path = Path(manifest_file)
    manifest = JobManifest.load(manifest_path)
    
    if submitted:
        count = manifest.mark_all_submitted(job_prefix)
        click.echo(f"✅ Marked {count} jobs as submitted")
        manifest.save(manifest_path)


# =============================================================================
# QUERY COMMANDS
# =============================================================================

@friction2d.command()
@click.option('--material', '-m', type=str, default=None,
              help='Filter by material')
@click.option('--layers', '-l', type=int, default=None,
              help='Filter by layer count')
@click.option('--force', '-f', type=float, default=None,
              help='Filter by force (nN)')
@click.option('--status', '-s', type=str, default=None,
              help='Filter by status')
@click.option('--limit', type=int, default=20,
              help='Maximum results to show')
@click.option('--export', '-e', type=click.Path(), default=None,
              help='Export results to CSV')
def query(material: Optional[str],
          layers: Optional[int],
          force: Optional[float],
          status: Optional[str],
          limit: int,
          export: Optional[str]):
    """Query the Friction2DDB database.
    
    Example:
        friction2d query -m h-MoS2 -l 2 --export results.csv
    """
    try:
        from src.aiida.db import Friction2DDB
    except ImportError:
        click.echo("❌ AiiDA not available", err=True)
        raise click.Abort()
    
    db = Friction2DDB()
    
    # Build query parameters
    query_params = {}
    if material:
        query_params['materials'] = [material]
    if layers:
        query_params['layers'] = layers
    if force:
        query_params['force_range'] = (force - 0.1, force + 0.1)
    if status:
        query_params['status'] = status
    query_params['limit'] = limit
    
    result = db.query(**query_params)
    
    click.echo(f"\n🔍 Found {result.total_count} simulations")
    
    if result.simulations:
        click.echo("\n" + "-" * 80)
        for sim in result.simulations:
            click.echo(
                f"{sim.material:15} L{sim.layers} F{sim.force:5.1f}nN "
                f"A{sim.scan_angle:4.0f}° [{sim.status}]"
            )
        click.echo("-" * 80)
    
    if export:
        export_path = result.export_csv(Path(export))
        click.echo(f"\n📁 Exported to: {export_path}")


@friction2d.command('stats')
def stats():
    """Show database statistics.
    
    Example:
        friction2d stats
    """
    try:
        from src.aiida.db import Friction2DDB
    except ImportError:
        click.echo("❌ AiiDA not available", err=True)
        raise click.Abort()
    
    db = Friction2DDB()
    statistics = db.get_statistics()
    
    click.echo("\n📊 Friction2DDB Statistics")
    click.echo("=" * 40)
    click.echo(f"Total simulations: {statistics['total_simulations']}")
    click.echo(f"Unique materials:  {statistics['n_materials']}")
    
    click.echo("\nBy Status:")
    for status, count in sorted(statistics['by_status'].items()):
        click.echo(f"  {status:15} {count:5}")
    
    click.echo("\nBy Type:")
    for sim_type, count in sorted(statistics['by_type'].items()):
        click.echo(f"  {sim_type:15} {count:5}")
    
    if statistics['n_materials'] <= 10:
        click.echo("\nBy Material:")
        for mat, count in sorted(statistics['by_material'].items(), 
                                  key=lambda x: -x[1]):
            click.echo(f"  {mat:20} {count:5}")


@friction2d.command('materials')
def list_materials():
    """List all available materials in the database.
    
    Example:
        friction2d materials
    """
    try:
        from src.aiida.db import Friction2DDB
    except ImportError:
        click.echo("❌ AiiDA not available", err=True)
        raise click.Abort()
    
    db = Friction2DDB()
    materials = db.get_available_materials()
    
    click.echo(f"\n📋 Available Materials ({len(materials)} total):")
    click.echo("-" * 40)
    
    # Group by prefix
    grouped = {}
    for mat in materials:
        prefix = mat.split('-')[0] if '-' in mat else 'other'
        grouped.setdefault(prefix, []).append(mat)
    
    for prefix, mats in sorted(grouped.items()):
        click.echo(f"\n{prefix}:")
        for mat in sorted(mats):
            click.echo(f"  {mat}")


# =============================================================================
# UTILITY COMMANDS
# =============================================================================

@friction2d.command('conditions')
def list_conditions():
    """List available experimental conditions in database.
    
    Example:
        friction2d conditions
    """
    try:
        from src.aiida.db import Friction2DDB
    except ImportError:
        click.echo("❌ AiiDA not available", err=True)
        raise click.Abort()
    
    db = Friction2DDB()
    conditions = db.get_available_conditions()
    
    click.echo("\n📋 Available Experimental Conditions:")
    click.echo("-" * 40)
    
    for key, values in conditions.items():
        if values:
            click.echo(f"\n{key.capitalize()}:")
            click.echo(f"  {values}")


# Entry point
def main():
    """Main entry point for the CLI."""
    friction2d()


if __name__ == '__main__':
    main()
