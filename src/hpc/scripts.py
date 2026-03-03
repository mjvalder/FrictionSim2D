"""HPC script generation for PBS and SLURM schedulers."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import shutil
import json

from jinja2 import Environment, FileSystemLoader

from src.hpc.manifest import JobManifest

TEMPLATES_DIR = Path(__file__).parent.parent / 'templates' / 'hpc'


@dataclass
class HPCConfig:
    """HPC job submission configuration.
    
    Can be created from GlobalSettings.hpc or instantiated directly.
    
    Attributes:
        scheduler_type: 'pbs' or 'slurm'.
        nodes: Number of compute nodes.
        cpus_per_node: CPUs per node.
        memory_gb: Memory per node in GB.
        walltime_hours: Maximum runtime in hours.
        job_name: Job name prefix.
        queue: PBS queue name.
        partition: SLURM partition name.
        account: Account/project code.
        hpc_home: HPC home directory path.
        log_dir: Directory for job logs.
        scratch_dir: Scratch directory path.
        modules: Environment modules to load.
        mpi_command: MPI launcher command.
        lmp_flags: Additional LAMMPS flags.
        use_tmpdir: Whether to use scratch directory.
        lammps_scripts: Script names to execute.
        max_array_size: Maximum array job size.
    """
    scheduler_type: Literal['pbs', 'slurm'] = 'pbs'
    nodes: int = 1
    cpus_per_node: int = 32
    memory_gb: int = 62
    walltime_hours: int = 20
    job_name: str = "friction2d"
    queue: Optional[str] = None
    partition: Optional[str] = None
    account: Optional[str] = None
    hpc_host: Optional[str] = None
    hpc_home: Optional[str] = None
    scratch_dir: Optional[str] = "$TMPDIR"
    modules: List[str] = field(default_factory=lambda: [
        'tools/prod',
        'LAMMPS/29Aug2024-foss-2023b-kokkos'
    ])
    mpi_command: str = "mpirun"
    lmp_flags: str = "-l none"
    use_tmpdir: bool = True
    lammps_scripts: List[str] = field(default_factory=lambda: ['system.in', 'slide.in'])
    max_array_size: int = 300

    @classmethod
    def from_settings(cls, hpc_settings, job_name: str = "friction2d") -> 'HPCConfig':
        """Create HPCConfig from GlobalSettings.hpc.
        
        Args:
            hpc_settings: HPCSettings instance from GlobalSettings.
            job_name: Job name prefix.
            
        Returns:
            New HPCConfig instance.
        """
        return cls(
            scheduler_type=hpc_settings.scheduler_type,
            nodes=hpc_settings.num_nodes,
            cpus_per_node=hpc_settings.num_cpus,
            memory_gb=hpc_settings.memory_gb,
            walltime_hours=hpc_settings.walltime_hours,
            job_name=job_name,
            queue=hpc_settings.queue or None,
            partition=hpc_settings.partition,
            account=hpc_settings.account or None,
            hpc_host=getattr(hpc_settings, 'hpc_host', None),
            hpc_home=getattr(hpc_settings, 'hpc_home', None),
            scratch_dir=getattr(hpc_settings, 'scratch_dir', None),
            modules=hpc_settings.modules,
            mpi_command=hpc_settings.mpi_command,
            use_tmpdir=hpc_settings.use_tmpdir,
            max_array_size=hpc_settings.max_array_size,
            lammps_scripts=getattr(hpc_settings, 'lammps_scripts', None) or ['system.in', 'slide.in'],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering.
        
        Returns:
            Dictionary with all config values and computed fields.
        """
        return {
            'nodes': self.nodes,
            'cpus_per_node': self.cpus_per_node,
            'memory_gb': self.memory_gb,
            'walltime': f"{self.walltime_hours}:00:00",
            'job_name': self.job_name,
            'queue': self.queue,
            'partition': self.partition,
            'account': self.account,
            'hpc_host': self.hpc_host,
            'hpc_home': self.hpc_home,
            'scratch_dir': self.scratch_dir,
            'modules': self.modules,
            'mpi_command': self.mpi_command,
            'lmp_flags': self.lmp_flags,
            'use_tmpdir': self.use_tmpdir,
            'lammps_scripts': self.lammps_scripts,
            'select_multi': f"1:ncpus={self.cpus_per_node}:mem={self.memory_gb}gb:mpiprocs={self.cpus_per_node}",
            'select_single': f"1:ncpus={self.cpus_per_node}:mem={self.memory_gb}gb",
            'ntasks_per_node': self.cpus_per_node,
            'cpus_per_task': 1,
            'mem': f"{self.memory_gb}G",
        }


class HPCScriptGenerator:
    """HPC batch script generator for LAMMPS simulations.
    
    Generates PBS and SLURM array job scripts with automatic chunking
    for large simulation sets.
    """

    def __init__(self, config: Optional[HPCConfig] = None):
        """Initialize generator.
        
        Args:
            config: HPC configuration (uses defaults if None).
        """
        self.config = config or HPCConfig()
        self.jinja_env = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _validate_config(self):
        """Validate required config fields.
        
        Raises:
            ValueError: If required fields are missing.
        """
        if not self.config.modules:
            raise ValueError("HPC modules list is empty.")
        if self.config.use_tmpdir and not self.config.scratch_dir:
            raise ValueError("HPC scratch_dir required when use_tmpdir is true.")

    def _generate_array_scripts(self, simulation_paths: List[str], output_dir: Path,
                                base_dir: str, scheduler: Literal['pbs', 'slurm']) -> List[Path]:
        """Generate array job scripts with automatic chunking.
        
        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable for HPC.
            scheduler: Scheduler type.
            
        Returns:
            List of generated script paths.
        """
        self._validate_config()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_sims = len(simulation_paths)
        n_scripts = math.ceil(n_sims / self.config.max_array_size)
        template_name = f"{scheduler}_array.j2"
        script_ext = ".pbs" if scheduler == 'pbs' else ".sh"
        scripts = []
        template = self.jinja_env.get_template(template_name)

        for i in range(n_scripts):
            start_idx = i * self.config.max_array_size
            end_idx = min((i + 1) * self.config.max_array_size, n_sims)
            chunk = simulation_paths[start_idx:end_idx]

            manifest_name = f"manifest_{i+1}.txt"
            (output_dir / manifest_name).write_text('\n'.join(chunk))

            context = self.config.to_dict()
            context.update({
                'array_size': len(chunk),
                'manifest_file': f"{base_dir}/{output_dir.name}/{manifest_name}",
                'manifest_filename': manifest_name,
                'base_dir': base_dir,
                'log_dir': f"{base_dir}/logs",
            })
            if n_scripts > 1:
                context['job_name'] = f"{self.config.job_name}_{i+1}"

            script_name = f"run_{i+1}{script_ext}" if n_scripts > 1 else f"run{script_ext}"
            script_path = output_dir / script_name
            script_path.write_text(template.render(context))
            scripts.append(script_path)

        self._write_master_script(output_dir, scripts, scheduler)
        return scripts

    def generate_pbs_scripts(self, simulation_paths: List[str], output_dir: Path,
                            base_dir: str = "$PBS_O_WORKDIR") -> List[Path]:
        """Generate PBS array job scripts.
        
        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable.
            
        Returns:
            List of generated script paths.
        """
        return self._generate_array_scripts(
            simulation_paths, output_dir, base_dir, 'pbs'
        )

    def generate_slurm_scripts(self, simulation_paths: List[str], output_dir: Path,
                                base_dir: str = "$SLURM_SUBMIT_DIR") -> List[Path]:
        """Generate SLURM array job scripts.
        
        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable.
            
        Returns:
            List of generated script paths.
        """
        return self._generate_array_scripts(
            simulation_paths, output_dir, base_dir, 'slurm'
        )

    def _write_master_script(self, output_dir: Path, scripts: List[Path],
                            scheduler: Literal['pbs', 'slurm']) -> Path:
        """Write master submission instructions.
        
        Args:
            output_dir: Directory to write script.
            scripts: List of generated script paths.
            scheduler: Scheduler type.
            
        Returns:
            Path to master script.
        """
        submit_cmd = 'qsub' if scheduler == 'pbs' else 'sbatch'
        sim_root = output_dir.parent
        hpc_host = self.config.hpc_host or "<HPC_HOST>"
        hpc_home = self.config.hpc_home or "<HPC_HOME>/"
        rsync_target = f"{hpc_host}:{hpc_home}"
        lines = [
            '# Submit all job arrays',
            '',
            '# Optional: upload simulations to HPC',
            f'# rsync -rvltoD {sim_root.name} {rsync_target}',
            '',
            '# Optional: download results from HPC',
            f'# rsync -avzhP --include="*/" --include="results/***"'
            f' --include="visuals/***" --exclude="*" {rsync_target}{sim_root.name} ./',
            '',
            f'cd {sim_root}',
            '',
        ]
        lines.extend(f'{submit_cmd} {output_dir.name}/{s.name}' for s in scripts)
        master_path = output_dir / 'submit_all.txt'
        master_path.write_text('\n'.join(lines))
        return master_path

    def generate_scripts(self, simulation_paths: List[str], output_dir: Path,
                        scheduler: Literal['pbs', 'slurm'] = 'pbs',
                        **kwargs) -> List[Path]:
        """Generate HPC scripts for specified scheduler.
        
        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            scheduler: Scheduler type ('pbs' or 'slurm').
            **kwargs: Additional arguments for specific generator.
            
        Returns:
            List of generated script paths.
            
        Raises:
            ValueError: If scheduler is unknown.
        """
        if scheduler == 'pbs':
            return self.generate_pbs_scripts(simulation_paths, output_dir, **kwargs)
        if scheduler == 'slurm':
            return self.generate_slurm_scripts(simulation_paths, output_dir, **kwargs)
        raise ValueError(f"Unknown scheduler: {scheduler}")

    def generate_two_phase_scripts(self, manifest: 'JobManifest', output_dir: Path,
                                    scheduler: Literal['pbs', 'slurm'] = 'pbs',
                                    base_dir: Optional[str] = None) -> Dict[str, List[Path]]:
        """Generate separate system and slide job scripts with dependencies.
        
        Creates two array jobs: system initialization and slide simulations.
        Slide jobs depend on system jobs completing first (for AFM simulations).
        
        Args:
            manifest: JobManifest containing all jobs.
            output_dir: Directory to write scripts.
            scheduler: Scheduler type.
            base_dir: Base directory variable (auto-detected if None).
            
        Returns:
            Dictionary with 'system' and 'slide' script lists.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_dir = base_dir or ('$PBS_O_WORKDIR' if scheduler == 'pbs' else '$SLURM_SUBMIT_DIR')

        system_jobs = manifest.get_system_jobs()
        slide_jobs = manifest.get_slide_jobs()
        result = {'system': [], 'slide': []}
        template_name = f"{scheduler}_array.j2"
        template = self.jinja_env.get_template(template_name)
        script_ext = '.pbs' if scheduler == 'pbs' else '.sh'

        if system_jobs:
            manifest.save_script_list(output_dir / 'manifest_system.txt', system_jobs)
            context = self.config.to_dict()
            context.update({
                'array_size': len(system_jobs),
                'manifest_file': f"{base_dir}/hpc/manifest_system.txt",
                'manifest_filename': 'manifest_system.txt',
                'base_dir': base_dir,
                'log_dir': f"{base_dir}/logs",
                'job_name': f"{self.config.job_name}_system",
            })
            script_path = output_dir / f"run_system{script_ext}"
            script_path.write_text(template.render(context))
            result['system'].append(script_path)

        if slide_jobs:
            manifest.save_script_list(output_dir / 'manifest_slide.txt', slide_jobs)
            context = self.config.to_dict()
            context.update({
                'array_size': len(slide_jobs),
                'manifest_file': f"{base_dir}/hpc/manifest_slide.txt",
                'manifest_filename': 'manifest_slide.txt',
                'base_dir': base_dir,
                'log_dir': f"{base_dir}/logs",
                'job_name': f"{self.config.job_name}_slide",
            })
            script_path = output_dir / f"run_slide{script_ext}"
            script_path.write_text(template.render(context))
            result['slide'].append(script_path)

        self._write_two_phase_submission(output_dir, result, scheduler,
                                            has_system=len(system_jobs) > 0)
        return result

    def _write_two_phase_submission(self, output_dir: Path,
                                    scripts: Dict[str, List[Path]],
                                    scheduler: Literal['pbs', 'slurm'],
                                    has_system: bool) -> Path:
        """Write two-phase submission script with job dependencies.
        
        Args:
            output_dir: Directory to write script.
            scripts: Dictionary with 'system' and 'slide' script lists.
            scheduler: Scheduler type.
            has_system: Whether system jobs exist.
            
        Returns:
            Path to submission script.
        """
        submit_cmd = 'qsub' if scheduler == 'pbs' else 'sbatch'
        sim_root = output_dir.parent
        hpc_host = self.config.hpc_host or "<HPC_HOST>"
        hpc_home = self.config.hpc_home or "<HPC_HOME>/"
        rsync_target = f"{hpc_host}:{hpc_home}"
        lines = [
            '#!/bin/bash',
            '# Two-phase submission: system jobs → slide jobs',
            '',
            '# Optional: upload to HPC',
            f'# rsync -rvltoD {sim_root.name} {rsync_target}',
            '',
        ]

        if has_system:
            if scheduler == 'slurm':
                lines.extend([
                    'echo "Submitting system initialization jobs..."',
                    f'SYSTEM_JOB_ID=$(sbatch --parsable hpc/{scripts["system"][0].name})',
                    'echo "System job ID: $SYSTEM_JOB_ID"',
                    '',
                    'echo "Submitting slide jobs (waiting for system jobs)..."',
                    f'SLIDE_JOB_ID=$(sbatch --parsable --dependency=afterok:$SYSTEM_JOB_ID hpc/{scripts["slide"][0].name})',
                    'echo "Slide job ID: $SLIDE_JOB_ID"',
                    '',
                    'echo "Monitor with: squeue -u $USER"',
                ])
            else:
                lines.extend([
                    'echo "Submitting system initialization jobs..."',
                    f'SYSTEM_JOB_ID=$({submit_cmd} hpc/{scripts["system"][0].name})',
                    'echo "System job ID: $SYSTEM_JOB_ID"',
                    '',
                    'echo "Submitting slide jobs (waiting for system jobs)..."',
                    f'SLIDE_JOB_ID=$({submit_cmd} -W depend=afterok:$SYSTEM_JOB_ID hpc/{scripts["slide"][0].name})',
                    'echo "Slide job ID: $SLIDE_JOB_ID"',
                    '',
                    'echo "Monitor with: qstat -u $USER"',
                ])
        else:
            monitor_cmd = 'squeue' if scheduler == 'slurm' else 'qstat'
            lines.extend([
                'echo "Submitting slide jobs..."',
                f'{submit_cmd} hpc/{scripts["slide"][0].name}',
                '',
                f'echo "Monitor with: {monitor_cmd} -u $USER"',
            ])

        lines.extend([
            '',
            '# Optional: download results after completion',
            f'# rsync -avzhP --include="*/" --include="results/***"'
            f' --include="visuals/***" --exclude="*" {rsync_target}{sim_root.name} ./',
        ])

        submit_path = output_dir / 'submit_jobs.sh'
        submit_path.write_text('\n'.join(lines))
        submit_path.chmod(0o755)
        return submit_path


def create_hpc_package(simulation_dir: Path, output_dir: Path,
                        scheduler: Literal['pbs', 'slurm'] = 'pbs',
                        config: Optional[HPCConfig] = None) -> Path:
    """Package simulations for HPC transfer.
    
    Creates self-contained directory with simulations, scripts, and metadata
    ready for transfer to HPC cluster.
    
    Args:
        simulation_dir: Directory containing FrictionSim2D outputs.
        output_dir: Directory to create package in.
        scheduler: HPC scheduler type.
        config: HPC configuration (uses defaults if None).
        
    Returns:
        Path to created package directory.
        
    Raises:
        ValueError: If no simulations found.
    """
    simulation_dir = Path(simulation_dir)
    output_dir = Path(output_dir)
    package_dir = output_dir / "friction2d_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    simulation_paths = [
        str(d.parent.relative_to(simulation_dir))
        for d in simulation_dir.rglob('lammps') if d.is_dir()
    ]
    if not simulation_paths:
        raise ValueError(f"No simulations found in {simulation_dir}")

    shutil.copytree(simulation_dir, package_dir / 'simulations',
                    ignore=shutil.ignore_patterns('*.lammpstrj'))

    generator = HPCScriptGenerator(config)
    generator.generate_scripts(simulation_paths, package_dir / 'scripts',
                                scheduler=scheduler, base_dir='../simulations')

    info = {'n_simulations': len(simulation_paths), 'scheduler': scheduler,
            'created_from': str(simulation_dir)}
    (package_dir / 'package_info.json').write_text(json.dumps(info, indent=2))

    monitor = 'qstat' if scheduler == 'pbs' else 'squeue'
    readme = f"""# FrictionSim2D HPC Package

{len(simulation_paths)} simulations for {scheduler.upper()}

## Usage
1. Transfer to HPC: `rsync -avz friction2d_package/ <hpc>:~/`
2. Submit jobs: `cd scripts && bash submit_all.txt`
3. Monitor: `{monitor} -u $USER`
4. Download results: `rsync -avz <hpc>:~/friction2d_package/simulations/*/results ./`
"""
    (package_dir / 'README.md').write_text(readme)
    return package_dir
