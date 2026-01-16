"""HPC script generation for PBS and SLURM schedulers.

This module generates batch job scripts for running LAMMPS simulations
on HPC clusters, supporting both PBS and SLURM schedulers.
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from jinja2 import Environment, BaseLoader


# PBS Array Job Template
PBS_TEMPLATE = """#!/bin/bash
#PBS -l select={{ select }}
#PBS -l walltime={{ walltime }}
#PBS -J 1-{{ array_size }}
#PBS -o {{ log_dir }}/{{ job_name }}_^array_index^_out.txt
#PBS -e {{ log_dir }}/{{ job_name }}_^array_index^_err.txt
#PBS -N {{ job_name }}
{% if queue %}
#PBS -q {{ queue }}
{% endif %}
{% if account %}
#PBS -A {{ account }}
{% endif %}

# Load modules
module purge
{% for mod in modules %}
module load {{ mod }}
{% endfor %}

# Get simulation path from manifest
SIM_PATH=$(sed -n "${PBS_ARRAY_INDEX}p" {{ manifest_file }})

# Handle empty line (shouldn't happen but safety check)
if [ -z "$SIM_PATH" ]; then
    echo "Error: No simulation path for index ${PBS_ARRAY_INDEX}"
    exit 1
fi

{% if use_tmpdir %}
# Copy files to TMPDIR for faster I/O
mkdir -p $TMPDIR/sim
rsync -av --exclude='*.lammpstrj' {{ base_dir }}/${SIM_PATH}/ $TMPDIR/sim/
cd $TMPDIR/sim
{% else %}
cd {{ base_dir }}/${SIM_PATH}
{% endif %}

# Run LAMMPS simulations
{% for script in lammps_scripts %}
{{ mpi_command }} lmp {{ lmp_flags }} -in lammps/{{ script }}
if [ $? -ne 0 ]; then
    echo "Error: LAMMPS failed on {{ script }}"
    exit 1
fi
{% endfor %}

{% if use_tmpdir %}
# Copy results back
cp -r results visuals {{ base_dir }}/${SIM_PATH}/
{% endif %}

echo "Simulation completed successfully: ${SIM_PATH}"
"""

# SLURM Array Job Template
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --nodes={{ nodes }}
#SBATCH --ntasks-per-node={{ ntasks_per_node }}
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH --time={{ walltime }}
#SBATCH --array=1-{{ array_size }}
#SBATCH --output={{ log_dir }}/{{ job_name }}_%a_out.txt
#SBATCH --error={{ log_dir }}/{{ job_name }}_%a_err.txt
#SBATCH --job-name={{ job_name }}
{% if partition %}
#SBATCH --partition={{ partition }}
{% endif %}
{% if account %}
#SBATCH --account={{ account }}
{% endif %}
{% if mem %}
#SBATCH --mem={{ mem }}
{% endif %}

# Load modules
module purge
{% for mod in modules %}
module load {{ mod }}
{% endfor %}

# Get simulation path from manifest
SIM_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" {{ manifest_file }})

# Handle empty line (shouldn't happen but safety check)
if [ -z "$SIM_PATH" ]; then
    echo "Error: No simulation path for index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

{% if use_tmpdir %}
# Copy files to local scratch for faster I/O
SCRATCH=${TMPDIR:-/tmp}/${SLURM_JOB_ID}
mkdir -p $SCRATCH
rsync -av --exclude='*.lammpstrj' {{ base_dir }}/${SIM_PATH}/ $SCRATCH/
cd $SCRATCH
{% else %}
cd {{ base_dir }}/${SIM_PATH}
{% endif %}

# Run LAMMPS simulations
{% for script in lammps_scripts %}
{{ mpi_command }} lmp {{ lmp_flags }} -in lammps/{{ script }}
if [ $? -ne 0 ]; then
    echo "Error: LAMMPS failed on {{ script }}"
    exit 1
fi
{% endfor %}

{% if use_tmpdir %}
# Copy results back
cp -r results visuals {{ base_dir }}/${SIM_PATH}/
{% endif %}

echo "Simulation completed successfully: ${SIM_PATH}"
"""


@dataclass
class HPCConfig:
    """Configuration for HPC job submission."""
    
    # Resource settings
    nodes: int = 1
    cpus_per_node: int = 32
    memory_gb: int = 62
    walltime_hours: int = 20
    
    # Job settings
    job_name: str = "friction2d"
    queue: Optional[str] = None  # PBS queue
    partition: Optional[str] = None  # SLURM partition
    account: Optional[str] = None
    
    # Modules to load
    modules: List[str] = field(default_factory=lambda: [
        'tools/prod',
        'LAMMPS/29Aug2024-foss-2023b-kokkos'
    ])
    
    # Execution settings
    mpi_command: str = "mpirun"
    lmp_flags: str = "-l none"
    use_tmpdir: bool = True
    
    # LAMMPS scripts to run (in order)
    lammps_scripts: List[str] = field(default_factory=lambda: [
        'system.lmp',
        'slide.lmp'
    ])
    
    # Array job settings
    max_array_size: int = 300  # Conservative limit per job
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering."""
        return {
            'nodes': self.nodes,
            'cpus_per_node': self.cpus_per_node,
            'memory_gb': self.memory_gb,
            'walltime': f"{self.walltime_hours}:00:00",
            'job_name': self.job_name,
            'queue': self.queue,
            'partition': self.partition,
            'account': self.account,
            'modules': self.modules,
            'mpi_command': self.mpi_command,
            'lmp_flags': self.lmp_flags,
            'use_tmpdir': self.use_tmpdir,
            'lammps_scripts': self.lammps_scripts,
            # PBS-specific
            'select': f"1:ncpus={self.cpus_per_node}:mem={self.memory_gb}gb:mpiprocs={self.cpus_per_node}",
            # SLURM-specific
            'ntasks_per_node': self.cpus_per_node,
            'cpus_per_task': 1,
            'mem': f"{self.memory_gb}G",
        }


class HPCScriptGenerator:
    """Generates HPC batch scripts for friction simulations.
    
    Supports PBS and SLURM schedulers, with automatic splitting of
    large job sets into multiple array jobs.
    """
    
    def __init__(self, config: Optional[HPCConfig] = None):
        """Initialize the script generator.
        
        Args:
            config: HPC configuration, uses defaults if not provided
        """
        self.config = config or HPCConfig()
        self.jinja_env = Environment(loader=BaseLoader(), 
                                      trim_blocks=True, 
                                      lstrip_blocks=True)
    
    def generate_pbs_scripts(self,
                             simulation_paths: List[str],
                             output_dir: Path,
                             base_dir: str = "$PBS_O_WORKDIR",
                             log_dir: str = "$HOME/logs") -> List[Path]:
        """Generate PBS array job scripts.
        
        Args:
            simulation_paths: List of relative paths to simulation directories
            output_dir: Directory to write scripts to
            base_dir: Base directory for simulations on HPC
            log_dir: Directory for log files
            
        Returns:
            List of paths to generated script files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into chunks if needed
        n_sims = len(simulation_paths)
        n_scripts = math.ceil(n_sims / self.config.max_array_size)
        
        scripts = []
        template = self.jinja_env.from_string(PBS_TEMPLATE)
        
        for i in range(n_scripts):
            start_idx = i * self.config.max_array_size
            end_idx = min((i + 1) * self.config.max_array_size, n_sims)
            chunk = simulation_paths[start_idx:end_idx]
            
            # Write manifest file for this chunk
            manifest_name = f"manifest_{i+1}.txt"
            manifest_path = output_dir / manifest_name
            manifest_path.write_text('\n'.join(chunk))
            
            # Generate script
            context = self.config.to_dict()
            context.update({
                'array_size': len(chunk),
                'manifest_file': manifest_name,
                'base_dir': base_dir,
                'log_dir': log_dir,
            })
            
            if n_scripts > 1:
                context['job_name'] = f"{self.config.job_name}_{i+1}"
            
            script_content = template.render(context)
            script_name = f"run_{i+1}.pbs" if n_scripts > 1 else "run.pbs"
            script_path = output_dir / script_name
            script_path.write_text(script_content)
            scripts.append(script_path)
        
        # Write master submission script
        self._write_master_script(output_dir, scripts, 'pbs')
        
        return scripts
    
    def generate_slurm_scripts(self,
                               simulation_paths: List[str],
                               output_dir: Path,
                               base_dir: str = "$SLURM_SUBMIT_DIR",
                               log_dir: str = "$HOME/logs") -> List[Path]:
        """Generate SLURM array job scripts.
        
        Args:
            simulation_paths: List of relative paths to simulation directories
            output_dir: Directory to write scripts to
            base_dir: Base directory for simulations on HPC
            log_dir: Directory for log files
            
        Returns:
            List of paths to generated script files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into chunks if needed
        n_sims = len(simulation_paths)
        n_scripts = math.ceil(n_sims / self.config.max_array_size)
        
        scripts = []
        template = self.jinja_env.from_string(SLURM_TEMPLATE)
        
        for i in range(n_scripts):
            start_idx = i * self.config.max_array_size
            end_idx = min((i + 1) * self.config.max_array_size, n_sims)
            chunk = simulation_paths[start_idx:end_idx]
            
            # Write manifest file for this chunk
            manifest_name = f"manifest_{i+1}.txt"
            manifest_path = output_dir / manifest_name
            manifest_path.write_text('\n'.join(chunk))
            
            # Generate script
            context = self.config.to_dict()
            context.update({
                'array_size': len(chunk),
                'manifest_file': manifest_name,
                'base_dir': base_dir,
                'log_dir': log_dir,
            })
            
            if n_scripts > 1:
                context['job_name'] = f"{self.config.job_name}_{i+1}"
            
            script_content = template.render(context)
            script_name = f"run_{i+1}.sh" if n_scripts > 1 else "run.sh"
            script_path = output_dir / script_name
            script_path.write_text(script_content)
            scripts.append(script_path)
        
        # Write master submission script
        self._write_master_script(output_dir, scripts, 'slurm')
        
        return scripts
    
    def _write_master_script(self, 
                             output_dir: Path, 
                             scripts: List[Path],
                             scheduler: Literal['pbs', 'slurm']) -> Path:
        """Write a master script to submit all job arrays.
        
        Args:
            output_dir: Directory to write script to
            scripts: List of generated script paths
            scheduler: Scheduler type ('pbs' or 'slurm')
            
        Returns:
            Path to master script
        """
        submit_cmd = 'qsub' if scheduler == 'pbs' else 'sbatch'
        
        lines = ['#!/bin/bash', '', '# Submit all job arrays', '']
        for script in scripts:
            lines.append(f'{submit_cmd} {script.name}')
        lines.append('')
        lines.append('echo "All jobs submitted."')
        
        master_path = output_dir / 'submit_all.sh'
        master_path.write_text('\n'.join(lines))
        master_path.chmod(0o755)
        
        return master_path
    
    def generate_scripts(self,
                        simulation_paths: List[str],
                        output_dir: Path,
                        scheduler: Literal['pbs', 'slurm'] = 'pbs',
                        **kwargs) -> List[Path]:
        """Generate HPC scripts for the specified scheduler.
        
        Args:
            simulation_paths: List of relative paths to simulation directories
            output_dir: Directory to write scripts to
            scheduler: Scheduler type ('pbs' or 'slurm')
            **kwargs: Additional arguments passed to the specific generator
            
        Returns:
            List of paths to generated script files
        """
        if scheduler == 'pbs':
            return self.generate_pbs_scripts(simulation_paths, output_dir, **kwargs)
        elif scheduler == 'slurm':
            return self.generate_slurm_scripts(simulation_paths, output_dir, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}. Use 'pbs' or 'slurm'.")


def create_hpc_package(simulation_dir: Path,
                       output_dir: Path,
                       scheduler: Literal['pbs', 'slurm'] = 'pbs',
                       config: Optional[HPCConfig] = None) -> Path:
    """Create a complete HPC package ready for transfer.
    
    This function creates a self-contained directory with all simulation
    files, HPC scripts, and manifests needed to run on an HPC cluster.
    
    Args:
        simulation_dir: Directory containing FrictionSim2D output
        output_dir: Directory to create the package in
        scheduler: HPC scheduler type
        config: HPC configuration
        
    Returns:
        Path to the created package directory
    """
    import shutil
    
    simulation_dir = Path(simulation_dir)
    output_dir = Path(output_dir)
    
    # Create package directory
    package_dir = output_dir / f"friction2d_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all simulation directories (those containing lammps/ subdirectory)
    simulation_paths = []
    for lammps_dir in simulation_dir.rglob('lammps'):
        if lammps_dir.is_dir():
            # Get relative path from simulation_dir to parent of lammps/
            rel_path = lammps_dir.parent.relative_to(simulation_dir)
            simulation_paths.append(str(rel_path))
    
    if not simulation_paths:
        raise ValueError(f"No simulation directories found in {simulation_dir}")
    
    # Copy simulation files
    sims_dir = package_dir / 'simulations'
    shutil.copytree(simulation_dir, sims_dir, 
                    ignore=shutil.ignore_patterns('*.lammpstrj'))
    
    # Generate HPC scripts
    scripts_dir = package_dir / 'scripts'
    generator = HPCScriptGenerator(config)
    generator.generate_scripts(
        simulation_paths,
        scripts_dir,
        scheduler=scheduler,
        base_dir='../simulations'
    )
    
    # Write package info
    info = {
        'n_simulations': len(simulation_paths),
        'scheduler': scheduler,
        'created_from': str(simulation_dir),
    }
    
    import json
    (package_dir / 'package_info.json').write_text(json.dumps(info, indent=2))
    
    # Write README
    readme = f"""# FrictionSim2D HPC Package

This package contains {len(simulation_paths)} simulations ready for HPC execution.

## Contents
- `simulations/`: All LAMMPS input files
- `scripts/`: HPC submission scripts ({scheduler.upper()})
- `package_info.json`: Package metadata

## Usage
1. Transfer this entire directory to your HPC cluster
2. Navigate to the `scripts/` directory
3. Run `./submit_all.sh` to submit all jobs
4. Monitor job status with `{'qstat' if scheduler == 'pbs' else 'squeue'}`
5. After completion, transfer results back

## Scheduler: {scheduler.upper()}
"""
    (package_dir / 'README.md').write_text(readme)
    
    return package_dir
