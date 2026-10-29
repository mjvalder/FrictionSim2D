"""HPC script generation for PBS and SLURM schedulers."""

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional

from jinja2 import Environment, FileSystemLoader

from .manifest import JobManifest
from ..core.config import GlobalSettings

TEMPLATES_DIR = Path(__file__).parent.parent / 'templates' / 'hpc'

#: What preflight.sh needs to know about the scheduler it is checking: the
#: variable holding the submit directory, how the job script spells its array
#: range, and the wrapper that runs the checks on a compute node.
_PREFLIGHT_DIALECT = {
    'pbs': {
        'submit_cmd': 'qsub',
        'workdir_var': 'PBS_O_WORKDIR',
        'preflight_job': 'preflight.pbs',
        'array_range_sed': r's/^#PBS -J 1-\([0-9][0-9]*\)$/\1/p',
    },
    'slurm': {
        'submit_cmd': 'sbatch',
        'workdir_var': 'SLURM_SUBMIT_DIR',
        'preflight_job': 'preflight.slurm',
        'array_range_sed': r's/^#SBATCH --array=1-\([0-9][0-9]*\)$/\1/p',
    },
}


def _default_hpc_settings():
    """Return default validated HPC settings from GlobalSettings."""
    return GlobalSettings().hpc


def _case_dir(simulation_root: Path, path: str) -> str:
    """The directory a PBS job copies to scratch for ``path`` (a simulation
    directory or a deck inside its ``lammps/``): the nearest ancestor holding
    ``provenance/`` or ``build/``, which is everything the decks read. Falls
    back to the simulation directory itself."""
    sim = PurePosixPath(path)
    if sim.suffix == '.in':
        sim = sim.parent.parent
    for candidate in (sim, *sim.parents):
        if str(candidate) == '.':
            break
        if any((simulation_root / candidate / name).is_dir()
               for name in ('provenance', 'build')):
            return str(candidate)
    return str(sim)


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
    hpc_settings: Any = field(default_factory=_default_hpc_settings)
    job_name: str = "friction2d"
    lmp_flags: str = "-l none"

    @classmethod
    def from_settings(cls, hpc_settings, job_name: str = "friction2d") -> 'HPCConfig':
        """Create HPCConfig from GlobalSettings.hpc.
        
        Args:
            hpc_settings: HPCSettings instance from GlobalSettings.
            job_name: Job name prefix.
            
        Returns:
            New HPCConfig instance.
        """
        return cls(hpc_settings=hpc_settings, job_name=job_name)

    @property
    def scheduler_type(self) -> Literal['pbs', 'slurm']:
        return self.hpc_settings.scheduler_type

    @property
    def nodes(self) -> int:
        return self.hpc_settings.num_nodes

    @property
    def cpus_per_node(self) -> int:
        return self.hpc_settings.num_cpus

    @property
    def memory_gb(self) -> int:
        return self.hpc_settings.memory_gb

    @property
    def walltime_hours(self) -> int:
        return self.hpc_settings.walltime_hours

    @property
    def queue(self) -> Optional[str]:
        return self.hpc_settings.queue or None

    @property
    def partition(self) -> Optional[str]:
        return self.hpc_settings.partition

    @property
    def account(self) -> Optional[str]:
        return self.hpc_settings.account or None

    @property
    def hpc_host(self) -> Optional[str]:
        return getattr(self.hpc_settings, 'hpc_host', None)

    @property
    def hpc_home(self) -> Optional[str]:
        return getattr(self.hpc_settings, 'hpc_home', None)

    @property
    def log_dir(self) -> Optional[str]:
        return getattr(self.hpc_settings, 'log_dir', None)

    @property
    def scratch_dir(self) -> Optional[str]:
        return getattr(self.hpc_settings, 'scratch_dir', None)

    @property
    def modules(self) -> List[str]:
        return list(self.hpc_settings.modules or [])

    @property
    def conda_env(self) -> Optional[str]:
        return getattr(self.hpc_settings, 'conda_env', None) or None

    @property
    def preflight(self) -> bool:
        return bool(getattr(self.hpc_settings, 'preflight', False))

    @property
    def mpi_command(self) -> str:
        return self.hpc_settings.mpi_command

    @property
    def use_tmpdir(self) -> bool:
        return self.hpc_settings.use_tmpdir

    @property
    def lammps_scripts(self) -> List[str]:
        return list(getattr(self.hpc_settings, 'lammps_scripts', None) or ['system.in', 'slide.in'])

    @lammps_scripts.setter
    def lammps_scripts(self, scripts: List[str]) -> None:
        self.hpc_settings.lammps_scripts = list(scripts)

    @property
    def max_array_size(self) -> int:
        return self.hpc_settings.max_array_size

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
            'log_dir': self.log_dir,
            'scratch_dir': self.scratch_dir,
            'modules': self.modules,
            'conda_env': self.conda_env,
            'mpi_command': self.mpi_command,
            'lmp_flags': self.lmp_flags,
            'use_tmpdir': self.use_tmpdir,
            'lammps_scripts': self.lammps_scripts,
            'select_multi': (
                f"{self.nodes}:ncpus={self.cpus_per_node}:mem={self.memory_gb}gb:"
                f"mpiprocs={self.cpus_per_node}"
            ),
            'select_single': (
                f"{self.nodes}:ncpus={self.cpus_per_node}:mem={self.memory_gb}gb:"
                f"mpiprocs={self.cpus_per_node}"
            ),
            'ntasks_per_node': self.cpus_per_node,
            'cpus_per_task': 1,
            'mem': f"{self.memory_gb}G",
        }


def _directive_log_dir(log_dir: str) -> str:
    """``log_dir`` as an ``#SBATCH --output`` path.

    SLURM does not expand shell variables in its directives -- a literal
    ``$SLURM_SUBMIT_DIR/logs/x.txt`` is taken as a directory called
    ``$SLURM_SUBMIT_DIR``, and the job is rejected before it starts. sbatch
    resolves relative output paths against the submit directory anyway, so
    dropping the prefix is exactly equivalent. (PBS expands ``$PBS_O_WORKDIR``
    in ``-o``, which is why the PBS header can use it verbatim.)
    """
    prefix = '$SLURM_SUBMIT_DIR/'
    return log_dir[len(prefix):] if log_dir.startswith(prefix) else log_dir


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
        self.config = config or HPCConfig.from_settings(_default_hpc_settings())
        self.jinja_env = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True
        )
        # One entry per job script written, for generate_preflight to check.
        self._jobs: List[Dict[str, Any]] = []
        self._log_dir: Optional[str] = None

    @staticmethod
    def _manifest_lines(paths: List[str], simulation_root: Path) -> List[str]:
        """Every manifest line is ``<case>\\t<path>``, for PBS and SLURM alike,
        so the job script reads what to copy to scratch instead of deriving it."""
        return [f"{_case_dir(simulation_root, p)}\t{p}" for p in paths]

    def _write_job_list(self, path: Path, jobs: list,
                        simulation_root: Path) -> None:
        decks = [f"{j.simulation_path}/lammps/{j.lammps_script or 'slide.in'}" for j in jobs]
        path.write_text('\n'.join(self._manifest_lines(decks, simulation_root)) + '\n',
                        encoding='utf-8')

    def _record_job(self, script: str, manifest: str, scheduler: str,
                    phases: Optional[List[str]] = None) -> None:
        self._jobs.append({'script': script, 'manifest': manifest,
                           'scheduler': scheduler, 'phases': phases or []})

    def _validate_config(self):
        """Validate required config fields.

        Raises:
            ValueError: If required fields are missing.
        """
        if not self.config.modules and not self.config.conda_env:
            raise ValueError("HPC modules list is empty and no conda_env is set.")
        if self.config.use_tmpdir and not self.config.scratch_dir:
            raise ValueError("HPC scratch_dir required when use_tmpdir is true.")

    def _resolve_log_dir(self, base_dir: str) -> str:
        """Resolve log directory for job scripts.

        Priority:
        1. Explicit log_dir in config.
        2. hpc_home/logs — a single shared directory for all jobs when hpc_home
           is set (avoids per-sim-root log subdirectories).
        3. base_dir/logs — fallback when no hpc_home (e.g. $PBS_O_WORKDIR/logs).
        """
        if self.config.log_dir:
            return self.config.log_dir
        if self.config.hpc_home:
            return self.config.hpc_home.rstrip('/') + '/logs'
        return f"{base_dir}/logs"

    def _log_context(self, base_dir: str) -> Dict[str, str]:
        """The two log-directory forms a template needs: the shell path the job
        body uses, and the scheduler-directive path the header uses."""
        log_dir = self._resolve_log_dir(base_dir)
        return {'log_dir': log_dir, 'directive_log_dir': _directive_log_dir(log_dir)}

    def _ensure_local_log_dir(self, output_dir: Path) -> None:
        """Create the job log directory locally so it exists after transfer.

        Schedulers open the ``-o``/``-e`` log files *before* the job script
        runs, so an in-script ``mkdir`` is too late. The default log path is
        ``<sim_root>/logs`` (relative to the submit dir), and the submit dir is
        the parent of the ``hpc/`` output directory. Creating it here means the
        empty ``logs/`` folder ships with the package via rsync and exists when
        the job starts.
        """
        sim_root = Path(output_dir).parent
        (sim_root / 'logs').mkdir(parents=True, exist_ok=True)
        # If an explicit, submit-dir-relative log_dir was configured, create it
        # too (skip absolute paths and shell-variable paths we cannot resolve).
        log_dir = self.config.log_dir
        if log_dir and not log_dir.startswith(('$', '/')):
            (sim_root / log_dir).mkdir(parents=True, exist_ok=True)

    def _generate_array_scripts(self, simulation_paths: List[str], output_dir: Path,
                                base_dir: str, scheduler: Literal['pbs', 'slurm'],
                                simulation_root: Optional[Path] = None) -> List[Path]:
        """Generate array job scripts with automatic chunking.

        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable for HPC.
            scheduler: Scheduler type.
            simulation_root: Tree the paths are relative to (default: output_dir's parent).

        Returns:
            List of generated script paths.
        """
        self._validate_config()
        output_dir = Path(output_dir)
        simulation_root = Path(simulation_root) if simulation_root else output_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_local_log_dir(output_dir)

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
            (output_dir / manifest_name).write_text(
                '\n'.join(self._manifest_lines(chunk, simulation_root)))

            context = self.config.to_dict()
            context.update({
                'array_size': len(chunk),
                'manifest_file': f"{base_dir}/{output_dir.name}/{manifest_name}",
                'manifest_filename': manifest_name,
                'base_dir': base_dir,
                **self._log_context(base_dir),
            })
            if n_scripts > 1:
                context['job_name'] = f"{self.config.job_name}_{i+1}"

            script_name = f"run_{i+1}{script_ext}" if n_scripts > 1 else f"run{script_ext}"
            script_path = output_dir / script_name
            script_path.write_text(template.render(context))
            scripts.append(script_path)
            self._record_job(script_name, manifest_name, scheduler)
            self._log_dir = context['log_dir']

        self._write_master_script(output_dir, scripts, scheduler)
        return scripts

    def generate_pbs_scripts(self, simulation_paths: List[str], output_dir: Path,
                            base_dir: str = "$PBS_O_WORKDIR",
                            simulation_root: Optional[Path] = None) -> List[Path]:
        """Generate PBS array job scripts.
        
        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable.
            
        Returns:
            List of generated script paths.
        """
        return self._generate_array_scripts(
            simulation_paths, output_dir, base_dir, 'pbs', simulation_root
        )

    def generate_slurm_scripts(self, simulation_paths: List[str], output_dir: Path,
                                base_dir: str = "$SLURM_SUBMIT_DIR",
                                simulation_root: Optional[Path] = None) -> List[Path]:
        """Generate SLURM array job scripts.
        
        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable.
            
        Returns:
            List of generated script paths.
        """
        return self._generate_array_scripts(
            simulation_paths, output_dir, base_dir, 'slurm', simulation_root
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
            # run*.log too: the settle loop's RAMP_STATUS lines live only in
            # those, and `hpc enrich-provenance -r` needs them on this side.
            f'# rsync -avzhP --include="*/" --include="results/***"'
            f' --include="visuals/***" --include="run*.log"'
            f' --exclude="*" {rsync_target}{sim_root.name} ./',
            '',
            f'cd {sim_root}',
            '',
        ]
        lines.extend(f'{submit_cmd} {output_dir.name}/{s.name}' for s in scripts)
        master_path = output_dir / 'submit_all.txt'
        master_path.write_text('\n'.join(lines))
        return master_path

    def generate_combined_scripts(self, manifest: 'JobManifest', output_dir: Path,
                                   scheduler: Literal['pbs', 'slurm'] = 'pbs',
                                   base_dir: Optional[str] = None,
                                   simulation_root: Optional[Path] = None) -> List[Path]:
        """Generate a single combined system+slide script for AFM simulations.

        Each array task runs system.in then slide.in sequentially within the
        same job, eliminating the need for PBS job dependencies. Use when each
        simulation directory has exactly one system script and one slide script.

        Args:
            manifest: JobManifest containing all jobs.
            output_dir: Directory to write scripts.
            scheduler: Scheduler type.
            base_dir: Base directory variable (auto-detected if None).

        Returns:
            List containing the single generated script path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_local_log_dir(output_dir)

        base_dir = base_dir or ('$PBS_O_WORKDIR' if scheduler == 'pbs' else '$SLURM_SUBMIT_DIR')

        system_jobs = manifest.get_system_jobs()
        # One entry per unique simulation directory, preserving order.
        seen: dict = {}
        for j in system_jobs:
            seen.setdefault(j.simulation_path, None)
        sim_dirs = list(seen.keys())

        simulation_root = Path(simulation_root) if simulation_root else output_dir.parent
        manifest_name = 'manifest_combined.txt'
        lines = self._manifest_lines(sim_dirs, simulation_root)
        (output_dir / manifest_name).write_text('\n'.join(lines) + '\n', encoding='utf-8')

        template_name = f"{scheduler}_combined.j2"
        template = self.jinja_env.get_template(template_name)
        script_ext = '.pbs' if scheduler == 'pbs' else '.sh'

        context = self.config.to_dict()
        context.update({
            'array_size': len(sim_dirs),
            'manifest_file': f"{base_dir}/hpc/{manifest_name}",
            'manifest_filename': manifest_name,
            'base_dir': base_dir,
            **self._log_context(base_dir),
        })

        script_path = output_dir / f"run{script_ext}"
        script_path.write_text(template.render(context))
        self._record_job(script_path.name, manifest_name, scheduler,
                         phases=['system.in', 'slide.in'])
        self._log_dir = context['log_dir']

        self._write_combined_submission(output_dir, script_path, scheduler)
        return [script_path]

    def generate_preflight(self, output_dir: Path) -> List[Path]:
        """Write preflight.sh and its submit wrapper for the jobs generated so far.

        Checks only, no simulation: modules or conda environment load, ``lmp``
        starts, each array range matches its manifest, every deck and the files
        it reads resolve, the pair styles exist in the build, and the log and
        scratch directories are writable.

        Raises:
            ValueError: If no job scripts were generated yet, or they mix
                schedulers.
        """
        if not self._jobs:
            raise ValueError("Generate the job scripts before their preflight check.")
        schedulers = {job['scheduler'] for job in self._jobs}
        if len(schedulers) > 1:
            raise ValueError(f"Cannot preflight a mix of schedulers: {sorted(schedulers)}.")
        scheduler = schedulers.pop()
        output_dir = Path(output_dir)
        context = self.config.to_dict()
        context.update({'jobs': self._jobs, 'log_dir': self._log_dir,
                        'directive_log_dir': _directive_log_dir(self._log_dir or ''),
                        **_PREFLIGHT_DIALECT[scheduler]})
        written = []
        for template_name, file_name in (
                ('preflight_sh.j2', 'preflight.sh'),
                (f'{scheduler}_preflight.j2', _PREFLIGHT_DIALECT[scheduler]['preflight_job'])):
            path = output_dir / file_name
            path.write_text(self.jinja_env.get_template(template_name).render(context),
                            encoding='utf-8')
            written.append(path)
        written[0].chmod(0o755)
        return written

    def _write_combined_submission(self, output_dir: Path, script: Path,
                                    scheduler: Literal['pbs', 'slurm']) -> Path:
        """Write submission helper for a combined system+slide script.

        Args:
            output_dir: Directory to write script.
            script: The combined script path.
            scheduler: Scheduler type.

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
            '# Combined system+slide submission (single job, no dependency needed)',
            '',
            '# Optional: upload to HPC',
            f'# rsync -rvltoD {sim_root.name} {rsync_target}',
            '',
            f'cd {sim_root}',
            '',
            f'{submit_cmd} hpc/{script.name}',
            '',
            f'# Monitor with: {"qstat" if scheduler == "pbs" else "squeue"} -u $USER',
            '',
            '# Optional: download results after completion',
            # run*.log too: the settle loop's RAMP_STATUS lines live only in
            # those, and `hpc enrich-provenance -r` needs them on this side.
            f'# rsync -avzhP --include="*/" --include="results/***"'
            f' --include="visuals/***" --include="run*.log"'
            f' --exclude="*" {rsync_target}{sim_root.name} ./',
        ]
        submit_path = output_dir / 'submit_jobs.sh'
        submit_path.write_text('\n'.join(lines))
        submit_path.chmod(0o755)
        return submit_path

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
                                    base_dir: Optional[str] = None,
                                    simulation_root: Optional[Path] = None
                                    ) -> Dict[str, List[Path]]:
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
        self._ensure_local_log_dir(output_dir)

        base_dir = base_dir or ('$PBS_O_WORKDIR' if scheduler == 'pbs' else '$SLURM_SUBMIT_DIR')

        simulation_root = Path(simulation_root) if simulation_root else output_dir.parent
        system_jobs = manifest.get_system_jobs()
        slide_jobs = manifest.get_slide_jobs()
        result = {'system': [], 'slide': []}
        template_name = f"{scheduler}_array.j2"
        template = self.jinja_env.get_template(template_name)
        script_ext = '.pbs' if scheduler == 'pbs' else '.sh'

        if system_jobs:
            self._write_job_list(output_dir / 'manifest_system.txt', system_jobs,
                                 simulation_root)
            context = self.config.to_dict()
            context.update({
                'array_size': len(system_jobs),
                'manifest_file': f"{base_dir}/hpc/manifest_system.txt",
                'manifest_filename': 'manifest_system.txt',
                'base_dir': base_dir,
                **self._log_context(base_dir),
                'job_name': f"{self.config.job_name}_system",
            })
            script_path = output_dir / f"run_system{script_ext}"
            script_path.write_text(template.render(context))
            result['system'].append(script_path)
            self._record_job(script_path.name, 'manifest_system.txt', scheduler)
            self._log_dir = context['log_dir']

        if slide_jobs:
            self._write_job_list(output_dir / 'manifest_slide.txt', slide_jobs,
                                 simulation_root)
            context = self.config.to_dict()
            context.update({
                'array_size': len(slide_jobs),
                'manifest_file': f"{base_dir}/hpc/manifest_slide.txt",
                'manifest_filename': 'manifest_slide.txt',
                'base_dir': base_dir,
                **self._log_context(base_dir),
                'job_name': f"{self.config.job_name}_slide",
            })
            script_path = output_dir / f"run_slide{script_ext}"
            script_path.write_text(template.render(context))
            result['slide'].append(script_path)
            self._record_job(script_path.name, 'manifest_slide.txt', scheduler)
            self._log_dir = context['log_dir']

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
                    # afterany, not afterok: a system job that died part-way still
                    # wrote the loads before the failure, and slide.in skips the rest.
                    'echo "Submitting slide jobs (waiting for system jobs)..."',
                    (
                        f'SLIDE_JOB_ID=$(sbatch --parsable '
                        f'--dependency=afterany:$SYSTEM_JOB_ID '
                        f'hpc/{scripts["slide"][0].name})'
                    ),
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
                    # afterany, not afterok: a system job that died part-way still
                    # wrote the loads before the failure, and slide.in skips the rest.
                    'echo "Submitting slide jobs (waiting for system jobs)..."',
                    (
                        f'SLIDE_JOB_ID=$({submit_cmd} '
                        f'-W depend=afterany:$SYSTEM_JOB_ID '
                        f'hpc/{scripts["slide"][0].name})'
                    ),
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
            # run*.log too: the settle loop's RAMP_STATUS lines live only in
            # those, and `hpc enrich-provenance -r` needs them on this side.
            f'# rsync -avzhP --include="*/" --include="results/***"'
            f' --include="visuals/***" --include="run*.log"'
            f' --exclude="*" {rsync_target}{sim_root.name} ./',
        ])

        submit_path = output_dir / 'submit_jobs.sh'
        submit_path.write_text('\n'.join(lines))
        submit_path.chmod(0o755)
        return submit_path
