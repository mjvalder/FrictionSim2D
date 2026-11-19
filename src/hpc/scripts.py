"""HPC script generation for PBS and SLURM schedulers."""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import shutil
import json

from jinja2 import Environment, FileSystemLoader

from .manifest import JobManifest, _mlip_job_fields
from ..core.config import GlobalSettings, resolve_profile
from ..core.mlip import get_mlip, is_mlip

TEMPLATES_DIR = Path(__file__).parent.parent / 'templates' / 'hpc'

#: Job-group key: ``(needs_gpu, framework)``. Jobs sharing a key can share one
#: array script; jobs that differ cannot (different resources, or — for
#: mace-vs-deepmd — outright incompatible LAMMPS builds).
GroupKey = Tuple[bool, Optional[str]]

#: The key a plain classical CPU job carries. A campaign whose only group is
#: this one renders exactly the pre-MLIP filenames and bytes.
CPU_GROUP: GroupKey = (False, None)


def _default_hpc_settings():
    """Return default validated HPC settings from GlobalSettings."""
    return GlobalSettings().hpc


def _group_key(entry) -> GroupKey:
    """Group key for a :class:`~src.hpc.manifest.JobEntry`.

    Tolerates entries predating the MLIP fields (``getattr`` defaults), so a
    manifest loaded from an older campaign groups as pure CPU.
    """
    return (
        bool(getattr(entry, 'needs_gpu', False)),
        getattr(entry, 'framework', None) or None,
    )


def _group_suffix(key: GroupKey) -> str:
    """Filename token for a group: ``cpu``, ``gpu_deepmd``, ``gpu_mace``, ..."""
    needs_gpu, framework = key
    parts = ['gpu' if needs_gpu else 'cpu']
    if framework:
        parts.append(framework)
    return '_'.join(parts)


def _ordered_groups(grouped: Dict[GroupKey, List[str]]) -> List[GroupKey]:
    """Deterministic group order: CPU first, then frameworks alphabetically."""
    return sorted(grouped, key=lambda k: (k[0], k[1] or ''))


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
        framework: MLIP framework this config renders for ('deepmd' | 'mace'),
            or None for a classical CPU job. Set per job group by
            :meth:`HPCScriptGenerator._config_for_group`; it selects the
            LauncherSpec that supplies the MPI rank count and extra lmp flags.
    """
    hpc_settings: Any = field(default_factory=_default_hpc_settings)
    job_name: str = "friction2d"
    lmp_flags: str = "-l none"
    framework: Optional[str] = None

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

    # ------------------------------------------------------------------
    # MLIP / GPU resolution
    # ------------------------------------------------------------------

    @property
    def gpu(self):
        """:class:`~src.core.config.GPUSettings` for this group, or None."""
        return getattr(self.hpc_settings, 'gpu', None)

    @property
    def lmp_binary(self) -> str:
        """LAMMPS binary for this group.

        Defaults to the bare ``lmp`` on PATH — which is exactly what the
        templates hard-coded before MLIP support, so a profile-free config
        renders an unchanged launch line.
        """
        return getattr(self.hpc_settings, 'lmp_binary', None) or 'lmp'

    @property
    def launcher(self):
        """:class:`~src.core.mlip.LauncherSpec` for this group's framework."""
        if self.framework and is_mlip(self.framework):
            return get_mlip(self.framework).launcher
        return None

    @property
    def mpi_ranks(self) -> int:
        """MPI ranks to request (the PBS ``mpiprocs`` token).

        The CPU path asks for one rank per core. A GPU/MLIP job does not: the
        LauncherSpec pins the rank count (both current frameworks are
        single-rank — MACE because symmetrix is OpenMP-threaded rather than
        rank-parallel, DeePMD because one process owns the GPU and extra ranks
        are monotonically slower at ~13 GB VRAM each), and requesting
        ``mpiprocs=ncpus`` there would contradict ``mpirun -np 1``.
        """
        launcher = self.launcher
        if launcher is not None:
            return launcher.mpi_ranks
        if self.gpu is not None:
            return 1
        return self.cpus_per_node

    @property
    def resolved_lmp_flags(self) -> str:
        """Base flags plus this framework's launcher flags."""
        launcher = self.launcher
        extra = launcher.lmp_flags if launcher else ''
        return ' '.join(part for part in (self.lmp_flags, extra) if part)

    @property
    def launch_command(self) -> str:
        """Fully-resolved LAMMPS invocation, minus the ``-in <script>`` part.

        Reproduces the templates' former ``{{ mpi_command }} lmp {{ lmp_flags }}``
        expression character-for-character for a profile-free config — including
        the trailing space when the flag string is empty, because the templates
        append ``" -in ..."`` straight after it.
        """
        return f"{self.mpi_command} {self.lmp_binary} {self.resolved_lmp_flags}"

    @property
    def env_lines(self) -> List[str]:
        """Shell lines to emit after the module loads, in precedence order.

        Settings-level lines first, then whatever the framework's LauncherSpec
        requires (e.g. DeePMD's ``LAMMPS_PLUGIN_PATH`` export). Duplicates are
        dropped so a profile that already exports a line does not export twice.
        """
        lines = list(getattr(self.hpc_settings, 'env_lines', None) or [])
        launcher = self.launcher
        if launcher is not None:
            for line in launcher.env_lines:
                if line not in lines:
                    lines.append(line)
        return lines

    def _select_line(self) -> str:
        """PBS ``select`` statement for this group.

        With no GPU configured this is byte-identical to the pre-MLIP string,
        because ``mpi_ranks`` collapses to ``cpus_per_node``.
        """
        line = (f"1:ncpus={self.cpus_per_node}:mem={self.memory_gb}gb"
                f":mpiprocs={self.mpi_ranks}")
        gpu = self.gpu
        if gpu is not None:
            line += f":ngpus={gpu.ngpus}"
            if gpu.gpu_type:
                line += f":gpu_type={gpu.gpu_type}"
        return line

    @property
    def gres(self) -> Optional[str]:
        """SLURM ``--gres`` value, or None when no GPU is requested.

        SLURM has no ``select`` line, so the GPU request has to travel through
        its own directive. None keeps the CPU rendering unchanged.
        """
        gpu = self.gpu
        if gpu is None:
            return None
        if gpu.gpu_type:
            return f"gpu:{gpu.gpu_type}:{gpu.ngpus}"
        return f"gpu:{gpu.ngpus}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering.

        Returns:
            Dictionary with all config values and computed fields.
        """
        select = self._select_line()
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
            'mpi_command': self.mpi_command,
            'lmp_flags': self.lmp_flags,
            'use_tmpdir': self.use_tmpdir,
            'lammps_scripts': self.lammps_scripts,
            'select_multi': select,
            'select_single': select,
            'ntasks_per_node': self.cpus_per_node,
            'cpus_per_task': 1,
            'mem': f"{self.memory_gb}G",
            'launch_command': self.launch_command,
            'env_lines': self.env_lines,
            'gres': self.gres,
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
        self.config = config or HPCConfig.from_settings(_default_hpc_settings())
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
        # A conda-based MLIP profile legitimately loads NO modules -- the whole
        # environment (lmp, the deepmd plugin, libtorch, CUDA) comes from one
        # conda prefix exported via env_lines. So an empty module list is only
        # an error when env_lines is *also* empty, which means nothing at all
        # was configured.
        if not self.config.modules and not self.config.env_lines:
            raise ValueError(
                "HPC settings configure neither 'modules' nor 'env_lines'; the "
                "generated job would have no way to find LAMMPS. Set one of "
                "them under 'hpc:' (or in the active profile) in settings.yaml."
            )
        if self.config.use_tmpdir and not self.config.scratch_dir:
            raise ValueError("HPC scratch_dir required when use_tmpdir is true.")

    def _config_for_group(self, key: GroupKey, split: bool) -> HPCConfig:
        """Return the :class:`HPCConfig` a given job group must render with.

        Two adjustments on top of the generator's base config:

        * if the group names an MLIP framework and an ``active_profile`` is set,
          that profile's ``frameworks:`` overlay is merged in — this is what
          supplies the framework's own ``lmp`` binary, modules and env lines;
        * if the campaign is *actually being split* (more than one group) and
          this group does not need a GPU, the GPU request is dropped so the CPU
          half of a mixed campaign does not idle an accelerator.

        The GPU strip is deliberately conditional on ``split``: a single-group
        campaign renders whatever the user's profile asked for, so
        ``--profile cx3`` on a wholly classical campaign still honours the
        profile rather than silently discarding half of it.

        Args:
            key: ``(needs_gpu, framework)``.
            split: whether the campaign produced more than one group.

        Returns:
            The base config itself when nothing needs changing (which is what
            keeps single-group CPU output byte-identical), else a copy.
        """
        needs_gpu, framework = key
        settings = self.config.hpc_settings

        active = getattr(settings, 'active_profile', None)
        profiles = getattr(settings, 'profiles', None) or {}
        if framework and active and active in profiles:
            settings = resolve_profile(settings, active, framework=framework)

        if split and not needs_gpu and getattr(settings, 'gpu', None) is not None:
            settings = settings.model_copy(update={'gpu': None})

        if settings is self.config.hpc_settings and not framework:
            return self.config
        return replace(self.config, hpc_settings=settings, framework=framework)

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

    @staticmethod
    def _group_paths(simulation_paths: List[str],
                     job_entries: Optional[List[Any]]) -> Dict[GroupKey, List[str]]:
        """Bucket ``simulation_paths`` by resource group, preserving order.

        ``job_entries`` is index-parallel to ``simulation_paths``. When it is
        absent -- the plain string API used by ``create_hpc_package`` and by
        every slide-only (sheet-on-sheet) campaign, which never reaches the
        manifest-carrying combined/two-phase generators -- the group is
        recovered from the run-directory flavour token instead.

        That fallback matters: without it an MLIP sheet-on-sheet job silently
        rendered as a plain CPU job, i.e. bare ``mpirun lmp`` with no GPU
        request, the wrong binary and no ``LAMMPS_PLUGIN_PATH`` -- which fails
        at runtime with "Unrecognized pair style 'deepmd'". Classical paths
        carry no token, so ``parse_mlip_flavor`` returns None for them and they
        stay in the CPU group exactly as before.
        """
        if not job_entries or len(job_entries) != len(simulation_paths):
            inferred: Dict[GroupKey, List[str]] = {}
            for path in simulation_paths:
                fields = _mlip_job_fields(path)
                key: GroupKey = (bool(fields.get('needs_gpu', False)),
                                 fields.get('framework') or None)
                inferred.setdefault(key, []).append(path)
            return inferred

        grouped: Dict[GroupKey, List[str]] = {}
        for path, entry in zip(simulation_paths, job_entries):
            grouped.setdefault(_group_key(entry), []).append(path)
        return grouped

    def _generate_array_scripts(self, simulation_paths: List[str], output_dir: Path,
                                base_dir: str, scheduler: Literal['pbs', 'slurm'],
                                job_entries: Optional[List[Any]] = None) -> List[Path]:
        """Generate array job scripts with automatic chunking.

        Jobs are first split by ``(needs_gpu, framework)`` — a GPU/MLIP job
        cannot share an array script with a CPU one, and the two MLIP builds
        cannot share one with each other — and only then chunked by
        ``max_array_size`` within each group. A campaign that yields a single
        group keeps the original ``run.pbs``/``manifest_1.txt`` names.

        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable for HPC.
            scheduler: Scheduler type.
            job_entries: Optional index-parallel JobEntry list carrying the
                resource fields to group on. Without it every path is treated
                as a plain CPU job.

        Returns:
            List of generated script paths.
        """
        self._validate_config()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_local_log_dir(output_dir)

        grouped = self._group_paths(simulation_paths, job_entries)
        group_keys = _ordered_groups(grouped)
        split = len(group_keys) > 1

        template_name = f"{scheduler}_array.j2"
        script_ext = ".pbs" if scheduler == 'pbs' else ".sh"
        template = self.jinja_env.get_template(template_name)
        scripts = []

        for key in group_keys:
            paths = grouped[key]
            config = self._config_for_group(key, split)
            group_token = f"_{_group_suffix(key)}" if split else ""
            n_sims = len(paths)
            n_scripts = math.ceil(n_sims / config.max_array_size)

            for i in range(n_scripts):
                start_idx = i * config.max_array_size
                end_idx = min((i + 1) * config.max_array_size, n_sims)
                chunk = paths[start_idx:end_idx]

                manifest_name = f"manifest{group_token}_{i+1}.txt"
                (output_dir / manifest_name).write_text('\n'.join(chunk))

                context = config.to_dict()
                context.update({
                    'array_size': len(chunk),
                    'manifest_file': f"{base_dir}/{output_dir.name}/{manifest_name}",
                    'manifest_filename': manifest_name,
                    'base_dir': base_dir,
                    'log_dir': self._resolve_log_dir(base_dir),
                })
                job_name = config.job_name + group_token
                if n_scripts > 1:
                    job_name = f"{job_name}_{i+1}"
                if job_name != config.job_name:
                    context['job_name'] = job_name

                chunk_token = f"_{i+1}" if n_scripts > 1 else ""
                script_path = output_dir / f"run{group_token}{chunk_token}{script_ext}"
                script_path.write_text(template.render(context))
                scripts.append(script_path)

        self._write_master_script(output_dir, scripts, scheduler)
        return scripts

    def generate_pbs_scripts(self, simulation_paths: List[str], output_dir: Path,
                            base_dir: str = "$PBS_O_WORKDIR",
                            job_entries: Optional[List[Any]] = None) -> List[Path]:
        """Generate PBS array job scripts.

        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable.
            job_entries: Optional index-parallel JobEntry list for GPU/framework
                splitting.

        Returns:
            List of generated script paths.
        """
        return self._generate_array_scripts(
            simulation_paths, output_dir, base_dir, 'pbs', job_entries=job_entries
        )

    def generate_slurm_scripts(self, simulation_paths: List[str], output_dir: Path,
                                base_dir: str = "$SLURM_SUBMIT_DIR",
                                job_entries: Optional[List[Any]] = None) -> List[Path]:
        """Generate SLURM array job scripts.

        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable.
            job_entries: Optional index-parallel JobEntry list for GPU/framework
                splitting.

        Returns:
            List of generated script paths.
        """
        return self._generate_array_scripts(
            simulation_paths, output_dir, base_dir, 'slurm', job_entries=job_entries
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

    def generate_combined_scripts(self, manifest: 'JobManifest', output_dir: Path,
                                   scheduler: Literal['pbs', 'slurm'] = 'pbs',
                                   base_dir: Optional[str] = None) -> List[Path]:
        """Generate a single combined system+slide script for AFM simulations.

        Each array task runs system.in then slide.in sequentially within the
        same job, eliminating the need for PBS job dependencies. Use when each
        simulation directory has exactly one system script and one slide script.

        Jobs are grouped by ``(needs_gpu, framework)`` and one script is emitted
        per group (``run_cpu.pbs``, ``run_gpu_deepmd.pbs``, ...), because a GPU
        job cannot share a resource request with a CPU one and the two MLIP
        LAMMPS builds are mutually incompatible binaries. A campaign with a
        single group keeps the original ``run.pbs``/``manifest_combined.txt``
        names and bytes.

        Args:
            manifest: JobManifest containing all jobs.
            output_dir: Directory to write scripts.
            scheduler: Scheduler type.
            base_dir: Base directory variable (auto-detected if None).

        Returns:
            List of generated script paths, one per job group.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_local_log_dir(output_dir)

        base_dir = base_dir or ('$PBS_O_WORKDIR' if scheduler == 'pbs' else '$SLURM_SUBMIT_DIR')

        system_jobs = manifest.get_system_jobs()
        # One entry per unique simulation directory, preserving order.
        seen: dict = {}
        for j in system_jobs:
            seen.setdefault(j.simulation_path, _group_key(j))

        grouped: Dict[GroupKey, List[str]] = {}
        for sim_path, key in seen.items():
            grouped.setdefault(key, []).append(sim_path)
        group_keys = _ordered_groups(grouped)
        split = len(group_keys) > 1

        template_name = f"{scheduler}_combined.j2"
        template = self.jinja_env.get_template(template_name)
        script_ext = '.pbs' if scheduler == 'pbs' else '.sh'
        scripts: List[Path] = []

        for key in group_keys:
            sim_dirs = grouped[key]
            config = self._config_for_group(key, split)
            group_token = f"_{_group_suffix(key)}" if split else ""

            manifest_name = f"manifest_combined{group_token}.txt"
            (output_dir / manifest_name).write_text(
                '\n'.join(sim_dirs) + '\n', encoding='utf-8')

            context = config.to_dict()
            context.update({
                'array_size': len(sim_dirs),
                'manifest_file': f"{base_dir}/hpc/{manifest_name}",
                'manifest_filename': manifest_name,
                'base_dir': base_dir,
                'log_dir': self._resolve_log_dir(base_dir),
            })
            if group_token:
                context['job_name'] = f"{config.job_name}{group_token}"

            script_path = output_dir / f"run{group_token}{script_ext}"
            script_path.write_text(template.render(context))
            scripts.append(script_path)

        self._write_combined_submission(output_dir, scripts, scheduler)
        return scripts

    def _write_combined_submission(self, output_dir: Path, scripts: List[Path],
                                    scheduler: Literal['pbs', 'slurm']) -> Path:
        """Write submission helper for the combined system+slide script(s).

        Args:
            output_dir: Directory to write script.
            scripts: The combined script paths, one per job group.
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
            *(f'{submit_cmd} hpc/{script.name}' for script in scripts),
            '',
            f'# Monitor with: {"qstat" if scheduler == "pbs" else "squeue"} -u $USER',
            '',
            '# Optional: download results after completion',
            f'# rsync -avzhP --include="*/" --include="results/***"'
            f' --include="visuals/***" --exclude="*" {rsync_target}{sim_root.name} ./',
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
        self._ensure_local_log_dir(output_dir)

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
                'log_dir': self._resolve_log_dir(base_dir),
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
                'log_dir': self._resolve_log_dir(base_dir),
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

    # base_dir '../simulations' means logs resolve under the simulations tree;
    # create it so the scheduler can open its log files at job start.
    (package_dir / 'simulations' / 'logs').mkdir(parents=True, exist_ok=True)

    generator = HPCScriptGenerator(config)
    generator.generate_scripts(simulation_paths, package_dir / 'hpc',
                                scheduler=scheduler, base_dir='../simulations')

    info = {'n_simulations': len(simulation_paths), 'scheduler': scheduler,
            'created_from': str(simulation_dir)}
    (package_dir / 'package_info.json').write_text(json.dumps(info, indent=2))

    monitor = 'qstat' if scheduler == 'pbs' else 'squeue'
    readme = f"""# FrictionSim2D HPC Package

{len(simulation_paths)} simulations for {scheduler.upper()}

## Usage
1. Transfer to HPC: `rsync -avz friction2d_package/ <hpc>:~/`
2. Submit jobs: `cd hpc && bash submit_all.txt`
3. Monitor: `{monitor} -u $USER`
4. Download results: `rsync -avz <hpc>:~/friction2d_package/simulations/*/results ./`
"""
    (package_dir / 'README.md').write_text(readme)
    return package_dir
