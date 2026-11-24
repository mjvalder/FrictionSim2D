"""HPC script generation for PBS and SLURM schedulers."""

import copy
import math
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional, Tuple

from .manifest import JobManifest, _mlip_job_fields
from ..core.config import GlobalSettings, apply_framework_overlay
from ..core.mlip import get_mlip, is_mlip
from ..interfaces.jinja import template_environment

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


#: Job-group key: ``(needs_gpu, framework)``. Jobs sharing a key can share one
#: job script; jobs that differ cannot (different resources, or -- for
#: mace-vs-deepmd -- two mutually incompatible LAMMPS builds).
GroupKey = Tuple[bool, Optional[str]]

#: The key a plain classical CPU job carries. A campaign whose only group is
#: this one renders exactly the pre-MLIP filenames and bytes.
CPU_GROUP: GroupKey = (False, None)


def scheduler_base_dir(scheduler: str) -> str:
    """The shell variable holding the directory a job was submitted from."""
    return '$' + _PREFLIGHT_DIALECT[scheduler]['workdir_var']


def _group_key(entry) -> GroupKey:
    """Group key for a :class:`~frictionsim2d.hpc.manifest.JobEntry`."""
    return (bool(entry.needs_gpu), entry.framework or None)


def _path_group_key(path: str) -> GroupKey:
    """Group key for a bare simulation/deck path, read from its ``pot_`` token.

    The string API (the array path every sheet-on-sheet campaign takes) has no
    JobEntry to carry the resource fields, so the run directory's MLIP flavour
    token is the only record of which build a job needs. Classical paths carry
    none and stay in the CPU group.
    """
    fields_ = _mlip_job_fields(path)
    return (bool(fields_.get('needs_gpu', False)), fields_.get('framework') or None)


def _group_suffix(key: GroupKey) -> str:
    """Filename token for a group: ``cpu``, ``gpu_deepmd``, ``gpu_mace``, ..."""
    needs_gpu, framework = key
    parts = ['gpu' if needs_gpu else 'cpu']
    if framework:
        parts.append(framework)
    return '_'.join(parts)


def _ordered_groups(keys) -> List[GroupKey]:
    """Deterministic group order: CPU first, then frameworks alphabetically."""
    return sorted(set(keys), key=lambda k: (k[0], k[1] or ''))


def _group_by(items, key_fn) -> Dict[GroupKey, list]:
    """Bucket ``items`` by ``key_fn``, preserving their order within a bucket."""
    grouped: Dict[GroupKey, list] = {}
    for item in items:
        grouped.setdefault(key_fn(item), []).append(item)
    return grouped


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
        use_tmpdir: Whether to use scratch directory.
        lammps_scripts: Script names to execute.
        max_array_size: Maximum array job size.
        framework: MLIP framework this config renders for ('deepmd' | 'mace'),
            or None for a classical job. Set per job group by
            :meth:`HPCScriptGenerator._config_for_group`; it selects the
            LauncherSpec that supplies the MPI rank count, extra lmp flags and
            environment lines.
    """
    hpc_settings: Any = field(default_factory=_default_hpc_settings)
    job_name: str = "friction2d"
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
        return self.hpc_settings.hpc_host

    @property
    def hpc_home(self) -> Optional[str]:
        return self.hpc_settings.hpc_home

    @property
    def log_dir(self) -> Optional[str]:
        return self.hpc_settings.log_dir

    @property
    def scratch_dir(self) -> Optional[str]:
        return self.hpc_settings.scratch_dir

    @property
    def modules(self) -> List[str]:
        return list(self.hpc_settings.modules or [])

    @property
    def conda_env(self) -> Optional[str]:
        return self.hpc_settings.conda_env or None

    @property
    def preflight(self) -> bool:
        return self.hpc_settings.preflight

    @property
    def mpi_command(self) -> str:
        return self.hpc_settings.mpi_command

    @property
    def use_tmpdir(self) -> bool:
        return self.hpc_settings.use_tmpdir

    @property
    def lammps_scripts(self) -> List[str]:
        return list(self.hpc_settings.lammps_scripts or ['system.in', 'slide.in'])

    @lammps_scripts.setter
    def lammps_scripts(self, scripts: List[str]) -> None:
        self.hpc_settings.lammps_scripts = list(scripts)

    @property
    def max_array_size(self) -> int:
        return self.hpc_settings.max_array_size

    # ------------------------------------------------------------------
    # MLIP / GPU resolution. With no framework and no ``hpc.gpu`` every
    # value below collapses to what the templates hard-coded before MLIP
    # support, which is what keeps CPU output byte-identical.
    # ------------------------------------------------------------------

    @property
    def gpu(self):
        """:class:`~frictionsim2d.core.config.GPUSettings` for this group, or None."""
        return self.hpc_settings.gpu

    @property
    def lmp_binary(self) -> str:
        """LAMMPS binary for this group (``hpc.lmp_binary``, or a framework
        overlay's ``binary``); the bare ``lmp`` on PATH by default."""
        return self.hpc_settings.lmp_binary or 'lmp'

    @property
    def launcher(self):
        """:class:`~frictionsim2d.core.mlip.LauncherSpec` for this group's framework."""
        if self.framework and is_mlip(self.framework):
            return get_mlip(self.framework).launcher
        return None

    @property
    def mpi_ranks(self) -> int:
        """MPI ranks per node (PBS ``mpiprocs``, SLURM ``--ntasks-per-node``).

        A classical job runs one rank per core. An MLIP job does not: its
        LauncherSpec pins the rank count (both current frameworks are
        single-rank -- MACE because symmetrix is OpenMP-threaded, DeePMD
        because one process owns the GPU), and asking for ``ncpus`` ranks
        would contradict that.
        """
        launcher = self.launcher
        if launcher is not None:
            return launcher.mpi_ranks
        if self.gpu is not None:
            return 1
        return self.cpus_per_node

    @property
    def cpus_per_task(self) -> int:
        """Cores each rank owns: 1 for a classical job, the node's share per
        rank for an MLIP/GPU one."""
        return max(1, self.cpus_per_node // max(1, self.mpi_ranks))

    @property
    def omp_threads(self) -> int:
        """``OMP_NUM_THREADS`` the job exports before its ``env_lines``.

        1 unless the framework is OpenMP-threaded (its launcher exports
        ``OMP_NUM_THREADS`` itself, as MACE's ``${OMP_NUM_THREADS:-...}`` does
        -- which would otherwise keep a hard 1). A settings-level env line
        exporting OMP_NUM_THREADS still wins: env_lines render after it.
        """
        launcher = self.launcher
        if launcher is not None and any(
                'OMP_NUM_THREADS' in line for line in launcher.env_lines):
            return self.cpus_per_task
        return 1

    @property
    def env_lines(self) -> List[str]:
        """Shell lines emitted after the environment setup, in order:
        settings/overlay lines first, then the framework LauncherSpec's
        (e.g. DeePMD's ``LAMMPS_PLUGIN_PATH`` export), without duplicates."""
        lines = list(self.hpc_settings.env_lines or [])
        launcher = self.launcher
        if launcher is not None:
            lines.extend(line for line in launcher.env_lines if line not in lines)
        return lines

    @property
    def launch_command(self) -> str:
        """LAMMPS invocation up to (not including) ``-in``.

        ``{mpi_command} {lmp_binary}`` plus the framework's launcher flags --
        exactly main's ``{{ mpi_command }} lmp`` for a classical config.
        """
        launcher = self.launcher
        command = f"{self.mpi_command} {self.lmp_binary}"
        if launcher is not None and launcher.lmp_flags:
            command += f" {launcher.lmp_flags}"
        return command

    @property
    def gres(self) -> Optional[str]:
        """SLURM ``--gres`` value, or None when no GPU is requested."""
        gpu = self.gpu
        if gpu is None:
            return None
        if gpu.gpu_type:
            return f"gpu:{gpu.gpu_type}:{gpu.ngpus}"
        return f"gpu:{gpu.ngpus}"

    def _select_line(self) -> str:
        """PBS ``select`` statement; the GPU request rides on it."""
        line = (f"{self.nodes}:ncpus={self.cpus_per_node}:mem={self.memory_gb}gb:"
                f"mpiprocs={self.mpi_ranks}")
        gpu = self.gpu
        if gpu is not None:
            line += f":ngpus={gpu.ngpus}"
            if gpu.gpu_type:
                line += f":gpu_type={gpu.gpu_type}"
        return line

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering.
        
        Returns:
            Dictionary with all config values and computed fields.
        """
        return {
            'nodes': self.nodes,
            'walltime': f"{self.walltime_hours}:00:00",
            'job_name': self.job_name,
            'queue': self.queue,
            'partition': self.partition,
            'account': self.account,
            'scratch_dir': self.scratch_dir,
            'modules': self.modules,
            'conda_env': self.conda_env,
            'mpi_command': self.mpi_command,
            'use_tmpdir': self.use_tmpdir,
            'select_single': self._select_line(),
            # SLURM: one task per MPI rank, each owning cpus_per_task cores.
            'ntasks_per_node': self.mpi_ranks,
            'cpus_per_task': self.cpus_per_task,
            'mem': f"{self.memory_gb}G",
            'framework': self.framework,
            'lmp_binary': self.lmp_binary,
            'launch_command': self.launch_command,
            'env_lines': self.env_lines,
            'omp_threads': self.omp_threads,
            'gres': self.gres,
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
        self.jinja_env = template_environment()
        # One entry per job script written, for generate_preflight to check.
        self._jobs: List[Dict[str, Any]] = []
        self._log_dir: Optional[str] = None

    def _template(self, name: str):
        """The packaged ``templates/hpc/<name>`` template."""
        return self.jinja_env.get_template(f'hpc/{name}')

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
                    phases: Optional[List[str]] = None,
                    config: Optional[HPCConfig] = None,
                    group: GroupKey = CPU_GROUP) -> None:
        self._jobs.append({'script': script, 'manifest': manifest,
                           'scheduler': scheduler, 'phases': phases or [],
                           'config': config or self.config, 'group': group})

    def _validate_config(self, config: Optional[HPCConfig] = None):
        """Validate required config fields.

        Raises:
            ValueError: If required fields are missing.
        """
        config = config or self.config
        # A conda- or env_lines-based MLIP setup legitimately loads no modules.
        if not config.modules and not config.conda_env and not config.env_lines:
            raise ValueError(
                "HPC modules list is empty and neither conda_env nor env_lines is set.")
        if config.use_tmpdir and not config.scratch_dir:
            raise ValueError("HPC scratch_dir required when use_tmpdir is true.")

    def _config_for_group(self, key: GroupKey, split: bool) -> HPCConfig:
        """The :class:`HPCConfig` a job group renders with.

        * An MLIP group gets ``hpc.frameworks[<framework>]`` merged in through
          :func:`~frictionsim2d.core.config.apply_framework_overlay` -- its own ``lmp``
          binary, modules, conda prefix and env lines -- and its LauncherSpec.
        * When the campaign is actually split (more than one group), a group
          that needs no GPU drops ``hpc.gpu`` so the CPU half of a mixed
          campaign does not idle an accelerator. A single-group campaign
          renders whatever the settings ask for.

        Returns the generator's own config, untouched, for a classical group
        that needs no change -- which keeps CPU output byte-identical.
        """
        needs_gpu, framework = key
        settings = self.config.hpc_settings
        strip_gpu = split and not needs_gpu and settings.gpu is not None
        if not framework and not strip_gpu:
            return self.config
        if framework:
            settings = apply_framework_overlay(settings, framework)
        else:
            settings = copy.deepcopy(settings)
        if strip_gpu:
            settings.gpu = None
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

    def generate_scripts(self, simulation_paths: List[str], output_dir: Path,
                         scheduler: Literal['pbs', 'slurm'] = 'pbs',
                         base_dir: Optional[str] = None,
                         simulation_root: Optional[Path] = None,
                         job_entries: Optional[List[Any]] = None) -> List[Path]:
        """Generate array job scripts with automatic chunking.

        Jobs are first split by ``(needs_gpu, framework)`` -- a GPU/MLIP job
        cannot share an array script with a CPU one, and the two MLIP builds
        cannot share one with each other -- and only then chunked by
        ``max_array_size`` within each group. A campaign that yields a single
        group keeps the plain ``run.pbs``/``manifest_1.txt`` names.

        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            scheduler: Scheduler type ('pbs' or 'slurm').
            base_dir: Base directory variable for HPC (the scheduler's submit
                directory if None).
            simulation_root: Tree the paths are relative to (default: output_dir's parent).
            job_entries: Optional index-parallel JobEntry list carrying the
                resource fields to group on. Without it the group is read from
                each path's ``pot_`` flavour token.

        Returns:
            List of generated script paths.

        Raises:
            ValueError: If the scheduler is unknown or the config is incomplete.
        """
        if scheduler not in _PREFLIGHT_DIALECT:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        base_dir = base_dir or scheduler_base_dir(scheduler)
        output_dir = Path(output_dir)
        simulation_root = Path(simulation_root) if simulation_root else output_dir.parent

        if job_entries and len(job_entries) == len(simulation_paths):
            keyed = list(zip(simulation_paths, (_group_key(e) for e in job_entries)))
        else:
            keyed = [(path, _path_group_key(path)) for path in simulation_paths]
        grouped = {k: [p for p, _ in v]
                   for k, v in _group_by(keyed, lambda pk: pk[1]).items()}
        group_keys = _ordered_groups(grouped)
        split = len(group_keys) > 1
        configs = {key: self._config_for_group(key, split) for key in group_keys}
        if not group_keys:
            self._validate_config()
        for config in configs.values():
            self._validate_config(config)

        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_local_log_dir(output_dir)

        template_name = f"{scheduler}_array.j2"
        script_ext = ".pbs" if scheduler == 'pbs' else ".sh"
        scripts = []
        template = self._template(template_name)

        for key in group_keys:
            paths = grouped[key]
            config = configs[key]
            group_token = f"_{_group_suffix(key)}" if split else ""
            n_sims = len(paths)
            n_scripts = math.ceil(n_sims / config.max_array_size)

            for i in range(n_scripts):
                start_idx = i * config.max_array_size
                end_idx = min((i + 1) * config.max_array_size, n_sims)
                chunk = paths[start_idx:end_idx]

                manifest_name = f"manifest{group_token}_{i+1}.txt"
                (output_dir / manifest_name).write_text(
                    '\n'.join(self._manifest_lines(chunk, simulation_root)))

                context = config.to_dict()
                context.update({
                    'array_size': len(chunk),
                    'manifest_filename': manifest_name,
                    **self._log_context(base_dir),
                })
                job_name = config.job_name + group_token
                if n_scripts > 1:
                    job_name = f"{job_name}_{i+1}"
                context['job_name'] = job_name

                chunk_token = f"_{i+1}" if n_scripts > 1 else ""
                script_name = f"run{group_token}{chunk_token}{script_ext}"
                script_path = output_dir / script_name
                script_path.write_text(template.render(context))
                scripts.append(script_path)
                self._record_job(script_name, manifest_name, scheduler,
                                 config=config, group=key)
                self._log_dir = context['log_dir']

        self._write_master_script(output_dir, scripts, scheduler)
        return scripts

    def _rsync_comments(self, sim_root: Path) -> Tuple[str, str]:
        """The commented-out upload and download ``rsync`` lines for a submit file."""
        target = (f"{self.config.hpc_host or '<HPC_HOST>'}:"
                  f"{self.config.hpc_home or '<HPC_HOME>/'}")
        upload = f'# rsync -rvltoD {sim_root.name} {target}'
        # run*.log too: the settle loop's RAMP_STATUS lines live only in
        # those, and `hpc enrich-provenance -r` needs them on this side.
        download = ('# rsync -avzhP --include="*/" --include="results/***"'
                    ' --include="visuals/***" --include="run*.log"'
                    f' --exclude="*" {target}{sim_root.name} ./')
        return upload, download

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
        submit_cmd = _PREFLIGHT_DIALECT[scheduler]['submit_cmd']
        sim_root = output_dir.parent
        upload, download = self._rsync_comments(sim_root)
        lines = [
            '# Submit all job arrays',
            '',
            '# Optional: upload simulations to HPC',
            upload,
            '',
            '# Optional: download results from HPC',
            download,
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
        """Generate combined system+slide scripts for AFM simulations.

        Each array task runs system.in then slide.in sequentially within the
        same job, eliminating the need for PBS job dependencies. Use when each
        simulation directory has exactly one system script and one slide script.

        Jobs are grouped by ``(needs_gpu, framework)`` and one script is emitted
        per group (``run_cpu.pbs``, ``run_gpu_mace.pbs``, ...): a GPU job cannot
        share a resource request with a CPU one, and the two MLIP LAMMPS builds
        are mutually incompatible binaries. A single-group campaign keeps the
        plain ``run.pbs``/``manifest_combined.txt`` names and bytes.

        Args:
            manifest: JobManifest containing all jobs.
            output_dir: Directory to write scripts.
            scheduler: Scheduler type.
            base_dir: Base directory variable (auto-detected if None).
            simulation_root: Tree the paths are relative to.

        Returns:
            List of generated script paths, one per job group.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_local_log_dir(output_dir)

        base_dir = base_dir or scheduler_base_dir(scheduler)

        system_jobs = manifest.get_system_jobs()
        # One entry per unique simulation directory, preserving order.
        seen: Dict[str, GroupKey] = {}
        for j in system_jobs:
            seen.setdefault(j.simulation_path, _group_key(j))
        grouped = {k: [p for p, _ in v]
                   for k, v in _group_by(seen.items(), lambda pk: pk[1]).items()}
        group_keys = _ordered_groups(grouped)
        split = len(group_keys) > 1

        simulation_root = Path(simulation_root) if simulation_root else output_dir.parent
        template_name = f"{scheduler}_combined.j2"
        template = self._template(template_name)
        script_ext = '.pbs' if scheduler == 'pbs' else '.sh'
        scripts: List[Path] = []

        for key in group_keys:
            sim_dirs = grouped[key]
            config = self._config_for_group(key, split)
            group_token = f"_{_group_suffix(key)}" if split else ""

            manifest_name = f"manifest_combined{group_token}.txt"
            lines = self._manifest_lines(sim_dirs, simulation_root)
            (output_dir / manifest_name).write_text('\n'.join(lines) + '\n',
                                                    encoding='utf-8')

            context = config.to_dict()
            context.update({
                'array_size': len(sim_dirs),
                'manifest_filename': manifest_name,
                'job_name': f"{config.job_name}{group_token}",
                **self._log_context(base_dir),
            })

            script_path = output_dir / f"run{group_token}{script_ext}"
            script_path.write_text(template.render(context))
            scripts.append(script_path)
            self._record_job(script_path.name, manifest_name, scheduler,
                             phases=['system.in', 'slide.in'], config=config, group=key)
            self._log_dir = context['log_dir']

        self._write_combined_submission(output_dir, scripts, scheduler)
        return scripts

    def generate_preflight(self, output_dir: Path) -> List[Path]:
        """Write preflight.sh and its submit wrapper for the jobs generated so far.

        Checks only, no simulation: modules or conda environment load, ``lmp``
        starts, each array range matches its manifest, every deck and the files
        it reads resolve, the pair styles exist in the build, and the log and
        scratch directories are writable.

        Jobs are checked per environment group: an MLIP group's checks run
        under its own binary, modules and env lines (MACE and DeePMD are two
        different LAMMPS builds). The submit wrapper requests the resources of
        the first GPU group when there is one, so the node it lands on can
        actually run the GPU build.

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

        by_group = _group_by(self._jobs, lambda job: job['group'])
        group_keys = _ordered_groups(by_group)
        split = len(group_keys) > 1
        groups = []
        for key in group_keys:
            config = by_group[key][0]['config']
            groups.append({
                'tag': f" ({_group_suffix(key)})" if split else "",
                'jobs': by_group[key],
                'modules': config.modules,
                'conda_env': config.conda_env,
                'framework': config.framework,
                'env_lines': config.env_lines,
                'omp_threads': config.omp_threads,
                'lmp_binary': config.lmp_binary,
            })
        wrapper_config = next((by_group[k][0]['config'] for k in group_keys if k[0]),
                              self.config)

        context = wrapper_config.to_dict()
        context.update({'job_name': self.config.job_name,
                        'jobs': self._jobs, 'groups': groups, 'log_dir': self._log_dir,
                        'directive_log_dir': _directive_log_dir(self._log_dir or ''),
                        **_PREFLIGHT_DIALECT[scheduler]})
        written = []
        for template_name, file_name in (
                ('preflight_sh.j2', 'preflight.sh'),
                (f'{scheduler}_preflight.j2', _PREFLIGHT_DIALECT[scheduler]['preflight_job'])):
            path = output_dir / file_name
            path.write_text(self._template(template_name).render(context),
                            encoding='utf-8')
            written.append(path)
        written[0].chmod(0o755)
        return written

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
        submit_cmd = _PREFLIGHT_DIALECT[scheduler]['submit_cmd']
        sim_root = output_dir.parent
        upload, download = self._rsync_comments(sim_root)
        lines = [
            '#!/bin/bash',
            '# Combined system+slide submission (single job, no dependency needed)',
            '',
            '# Optional: upload to HPC',
            upload,
            '',
            f'cd {sim_root}',
            '',
            *(f'{submit_cmd} hpc/{script.name}' for script in scripts),
            '',
            f'# Monitor with: {"qstat" if scheduler == "pbs" else "squeue"} -u $USER',
            '',
            '# Optional: download results after completion',
            download,
        ]
        submit_path = output_dir / 'submit_jobs.sh'
        submit_path.write_text('\n'.join(lines))
        submit_path.chmod(0o755)
        return submit_path

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

        base_dir = base_dir or scheduler_base_dir(scheduler)

        simulation_root = Path(simulation_root) if simulation_root else output_dir.parent
        system_jobs = manifest.get_system_jobs()
        slide_jobs = manifest.get_slide_jobs()
        result = {'system': [], 'slide': []}
        template_name = f"{scheduler}_array.j2"
        template = self._template(template_name)
        script_ext = '.pbs' if scheduler == 'pbs' else '.sh'

        # Same (needs_gpu, framework) split as the other layouts: each group
        # gets its own system/slide pair, and its slide job depends on its own
        # system job only.
        phase_jobs = {'system': _group_by(system_jobs, _group_key),
                      'slide': _group_by(slide_jobs, _group_key)}
        group_keys = _ordered_groups([*phase_jobs['system'], *phase_jobs['slide']])
        split = len(group_keys) > 1
        pairs = []

        for key in group_keys:
            config = self._config_for_group(key, split)
            group_token = f"_{_group_suffix(key)}" if split else ""
            pair = {}
            for phase in ('system', 'slide'):
                jobs = phase_jobs[phase].get(key)
                if not jobs:
                    continue
                manifest_name = f"manifest_{phase}{group_token}.txt"
                self._write_job_list(output_dir / manifest_name, jobs, simulation_root)
                context = config.to_dict()
                context.update({
                    'array_size': len(jobs),
                    'manifest_filename': manifest_name,
                    **self._log_context(base_dir),
                    'job_name': f"{config.job_name}_{phase}{group_token}",
                })
                script_path = output_dir / f"run_{phase}{group_token}{script_ext}"
                script_path.write_text(template.render(context))
                result[phase].append(script_path)
                pair[phase] = script_path
                self._record_job(script_path.name, manifest_name, scheduler,
                                 config=config, group=key)
                self._log_dir = context['log_dir']
            pairs.append(pair)

        self._write_two_phase_submission(output_dir, pairs, scheduler)
        return result

    def _write_two_phase_submission(self, output_dir: Path,
                                    pairs: List[Dict[str, Path]],
                                    scheduler: Literal['pbs', 'slurm']) -> Path:
        """Write two-phase submission script with job dependencies.

        Args:
            output_dir: Directory to write script.
            pairs: One ``{'system': path, 'slide': path}`` dict per job group
                (either key may be absent).
            scheduler: Scheduler type.

        Returns:
            Path to submission script.
        """
        submit_cmd = _PREFLIGHT_DIALECT[scheduler]['submit_cmd']
        monitor_cmd = 'squeue' if scheduler == 'slurm' else 'qstat'
        sim_root = output_dir.parent
        upload, download = self._rsync_comments(sim_root)
        lines = [
            '#!/bin/bash',
            '# Two-phase submission: system jobs → slide jobs',
            '',
            '# Optional: upload to HPC',
            upload,
            '',
        ]

        for index, pair in enumerate(pairs):
            last = index == len(pairs) - 1
            system, slide = pair.get('system'), pair.get('slide')
            if system is not None:
                if scheduler == 'slurm':
                    submit_system = f'sbatch --parsable hpc/{system.name}'
                    # afterany, not afterok: a system job that died part-way still
                    # wrote the loads before the failure, and slide.in skips the rest.
                    depend = '--dependency=afterany:$SYSTEM_JOB_ID'
                    submit_slide = 'sbatch --parsable'
                else:
                    submit_system = f'{submit_cmd} hpc/{system.name}'
                    depend = '-W depend=afterany:$SYSTEM_JOB_ID'
                    submit_slide = submit_cmd
                lines.extend([
                    'echo "Submitting system initialization jobs..."',
                    f'SYSTEM_JOB_ID=$({submit_system})',
                    'echo "System job ID: $SYSTEM_JOB_ID"',
                    '',
                ])
                if slide is not None:
                    lines.extend([
                        'echo "Submitting slide jobs (waiting for system jobs)..."',
                        f'SLIDE_JOB_ID=$({submit_slide} {depend} hpc/{slide.name})',
                        'echo "Slide job ID: $SLIDE_JOB_ID"',
                        '',
                    ])
                if last:
                    lines.append(f'echo "Monitor with: {monitor_cmd} -u $USER"')
            elif slide is not None:
                lines.extend([
                    'echo "Submitting slide jobs..."',
                    f'{submit_cmd} hpc/{slide.name}',
                    '',
                ])
                if last:
                    lines.append(f'echo "Monitor with: {monitor_cmd} -u $USER"')

        lines.extend([
            '',
            '# Optional: download results after completion',
            download,
        ])

        submit_path = output_dir / 'submit_jobs.sh'
        submit_path.write_text('\n'.join(lines))
        submit_path.chmod(0o755)
        return submit_path
