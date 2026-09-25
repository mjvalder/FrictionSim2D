"""Base class for simulation setup.

This module provides the abstract base class `SimulationBase`, which defines
the common interface and utility methods (template rendering, directory setup)
required by all specific simulation types (AFM, Sheet-on-Sheet, etc.).
"""
import logging
import random
import subprocess
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import shutil
import json
from datetime import datetime
import hashlib

from jinja2 import Environment

from ..interfaces.atomsk import AtomskWrapper
from ..interfaces.jinja import PackageLoader
from ..core.utils.paths import get_material_path, get_potential_path
from ..hpc import HPCScriptGenerator, HPCConfig
from ..hpc.manifest import JobManifest

logger = logging.getLogger(__name__)

POTENTIAL_EXTENSIONS = {
    '.sw', '.tersoff', '.rebo', '.airebo', '.reaxff', '.meam',
    '.extep', '.vashishta', '.adp', '.bop', '.eam', '.edip',
    '.comb', '.comb3', '.morse', '.rebomos', '.mod'
}

#: The target temperature the halt threshold is floored at (K). A cold run is
#: not a cold system: releasing the minimiser's strain energy into 500-odd
#: atoms puts a 1 K-target smoke run at tens of K within a thousand steps, and
#: it is perfectly healthy there. Measured 2026-09-22: 119 K on the AFM smoke,
#: 13-78 K across the hetero and flake slides, every one of them a good run
#: that a bare 10x1 K threshold aborted.
HALT_FLOOR_TEMPERATURE_K = 300.0


def halt_temperature(settings: Any, temp: Optional[float]) -> Optional[float]:
    """The absolute temperature (K) at which a run aborts, or None for no guard.

    ``factor`` x the target, floored at ``factor`` x
    :data:`HALT_FLOOR_TEMPERATURE_K`: the guard exists to catch a blow-up
    (thousands of K), not to police the ordinary heating of a cold run, and
    multiplying a 1 K target by 10 would do exactly that.

    One rule in one place for every template: the guard is off when the factor
    is 0 (its documented off switch) and when the config carries no target
    temperature at all.
    """
    factor = settings.simulation.halt_temperature_factor
    if not factor or not temp:
        return None
    return max(float(temp), HALT_FLOOR_TEMPERATURE_K) * float(factor)


def _infer_provenance_category(file_path: Path) -> str:
    """Infer provenance category from file extension."""
    ext = file_path.suffix.lower()
    if ext == '.cif':
        return 'cif'
    if ext in POTENTIAL_EXTENSIONS:
        return 'potential'
    return 'other'

class SimulationBase(ABC):
    """Abstract base class for simulation setup.

    Provides common infrastructure for directory creation, template rendering,
    and file writing. Concrete simulation classes (AFM, SheetOnSheet) inherit
    from this.

    Attributes:
        config: The validated configuration object for the simulation.
        output_dir: The root directory for simulation output.
        atomsk: Optional Atomsk interface for geometry manipulation.
        jinja_env: Template engine environment.
    """

    #: ``[general] outer_loop`` values whose run phase this builder actually
    #: splits into one ``slide_*.in`` per value. Empty here: a builder that
    #: writes a single run script regardless (the hetero/flake slides, both PES
    #: scans) inherits the default and needs no declaration.
    #:
    #: This is the one place each builder states what it splits: the builders
    #: gate their split on it, and the job-count prediction
    #: (``src.run.expand._run_scripts_per_build``) reads it off the builder class
    #: the config routes to, so the predicted and generated counts cannot drift.
    SPLIT_OUTER_LOOPS: frozenset = frozenset()

    #: The results-file schema this builder emits, one of
    #: :data:`src.core.observables.FAMILIES`. It is what `build_obs_ctx` is
    #: called with (`sim_type=`) and what goes into `provenance.json`.
    FAMILY: str = 'afm'

    def set_model(self, model: str) -> None:
        """Record which `run <model>` produced this build.

        `FAMILY` is a property of the builder class -- the schema its results
        files follow; `model` is the CLI/top-directory axis and only the
        runner knows it. The two can diverge: a sheet-on-sheet config with two
        materials routes `model='sheetonsheet'` to
        `HeteroSheetOnSheetSimulation`, whose `FAMILY` is `'hetero'`.
        `run_simulations` calls this right after construction; `model` falls
        back to `FAMILY` when it has not.
        """
        self._model = model

    @property
    def model(self) -> str:
        return getattr(self, '_model', '') or self.FAMILY

    @staticmethod
    def _git_identity() -> "tuple[str, bool]":
        """``(commit, dirty)`` for a provenance record. Never raises.

        Shared by :meth:`~src.builders.afm.base.AFMSimulation._write_provenance`
        and :func:`~src.builders.sheetonsheet.base.write_sheet_provenance` so
        the two families record build identity the same way.

        ``check=False`` throughout: git being absent, or the tree not being a
        repo, must not abort a build. That case is recorded as
        ``('unknown', False)`` -- a provenance record saying "I could not tell"
        is worth more than a failed build.
        """
        try:
            commit = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                cwd=Path(__file__).parent, timeout=10, check=False,
            ).stdout.strip() or 'unknown'
            dirty = bool(subprocess.run(
                ['git', 'status', '--porcelain'], capture_output=True,
                text=True, cwd=Path(__file__).parent, timeout=10, check=False,
            ).stdout.strip())
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Could not read git identity for provenance (git absent or "
                "not a repo?): %s. Recording git_commit='unknown', "
                "git_dirty=False.", exc)
            return 'unknown', False
        return commit, dirty

    def _run_seed(self, channel: str) -> int:
        """A fresh random seed for one RNG channel, drawn once per build.

        The seed exists to make runs independent samples of the thermostat
        ensemble, not to make a run reproducible: a friction result that
        depends on which seed it drew is a sampling problem, not a provenance
        one. So this is a consistency mechanism, not a correctness one.

        It is drawn here rather than by a template-side ``| random`` for two
        practical reasons: one channel gets one seed no matter how many
        templates read it (``| random`` re-draws per occurrence, so a velocity
        and a thermostat line in the same script would silently disagree), and
        the value is available to the builder afterwards -- for a provenance
        record, or simply to log. Templates that take a seed should be handed
        one so every family behaves the same way.
        """
        if channel not in self._seeds:
            self._seeds[channel] = random.randint(1, 900_000_000)
        return self._seeds[channel]

    def __init__(self, config: Any, output_dir: Union[str, Path],
                    config_path: Optional[Union[str, Path]] = None):
        """Initialize the simulation base.

        Args:
            config: A Pydantic configuration object (specific to the simulation type).
            output_dir: The directory where files will be generated (e.g., 'afm/InO/...').
            config_path: Path to the original config .ini file (for provenance).
        """
        self.config = config
        # Store both the original relative path (for LAMMPS) and absolute path (for file I/O)
        self.relative_run_dir = Path(output_dir)
        self.output_dir = Path(output_dir).resolve()
        self.config_path = Path(config_path).resolve() if config_path else None
        self.base_output_dir: Optional[Path] = None
        self._seeds: Dict[str, int] = {}
        #: Set by :meth:`set_model`; '' (falsy) means "not set yet", so the
        #: `model` property's fallback to `FAMILY` stays well-defined for a
        #: builder built directly (no `run_simulations` in front of it).
        self._model: str = ''
        try:
            self.atomsk: Optional[AtomskWrapper] = AtomskWrapper()
        except RuntimeError:
            self.atomsk = None
            logger.info(
                "Atomsk not found. Continuing with native geometry pipeline; "
                "Atomsk-only operations will be unavailable."
            )

        self.jinja_env = Environment(
            loader=PackageLoader('src.templates'),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _create_directories(self, output_dir: Optional[Path] = None,
                            subdirs: Optional[List[str]] = None) -> None:
        """Creates standard simulation subdirectories.

        Args:
            output_dir: Directory in which to create subdirectories. Defaults to self.output_dir.
            subdirs: Optional list of additional subdirectories to create.
                    Defaults to ['visuals', 'results', 'lammps', 'data'].
        """
        target_dir = output_dir if output_dir is not None else self.output_dir
        default_dirs = ['visuals', 'results', 'lammps', 'data']
        for d in default_dirs + (subdirs or []):
            (target_dir / d).mkdir(parents=True, exist_ok=True)

    def render_template(self, template_name: str,
                        context: Dict[str, Any]) -> str:
        """Render a Jinja2 template with the provided context.

        Args:
            template_name: Relative path to the template.
            context: Dictionary of variables to pass to the template.

        Returns:
            The rendered template string.
        """
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(context)
        except Exception as e:
            logger.exception("Failed to render template '%s': %s", template_name, e)
            raise

    def set_base_output_dir(self, base_output_dir: Union[str, Path]) -> None:
        """Set the base output directory and update relative run paths.

        Args:
            base_output_dir: Root directory that contains all simulations.
        """
        base_output_dir = Path(base_output_dir).resolve()
        self.base_output_dir = base_output_dir
        try:
            self.relative_run_dir = self.output_dir.relative_to(base_output_dir)
        except ValueError:
            self.relative_run_dir = Path(self.output_dir)

    def _run_dir_path(self, *parts: str) -> str:
        """A file inside this run's directory, named as a LAMMPS script must name it.

        The generated HPC scripts run LAMMPS from the simulation ROOT, not from
        the run directory, so every path a script emits has to carry
        ``relative_run_dir``. Standalone (no ``set_base_output_dir``) that
        prefix is the output directory itself, so this is correct either way.
        """
        return str(Path(self.relative_run_dir, *parts))

    def _potentials_prefix(self) -> str:
        """How written ``pair_coeff`` lines name the staged potential files.

        Sim-root-relative for the reason above: a bare
        ``provenance/potentials/...`` would be looked up beside the root rather
        than inside this run's directory, and LAMMPS would abort with "cannot
        open potential file" the moment the job was submitted.
        """
        return self._run_dir_path("provenance", "potentials")

    def write_file(self, filename: Union[str, Path],
                    content: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Write string content to a file in the output directory.

        Args:
            filename: Relative path or filename (e.g., 'lammps/system.in').
            content: The string content to write.
            output_dir: Optional custom output directory. Defaults to self.output_dir.

        Returns:
            The full path to the written file.
        """
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        full_path = target_dir / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')
        return full_path

    def add_to_provenance(self, file_path: Union[str, Path], category: str = 'auto',
                            component: Optional[str] = None) -> Optional[Path]:
        """Add a file to the provenance folder for reproducibility tracking.
        
        All files (CIFs, potentials, configs) are copied into their respective
        provenance subdirectories. This ensures portability when folders are
        transferred to HPC or moved elsewhere.
        
        Also updates the provenance manifest to track component usage.
        
        Args:
            file_path: Path to the file to add
            category: 'cif', 'potential', or 'auto' (detect by extension)
            component: Name of component using this file (e.g., 'tip', 'substrate').
                        Optional but recommended for AiiDA traceability.
            
        Returns:
            Path to the file in provenance folder
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning("Cannot add to provenance, file not found: %s", file_path)
            return None

        prov_dir = self.output_dir / 'provenance'

        if category == 'auto':
            category = _infer_provenance_category(file_path)

        if category == 'cif':
            dest_dir = prov_dir / 'cif'
        elif category == 'potential':
            dest_dir = prov_dir / 'potentials'
        else:
            dest_dir = prov_dir

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_path.name

        if not dest_path.exists():
            shutil.copy2(file_path, dest_path)
            logger.debug("Added to provenance: %s", file_path.name)

        self._update_provenance_manifest(file_path, category, component, dest_path)

        return dest_path

    def _load_or_init_manifest(self) -> Dict[str, Any]:
        """Read ``provenance/manifest.json``, or start a fresh one.

        Shared by :meth:`_update_provenance_manifest` and by the two
        ``_record_twist_manifest`` methods that add a ``twist`` block to the
        same file (``sheetonsheet.hetero.HeteroSheetOnSheetSimulation`` and
        ``sheetonsheet.flake_slide.FlakeSlideSimulation``), which had grown
        their own copy of this read-or-init -- one of the ``duplicate-code``
        blocks the Phase 2c review tabulated.

        The copies had also DIVERGED: both wrote ``{'version': '1.0',
        'files': []}`` where this method writes a ``created`` timestamp too, so
        a manifest first created by the twist path permanently lacked it.
        Harmless in itself (only ``files`` is ever read back) but exactly the
        kind of silent schema drift a third copy invites, and the reason to
        collapse them rather than pragma the duplication away.
        """
        manifest_path = self.output_dir / 'provenance' / 'manifest.json'
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        return {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'files': [],
        }

    def _write_manifest(self, manifest_data: Dict[str, Any]) -> Path:
        """Write ``provenance/manifest.json``, stamping ``last_updated``."""
        manifest_path = self.output_dir / 'provenance' / 'manifest.json'
        manifest_data['last_updated'] = datetime.now().isoformat()
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding='utf-8')
        return manifest_path

    def _update_provenance_manifest(self, original_path: Path, category: str,
                                        component: Optional[str], stored_path: Path) -> None:
        """Update the provenance manifest with file metadata.
        
        Creates/updates a JSON manifest file that tracks which files are used for
        which components. This enables AiiDA to trace potentials to specific
        components (e.g., tip, substrate) via the config.ini file.
        
        Manifest structure:
        {
            "version": "1.0",
            "created": "2025-01-19T...",
            "last_updated": "2025-01-19T...",
            "files": [
                {
                    "filename": "Au.sw",
                    "original_path": "/path/to/repo/potentials/Au.sw",
                    "stored_path": "provenance/potentials/Au.sw",
                    "category": "potential",
                    "component": "tip",
                    "checksum": "sha256...",
                    "added_at": "2025-01-19T..."
                }
            ]
        }
        """
        manifest_data = self._load_or_init_manifest()

        original_path_resolved = str(original_path.resolve())
        checksum = hashlib.sha256(original_path.read_bytes()).hexdigest()

        search_criteria = (
            f for f in manifest_data['files']
            if f['category'] == category
            and (
                f.get('original_path') == original_path_resolved
                or f.get('checksum') == checksum
            )
        )
        existing = next(search_criteria, None)

        def ensure_list(val: Optional[Union[str, List[str]]]) -> List[str]:
            if val is None:
                return []
            if isinstance(val, list):
                return [entry for entry in val if isinstance(entry, str)]
            return [val]

        new_components = ensure_list(component)

        if existing:
            existing_components = ensure_list(existing.get('components'))
            for c in new_components:
                if c and c not in existing_components:
                    existing_components.append(c)
            existing['components'] = existing_components
            existing['original_path'] = original_path_resolved
            existing['stored_path'] = str(stored_path.relative_to(self.output_dir))
            existing['checksum'] = checksum
            existing['added_at'] = datetime.now().isoformat()
        else:
            manifest_data['files'].append({
                'filename': original_path.name,
                'original_path': original_path_resolved,
                'stored_path': str(stored_path.relative_to(self.output_dir)),
                'category': category,
                'components': new_components,
                'checksum': checksum,
                'added_at': datetime.now().isoformat()
            })

        manifest_path = self._write_manifest(manifest_data)
        logger.debug("Updated provenance manifest: %s", manifest_path)

    def _add_component_files_to_provenance(self, component_name: str, config: Any) -> None:
        """Find and add CIF and potential files for a component config to provenance.

        Uses canonical component paths from the validated schema: `cif_path` and
        `pot_path`.

        Args:
            component_name: Name of the component ('tip', 'sub', 'sheet', etc.)
            config: Component config object (TipConfig, SubstrateConfig, SheetConfig, etc.)
        """
        cif = None
        try:
            cif = getattr(config, 'cif_path', None) or (
                get_material_path(config.mat) if hasattr(config, 'mat') and config.mat else None
            )
        except (FileNotFoundError, ValueError, KeyError):
            pass

        if cif:
            try:
                self.add_to_provenance(cif, 'cif', component=component_name)
            except (FileNotFoundError, ValueError, KeyError):
                pass

        pot = None
        try:
            pot_path = getattr(config, 'pot_path', None)
            pot = get_potential_path(pot_path) if pot_path else None
        except (FileNotFoundError, ValueError, KeyError):
            pass

        if pot:
            try:
                self.add_to_provenance(pot, 'potential', component=component_name)
            except (FileNotFoundError, ValueError, KeyError):
                pass

    def _generate_hpc_scripts(self) -> None:
        """Generate HPC job submission scripts for all simulations.

        This method should be called after all simulation files are written.
        It creates a manifest of all jobs and generates appropriate HPC scripts.
        For AFM simulations with system.in files, creates two-phase jobs with dependencies.
        """

        if self.base_output_dir is not None and self.base_output_dir != self.output_dir:
            logger.info(
                "Skipping per-simulation HPC script generation; use base output dir instead."
            )
            return

        hpc_dir = self._get_hpc_output_dir()
        hpc_dir.mkdir(parents=True, exist_ok=True)

        sim_root = self.base_output_dir if self.base_output_dir is not None else self.output_dir
        manifest = JobManifest.from_simulation_directory(
            sim_root,
            name=self._get_hpc_job_name()
        )

        if manifest.n_jobs == 0:
            logger.warning("No simulation jobs found for HPC script generation")
            return

        logger.info("Created manifest with %d jobs", manifest.n_jobs)

        job_name = self._get_hpc_job_name()

        hpc_config = HPCConfig.from_settings(
            self.config.settings.hpc,
            job_name=job_name
        )

        if not hpc_config.modules and not hpc_config.conda_env:
            raise ValueError(
                "HPC modules list is empty; set hpc.modules or hpc.conda_env in settings.")
        if hpc_config.use_tmpdir and not hpc_config.scratch_dir:
            raise ValueError("HPC scratch_dir is required when use_tmpdir is true.")

        generator = HPCScriptGenerator(hpc_config)

        if hpc_config.hpc_home:
            base_dir = hpc_config.hpc_home.rstrip('/') + '/' + sim_root.name
        elif hpc_config.scheduler_type == 'pbs':
            base_dir = '$PBS_O_WORKDIR'
        else:
            base_dir = '$SLURM_SUBMIT_DIR'

        manifest_json_path = hpc_dir / 'manifest.json'
        manifest.save(manifest_json_path)
        logger.info("Saved manifest to %s", manifest_json_path)

        if manifest.has_system_jobs():
            logger.info("Detected system initialization jobs - using two-phase submission")
            scripts = generator.generate_two_phase_scripts(
                manifest=manifest,
                output_dir=hpc_dir,
                scheduler=self.config.settings.hpc.scheduler_type,
                base_dir=base_dir,
                simulation_root=sim_root,
            )
            n_scripts = len(scripts['system']) + len(scripts['slide'])
            logger.info("Generated %d HPC scripts (system + slide) in %s", n_scripts, hpc_dir)
        else:
            logger.info("No system jobs detected - using single-phase submission")
            decks = [f"{j.simulation_path}/lammps/{j.lammps_script or 'slide.in'}"
                     for j in manifest.jobs]
            generator.generate_scripts(decks, hpc_dir,
                                       scheduler=hpc_config.scheduler_type,
                                       base_dir=base_dir, simulation_root=sim_root)
            logger.info("Generated HPC script in %s", hpc_dir)

    def _get_hpc_output_dir(self) -> Path:
        """Get the output directory for HPC scripts.
        
        Override this method if you need custom HPC script location.
        Default is output_dir/hpc for single simulations,
        or base_output_dir/hpc for multi-simulation setups.
        
        Returns:
            Path to HPC scripts directory
        """
        if self.base_output_dir is not None:
            return self.base_output_dir / 'hpc'
        return self.output_dir / 'hpc'

    def _get_hpc_job_name(self) -> str:
        """Get the job name for HPC scripts.
        
        Override this method to customize job names.
        
        Returns:
            Job name string
        """
        material = getattr(self.config, 'sheet', None) or getattr(self.config, '2D', None)
        if material and hasattr(material, 'mat'):
            return f"friction_{material.mat}"
        return "friction2d"
