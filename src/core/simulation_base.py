"""Base class for simulation setup.

This module provides the abstract base class `SimulationBase`, which defines
the common interface and utility methods (template rendering, directory setup)
required by all specific simulation types (AFM, Sheet-on-Sheet, etc.).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union
from importlib import resources

from jinja2 import Environment, BaseLoader, TemplateNotFound
import shutil
from src.interfaces.atomsk import AtomskWrapper

logger = logging.getLogger(__name__)


class PackageLoader(BaseLoader):
    """Jinja2 loader that works with importlib.resources Traversable objects.

    Allows Jinja2 to load templates from package resources instead of the
    filesystem, supporting both installed packages and development mode.
    """
    
    def __init__(self, package_name: str):
        self._package = resources.files(package_name)
    
    def get_source(self, environment, template):
        try:
            parts = template.split('/')
            current = self._package
            for part in parts:
                current = current.joinpath(part)
            
            if not current.is_file():
                raise TemplateNotFound(template)
            
            source = current.read_text(encoding='utf-8')
            return source, template, lambda: True
        except (FileNotFoundError, TypeError):
            raise TemplateNotFound(template)

class SimulationBase(ABC):
    """Abstract base class for simulation setup.

    Provides common infrastructure for directory creation, template rendering,
    and file writing. Concrete simulation classes (AFM, SheetOnSheet) inherit
    from this and implement the build() and write_inputs() methods.

    Attributes:
        config: The validated configuration object for the simulation.
        output_dir: The root directory for simulation output.
        atomsk: Interface for geometry manipulation.
        jinja_env: Template engine environment.
    """

    def __init__(self, config: Any, output_dir: Union[str, Path]):
        """Initialize the simulation base.

        Args:
            config: A Pydantic configuration object (specific to the simulation type).
            output_dir: The directory where files will be generated.
        """
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.atomsk = AtomskWrapper()
        
        self.jinja_env = Environment(
            loader=PackageLoader('src.templates'),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _create_directories(self, subdirs: List[str] = None) -> None:
        """Creates standard simulation subdirectories.

        Args:
            subdirs: Optional list of additional subdirectories to create.
                     Defaults to ['visuals', 'results', 'lammps', 'data', 'build'].
        """
        default_dirs = ['visuals', 'results', 'lammps', 'data', 'build']
        for d in default_dirs + (subdirs or []):
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)

    def render_template(self, template_name: str,
                       context: Dict[str, Any]) -> str:
        """Render a Jinja2 template with the provided context.

        Args:
            template_name: Relative path to the template (e.g.,
                'afm/slide.lmp').
            context: Dictionary of variables to pass to the template.

        Returns:
            The rendered template string.
        """
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(context)
        except Exception as e:
            logger.error(f"Failed to render template '{template_name}': {e}")
            raise

    def write_file(self, filename: Union[str, Path],
                   content: str) -> Path:
        """Write string content to a file in the output directory.

        Args:
            filename: Relative path or filename (e.g., 'lammps/system.in').
            content: The string content to write.

        Returns:
            The full path to the written file.
        """
        full_path = self.output_dir / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return full_path

    def add_to_provenance(self, file_path: Union[str, Path], category: str = 'auto') -> Path:
        """Add a file to the provenance folder for reproducibility tracking.
        
        Call this when using a CIF or potential file during simulation setup.
        
        Args:
            file_path: Path to the file to add
            category: 'cif', 'potential', or 'auto' (detect by extension)
            
        Returns:
            Path to the copied file in provenance folder
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Cannot add to provenance, file not found: {file_path}")
            return None
        
        prov_dir = self.output_dir / 'provenance'
        
        # Auto-detect category
        if category == 'auto':
            ext = file_path.suffix.lower()
            if ext == '.cif':
                category = 'cif'
            elif ext in ('.sw', '.tersoff', '.eam', '.meam', '.rebo', '.airebo'):
                category = 'potential'
            else:
                category = 'other'
        
        # Determine destination
        if category == 'cif':
            dest_dir = prov_dir / 'cif'
        elif category == 'potential':
            dest_dir = prov_dir / 'potentials'
        else:
            dest_dir = prov_dir
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_path.name
        
        # Only copy if not already there
        if not dest_path.exists():
            shutil.copy2(file_path, dest_path)
            logger.debug(f"Added to provenance: {file_path.name}")
        
        return dest_path
    
    def init_provenance(self, 
                        config_path: Union[str, Path] = None,
                        settings_path: Union[str, Path] = None,
                        materials_list_path: Union[str, Path] = None) -> Path:
        """Initialize provenance folder with config files.
        
        Call this once at the start of simulation setup. CIF and potential
        files are added later via add_to_provenance() when they're used.
        
        Args:
            config_path: Path to the config.ini file
            settings_path: Path to settings.yaml (optional)
            materials_list_path: Path to materials list file (optional)
            
        Returns:
            Path to the provenance directory
        """
        prov_dir = self.output_dir / 'provenance'
        prov_dir.mkdir(parents=True, exist_ok=True)
        
        if config_path and Path(config_path).exists():
            shutil.copy2(config_path, prov_dir / Path(config_path).name)
        
        if settings_path and Path(settings_path).exists():
            shutil.copy2(settings_path, prov_dir / 'settings.yaml')
        
        if materials_list_path and Path(materials_list_path).exists():
            shutil.copy2(materials_list_path, prov_dir / Path(materials_list_path).name)
        
        logger.info(f"Initialized provenance folder: {prov_dir}")
        return prov_dir

    @abstractmethod
    def build(self) -> None:
        """Orchestrate the construction of atomic structures.

        Subclasses must implement this to create all necessary structures
        and prepare the simulation.
        """
        pass

    @abstractmethod
    def write_inputs(self) -> None:
        """Generate the LAMMPS input scripts and potential files.

        Subclasses must implement this to create all necessary LAMMPS
        configuration files and scripts.
        """
        pass