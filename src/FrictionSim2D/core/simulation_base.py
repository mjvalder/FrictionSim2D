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

from FrictionSim2D.interfaces.atomsk import AtomskWrapper

logger = logging.getLogger(__name__)


class PackageLoader(BaseLoader):
    """Jinja2 loader that works with importlib.resources Traversable objects."""
    
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
            loader=PackageLoader('FrictionSim2D.templates'),
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

    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Renders a Jinja2 template with the provided context.

        Args:
            template_name: Relative path to the template (e.g., 'afm/slide.lmp').
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

    def write_file(self, filename: Union[str, Path], content: str) -> Path:
        """Writes string content to a file in the output directory.
        
        Args:
            filename: Relative path or filename (e.g. 'lammps/system.in').
            content: The string content to write.
            
        Returns:
            The full path to the written file.
        """
        full_path = self.output_dir / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return full_path

    @abstractmethod
    def build(self) -> None:
        """Orchestrates the construction of atomic structures (Tips, Sheets)."""
        pass

    @abstractmethod
    def write_inputs(self) -> None:
        """Generates the LAMMPS input scripts and potential files."""
        pass