"""Base class for all simulation builders.

This module provides the abstract base class `BaseBuilder`, which defines
the common interface and utility methods (template rendering, directory setup)
required by all specific simulation types (AFM, Sheet-on-Sheet, etc.).
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape
from importlib import resources

# Import the configuration model base for type hinting
from FrictionSim2D.core.config import AFMSimulationConfig, GlobalSettings
from FrictionSim2D.interfaces.atomsk import AtomskWrapper

logger = logging.getLogger(__name__)

class BaseBuilder(ABC):
    """Abstract base class for simulation builders.

    Attributes:
        config (AFMSimulationConfig): The validated configuration object.
        output_dir (Path): The root directory for the simulation output.
        atomsk (AtomskWrapper): Interface for geometry manipulation.
        jinja_env (jinja2.Environment): Template engine environment.
    """

    def __init__(self, config: Any, output_dir: Union[str, Path]):
        """Initialize the builder.

        Args:
            config: A Pydantic configuration object (specific to the simulation type).
            output_dir: The directory where files will be generated.
        """
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.atomsk = AtomskWrapper()
        
        # Initialize Jinja2 Environment
        # We point it to the installed 'templates' directory package
        template_path = resources.files('FrictionSim2D.templates')
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_path)),
            autoescape=select_autoescape(),
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
        dirs_to_create = default_dirs + (subdirs or [])
        
        for d in dirs_to_create:
            path = self.output_dir / d
            path.mkdir(parents=True, exist_ok=True)

    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Renders a Jinja2 template with the provided context.

        Args:
            template_name: Relative path to the template (e.g., 'afm/slide.lmp').
            context: Dictionary of variables to pass to the template.

        Returns:
            str: The rendered template string.
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
            Path: The full path to the written file.
        """
        full_path = self.output_dir / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w') as f:
            f.write(content)
            
        return full_path

    @abstractmethod
    def build(self) -> None:
        """Orchestrates the construction of atomic structures (Tips, Sheets)."""
        pass

    @abstractmethod
    def write_inputs(self) -> None:
        """Generates the LAMMPS input scripts and potential files."""
        pass