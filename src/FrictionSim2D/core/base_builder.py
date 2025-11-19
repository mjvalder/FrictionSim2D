"""Base builder class for FrictionSim2D simulations.

This module defines the abstract base class for all simulation builders.
It handles common tasks such as directory creation, configuration loading,
Atomsk wrapper initialization, and Jinja2 template rendering.
"""

import logging
from pathlib import Path
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod

import jinja2
from importlib import resources

from FrictionSim2D.core.config import GlobalSettings, AFMSimulationConfig
from FrictionSim2D.interfaces.atomsk import AtomskWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseBuilder(ABC):
    """Abstract base class for simulation builders.

    Attributes:
        config (AFMSimulationConfig): The validated configuration object.
        settings (GlobalSettings): The global software settings.
        atomsk (AtomskWrapper): The wrapper for the Atomsk binary.
        work_dir (Path): The root directory for the current simulation output.
    """

    def __init__(self, config: AFMSimulationConfig, output_dir: Optional[Path] = None):
        """Initialize the BaseBuilder.

        Args:
            config (AFMSimulationConfig): Validated configuration object.
            output_dir (Optional[Path]): Root directory for output. If None,
                defaults to the current working directory.
        """
        self.config = config
        self.settings = config.settings
        
        # Initialize Atomsk Wrapper
        self.atomsk = AtomskWrapper()

        # Setup Output Directory
        self.work_dir = output_dir if output_dir else Path.cwd()
        
        # Setup Jinja2 Template Environment
        # We point the loader to the 'templates' directory inside the package
        template_dir = resources.files('FrictionSim2D.templates')
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def setup_directories(self, subdirs: list[str]) -> None:
        """Creates the simulation directory structure.

        Args:
            subdirs (list[str]): List of subdirectory names to create 
                                 (e.g., ['visuals', 'results']).
        """
        for subdir in subdirs:
            path = self.work_dir / subdir
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create directory {path}: {e}")
                raise

    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Renders a Jinja2 template with the provided context.

        Args:
            template_name (str): Relative path to the template (e.g., 'afm/slide.lmp').
            context (Dict[str, Any]): Dictionary of variables to pass to the template.

        Returns:
            str: The rendered content.
        """
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(context)
        except jinja2.TemplateError as e:
            logger.error(f"Failed to render template '{template_name}': {e}")
            raise

    @abstractmethod
    def build(self) -> None:
        """Main execution method to build the simulation.
        
        Must be implemented by child classes (e.g., AFMSimulation).
        """
        pass

    @abstractmethod
    def write_inputs(self) -> None:
        """Writes the LAMMPS input scripts.
        
        Must be implemented by child classes.
        """
        pass