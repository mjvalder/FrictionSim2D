"""Jinja2 environment and rendering over the packaged templates."""

from importlib import resources
from typing import Any, Dict, List

from jinja2 import BaseLoader, Environment, TemplateNotFound


class PackageLoader(BaseLoader):
    """Jinja2 loader reading templates from package resources."""

    def __init__(self, package_name: str):
        """Load templates from the package ``package_name``."""
        self._package = resources.files(package_name)

    def get_source(self, environment, template):
        """Load a template from package resources.

        Args:
            environment: The Jinja2 environment. Required by the interface but unused.
            template: Template path relative to the package (e.g., 'afm/slide.lmp').
            
        Returns:
            Tuple of (source, filename, uptodate_function).

        Raises:
            TemplateNotFound: If the template file does not exist.
        """
        try:
            parts = template.split('/')
            current = self._package
            for part in parts:
                current = current.joinpath(part)

            if not current.is_file():
                raise TemplateNotFound(template)

            source = current.read_text(encoding='utf-8')
            return source, template, lambda: True
        except FileNotFoundError as exc:
            raise TemplateNotFound(template) from exc


def template_environment() -> Environment:
    """A Jinja2 environment over the packaged templates (blocks trimmed)."""
    return Environment(
        loader=PackageLoader('frictionsim2d.templates'),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(template: str, context: Dict[str, Any]) -> str:
    """Render a packaged template with ``context``."""
    return template_environment().get_template(template).render(context)


def render_lammps_commands(template: str, context: Dict[str, Any]) -> List[str]:
    """Render a packaged LAMMPS template into its command lines.

    Blank lines and ``#`` comment lines are dropped, the rest stripped.
    """
    rendered = render_template(template, context)
    lines = (line.strip() for line in rendered.strip().split('\n'))
    return [line for line in lines if line and not line.startswith('#')]
