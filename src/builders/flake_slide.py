"""D4 flake-driven slide: a finite flake driven across a substrate sheet.

A finite flake (single-material or a hetero bilayer) sits on a substrate sheet
at the minimized flake--sheet gap and is driven laterally by a virtual-atom /
COM spring (no tip, no substrate slab). The substrate sheet's bottom band is
fixed; its interior + the flake are thermostatted; lateral friction is read
from the bottom-sheet reaction force (the many-body ``sw`` observable).

This reuses the D1 slide body (``templates/hetero/slide.lmp``) via a dedicated
``templates/flake/slide.lmp`` in which the driven group is the flake (by type),
the tip/load blocks are dropped, and the substrate-sheet layer groups are the
z-region bands.
"""
import logging
from typing import Dict

from jinja2 import Environment

from ..interfaces.jinja import PackageLoader

logger = logging.getLogger(__name__)

# Same Jinja environment construction as SimulationBase.__init__ and
# hetero_slide.render_hetero_slide: package-resource loader, trimmed blocks.
_ENV = Environment(
    loader=PackageLoader('src.templates'),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_flake_slide(context: Dict) -> str:
    """Render ``flake/slide.lmp`` with the given context.

    Thin module-level helper mirroring :func:`hetero_slide.render_hetero_slide`:
    renders the flake-driven slide template without needing a full builder
    instance (used by the render unit tests).
    """
    return _ENV.get_template("flake/slide.lmp").render(context)
