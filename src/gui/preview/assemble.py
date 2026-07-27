"""Building the previewable system from parsed placements and identity.

Applies exactly what the generated script says: read each component, add its
type offset, translate it by its shift, and put the result in the script's
cell. Nothing here decides *where* anything goes.

Imports ovito (for the LAMMPS data reader and the resulting data collection)
but not Qt, so it can be exercised without a display.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .init_script import Assembly, Placement
from .settings_file import Identity

logger = logging.getLogger(__name__)

__all__ = ['AssembledSystem', 'ComponentSlice', 'assemble']

#: Group id used for atoms whose component could not be determined.
UNGROUPED = -1


@dataclass
class ComponentSlice:
    """One component's contribution to the assembled system."""

    name: str
    group_id: int
    first: int
    count: int
    type_ids: List[int] = field(default_factory=list)
    source: Optional[Path] = None

    @property
    def last(self) -> int:
        return self.first + self.count


@dataclass
class AssembledSystem:
    """A previewable system, plus what is needed to explain it."""

    positions: np.ndarray
    type_ids: np.ndarray
    group_ids: np.ndarray
    cell: Optional[Tuple[float, float, float, float, float, float]]
    components: List[ComponentSlice] = field(default_factory=list)
    legend: Dict[int, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def group_names(self) -> Dict[int, str]:
        return {c.group_id: c.name for c in self.components}

    def extent(self) -> Optional[np.ndarray]:
        """Bounding box of the atoms themselves, as ``[[lo], [hi]]``."""
        if self.count == 0:
            return None
        return np.vstack([self.positions.min(axis=0), self.positions.max(axis=0)])

    def touches_boundary(self, tolerance: float = 1.0) -> Dict[str, bool]:
        """Whether each component reaches the periodic x/y walls.

        Relevant because a component that meets the boundary interacts with its
        own periodic image -- the failure the >=2x tip-diameter rule exists to
        avoid.
        """
        if self.cell is None or self.count == 0:
            return {}
        xlo, xhi, ylo, yhi = self.cell[:4]
        result = {}
        for component in self.components:
            block = self.positions[component.first:component.last]
            if block.size == 0:
                continue
            result[component.name] = bool(
                block[:, 0].min() <= xlo + tolerance
                or block[:, 0].max() >= xhi - tolerance
                or block[:, 1].min() <= ylo + tolerance
                or block[:, 1].max() >= yhi - tolerance
            )
        return result


def _read_component(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Positions and type ids from a LAMMPS data file."""
    from ovito.io import import_file  # noqa: PLC0415  (keeps import cost off module load)

    data = import_file(str(path)).compute()
    positions = np.asarray(data.particles['Position'], dtype=float).copy()
    if 'Particle Type' in data.particles:
        types = np.asarray(data.particles['Particle Type'], dtype=int).copy()
    else:
        types = np.ones(len(positions), dtype=int)
    return positions, types


def _component_name(placement: Placement,
                    identity: Identity,
                    global_types: Sequence[int]) -> str:
    """Best available name for a component.

    The script's own ``group`` label is used when present. Sheet-like models
    have none, so the name is recovered from ``system.in.settings`` via the
    types the component contributes.
    """
    if placement.group:
        return placement.group
    for type_id in global_types:
        name = identity.group_of(int(type_id))
        if name:
            return name
    return Path(placement.file).stem


def assemble(assembly: Assembly, identity: Optional[Identity] = None) -> AssembledSystem:
    """Combine an assembly's components into one previewable system.

    Args:
        assembly: Parsed assembly script.
        identity: Parsed ``system.in.settings``. Optional, but without it there
            is no legend and sheet-like components cannot be named.

    Returns:
        An :class:`AssembledSystem`. Components that do not resolve -- run-time
        state files, or a component skipped because its amorphous structure was
        not cached -- are recorded in ``warnings`` and omitted.
    """
    identity = identity or Identity()

    position_blocks: List[np.ndarray] = []
    type_blocks: List[np.ndarray] = []
    group_blocks: List[np.ndarray] = []
    components: List[ComponentSlice] = []
    warnings: List[str] = []
    cursor = 0

    for group_id, placement in enumerate(assembly.placements):
        path = placement.resolve(assembly.simulation_root)
        if path is None:
            reason = ("names run-time state, not a component"
                      if placement.is_runtime_state else "file not found")
            warnings.append(f"skipped '{placement.file}': {reason}")
            continue

        positions, local_types = _read_component(path)
        positions = positions + np.asarray(placement.shift, dtype=float)
        global_types = local_types + placement.type_offset

        name = _component_name(placement, identity, np.unique(global_types))
        components.append(ComponentSlice(
            name=name,
            group_id=group_id,
            first=cursor,
            count=len(positions),
            type_ids=sorted(int(t) for t in np.unique(global_types)),
            source=path,
        ))
        cursor += len(positions)

        position_blocks.append(positions)
        type_blocks.append(global_types)
        group_blocks.append(np.full(len(positions), group_id, dtype=int))

    if position_blocks:
        positions = np.vstack(position_blocks)
        types = np.concatenate(type_blocks)
        groups = np.concatenate(group_blocks)
    else:
        positions = np.zeros((0, 3), dtype=float)
        types = np.zeros(0, dtype=int)
        groups = np.zeros(0, dtype=int)

    present = {int(t) for t in np.unique(types)}
    legend = identity.legend(present_types=present)

    if identity.overlaps:
        for earlier, later, shared in identity.overlaps:
            warnings.append(
                f"overlapping mass ranges: '{later}' overwrites '{earlier}' "
                f"for type(s) {shared}"
            )

    return AssembledSystem(
        positions=positions,
        type_ids=types,
        group_ids=groups,
        cell=assembly.effective_cell(),
        components=components,
        legend=legend,
        warnings=warnings,
    )


def to_ovito(system: AssembledSystem):
    """Convert an assembled system into an ovito pipeline for display.

    Returns:
        An ``ovito.pipeline.Pipeline`` whose source holds the assembled data,
        carrying ``Position``, ``Particle Type`` and ``Group`` properties.
    """
    from ovito.data import DataCollection, ElementType  # noqa: PLC0415
    from ovito.pipeline import Pipeline, StaticSource  # noqa: PLC0415

    data = DataCollection()

    if system.cell is not None:
        xlo, xhi, ylo, yhi, zlo, zhi = system.cell
        matrix = np.array([
            [xhi - xlo, 0.0, 0.0, xlo],
            [0.0, yhi - ylo, 0.0, ylo],
            [0.0, 0.0, zhi - zlo, zlo],
        ], dtype=float)
        # x/y are periodic in every model here; z is the free direction.
        data.create_cell(matrix, pbc=(True, True, False))

    particles = data.create_particles(count=system.count)
    particles.create_property('Position', data=system.positions)
    particles.create_property('Particle Type', data=system.type_ids)
    particles.create_property('Group', data=system.group_ids)

    # A property built from raw ids carries no type definitions, so the element
    # names have to be registered explicitly -- this is what gives the viewport
    # a legend instead of bare numbers.
    types_property = particles['Particle Type']
    for type_id, element in sorted(system.legend.items()):
        types_property.types.append(ElementType(id=int(type_id), name=element))

    return Pipeline(source=StaticSource(data=data))
