"""What introspection cannot know: labels, units, order and dependencies.

Kept declarative and small on purpose. Introspection supplies structure and
types; this supplies meaning. A field with no entry still renders -- from its
name and its ``Field(description=...)`` -- so an override is an improvement,
never a requirement, and the coverage test reports what is missing rather than
failing.

**Units are Angstrom.** The simulations run in LAMMPS ``metal`` units. Three
example configs commented these fields as nm, which was wrong and has been
corrected; do not take a unit from an example comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .simulation_types import MODES

__all__ = ['FieldOverride', 'SectionOverride', 'overrides_for', 'section_override']


@dataclass(frozen=True)
class FieldOverride:
    """Display metadata for one field."""

    label: Optional[str] = None
    unit: Optional[str] = None
    help: Optional[str] = None
    group: Optional[str] = None
    order: int = 100
    #: Hide unless another field in the same section has a given value.
    show_if: Optional[Tuple[str, object]] = None
    #: Render read-only.
    read_only: bool = False
    #: Show but do not offer editing, with a reason.
    disabled_reason: Optional[str] = None


@dataclass(frozen=True)
class SectionOverride:
    """Display metadata for one section."""

    label: Optional[str] = None
    help: Optional[str] = None
    order: int = 100
    fields: Dict[str, FieldOverride] = field(default_factory=dict)
    #: Section cannot be switched off, with a reason, even though the model
    #: marks it optional.
    required_reason: Optional[str] = None


A = 'Å'

_COMPONENT_FIELDS = {
    'mat': FieldOverride(label='Material', order=1,
                         help='Material identifier, e.g. h-MoS2.'),
    'cif_path': FieldOverride(label='Structure file', order=2,
                              help='CIF describing the crystal structure.'),
    'pot_type': FieldOverride(label='Potential type', order=3),
    'pot_path': FieldOverride(label='Potential file', order=4),
}

#: Section overrides shared by both families.
_COMMON: Dict[str, SectionOverride] = {
    '2D': SectionOverride(
        label='2D sheet', order=1,
        help='The sheet being slid or indented.',
        fields={
            **_COMPONENT_FIELDS,
            'x': FieldOverride(label='Length (x)', unit=A, order=10),
            'y': FieldOverride(label='Width (y)', unit=A, order=11),
            'layers': FieldOverride(
                label='Layers', order=12,
                help=('Number of sheet layers. A list builds one L{n} directory '
                      'per entry -- it multiplies builds, not configs.'),
            ),
            'stack_type': FieldOverride(label='Stacking', order=13),
            'lat_c': FieldOverride(label='Interlayer spacing', unit=A, order=14),
            'materials_list': FieldOverride(
                label='Material list file', order=5,
                help=('Text file of material names. Each becomes a separate '
                      'simulation, so this multiplies the run count.'),
            ),
        },
    ),
    'tip': SectionOverride(
        label='Tip', order=2,
        help='The indenting tip.',
        fields={
            **_COMPONENT_FIELDS,
            'r': FieldOverride(label='Radius', unit=A, order=10,
                               help='Sheet footprint should be at least twice '
                                    'the tip diameter to avoid the tip meeting '
                                    'its own periodic image.'),
            'amorph': FieldOverride(
                label='Structure', order=11,
                help="'a' uses a melt-quenched amorphous block; 'c' is crystalline.",
            ),
            'dspring': FieldOverride(label='Damping override', order=12),
        },
    ),
    'sub': SectionOverride(
        label='Substrate', order=3,
        help='The supporting block beneath the sheet.',
        fields={
            **_COMPONENT_FIELDS,
            'thickness': FieldOverride(label='Thickness', unit=A, order=10),
            'x': FieldOverride(label='Length (x)', unit=A, order=11),
            'y': FieldOverride(label='Width (y)', unit=A, order=12),
            'amorph': FieldOverride(label='Structure', order=13),
            'match': FieldOverride(label='Lattice match', order=14),
        },
    ),
    'general': SectionOverride(
        label='Run parameters', order=4,
        fields={
            'temp': FieldOverride(
                label='Temperature', unit='K', order=1,
                help=('A list runs each temperature as a separate simulation '
                      '-- it appears in the output directory name.'),
            ),
            'force': FieldOverride(
                label='Normal load', unit='nN', order=2,
                help=('Looped inside one LAMMPS script. A list does not '
                      'multiply the run count unless outer_loop names it.'),
            ),
            'pressure': FieldOverride(label='Pressure', unit='GPa', order=3),
            'scan_speed': FieldOverride(label='Scan speed', unit='Å/ps', order=4),
            'scan_angle': FieldOverride(label='Scan angle', unit='deg', order=5),
            'scan_angle_force': FieldOverride(
                label='Scan-angle load selector', unit='nN', order=6,
                help='Which loads the scan-angle list applies to.',
            ),
            'outer_loop': FieldOverride(
                label='Split run phase by', order=7,
                help=('Emits one run script per value of the named parameter '
                      'instead of looping inside a single script. This is what '
                      'multiplies the HPC job count.'),
            ),
            'finite_sheet': FieldOverride(label='Finite sheet', order=8),
            'bond_spring': FieldOverride(label='Bond spring constant', order=9),
            'driving_spring': FieldOverride(label='Driving spring constant', order=10),
            'twist_angle': FieldOverride(label='Twist angle', unit='deg', order=11),
            'hetero_stacking': FieldOverride(label='Heterostructure stacking', order=12),
        },
    ),
    'flake': SectionOverride(
        label='Flake', order=5,
        help='A finite patch placed between tip and sheet.',
        fields={
            **_COMPONENT_FIELDS,
            'edge_length': FieldOverride(label='Edge length', unit=A, order=10),
            'shape': FieldOverride(label='Shape', order=11),
            'x': FieldOverride(label='Supercell x', unit=A, order=12),
            'y': FieldOverride(label='Supercell y', unit=A, order=13),
            'center_x': FieldOverride(label='Centre x', unit=A, order=14),
            'center_y': FieldOverride(label='Centre y', unit=A, order=15),
        },
    ),
    'pes': SectionOverride(
        label='PES scan', order=6,
        help='Static potential-energy-surface scan settings.',
        fields={
            'grid_n': FieldOverride(label='Grid points', order=1),
            'grid_n_refine': FieldOverride(
                label='Refinement grid points', order=2,
                help='Adds a second, finer scan script for a convergence check.',
            ),
        },
    ),
    'indent': SectionOverride(
        label='Indent', order=7,
        help='Hold-and-retract settings for an indent-only run.',
    ),
    'lattice_match': SectionOverride(label='Lattice matching', order=8),
    'lj_override': SectionOverride(
        label='Lennard-Jones overrides', order=9,
        help='Free-form pair overrides; keys are chosen by you.',
    ),
}


def section_override(section_key: str) -> SectionOverride:
    """Display metadata for a section, empty if none is declared."""
    return _COMMON.get(section_key, SectionOverride())


def overrides_for(model: str) -> Dict[str, SectionOverride]:
    """Section overrides for a simulation model, with per-mode adjustments.

    The one real per-model difference: a substrate-free config is only accepted
    by the PES tip scan. Every other model raises ``FreestandingConfigError``
    when ``[sub]`` is absent, so offering a checkbox to remove it would offer a
    guaranteed failure.
    """
    entry = MODES.get(model)
    result = dict(_COMMON)

    if entry is None:
        return result

    if model != 'pes_tip':
        substrate = result.get('sub', SectionOverride())
        result['sub'] = SectionOverride(
            label=substrate.label, help=substrate.help, order=substrate.order,
            fields=substrate.fields,
            required_reason=(
                'Only the PES tip scan supports a substrate-free (freestanding) '
                'configuration; other models abort without one.'
            ),
        )

    return result
