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
from typing import Dict, Optional, Tuple

from .simulation_types import MODES

__all__ = [
    'FieldOverride', 'SectionOverride', 'overrides_for', 'section_override',
    'ENABLED', 'field_override', 'CAPTION_BUDGET',
]

#: Sentinel for ``enable_if``: the other field just has to hold a truthy value.
ENABLED = object()

#: Caption length budget, in characters. Enforced by tests/test_gui_schema.py.
CAPTION_BUDGET = 70


@dataclass(frozen=True)
class FieldOverride:
    """Display metadata for one field."""

    label: Optional[str] = None
    unit: Optional[str] = None
    help: Optional[str] = None
    group: Optional[str] = None
    order: int = 100
    #: Disable unless another field in the same section is set. Given as
    #: ``(field key, required value)``; ``ENABLED`` means "any truthy value".
    #: A greyed control says "this does nothing yet" far more clearly than a
    #: value that is quietly ignored.
    enable_if: Optional[Tuple[str, object]] = None
    #: Reason shown when the control is disabled by ``enable_if``.
    enable_hint: Optional[str] = None
    #: Render read-only.
    read_only: bool = False
    #: Restrict an enum's choices below what the model allows.
    choices: Optional[Tuple[object, ...]] = None
    #: Long-form explanation. Tooltip only -- never rendered as a caption,
    #: because a caption under every control is what made the form unreadable.
    detail: Optional[str] = None
    #: Slug requesting a purpose-built editor; see widgets.fields.EDITOR_REGISTRY.
    editor: Optional[str] = None


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
    #: Long-form explanation. Tooltip only, same rationale as FieldOverride.detail.
    detail: Optional[str] = None


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
            'layers': FieldOverride(label='Layers', order=12),
            'stack_type': FieldOverride(
                label='Stacking', order=13,
                choices=('AA', 'AB'),
                help='AA: layers aligned. AB: alternate layers shifted.',
                detail=('Only affects runs with more than one layer. The builders test '
                        'for AA and treat every other value as AB.'),
            ),
            'lat_c': FieldOverride(label='Interlayer spacing', unit=A, order=14),
            'materials_list': FieldOverride(
                label='Materials', order=5, editor='material_list',
                detail=('Each material becomes a separate simulation. The structure '
                        'and potential directories decide which are available: a '
                        'material needs a file in both.'),
            ),
        },
    ),
    'tip': SectionOverride(
        label='Tip', order=2,
        help='The indenting tip.',
        fields={
            **_COMPONENT_FIELDS,
            'r': FieldOverride(
                label='Radius', unit=A, order=10,
                help='Tip radius.',
                detail=('Sheet footprint should be at least twice the tip diameter to '
                        'avoid the tip meeting its own periodic image.'),
            ),
            'amorph': FieldOverride(
                label='Structure', order=11,
                help="'a' melt-quenched amorphous, 'c' crystalline.",
            ),
            'dspring': FieldOverride(
                label='Damping override', order=12,
                help='Raw viscous damping; overrides Damping ratio when set.',
            ),
        },
    ),
    'sub': SectionOverride(
        label='Substrate', order=3,
        help='The supporting block beneath the sheet.',
        fields={
            **_COMPONENT_FIELDS,
            'thickness': FieldOverride(label='Thickness', unit=A, order=10),
            'x': FieldOverride(
                label='Length (x)', unit=A, order=11,
                help='Finite-sheet mode only.',
                detail='Otherwise the substrate takes the sheet footprint.',
            ),
            'y': FieldOverride(label='Width (y)', unit=A, order=12,
                               help='Finite-sheet mode only.'),
            'amorph': FieldOverride(label='Structure', order=13),
            'match': FieldOverride(
                label='Lattice match', order=14,
                help='Strain the substrate commensurate with the sheet.',
            ),
            'match_strain_tol': FieldOverride(
                label='Match strain tolerance', order=15,
                enable_if=('match', ENABLED),
                enable_hint='Tick "Lattice match" to set a strain tolerance.',
                help='Largest strain accepted when matching.',
            ),
        },
    ),
    'general': SectionOverride(
        label='Run parameters', order=4,
        fields={
            'temp': FieldOverride(label='Temperature', unit='K', order=1),
            'force': FieldOverride(label='Normal load', unit='nN', order=2),
            'pressure': FieldOverride(label='Pressure', unit='GPa', order=3),
            'scan_speed': FieldOverride(
                label='Scan speed', unit='m/s', order=4,
                help='Converted in the LAMMPS script: 100 m/s = 1 Å/ps.',
                detail=('The stored value is metres per second; the generated script '
                        'divides by 100 to get the Å/ps LAMMPS drives with.'),
            ),
            'scan_angle': FieldOverride(label='Scan angle', unit='deg', order=5),
            'scan_angle_force': FieldOverride(
                label='Scan angles apply at these loads', unit='nN', order=6,
                enable_if=('scan_angle', ENABLED),
                enable_hint='Set one or more scan angles first.',
                help='Empty: every angle at every load.',
                detail=('Give loads here to pair them with the scan angles instead, in '
                        'the same order.'),
            ),
            'outer_loop': FieldOverride(
                label='Split run phase by', order=7,
                help='One run script per value instead of one in-script loop.',
                detail=('Emits one run script per value of the named parameter instead '
                        'of looping inside a single script. This is what multiplies '
                        'the HPC job count.'),
            ),
            'finite_sheet': FieldOverride(
                label='Finite sheet', order=8,
                help='Build free edges rather than a periodic sheet.',
            ),
            'bond_spring': FieldOverride(
                label='Bond spring constant', order=9, unit='N/m',
                help='Harmonic bonds holding a sheet together.',
                detail='Converted to eV/Å² in the builder (divide by 16.02).',
            ),
            'driving_spring': FieldOverride(
                label='Driving spring constant', order=10, unit='N/m',
                help='Stiffness of the spring dragging the moving sheet.',
                detail='Converted to eV/Å² in the builder (divide by 16.02).',
            ),
            'twist_angle': FieldOverride(
                label='Twist angle', unit='deg', order=11,
                help='Relative rotation between the two sheets.',
                detail=('Setting this builds a twisted (moiré) cell and routes the run '
                        'to the heterostructure builder. The achieved angle is the '
                        'nearest commensurate one, not necessarily this target.'),
            ),
            'twist_angle_tol': FieldOverride(
                label='Twist angle tolerance', unit='deg', order=12,
                enable_if=('twist_angle', ENABLED),
                enable_hint='Set a twist angle first.',
                help='How far the achieved angle may fall from the requested one.',
            ),
            'twist_strain_tol': FieldOverride(
                label='Twist strain tolerance', order=13,
                enable_if=('twist_angle', ENABLED),
                enable_hint='Set a twist angle first.',
            ),
            'twist_max_supercell': FieldOverride(
                label='Twist max supercell', order=14,
                enable_if=('twist_angle', ENABLED),
                enable_hint='Set a twist angle first.',
                help='Largest supercell searched for a coincidence cell.',
            ),
            'hetero_stacking': FieldOverride(
                label='Heterostructure stacking', order=15,
                help='Layer ordering across multiple sheet materials.',
                detail='Has no effect on a single-material run.',
            ),
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


#: Model name used by the settings document, so its sections get their own
#: labels rather than falling back to raw field names.
SETTINGS_MODEL = '__settings__'

#: Settings groups. Only the fields worth explaining are listed; the rest fall
#: back to their model description.
_SETTINGS: Dict[str, SectionOverride] = {
    'hpc': SectionOverride(
        label='HPC',
        help='Cluster and job-script settings.',
        detail='Job scripts are only written when Modules is filled in.',
        fields={
            'modules': FieldOverride(
                label='Modules', order=1,
                help='Required — with this empty, no job scripts are written.',
                detail=('Environment modules your cluster loads before running, comma '
                        'separated, e.g. lammps, openmpi. With this empty no job '
                        'scripts are written at all, whatever the other settings say.'),
            ),
            'scheduler_type': FieldOverride(
                label='Scheduler', order=2,
                help='Which submission system the scripts target.',
            ),
            'queue': FieldOverride(label='Queue', order=3),
            'partition': FieldOverride(label='Partition', order=4),
            'account': FieldOverride(
                label='Account', order=5,
                help='Project or budget code to charge the job to.',
            ),
            'num_nodes': FieldOverride(label='Nodes', order=6),
            'num_cpus': FieldOverride(label='CPUs per node', order=7),
            'memory_gb': FieldOverride(label='Memory', unit='GB', order=8),
            'walltime_hours': FieldOverride(label='Walltime', unit='h', order=9),
            'max_array_size': FieldOverride(
                label='Max array size', order=10,
                help='Longer runs are split across several arrays.',
                detail='Largest job array the scheduler accepts.',
            ),
            'hpc_host': FieldOverride(label='Host', order=11),
            'hpc_home': FieldOverride(
                label='Remote working directory', order=12,
                help='Where the simulation tree will live on the cluster.',
            ),
            'mpi_command': FieldOverride(label='MPI launcher', order=13),
            'use_tmpdir': FieldOverride(
                label='Run in scratch', order=14,
                help='Copy the run into node-local scratch and back afterwards.',
            ),
            'scratch_dir': FieldOverride(
                label='Scratch directory', order=15,
                enable_if=('use_tmpdir', ENABLED),
                enable_hint='Enable "Run in scratch" to set this.',
            ),
            'lammps_scripts': FieldOverride(
                label='Scripts to submit', order=16,
                help='Which generated run scripts get job entries.',
            ),
        },
    ),
    'quench': SectionOverride(
        label='Quench',
        help='Melt-quench parameters for amorphous structures.',
        fields={
            'allow_generate': FieldOverride(
                label='Allow generating amorphous structures', order=1,
                help='When off, a missing structure is an error.',
                detail=('Rather than a melt-quench. The preview always runs with this '
                        'off, whatever this setting says.'),
            ),
            'run_local': FieldOverride(label='Run locally', order=2),
            'n_procs': FieldOverride(label='Processes', order=3),
            'quench_slab_dims': FieldOverride(
                label='Quench slab', unit=A, order=4, editor='xyz_dims'),
        },
    ),
    'thermostat': SectionOverride(
        label='Thermostat',
        fields={
            'langevin_boundaries': FieldOverride(
                label='Langevin boundaries', order=3,
                editor='langevin_bands',
                help='Fixed and thermostatted bands per component.',
                detail=('Tip bands are absolute Å below the tip top face; '
                        'substrate bands are a fraction of its thickness from '
                        'the bottom.'),
            ),
        },
    ),
    'geometry': SectionOverride(label='Geometry'),
    'simulation': SectionOverride(label='Simulation'),
    'output': SectionOverride(
        label='Output',
        fields={
            'dump': FieldOverride(label='Dump phases', order=1,
                                  editor='key_value'),
            'dump_frequency': FieldOverride(
                label='Dump frequency', order=2, editor='key_value',
                help='Steps between dumps, per phase.'),
        },
    ),
    'potential': SectionOverride(label='Potential'),
    'aiida': SectionOverride(label='AiiDA'),
    'database': SectionOverride(label='Database'),
}


def section_override(section_key: str) -> SectionOverride:
    """Display metadata for a section, empty if none is declared."""
    return _COMMON.get(section_key, SectionOverride())


def field_override(model: str, section: str, key: str) -> Optional[FieldOverride]:
    """Display metadata for one field of one model."""
    return overrides_for(model).get(section, SectionOverride()).fields.get(key)


def overrides_for(model: str) -> Dict[str, SectionOverride]:
    """Section overrides for a simulation model, with per-family adjustments.

    Two things vary by family, both because offering them anyway would let
    someone configure something that has no effect:

    * a substrate-free config is only accepted by the PES tip scan -- every
      other model raises ``FreestandingConfigError`` without ``[sub]``;
    * the AFM templates read a normal load and never a pressure, while the
      sheet-on-sheet templates are the reverse and are the only ones that read
      the two spring constants.
    """
    if model == SETTINGS_MODEL:
        return dict(_SETTINGS)

    entry = MODES.get(model)
    result = dict(_COMMON)

    if entry is None:
        return result

    family, _mode = entry

    if model != 'pes_tip':
        substrate = result.get('sub', SectionOverride())
        result['sub'] = SectionOverride(
            label=substrate.label, help=substrate.help, order=substrate.order,
            fields=substrate.fields, detail=substrate.detail,
            required_reason=(
                'Only the PES tip scan supports a substrate-free (freestanding) '
                'configuration; other models abort without one.'
            ),
        )

    general = result.get('general', SectionOverride())
    fields = {
        key: value for key, value in general.fields.items()
        if family.supports('general', key)
    }
    if family.outer_loop_choices and 'outer_loop' in fields:
        existing = fields['outer_loop']
        fields['outer_loop'] = FieldOverride(
            label=existing.label, unit=existing.unit, help=existing.help,
            order=existing.order, choices=family.outer_loop_choices,
            detail=existing.detail, editor=existing.editor,
        )
    result['general'] = SectionOverride(
        label=general.label, help=general.help, order=general.order, fields=fields,
        detail=general.detail,
    )

    return result


def hidden_fields(model: str, section: str) -> Tuple[str, ...]:
    """Keys this model's templates never read, so the form should not offer them."""
    entry = MODES.get(model)
    if entry is None or section != 'general':
        return ()
    return entry[0].unsupported_general
