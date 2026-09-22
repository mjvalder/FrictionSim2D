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

import re
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

from .simulation_types import MODES

__all__ = [
    'FieldOverride', 'SectionOverride', 'overrides_for', 'section_override',
    'override_for_section', 'ENABLED', 'SET', 'field_override',
    'CAPTION_BUDGET', 'SETTINGS_MODEL', 'PLOT_MODEL',
]

#: A numbered sheet section as it is written to disk: ``[2D-1]``, ``[2D-2]``.
_NUMBERED_SHEET = re.compile(r'^2D-\d+$')

#: Sentinel for ``enable_if``: the other field just has to hold a truthy value.
#: Wrong for a field whose valid range includes a falsy-but-meaningful value
#: (0 is a real scan angle, a straight-line scan) -- use ``SET`` for those.
ENABLED = object()

#: Sentinel for ``enable_if``: the other field just has to hold a value at
#: all -- not ``None``, not ``''``, not ``[]``. Unlike ``ENABLED`` this does
#: not test truthiness, so 0 / 0.0 count as set.
SET = object()

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
    #: ``(field key, required value)``; ``ENABLED`` means "any truthy value",
    #: ``SET`` means "holds a value at all" (0 / 0.0 count).
    #: A greyed control says "this does nothing yet" far more clearly than a
    #: value that is quietly ignored.
    enable_if: Optional[Tuple[str, object]] = None
    #: Reason shown when the control is disabled by ``enable_if``.
    enable_hint: Optional[str] = None
    #: Render read-only.
    read_only: bool = False
    #: Restrict an enum's choices below what the model allows.
    choices: Optional[Tuple[object, ...]] = None
    #: Display text for individual enum choices, keyed by the stored value.
    #: A choice absent here still renders (as `str(choice)`, the existing
    #: behaviour) -- this only relabels what is shown, never what is stored.
    #: `_EnumEditor` keeps the item's `itemData` as the real value, so this
    #: is purely cosmetic: the config still reads/writes the original string.
    choice_labels: Optional[Dict[object, str]] = None
    #: Long-form explanation. Tooltip only -- never rendered as a caption,
    #: because a caption under every control is what made the form unreadable.
    detail: Optional[str] = None
    #: Slug requesting a purpose-built editor; see widgets.fields.EDITOR_REGISTRY.
    editor: Optional[str] = None
    #: Overrides the introspected requiredness. ``None`` defers to the
    #: model's own required flag; ``True``/``False`` force it either way --
    #: some fields the builder treats as obligatory even though the Pydantic
    #: model defaults them (e.g. ``temp``), so introspection alone would
    #: never star them.
    required: Optional[bool] = None
    #: The field is only obligatory given another *settings* value (not
    #: another field in the same section -- that is ``enable_if``). Given as
    #: ``('dotted.settings.path', expected value)``; evaluated against the
    #: `GlobalSettings` handed to the form, not the document being edited.
    required_if_settings: Optional[Tuple[str, object]] = None
    #: Whether leaving this starred field empty blocks Generate (True) or
    #: only warns (False). Only meaningful when the field is required (by
    #: either of the two mechanisms above).
    required_blocks: bool = True
    #: Key into the owning SectionOverride's ``gates``: the field is only
    #: shown, pre-filled, and committed while that gate's checkbox is ticked.
    #: Unlike ``enable_if`` (greyed out but still visible and still written),
    #: a gated field is hidden entirely and its key is kept out of the
    #: document while the gate is off -- for fields that are actively
    #: confusing to see unless deliberately in use, not merely inert.
    gate: Optional[str] = None


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
    #: Checkbox gates for this section's fields: gate key -> checkbox label.
    #: A field opts in via ``FieldOverride.gate``.
    gates: Dict[str, str] = field(default_factory=dict)


A = 'Å'

_COMPONENT_FIELDS = {
    'mat': FieldOverride(label='Material', order=1,
                         help='Material identifier, e.g. h-MoS2.'),
    'cif_path': FieldOverride(label='Structure file', order=2,
                              help='CIF describing the crystal structure.'),
    'pot_type': FieldOverride(label='Potential type', order=3),
    'pot_path': FieldOverride(label='Potential file', order=4),
}

#: Field overrides for a sheet section. Shared by `[2D]` and `[2D-2]`: they
#: are the same `SheetConfig`, so restating them would only let the two
#: sheets of one heterostructure drift into different labels and units.
_SHEET_FIELDS = {
    **_COMPONENT_FIELDS,
    # Shifted down from _COMPONENT_FIELDS' shared order (2/3/4) to
    # leave a gap for the material-list block, which SectionForm
    # inserts as its own row right after `mat` -- see
    # SectionForm._insert_material_list_row. Only this section
    # reorders them; tip/sub/flake keep _COMPONENT_FIELDS' order as
    # they have no materials_list field to make room for.
    'cif_path': replace(_COMPONENT_FIELDS['cif_path'], order=3),
    'pot_type': replace(_COMPONENT_FIELDS['pot_type'], order=4),
    'pot_path': replace(_COMPONENT_FIELDS['pot_path'], order=5),
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
        label='Sweep over materials', order=0, editor='material_list',
        detail=('Each material becomes a separate simulation. The structure '
                'and potential directories decide which are available: a '
                'material needs a file in both.'),
    ),
}

#: Section overrides shared by both families.
_COMMON: Dict[str, SectionOverride] = {
    '2D': SectionOverride(
        label='2D sheet', order=1,
        help='The sheet being slid or indented.',
        fields=dict(_SHEET_FIELDS),
    ),
    # The second sheet of a heterostructure. `ConfigDocument.SECOND_SHEET`
    # spells the same key; it is repeated rather than imported because the
    # document imports this module, not the other way round.
    #
    # No `order`: this section is never sorted. It has no field on the root
    # model at all (the models fold numbered sections into a derived `sheets`
    # list), so it never appears in `ConfigDocument.sections()` -- `ConfigForm`
    # builds it from `second_material_spec()` and inserts it directly after
    # `[2D]`, where a heterostructure's two materials have to be read together.
    '2D-2': SectionOverride(
        label='Material 2 (heterostructure)',
        help='The second material; the two are stacked into one cell.',
        # Minus `materials_list`: only the first sheet carries the sweep list,
        # and each of its lines names *both* materials, so a second list would
        # be a second, conflicting sweep.
        fields={key: value for key, value in _SHEET_FIELDS.items()
                if key != 'materials_list'},
    ),
    # Ahead of tip/substrate/run parameters: lattice matching is how the sheets
    # are made commensurate with each other, so it is read with them.
    'lattice_match': SectionOverride(label='Lattice matching', order=2),
    'tip': SectionOverride(
        label='Tip', order=3,
        help='The indenting tip.',
        gates={'tip_position': 'Set tip position'},
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
            'tip_x': FieldOverride(
                label='Tip x', unit=A, order=20, gate='tip_position',
                help='Absolute tip x-position in box coordinates.',
            ),
            'tip_y': FieldOverride(
                label='Tip y', unit=A, order=21, gate='tip_position',
                help='Absolute tip y-position in box coordinates.',
            ),
            'tip_x_offset': FieldOverride(
                label='Tip x offset', unit=A, order=22, gate='tip_position',
                help='Used only when tip x is not set.',
            ),
            'tip_y_offset': FieldOverride(
                label='Tip y offset', unit=A, order=23, gate='tip_position',
                help='Used only when tip y is not set.',
            ),
        },
    ),
    'sub': SectionOverride(
        label='Substrate', order=4,
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
        label='Run parameters', order=5,
        fields={
            'temp': FieldOverride(label='Temperature', unit='K', order=1, required=True),
            'force': FieldOverride(
                label='Normal load', unit='nN', order=2,
                required=True, required_blocks=True,
                detail=('An empty load list is not a zero-load run: system_init.lmp '
                        'and slide.lmp both render `variable find index` with no '
                        'values (or `variable find equal None` when the field is '
                        'simply absent) -- LAMMPS aborts on that line, it does not '
                        'produce a no-load run. There is no builder default for '
                        'force in any code path, so this blocks Generate.'),
            ),
            'pressure': FieldOverride(
                label='Pressure', unit='GPa', order=3,
                required=True, required_blocks=True,
                detail=('The `pressures or [0.0]` fallback only exists in the two '
                        'outer_loop branches (sheetonsheet.py outer_loop == '
                        '"pressure"/"scan_speed"). The default single-script path '
                        '(outer_loop unset) passes `self.config.general.pressure` '
                        'straight into the template unguarded, and an absent '
                        'pressure renders `variable pressure equal None` in '
                        'sheetonsheet/slide.lmp -- the same unparseable-LAMMPS '
                        'failure as an empty force list. Such configs were never '
                        'actually valid on the default path, so this blocks '
                        'Generate rather than only warning.'),
            ),
            'scan_speed': FieldOverride(
                label='Scan speed', unit='m/s', order=4, required=True,
                help='Converted in the LAMMPS script: 100 m/s = 1 Å/ps.',
                detail=('The stored value is metres per second; the generated script '
                        'divides by 100 to get the Å/ps LAMMPS drives with.'),
            ),
            'scan_angle': FieldOverride(label='Scan angle', unit='deg', order=5),
            'scan_angle_force': FieldOverride(
                label='Scan angles apply at these loads', unit='nN', order=6,
                enable_if=('scan_angle', SET),
                enable_hint='Set one or more scan angles first.',
                help='Empty: every angle at every load.',
                detail=('Give loads here to pair them with the scan angles instead, in '
                        'the same order.'),
            ),
            'outer_loop': FieldOverride(
                label='Split run phase by', order=7,
                help='One run script per value instead of one in-script loop.',
                detail=('By default every value of a swept parameter -- e.g. every '
                        'scan angle -- loops inside one run script. Choosing a '
                        'parameter here instead emits one run script, and one HPC '
                        'job, per value; this is what multiplies the job count. A '
                        'parameter with only one configured value is never split, '
                        'whether or not it is chosen here.'),
                # Display only -- the stored/written value stays 'scan_angle'
                # (see FieldOverride.choice_labels). run.py's job-count lookup
                # and the builders both key off the literal 'scan_angle'.
                choice_labels={'scan_angle': 'angle'},
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
                required_if_settings=('simulation.drive_method', 'virtual_atom'),
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
            # Written to [general], but shown on the second sheet's section
            # (see `hidden_fields`) -- the label and help below are what that
            # proxy row renders, so this entry is still the single source of
            # both, and `ConfigDocument.comments()` still annotates the written
            # key from it.
            'hetero_stacking': FieldOverride(
                label='Layer stacking', order=15,
                help='How the two materials interleave in the stack.',
                detail=("'grouped' puts every layer of the first material "
                        "below every layer of the second; 'alternating' "
                        "interleaves them. Only affects multi-layer sheets."),
            ),
        },
    ),
    'flake': SectionOverride(
        label='Flake', order=6,
        help='A finite patch placed between tip and sheet.',
        fields={
            **_COMPONENT_FIELDS,
            'edge_length': FieldOverride(label='Edge length', unit=A, order=10),
            'shape': FieldOverride(label='Shape', order=11),
            'x': FieldOverride(label='Supercell x', unit=A, order=12),
            'y': FieldOverride(label='Supercell y', unit=A, order=13),
            'center_x': FieldOverride(label='Centre x', unit=A, order=14),
            'center_y': FieldOverride(label='Centre y', unit=A, order=15),
            'corner_hold': FieldOverride(
                label='Hold corners', order=16,
                help='Sheet-on-sheet flake slide only: tether the flake corners in place.',
                detail=("Pins the flake's corner atoms to their starting xy "
                        "position while it is driven, so the flake cannot "
                        "translate -- only use it for static/indent-like "
                        "studies. Default off, so the flake is deformable and "
                        "moves under its drive spring. AFM indentation does "
                        "not read this field: it always holds its corners."),
            ),
        },
    ),
    'pes': SectionOverride(
        label='PES scan', order=7,
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
        label='Indent', order=8,
        help='Hold-and-retract settings for an indent-only run.',
    ),
    'lj_override': SectionOverride(
        label='Lennard-Jones overrides', order=9,
        help='Free-form pair overrides; keys are chosen by you.',
    ),
}


#: Model name used by the settings document, so its sections get their own
#: labels rather than falling back to raw field names.
SETTINGS_MODEL = '__settings__'

#: Sentinel model name for the plot editor, like SETTINGS_MODEL. Not a
#: simulation model: `overrides_for` branches on it before consulting MODES.
PLOT_MODEL = '__plot__'

#: Section labels for the plot editor. The fields themselves carry their
#: Field(description=...) from plot_config, so only the sections need naming.
_PLOT: Dict[str, SectionOverride] = {
    'common': SectionOverride(
        label='Common',
        fields={
            'tier': FieldOverride(
                label='Sampling tier', order=1,
                help='Which results tier to plot.',
                detail='"window" is the block-averaged tier every summary '
                       'uses; "raw" is 100 fs samples; "peratom" holds the '
                       'per-atom computes.'),
            'lf_source': FieldOverride(
                label='Lateral-force source', order=2,
                help='Which lateral-force definition to plot.',
                detail='Refused rather than aliased when the dataset does '
                       'not carry it -- a db:// dataset has one.'),
        }),
    'summary': SectionOverride(label='Summary'),
    'timeseries': SectionOverride(label='Timeseries'),
    'stick_slip_analysis': SectionOverride(label='Stick-slip'),
    'scatter_comparison': SectionOverride(label='Scatter comparison'),
    'rank_friction': SectionOverride(label='Ranking'),
    'correlation': SectionOverride(label='Correlation'),
    'cof_histogram': SectionOverride(label='COF-slope histogram'),
}

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
            'conda_env': FieldOverride(
                label='Conda environment', order=14,
                help='Conda prefix put on PATH before running, e.g. $HOME/envs/lammps.',
            ),
            'use_tmpdir': FieldOverride(
                label='Run in scratch', order=15,
                help='Copy the run into node-local scratch and back afterwards.',
            ),
            'scratch_dir': FieldOverride(
                label='Scratch directory', order=16,
                enable_if=('use_tmpdir', ENABLED),
                enable_hint='Enable "Run in scratch" to set this.',
            ),
            'lammps_scripts': FieldOverride(
                label='Scripts to submit', order=17,
                help='Which generated run scripts get job entries.',
            ),
            'preflight': FieldOverride(
                label='Write preflight check', order=18,
                help='Also write hpc/preflight.sh and preflight.pbs (PBS only).',
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
            'channels': FieldOverride(
                label='Observable channels', order=3, editor='key_value',
                help='Sampling tiers written as fix ave/time.',
                detail=('"window" is the COF source. "raw" (100 fs '
                        'instantaneous samples) feeds the slip statistics '
                        'and "peratom" (per-atom computes) is plottable by '
                        'column name -- postprocessing reads all three. '
                        'Disabling "peratom" also drops its group/group '
                        'computes, which rebuild a neighbour list per '
                        'invocation.')),
        },
    ),
    'potential': SectionOverride(label='Potential'),
    'aiida': SectionOverride(label='AiiDA'),
    'database': SectionOverride(label='Database'),
}


def section_override(section_key: str) -> SectionOverride:
    """Display metadata for a section, empty if none is declared."""
    return _COMMON.get(section_key, SectionOverride())


def override_for_section(model: str, section_key: str) -> Optional[SectionOverride]:
    """The override entry a section takes its labels, units and help from.

    The one rule, for both readers: `widgets.form.SectionForm`, which puts
    that text on screen, and `ConfigDocument.comments()`, which writes the
    same text back into the saved `.ini`. A section with an entry of its own
    uses it -- `[2D-2]` has one, because its field list differs (no
    `materials_list`: one list names both materials of every pair) -- and
    every other numbered sheet folds onto `[2D]`, since `[2D-1]` *is* the
    first sheet under its written name. `src.run.naming.loopability_section` folds
    them the same way for the same reason.

    The two readers used to resolve this separately: the form looked the key
    up directly, `comments()` folded every numbered sheet onto `[2D]`. They
    agreed only because the two entries happen to carry the same text -- an
    override added to `[2D-2]` alone would have rendered on screen and been
    silently dropped from the file.
    """
    table = overrides_for(model)
    if section_key in table:
        return table[section_key]
    if _NUMBERED_SHEET.match(section_key):
        return table.get('2D')
    return None


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

    if model == PLOT_MODEL:
        return dict(_PLOT)

    entry = MODES.get(model)
    result = dict(_COMMON)

    if entry is None:
        return result

    family, mode = entry

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
    outer_loop_choices = family.outer_loop_choices_for(mode)
    # outer_loop_choices_for always returns a tuple (never None), so an empty
    # tuple here is deliberate -- a mode whose build never reads outer_loop at
    # all (e.g. the PES scans) -- not "leave unrestricted". Narrow whenever
    # the field exists, so that case actually offers zero choices instead of
    # silently falling back to every value the underlying model allows. The
    # zero-choice case renders no row at all (see hidden_fields below): an
    # empty combo reads as a broken control rather than as "not applicable".
    if 'outer_loop' in fields:
        existing = fields['outer_loop']
        fields['outer_loop'] = FieldOverride(
            label=existing.label, unit=existing.unit, help=existing.help,
            order=existing.order, choices=outer_loop_choices,
            detail=existing.detail, editor=existing.editor,
            choice_labels=existing.choice_labels,
        )
    result['general'] = SectionOverride(
        label=general.label, help=general.help, order=general.order, fields=fields,
        detail=general.detail,
    )

    return result


def hidden_fields(model: str, section: str) -> Tuple[str, ...]:
    """Keys this model's templates never read, so the form should not offer them.

    ``outer_loop`` joins them for a mode that offers no split at all (both PES
    scans): the combo would render holding nothing but its '(unset)' entry,
    which reads as a broken control rather than as "this mode has nothing to
    split".

    ``hetero_stacking`` is hidden for *every* mode, and for a different reason:
    it is not unsupported, it has simply moved. It orders the layers of a
    two-material stack, so the form shows it on the second sheet's own section
    (``widgets.form.SectionForm``'s stacking proxy), where there is a second
    material for it to order. Left in Run parameters as well it would be the
    same setting in two places, and on a single-material config it would be a
    control that silently does nothing at all.
    """
    entry = MODES.get(model)
    if entry is None or section != 'general':
        return ()
    family, mode = entry
    hidden = tuple(family.unsupported_general) + ('hetero_stacking',)
    if not family.outer_loop_choices_for(mode):
        hidden += ('outer_loop',)
    return hidden
