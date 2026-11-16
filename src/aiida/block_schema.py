"""The block schema: which block each field belongs to, its unit and its default.

This is the declarative half of the parameter blocks (spec §5.2, plan
*Revision 2026-11-16*); :mod:`src.aiida.blocks` is the half that splits a
run with it, and re-exports the names defined here.

- :data:`FIELD_MAP` assigns every requested leaf of ``AFMSimulationConfig``,
  ``SheetOnSheetSimulationConfig`` and ``GlobalSettings`` (as ``settings.<...>``)
  to a block kind. :data:`SWEPT` names the swept axes, which are point
  attributes, never block content. :data:`EXCLUDED` names what is not physics.
- :data:`PROVENANCE_MAP` assigns every ``ProvenanceRecord`` leaf to a block
  kind, to ``'derived'`` (builder-computed, stored unhashed), to ``'point'``
  or to ``'run_info'``.
- :data:`UNITS` gives every numeric field a unit; ``tests/aiida/
  test_block_schema.py`` fails for a numeric field without one. A unit change
  is a :data:`BLOCK_SCHEMA_VERSION` bump with a converter, never silent.
- :func:`defaults_for` is what default elision compares against, and what an
  elided field is refilled from (:func:`schema_document`).

Keep this module free of aiida imports and of :mod:`src.aiida.blocks`: it
runs in the fast suite, and the split imports it, not the other way round.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import (Any, Dict, Iterator, List, Literal, Mapping, Optional, Tuple,
                    Union, get_args, get_origin)

import numpy as np
from pydantic import BaseModel

from ..core.config import AFMSimulationConfig, SheetOnSheetSimulationConfig
from ..data.models import NM_TO_EV_A2
from ..data.provenance import BuiltGeometry, ProvenanceRecord
from ..run.naming import AFM_LIKE_MODELS, SHEET_LIKE_MODELS

#: Version of the block layout, the canonical form and the units below.
#: Stamped into every block's content, so a bump changes every hash.
BLOCK_SCHEMA_VERSION = '1.0.0'

#: Schema version -> {FIELD_MAP path: its default in that version}, for each
#: default that changed in the bump *out of* that version. It is a refill
#: rule, never an extra elision value: an elided field in a block stamped
#: with an older version meant that version's default, so
#: :func:`src.aiida.blocks.upgrade_block` writes the old value back, then
#: elides against the current default. A block's field takes the value from
#: the earliest recorded version at or after the block's own.
#:
#: Changing a model default therefore needs, in one commit: bump
#: :data:`BLOCK_SCHEMA_VERSION`, record the old value here under the old
#: version, and regenerate ``tests/fixtures/block_defaults.json``
#: (``test_block_schema.py::test_defaults_match_the_snapshot`` fails until then).
LEGACY_DEFAULTS: Dict[str, Dict[str, Any]] = {}


def version_key(version: str) -> Tuple[int, ...]:
    """``'1.10.0'`` -> ``(1, 10, 0)``, for ordering schema versions."""
    return tuple(int(part) for part in str(version).split('.'))

BLOCK_KINDS = ('material', 'potential', 'interaction', 'thermostat', 'integration',
               'drive', 'geometry', 'protocol', 'output', 'realization')

(_MATERIAL, _POTENTIAL, _INTERACTION, _THERMOSTAT, _INTEGRATION, _DRIVE, _GEOMETRY,
 _PROTOCOL, _OUTPUT, _REALIZATION) = BLOCK_KINDS

#: Non-block destinations of a provenance field.
DERIVED, POINT, RUN_INFO = 'derived', 'point', 'run_info'

#: Build families and the config model each one validates against.
AFM_FAMILIES = frozenset(AFM_LIKE_MODELS)
SHEET_FAMILIES = frozenset(SHEET_LIKE_MODELS | {'hetero', 'flake'})


def _fields(kind: str, *names: str) -> Dict[str, str]:
    return {name: kind for name in names}


# --- per-section field tables -------------------------------------------------

_COMPONENT = _fields(_MATERIAL, 'mat', 'cif_path')

#: Component fields whose block content comes from provenance.json instead:
#: ``pot_type``/``pot_path`` -> ``materials.pot_type``/``pot_file_sha256``
#: (what the build resolved).
_COMPONENT_FROM_PROVENANCE = _fields(_POTENTIAL, 'pot_type', 'pot_path')

_SHEET = {
    **_COMPONENT,
    **_fields(_GEOMETRY, 'x', 'y', 'layers', 'stack_type', 'lat_c'),
    # A Gaussian strength in meV: interaction, not geometry (reverses P2-R3).
    'interlayer_g': _INTERACTION,
}

_TIP = {
    **_COMPONENT,
    **_fields(_GEOMETRY, 'r', 'amorph'),
    # The viscous damping damp_ev derives from: a thermostat, not the drive.
    **_fields(_THERMOSTAT, 'dspring', 'damping_ratio'),
    **_fields(_DRIVE, 'tip_x', 'tip_y', 'tip_x_offset', 'tip_y_offset'),
}

_SUBSTRATE = {
    **_COMPONENT,
    **_fields(_GEOMETRY, 'thickness', 'x', 'y', 'amorph', 'match',
              'match_strain_tol'),
}

_FLAKE = {
    **_COMPONENT,
    **_fields(_GEOMETRY, 'shape', 'edge_length', 'rotation_deg', 'x', 'y',
              'corner_radius', 'corner_spring_k', 'corner_hold', 'center_x',
              'center_y'),
}

_GENERAL = {
    'temp': _THERMOSTAT,
    **_fields(_DRIVE, 'bond_spring', 'driving_spring'),
    **_fields(_GEOMETRY, 'finite_sheet', 'hetero_stacking', 'twist_angle',
              'twist_angle_tol', 'twist_strain_tol', 'twist_max_supercell'),
    **_fields(_INTERACTION, 'interlayer_style', 'interlayer_file'),
    # How friction is measured (reaction force vs group/group), not the
    # interaction itself.
    'interlayer_comparison': _PROTOCOL,
}

_PES = _fields(_PROTOCOL, 'grid_n', 'grid_n_refine', 'z_relax', 'tip_load',
               'scan_cells', 'eval_mode', 'md_steps')
_INDENT = {**_fields(_PROTOCOL, 'hold_steps', 'z_step', 'n_steps'),
           'seed': _REALIZATION}
_LATTICE_MATCH = _fields(_GEOMETRY, 'method', 'strain_tol', 'max_supercell',
                         'area_tol')

_SETTINGS = {
    'geometry': _fields(
        _GEOMETRY, 'tip_reduction_factor', 'rigid_tip', 'tip_start_clearance',
        'lat_c_default', 'finite_sheet_edge_width', 'finite_sheet_afm_edge_mode',
        'finite_sheet_edge_spring_k', 'finite_sheet_sheetonsheet_edge_mode'),
    'thermostat': _fields(_THERMOSTAT, 'type', 'time_integration',
                          'langevin_boundaries'),
    'simulation': {
        **_fields(_INTEGRATION, 'timestep', 'min_style', 'minimization_command',
                  'neighbor_list', 'neigh_modify_command', 'slide_run_steps',
                  'sheet_hold_steps', 'ramp_max_iterations'),
        **_fields(_DRIVE, 'drive_method', 'constraint_mode'),
        'halt_temperature_factor': _PROTOCOL,
        'thermo': _OUTPUT,
    },
    # The recipe builds the amorphous substrate: geometry, its timestep included.
    'quench': _fields(_GEOMETRY, 'quench_slab_dims', 'quench_rate', 'melt_temp',
                      'quench_temp', 'timestep', 'melt_steps', 'equilibrate_steps'),
    'output': _fields(_OUTPUT, 'dump', 'dump_frequency', 'results_frequency',
                      'channels', 'optional_channels'),
    'potential': _fields(_INTERACTION, 'lj_cutoff', 'lj_type', 'reaxff_safezone',
                         'reaxff_mincap'),
}

#: config.json sections holding one component (or a list of them).
COMPONENT_SECTIONS: Dict[str, Dict[str, str]] = {
    'sheet': _SHEET, 'sheets': _SHEET, 'flake_sheets': _SHEET,
    'tip': _TIP, 'sub': _SUBSTRATE, 'flake': _FLAKE,
}

#: config.json sections that are lists of components.
LIST_SECTIONS = ('sheets', 'flake_sheets')

_OTHER_SECTIONS: Dict[str, Dict[str, str]] = {
    'general': _GENERAL, 'pes': _PES, 'indent': _INDENT,
    'lattice_match': _LATTICE_MATCH,
}

#: Field path -> block kind, for every requested leaf of AFMSimulationConfig,
#: SheetOnSheetSimulationConfig and GlobalSettings (as ``settings.<...>``).
#: A field of a list of models (``sheets``) is named without an index; block
#: content indexes it (``sheets.0.x``).
FIELD_MAP: Dict[str, str] = {
    **{f'{section}.{name}': kind
       for section, table in {**COMPONENT_SECTIONS, **_OTHER_SECTIONS}.items()
       for name, kind in table.items()},
    'lj_override': _INTERACTION,
    **{f'settings.{section}.{name}': kind
       for section, table in _SETTINGS.items() for name, kind in table.items()},
}

#: Swept axes -> the point attribute they become. Assigned, never block
#: content: a load sweep must not put each campaign on its own block (C1a).
SWEPT: Dict[str, str] = {
    'general.force': 'load', 'general.pressure': 'load',
    'general.scan_angle': 'angle', 'general.scan_speed': 'speed',
}

#: Config field path -> block kind for the component fields a block takes
#: from provenance.json (see _COMPONENT_FROM_PROVENANCE).
PROVENANCE_SOURCED: Dict[str, str] = {
    f'{section}.{name}': kind
    for section in COMPONENT_SECTIONS
    for name, kind in _COMPONENT_FROM_PROVENANCE.items()
}

#: Every ProvenanceRecord leaf (``materials.*`` per entry, ``geometry.*`` is
#: ``BuiltGeometry``) -> a block kind, ``'derived'``, ``'point'`` or
#: ``'run_info'``; the rest are in :data:`EXCLUDED`. ``interlayer`` is a dict
#: leaf: its ``sha256`` is the interaction file, everything else is derived.
PROVENANCE_MAP: Dict[str, str] = {
    'materials.name': _MATERIAL,
    'materials.pot_type': _POTENTIAL,
    'materials.pot_file_sha256': _POTENTIAL,
    'interlayer.sha256': _INTERACTION,
    **_fields(_REALIZATION, 'seed_thermal_tip', 'seed_thermal_sub', 'seed_velocity',
              'seed_sheet_center', 'substrate_realization_id'),
    **{f'geometry.{name}': DERIVED for name in list(BuiltGeometry.model_fields)},
    **_fields(DERIVED, 'twist_angle', 'drive_k', 'drive_rest_length', 'damping_ratio',
              'damp_ev', 'thermostat_scheme', 'contact_cutoff', 'cutoff_source',
              'layer_type_map', 'tip_group_sizes', 'interlayer'),
    **_fields(POINT, 'load_requested', 'pressure_requested', 'load_achieved',
              'pressure_achieved', 'load_achieved_by_load', 'ramp_by_load'),
    **_fields(RUN_INFO, 'lammps_version', 'wall_time_s', 'n_procs', 'atoms_per_proc',
              'ramp_converged', 'ramp_r', 'ramp_iterations', 'unloaded_reference_z'),
}

#: Provenance fields a ``realization`` block reads, in order.
REALIZATION_FIELDS: Tuple[str, ...] = tuple(
    k for k, v in PROVENANCE_MAP.items() if v == _REALIZATION)

#: Keys of ``ProvenanceRecord.interlayer`` (``PotentialManager.
#: interlayer_provenance``) other than ``sha256``: derived, as ``interlayer.<key>``.
INTERLAYER_DERIVED = ('style', 'file', 'g_meV', 'generated_files', 'comparison')

#: Flat keys of :attr:`RunBlocks.derived`.
DERIVED_PATHS: Tuple[str, ...] = tuple(
    [k for k, v in PROVENANCE_MAP.items() if v == DERIVED and k != 'interlayer']
    + [f'interlayer.{k}' for k in INTERLAYER_DERIVED])

#: Point attributes with a unit (the swept axes; the load per family).
POINT_PATHS = ('point.load.force', 'point.load.pressure', 'point.angle', 'point.speed')

_HPC = ('scheduler_type', 'queue', 'partition', 'account', 'hpc_host', 'hpc_home',
        'log_dir', 'scratch_dir', 'num_nodes', 'num_cpus', 'memory_gb',
        'walltime_hours', 'max_array_size', 'modules', 'conda_env', 'mpi_command',
        'use_tmpdir', 'preflight', 'lammps_scripts')
_AIIDA = ('profile', 'computer_label', 'max_stored_file_mb', 'skip_fraction',
          'machine')
_DATABASE = ('host', 'port', 'dbname', 'user', 'password', 'api_key', 'api_url',
             'skip_fraction')
# Analysis thresholds of the post-run integrity check, not simulation physics:
# a new threshold set judges the same run differently, it is not a new run.
_INTEGRITY = ('hole_bonds', 'isolated_radius_bonds', 'isolated_atoms',
              'ejected_atoms', 'ejected_bonds', 'heating_sigma', 'gap_rise',
              'collapse', 'frozen_std', 'gap_start_samples',
              'sheet_heating_skip_samples', 'sheet_gap_end_samples', 'margins',
              'version')

#: ProvenanceRecord leaves that identify or describe the record, not the run's
#: physics. ``run_uuid`` (and a legacy record's ``run_uuid_stamped_at``) is the
#: run's identity: it lives on the CalcJob (``parameters``, extra
#: ``friction2d_run_uuid``) and on its points, never in a block, or every run
#: would get blocks of its own. ``materials.role`` is the block
#: name's role, not its content; ``pot_path``/``pot_resolution_route`` say
#: where and how the file was found, ``pot_file_sha256`` what it is (M6).
PROVENANCE_EXCLUDED = frozenset((
    'schema_version', 'run_uuid', 'run_uuid_stamped_at', 'family', 'model',
    'run_role', 'git_commit', 'git_dirty', 'config_hash', 'materials.role', 'materials.pot_path',
    'materials.pot_resolution_route'))

#: Leaf fields that are not simulation physics, each named. Job submission
#: (``hpc``), the import itself (``aiida``), the central DB (``database``,
#: until Task 11 removes it), integrity thresholds (``integrity``), how and
#: where the amorphous quench runs (not what it produces), ``materials_list``
#: (consumed by sweep expansion), the deck shape and point set
#: (``outer_loop``, ``scan_angle_force``), and :data:`PROVENANCE_EXCLUDED`.
EXCLUDED = frozenset(
    [f'settings.hpc.{n}' for n in _HPC]
    + [f'settings.aiida.{n}' for n in _AIIDA]
    + [f'settings.database.{n}' for n in _DATABASE]
    + [f'settings.integrity.{n}' for n in _INTEGRITY]
    + ['settings.quench.run_local', 'settings.quench.n_procs',
       'settings.quench.allow_generate']
    + [f'{s}.materials_list' for s in ('sheet', 'sheets', 'flake_sheets')]
    + ['general.outer_loop', 'general.scan_angle_force']
) | PROVENANCE_EXCLUDED


# --- units ----------------------------------------------------------------------

_A = 'angstrom'
_SPRING_EV = 'eV/angstrom^2'
_SHEETS = ('sheet', 'sheets', 'flake_sheets')

#: Field path -> unit (ASCII). Keys are FIELD_MAP paths, realization fields,
#: ``derived.<path>`` and ``point.<attr>``. Each unit is checked against the
#: builder or template that consumes the value (trailing comment). Lengths
#: are Angstrom, LAMMPS ``metal`` units throughout.
UNITS: Dict[str, str] = {
    # sheets ([2D], [2D-N], [Flake-N])
    **{f'{s}.x': _A for s in _SHEETS},  # gui/schema/overrides.py:219
    **{f'{s}.y': _A for s in _SHEETS},  # gui/schema/overrides.py:220
    **{f'{s}.layers': '1' for s in _SHEETS},  # layer count, read_data.py:518
    # overrides.py:229 "Interlayer spacing"; sheetonsheet/slide.lmp:77 bond r0
    **{f'{s}.lat_c': _A for s in _SHEETS},
    # overrides.py:231; potentials/manager.py:1135 g_meV
    **{f'{s}.interlayer_g': 'meV' for s in _SHEETS},
    # tip
    'tip.r': _A,  # overrides.py:281; afm/base.py:448 tip height = r / factor
    'tip.dspring': 'nN*s/m',  # afm/base.py:795 damp_ev = dspring/0.016: 1 nN s/m = 62.4 eV ps/A^2
    'tip.damping_ratio': '1',  # zeta of critical, core/config.py:474
    'tip.tip_x': _A, 'tip.tip_y': _A,  # overrides.py:295,299
    'tip.tip_x_offset': _A, 'tip.tip_y_offset': _A,  # overrides.py:303,307
    # substrate
    'sub.thickness': _A,  # overrides.py:317; construction/components.py:2284 fraction * thickness
    'sub.x': _A, 'sub.y': _A,  # overrides.py:319,323
    'sub.match_strain_tol': '1',  # fractional strain, core/config.py:525
    # flake
    'flake.edge_length': _A,  # overrides.py:472; flake/slide.lmp:93
    'flake.rotation_deg': 'deg',  # construction/components.py:170
    'flake.x': _A, 'flake.y': _A,  # overrides.py:474,475
    'flake.corner_radius': _A,  # core/config.py:599 capture radius (A)
    'flake.corner_spring_k': _SPRING_EV,  # flake/slide.lmp:153 spring/self K (metal: eV/A^2)
    'flake.center_x': _A, 'flake.center_y': _A,  # overrides.py:476,477
    # general
    'general.temp': 'K',  # overrides.py:341; afm/slide.lmp:85 langevin T
    'general.bond_spring': 'N/m',  # overrides.py:400; sheetonsheet/base.py:798 / NM_TO_EV_A2
    'general.driving_spring': 'N/m',  # overrides.py:405; afm/base.py:864 / NM_TO_EV_A2
    'general.twist_angle': 'deg',  # overrides.py:411
    'general.twist_angle_tol': 'deg',  # overrides.py:418
    'general.twist_strain_tol': '1',  # fractional strain, core/config.py:708
    'general.twist_max_supercell': '1',  # repeat count, core/config.py:715
    # pes
    'pes.grid_n': '1', 'pes.grid_n_refine': '1',  # grid points, afm/pes_tip.py:83
    'pes.tip_load': 'nN',  # afm/freestanding.py:39 "tip_load=%s nN"
    'pes.scan_cells': '1',  # cell count, afm/pes_tip.py:91
    'pes.md_steps': 'steps',  # MD steps per grid point, core/config.py:801
    # indent
    'indent.hold_steps': 'steps',  # afm/indent.lmp:143 run {{ hold_steps }}
    'indent.z_step': _A,  # afm/indent.lmp:204 variable zstep (A)
    'indent.n_steps': '1',  # retract increments, afm/indent.lmp:205 loop count
    'indent.seed': '1',  # RNG seed
    # lattice_match
    'lattice_match.strain_tol': '1', 'lattice_match.area_tol': '1',  # fractions, core/config.py:865
    'lattice_match.max_supercell': '1',  # repeat count, core/config.py:866
    # settings.geometry
    'settings.geometry.tip_reduction_factor': '1',  # afm/base.py:448 r / factor
    'settings.geometry.tip_start_clearance': _A,  # afm/base.py:566; core/config.py:27
    'settings.geometry.lat_c_default': _A,  # construction/components.py:2083 lat_c fallback
    'settings.geometry.finite_sheet_edge_width': _A,  # construction/hetero_stack.py:1879
    'settings.geometry.finite_sheet_edge_spring_k': _SPRING_EV,  # afm/slide.lmp:70 spring/self
    # settings.thermostat: tip bands are A below the tip top, sub bands are
    # fractions of the substrate thickness (construction/components.py:2275,2284)
    'settings.thermostat.langevin_boundaries': 'tip=angstrom;sub=1',
    # settings.simulation
    'settings.simulation.timestep': 'ps',  # afm/slide.lmp:44 timestep (metal: ps)
    'settings.simulation.thermo': 'steps',  # afm/slide.lmp:45 thermo N
    'settings.simulation.neighbor_list': _A,  # common/init.lmp:10 neighbor skin (A)
    'settings.simulation.slide_run_steps': 'steps',  # afm/base.py:859 run_steps
    'settings.simulation.sheet_hold_steps': 'steps',  # sheetonsheet/base.py:827 hold_steps
    'settings.simulation.halt_temperature_factor': '1',  # simulation_base.py:55 x target T
    'settings.simulation.ramp_max_iterations': '1',  # afm/system_init.lmp:161 loop count
    # settings.quench
    'settings.quench.quench_slab_dims': _A,  # overrides.py:623; components.py:1727 vs target_x (A)
    # K/s is the intended meaning, but construction/components.py:1765-1766 computes
    # nsteps = (melt_temp - quench_temp) * rate * 1e-12 / dt, i.e. proportional to
    # the rate; a K/s rate gives nsteps = dT / (rate * 1e-12 * dt). Both agree only
    # at the 1e12 default. Pinned by an xfail in test_block_schema.py; the builder
    # fix awaits the user.
    'settings.quench.quench_rate': 'K/s',
    'settings.quench.melt_temp': 'K',  # common/make_amorphous.lmp:25 velocity create T
    'settings.quench.quench_temp': 'K',  # common/make_amorphous.lmp:32 npt temp
    'settings.quench.timestep': 'ps',  # common/make_amorphous.lmp:21 timestep
    'settings.quench.melt_steps': 'steps',  # common/make_amorphous.lmp:29 run
    'settings.quench.equilibrate_steps': 'steps',  # common/make_amorphous.lmp:37 run
    # settings.output
    'settings.output.dump_frequency': 'steps',  # afm/base.py:868 dump_freq
    'settings.output.results_frequency': 'steps',  # afm/base.py:867 results_freq
    # settings.potential
    'settings.potential.lj_cutoff': _A,  # construction/components.py:604 gap <= lj_cutoff
    'settings.potential.reaxff_safezone': '1',  # potentials/manager.py:1359 multiplier
    'settings.potential.reaxff_mincap': '1',  # potentials/manager.py:1360 array capacity
    # realization (provenance)
    'seed_thermal_tip': '1', 'seed_thermal_sub': '1',  # afm/slide.lmp:85,89 langevin seeds
    'seed_velocity': '1', 'seed_sheet_center': '1',  # flake/slide.lmp:117 velocity seed
    # derived (provenance, builder-computed)
    'derived.geometry.lat_c': _A,  # data/provenance.py:48 "lengths in Angstrom"
    'derived.geometry.unit_cell': _A,  # data/provenance.py:53 xlo..zhi
    'derived.geometry.cell_a': _A, 'derived.geometry.cell_b': _A,  # provenance.py:54,55
    'derived.geometry.sheet_dims': _A,  # data/provenance.py:56 box xlo..zhi
    'derived.geometry.atoms_per_layer': '1',  # count
    'derived.geometry.layer_thickness': _A,  # data/provenance.py:58 z-extent
    'derived.geometry.tip_r': _A,  # data/provenance.py:48
    'derived.twist_angle': 'deg',  # data/provenance.py:73 built stacking twist
    'derived.drive_k': _SPRING_EV,  # data/provenance.py:120; simulation_base.drive_spring
    'derived.drive_rest_length': _A,  # data/provenance.py:121; afm/base.py:866 1.5 r
    'derived.damping_ratio': '1',  # afm/base.py:799 zeta
    'derived.damp_ev': 'eV*ps/angstrom^2',  # afm/slide.lmp:121 fix viscous gamma (metal)
    'derived.contact_cutoff': _A,  # builders/utils/obs_ctx.py:28 contact distance
    'derived.layer_type_map': '1',  # atom type ids
    'derived.tip_group_sizes': '1',  # atom counts
    'derived.interlayer.g_meV': 'meV',  # potentials/manager.py:1135 g_meV values
    # points
    'point.load.force': 'nN',  # overrides.py:343; afm/system_init.lmp:140 / ev_a_to_nn
    'point.load.pressure': 'GPa',  # overrides.py:353; sheetonsheet/slide.lmp:131 / ev_a3_to_gpa
    'point.angle': 'deg',  # overrides.py:372; afm/slide.lmp:129 cos(a*PI/180)
    'point.speed': 'm/s',  # overrides.py:367; afm/slide.lmp:131 speed/100 -> A/ps
}

#: ``load_kind`` -> the unit of each identity attribute of a point (its
#: ``units`` attribute). One entry per ``load_kind`` that
#: :func:`src.aiida.ingest.load_points` produces: ``'force'`` (tip families),
#: ``'pressure'`` (sheet families) and ``'none'`` (a PES grid, whose load,
#: angle and speed are placeholders 0.0 -- so they carry no unit).
POINT_UNITS: Dict[str, Dict[str, str]] = {
    'force': {'load': UNITS['point.load.force'], 'angle': UNITS['point.angle'],
              'speed': UNITS['point.speed']},
    'pressure': {'load': UNITS['point.load.pressure'], 'angle': UNITS['point.angle'],
                 'speed': UNITS['point.speed']},
    'none': {},
}


# --- model introspection --------------------------------------------------------

def model_in(annotation: Any) -> Optional[type]:
    """The pydantic model inside ``annotation`` (bare, Optional or List), if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (dict, Mapping):
        return None
    for arg in get_args(annotation):
        found = model_in(arg)
        if found is not None:
            return found
    return None


def _leaves(model: type, prefix: str = '') -> Iterator[Tuple[str, Any]]:
    """``(path, FieldInfo)`` of every leaf of ``model``, recursing into models."""
    for name, info in model.model_fields.items():
        sub = model_in(info.annotation)
        if sub is None:
            yield f'{prefix}{name}', info
        else:
            yield from _leaves(sub, f'{prefix}{name}.')


def leaf_field_paths(model: type, prefix: str = '') -> Iterator[str]:
    """Every leaf field path of ``model``, recursing into nested models.

    A field whose type is (an Optional or a List of) a model is descended into;
    everything else -- scalars, lists of scalars, dicts -- is a leaf.
    """
    for path, _info in _leaves(model, prefix):
        yield path


#: Field path -> annotation, for every config/settings and provenance leaf.
ANNOTATIONS: Dict[str, Any] = {
    **{p: i.annotation for p, i in _leaves(ProvenanceRecord)},
    **{p: i.annotation for p, i in _leaves(SheetOnSheetSimulationConfig)},
    **{p: i.annotation for p, i in _leaves(AFMSimulationConfig)},
}


def unindexed(path: str) -> str:
    """``sheets.0.x`` -> ``sheets.x``: the :data:`FIELD_MAP` path of a content path."""
    return '.'.join(part for part in path.split('.') if not part.isdigit())


def _types(annotation: Any) -> set:
    """The scalar types an annotation can hold (dict keys excluded)."""
    if annotation is Any:
        return {Any}
    origin = get_origin(annotation)
    if origin is Literal:
        return {type(arg) for arg in get_args(annotation)}
    args = get_args(annotation)
    if not args:
        return {annotation}
    if origin in (dict, Mapping):
        args = args[1:]
    out: set = set()
    for arg in args:
        out |= _types(arg)
    return out


def is_numeric(annotation: Any) -> bool:
    """Does the annotation hold numbers (bools are not numbers here)?"""
    return bool(_types(annotation) & {int, float})


_NONE = type(None)


def is_sweep_union(annotation: Any) -> bool:
    """``Union[float, List[float]]`` (Optional or not): a value or a sweep list."""
    if get_origin(annotation) is not Union:
        return False
    args = [a for a in get_args(annotation) if a is not _NONE]
    has_list = any(get_origin(a) is list for a in args)
    has_scalar = any(get_origin(a) is None for a in args)
    return has_list and has_scalar


def _plain(value: Any) -> Any:
    """numpy scalars and tuples as plain JSON values."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _to_float(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return float(value)
    if isinstance(value, Mapping):
        return {k: _to_float(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_float(v) for v in value]
    return value


def coerce(path: str, value: Any) -> Any:
    """``value`` as plain JSON, ints made floats where the field holds floats only.

    ``path`` may be indexed (``sheets.0.x``). An unknown path is only made plain.
    """
    value = _plain(value)
    annotation = ANNOTATIONS.get(unindexed(path))
    if annotation is not None:
        types = _types(annotation)
        if float in types and int not in types and Any not in types:
            value = _to_float(value)
    return value


def family_model(family: str) -> type:
    """The config model a build family validates against."""
    if family in AFM_FAMILIES:
        return AFMSimulationConfig
    if family in SHEET_FAMILIES:
        return SheetOnSheetSimulationConfig
    raise ValueError(f'unknown family {family!r}; expected one of '
                     f'{sorted(AFM_FAMILIES | SHEET_FAMILIES)}')


_DEFAULTS_CACHE: Dict[type, Dict[str, Any]] = {}


def defaults_for(family: str) -> Dict[str, Any]:
    """Model default per :data:`FIELD_MAP` path (coerced) for ``family``.

    Required fields are absent. AFM families read ``AFMSimulationConfig``,
    sheet families ``SheetOnSheetSimulationConfig``; ``settings.*`` comes from
    ``GlobalSettings`` through either. Only block fields are listed: the
    excluded ones (``settings.database.*`` reads the environment) have no
    place in identity. Pinned by ``tests/fixtures/block_defaults.json``.
    """
    model = family_model(family)
    if model not in _DEFAULTS_CACHE:
        _DEFAULTS_CACHE[model] = {
            path: coerce(path, info.get_default(call_default_factory=True))
            for path, info in _leaves(model)
            if path in FIELD_MAP and not info.is_required()}
    return dict(_DEFAULTS_CACHE[model])


# --- the split's result, and checks on it ----------------------------------------

@dataclass(frozen=True)
class RunBlocks:
    """One run's parameters: canonical blocks (hashed) and derived values (not).

    Attributes:
        blocks: Block name -> canonical content ``{'kind', 'block_schema',
            'fields'}``.
        derived: Builder-computed values (``geometry.lat_c``, ``drive_k``, ...),
            flat, never hashed.
        family: The build family the blocks were canonicalised against.
    """
    blocks: Dict[str, dict]
    derived: dict
    family: str


class UnitMismatch(ValueError):
    """A recorded value disagrees with the unit its schema states."""


#: drive_method values whose drive has a spring (``simulation_base.drive_spring``).
SPRING_METHODS = frozenset({'smd', 'virtual_atom'})


#: The N/m -> eV/A^2 factor AFM builds used before ``b501bf5`` (a literal
#: ``/ 16.02`` in ``afm/base.py``; exact is NM_TO_EV_A2 = 16.02176565). Their
#: ``drive_k`` is 1.1e-4 relative off the exact constant, so the cross-check
#: accepts it for legacy AFM-family records only (:func:`_legacy_afm_spring`).
LEGACY_AFM_SPRING_FACTOR = 16.02

#: The provenance schema of the first builds with the exact AFM factor.
_EXACT_SPRING_SCHEMA = '2.2.0'


def _legacy_afm_spring(family: str, record: Optional[Mapping[str, Any]]) -> bool:
    """True for an AFM-family record built before ``b501bf5``: provenance
    ``schema_version`` < 2.2.0, or a ``run_uuid_stamped_at`` (a legacy run the
    stamper gave an identity). No record means a current build."""
    if family not in AFM_FAMILIES or not record:
        return False
    if record.get('run_uuid_stamped_at'):
        return True
    try:
        return version_key(record.get('schema_version', '1.0.0')) < \
            version_key(_EXACT_SPRING_SCHEMA)
    except (TypeError, ValueError):
        return False


def _spring_fallback(family: str) -> float:
    """The N/m a blank ``driving_spring`` falls back to in the family's builder:
    8.0 for AFM (``afm/base.py``), 50.0 for the sheet families
    (``sheetonsheet/base.py``, ``hetero.py``, ``flake_slide.py``). Every one of
    them converts with :data:`~src.data.models.NM_TO_EV_A2`."""
    if family in AFM_FAMILIES:
        return 8.0
    family_model(family)
    return 50.0


def check_units(blocks: RunBlocks, *, record: Optional[Mapping[str, Any]] = None) -> None:
    """Raise :class:`UnitMismatch` when ``derived.drive_k`` is not the requested
    ``general.driving_spring`` (N/m) in eV/A^2.

    Pins the convention that ``drive_k`` is eV/A^2: a drive spring recorded in
    N/m (or converted twice) fails the import. Runs without a spring, or
    without a recorded ``drive_k``, pass. ``record`` is the run's
    ``provenance.json``: a legacy AFM-family record may also match
    :data:`LEGACY_AFM_SPRING_FACTOR`; every other run uses only NM_TO_EV_A2.
    """
    defaults = defaults_for(blocks.family)
    fields = (blocks.blocks.get(_DRIVE) or {}).get('fields') or {}
    method = fields.get('settings.simulation.drive_method',
                        defaults.get('settings.simulation.drive_method'))
    drive_k = blocks.derived.get('drive_k')
    if method not in SPRING_METHODS or drive_k is None:
        return
    spring = fields.get('general.driving_spring', defaults.get('general.driving_spring'))
    factors = [NM_TO_EV_A2]
    if _legacy_afm_spring(blocks.family, record):
        factors.append(LEGACY_AFM_SPRING_FACTOR)
    candidates = [(spring or _spring_fallback(blocks.family)) / f for f in factors]
    tolerance = 1e-9 * max(1.0, abs(float(drive_k)))
    expected = candidates[0]
    if all(abs(float(drive_k) - c) > tolerance for c in candidates):
        raise UnitMismatch(
            f'derived.drive_k = {drive_k} {UNITS["derived.drive_k"]}, but '
            f'general.driving_spring = {spring} {UNITS["general.driving_spring"]} '
            f'gives {expected} for family {blocks.family!r} ({method})')


def numeric_paths() -> List[str]:
    """Every path whose value is numeric: FIELD_MAP and realization fields,
    ``derived.*`` and ``point.*``."""
    paths = [p for p in FIELD_MAP if is_numeric(ANNOTATIONS[p])]
    paths += [p for p in REALIZATION_FIELDS if is_numeric(ANNOTATIONS[p])]
    paths += [f'derived.{p}' for p in DERIVED_PATHS
              if p in ANNOTATIONS and is_numeric(ANNOTATIONS[p])]
    paths.append('derived.interlayer.g_meV')                 # Dict[str, Any] in the model
    paths += list(POINT_PATHS)
    return sorted(set(paths))


def missing_units() -> List[str]:
    """Numeric paths with no entry in :data:`UNITS`, sorted."""
    return sorted(p for p in numeric_paths() if p not in UNITS)


def schema_document(family: str) -> dict:
    """The schema as JSON: ``{'block_schema', 'family', 'fields': {path: {'kind',
    'unit', 'default'}}}``. ``default`` is None for a required field."""
    defaults = defaults_for(family)
    model_paths = set(leaf_field_paths(family_model(family)))
    fields: Dict[str, Dict[str, Any]] = {
        path: {'kind': kind, 'unit': UNITS.get(path), 'default': defaults.get(path)}
        for path, kind in FIELD_MAP.items() if path in model_paths}
    fields.update({path: {'kind': _REALIZATION, 'unit': UNITS.get(path), 'default': None}
                   for path in REALIZATION_FIELDS})
    fields.update({f'derived.{p}': {'kind': DERIVED, 'unit': UNITS.get(f'derived.{p}'),
                                    'default': None} for p in DERIVED_PATHS})
    fields.update({p: {'kind': POINT, 'unit': UNITS[p], 'default': None}
                   for p in POINT_PATHS})
    return {'block_schema': BLOCK_SCHEMA_VERSION, 'family': family, 'fields': fields}
