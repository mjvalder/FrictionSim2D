"""Pydantic schemas for every simulation parameter, validated before a build.

Also holds the ``.ini``/YAML/JSON reader and writer pair that feeds them.
"""
import configparser
import json
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Literal, Mapping, cast
import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .utils.paths import get_material_path, get_potential_path
from .utils.tokens import _is_float

SUPPORTED_WRITE_FORMATS = ('.ini', '.yaml', '.yml', '.json')

# --- Internal Settings ---

class GeometrySettings(BaseModel):
    """Geometry settings for tip and substrate positioning."""
    tip_reduction_factor: float = 2.25
    rigid_tip: bool = False
    tip_start_clearance: float = Field(
        default=20.0,
        description=(
            "Extra height (Å) above the contact gap, keeping the tip clear of the "
            "surface during equilibration; the indentation script then lowers its "
            "lowest atom onto the measured gap."
        ),
    )
    lat_c_default: float = 6.0
    finite_sheet_edge_width: float = Field(
        default=4.0,
        description="Edge band width (Å) for finite-sheet runs; typically 1-2 atoms.",
    )
    finite_sheet_afm_edge_mode: Literal['none', 'spring'] = 'none'
    finite_sheet_edge_spring_k: float = Field(
        default=10.0,
        description=(
            "Spring constant (eV/Å²) tethering finite-sheet edge atoms to their "
            "initial positions, in 'spring' edge mode."
        ),
    )
    finite_sheet_sheetonsheet_edge_mode: Literal['none', 'rigid'] = 'none'

class ThermostatSettings(BaseModel):
    """Thermostat and time integration settings."""
    type: Literal['langevin', 'nose-hoover'] = 'langevin'
    time_integration: Literal['verlet', 'respa', 'nvt', 'nve'] = Field(
        default='nve', alias='time_int_type')
    langevin_boundaries: Dict[str, Dict[str, List[float]]] = Field(
        default_factory=lambda: {
            'tip': {'fix': [3.0, 0.0], 'thermo': [6.0, 3.0]},
            'sub': {'fix': [0.0, 0.3], 'thermo': [0.3, 0.6]}
        })
    model_config = {'validate_by_name': True}

class SimulationSettings(BaseModel):
    """Simulation run parameters."""
    timestep: float = Field(default=0.001, description="Timestep in picoseconds")
    thermo: int = 100000
    min_style: str = 'cg'
    minimization_command: str = 'minimize 1e-4 1e-8 1000000 1000000'
    neighbor_list: float = 0.3
    neigh_modify_command: str = 'neigh_modify every 1 delay 0 check yes'
    slide_run_steps: int = 500000
    drive_method: Literal['smd', 'fix_move', 'virtual_atom'] = 'virtual_atom'
    constraint_mode: Literal['atom_bonds', 'com_spring', 'none'] = 'none'
    sheet_hold_steps: int = Field(
        default=20000,
        description=(
            "Constant-pressure compression hold before sheet sliding, the sheet's "
            "analogue of tip penetration. Logs interlayer separation at the "
            "shearing interface (layer_{n//2+1}/layer_{n//2})."
        ),
    )
    ramp_max_iterations: int = Field(
        default=40,
        description=(
            "Cap on the post-ramp settle loop in system_init, so a seized or "
            "punched-through contact cannot spin forever. On exhaustion the run "
            "records RAMP_STATUS converged=0 and continues."
        ),
    )


class QuenchSettings(BaseModel):
    """Quenching parameters for amorphous material generation."""
    run_local: bool = True
    n_procs: int = 16
    quench_slab_dims: List[int] = Field(default_factory=lambda: [200, 200, 50])
    quench_rate: float = 1e12
    melt_temp: float = Field(default=2500.0, alias='quench_melt_temp')
    quench_temp: float = Field(default=300.0, alias='quench_target_temp')
    timestep: float = 0.002
    melt_steps: int = 50000
    equilibrate_steps: int = 20000
    allow_generate: bool = Field(
        default=True,
        description=(
            "May a missing or undersized amorphous structure be melt-quenched into "
            "existence? False makes make_amorphous raise instead — for interactive "
            "callers (the GUI preview) that cannot spend minutes in mpiexec or write "
            "into the packaged materials directory."
        ),
    )
    model_config = {'validate_by_name': True}

class OutputSettings(BaseModel):
    """Output and dump settings."""
    dump: Dict[str, bool] = Field(
        default_factory=lambda: {'system_init': True, 'slide': True})
    dump_frequency: Dict[str, int] = Field(
        default_factory=lambda: {'system_init': 1000, 'slide': 10000})
    results_frequency: int = 1000
    channels: Dict[str, bool] = Field(
        default_factory=lambda: {'window': True, 'raw': True, 'peratom': True},
        description=(
            "Which observable sampling tiers get a fix ave/time. 'window' is the COF "
            "source and the only tier postprocessing ingests; 'raw' (100 fs samples) "
            "and 'peratom' (per-atom computes, including the neighbour-list-rebuilding "
            "group/group) cost runtime and disk. All on by default."
        ),
    )
    optional_channels: List[str] = Field(
        default_factory=list,
        description=(
            "Names of opt-in observable channels to enable, beyond the default "
            "set. Channels declared opt_in=True in the registry cost measurable "
            "runtime and are emitted only when named here. A name that matches "
            "no registry channel is a build error, not a silent no-op."
        ),
    )

class PotentialSettings(BaseModel):
    """Settings for interatomic potentials."""
    lj_cutoff: float = Field(default=11.0, alias='LJ_cutoff')
    lj_type: str = Field(default='LJ_base', alias='LJ_type')
    reaxff_safezone: float = Field(
        default=1.2,
        description="Memory safety multiplier for ReaxFF neighbor lists"
    )
    reaxff_mincap: int = Field(
        default=50,
        description="Minimum array capacity for ReaxFF internals"
    )
    model_config = {'validate_by_name': True}

class HPCSettings(BaseModel):
    """HPC cluster and job submission settings."""
    scheduler_type: Literal['pbs', 'slurm'] = 'pbs'
    queue: Optional[str] = None
    partition: Optional[str] = None
    account: str = ''
    hpc_host: Optional[str] = None
    hpc_home: Optional[str] = None
    log_dir: Optional[str] = None
    scratch_dir: Optional[str] = "$TMPDIR"
    num_nodes: int = 1
    num_cpus: int = 32
    memory_gb: int = 62
    walltime_hours: int = 20
    max_array_size: int = 300
    modules: Optional[List[str]] = Field(default_factory=lambda: None)
    mpi_command: str = "mpirun"
    use_tmpdir: bool = True
    lammps_scripts: List[str] = Field(default_factory=lambda: [
        'system.in',
        'slide.in'
    ])


class AiidaSettings(BaseModel):
    """AiiDA-specific workflow settings."""
    enabled: bool = False
    lammps_code_label: str = 'lammps@my_hpc'
    postprocess_code_label: str = 'python@my_hpc'
    postprocess_script_path: str = ''
    create_provenance: bool = True
    auto_import_results: bool = False
    hpc_mode: Literal['local', 'remote', 'offline'] = 'offline'

    # AiiDA remote computer configuration (optional, only for 'aiida setup')
    computer_label: Optional[str] = 'localhost'
    transport: Literal['local', 'ssh'] = 'local'
    hostname: Optional[str] = None
    workdir: Optional[str] = None
    username: Optional[str] = None
    ssh_port: int = 22
    key_filename: Optional[str] = None


class DatabaseProfileSettings(BaseModel):
    """Connection parameters for a single database profile."""
    host: str = 'localhost'
    port: int = 5432
    dbname: str = 'frictionsim2ddb'
    user: str = ''
    password: str = ''
    api_key: str = ''


class DatabaseSettings(BaseModel):
    """Database connection and staging pipeline configuration."""
    active_profile: str = 'local'
    local: DatabaseProfileSettings = Field(default_factory=DatabaseProfileSettings)
    central: DatabaseProfileSettings = Field(
        default_factory=lambda: DatabaseProfileSettings(
            host=os.environ.get('FRICTION_CENTRAL_DB_HOST', ''),
            port=int(os.environ.get('FRICTION_CENTRAL_DB_PORT', '5432')),
            dbname=os.environ.get('FRICTION_CENTRAL_DB_NAME', 'frictionsim2ddb'),
            user=os.environ.get('FRICTION_CENTRAL_DB_USER', ''),
            password=os.environ.get('FRICTION_CENTRAL_DB_PASSWORD', ''),
            api_key=os.environ.get('FRICTION_DB_API_KEY', ''),
        )
    )
    skip_fraction: float = 0.2
    api_url: str = os.environ.get('FRICTION_CENTRAL_API_URL', 'http://localhost:8000')
    api_host: str = '0.0.0.0'
    api_port: int = 8000


class GlobalSettings(BaseModel):
    """Represents the full structure of settings.yaml with hardcoded defaults."""
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    thermostat: ThermostatSettings = Field(default_factory=ThermostatSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    quench: QuenchSettings = Field(default_factory=QuenchSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    potential: PotentialSettings = Field(default_factory=PotentialSettings)
    hpc: HPCSettings = Field(default_factory=HPCSettings)
    aiida: AiidaSettings = Field(default_factory=AiidaSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

# --- User Input Settings (From .ini files) ---

class ComponentConfig(BaseModel):
    """Base configuration for any material component.

    Attributes:
        mat: Material identifier.
        pot_type: Potential type ('sw', 'tersoff', ...).
        pot_path, cif_path: Potential and CIF structure files.
    """
    mat: str
    pot_type: str
    pot_path: str
    cif_path: str

    @field_validator('pot_path', 'cif_path', mode='after')
    @classmethod
    def resolve_path(cls, v: str, info: ValidationInfo) -> str:
        """Resolve a config path value against the materials/potentials library.

        A ``base_dir`` passed through the validation context
        (``model_validate(..., context={'base_dir': ...})``) resolves relative
        paths, so ``examples/potentials/...`` works from any working directory.
        Unresolvable values are returned unchanged.
        """
        if v is None:
            return v
        base_dir = None
        if isinstance(info.context, dict):
            base_dir = info.context.get('base_dir')

        path = Path(v)
        if path.exists():
            return str(path)

        resolver = get_potential_path if info.field_name == 'pot_path' else get_material_path
        resolved = resolver(v) if base_dir is None else resolver(v, base_dir)

        return str(resolved) if resolved.exists() else v

class TipConfig(ComponentConfig):
    """Tip configuration parameters."""
    r: float = Field(..., description="Tip radius in Angstroms")
    amorph: Literal['c', 'a'] = Field('c', description="'c' for crystalline, 'a' for amorphous")
    dspring: Optional[float] = Field(
        0.0,
        description=(
            "Explicit viscous damping, passed to LAMMPS as damp_ev = dspring/0.016. "
            "Any set value — including the 0.0 default, i.e. no damping — wins over "
            "damping_ratio; blank it to fall back to damping_ratio."
        ),
    )
    damping_ratio: Optional[float] = Field(
        0.0,
        description=(
            "Target damping ratio (zeta) of critical for the spring-driven tip. LAMMPS "
            "derives the viscous coefficient from the driving spring constant and the "
            "real tip mass, so the ratio holds for any tip or spring. Read only when "
            "dspring is blank."
        ),
    )
    tip_x: Optional[float] = Field(
        default=None,
        description="Optional absolute AFM tip x-position in box coordinates.",
    )
    tip_y: Optional[float] = Field(
        default=None,
        description="Optional absolute AFM tip y-position in box coordinates.",
    )
    tip_x_offset: float = Field(
        default=0.0,
        description="AFM tip x-offset from center applied only when tip_x is not explicitly set.",
    )
    tip_y_offset: float = Field(
        default=0.0,
        description="AFM tip y-offset from center applied only when tip_y is not explicitly set.",
    )

    @field_validator('amorph', mode='before')
    @classmethod
    def handle_empty_amorph(cls, v):
        """Convert None to default 'c' value."""
        return 'c' if v is None else v

class SubstrateConfig(ComponentConfig):
    """Substrate configuration parameters."""
    thickness: float
    x: Optional[float] = Field(
        default=None,
        description="Finite-sheet substrate size in x (same units as sheet x).",
    )
    y: Optional[float] = Field(
        default=None,
        description="Finite-sheet substrate size in y (same units as sheet y).",
    )
    amorph: Literal['c', 'a'] = Field('c', description="'c' for crystalline, 'a' for amorphous")
    match: bool = Field(
        False,
        description=(
            "With amorph='c', strain the substrate surface cell onto the fixed hetero "
            "box (D2b) so it tiles the periodic cell with no seam."
        ),
    )
    match_strain_tol: float = Field(
        default=0.02, ge=0.0, le=1.0,
        description=(
            "Base strain tolerance for the fixed-reference substrate match; the accept "
            "bound is 2× this, and residual strain beyond it fails loud."
        ),
    )
    @field_validator('amorph', mode='before')
    @classmethod
    def handle_empty_amorph(cls, v):
        """Convert None to default 'c' value."""
        return 'c' if v is None else v

class SheetConfig(ComponentConfig):
    """Sheet configuration parameters."""
    x: Union[float, List[float]]
    y: Union[float, List[float]]
    layers: List[int]
    stack_type: str = 'AA'
    lat_c: Optional[float] = None
    materials_list: Optional[str] = Field(
        default=None,
        description=(
            "Path to a .txt of material names, one per line, each becoming its own "
            "simulation. expand_config_sweeps consumes and strips it before the model "
            "is built; declared here only so it counts as a known key."
        ),
    )

class FlakeConfig(ComponentConfig):
    """A finite patch of 2D material placed between tip and sheet.

    The flake is cut from a supercell of its own material into a triangle, square
    or hexagon. AFM indentation tethers its corner atoms in xy (free in z) to hold
    that shape, and releases them before sliding.

    Attributes:
        shape: Flake outline ('triangle', 'square', 'hexagon').
        edge_length: Characteristic size (Å). Triangle/square: side length;
            hexagon: circumradius (center-to-vertex distance).
        rotation_deg: In-plane rotation of the shape (degrees, CCW).
        x, y: Supercell footprint (Å) to carve the flake from. Defaults to
            roughly twice ``edge_length`` so the shape always fits.
        center_x, center_y: Optional flake center override (box coords). When
            omitted the flake is centered under the tip.
    """
    # Optional because a hetero flake takes its two materials from
    # [Flake-1]/[Flake-2] (``flake_sheets``), leaving [Flake] a pure shape spec.
    mat: Optional[str] = None
    pot_type: Optional[str] = None
    pot_path: Optional[str] = None
    cif_path: Optional[str] = None
    shape: Literal['triangle', 'square', 'hexagon'] = 'hexagon'
    edge_length: float = Field(..., description="Characteristic flake size in Angstroms")
    rotation_deg: float = 0.0
    x: Optional[float] = None
    y: Optional[float] = None
    corner_radius: float = Field(
        default=5.0,
        description="Capture radius (Å) around each vertex for corner-atom selection.",
    )
    corner_spring_k: float = Field(
        default=10.0,
        description="xy spring constant (eV/Å²) tethering corner atoms during indentation.",
    )
    center_x: Optional[float] = None
    center_y: Optional[float] = None

    @model_validator(mode='after')
    def default_supercell_size(self) -> 'FlakeConfig':
        """Size the carving supercell to comfortably contain the shape."""
        margin = max(2.0 * self.corner_radius, 10.0)
        # 2x is safe for both readings of edge_length: hexagon circumradius
        # (half-width) and triangle/square side.
        span = 2.0 * self.edge_length + margin
        if self.x is None:
            self.x = span
        if self.y is None:
            self.y = span
        return self


class GeneralConfig(BaseModel):
    """General simulation parameters."""
    temp: float = 300.0
    force: Optional[Union[float, List[float]]] = None
    pressure: Optional[Union[float, List[float]]] = None
    scan_angle: Optional[Union[float, List[float]]] = 0.0
    scan_angle_force: Optional[Union[float, List[float]]] = Field(
        None,
        description=(
            "Which force/pressure values the scan_angle list applies to; one value "
            "or a list. Omitted: every angle is applied to every force/pressure."
        )
    )
    scan_speed: Optional[Union[float, List[float]]] = 2.0
    outer_loop: Optional[Literal['pressure', 'scan_speed', 'force', 'scan_angle']] = Field(
        None,
        description=(
            "Parameter expanded into one slide_*.in per value instead of a single "
            "in-script loop; omitted gives the single-script behaviour. "
            "'pressure'/'scan_speed' are sheet-on-sheet, 'force' is AFM "
            "(slide_f<load>N.in per load), 'scan_angle' works for either. Every "
            "emitted script shares the one system.in indentation/pre-slide phase."
        )
    )
    bond_spring: Optional[float] = Field(
        80.0, description="Spring constant for harmonically bonded sheets.")
    driving_spring: Optional[float] = Field(
        50, description="Driving spring constant for the virtual-atom method.")
    finite_sheet: bool = Field(
        default=False,
        description="Enable finite-sheet AFM geometry for this run.",
    )
    hetero_stacking: Literal['grouped', 'alternating'] = Field(
        default='grouped',
        description=(
            "Layer ordering in a multi-material stack: 'grouped' finishes one "
            "material before the next, 'alternating' interleaves them."
        ),
    )
    twist_angle: Optional[float] = Field(
        default=None,
        description=(
            "Relative twist (deg) between the two stacked layers for a D3 "
            "commensurate-moiré build; None leaves every twist path inert. Range "
            "[0, 60), the hexagonal irreducible wedge. One [2D] material gives a "
            "same-material moiré, two a twisted heterostructure. Commensurate angles "
            "are discrete, so the ACHIEVED angle is usually the nearest one."
        ),
    )
    twist_angle_tol: float = Field(
        default=2.0,
        description=(
            "Max |achieved - target| twist (deg) before the moiré search fails loud."
        ),
    )
    twist_strain_tol: float = Field(
        default=0.02,
        description=(
            "Max residual per-material strain when closing the twisted moiré "
            "approximant onto its shared box."
        ),
    )
    twist_max_supercell: int = Field(
        default=12,
        description=(
            "Cap on the moiré supercell repeat per axis, bounding cell explosion. "
            "Exceeded, the search fails loud with the nearest achievable angle and "
            "its atom-count cost."
        ),
    )

    @model_validator(mode='after')
    def _validate_twist_angle(self) -> 'GeneralConfig':
        """Restrict twist_angle to the hexagonal irreducible wedge [0, 60)."""
        if self.twist_angle is not None and not (0.0 <= self.twist_angle < 60.0):
            raise ValueError(
                f"general.twist_angle must be in [0, 60) deg (the hexagonal "
                f"irreducible wedge); got {self.twist_angle}."
            )
        return self


class PESConfig(BaseModel):
    """Static lateral potential-energy-surface scan parameters.

    A PES scan replaces the dynamic sliding run with a cheap energy-vs-lateral-
    position map over one surface unit cell. Both flavours share this model, and
    each one's fields are inert for the other:

        - ``pes_sheet`` (``builders.sheetonsheet.pes_sheet``) grid-scans a frozen
          bilayer's top layer, relaxing only its out-of-plane coordinate.
        - ``pes_tip`` (``builders.afm.pes_tip``) grid-scans the AFM tip over a
          relaxing sheet + substrate, recording lateral force as well as energy.
    """
    grid_n: int = Field(
        default=12,
        ge=2,
        description="N for the N×N lateral grid over one surface unit cell.",
    )
    grid_n_refine: Optional[int] = Field(
        default=None,
        ge=2,
        description=(
            "Finer grid for a convergence check; emits a second scan script at this "
            "resolution alongside the grid_n one."
        ),
    )
    z_relax: bool = Field(
        default=True,
        description=(
            "Sheet scan only — relax top-layer z at each grid point (relaxed PES). "
            "False keeps the layers rigid (rigid PES)."
        ),
    )
    tip_load: Optional[float] = Field(
        default=5.0,
        description=(
            "Tip scan only — target grid-mean normal load (nN). The scan itself is "
            "constant-height, so the campaign runner calibrates each material's "
            "fz(gap) curve and scans at the gap whose grid-mean load hits this target. "
            "Calibration (scripts/run_pes_tip_bilayer/gap_load_calibration/) put the "
            "bilayer's stable window at ~2-12 nN, hence the 5 nN default. Null/0 → the "
            "fixed contact gap, an adhesion descriptor."
        ),
    )
    scan_cells: int = Field(
        default=1,
        ge=1,
        description=(
            "Tip scan only — the scan spans M×M surface unit cells at grid_n points "
            "*per cell*, so the emitted grid is M·grid_n square over M·a × M·b. M=1 "
            "(default) changes nothing. M>1 is a representativeness diagnostic: over a "
            "disordered backing one cell is a random patch of a non-periodic PES, and "
            "only a multi-cell scan measures the cell-to-cell spread (the error bar) "
            "and tells real corrugation from substrate noise. Needs a sheet footprint "
            "wide enough that the tip clears its own periodic image over the M-cell "
            "travel; the builder warns if not."
        ),
    )
    eval_mode: Literal['minimize', 'md'] = Field(
        default='minimize',
        description=(
            "Tip freestanding scan only — how each grid point is evaluated. "
            "'minimize': static energy minimization (0 K, cheapest). 'md': a short "
            "NVE + Langevin run of md_steps against the layer_2 thermostat band, "
            "time-averaging energy/force into a finite-T potential of mean force."
        ),
    )
    md_steps: int = Field(
        default=2000,
        ge=1,
        description="Steps per grid point when eval_mode = 'md'; ignored otherwise.",
    )


class IndentConfig(BaseModel):
    """AFM indentation parameters — damped hold + pull-off retract, no sliding.

    After the usual AFM build and indentation ramp to each load in
    ``[general] force``, an indent run does one hold+retract per load (see
    :mod:`src.builders.afm.indent`, and ``ML analysis/docs/POKE_SIM_RECIPE.md``
    outside this repo):

        1. a short *damped* finite-T hold of the rigid, laterally-fixed tip at
           the target load. Its thermally-averaged penetration depth
           (``tip_z − sheet_com_z``) is the vertical-compliance descriptor a
           lateral PES scan cannot see.
        2. a quasi-static T=0 retract in small z-steps, through zero load and
           past the pull-off instability to full detachment. Its force–distance
           curve gives pull-off force, work of adhesion and contact stiffness.

    Subsumes the former separate poke (hold-only) and adhesion (retract-only)
    types. Sizing constraints, on top of the field descriptions:

        - ``z_step`` must resolve the pull-off instability, and
          ``n_steps × z_step`` must clear the adhesive tail;
        - **a fresh seed per run is mandatory**: reusing a campaign seed replays
          the campaign contact trajectory and re-imports the same-run optimism
          the verification exists to exclude.
    """
    hold_steps: int = Field(
        default=10000,
        ge=1000,
        description=(
            "Damped-hold length in steps at the target load; 10 000 = 10 ps, ~10 "
            "samples at the 1 ps campaign cadence, over which penetration is stable."
        ),
    )
    z_step: float = Field(
        default=0.2,
        gt=0.0,
        description="Retract increment per step in Å (recipe: 0.1–0.2 Å).",
    )
    n_steps: int = Field(
        default=100,
        ge=1,
        description="Retract steps; n_steps × z_step is the total distance (20 Å).",
    )
    seed: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Base velocity seed. Each (material, layer) derives its own seed from it, "
            "recorded in the manifest. Omit to draw a fresh base at build time; never "
            "reuse a campaign seed."
        ),
    )


class LatticeMatchConfig(BaseModel):
    """Lattice-matching parameters for heterostructure interfaces."""
    method: Literal['zur_mcgill'] = 'zur_mcgill'
    strain_tol: float = Field(default=0.02, ge=0.0, le=1.0)
    max_supercell: int = Field(default=10, ge=1)
    area_tol: float = Field(default=0.10, ge=0.0, le=1.0)


def _fold_flake_sheet_sections(folded: Dict[str, Any]) -> Dict[str, Any]:
    """Fold [Flake-1], [Flake-2], ... into ``flake_sheets``, in numeric order.

    These carry the two dedicated hetero-flake materials, mirroring the ``2D-N``
    sheet folding. With no ``[Flake-N]`` present the key is left alone (an empty
    list), leaving the single-material path — a lone ``[Flake]`` shape spec —
    unchanged.
    """
    flake_keys = sorted(
        (k for k in folded if isinstance(k, str) and k.startswith('Flake-')),
        key=lambda key: (
            int(key.split('-', maxsplit=1)[1])
            if key.split('-', maxsplit=1)[1].isdigit() else 10**9
        ),
    )
    if flake_keys:
        folded['flake_sheets'] = [folded[k] for k in flake_keys]
    return folded


class AFMSimulationConfig(BaseModel):
    """Master configuration object for an AFM simulation run."""
    general: GeneralConfig
    tip: TipConfig
    sub: Optional[SubstrateConfig] = Field(default=None, alias='sub')
    sheet: SheetConfig = Field(..., alias='2D')
    sheets: List[SheetConfig] = Field(
        default_factory=list,
        description=(
            "Per-material sheet stack, folded from [2D-1], [2D-2], ... by "
            "_fold_sheet_sections (or [2D] alone, wrapped). Two or more entries route "
            "`run afm` to the periodic hetero AFM builder (D2)."
        ),
    )
    flake: Optional[FlakeConfig] = Field(default=None, alias='flake')
    flake_sheets: List[SheetConfig] = Field(
        default_factory=list,
        description=(
            "Dedicated hetero-flake materials, folded from [Flake-1], [Flake-2]. "
            "Two entries carve the flake from a matched hetero bilayer (B2); empty "
            "leaves the single-material path alone. [flake] holds the shape spec."
        ),
    )
    pes: Optional[PESConfig] = Field(default=None, alias='pes')
    indent: Optional[IndentConfig] = Field(default=None, alias='indent')
    lattice_match: Optional[LatticeMatchConfig] = Field(default=None, alias='lattice_match')
    lj_override: Dict[str, Any] = Field(default_factory=dict, alias='lj_override')
    settings: GlobalSettings

    @model_validator(mode='before')
    @classmethod
    def _fold_sheet_sections(cls, data: Any) -> Any:
        """Fold [2D-1], [2D-2], ... into ``sheets``, in numeric order.

        A lone ``[2D]`` is wrapped as a one-element list instead, keeping the
        single-material path intact. ``2D``/``sheet`` stays required either way:
        given only numbered sections, the first is mirrored onto ``2D`` so
        single-sheet code paths keep working.
        """
        if not isinstance(data, dict):
            return data

        folded = dict(data)
        two_d_keys = sorted(
            (k for k in folded if isinstance(k, str) and k.startswith('2D-')),
            key=lambda key: (
                int(key.split('-', maxsplit=1)[1])
                if key.split('-', maxsplit=1)[1].isdigit() else 10**9
            ),
        )

        if two_d_keys:
            folded['sheets'] = [folded[k] for k in two_d_keys]
            folded.setdefault('2D', folded[two_d_keys[0]])
        elif '2D' in folded:
            folded['sheets'] = [folded['2D']]

        return _fold_flake_sheet_sections(folded)

    @model_validator(mode='after')
    def validate_flake_material(self) -> 'AFMSimulationConfig':
        """Require an AFM [flake] to name its own material.

        ``build_flake_component`` and ``PotentialManager.register_component`` both
        read [flake]'s own mat/cif_path/pot_path/pot_type, neither falls back, and
        ``[Flake-N]`` does not help (only the flake slide reads ``flake_sheets``).
        AFM-only, because the sheet-on-sheet flake slide *does* fall back to [2D]
        via ``FlakeSlideSimulation._filled_flake_config`` — which is why the fields
        stay optional on the shared ``FlakeConfig``. Unchecked, the config validates
        and then dies inside ``build_monolayer`` on a ``None`` path.
        """
        if self.flake is None:
            return self
        missing = [name for name in ('mat', 'cif_path', 'pot_path', 'pot_type')
                   if getattr(self.flake, name) is None]
        if missing:
            raise ValueError(
                f"AFM [flake] is missing {', '.join(missing)}. The AFM flake is built "
                "from its own material and has no [2D] fallback (that is a "
                "sheet-on-sheet flake-slide feature), so mat, cif_path, pot_path and "
                "pot_type must all be set on [flake]."
            )
        return self

    @model_validator(mode='after')
    def validate_non_finite_sheet_substrate(self) -> 'AFMSimulationConfig':
        """Require an amorphous substrate under a non-finite AFM sheet.

        Two exemptions: a freestanding run (no [sub]) has nothing to constrain, and
        a lattice-matched crystalline substrate (``amorph='c'`` with ``match``) is
        strained to tile the periodic cell seamlessly (D2b).
        """
        if (self.sub is not None and not self.general.finite_sheet
                and self.sub.amorph != 'a'
                and not (self.sub.amorph == 'c' and self.sub.match)):
            raise ValueError(
                "Non-finite-sheet AFM requires an amorphous substrate "
                "([sub] amorph = a), or a lattice-matched crystalline substrate "
                "([sub] amorph = c, match = true)."
            )
        return self

    @model_validator(mode='after')
    def validate_finite_sheet_geometry(self) -> 'AFMSimulationConfig':
        """Validate finite-sheet substrate sizing against the sheet footprint."""
        if not self.general.finite_sheet:
            return self

        if self.sub is None:
            raise ValueError(
                "Finite-sheet AFM requires a [sub] section: a finite sheet must rest "
                "on a substrate. For a substrate-free scan, use the freestanding tip "
                "PES (omit [sub]) with finite_sheet = false."
            )

        if self.sub.x is None or self.sub.y is None:
            raise ValueError(
                "Finite-sheet AFM requires substrate x and y to be set in the [sub] section."
            )

        sheet_x = float(self.sheet.x[0] if isinstance(self.sheet.x, list) else self.sheet.x)
        sheet_y = float(self.sheet.y[0] if isinstance(self.sheet.y, list) else self.sheet.y)
        sub_x = float(self.sub.x)
        sub_y = float(self.sub.y)

        if sub_x < sheet_x:
            raise ValueError(
                f"Finite-sheet substrate x ({sub_x}) cannot be smaller than sheet x ({sheet_x})."
            )
        if sub_y < sheet_y:
            raise ValueError(
                f"Finite-sheet substrate y ({sub_y}) cannot be smaller than sheet y ({sheet_y})."
            )

        lj_cutoff = float(self.settings.potential.lj_cutoff)
        margin_x = sub_x - sheet_x
        margin_y = sub_y - sheet_y
        if margin_x <= lj_cutoff:
            raise ValueError(
                "Finite-sheet AFM requires (substrate_x - sheet_x) to be greater than LJ cutoff "
                f"({lj_cutoff}). Got {margin_x}."
            )
        if margin_y <= lj_cutoff:
            raise ValueError(
                "Finite-sheet AFM requires (substrate_y - sheet_y) to be greater than LJ cutoff "
                f"({lj_cutoff}). Got {margin_y}."
            )

        return self

class SheetOnSheetSimulationConfig(BaseModel):
    """Master configuration object for a Sheet-on-Sheet simulation run."""
    general: GeneralConfig
    sheet: SheetConfig = Field(..., alias='2D')
    sheets: List[SheetConfig] = Field(
        default_factory=list,
        description=(
            "Per-material sheet stack, folded from [2D-1], [2D-2], ... by "
            "_fold_sheet_sections (or [2D] alone, wrapped)."
        ),
    )
    flake: Optional[FlakeConfig] = Field(default=None, alias='Flake')
    flake_sheets: List[SheetConfig] = Field(
        default_factory=list,
        description=(
            "Dedicated hetero-flake materials, folded from [Flake-1], [Flake-2]. "
            "Two entries carve the flake from a matched hetero bilayer (B2); empty "
            "leaves the single-material path alone. [Flake] holds the shape spec."
        ),
    )
    pes: Optional[PESConfig] = Field(default=None, alias='pes')
    lattice_match: Optional[LatticeMatchConfig] = Field(default=None, alias='lattice_match')
    lj_override: Dict[str, Any] = Field(default_factory=dict, alias='lj_override')
    settings: GlobalSettings

    @model_validator(mode='before')
    @classmethod
    def _fold_sheet_sections(cls, data: Any) -> Any:
        """Fold [2D-1], [2D-2], ... into ``sheets``, in numeric order.

        Mirrors :meth:`AFMSimulationConfig._fold_sheet_sections`: a lone ``[2D]``
        is wrapped as a one-element list, and given only numbered sections the
        first is mirrored back onto the still-required ``2D`` field.
        """
        if not isinstance(data, dict):
            return data

        folded = dict(data)
        two_d_keys = sorted(
            (k for k in folded if isinstance(k, str) and k.startswith('2D-')),
            key=lambda key: (
                int(key.split('-', maxsplit=1)[1])
                if key.split('-', maxsplit=1)[1].isdigit() else 10**9
            ),
        )

        if two_d_keys:
            folded['sheets'] = [folded[k] for k in two_d_keys]
            folded.setdefault('2D', folded[two_d_keys[0]])
        elif '2D' in folded:
            folded['sheets'] = [folded['2D']]

        return _fold_flake_sheet_sections(folded)

# --- Helper Functions ---

def _global_settings_path() -> Path:
    """The per-user settings path, under ``$XDG_CONFIG_HOME`` or ``~/.config``."""
    xdg_config_home = os.environ.get('XDG_CONFIG_HOME')
    base = Path(xdg_config_home) if xdg_config_home else (Path.home() / '.config')
    return base / 'FrictionSim2D' / 'settings.yaml'


def _settings_paths_in_precedence_order(settings_file: Optional[Union[str, Path]] = None) -> List[Path]:
    """Candidate settings files, highest precedence first.

    An explicit ``settings_file`` for this run beats the semi-permanent
    ``~/.config/FrictionSim2D/settings.yaml``.
    """
    paths: List[Path] = []

    if settings_file:
        paths.append(Path(settings_file).expanduser())

    paths.append(_global_settings_path())
    return paths


def load_settings(settings_file: Optional[Union[str, Path]] = None) -> GlobalSettings:
    """Load the first readable settings file onto the hardcoded defaults.

    Candidates come from :func:`_settings_paths_in_precedence_order`; with none
    of them present, :class:`GlobalSettings` defaults are returned unchanged.
    """
    for path in _settings_paths_in_precedence_order(settings_file=settings_file):
        try:
            if not path.is_file():
                continue
            with path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            if data:
                return cast(GlobalSettings, GlobalSettings.model_validate(data))
        except (FileNotFoundError, OSError, yaml.YAMLError):
            continue

    return cast(GlobalSettings, GlobalSettings())


def settings_origin(settings_file: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """The settings file currently in effect, or ``None`` (for ``settings show --origin``)."""
    for path in _settings_paths_in_precedence_order(settings_file=settings_file):
        if path.is_file():
            return path
    return None


def parse_config(config_source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load a config from any supported source into a dict Pydantic can validate.

    Takes a path to an ``.ini``, ``.yaml``/``.yml`` or ``.json`` file, or an
    already-built dict (from the CLI or a UI form), which passes through unchanged.

    Raises:
        ValueError: If the file extension is not supported.
        TypeError: If the source is neither a path nor a dict.
    """
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        ext = path.suffix.lower()

        if ext == '.ini':
            return read_config(path)

        if ext in ('.yaml', '.yml'):
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}

        if ext == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

        else:
            raise ValueError(f"Unsupported configuration file format: {ext}. Supported formats: .ini, .yaml, .yml, .json")

    elif isinstance(config_source, dict):
        return config_source

    else:
        raise TypeError(f"Unsupported configuration source type: {type(config_source)}")

#: Mapping of ``section -> key -> comment`` used to annotate ``.ini`` output.
CommentMap = Mapping[str, Mapping[str, str]]

def format_ini_value(value: Any) -> str:
    """Render a Python value as an ``.ini`` right-hand side.

    The exact inverse of the value grammar documented on :func:`read_config` —
    change one and you must change the other.

    Raises:
        ValueError: If the value contains ``#``, which the reader would strip.
    """
    if value is None:
        return ''

    if isinstance(value, bool):
        # The reader has no bool rule: these come back as strings for Pydantic to
        # coerce, and 'True'/'False' is what the existing configs use.
        return 'True' if value else 'False'

    if isinstance(value, (list, tuple)):
        rendered = json.dumps(list(value))
    elif isinstance(value, float):
        rendered = repr(value)
        # repr() already gives an integral float its '.0'; this guards oddities
        # (a float subclass) that would otherwise read back as an int.
        if rendered.isdigit():
            rendered += '.0'
    elif isinstance(value, int):
        rendered = str(value)
    else:
        rendered = str(value)

    if '#' in rendered:
        raise ValueError(
            f"Config values cannot contain '#': {rendered!r}. The reader treats "
            "it as the start of an inline comment and would truncate the value."
        )
    return rendered

def _render_ini(config: Mapping[str, Mapping[str, Any]],
                comments: Optional[CommentMap] = None,
                header: Optional[str] = None) -> str:
    """Render a section mapping as ``.ini`` text.

    Hand-written rather than ``configparser.write`` so inline comments can be
    attached and ``%`` needs no interpolation escaping.
    """
    comments = comments or {}
    lines = []

    if header:
        lines.extend(f"# {line}" if line else "#" for line in header.splitlines())
        lines.append("")

    for section, values in config.items():
        lines.append(f"[{section}]")
        section_comments = comments.get(section, {})
        for key, value in values.items():
            rendered = format_ini_value(value)
            comment = section_comments.get(key)
            if comment:
                lines.append(f"{key} = {rendered} # {comment}")
            else:
                lines.append(f"{key} = {rendered}")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"

def write_config(config: Mapping[str, Any],
                 path: Union[str, Path],
                 *,
                 comments: Optional[CommentMap] = None,
                 header: Optional[str] = None) -> Path:
    """Write a config mapping (as ``parse_config`` returns it) to ``path``.

    The suffix picks the format. ``comments`` (``section -> key -> comment``) and
    ``header`` annotate ``.ini`` output only; the other writers ignore them.

    Raises:
        ValueError: If the extension is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == '.ini':
        text = _render_ini(config, comments=comments, header=header)
    elif ext in ('.yaml', '.yml'):
        text = yaml.safe_dump(_to_plain(config), sort_keys=False)
    elif ext == '.json':
        text = json.dumps(_to_plain(config), indent=2) + "\n"
    else:
        raise ValueError(
            f"Unsupported configuration file format: {ext}. "
            f"Supported formats: {', '.join(SUPPORTED_WRITE_FORMATS)}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path

def _to_plain(value: Any) -> Any:
    """Convert nested mappings to plain dicts for the yaml/json writers."""
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value

def _remove_inline_comments(config: configparser.ConfigParser) -> configparser.ConfigParser:
    """Strip everything after a ``#`` from every value, in place.

    Returns the same object it was given, for chaining.
    """
    for section in config.sections():
        for item in config.items(section):
            config.set(section, item[0], item[1].split("#")[0].strip())
    return config

def read_config(filepath: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Read an ``.ini`` file into a ``section -> key -> value`` mapping.

    The value grammar is implicit and easy to violate; :func:`format_ini_value`
    is its exact inverse:

    * an empty right-hand side reads back as ``None``, so never write ``"None"``;
    * a trailing ``]`` is what triggers list detection, so write lists as JSON;
    * ``isdigit()`` decides int-vs-float, so a float needs a decimal point or an
      exponent. ``"-5".isdigit()`` is False, so a *negative* integer reads back as
      a float — long-standing, reproduced by the writer, pinned by the round-trip
      test;
    * a ``#`` starts an inline comment, so the writer rejects values holding one
      rather than let the next read truncate them.
    """

    # interpolation=None: the default BasicInterpolation raises on a bare '%'.
    # Nothing here uses '%(name)s' substitution, so the feature bought only
    # crashes on values a user could reasonably type.
    config = configparser.ConfigParser(interpolation=None)
    config.read(filepath)
    config = _remove_inline_comments(config)

    params = {}
    for section in config.sections():
        params[section] = {}
        for key in config[section]:
            value = config.get(section, key)

            if value == '':
                params[section][key] = None
                continue

            if value.endswith(']'):
                try:
                    params[section][key] = json.loads(value)
                except json.JSONDecodeError:
                    cleaned = value.strip('[]')
                    items = [item.strip() for item in cleaned.split(',')]
                    parsed_items = []
                    for item in items:
                        if item.isdigit():
                            parsed_items.append(int(item))
                        elif _is_float(item):
                            parsed_items.append(float(item))
                        else:
                            parsed_items.append(item)
                    params[section][key] = parsed_items
            elif value.isdigit():
                params[section][key] = int(value)
            elif _is_float(value):
                params[section][key] = float(value)
            else:
                params[section][key] = value
    return params
