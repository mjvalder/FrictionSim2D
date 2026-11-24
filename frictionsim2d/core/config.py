"""Pydantic schemas for every simulation parameter, validated before a build.

Also holds the ``.ini``/YAML/JSON reader and writer pair that feeds them.
"""
import configparser
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, ClassVar, Literal, Mapping, Sequence, cast
import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .mlip import check_bound_stack, get_mlip, is_mlip
from .utils.paths import get_material_path, get_potential_path
from .utils.tokens import is_float

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ('.ini', '.yaml', '.yml', '.json')

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
    z_vacuum_margin: float = Field(
        default=8.0,
        ge=0.0,
        description=(
            "Extra periodic-z vacuum (Å) on top of the largest pair cutoff plus "
            "the neighbor skin. Every box is periodic in z, so the vacuum between "
            "the top atoms and the image of the bottom atoms must exceed the "
            "cutoff or the stack interacts with itself; the margin absorbs the "
            "atoms' motion during the run (thermal, load, tip travel)."
        ),
    )

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
    thermo: int = Field(default=100000, description="Steps between thermo output lines.")
    min_style: str = 'cg'
    minimization_command: str = 'minimize 1e-4 1e-8 1000000 1000000'
    neighbor_list: float = Field(default=0.3, description="Neighbor-list skin distance (Å).")
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
    halt_temperature_factor: float = Field(
        default=10.0,
        ge=0.0,
        description=(
            "Halt the current load point when the global temperature exceeds "
            "this multiple of the target temperature -- a blow-up that keeps "
            "all its atoms would otherwise spend the whole step budget. The "
            "rest of the sweep still runs. 0 disables the guard."
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
            "source; 'raw' (100 fs samples) and 'peratom' (per-atom computes, including "
            "the neighbour-list-rebuilding group/group) cost runtime and disk. "
            "Postprocessing reads all three tiers: the raw tier feeds the slip "
            "statistics and the peratom tier is plottable by column name. All on by "
            "default."
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

    @field_validator('optional_channels', mode='before')
    @classmethod
    def _coerce_optional_channels(cls, v: Any) -> Any:
        """Accept ``None``, an empty string, or a comma/whitespace-separated
        string (as a hand-edited ``.ini`` or an emptied GUI field produces),
        in addition to a real list/tuple, which passes through unchanged."""
        if v is None:
            return []
        if isinstance(v, str):
            if not v.strip():
                return []
            return [tok for tok in (t.strip() for t in v.replace(',', ' ').split()) if tok]
        return v

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
    d3_functional: str = Field(
        default='pbe',
        description="DFT functional whose D3 parameters are used by pair_style dispersion/d3."
    )
    d3_damping: Literal['original', 'bj', 'zerom', 'bjm'] = Field(
        default='bj',
        description=(
            "D3 damping name, emitted verbatim by "
            "frictionsim2d/core/mlip.py::d3_style_token; 'original' is zero-damping."
        )
    )
    d3_cutoff: float = Field(
        default=16.0,
        description="D3 pair cutoff in Angstrom."
    )
    d3_cn_cutoff: float = Field(
        default=8.0,
        description="D3 coordination-number cutoff in Angstrom."
    )
    model_config = {'validate_by_name': True}

class IntegritySettings(BaseModel):
    """Thresholds of the post-run integrity check (a faithful port of ML_taxonomy s48).

    Every threshold is ``Optional``; ``None`` turns that rule off. Bond lengths
    are each layer's median nearest-neighbour distance at the loaded state.
    """
    hole_bonds: Optional[float] = Field(
        default=1.5,
        description="Fail when the largest atom-free disk in either layer's xy "
                    "projection, end of the slide, is >= this many bond lengths")
    isolated_radius_bonds: Optional[float] = Field(
        default=1.5,
        description="Radius, in bond lengths, within which a sheet atom needs a "
                    "sheet neighbour to count as not isolated")
    isolated_atoms: Optional[int] = Field(
        default=2,
        description="Fail when >= this many sheet atoms have no neighbour within "
                    "isolated_radius_bonds (end of the slide)")
    ejected_atoms: Optional[int] = Field(
        default=2,
        description="Fail when >= this many sheet atoms lie more than ejected_bonds "
                    "bond lengths from their layer's median plane")
    ejected_bonds: Optional[float] = Field(
        default=8.0,
        description="Distance, in bond lengths, from the layer's median plane "
                    "beyond which an atom counts as ejected")
    heating_sigma: Optional[float] = Field(
        default=8.0,
        description="Fail when the peak sheet temperature is >= this many sigma_T "
                    "above the target, sigma_T = T sqrt(2 / 3N), N = sheet atoms")
    gap_rise: Optional[float] = Field(
        default=0.05,
        description="Fail when the interlayer gap rises >= this fraction above "
                    "its own starting value (bilayers)")
    collapse: bool = Field(
        default=True,
        description="Fail a run whose tip has collapsed into the sheet")
    frozen_std: Optional[float] = Field(
        default=1e-6,
        description="Fail when an output channel's standard deviation is below "
                    "this (a frozen channel)")
    gap_start_samples: int = Field(
        default=25, description="Samples averaged for the starting interlayer gap")
    sheet_heating_skip_samples: int = Field(
        default=5,
        description="Leading samples skipped before taking the peak sheet temperature")
    sheet_gap_end_samples: int = Field(
        default=50, description="Samples averaged for the end-of-slide sheet gap")
    margins: Dict[str, float] = Field(
        default_factory=dict,
        description="Rule name -> fraction of its threshold below which a "
                    "non-firing value quarantines the run")
    version: str = Field(
        default='1.0.0', description="Version of the integrity rule set")
    model_config = {'validate_by_name': True}


#: What a ``margins`` key may name: a rule or its measure key
#: (``postprocessing.integrity.MARGIN_KEYS`` is derived from the rules and pinned equal by a test).
_MARGIN_KEYS = frozenset({
    'non_finite', 'frozen', 'hole', 'isolated_atoms', 'isolated', 'ejected_atoms', 'ejected',
    'sheet_heating', 'T_sigma', 'gap_rise', 'collapse', 'gap_min_any'})


def _integrity_fields() -> set:
    return set(IntegritySettings.model_fields)


def integrity_overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    """``['hole_bonds=2', 'margins.hole=0.1']`` -> ``{'hole_bonds': 2, 'margins': {'hole': 0.1}}``.

    Values are YAML scalars; ``none`` turns a threshold off. ``margins.RULE`` sets one
    margin; every other key must be an :class:`IntegritySettings` field.

    Raises:
        ValueError: A pair without ``=``, or an unknown key.
    """
    out: Dict[str, Any] = {}
    for pair in pairs:
        key, sep, text = pair.partition('=')
        key = key.strip()
        if not sep or not key:
            raise ValueError(f'{pair!r} is not KEY=VALUE')
        value = None if text.strip().lower() == 'none' else yaml.safe_load(text)
        head, dot, rule = key.partition('.')
        if head == 'margins' and dot and rule:
            if rule not in _MARGIN_KEYS:
                raise ValueError(f'unknown integrity rule {rule!r} in {key!r}; known: '
                                 f'{", ".join(sorted(_MARGIN_KEYS))}')
            out.setdefault('margins', {})[rule] = value
        elif key != 'margins' and key in _integrity_fields():
            out[key] = value
        else:
            known = sorted(_integrity_fields() - {'margins'})
            raise ValueError(f'unknown integrity key {key!r}; known: {", ".join(known)}, '
                             'margins.RULE')
    return out


def merged_integrity(base: Mapping[str, Any], overrides: Optional[Mapping[str, Any]]
                     ) -> Dict[str, Any]:
    """``base`` (an ``IntegritySettings`` dump) with ``overrides`` applied
    (``margins`` merged key by key), re-validated.

    Raises:
        pydantic.ValidationError: An overridden value does not validate.
    """
    merged = {**base, **(overrides or {})}
    if overrides and 'margins' in overrides:
        merged['margins'] = {**(base.get('margins') or {}), **overrides['margins']}
    return IntegritySettings(**merged).model_dump(mode='json')


class GPUSettings(BaseModel):
    """GPU resources requested for a job.

    Rendered into the PBS ``select`` line as ``ngpus``/``gpu_type``. ``None``
    ``gpu_type`` means "whatever the queue's default accelerator is" and the
    token is then omitted entirely rather than emitted empty.
    """
    ngpus: int = 1
    gpu_type: Optional[str] = None


class FrameworkOverlay(BaseModel):
    """Per-MLIP-framework HPC overrides.

    MACE (symmetrix + Kokkos) and DeePMD (pinned mainline + deepmd-kit plugin)
    are two *incompatible* LAMMPS builds, so a campaign that mixes them must
    resolve a different binary — and often different modules and environment
    exports — per framework. Keyed by framework name (``hpc.frameworks``).
    """
    binary: Optional[str] = None
    modules: List[str] = Field(default_factory=list)
    env_lines: List[str] = Field(default_factory=list)
    conda_env: Optional[str] = Field(
        default=None,
        description=(
            "Conda prefix holding this framework's LAMMPS build. When set it "
            "replaces hpc.conda_env for this framework's job group only."
        ),
    )


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
    # Conda prefix whose bin/ and lib/ the job puts on PATH (e.g. $HOME/envs/lammps).
    conda_env: Optional[str] = None
    mpi_command: str = "mpirun"
    use_tmpdir: bool = True
    # Also write hpc/preflight.sh + the scheduler's submit wrapper.
    preflight: bool = False
    lammps_scripts: List[str] = Field(default_factory=lambda: [
        'system.in',
        'slide.in'
    ])
    # --- MLIP additions (all default to "nothing configured" so an untouched
    # settings.yaml renders byte-identical scripts to before). Clusters are
    # switched with --settings-file; there are no named profiles. ---
    env_lines: List[str] = Field(default_factory=list)
    gpu: Optional[GPUSettings] = None
    frameworks: Dict[str, FrameworkOverlay] = Field(
        default_factory=dict,
        description=(
            "Per-MLIP-framework overrides keyed by framework name (mace, deepmd). "
            "Applied per job group by apply_framework_overlay()."
        ),
    )
    lmp_binary: Optional[str] = Field(
        default=None,
        description=(
            "LAMMPS binary for this job group. Set directly, or populated by "
            "apply_framework_overlay() from a FrameworkOverlay's binary; None "
            "means the plain 'lmp' on PATH."
        ),
    )


def apply_framework_overlay(hpc: HPCSettings,
                            framework: Optional[str]) -> HPCSettings:
    """Return a copy of ``hpc`` with ``hpc.frameworks[framework]`` merged in.

    MACE and DeePMD are two incompatible LAMMPS builds, so one campaign can
    need a different binary, modules and environment per framework. Merge
    rules: the overlay's ``modules`` and ``env_lines`` are appended to the
    base ones (a framework needs the cluster's modules too); a set ``binary``
    replaces ``lmp_binary``; a set ``conda_env`` replaces ``conda_env``.

    Args:
        hpc: base settings, never mutated.
        framework: MLIP framework name of the job group, or ``None`` for a
            classical group. A framework with no overlay entry is a no-op.

    Returns:
        A deep copy of ``hpc``, merged when an overlay applies.
    """
    merged = hpc.model_copy(deep=True)
    overlay = merged.frameworks.get(framework) if framework else None
    if overlay is None:
        return merged
    if overlay.binary:
        merged.lmp_binary = overlay.binary
    if overlay.conda_env:
        merged.conda_env = overlay.conda_env
    if overlay.modules:
        merged.modules = list(merged.modules or []) + list(overlay.modules)
    if overlay.env_lines:
        merged.env_lines = list(merged.env_lines or []) + list(overlay.env_lines)
    return merged


class AiidaSettings(BaseModel):
    """AiiDA import settings.

    AiiDA is import-only: simulations never touch it, results are ingested
    afterwards. `enabled` and `lammps_code_label` belonged to the old
    run-time submission path and are gone (an old settings.yaml that still
    names them loads, with a warning).
    """
    #: The profile every profile-taking function falls back to when its own
    #: `profile: Optional[str] = None` argument is omitted. `None` here means
    #: "AiiDA's own default profile".
    profile: Optional[str] = None
    #: Computer used when `machine` resolves to 'localhost'.
    computer_label: str = 'localhost'
    #: Files larger than this are recorded by sha256 + path instead of stored.
    max_stored_file_mb: int = 20
    #: Leading fraction of each run's friction trace discarded as transient
    #: before the COF statistics are taken.
    skip_fraction: float = 0.2
    #: The machine a run tree was produced on; `None` lets ingest resolve it.
    machine: Optional[str] = None


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
    integrity: IntegritySettings = Field(default_factory=IntegritySettings)

#: Credential-named keys, matched case-insensitively at any depth; rename a field
#: rather than carve it out.
_SECRET_KEY_NAMES = frozenset({
    'password', 'passwd', 'api_key', 'apikey', 'secret', 'token',
})
#: Suffixed forms of the same names, e.g. `db_password`, `remote_api_key`.
_SECRET_KEY_SUFFIXES = ('_password', '_secret', '_token', '_api_key')


def is_secret_key(key: Any) -> bool:
    """True when `key` (one mapping key of parsed YAML/JSON) names a
    credential, by exact name or suffix, case-insensitively."""
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    return lowered in _SECRET_KEY_NAMES or lowered.endswith(_SECRET_KEY_SUFFIXES)


def redact_secrets_mapping(data: Any) -> Any:
    """A copy of a raw settings/config structure with every credential blanked.

    Copies every mapping and sequence, blanks the value of any key that
    :func:`is_secret_key` at any depth, and never mutates the input.
    """
    if isinstance(data, dict):
        return {key: ('' if is_secret_key(key) else redact_secrets_mapping(value))
                for key, value in data.items()}
    if isinstance(data, list):
        return [redact_secrets_mapping(item) for item in data]
    return data


def redact_secrets(settings: GlobalSettings) -> GlobalSettings:
    """A copy of `settings` with every credential blanked; `settings` is not mutated."""
    redacted = redact_secrets_mapping(settings.model_dump())
    return cast(GlobalSettings, GlobalSettings.model_validate(redacted))


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

    @field_validator('pot_type', mode='after')
    @classmethod
    def check_pot_type(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Refuse a pair style LAMMPS will not recognise.

        ``None`` passes (a flake without its own pot_type). The import is deferred
        to keep ``core -> potentials`` one-way.
        """
        if v is None:
            return v
        from ..potentials.utils import validate_pot_type  # pylint: disable=import-outside-toplevel,cyclic-import
        return validate_pot_type(v, source=f'{info.field_name}')

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

    @model_validator(mode='after')
    def validate_mlip_model_extension(self) -> 'ComponentConfig':
        """A MACE/DeePMD ``pot_path`` must carry that framework's extension.

        The two frameworks ship *incompatible* serialized formats and neither
        errors cleanly when handed the other's file — LAMMPS either segfaults
        or silently mis-loads. Catching the mismatch here is the only cheap
        place. Classical potentials are untouched.

        The accepted extensions live in ``MLIPSpec.file_extensions`` and are
        **not** restated here, because MACE's changed: it is driven through
        wcwitt/symmetrix, which consumes a ``.json`` extracted from a torch
        checkpoint by ``symmetrix_extract_mace``, not the ACEsuit fork's
        ``.model-lammps.pt``. DeePMD takes a frozen ``.pb``/``.pth``.
        """
        pot_type = self.pot_type
        if not pot_type or not is_mlip(pot_type) or not self.pot_path:
            return self

        spec = get_mlip(pot_type)
        if not str(self.pot_path).lower().endswith(
                tuple(ext.lower() for ext in spec.file_extensions)):
            raise ValueError(
                f"pot_type = '{spec.name}' needs a model file whose extension is "
                f"one of {', '.join(spec.file_extensions)}; got "
                f"'{self.pot_path}'. A model of the other framework will not "
                f"load."
            )
        return self


class _EmptyAmorphDefault(BaseModel):
    """Mixin: an empty ``amorph`` reads as the default ``'c'``."""

    @field_validator('amorph', mode='before', check_fields=False)
    @classmethod
    def handle_empty_amorph(cls, v):
        """Convert None to default 'c' value."""
        return 'c' if v is None else v


class TipConfig(_EmptyAmorphDefault, ComponentConfig):
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

class SubstrateConfig(_EmptyAmorphDefault, ComponentConfig):
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
#: Config keys struck by the MLIP redesign. Pydantic *ignores* unknown keys
#: by default, which for these would silently change a run's physics, so
#: :meth:`SheetConfig.reject_removed_mlip_keys` rejects them by name with a
#: pointer at documentation/mlip.md.
REMOVED_SHEET_KEYS = {
    'mlip_interlayer': (
        "It only ever selected between a partitioned cell ('lj') and full "
        "coverage ('native'), and the partition is unavailable for BOTH "
        "frameworks: pair_deepmd rejects NULL type maps ('Pair hybrid "
        "sub-style deepmd is not used') and refuses type-range pair_coeff "
        "('deepmd requires that the scale should be set to all atom types', "
        "pair_deepmd.cpp:816); pair_mace segfaults on NULL (ACEsuit #1314). "
        "Full-cell is now the ONLY topology, so the key carries no "
        "information -- delete the line. Interlayer coupling comes from the "
        "model itself; use 'dispersion = d3' for a vdW correction."
    ),
}


class SheetConfig(ComponentConfig):
    """Sheet configuration parameters."""
    x: Union[float, List[float]]
    y: Union[float, List[float]]
    layers: List[int]
    stack_type: str = 'AA'
    lat_c: Optional[float] = None
    interlayer_g: Optional[float] = Field(
        default=None,
        description="Gaussian strength g (meV) for interlayer_style = lj_g; "
                    "overrides the built-in table.",
    )
    materials_list: Optional[str] = Field(
        default=None,
        description=(
            "Path to a .txt of material names, one per line, each becoming its own "
            "simulation. expand_config_sweeps consumes and strips it before the model "
            "is built; declared here only so it counts as a known key."
        ),
    )
    dispersion: Literal['none', 'd3'] = Field(
        default='none',
        description=(
            "Optional Grimme-D3 dispersion overlay (pair_style dispersion/d3). "
            "Full-cell MLIP coverage only: stock dispersion/d3 needs a real "
            "element for every atom type, so a partial cell would double-count "
            "dispersion over the SW/LJ pairs."
        ),
    )
    allow_unbound: bool = Field(
        default=False,
        description=(
            "Escape hatch for running a deliberately unbound multilayer MLIP "
            "stack (dispersion = 'none' on a framework whose bare model has no "
            "interlayer binding). This is ONLY for no-D3 control arms, where "
            "demonstrating the failure is the point. The layers will drift "
            "apart and interlayer friction/adhesion numbers will be physically "
            "meaningless -- never set this on a production run."
        ),
    )

    @model_validator(mode='before')
    @classmethod
    def reject_removed_mlip_keys(cls, data):
        """Fail loudly on keys the MLIP redesign removed."""
        if not isinstance(data, dict):
            return data
        for key, reason in REMOVED_SHEET_KEYS.items():
            if key in data:
                raise ValueError(
                    f"'{key}' was removed and is no longer accepted (got "
                    f"{data[key]!r}). {reason} See "
                    f"documentation/mlip.md §2."
                )
        return data

    @model_validator(mode='after')
    def validate_mlip_capability_rules(self) -> 'SheetConfig':
        """Enforce the config-layer rules for MLIP sheets.

        These come straight from the binding capability matrix
        (documentation/mlip.md §1-§2);
        each message names the rule it enforces so a rejected config is
        self-diagnosing. Cell-level rules that need to see *sibling* components
        (no classical component alongside an MLIP; no multi-component full cell)
        cannot be decided here and live in ``mlip.validate_mlip_cell`` instead.
        """
        mlip = bool(self.pot_type) and is_mlip(self.pot_type)

        if not mlip:
            if self.dispersion != 'none':
                raise ValueError(
                    f"dispersion = '{self.dispersion}' requires an MLIP pot_type "
                    f"(mace or deepmd); got '{self.pot_type}'. The dispersion/d3 "
                    f"overlay is only defined for full-cell MLIP coverage."
                )
            return self

        if self.lat_c is None:
            raise ValueError(
                f"pot_type = '{self.pot_type}' requires an explicit lat_c. The "
                f"interlayer distance is normally found by minimizing with the "
                f"sheet's own potential, but MLIP stacking runs under a "
                f"pair_style zero surrogate (no MLIP-enabled LAMMPS is assumed on "
                f"the generation machine), so no minimization can supply it. Use "
                f"the value from scripts/mlip_bilayer_benchmark.py."
            )

        layers = self.layers if isinstance(self.layers, list) else [self.layers]
        check_bound_stack(self.pot_type, self.dispersion, self.allow_unbound,
                          max(int(n) for n in layers) if layers else 1)
        return self


class FlakeConfig(ComponentConfig):
    """A finite patch of 2D material placed between tip and sheet.

    The flake is cut from a supercell of its own material into a triangle, square
    or hexagon. AFM indentation always tethers its corner atoms in xy (free in
    z) to hold that shape, and releases them before sliding -- that path does
    not read ``corner_hold`` and is unaffected by it. A sheet-on-sheet flake
    slide (``FlakeSlideSimulation``) reads ``corner_hold`` to decide whether the
    same tether applies while the flake is driven; default off, since holding
    every corner in place stops the flake translating at all. Turn it on only
    for static/indent-like studies that need the flake kept rigid in-plane.

    Attributes:
        shape: Flake outline ('triangle', 'square', 'hexagon').
        edge_length: Characteristic size (Å). Triangle/square: side length;
            hexagon: circumradius (center-to-vertex distance).
        rotation_deg: In-plane rotation of the shape (degrees, CCW).
        x, y: Supercell footprint (Å) to carve the flake from. Defaults to
            roughly twice ``edge_length`` so the shape always fits.
        center_x, center_y: Optional flake center override (box coords). When
            omitted the flake is centered under the tip.
        corner_hold: Sheet-on-sheet flake slide only. Tether the flake's
            corner atoms (within ``corner_radius`` of each vertex) in xy with
            ``corner_spring_k`` while it is driven, pinning the flake in place.
            Default off. Not read by AFM, which always applies its own
            corner-hold during indentation.
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
    corner_hold: bool = Field(
        default=False,
        description=(
            "Sheet-on-sheet flake slide only: tether the flake's corner atoms "
            "in xy while it is driven, pinning it in place. Default off. AFM "
            "indentation does not read this field and always holds its "
            "corners."
        ),
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
    interlayer_style: Literal['lj', 'lj_g', 'ilp', 'kc'] = Field(
        default='lj',
        description=(
            "Interaction between the layers of the sheet stack: 'lj' (the run's "
            "normal LJ), 'lj_g' (LJ + Jiang-Park Gaussian), 'ilp' or 'kc'. "
            "Tip-sheet and substrate-sheet always stay LJ."
        ),
    )
    interlayer_file: Optional[str] = Field(
        default=None,
        description="ILP/KC parameter file; None picks a bundled default by element.",
    )
    interlayer_comparison: bool = Field(
        default=False,
        description=(
            "Measure friction by the bottom-layer reaction force under every "
            "interlayer style (lj included, instead of group/group), so a sweep "
            "over styles is like for like."
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
        if self.twist_angle is not None and not 0.0 <= self.twist_angle < 60.0:
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
    :mod:`frictionsim2d.builders.afm.indent`):

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


def _numbered_keys(folded: Dict[str, Any], prefix: str) -> List[str]:
    """Keys ``<prefix>N`` of ``folded``, in numeric order of N."""
    return sorted(
        (k for k in folded if isinstance(k, str) and k.startswith(prefix)),
        key=lambda key: (
            int(key.split('-', maxsplit=1)[1])
            if key.split('-', maxsplit=1)[1].isdigit() else 10**9
        ),
    )


def _fold_sheet_sections(data: Any) -> Any:
    """Fold [2D-1], [2D-2], ... into ``sheets`` and [Flake-N] into ``flake_sheets``.

    A lone ``[2D]`` is wrapped as a one-element list, keeping the single-material
    path intact. ``2D`` stays required either way: given only numbered sections,
    the first is mirrored onto it. With no ``[Flake-N]`` the ``flake_sheets`` key
    is left alone.
    """
    if not isinstance(data, dict):
        return data

    folded = dict(data)
    two_d_keys = _numbered_keys(folded, '2D-')
    if two_d_keys:
        folded['sheets'] = [folded[k] for k in two_d_keys]
        folded.setdefault('2D', folded[two_d_keys[0]])
    elif '2D' in folded:
        folded['sheets'] = [folded['2D']]

    flake_keys = _numbered_keys(folded, 'Flake-')
    if flake_keys:
        folded['flake_sheets'] = [folded[k] for k in flake_keys]
    return folded


def _first(value: Any) -> float:
    """A sheet ``x``/``y`` as a float; a list (a sweep) contributes its first entry."""
    return float(value[0] if isinstance(value, list) else value)


def _check_mlip_interlayer_choice(config: Any) -> None:
    """Refuse a non-``lj`` interlayer style or the comparison mode under an MLIP.

    An MLIP cell is full-cell only (one instance spans every atom type), so
    there is no interlayer pair slot left for ILP/KC/LJ+Gaussian, and the only
    vdW lever is ``dispersion = d3`` (documentation/mlip.md). Accepting the
    choice would be a silent no-op or a broken deck.

    Raises:
        ValueError: If any component of ``config`` is an MLIP while
            ``general.interlayer_style != 'lj'`` or
            ``general.interlayer_comparison`` is set.
    """
    general = config.general
    style = general.interlayer_style
    comparison = general.interlayer_comparison
    if style == 'lj' and not comparison:
        return
    components = [getattr(config, name, None) for name in ('tip', 'sub', 'flake', 'sheet')]
    components += list(getattr(config, 'sheets', None) or [])
    components += list(getattr(config, 'flake_sheets', None) or [])
    mlip_types = sorted({c.pot_type for c in components
                         if c is not None and c.pot_type and is_mlip(c.pot_type)})
    if not mlip_types:
        return
    what = (f"interlayer_style = '{style}'" if style != 'lj'
            else "interlayer_comparison = true")
    raise ValueError(
        f"{what} cannot be combined with an MLIP component "
        f"(pot_type {', '.join(mlip_types)}): an MLIP cell is full-cell only, so "
        "there is no interlayer pair slot to replace. Use interlayer_style = lj "
        "and dispersion = d3 for the vdW term (documentation/mlip.md).")


def _check_sheet_sections_agree(config: Any, include_flake_sheets: bool) -> None:
    """Require the sheet sections of an MLIP hetero or flake cell to agree on D3.

    ``dispersion`` and ``allow_unbound`` are properties of the whole cell, but the
    builders read them from the first sheet only, so a disagreeing section would
    be silently ignored (or wrongly refused). Only MLIP cells are checked, since
    both settings are inert for classical potentials, and only the sheet lists
    the config's builders actually read (``flake_sheets`` is read by the flake
    slide only).

    Raises:
        ValueError: Naming the sections that disagree.
    """
    sheets = list(getattr(config, 'sheets', None) or [])
    flake_sheets = list(getattr(config, 'flake_sheets', None) or []) if include_flake_sheets else []
    if len(sheets) < 2 and not flake_sheets:
        return
    if len(sheets) >= 2:
        labelled = [(f"[2D-{i}]", sh) for i, sh in enumerate(sheets, 1)]
    else:
        labelled = [("[2D]/[2D-1]", config.sheet)]
    labelled += [(f"[Flake-{i}]", sh) for i, sh in enumerate(flake_sheets, 1)]
    if not any(sh.pot_type and is_mlip(sh.pot_type) for _, sh in labelled):
        return
    for field in ('dispersion', 'allow_unbound'):
        values = {label: getattr(sh, field) for label, sh in labelled}
        if len(set(values.values())) > 1:
            listed = ', '.join(f"{label} {field} = {value}" for label, value in values.items())
            raise ValueError(
                f"{field} is a property of the whole cell, so every sheet section "
                f"must set it identically; got {listed}. See documentation/mlip.md.")


class _InterlayerChecked(BaseModel):
    """Mixin: reject interlayer settings that cannot produce a valid run."""

    #: Whether this config's builders read ``flake_sheets`` (the flake slide does).
    _READS_FLAKE_SHEETS: ClassVar[bool] = False

    @model_validator(mode='after')
    def _check_interlayer(self, info: ValidationInfo):
        """Refuse a non-lj interlayer style under an MLIP, then check the style itself."""
        from ..potentials.interlayer import check_interlayer_choice  # pylint: disable=import-outside-toplevel,cyclic-import
        _check_mlip_interlayer_choice(self)
        base_dir = info.context.get('base_dir') if isinstance(info.context, dict) else None
        check_interlayer_choice(self, base_dir)
        return self

    @model_validator(mode='after')
    def _check_sheet_sections_agree(self):
        """Refuse hetero/flake sheet sections that disagree on dispersion or allow_unbound."""
        _check_sheet_sections_agree(self, self._READS_FLAKE_SHEETS)
        return self


class AFMSimulationConfig(_InterlayerChecked):
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
        """Fold numbered ``[2D-N]``/``[Flake-N]`` sections into lists."""
        return _fold_sheet_sections(data)

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
        sub = self.sub
        if sub is not None:
            amorph = sub.amorph  # pylint: disable=no-member  # pydantic FieldInfo FP
            match = sub.match  # pylint: disable=no-member  # pydantic FieldInfo FP
            if (not self.general.finite_sheet and amorph != 'a'
                    and not (amorph == 'c' and match)):
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
        sub = self.sub

        if sub.x is None or sub.y is None:  # pylint: disable=no-member  # pydantic FieldInfo FP
            raise ValueError(
                "Finite-sheet AFM requires substrate x and y to be set in the [sub] section."
            )

        sheet_x = _first(self.sheet.x)  # pylint: disable=no-member  # pydantic FieldInfo FP
        sheet_y = _first(self.sheet.y)  # pylint: disable=no-member  # pydantic FieldInfo FP
        sub_x = float(sub.x)  # pylint: disable=no-member  # pydantic FieldInfo FP
        sub_y = float(sub.y)  # pylint: disable=no-member  # pydantic FieldInfo FP

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

    @model_validator(mode='after')
    def validate_tip_sheet_footprint(self) -> 'AFMSimulationConfig':
        """Reject a periodic AFM sheet too small for the tip diameter.

        The footprint must be >= 2x the tip diameter (``4 * tip.r``), else the tip
        meets its own periodic image. Exempt: a hetero stack (checked by
        ``compute_replication_factors``), a PES scan, and an indent run, which
        only presses down and gets a 1.5x margin (``3 * tip.r``).
        """
        if len(self.sheets) >= 2 or self.pes is not None:
            return self

        tip_r = float(self.tip.r)  # pylint: disable=no-member  # pydantic FieldInfo FP
        sheet_x = _first(self.sheet.x)  # pylint: disable=no-member  # pydantic FieldInfo FP
        sheet_y = _first(self.sheet.y)  # pylint: disable=no-member  # pydantic FieldInfo FP
        is_indent = self.indent is not None
        margin_factor = 3.0 if is_indent else 4.0
        min_footprint = margin_factor * tip_r

        if sheet_x < min_footprint or sheet_y < min_footprint:
            margin_label = "1.5x" if is_indent else "2x"
            raise ValueError(
                f"AFM sheet footprint ({sheet_x} x {sheet_y}) is smaller than {margin_label} "
                f"the tip diameter ({min_footprint}, from [tip] r={tip_r}): the tip would "
                "interact with its own periodic image, corrupting the recorded forces. "
                "Enlarge [2D] x/y, shrink [tip] r, or add a second [2D-N] material to route "
                "to the hetero builder (which replicates the cell automatically)."
            )
        return self


class SheetOnSheetSimulationConfig(_InterlayerChecked):
    """Master configuration object for a Sheet-on-Sheet simulation run."""
    _READS_FLAKE_SHEETS: ClassVar[bool] = True
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
        """Fold numbered ``[2D-N]``/``[Flake-N]`` sections into lists."""
        return _fold_sheet_sections(data)


# --- Helper Functions ---

def _global_settings_path() -> Path:
    """The per-user settings path, under ``$XDG_CONFIG_HOME`` or ``~/.config``."""
    xdg_config_home = os.environ.get('XDG_CONFIG_HOME')
    base = Path(xdg_config_home) if xdg_config_home else (Path.home() / '.config')
    return base / 'FrictionSim2D' / 'settings.yaml'


def _settings_paths_in_precedence_order(
    settings_file: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Candidate settings files, highest precedence first.

    An explicit ``settings_file`` for this run beats the semi-permanent
    ``~/.config/FrictionSim2D/settings.yaml``.
    """
    paths: List[Path] = []

    if settings_file:
        paths.append(Path(settings_file).expanduser())

    paths.append(_global_settings_path())
    return paths


#: `AiidaSettings` keys removed when AiiDA became import-only.
_REMOVED_AIIDA_KEYS = ('enabled', 'lammps_code_label')


def _warn_dropped_settings_keys(path: Path, data: Mapping[str, Any]) -> None:
    """Log ONE warning naming every removed key a settings file still carries."""
    dropped = []
    if 'database' in data:
        dropped.append(
            "the database block (the central database is gone; the AiiDA store "
            "is the only database)")
    aiida_block = data.get('aiida')
    if isinstance(aiida_block, dict):
        dropped.extend(f"aiida.{key}" for key in _REMOVED_AIIDA_KEYS
                       if key in aiida_block)
    if dropped:
        logger.warning("settings file %s carries keys that no longer exist and "
                       "are being ignored: %s", path, "; ".join(dropped))


def _settings_candidates(settings_file: Optional[Union[str, Path]] = None):
    """Yield ``(path, data)`` for each existing settings file, highest precedence first.

    ``data`` is the parsed mapping (``{}`` for an empty file), or ``None`` when
    the file cannot be read; that failure is logged here.
    """
    for path in _settings_paths_in_precedence_order(settings_file=settings_file):
        try:
            if not path.is_file():
                continue
            with path.open('r', encoding='utf-8') as f:
                yield path, yaml.safe_load(f) or {}
        except (FileNotFoundError, OSError, yaml.YAMLError) as exc:
            logger.warning("skipping settings file %s: %s", path, exc)
            yield path, None


def load_settings(settings_file: Optional[Union[str, Path]] = None) -> GlobalSettings:
    """Load the first readable settings file onto the hardcoded defaults.

    Candidates come from :func:`_settings_paths_in_precedence_order`; with none
    of them present, :class:`GlobalSettings` defaults are returned unchanged.
    """
    for path, data in _settings_candidates(settings_file):
        if data:
            _warn_dropped_settings_keys(path, data)
            return cast(GlobalSettings, GlobalSettings.model_validate(data))
    return cast(GlobalSettings, GlobalSettings())


def settings_origin(settings_file: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """The file ``load_settings`` would use, or ``None`` (for ``settings show --origin``).

    An unreadable file is still reported, so callers can diagnose it; an empty
    one is skipped, as ``load_settings`` skips it.
    """
    for path, data in _settings_candidates(settings_file):
        if data is None or data:
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
            raise ValueError(
                f"Unsupported configuration file format: {ext}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )

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
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
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
                        elif is_float(item):
                            parsed_items.append(float(item))
                        else:
                            parsed_items.append(item)
                    params[section][key] = parsed_items
            elif value.isdigit():
                params[section][key] = int(value)
            elif is_float(value):
                params[section][key] = float(value)
            else:
                params[section][key] = value
    return params
