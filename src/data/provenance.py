"""Per-run provenance record.

One ``provenance.json`` per run, written by the builder at build time and
enriched post-run from the LAMMPS log (``parse_log_extras``). A missing
required field fails the *build*, not the analysis: the record is a Pydantic
model and builders must construct it completely.
"""
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel

logger = logging.getLogger(__name__)

PROVENANCE_SCHEMA_VERSION = '2.1.0'

#: Which part of the built system a MaterialProvenance entry describes.
MATERIAL_ROLES = ('sheet', 'flake', 'tip', 'substrate')

# Anchored: LAMMPS echoes the ``print "# RAMP_STATUS ..."`` command into the log
# too, and that line (starting with ``print``) must not count as a result.
_RAMP_RE = re.compile(
    r'^# RAMP_STATUS converged=(\d) r=([\d.eE+-]+) iterations=(\d+)'
    r'(?: load=([\d.eE+-]+))?', re.M)
_HOLD_FILE_RE = re.compile(r'^hold_(-?[\d.]+)N\.txt$')
_ZREF_RE = re.compile(r'# UNLOADED_REF layer=(\d+) z=([\d.eE+-]+)')


class MaterialProvenance(BaseModel):
    """One material of the built system and the potential it was given.

    A hetero run has two or more 'sheet' entries; the flake, tip and substrate
    each get one. Replaces the single pot_path/pot_type/pot_file_sha256 triple
    of schema 1.x, which silently recorded only the primary sheet.
    """
    name: str
    role: str
    pot_path: Optional[str] = None
    pot_type: Optional[str] = None
    pot_resolution_route: Optional[str] = None
    pot_file_sha256: Optional[str] = None


class BuiltGeometry(BaseModel):
    """Geometry the builder computed and, until now, discarded (lengths in Angstrom)."""
    lat_c: Optional[float] = None
    lat_c_source: Optional[
        Literal['config', 'minimised', 'default', 'not_applicable']] = None
    unit_cell: Optional[Dict[str, float]] = None   # xlo..zhi of the ortho cell
    cell_a: Optional[float] = None                 # unit_cell xhi - xlo
    cell_b: Optional[float] = None                 # unit_cell yhi - ylo
    sheet_dims: Optional[Dict[str, float]] = None  # supercell box xlo..zhi
    atoms_per_layer: Optional[int] = None
    layer_thickness: Optional[float] = None        # z-extent of one layer's atoms
    tip_r: Optional[float] = None


class ProvenanceRecord(BaseModel):
    """Everything needed to reproduce and audit one run (spec §5 F5)."""
    schema_version: str = PROVENANCE_SCHEMA_VERSION

    # what this run is (spec section 3 F1: model = where it lives, family = its schema)
    family: str
    model: str

    # every material of the built system, replacing the single pot_* triple
    materials: List[MaterialProvenance] = []

    # the stacking twist, not the slide direction. The only thing that
    # distinguishes two otherwise-identical hetero runs, and nothing recorded
    # it before: `scan_angle` is where the slide goes, `twist_angle` is how the
    # stack was built.
    twist_angle: Optional[float] = None

    # seeds — drawn in the builder (not in the template) so they land here
    seed_thermal_tip: Optional[int]
    seed_thermal_sub: Optional[int]
    seed_velocity: Optional[int]
    seed_sheet_center: Optional[int] = None    # sheet runs only

    # the hidden constant of the 2026 campaign (spec §2.4): every AFM run
    # shares one cached amorphous-substrate realization
    substrate_realization_id: Optional[str]

    # damping: previously unrecorded free parameters of the friction result
    damping_ratio: Optional[float]
    damp_ev: Optional[float]

    # build identity
    git_commit: str
    git_dirty: bool
    config_hash: str
    lammps_version: Optional[str] = None       # enriched post-run

    # geometry / instrumentation identity
    layer_type_map: Dict[str, List[int]]
    unloaded_reference_z: Optional[Dict[str, float]] = None  # enriched post-run
    contact_cutoff: float
    cutoff_source: str
    thermostat_scheme: str
    tip_group_sizes: Optional[Dict[str, int]] = None

    # run identity / roles
    run_role: Literal['label', 'probe', 'both_forbidden'] = 'label'
    load_requested: Optional[float]
    load_achieved: Optional[float] = None      # enriched post-run
    pressure_requested: Optional[float] = None
    pressure_achieved: Optional[float] = None

    # ramp / settle diagnostics (enriched post-run from RAMP_STATUS)
    ramp_converged: Optional[bool] = None
    ramp_r: Optional[float] = None
    ramp_iterations: Optional[int] = None

    # drive spring (builder-computed: unit conversion and fallback live in Python)
    drive_k: Optional[float] = None             # eV/A^2
    drive_rest_length: Optional[float] = None   # A

    geometry: Optional[BuiltGeometry] = None

    # per-load diagnostics, keyed f'{load:g}' (enriched post-run)
    ramp_by_load: Optional[Dict[str, Dict[str, Any]]] = None
    load_achieved_by_load: Optional[Dict[str, float]] = None

    # execution (enriched post-run)
    wall_time_s: Optional[float] = None
    n_procs: Optional[int] = None
    atoms_per_proc: Optional[float] = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def material_provenance(component: Any, role: str) -> MaterialProvenance:
    """One :class:`MaterialProvenance` from a component config.

    The single construction site for the ``pot_path``/``pot_type``/
    ``pot_resolution_route``/``pot_file_sha256`` quadruple -- including the
    ``sha256_file`` existence guard -- shared by every builder family (Task 6,
    spec §3 F3) instead of each family re-deriving it. ``component`` is any
    ``ComponentConfig``-shaped object (``SheetConfig``, ``TipConfig``,
    ``SubstrateConfig``, ``FlakeConfig``, ...) with at least a ``mat``
    attribute; a component with no ``pot_path`` (e.g. an unfilled ``[Flake]``
    shape spec) records every ``pot_*`` field as ``None`` rather than raising.
    """
    pot = getattr(component, 'pot_path', None)
    pot_path = Path(str(pot)) if pot else None
    return MaterialProvenance(
        name=str(getattr(component, 'mat', '') or ''),
        role=role,
        pot_path=str(pot) if pot else None,
        pot_type=getattr(component, 'pot_type', None),
        pot_resolution_route=getattr(component, 'pot_resolution_route', None),
        pot_file_sha256=(sha256_file(pot_path)
                         if pot_path and pot_path.is_file() else None),
    )


def write_provenance(run_dir: Path, record: ProvenanceRecord) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / 'provenance.json'
    out.write_text(record.model_dump_json(indent=2), encoding='utf-8')
    return out


def read_provenance(path: Path, *, family: str = 'afm', model: str = 'afm',
                    material_name: str = '') -> ProvenanceRecord:
    """Read a provenance.json, upgrading a schema-1.x file in memory.

    A 1.x file predates ``family``/``model``/``materials``: the keyword
    arguments supply the first two (``src.run.tree.discover_runs`` passes the
    run's tokens) and the single ``pot_*`` triple becomes one ``'sheet'``-role
    :class:`MaterialProvenance` named ``material_name``. A 2.x file ignores all
    three. The file on disk is never rewritten.
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if str(data.get('schema_version', '1.0.0')).startswith('1.'):
        pot = {k: data.pop(k, None) for k in
               ('pot_path', 'pot_type', 'pot_resolution_route', 'pot_file_sha256')}
        data['schema_version'] = PROVENANCE_SCHEMA_VERSION
        data.setdefault('family', family)
        data.setdefault('model', model)
        data['materials'] = [MaterialProvenance(name=material_name, role='sheet', **pot)]
    return ProvenanceRecord(**data)


def load_key(load: float) -> str:
    """The one spelling of a load as a dict key (10, 10.0 -> '10'; 2.5 -> '2.5')."""
    return f'{float(load):g}'


def _ramp_by_load(matches: list, loads: Optional[Sequence[float]],
                  log_path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    """Key each RAMP_STATUS entry by load: its own ``load=``, else by position."""
    entries = [{'converged': conv == '1', 'r': float(r_val),
                'iterations': int(iters)} for conv, r_val, iters, _ in matches]
    if all(m[3] for m in matches):
        keys = [load_key(float(m[3])) for m in matches]
    elif loads is not None and len(loads) == len(matches):
        keys = [load_key(x) for x in loads]
    else:
        logger.warning(
            "%s: %d RAMP_STATUS lines without load= cannot be matched to %s "
            "configured loads; ramp_by_load omitted.", log_path, len(matches),
            'no' if loads is None else len(loads))
        return None
    return dict(zip(keys, entries))


def parse_log_extras(log_path: Path,
                     loads: Optional[Sequence[float]] = None) -> dict:
    """Post-run enrichment: RAMP_STATUS + UNLOADED_REF lines from the log.

    Returns a dict of ProvenanceRecord field updates. The scalar ramp fields
    keep the last RAMP_STATUS line (write order); ``ramp_by_load`` keeps all of
    them, keyed by each line's ``load=`` or, for older logs without it, paired
    by position with ``loads`` when the counts agree.
    """
    text = Path(log_path).read_text(encoding='utf-8', errors='ignore')
    updates: dict = {}
    ramp_matches = _RAMP_RE.findall(text)
    if ramp_matches:
        conv, r_val, iters, _ = ramp_matches[-1]
        updates['ramp_converged'] = conv == '1'
        updates['ramp_r'] = float(r_val)
        updates['ramp_iterations'] = int(iters)
        by_load = _ramp_by_load(ramp_matches, loads, Path(log_path))
        if by_load:
            updates['ramp_by_load'] = by_load
    zrefs = {f'layer_{lyr}': float(z) for lyr, z in _ZREF_RE.findall(text)}
    if zrefs:
        updates['unloaded_reference_z'] = zrefs
    m = re.search(r'^LAMMPS \(([^)]+)\)', text, re.M)
    if m:
        updates['lammps_version'] = m.group(1)
    return updates


def _mean_hold_fz(path: Path) -> float:
    """Mean of column ``v_hold_fz`` of an ``ave/time`` file.

    A data row with fewer fields than the header (a job killed mid-write) is
    skipped rather than read: its last field may itself be cut short.
    """
    names: List[str] = []
    values: List[float] = []
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if line.startswith('#'):
            if 'TimeStep' in line:
                names = line.lstrip('# ').split()
            continue
        fields = line.split()
        if not fields or (names and len(fields) < len(names)):
            continue        # blank, or a row cut short by a mid-write kill
        values.append(float(fields[names.index('v_hold_fz')]))
    return sum(values) / len(values)


def parse_hold_files(data_dir: Path) -> Dict[str, float]:
    """Mean hold normal force per load from ``hold_<load>N.txt`` files."""
    out: Dict[str, float] = {}
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return out
    for path in sorted(data_dir.iterdir()):
        m = _HOLD_FILE_RE.match(path.name)
        if not m:
            continue
        try:
            out[load_key(float(m.group(1)))] = _mean_hold_fz(path)
        except (ValueError, ZeroDivisionError, OSError) as exc:
            logger.warning("Skipping unparseable hold file %s (%s).", path, exc)
    return out
