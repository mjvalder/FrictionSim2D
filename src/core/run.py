"""Shared simulation runner utilities.

Provides config expansion, shared simulation-root layout, and HPC script
generation used by both the CLI and Python scripts.
"""

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from types import SimpleNamespace
import itertools
import re

from pydantic import TypeAdapter, ValidationError

from .config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
    parse_config,
)
from .path_utils import format_dimension_token
from .utils import _find_existing  # noqa: PLC2701  (intentional internal reuse)
from ..builders.afm import AFMSimulation
from ..builders.afm_freestanding import FreestandingPESTipSimulation
from ..builders.sheetonsheet import SheetOnSheetSimulation
from ..builders.hetero_slide import HeteroSheetOnSheetSimulation
from ..builders.pes_scan import PESSheetSimulation, PESTipSimulation
from ..builders.indent import IndentSimulation
from ..hpc import HPCScriptGenerator, HPCConfig

logger = logging.getLogger(__name__)

#: Simulation models sharing the AFM stack (tip + sheet + substrate) and its
#: two-phase system.in → slide.in layout. The indent reuses the AFM build and
#: system.in indentation, replacing only the run phase (a hold + retract, no slide).
AFM_LIKE_MODELS = frozenset({'afm', 'pes_tip', 'indent'})
#: Simulation models sharing the sheet-on-sheet stack (single slide.in run).
SHEET_LIKE_MODELS = frozenset({'sheetonsheet', 'pes_sheet'})


def layer_aware_path_sort_key(path_str: str) -> Tuple[Tuple[int, Union[int, str]], ...]:
    """Sort paths naturally for layer segments like L1, L2, L10."""
    parts = Path(path_str).parts
    key_parts: List[Tuple[int, Union[int, str]]] = []
    for part in parts:
        match = re.fullmatch(r"L(\d+)", part)
        if match:
            key_parts.append((0, int(match.group(1))))
        else:
            key_parts.append((1, part))
    return tuple(key_parts)


def _as_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def _is_ascending(values: List[float]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def _is_subsequence(subseq: List[float], seq: List[float], tol: float = 1.0e-12) -> bool:
    if not subseq:
        return True
    idx = 0
    for value in seq:
        if abs(value - subseq[idx]) <= tol:
            idx += 1
            if idx == len(subseq):
                return True
    return False


def _validate_runtime_sweep_ordering(general_cfg: Dict[str, Any]) -> None:
    """Validate sweep ordering in runtime path (CLI/script flow)."""
    for field_name in ('force', 'pressure', 'scan_angle', 'scan_angle_force'):
        values = _as_float_list(general_cfg.get(field_name))
        if len(values) > 1 and not _is_ascending(values):
            raise ValueError(f"{field_name} must be in ascending order")

    selector = _as_float_list(general_cfg.get('scan_angle_force'))
    if len(selector) <= 1:
        return

    force_values = _as_float_list(general_cfg.get('force'))
    pressure_values = _as_float_list(general_cfg.get('pressure'))
    target_field = 'force' if force_values else 'pressure'
    target_values = force_values if force_values else pressure_values

    if not target_values:
        raise ValueError("scan_angle_force list requires force or pressure values")

    if not _is_subsequence(selector, target_values):
        raise ValueError(
            f"scan_angle_force must follow the same order as {target_field} values"
        )


def read_material_list(path: Union[str, Path],
                       base_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Read a material list file, tolerating the ways people actually write them.

    Blank lines, surrounding whitespace, ``#`` comments (whole-line or trailing)
    and duplicate entries are all handled. Entries may be one per line or
    comma-separated, or a mixture.

    Args:
        path: Path to the list file.
        base_dir: Directory to resolve a relative ``path`` against.

    Returns:
        Material names, in file order, without duplicates.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    found = _find_existing(str(path), base_dir)
    if found is None:
        raise FileNotFoundError(f"Material list file '{path}' not found.")

    materials: List[str] = []
    seen = set()
    for raw_line in found.read_text(encoding='utf-8').splitlines():
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue
        for entry in line.split(','):
            name = entry.strip()
            if name and name not in seen:
                seen.add(name)
                materials.append(name)
    return materials


@dataclass
class MaterialPreflight:
    """Which materials in a sweep can actually be built."""

    usable: List[str] = dataclass_field(default_factory=list)
    skipped: List[Tuple[str, str]] = dataclass_field(default_factory=list)

    def summary(self) -> str:
        if not self.skipped:
            return f"{len(self.usable)} material(s) resolved"
        detail = "; ".join(f"{mat} ({reason})" for mat, reason in self.skipped)
        return (
            f"{len(self.usable)} material(s) resolved, "
            f"{len(self.skipped)} skipped: {detail}"
        )


def preflight_materials(materials: List[str],
                        section_config: Dict[str, Any],
                        base_dir: Optional[Union[str, Path]] = None) -> MaterialPreflight:
    """Check which materials have the files their config template refers to.

    A missing CIF used to surface only part-way through a build, after other
    materials had already been generated. Resolving up front means one clear
    warning instead of a late crash, and the rest of the sweep still runs.

    Args:
        materials: Candidate material names.
        section_config: The config section holding the ``{mat}`` templates
            (``cif_path``, ``pot_path``).
        base_dir: Directory to resolve relative paths against.

    Returns:
        The usable materials and the reason each skipped one was dropped.
    """
    result = MaterialPreflight()
    for mat in materials:
        missing = []
        for key in ('cif_path', 'pot_path'):
            template = section_config.get(key)
            if not isinstance(template, str) or not template:
                continue
            candidate = template.replace('{mat}', mat)
            if _find_existing(candidate, base_dir) is None:
                missing.append(f"{key}={candidate}")
        if missing:
            result.skipped.append((mat, "missing " + ", ".join(missing)))
        else:
            result.usable.append(mat)
    return result


def resolve_lj_override_file(section: Dict[str, Any],
                             base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Merge a shared LJ-override table into an ``[lj_override]`` section.

    ``file`` names a plain-text table of ``pair  epsilon  sigma`` rows (comma
    or whitespace separated, ``#`` comments and blank lines allowed) meant to
    cover a whole material sweep: its keys are element pairs, not materials,
    so pairs whose elements are absent from a given run are simply unused.
    Inline keys already in ``section`` win over the file, so a single run can
    still override one pair without editing the shared table. The file is
    resolved and the ``'file'`` key dropped either way -- the rest of the
    pipeline (``PotentialManager.set_lj_overrides``) never sees it.

    Missing or unreadable files are skipped silently, same as
    ``materials_list``: a half-typed path is worked on more often than it is
    finished, and this is advisory convenience, not a required input.
    """
    result = dict(section)
    file_path = result.pop('file', None)
    if not file_path:
        return result

    found = _find_existing(str(file_path), base_dir)
    if found is None:
        logger.warning("LJ override file '%s' not found; ignoring.", file_path)
        return result

    try:
        text = found.read_text(encoding='utf-8')
    except OSError as exc:
        logger.warning("LJ override file '%s' could not be read (%s); ignoring.",
                       file_path, exc)
        return result

    for raw_line in text.splitlines():
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue
        tokens = [tok for tok in re.split(r'[\s,]+', line) if tok]
        if len(tokens) != 3:
            continue
        pair_key, epsilon, sigma = tokens
        try:
            values = [float(epsilon), float(sigma)]
        except ValueError:
            continue
        # Inline wins: only fill in a pair the section did not already name.
        result.setdefault(pair_key, values)

    return result


def _resolve_lj_override_files(configs: List[Dict[str, Any]],
                               base_dir: Optional[Union[str, Path]]) -> List[Dict[str, Any]]:
    """Apply `resolve_lj_override_file` to every config's [lj_override].

    A shared list, not a per-config check for `'file' in section`: this has
    to run on every config `expand_config_sweeps` returns, including the
    no-sweep early return below, which is the common case.
    """
    for config in configs:
        section = config.get('lj_override')
        if isinstance(section, dict):
            config['lj_override'] = resolve_lj_override_file(section, base_dir)
    return configs


def expand_config_sweeps(base_config: Dict[str, Any],
                         base_dir: Optional[Union[str, Path]] = None,
                         notes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Expand config with material lists and parameter sweeps.

    Args:
        base_config: Raw config mapping.
        base_dir: Directory to resolve relative material-list and component
            paths against. Optional; without it, resolution is relative to the
            current working directory as before.
        notes: Optional list to collect informational messages about expansion
            decisions (e.g., when a materials_list file is ignored).

    Raises:
        ValueError: If a material list was given but none of its materials can
            be resolved -- silently generating nothing is worse than failing.
    """
    sweep_params = {}

    def resolve_materials_list(section_config: Dict[str, Any]) -> List[str]:
        if 'materials_list' not in section_config:
            return []

        mat_list_path = section_config.get('materials_list', '')
        if not isinstance(mat_list_path, str) or not mat_list_path.endswith(".txt"):
            if mat_list_path and notes is not None:
                notes.append(
                    f"[2D] materials_list: '{mat_list_path}' is not a .txt file, "
                    f"so no material sweep was expanded.")
            return []

        try:
            materials = read_material_list(mat_list_path, base_dir)
        except (FileNotFoundError, OSError) as exc:
            logger.warning("%s", exc)
            if notes is not None:
                notes.append(
                    f"[2D] materials_list: '{mat_list_path}' could not be read "
                    f"({exc}), so no material sweep was expanded.")
            return []

        preflight = preflight_materials(materials, section_config, base_dir)
        if preflight.skipped:
            logger.warning(
                "Material list %s: %s", mat_list_path, preflight.summary()
            )
        if materials and not preflight.usable:
            raise ValueError(
                f"None of the {len(materials)} materials in '{mat_list_path}' could "
                f"be resolved. {preflight.summary()}"
            )
        return preflight.usable

    def expand_mat_template(section_config: Dict[str, Any], mat_value: str) -> Dict[str, Any]:
        expanded = {}
        for key, val in section_config.items():
            if key == 'materials_list':
                continue
            if isinstance(val, str) and '{mat}' in val:
                expanded[key] = val.replace('{mat}', mat_value)
            else:
                expanded[key] = val
        return expanded

    if '2D' in base_config:
        materials = resolve_materials_list(base_config['2D'])
        if materials:
            sweep_params[('2D', '_material_expand')] = materials

    lammps_loop_params = {'force', 'scan_angle', 'pressure', 'scan_speed'}

    if 'general' in base_config:
        for key, val in base_config['general'].items():
            if isinstance(val, list) and key not in lammps_loop_params:
                sweep_params[('general', key)] = val

    if not sweep_params:
        return _resolve_lj_override_files([base_config], base_dir)

    keys, values = zip(*sweep_params.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    expanded_configs = []
    for perm in permutations:
        new_conf = deepcopy(base_config)

        mat_value = perm.pop(('2D', '_material_expand'), None)
        if mat_value and '2D' in new_conf:
            new_conf['2D'] = expand_mat_template(new_conf['2D'], mat_value)

        for (section, key), val in perm.items():
            new_conf[section][key] = val

        expanded_configs.append(new_conf)

    return _resolve_lj_override_files(expanded_configs, base_dir)


#: How a list value in each config key is interpreted. A parameter may only be
#: *swept* into separate runs if it changes the output directory name (see
#: ``run_simulations``); otherwise every member would write into the same
#: directory and overwrite its predecessor.
#:
#:   'sweep'    expanded into separate configs, each with its own directory
#:   'layer'    looped inside build() into L{n} subdirectories
#:   'in_file'  looped inside one LAMMPS script, or split into one run script
#:              per value when named by ``outer_loop``
#:   'scalar'   a list here is not supported
LOOPABILITY: Dict[Tuple[str, str], str] = {
    ('2D', 'materials_list'): 'sweep',
    ('general', 'temp'): 'sweep',
    ('2D', 'layers'): 'layer',
    ('general', 'force'): 'in_file',
    ('general', 'scan_angle'): 'in_file',
    ('general', 'scan_angle_force'): 'in_file',
    ('general', 'pressure'): 'in_file',
    ('general', 'scan_speed'): 'in_file',
}

#: Keys the LAMMPS scripts loop over internally; excluded from sweep expansion.
LAMMPS_LOOP_PARAMS = frozenset({'force', 'scan_angle', 'pressure', 'scan_speed'})


def loopability(section: str, key: str) -> str:
    """How a list value in ``[section] key`` would be interpreted."""
    return LOOPABILITY.get((section, key), 'scalar')


@dataclass
class ExpansionSummary:
    """How much work a config will produce.

    Three different numbers, all of which have been mistaken for each other:

        configs  separate configs, each with its own output directory
        builds   configs x layer counts -- the L{n} trees actually written
        jobs     run scripts, i.e. HPC array entries
    """

    configs: int
    builds: int
    jobs: int
    notes: List[str] = dataclass_field(default_factory=list)

    def render(self) -> str:
        return f"{self.configs} config(s) -> {self.builds} build(s) -> {self.jobs} job(s)"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


#: Coerces a raw value the way the pydantic models do. `read_config` leaves
#: `.ini` values as strings, and `bool('false')` is True -- the opposite of
#: what a config saying `finite_sheet = false` means (and of what the model
#: would produce). Reusing pydantic's own parser keeps the two in step.
_BOOL_ADAPTER = TypeAdapter(bool)


def _as_model_bool(value: Any, default: bool = False) -> bool:
    """`value` as pydantic would read it, or `default` if it would not."""
    if value is None:
        return default
    try:
        return _BOOL_ADAPTER.validate_python(value)
    except ValidationError:
        return default


def _routing_view(config: Dict[str, Any], model: str) -> SimpleNamespace:
    """A stand-in for the validated config object, built from a raw mapping.

    The builder-routing functions (:func:`_select_afm_builder_cls`,
    :func:`_select_sheet_builder_cls`) read a handful of fields off a validated
    config object. The prediction path has only the raw mapping and cannot
    validate it -- ``AFMSimulationConfig`` / ``SheetOnSheetSimulationConfig``
    resolve potential files and require a ``GlobalSettings``, so they raise on
    configs that are perfectly countable -- so they are handed this view
    instead.

    Only the *fields* are mirrored here (the ``2D-N`` folding of
    ``_fold_sheet_sections``, and the ``flake``/``Flake`` section alias, which
    differs between the two families). The routing *rules* stay in the two
    selector functions, so there is exactly one copy of them.
    """
    general = config.get('general', {}) or {}
    numbered = [k for k in config if isinstance(k, str) and k.startswith('2D-')]
    sheets = numbered or (['2D'] if '2D' in config else [])
    flake_key = 'flake' if model in AFM_LIKE_MODELS else 'Flake'
    twist = general.get('twist_angle')
    return SimpleNamespace(
        sub=config.get('sub'),
        sheets=sheets,
        flake=config.get(flake_key),
        general=SimpleNamespace(
            # A blank value is an unset key, not a twist of nothing: configparser
            # gives None, but a half-typed GUI field or a JSON config can hand
            # over ''. Only presence is read, so '' would route to hetero.
            twist_angle=None if twist == '' else twist,
            finite_sheet=_as_model_bool(general.get('finite_sheet')),
        ),
    )


def _builder_cls_for_count(config: Dict[str, Any], model: str) -> Optional[type]:
    """The builder class a raw config would route to, or None if undecidable.

    None means "route it like the model's default builder": an unroutable
    config (e.g. a substrate-free AFM, which raises
    :class:`FreestandingConfigError`) has no build to predict anyway, so the
    caller keeps its previous, model-only answer rather than inventing one.
    """
    view = _routing_view(config, model)
    try:
        if model in AFM_LIKE_MODELS:
            return _select_afm_builder_cls(model, view)
        if model in SHEET_LIKE_MODELS:
            return _select_sheet_builder_cls(model, view)
    except Exception:  # pylint: disable=broad-except
        return None
    return None


def outer_loop_conflict(config: Dict[str, Any], model: str) -> Optional[str]:
    """Why this config's ``outer_loop`` cannot be honoured, or None.

    One conflict today: ``outer_loop='scan_angle'`` with ``scan_angle_force``.
    The selector restricts which force/pressure value(s) the ``scan_angle``
    list applies to (the templates' angle_loop gate), but a per-angle script
    applies its one fixed angle to every force in it -- so the selector would
    be silently dropped, changing what a gated scan means.

    Routing-aware, and it has to be: the conflict exists only where the split
    exists. The indent, both PES scans and the hetero/flake slides never split
    by angle (``SPLIT_OUTER_LOOPS``), and the hetero/flake slide templates
    genuinely honour the in-script gate -- so for them a stray ``outer_loop``
    beside a real gate is inert, and rejecting it would fail a config that
    builds and runs correctly. Only a config whose selected builder really
    splits is a conflict, and only with more than one angle (a single angle
    falls through to the in-script loop, which honours the gate).

    An unroutable config reports nothing: it has no build to protect, and the
    builder-side raises remain the backstop.

    Called before anything is built -- by :func:`run_simulations` (which
    raises) and by the GUI's validation panel (which lists it) -- because the
    builders only raise from ``write_inputs``, by which point the geometry is
    built and a partially written tree is on disk.
    """
    general = config.get('general', {}) or {}
    if general.get('outer_loop') != 'scan_angle':
        return None
    if general.get('scan_angle_force') is None:
        return None
    if len(_as_list(general.get('scan_angle'))) <= 1:
        return None

    builder_cls = _builder_cls_for_count(config, model)
    if builder_cls is None or 'scan_angle' not in builder_cls.SPLIT_OUTER_LOOPS:
        return None

    return (
        "outer_loop='scan_angle' cannot be combined with scan_angle_force: "
        f"{builder_cls.__name__} splits the run phase into one script per "
        "angle, and each script applies its angle to every force/pressure "
        "value -- silently dropping the selective filter. Remove "
        "scan_angle_force, or drop outer_loop to keep the in-script angle "
        "loop (which honors it)."
    )


def _run_scripts_per_build(config: Dict[str, Any], model: str) -> Tuple[int, List[str]]:
    """How many run scripts one built directory will contain.

    Asks the builder the config actually routes to: ``outer_loop`` promotes one
    in-file loop to separate ``slide_*.in`` files only when that builder
    declares the parameter in ``SPLIT_OUTER_LOOPS`` (the indent's hold+retract
    and the hetero/flake/PES builders write a single run script whatever
    ``outer_loop`` says), and a PES sheet scan adds a refinement script. Pinned
    against real generated trees by ``test_expansion_count``.
    """
    general = config.get('general', {}) or {}
    outer_loop = general.get('outer_loop')
    notes: List[str] = []

    scripts = 1
    if outer_loop:
        values = _as_list(general.get(outer_loop))
        builder_cls = _builder_cls_for_count(config, model)
        splits = (outer_loop in builder_cls.SPLIT_OUTER_LOOPS
                  if builder_cls is not None else True)
        if len(values) > 1 and splits:
            scripts = len(values)
            notes.append(f"outer_loop={outer_loop} splits the run phase into {scripts} scripts")
        elif len(values) > 1:
            notes.append(
                f"outer_loop={outer_loop} is ignored by {builder_cls.__name__}: "
                f"the run phase stays a single script"
            )

    if model == 'pes_sheet':
        pes = config.get('pes', {}) or {}
        if pes.get('grid_n_refine'):
            scripts += 1
            notes.append("pes.grid_n_refine adds a refinement script")

    return scripts, notes


def _numbered_sheet_keys(run_dict: Dict[str, Any]) -> List[str]:
    """Every '2D-N' key in ``run_dict``, sorted by N ascending.

    Keys are matched by prefix and sorted numerically (so '2D-2' sorts
    before '2D-10'); a non-numeric suffix sorts last rather than raising.
    """
    return sorted(
        (k for k in run_dict if isinstance(k, str) and k.startswith('2D-')),
        key=lambda key: int(key.split('-', 1)[1])
        if key.split('-', 1)[1].isdigit() else 10**9,
    )


def primary_sheet(run_dict: Dict[str, Any]) -> Dict[str, Any]:
    """The primary [2D] section: plain '2D', else the first numbered '2D-N'.

    Numbered-only configs are legal -- the pydantic fold mirrors 2D-1 onto
    '2D' -- but the naming and counting code below runs before any model is
    built, straight off the raw config dict, so it needs its own fallback.
    """
    if '2D' in run_dict:
        return run_dict['2D']
    numbered = _numbered_sheet_keys(run_dict)
    if numbered:
        return run_dict[numbered[0]]
    raise KeyError('2D')


def material_token(run_dict: Dict[str, Any]) -> str:
    """Directory token for the material(s): 'mat' or 'mat1+mat2+...'.

    Two hetero runs differing only in material 2 must not share a directory,
    so every numbered sheet section contributes its 'mat' in numeric order.
    Joined with '+' rather than '-': material names themselves contain '-'
    (e.g. 'h-MoS2'), which would make a '-' join unsplittable, and
    postprocessing already folds '-' into '_' for its own filter keys.
    """
    numbered = _numbered_sheet_keys(run_dict)
    if numbered:
        return '+'.join(str(run_dict[k].get('mat', 'unknown')) for k in numbered)
    return str(primary_sheet(run_dict).get('mat', 'unknown'))


def count_expansion(base_config: Dict[str, Any],
                    model: str,
                    base_dir: Optional[Union[str, Path]] = None) -> ExpansionSummary:
    """Predict how many configs, builds and jobs a config will generate.

    Args:
        base_config: Raw config mapping.
        model: Simulation model name (``afm``, ``indent``, ``pes_tip``,
            ``sheetonsheet``, ``pes_sheet``).
        base_dir: Directory to resolve relative paths against.

    Returns:
        An :class:`ExpansionSummary`.
    """
    notes: List[str] = []
    expanded = expand_config_sweeps(base_config, base_dir, notes)
    configs = len(expanded)

    builds = 0
    jobs = 0
    for config in expanded:
        layers = _as_list(primary_sheet(config).get('layers')) or [1]
        scripts, script_notes = _run_scripts_per_build(config, model)
        builds += len(layers)
        jobs += len(layers) * scripts
        for note in script_notes:
            if note not in notes:
                notes.append(note)

    return ExpansionSummary(configs=configs, builds=builds, jobs=jobs, notes=notes)


def count_generated_jobs(simulation_root: Path) -> int:
    """Exact job count for an already-generated tree.

    Uses the same discovery as the HPC manifest: every ``*slide*.in`` in a
    simulation directory, falling back to ``*system*.in`` when there are none.
    """
    paths = collect_hpc_simulation_paths(simulation_root)
    entries, _ = _build_hpc_manifest_entries(simulation_root, paths, [])
    return len(entries)


def collect_hpc_simulation_paths(
    simulation_root: Path,
    lammps_scripts: Optional[List[str]] = None,
) -> List[str]:
    """Collect simulation paths relative to a shared simulation root."""
    scripts = ['system.in', 'slide.in'] if lammps_scripts is None else lammps_scripts

    def script_exists(lammps_dir: Path, script_name: str) -> bool:
        if '*' in script_name or '?' in script_name:
            return any(lammps_dir.glob(script_name))
        if script_name == 'system.in':
            return (
                (lammps_dir / 'system.in').exists()
                or any(lammps_dir.glob('*system*.in'))
            )
        if script_name == 'slide.in':
            return (
                (lammps_dir / 'slide.in').exists()
                or any(lammps_dir.glob('*slide*.in'))
            )
        return (lammps_dir / script_name).exists()

    simulation_paths = []
    for lammps_dir in simulation_root.rglob('lammps'):
        if not lammps_dir.is_dir():
            continue

        if scripts:
            if not any(script_exists(lammps_dir, script) for script in scripts):
                continue
        elif not list(lammps_dir.glob('*.in')):
            continue

        rel_path = lammps_dir.parent.relative_to(simulation_root)
        simulation_paths.append(str(rel_path))
    return sorted(set(simulation_paths), key=layer_aware_path_sort_key)


def _build_hpc_manifest_entries(
    simulation_root: Path,
    simulation_paths: List[str],
    configured_scripts: List[str],
) -> Tuple[List[str], List[str]]:
    """Build manifest rows as full script paths discovered from built outputs.

    Each row is a full script path relative to ``simulation_root`` such as
    ``sheetonsheet/.../lammps/slide_p1gpa.in``.

    Discovery strategy per simulation directory:
        - If any ``*slide*.in`` files exist, add all of them (array-per-slide).
        - Otherwise add all ``*system*.in`` files.
    """
    manifest_entries: List[str] = []

    for sim_rel in simulation_paths:
        lammps_dir = simulation_root / sim_rel / 'lammps'
        if not lammps_dir.exists():
            continue

        system_scripts = sorted(
            p.relative_to(simulation_root).as_posix()
            for p in lammps_dir.glob('*.in')
            if 'system' in p.name
        )
        slide_scripts = sorted(
            p.relative_to(simulation_root).as_posix()
            for p in lammps_dir.glob('*.in')
            if 'slide' in p.name
        )

        if slide_scripts:
            manifest_entries.extend(slide_scripts)
        elif system_scripts:
            manifest_entries.extend(system_scripts)

    if manifest_entries:
        return manifest_entries, ['__MANIFEST_PATH__']

    return simulation_paths, configured_scripts


def dispatch_hpc_generation(
    generator: 'HPCScriptGenerator',
    simulation_root: Path,
    output_dir: Path,
    scheduler: str,
    base_dir: str,
    manifest_entries: List[str],
):
    """Generate HPC scripts, choosing combined, two-phase, or single-phase layout.

    AFM simulations have one system.in and one slide.in per simulation
    directory. When both phases are present in equal numbers (1:1 per sim dir)
    we emit a single combined job that runs system then slide sequentially —
    no PBS dependency needed. When counts differ (unusual) we fall back to two
    dependent array jobs. Sheet-on-sheet simulations have only slide scripts
    and use a single array job.

    Returns whatever the chosen generator method returns.
    """
    from ..hpc.manifest import JobManifest

    manifest = JobManifest.from_simulation_directory(simulation_root)
    if manifest.has_system_jobs():
        system_jobs = manifest.get_system_jobs()
        slide_jobs = manifest.get_slide_jobs()
        if len(system_jobs) == len(slide_jobs):
            logger.info(
                "System+slide jobs detected (1:1) — generating combined HPC script."
            )
            return generator.generate_combined_scripts(
                manifest, output_dir, scheduler=scheduler, base_dir=base_dir
            )
        logger.info("System/slide counts differ — generating two-phase HPC scripts.")
        return generator.generate_two_phase_scripts(
            manifest, output_dir, scheduler=scheduler, base_dir=base_dir
        )
    return generator.generate_scripts(
        simulation_paths=manifest_entries,
        output_dir=output_dir,
        scheduler=scheduler,
        base_dir=base_dir,
    )


def hpc_generation_blocker(settings) -> Optional[str]:
    """Why HPC script generation would be skipped, or None if it would run.

    Generation has a precondition that is easy to miss: ``settings.hpc.modules``
    defaults to unset, and without it nothing is written. Exposing the reason
    lets a caller say so up front instead of appearing to succeed while
    producing no scripts.
    """
    if not getattr(settings.hpc, 'modules', None):
        return (
            "The HPC 'Modules' setting is empty. It is the only one that gates "
            "job-script generation: scheduler, nodes, CPUs and walltime are all "
            "written into the scripts but none of them substitute for it. "
            "Set it to the environment modules your cluster loads before "
            "running, e.g. 'lammps, openmpi'."
        )
    return None


def generate_hpc_scripts_for_root(simulation_root: Path, settings) -> Optional[str]:
    """Generate shared HPC scripts for a simulation root directory.

    Returns:
        None when scripts were generated, otherwise the reason they were not.
    """

    blocker = hpc_generation_blocker(settings)
    if blocker is not None:
        logger.warning("Skipping HPC script generation: %s", blocker)
        return blocker

    hpc_config = HPCConfig.from_settings(settings.hpc, job_name=simulation_root.name)

    simulation_paths = collect_hpc_simulation_paths(
        simulation_root,
        lammps_scripts=hpc_config.lammps_scripts,
    )
    if not simulation_paths:
        reason = "No simulations were found to write job scripts for."
        logger.warning("%s", reason)
        return reason

    manifest_entries, resolved_scripts = _build_hpc_manifest_entries(
        simulation_root,
        simulation_paths,
        hpc_config.lammps_scripts,
    )
    hpc_config.lammps_scripts = resolved_scripts

    hpc_dir = simulation_root / 'hpc'
    hpc_dir.mkdir(parents=True, exist_ok=True)

    if hpc_config.hpc_home:
        base_dir = hpc_config.hpc_home.rstrip('/') + '/' + simulation_root.name
    elif hpc_config.scheduler_type == 'pbs':
        base_dir = '$PBS_O_WORKDIR'
    else:
        base_dir = '$SLURM_SUBMIT_DIR'
    generator = HPCScriptGenerator(hpc_config)
    dispatch_hpc_generation(
        generator,
        simulation_root,
        hpc_dir,
        hpc_config.scheduler_type,
        base_dir,
        manifest_entries,
    )
    return None


class FreestandingConfigError(ValueError):
    """A substrate-free config was used with a model lacking a freestanding builder.

    Raised by :func:`_select_afm_builder_cls` and deliberately NOT caught by
    ``run_simulations``'s per-config broad-except (which exists to let a
    material/parameter sweep continue past isolated build failures) — this is
    a hard misconfiguration, not a recoverable per-config build error, so it
    must propagate and abort the whole run.
    """


def _select_afm_builder_cls(model, config_obj):
    """Pick the builder class, routing substrate-free configs to freestanding.

    A substrate-free config (no ``[sub]`` section) is only supported today for
    the freestanding tip PES scan. Freestanding dynamic slide/indent is a
    future (Phase 2) feature, so any other model without a substrate is a
    hard error rather than a silently broken build.
    """
    afm_builders = {'pes_tip': PESTipSimulation, 'indent': IndentSimulation}
    default_cls = afm_builders.get(model, AFMSimulation)
    if getattr(config_obj, 'sub', None) is None:
        freestanding = {'pes_tip': FreestandingPESTipSimulation}
        if model in freestanding:
            return freestanding[model]
        raise FreestandingConfigError(
            f"A substrate-free config (no [sub] section) is only supported for "
            f"'pes-tip' (the freestanding tip PES). Model '{model}' requires a [sub] "
            f"section; freestanding dynamic slide/indent is a future (Phase 2) feature."
        )
    # Hetero AFM when >=2 materials are configured (amorphous sub). ONE hetero
    # decision point: a finite ([general] finite_sheet) stack routes to the D5
    # edge-typed FiniteHeteroAFMSimulation, else the D2 periodic HeteroAFMSimulation.
    # pes_tip/indent are NOT rerouted here — their hetero variants are later phases.
    # TODO(future): hetero pes_tip/indent.
    sheets = getattr(config_obj, 'sheets', []) or []
    if model == 'afm' and len(sheets) >= 2:
        from src.builders.hetero_afm import HeteroAFMSimulation, FiniteHeteroAFMSimulation
        if getattr(config_obj.general, 'finite_sheet', False):
            return FiniteHeteroAFMSimulation
        return HeteroAFMSimulation
    return default_cls


def _select_sheet_builder_cls(model, config_obj):
    """Route sheet-on-sheet to the hetero builder for >=2 materials or a twist.

    A single ``[2D]`` material with ``general.twist_angle`` set is a genuine
    same-material moiré, so it must also reach the hetero builder (which
    synthesises the two sheets). A single material with no twist stays on the
    homogeneous builder (unchanged).
    """
    if model == 'pes_sheet':
        return PESSheetSimulation
    # A [Flake] section switches sheet-on-sheet to the D4 flake-driven slide,
    # regardless of the substrate being homogeneous or hetero (precedence).
    if model == 'sheetonsheet' and getattr(config_obj, 'flake', None) is not None:
        from src.builders.flake_slide import FlakeSlideSimulation
        return FlakeSlideSimulation
    sheets = getattr(config_obj, 'sheets', None) or []
    twist = getattr(getattr(config_obj, 'general', None), 'twist_angle', None)
    if len(sheets) >= 2 or twist is not None:
        return HeteroSheetOnSheetSimulation
    return SheetOnSheetSimulation


def output_dir_for(run_dict: Dict[str, Any], model: str) -> Path:
    """The output directory a run config builds into, relative to the
    simulation root.

    Extracted verbatim from the naming block in `run_simulations` so callers
    that never build anything -- the GUI's Generate preview, in particular --
    can predict a run's directory (and so spot sweep members that would
    collide) without duplicating the layout rules here.

    ``mat`` covers every numbered sheet section (see :func:`material_token`)
    so two hetero runs differing only in the second material do not collide;
    ``x``/``y`` come from the primary sheet (see :func:`primary_sheet`).
    """
    mat = material_token(run_dict)
    sheet = primary_sheet(run_dict)
    x = sheet.get('x', 100)
    y = sheet.get('y', 100)
    temp = run_dict['general'].get('temp', 300)
    size_token = f"{format_dimension_token(x)}x_{format_dimension_token(y)}y"

    top_dir = model  # 'afm' | 'sheetonsheet' | 'pes_tip' | 'pes_sheet'
    if model in AFM_LIKE_MODELS:
        tip_mat = run_dict.get('tip', {}).get('mat', 'Si')
        tip_amorph = run_dict.get('tip', {}).get('amorph', 'c')
        tip_r = run_dict.get('tip', {}).get('r', 25)
        sub_mat = run_dict.get('sub', {}).get('mat', 'Si')
        sub_amorph = run_dict.get('sub', {}).get('amorph', 'a')

        sub_str = f"{sub_amorph}{sub_mat}" if sub_amorph == 'a' else sub_mat
        tip_str = f"{tip_amorph}{tip_mat}" if tip_amorph == 'a' else tip_mat

        return (
            Path(top_dir) / mat / size_token /
            f"sub_{sub_str}_tip_{tip_str}_r{int(tip_r)}" /
            f"K{int(temp)}"
        )

    return Path(top_dir) / mat / size_token / f"K{int(temp)}"


def run_simulations(
    config_file: str,
    model: str,
    settings_file: Optional[Union[str, Path]] = None,
    output_root: Optional[Union[str, Path]] = None,
    ensure_hpc_settings=None,
    use_aiida: bool = False,
    generate_hpc: bool = False,
    simulation_root_name: Optional[str] = None,
    on_failure: Optional[Callable[[Path, Dict[str, Any], Exception], None]] = None,
    allow_generate_amorphous: bool = True,
) -> Tuple[List[Path], Path, List[Dict[str, Any]], Any]:
    """Run all simulations defined in a config file.

    Args:
        on_failure: Called as ``(output_dir, run_dict, exception)`` for each
            config whose build fails. Failures are otherwise only logged, and
            the caller sees just a success count with no way to say which
            member failed.
        allow_generate_amorphous: When False, a missing amorphous structure
            raises instead of triggering a melt-quench. See
            :func:`src.builders.components.make_amorphous`.

    Returns a tuple of (created_simulations, simulation_root, configs_to_run, settings).
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_dict = parse_config(config_path)
    defaults = load_settings(settings_file=settings_file)

    if use_aiida:
        defaults.aiida.enabled = True
        defaults.aiida.create_provenance = True

    # Carried on the settings rather than threaded through every builder: the
    # builders already pass `settings` down to make_amorphous.
    if not allow_generate_amorphous:
        defaults.quench.allow_generate = False

    if model in SHEET_LIKE_MODELS:
        if defaults.hpc.lammps_scripts == ['system.in', 'slide.in']:
            defaults.hpc.lammps_scripts = ['slide.in']
    elif model in AFM_LIKE_MODELS:
        if defaults.hpc.lammps_scripts == ['slide.in']:
            defaults.hpc.lammps_scripts = ['system.in', 'slide.in']

    # pes_tip uses the AFM stack (geometry/output layout) but is self-contained:
    # a single static scan script, no separate system.in indentation phase.
    if model == 'pes_tip':
        defaults.hpc.lammps_scripts = ['slide.in']

    if ensure_hpc_settings is not None and generate_hpc:
        ensure_hpc_settings(defaults.hpc)

    configs_to_run = expand_config_sweeps(base_dict)

    for run_dict in configs_to_run:
        _validate_runtime_sweep_ordering(run_dict.get('general', {}))
        # Before the simulation root exists: a conflict the builders would
        # only raise on from write_inputs, i.e. after a full geometry build
        # onto a tree that then stays half written.
        conflict = outer_loop_conflict(run_dict, model)
        if conflict is not None:
            raise ValueError(conflict)

    root_name = simulation_root_name or datetime.now().strftime("simulation_%Y%m%d_%H%M%S")
    base_root = Path(output_root) if output_root is not None else Path.cwd()
    simulation_root = base_root / root_name
    simulation_root.mkdir(parents=True, exist_ok=True)
    (simulation_root / 'logs').mkdir(parents=True, exist_ok=True)

    created_simulations: List[Path] = []

    for run_dict in configs_to_run:
        run_dict['settings'] = defaults.model_dump()

        output_dir = simulation_root / output_dir_for(run_dict, model)

        prov_dir = output_dir / 'provenance'
        prov_dir.mkdir(parents=True, exist_ok=True)

        try:
            if model in AFM_LIKE_MODELS:
                config_obj = AFMSimulationConfig(**run_dict)
                config_json_path = prov_dir / 'config.json'
                config_json_path.write_text(config_obj.model_dump_json(indent=2), encoding='utf-8')
                builder_cls = _select_afm_builder_cls(model, config_obj)
                builder = builder_cls(config_obj, output_dir, config_path=str(config_json_path))
            else:
                config_obj = SheetOnSheetSimulationConfig(**run_dict)
                config_json_path = prov_dir / 'config.json'
                config_json_path.write_text(config_obj.model_dump_json(indent=2), encoding='utf-8')
                builder_cls = _select_sheet_builder_cls(model, config_obj)
                builder = builder_cls(config_obj, output_dir, config_path=str(config_json_path))

            builder.set_base_output_dir(simulation_root)
            builder.build()
            created_simulations.append(output_dir)

        except FreestandingConfigError:
            # Hard misconfiguration (substrate-free config on a model with no
            # freestanding builder) — abort the whole run instead of silently
            # skipping this config and reporting a false "N/M successful".
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Build failed for %s: %s", output_dir, exc)
            # Logging alone leaves callers with only "N/M successful" and no way
            # to say which member failed or why. The callback lets them report it.
            if on_failure is not None:
                on_failure(output_dir, run_dict, exc)
            continue

    if generate_hpc and created_simulations:
        generate_hpc_scripts_for_root(simulation_root, defaults)

    return created_simulations, simulation_root, configs_to_run, defaults
