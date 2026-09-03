"""Sweep expansion, material preflight, and build counting.

Split out of src/core/run.py. Turns one raw config into the list of configs a
sweep expands to, validates the material/template surface before any build
runs, and predicts how many builds and run scripts a config will produce.
"""
from dataclasses import dataclass, field as dataclass_field
import itertools
import logging
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import TypeAdapter, ValidationError

from ..core.config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
    parse_config,
)
from .placeholders import (
    TEMPLATE_MARKERS,
    has_template_marker,
    substitute_materials,
)
from ..core.utils.paths import _find_existing  # noqa: PLC2701
from .naming import (
    AFM_LIKE_MODELS,
    LAMMPS_LOOP_PARAMS,
    LOOPABILITY,
    SHEET_LIKE_MODELS,
    _numbered_sheet_keys,
    loopability,
    loopability_section,
    primary_sheet,
)

logger = logging.getLogger(__name__)


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


def read_material_entries(path: Union[str, Path],
                          base_dir: Optional[Union[str, Path]] = None
                          ) -> List[Tuple[str, ...]]:
    """Read a material list file as ordered entries -- singles or pairs.

    Same line reading as the old single-material ``read_material_list``:
    blank lines, surrounding whitespace, ``#`` comments (whole-line or
    trailing) all handled. A comma-separated line is still multiple single
    materials (``'MoS2, WS2'`` -> ``('MoS2',), ('WS2',)``) -- that behaviour
    is pinned by ``tests/run/test_material_list_robustness.py`` and predates
    heterostructure pairs, so it is untouched here. A ``'+'`` line is new: it
    names one ORDERED pair for a two-sheet sweep (``'h-MoS2 + h-WS2'`` ->
    ``('h-MoS2', 'h-WS2')``) -- order is never normalised, since sheet 1 and
    sheet 2 are different positions in the stack, not a set.

    A line cannot mix the two separators (a comma implies multiple single
    entries; a pair is exactly one entry, so a comma on a ``'+'`` line is
    ambiguous) and a ``'+'`` line naming more than two materials has no
    ``[2D-N]`` section to hold the extra members. Both raise ``ValueError``
    naming the offending line.

    Args:
        path: Path to the list file.
        base_dir: Directory to resolve a relative ``path`` against.

    Returns:
        Entries in file order. Duplicates dedupe order-preservingly, matching
        the old single-material behaviour; a duplicated pair also logs a
        warning, since two pair lines colliding is more likely to be a
        copy-paste mistake than a duplicated single name is.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
        ValueError: If a line mixes ',' and '+', or a '+' line does not name
            exactly two materials.
    """
    found = _find_existing(str(path), base_dir)
    if found is None:
        raise FileNotFoundError(f"Material list file '{path}' not found.")

    entries: List[Tuple[str, ...]] = []
    seen = set()
    for raw_line in found.read_text(encoding='utf-8').splitlines():
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue

        if '+' in line:
            if ',' in line:
                raise ValueError(
                    f"Material list line '{line}' mixes ',' and '+': a pair "
                    "line ('mat1 + mat2') cannot also list comma-separated "
                    "materials."
                )
            members = tuple(m.strip() for m in line.split('+'))
            if not all(members):
                raise ValueError(
                    f"Material list line '{line}' has an empty member around '+'."
                )
            if len(members) != 2:
                raise ValueError(
                    f"Material list line '{line}' names {len(members)} materials; "
                    "an ordered pair needs exactly 2, e.g. 'mat1 + mat2'."
                )
            if members in seen:
                logger.warning("Duplicate material pair '%s' ignored.",
                               ' + '.join(members))
                continue
            seen.add(members)
            entries.append(members)
            continue

        for raw_name in line.split(','):
            name = raw_name.strip()
            if not name:
                continue
            single = (name,)
            if single in seen:
                continue
            seen.add(single)
            entries.append(single)

    return entries


def read_material_list(path: Union[str, Path],
                       base_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Read a material list file of single materials only.

    A thin wrapper over :func:`read_material_entries` that keeps the original
    ``List[str]`` contract (and its comma-means-multiple-singles behaviour)
    unchanged for callers that do not know about ordered pairs. A ``'+'``
    pair line is not a single material, so it raises rather than silently
    collapsing to one of its members.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
        ValueError: If the file contains a ``'+'`` pair line.
    """
    entries = read_material_entries(path, base_dir)
    pairs = [e for e in entries if len(e) != 1]
    if pairs:
        raise ValueError(
            f"Material list '{path}' contains an ordered pair "
            f"('{' + '.join(pairs[0])}'), which read_material_list does not "
            "support -- use read_material_entries for a two-sheet sweep."
        )
    return [m for e in entries for m in (e if len(e) == 1 else ())]


@dataclass
class MaterialPreflight:
    """Which materials in a sweep can actually be built.

    ``usable``/``skipped`` hold plain material names for
    :func:`preflight_materials`, or ordered ``(mat1, mat2)`` pairs for
    :func:`preflight_material_pairs` -- both share this result type and
    ``summary()`` rendering.
    """

    usable: List[Any] = dataclass_field(default_factory=list)
    skipped: List[Tuple[Any, str]] = dataclass_field(default_factory=list)

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
            candidate = substitute_materials(template, mat)
            if _find_existing(candidate, base_dir) is None:
                missing.append(f"{key}={candidate}")
        if missing:
            result.skipped.append((mat, "missing " + ", ".join(missing)))
        else:
            result.usable.append(mat)
    return result


def preflight_material_pairs(entries: List[Tuple[str, str]],
                             primary_section: Dict[str, Any],
                             second_section: Dict[str, Any],
                             base_dir: Optional[Union[str, Path]] = None
                             ) -> MaterialPreflight:
    """Check which ordered material pairs have both sheets' files on disk.

    The pair analogue of :func:`preflight_materials`: member 1 (``mat1``) is
    checked against the primary section's ``cif_path``/``pot_path``
    templates, member 2 (``mat2``) against ``second_section``'s -- a pair is
    usable only when both sides resolve, since a hetero build needs both
    sheets.

    Args:
        entries: Ordered 2-tuples, e.g. from :func:`read_material_entries`.
        primary_section: The ``[2D]``/``[2D-1]`` config section, checked
            against ``mat1``.
        second_section: The ``[2D-2]`` config section, checked against
            ``mat2``.
        base_dir: Directory to resolve relative paths against.

    Returns:
        The usable pairs and the reason each skipped one was dropped.
    """
    result = MaterialPreflight()
    for mat1, mat2 in entries:
        missing = []
        for label, section, key in (
            ('2D-1', primary_section, 'cif_path'),
            ('2D-1', primary_section, 'pot_path'),
            ('2D-2', second_section, 'cif_path'),
            ('2D-2', second_section, 'pot_path'),
        ):
            template = (section or {}).get(key)
            if not isinstance(template, str) or not template:
                continue
            candidate = substitute_materials(template, mat1, mat2)
            if _find_existing(candidate, base_dir) is None:
                missing.append(f"[{label}] {key}={candidate}")
        if missing:
            result.skipped.append((f"{mat1} + {mat2}", "missing " + ", ".join(missing)))
        else:
            result.usable.append((mat1, mat2))
    return result


#: The placeholders each sheet of an ordered pair may template on. The primary
#: sheet accepts both spellings of the first material: ``{mat}`` predates the
#: pair form and substitutes identically to ``{mat1}`` (see
#: :func:`substitute_materials`), so a single-material config that later gains
#: a second sheet keeps sweeping without being rewritten.
PRIMARY_PAIR_MARKERS = ('{mat}', '{mat1}')
SECOND_PAIR_MARKERS = ('{mat2}',)


def _section_markers(section: Optional[Dict[str, Any]]) -> Tuple[str, ...]:
    """Every placeholder a sheet section templates on, in TEMPLATE_MARKERS order.

    Drawn from the shared marker vocabulary rather than matched literally, so
    this cannot drift from what :func:`substitute_materials` actually replaces.

    ``materials_list`` is excluded because it names the file that *drives* the
    sweep rather than a value the sweep fills in -- a placeholder there would
    never be substituted, so it must not count as templating.
    """
    values = [value for key, value in (section or {}).items()
              if key != 'materials_list']
    return tuple(marker for marker in TEMPLATE_MARKERS
                 if any(isinstance(value, str) and marker in value
                        for value in values))


def _pair_template_fault(section: Optional[Dict[str, Any]],
                         accepted: Tuple[str, ...],
                         canonical: str,
                         pinned: str) -> Optional[str]:
    """Why this sheet cannot take its member of an ordered pair, or None.

    Two faults, reported separately because they mislead differently. A
    section carrying no marker is pinned to its literal value for every member
    of the sweep. A section carrying the *other* sheet's marker does vary --
    it just varies with the wrong member, so both sheets build one material
    while the directory is still named after the pair that was asked for.

    The wrong marker is a fault even when the right one is also present,
    because the markers are per *key*, not per section. The realistic mistake
    is a ``[2D-2]`` pasted from ``[2D-1]`` whose ``mat`` was updated to
    ``{mat2}`` while its ``cif_path``/``pot_path`` still read ``{mat1}``: the
    section holds ``{mat2}``, yet both sheets load the first material's files.
    """
    found = _section_markers(section)
    wrong = [marker for marker in found if marker not in accepted]
    if wrong:
        return (
            f"templates on {', '.join(repr(marker) for marker in wrong)}, "
            "which names the other sheet's material, so both sheets would "
            f"build the same one. Use '{canonical}' in its mat, cif_path and "
            "pot_path"
        )
    if not found:
        return (
            f"holds no material template, so {pinned}. Write '{canonical}' "
            "into its mat, cif_path and pot_path"
        )
    return None


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

    # The primary sheet section: plain '2D', else the first numbered '2D-N'
    # (mirrors primary_sheet(), but this runs on the raw dict before any
    # model exists, and needs the *key* to write expanded values back under,
    # not just the section). A numbered-only config used to be silently
    # skipped here -- only a plain '2D' was ever checked -- so its
    # materials_list never expanded into a sweep at all.
    if '2D' in base_config:
        primary_key: Optional[str] = '2D'
    else:
        numbered = _numbered_sheet_keys(base_config)
        primary_key = numbered[0] if numbered else None

    has_second_sheet = '2D-2' in base_config

    def resolve_material_entries(section_config: Dict[str, Any]) -> List[Tuple[str, ...]]:
        if 'materials_list' not in section_config:
            return []

        mat_list_path = section_config.get('materials_list', '')
        if not isinstance(mat_list_path, str) or not mat_list_path.endswith(".txt"):
            if mat_list_path and notes is not None:
                # Literally '[2D]', even when the primary section is numbered:
                # MainWindow._validate (src/gui/app.py) filters expansion
                # notes on this exact prefix.
                notes.append(
                    f"[2D] materials_list: '{mat_list_path}' is not a .txt file, "
                    f"so no material sweep was expanded.")
            return []

        try:
            entries = read_material_entries(mat_list_path, base_dir)
        except (FileNotFoundError, OSError) as exc:
            logger.warning("%s", exc)
            if notes is not None:
                notes.append(
                    f"[2D] materials_list: '{mat_list_path}' could not be read "
                    f"({exc}), so no material sweep was expanded.")
            return []

        # Each sheet of a pair must template on its OWN marker. A section with
        # the wrong marker, or none, still expands and still lands in a
        # directory named after the pair the list asked for, while building
        # something else. Nothing downstream catches that: the substituted
        # paths exist, so the preflight below is happy and validation is
        # clean. A config-level property, so it is decided once here rather
        # than per entry.
        pair_faults = [
            (label, _pair_template_fault(section, accepted, canonical, pinned))
            for label, section, accepted, canonical, pinned in (
                (primary_key, section_config, PRIMARY_PAIR_MARKERS, '{mat1}',
                 'every run would build the same first material'),
                ('2D-2', base_config.get('2D-2'), SECOND_PAIR_MARKERS, '{mat2}',
                 'every run would build the same second material'),
            )
        ]

        for entry in entries:
            is_pair = len(entry) == 2
            if is_pair and not has_second_sheet:
                raise ValueError(
                    f"Material list '{mat_list_path}' line "
                    f"'{' + '.join(entry)}' names an ordered pair, but this "
                    "config has no [2D-2] section to hold the second material."
                )
            if not is_pair and has_second_sheet:
                raise ValueError(
                    f"Material list '{mat_list_path}' line '{entry[0]}' names "
                    "a single material, but this config has a [2D-2] section "
                    "and expects ordered pairs ('mat1 + mat2')."
                )
            if is_pair:
                for label, fault in pair_faults:
                    if fault is not None:
                        raise ValueError(
                            f"Material list '{mat_list_path}' line "
                            f"'{' + '.join(entry)}' names an ordered pair, but "
                            f"the [{label}] section {fault}."
                        )

        if has_second_sheet:
            preflight = preflight_material_pairs(
                entries, section_config, base_config.get('2D-2', {}), base_dir)
        else:
            preflight = preflight_materials(
                [e[0] for e in entries], section_config, base_dir)
            preflight = MaterialPreflight(
                usable=[(mat,) for mat in preflight.usable],
                skipped=preflight.skipped,
            )

        if preflight.skipped:
            logger.warning(
                "Material list %s: %s", mat_list_path, preflight.summary()
            )
        if entries and not preflight.usable:
            raise ValueError(
                f"None of the {len(entries)} materials in '{mat_list_path}' could "
                f"be resolved. {preflight.summary()}"
            )
        return preflight.usable

    def expand_mat_template(section_config: Dict[str, Any], mat1: str,
                            mat2: Optional[str] = None) -> Dict[str, Any]:
        expanded = {}
        for key, val in section_config.items():
            if key == 'materials_list':
                continue
            if isinstance(val, str) and has_template_marker(val):
                expanded[key] = substitute_materials(val, mat1, mat2)
            else:
                expanded[key] = val
        return expanded

    if primary_key is not None:
        entries = resolve_material_entries(base_config[primary_key])
        if entries:
            sweep_params[(primary_key, '_material_expand')] = entries

    lammps_loop_params = {'force', 'scan_angle', 'pressure', 'scan_speed'}

    if 'general' in base_config:
        for key, val in base_config['general'].items():
            if isinstance(val, list) and key not in lammps_loop_params:
                sweep_params[('general', key)] = val

    # Potential sweep (spec §13): a pot_path LIST on the primary sheet sweeps
    # into separate runs, one directory per member (pot_ token appended by
    # output_dir_for — the LOOPABILITY rule). Homogeneous sheets only: two
    # distinct 2D materials or a flake make a single potential axis ambiguous.
    if primary_key is not None:
        pot_val = base_config.get(primary_key, {}).get('pot_path')
        if isinstance(pot_val, list):
            hetero_markers = [k for k in base_config
                              if k.startswith('2D-') and k != primary_key]
            has_flake = any(k == 'flake' or k.startswith('flake-')
                            for k in base_config)
            if hetero_markers or has_flake:
                raise ValueError(
                    "potential sweep is not supported on heterostructure or "
                    "flake configs (spec §13.2): a pot_path list needs a "
                    "single homogeneous 2D material"
                )
            from ..potentials.utils import resolve_pot_type
            root = Path(base_dir) if base_dir else Path('.')
            resolved = [resolve_pot_type(entry, root) for entry in pot_val]
            sweep_params[(primary_key, '_pot_expand')] = resolved

    if not sweep_params:
        return _resolve_lj_override_files([base_config], base_dir)

    keys, values = zip(*sweep_params.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    expanded_configs = []
    for perm in permutations:
        new_conf = deepcopy(base_config)

        mat_entry = (perm.pop((primary_key, '_material_expand'), None)
                    if primary_key is not None else None)
        if mat_entry and primary_key in new_conf:
            mat1 = mat_entry[0]
            mat2 = mat_entry[1] if len(mat_entry) > 1 else None
            new_conf[primary_key] = expand_mat_template(new_conf[primary_key], mat1, mat2)
            if mat2 is not None and '2D-2' in new_conf:
                new_conf['2D-2'] = expand_mat_template(new_conf['2D-2'], mat1, mat2)

        pot_entry = (perm.pop((primary_key, '_pot_expand'), None)
                     if primary_key is not None else None)
        if pot_entry is not None and primary_key in new_conf:
            new_conf[primary_key]['pot_path'] = pot_entry.path
            new_conf[primary_key]['pot_type'] = pot_entry.pot_type
            new_conf[primary_key]['pot_resolution_route'] = pot_entry.route

        for (section, key), val in perm.items():
            new_conf[section][key] = val

        expanded_configs.append(new_conf)

    return _resolve_lj_override_files(expanded_configs, base_dir)


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
    # Lazy: simulate.py imports this module for expand_config_sweeps, so a
    # module-level import here would close the loop. The selectors are the
    # runner's, not the counter's -- the counter merely asks them what the
    # runner would have chosen.
    from .simulate import _select_afm_builder_cls, _select_sheet_builder_cls

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