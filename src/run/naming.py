"""Output-tree naming: tokens, directory layout, loop classification.

Split out of run.py. Pure string/path logic -- it imports no builder and no
simulation machinery, which is what lets src/run/ later split without a cycle.

It also owns AFM_LIKE_MODELS / SHEET_LIKE_MODELS. Those are model-name
vocabulary rather than simulation machinery, and the flake-section rules here
need them, so they live in the leaf module and the runner imports them from
here -- not the other way round, which would be a cycle.
"""
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.utils.tokens import format_dimension_token

#: Simulation models sharing the AFM stack (tip + sheet + substrate) and its
#: two-phase system.in → slide.in layout. The indent reuses the AFM build and
#: system.in indentation, replacing only the run phase (a hold + retract, no slide).
AFM_LIKE_MODELS = frozenset({'afm', 'pes_tip', 'indent'})
#: Simulation models sharing the sheet-on-sheet stack (single slide.in run).
SHEET_LIKE_MODELS = frozenset({'sheetonsheet', 'pes_sheet'})

#: The CLI / top-directory axis: `run <model>` and `<root>/<model>/...`.
#: Every entry is also a family (see `core.observables.FAMILIES`), but the
#: reverse does not hold -- 'hetero' and 'flake' are schemas, never top
#: directories.
MODELS: Tuple[str, ...] = ('afm', 'indent', 'pes_tip', 'sheetonsheet', 'pes_sheet')


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
    ('2D', 'pot_path'): 'sweep',   # spec §13: one pot_ directory per member
    ('general', 'temp'): 'sweep',
    ('general', 'interlayer_style'): 'sweep',
    ('2D', 'layers'): 'layer',
    ('general', 'force'): 'in_file',
    ('general', 'scan_angle'): 'in_file',
    ('general', 'scan_angle_force'): 'in_file',
    ('general', 'pressure'): 'in_file',
    ('general', 'scan_speed'): 'in_file',
}


#: Keys the LAMMPS scripts loop over internally; excluded from sweep expansion.
#: Derived from :data:`LOOPABILITY`'s ``'in_file'`` entries under ``'general'``
#: rather than listed again by hand, so this set and ``LOOPABILITY`` cannot
#: drift apart the way they did when ``scan_angle_force`` was added to
#: ``LOOPABILITY`` but not here: a list-valued ``scan_angle_force`` was
#: silently swept into separate configs that all built to the identical
#: output directory (``output_dir_for`` never looks at it) and overwrote each
#: other. See :func:`~src.run.expand.expand_config_sweeps`, the sole consumer.
LAMMPS_LOOP_PARAMS = frozenset(
    key for (section, key), kind in LOOPABILITY.items()
    if section == 'general' and kind == 'in_file'
)


#: A heterostructure config splits the sheet into ``[2D-1]``/``[2D-2]`` instead
#: of a single ``[2D]``. Every one of them is a sheet section obeying the sheet
#: rules above, so the lookup is keyed on the plain name.
_NUMBERED_SHEET_SECTION = re.compile(r'^2D-\d+$')


def loopability_section(section: str) -> str:
    """The :data:`LOOPABILITY` key a config section is classified under.

    Only ``2D-N`` folds (onto ``2D``); everything else is its own key. Nothing
    that merely *looks* like a sheet section (``2D-``, ``2Dx``, ``flake-1``)
    may borrow the sheet rules.
    """
    return '2D' if _NUMBERED_SHEET_SECTION.match(section) else section


def loopability(section: str, key: str) -> str:
    """How a list value in ``[section] key`` would be interpreted."""
    return LOOPABILITY.get((loopability_section(section), key), 'scalar')


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


def _numbered_flake_keys(run_dict: Dict[str, Any]) -> List[str]:
    """Every 'Flake-N' key in ``run_dict``, sorted by N ascending.

    Mirrors :func:`_numbered_sheet_keys` for the dedicated hetero-flake
    sections (folded into ``flake_sheets`` by the pydantic model).
    """
    return sorted(
        (k for k in run_dict if isinstance(k, str) and k.startswith('Flake-')),
        key=lambda key: int(key.split('-', 1)[1])
        if key.split('-', 1)[1].isdigit() else 10**9,
    )


def _flake_section(run_dict: Dict[str, Any], model: Optional[str]) -> Any:
    """The raw ``[Flake]``/``[flake]`` section, whichever alias applies.

    The single-flake-shape section's key differs by family:
    ``SheetOnSheetSimulationConfig.flake`` aliases ``'Flake'`` (capital),
    ``AFMSimulationConfig.flake`` aliases ``'flake'`` (lowercase) -- same
    ``FlakeConfig`` model, different ini/dict key. This mirrors
    :func:`_routing_view`'s already-established
    ``flake_key = 'flake' if model in AFM_LIKE_MODELS else 'Flake'`` rule
    (the routing functions hit this exact casing split first) rather than
    inventing a second convention. When ``model`` isn't known (the two
    existing bare ``material_token(run_dict)`` call sites, with no model
    param, predate this fix) both keys are checked -- a real run_dict from a
    single family only ever populates one of them.
    """
    if model is not None:
        return run_dict.get('flake' if model in AFM_LIKE_MODELS else 'Flake')
    flake = run_dict.get('Flake')
    return flake if flake is not None else run_dict.get('flake')


def flake_material_token(run_dict: Dict[str, Any], model: Optional[str] = None) -> Optional[str]:
    """Directory-token contribution for a ``[Flake]``/``[flake]``/``[Flake-N]``
    section, or ``None`` when the flake doesn't change the built system's
    material identity.

    A flake config -- :class:`~src.builders.sheetonsheet.flake_slide.FlakeSlideSimulation`
    (routed from ``run sheetonsheet`` by a ``[Flake]`` section) or an AFM
    ``[flake]`` (the finite shaped flake between tip and sheet, built by
    :class:`~src.builders.afm.AFMSimulation`) -- was named off the substrate
    material only: ``material_token``/``output_dir_for`` never looked at
    ``flake_sheets``/``[Flake-N]``/``[flake]`` at all, so two flake configs
    differing ONLY in their flake material silently built into the same
    output directory and overwrote each other. (The first fix for this only
    checked the capital-F ``'Flake'`` key, so it closed the sheet-on-sheet
    flake path but missed AFM's lowercase ``'flake'`` alias entirely --
    ``run_dict.get('Flake')`` is always ``None`` for an AFM config, so the
    AFM ``[flake]`` collision was never actually closed. See
    :func:`_flake_section` for the casing rule.)

    A hetero flake (``[Flake-1]``/``[Flake-2]``, folded into ``flake_sheets``
    for both families) always names two materials explicitly (``mat`` is
    required on those sections), so it always contributes -- there is no
    established directory convention for it to stay byte-identical to (today
    it always collides).

    A single flake section may omit ``mat`` entirely for the sheet-on-sheet
    family, in which case the builder reuses the substrate ``[2D]``'s
    material
    (:meth:`~src.builders.sheetonsheet.flake_slide.FlakeSlideSimulation._filled_flake_config`).
    That's the common case there, and it must keep building into today's
    directory (no flake section at all), so it contributes nothing here. AFM's
    own ``[flake]`` has no such substrate fallback (its builder always needs
    an explicit ``mat``/``cif_path``/``pot_path`` to carve the flake), so a
    configured AFM flake always contributes. Either way, only an EXPLICIT
    ``mat`` contributes -- that's the case that actually changes what gets
    built and so must change the directory.
    """
    numbered = _numbered_flake_keys(run_dict)
    if numbered:
        return '+'.join(str(run_dict[k].get('mat', 'unknown')) for k in numbered)
    flake = _flake_section(run_dict, model)
    if isinstance(flake, dict) and flake.get('mat') is not None:
        return str(flake['mat'])
    return None


def material_token(run_dict: Dict[str, Any], model: Optional[str] = None) -> str:
    """Directory token for the material(s): 'mat' or 'mat1+mat2+...'.

    Two hetero runs differing only in material 2 must not share a directory,
    so every numbered sheet section contributes its 'mat' in numeric order.
    Joined with '+' rather than '-': material names themselves contain '-'
    (e.g. 'h-MoS2'), which would make a '-' join unsplittable, and
    postprocessing already folds '-' into '_' for its own filter keys.

    A flake section (see :func:`flake_material_token`) that names a distinct
    flake material contributes a '+flake-<mat...>' suffix, so a flake config
    doesn't collide with a hetero substrate ([2D-1]/[2D-2]) config whose
    material token happens to look the same joined with a bare '+'. ``model``
    is optional (existing callers predate it) but should be passed whenever
    known -- see :func:`_flake_section`.
    """
    numbered = _numbered_sheet_keys(run_dict)
    if numbered:
        base = '+'.join(str(run_dict[k].get('mat', 'unknown')) for k in numbered)
    else:
        base = str(primary_sheet(run_dict).get('mat', 'unknown'))
    flake_suffix = flake_material_token(run_dict, model)
    if flake_suffix is not None:
        return f"{base}+flake-{flake_suffix}"
    return base


def _truthy(value: Any) -> bool:
    """A bool as the model reads it; ``.ini`` values arrive as strings."""
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def interlayer_dir_token(style: str, comparison: bool) -> Optional[str]:
    """Directory segment for a non-default interlayer style, else ``None``.

    ``('lj', False)`` is today's behaviour and keeps today's paths; anything
    else gets ``il-<style>`` (underscores dropped) plus ``-cmp`` in comparison
    mode, so a style sweep can never collide.
    """
    if style == 'lj' and not comparison:
        return None
    token = f"il-{str(style).replace('_', '')}"
    return f"{token}-cmp" if comparison else token


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
    mat = material_token(run_dict, model)
    sheet = primary_sheet(run_dict)
    x = sheet.get('x', 100)
    y = sheet.get('y', 100)
    temp = run_dict['general'].get('temp', 300)
    size_token = f"{format_dimension_token(x)}x_{format_dimension_token(y)}y"

    # Potential token (spec §13.4): always emitted when the sheet names a
    # potential, so paths are self-describing and a pot_path sweep can never
    # collide (the LOOPABILITY rule: sweepable only if it changes the
    # directory name). LEGACY_2026 layouts without the segment stay readable.
    pot_segment = None
    pot_path = sheet.get('pot_path')
    if pot_path:
        from ..potentials.utils import pot_dir_token  # pylint: disable=import-outside-toplevel
        # deferred: potentials.utils imports the UFF table; only pot_path-bearing configs pay for it
        pot_segment = pot_dir_token(sheet.get('pot_type', 'sw'), pot_path)

    general = run_dict['general']
    il_segment = interlayer_dir_token(
        general.get('interlayer_style', 'lj'),
        _truthy(general.get('interlayer_comparison', False)))
    if il_segment:
        pot_segment = f"{pot_segment}/{il_segment}" if pot_segment else il_segment

    top_dir = model  # 'afm' | 'sheetonsheet' | 'pes_tip' | 'pes_sheet'
    if model in AFM_LIKE_MODELS:
        tip_mat = run_dict.get('tip', {}).get('mat', 'Si')
        tip_amorph = run_dict.get('tip', {}).get('amorph', 'c')
        tip_r = run_dict.get('tip', {}).get('r', 25)
        sub_mat = run_dict.get('sub', {}).get('mat', 'Si')
        sub_amorph = run_dict.get('sub', {}).get('amorph', 'a')

        sub_str = f"amorph-{sub_mat}" if sub_amorph == 'a' else sub_mat
        tip_str = f"amorph-{tip_mat}" if tip_amorph == 'a' else tip_mat

        base = (
            Path(top_dir) / mat / size_token /
            f"sub_{sub_str}_tip_{tip_str}_r{int(tip_r)}" /
            f"K{int(temp)}"
        )
        return base / pot_segment if pot_segment else base

    base = Path(top_dir) / mat / size_token / f"K{int(temp)}"
    return base / pot_segment if pot_segment else base


_SIZE_RE = re.compile(r'^(?P<x>-?[\d.]+(?:[eE][+-]?\d+)?)x_'
                      r'(?P<y>-?[\d.]+(?:[eE][+-]?\d+)?)y$')
_SUBTIP_RE = re.compile(r'^sub_(?P<sub>.+?)_tip_(?P<tip>.+?)_r(?P<r>\d+)$')
_TEMP_RE = re.compile(r'^K(?P<t>-?\d+)$')
_LAYER_RE = re.compile(r'^L(?P<n>\d+)$')
_FLAKE_SEP = '+flake-'


@dataclass(frozen=True)
class RunTokens:
    """Everything :func:`output_dir_for` encoded into a run's directory path.

    ``family`` cannot be read from the path (a hetero run lives under
    ``sheetonsheet/``), so :func:`parse_run_path` sets it to ``model`` and
    :func:`src.run.tree.discover_runs` replaces it from ``provenance.json``.
    ``substrate`` and ``tip_material`` are the bare material name with any
    amorphous marker stripped (``'Si'``, never ``'amorph-Si'``/``'aSi'``);
    ``substrate_amorphous`` carries the substrate's flag separately (see
    :func:`_split_amorph_token`). There is no ``tip_amorphous`` -- nothing
    downstream needs the tip's amorphous flag as a standalone field.
    """
    model: str
    family: str
    materials: Tuple[str, ...]
    flake_material: Optional[str]
    size_x: float
    size_y: float
    substrate: Optional[str]
    substrate_amorphous: Optional[bool]
    tip_material: Optional[str]
    tip_radius: Optional[float]
    temperature: float
    layers: Optional[int]
    pot_token: Optional[str]
    rel_path: str
    interlayer_token: Optional[str] = None


_LEGACY_AMORPH_RE = re.compile(r'^a(?P<mat>[A-Z].*)$')
_AMORPH_PREFIX = 'amorph-'


def _split_amorph_token(token: str) -> Tuple[str, bool]:
    """Split a substrate/tip token into ``(material, is_amorphous)``.

    ``output_dir_for`` writes the literal prefix ``'amorph-'`` when
    ``amorph == 'a'`` (2026-09-21 user decision, Task 3b) -- an unambiguous
    encoding that needs no heuristic, so ``'amorph-Si'`` -> ``('Si', True)``
    and ``'h-MoS2'``/``'h_MoS2'`` (materials that carry their own structure
    prefix -- see the domain conventions) -> unchanged, ``False``.

    Legacy read: trees on disk from before Task 3b instead wrote a bare
    leading ``'a'`` followed by the material's own capitalised first letter
    (``'aSi'`` -> ``('Si', True)``). That legacy shape is recognised ONLY
    when the remainder starts with an uppercase letter -- the ``'^[a-z][-_]'``
    heuristic Task 3 added to also catch prefixed 2D names (``'ah-MoS2'``)
    was guesswork and is gone: ``'ah-MoS2'`` is now just a crystalline
    material literally named that (no such material exists; that is
    accepted).
    """
    if token.startswith(_AMORPH_PREFIX):
        return token[len(_AMORPH_PREFIX):], True
    legacy = _LEGACY_AMORPH_RE.match(token)
    if legacy:
        return legacy.group('mat'), True
    return token, False


def _split_materials(token: str) -> Tuple[Tuple[str, ...], Optional[str]]:
    """``'h-MoS2+h-WS2+flake-h-WSe2'`` -> ``(('h-MoS2', 'h-WS2'), 'h-WSe2')``."""
    flake = None
    base = token
    if _FLAKE_SEP in token:
        base, flake = token.split(_FLAKE_SEP, 1)
    return tuple(base.split('+')), flake


def parse_run_path(rel_path: str) -> RunTokens:
    """Parse a run directory path (relative to the simulation root).

    The exact inverse of :func:`output_dir_for`, plus the optional trailing
    ``L{n}`` level the AFM-family builders add. Raises ``ValueError`` on
    anything it cannot account for -- silently returning partial tokens is
    how the old duplicated parser mis-filed hetero runs. A substrate or tip
    token is amorphous iff it starts with ``'amorph-'``, or (legacy trees,
    see :func:`_split_amorph_token`) is a bare ``'a'`` + capitalised letter;
    either way the token's material is the remainder.
    """
    text = str(rel_path).replace('\\', '/').strip('/')
    parts = PurePosixPath(text).parts
    if len(parts) < 4:
        raise ValueError(f"not a run path (too short): {rel_path!r}")

    model = parts[0]
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r} in {rel_path!r}; known: {MODELS}")

    materials, flake_material = _split_materials(parts[1])

    size = _SIZE_RE.fullmatch(parts[2])
    if size is None:
        raise ValueError(f"unparseable size token {parts[2]!r} in {rel_path!r}")

    idx = 3
    substrate = tip_material = None
    substrate_amorphous: Optional[bool] = None
    tip_radius: Optional[float] = None
    if model in AFM_LIKE_MODELS:
        subtip = _SUBTIP_RE.fullmatch(parts[idx])
        if subtip is None:
            raise ValueError(
                f"unparseable sub/tip token {parts[idx]!r} in {rel_path!r}")
        substrate, substrate_amorphous = _split_amorph_token(subtip.group('sub'))
        tip_material, _tip_amorphous = _split_amorph_token(subtip.group('tip'))
        tip_radius = float(subtip.group('r'))
        idx += 1

    if idx >= len(parts):
        raise ValueError(f"no K<temp> level in {rel_path!r}")
    temp = _TEMP_RE.fullmatch(parts[idx])
    if temp is None:
        raise ValueError(f"unparseable temperature token {parts[idx]!r} in {rel_path!r}")
    idx += 1

    pot_token = None
    if idx < len(parts) and parts[idx].startswith('pot_'):
        pot_token = parts[idx]
        idx += 1

    interlayer_token = None
    if idx < len(parts) and parts[idx].startswith('il-'):
        interlayer_token = parts[idx]
        idx += 1

    layers = None
    if idx < len(parts):
        layer = _LAYER_RE.fullmatch(parts[idx])
        if layer is None:
            raise ValueError(f"unexpected path level {parts[idx]!r} in {rel_path!r}")
        layers = int(layer.group('n'))
        idx += 1

    if idx != len(parts):
        raise ValueError(f"trailing path levels {parts[idx:]} in {rel_path!r}")

    return RunTokens(
        model=model,
        family=model,
        materials=materials,
        flake_material=flake_material,
        size_x=float(size.group('x')),
        size_y=float(size.group('y')),
        substrate=substrate,
        substrate_amorphous=substrate_amorphous,
        tip_material=tip_material,
        tip_radius=tip_radius,
        temperature=float(temp.group('t')),
        layers=layers,
        pot_token=pot_token,
        rel_path=text,
        interlayer_token=interlayer_token,
    )
