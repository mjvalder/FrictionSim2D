"""Output-tree naming: tokens, directory layout, loop classification.

Split out of run.py. Pure string/path logic -- it imports no builder and no
simulation machinery, which is what lets src/run/ later split without a cycle.

It also owns AFM_LIKE_MODELS / SHEET_LIKE_MODELS. Those are model-name
vocabulary rather than simulation machinery, and the flake-section rules here
need them, so they live in the leaf module and the runner imports them from
here -- not the other way round, which would be a cycle.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .utils import format_dimension_token

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
    ('2D', 'layers'): 'layer',
    ('general', 'force'): 'in_file',
    ('general', 'scan_angle'): 'in_file',
    ('general', 'scan_angle_force'): 'in_file',
    ('general', 'pressure'): 'in_file',
    ('general', 'scan_speed'): 'in_file',
}


#: Keys the LAMMPS scripts loop over internally; excluded from sweep expansion.
LAMMPS_LOOP_PARAMS = frozenset({'force', 'scan_angle', 'pressure', 'scan_speed'})


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

    A flake config -- :class:`~src.builders.flake_slide.FlakeSlideSimulation`
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
    (:meth:`~src.builders.flake_slide.FlakeSlideSimulation._filled_flake_config`).
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
        from ..potentials.utils import pot_dir_token
        pot_segment = pot_dir_token(sheet.get('pot_type', 'sw'), pot_path)

    top_dir = model  # 'afm' | 'sheetonsheet' | 'pes_tip' | 'pes_sheet'
    if model in AFM_LIKE_MODELS:
        tip_mat = run_dict.get('tip', {}).get('mat', 'Si')
        tip_amorph = run_dict.get('tip', {}).get('amorph', 'c')
        tip_r = run_dict.get('tip', {}).get('r', 25)
        sub_mat = run_dict.get('sub', {}).get('mat', 'Si')
        sub_amorph = run_dict.get('sub', {}).get('amorph', 'a')

        sub_str = f"{sub_amorph}{sub_mat}" if sub_amorph == 'a' else sub_mat
        tip_str = f"{tip_amorph}{tip_mat}" if tip_amorph == 'a' else tip_mat

        base = (
            Path(top_dir) / mat / size_token /
            f"sub_{sub_str}_tip_{tip_str}_r{int(tip_r)}" /
            f"K{int(temp)}"
        )
        return base / pot_segment if pot_segment else base

    base = Path(top_dir) / mat / size_token / f"K{int(temp)}"
    return base / pot_segment if pot_segment else base
