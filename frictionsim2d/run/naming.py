"""Output-tree naming: tokens, directory layout, loop classification; owns the model vocabulary."""
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
    ('2D', 'pot_path'): 'sweep',
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
#: Derived from the ``'in_file'`` entries of :data:`LOOPABILITY`; its consumer is
#: :func:`~frictionsim2d.run.expand.expand_config_sweeps`.
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


def _numbered_keys(run_dict: Dict[str, Any], prefix: str) -> List[str]:
    """Every ``<prefix>N`` key in ``run_dict`` ('2D-', 'Flake-'), sorted by N.

    A non-numeric suffix sorts last rather than raising.
    """
    return sorted(
        (k for k in run_dict if isinstance(k, str) and k.startswith(prefix)),
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
    numbered = _numbered_keys(run_dict, '2D-')
    if numbered:
        return run_dict[numbered[0]]
    raise KeyError('2D')


def _flake_section(run_dict: Dict[str, Any], model: Optional[str]) -> Any:
    """The raw single ``[Flake]``/``[flake]`` section for ``model``.

    The key is ``'flake'`` for the AFM family and ``'Flake'`` for sheet-on-sheet;
    with no ``model`` both are checked.
    """
    if model is not None:
        return run_dict.get('flake' if model in AFM_LIKE_MODELS else 'Flake')
    flake = run_dict.get('Flake')
    return flake if flake is not None else run_dict.get('flake')


def flake_material_token(run_dict: Dict[str, Any], model: Optional[str] = None) -> Optional[str]:
    """Directory token for the flake section(s), or ``None`` when the flake does
    not change the built system's material.

    - ``[Flake-N]`` sections (hetero flake) always contribute: ``mat1+mat2+...``.
    - A sheet-on-sheet ``[Flake]`` without ``mat`` contributes nothing; the
      builder reuses the substrate's material.
    - A single flake with an explicit ``mat`` (sheet-on-sheet ``[Flake]`` or AFM
      ``[flake]``) contributes that material.
    """
    numbered = _numbered_keys(run_dict, 'Flake-')
    if numbered:
        return '+'.join(str(run_dict[k].get('mat', 'unknown')) for k in numbered)
    flake = _flake_section(run_dict, model)
    if isinstance(flake, dict) and flake.get('mat') is not None:
        return str(flake['mat'])
    return None


def material_token(run_dict: Dict[str, Any], model: Optional[str] = None) -> str:
    """Directory token for the material(s): 'mat' or 'mat1+mat2+...'.

    Every numbered sheet section contributes its 'mat' in numeric order. The
    join is '+' because material names contain '-' (e.g. 'h-MoS2'). A flake
    with its own material (see :func:`flake_material_token`) appends
    ``+flake-<mat...>``; pass ``model`` whenever it is known.
    """
    numbered = _numbered_keys(run_dict, '2D-')
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
    """The output directory a run config builds into, relative to the simulation root.

    Pure naming, so callers that build nothing (the GUI's Generate preview) can
    predict a run's directory. ``x``/``y`` come from :func:`primary_sheet`.
    """
    mat = material_token(run_dict, model)
    sheet = primary_sheet(run_dict)
    x = sheet.get('x', 100)
    y = sheet.get('y', 100)
    temp = run_dict['general'].get('temp', 300)
    size_token = f"{format_dimension_token(x)}x_{format_dimension_token(y)}y"

    # Emitted whenever the sheet names a potential.
    pot_segment = None
    pot_path = sheet.get('pot_path')
    if pot_path:
        from ..potentials.utils import pot_dir_token  # pylint: disable=import-outside-toplevel
        # deferred: potentials.utils imports the UFF table; only pot_path-bearing configs pay for it
        # An MLIP sheet's D3 choice rides in the slug (pot_mace-d3_<stem>), so
        # with/without-D3 runs of one model never share a directory.
        pot_segment = pot_dir_token(sheet.get('pot_type', 'sw'), pot_path,
                                    dispersion=sheet.get('dispersion'))

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
    :func:`frictionsim2d.run.tree.discover_runs` replaces it from ``provenance.json``.
    ``substrate`` and ``tip_material`` are the bare material name with any
    amorphous marker stripped; ``substrate_amorphous`` carries the flag.
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
    #: MLIP framework decoded from ``pot_token`` ('mace', 'deepmd'), else None.
    mlip_framework: Optional[str] = None
    #: ``'d3'`` or ``'none'`` for an MLIP run (from the ``-d3`` slug suffix);
    #: None for a classical run, whose path carries no dispersion choice.
    dispersion: Optional[str] = None


def mlip_flavor_from_pot_token(pot_token: Optional[str]) -> Optional[Tuple[str, str]]:
    """Decode ``(framework, dispersion)`` from a ``pot_<slug>_<stem>`` segment.

    The inverse of the MLIP branch of
    :func:`frictionsim2d.potentials.utils.pot_dir_token`: ``pot_mace_<stem>`` ->
    ``('mace', 'none')``, ``pot_deepmd-d3_<stem>`` -> ``('deepmd', 'd3')``.
    Slugs never contain ``'_'`` ('/' is mapped to '-'), so the slug is the
    text between ``pot_`` and the next underscore. A classical slug, or
    anything that is not a pot token, gives ``None``.

    """
    if not pot_token or not pot_token.startswith('pot_'):
        return None
    slug = pot_token[len('pot_'):].split('_', 1)[0]
    from ..core.mlip import MLIP_REGISTRY  # pylint: disable=import-outside-toplevel
    dispersion = 'none'
    if slug.endswith('-d3'):
        slug, dispersion = slug[:-len('-d3')], 'd3'
    if slug not in MLIP_REGISTRY:
        return None
    return slug, dispersion


_LEGACY_AMORPH_RE = re.compile(r'^a(?P<mat>[A-Z].*)$')
_AMORPH_PREFIX = 'amorph-'


def _split_amorph_token(token: str) -> Tuple[str, bool]:
    """Split a substrate/tip token into ``(material, is_amorphous)``.

    The ``amorph-`` prefix marks an amorphous material (``'amorph-Si'`` ->
    ``('Si', True)``). A bare ``'a'`` followed by a capital (``'aSi'``) is also
    read as amorphous; anything else is returned unchanged.
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
    anything it cannot account for rather than returning partial tokens.
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

    flavor = mlip_flavor_from_pot_token(pot_token)
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
        mlip_framework=flavor[0] if flavor else None,
        dispersion=flavor[1] if flavor else None,
    )
