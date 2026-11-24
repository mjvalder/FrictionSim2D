"""Read and process friction simulation output data.

One walk, through `frictionsim2d.run.tree.discover_runs`: a run is one
`provenance.json`, its directory is parsed by `frictionsim2d.run.naming.parse_run_path`,
and each of its load points becomes one flat `RunFrames`. The eight-deep
nested dict this replaced had no dimension for the potential or the
temperature, so two runs differing only in those collapsed onto one leaf and
the last one walked won, silently.

Exports the schema-3 `output_full_<size>.json` and `summary_<size>.csv`; every
statistic in the latter comes from `frictionsim2d.data.models.summarise_run`, the single
COF on the store path.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from frictionsim2d.core.config import IntegritySettings
from frictionsim2d.core.observables import (AFM_LIKE, FAMILIES, SHEET_LIKE, expected_tier_rows,
                                            rename_map)
from frictionsim2d.data.models import EV_A_TO_NN, compute_derived_columns

from .integrity import (SKIP_DETAILS, IntegrityVerdict, check_point,
                        writes_end_snapshot)
from .integrity_rulings import apply_rulings, load_rulings
from .post_channels import post_channels_for
from .run_provenance import (point_provenance, pool_provenance,
                             run_provenance)
from .run_status import (GRID_FAMILIES, LEGACY_TRUNCATION_DETAIL, STATUSES,
                         PointKey, classify,
                         expected_chains, halt_detail, legacy_truncated,
                         load_kind_for_family, mark_not_reached, point_key,
                         read_halt_registry, read_run_config)

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Results file layout does not match any known schema (spec §1.1).

    Raised instead of silently mis-aligning: a header-less file whose column
    count differs from the LEGACY_2026 positional list is corrupt or from an
    unknown template revision, and parsing it positionally would shift every
    value.
    """


#: eV/A -> nN (the legacy schema-1 lateral-force correction).
LF_SCHEMA1_CORRECTION = EV_A_TO_NN
#: Output-schema version this reader understands.
LF_SCHEMA_CURRENT = 2
#: Column names matching `fix fc_ave` output order in afm/slide.lmp.
AFM_FILE_COLUMNS = [
    'time', 'nf', 'lfx', 'lfy', 'comx', 'comy', 'comz',
    'tipx', 'tipy', 'tipz',
]


def parse_lf_schema(line1: str) -> int:
    """Output-schema version from a results file's first line.

    Parsed as an INTEGER, not matched as the substring 'schema=2'. A
    substring test treats every future version as schema 1 and silently
    applies the legacy unit correction to a file that never needed it --
    so bumping the marker would corrupt data rather than fail loudly.
    Absent marker means the file predates the marker entirely: schema 1.
    """
    m = re.search(r'schema=(\d+)', line1 or '')
    return int(m.group(1)) if m else 1


def branch_column(df):
    """The reaction-force-x column, under whichever name it parsed to.

    The registry declares `react_x` with rename_to='v_sx', so a
    header-carrying file renames the native away. Retained files parsed
    before that declaration, and any file read through a path that skips
    the rename, still carry 'v_sx'. Both are accepted; neither is
    guessed at.
    """
    for candidate in ('react_x', 'v_sx'):
        if candidate in df.columns:
            return df[candidate]
    return None


def lateral_columns(df):
    """The (x, y) lateral-force column names present in the frame.

    Symmetric with `branch_column`: the registry renames the sheet's
    native `v_xfrict`/`v_yfrict` to `lfx`/`lfy` for window/peratom tiers
    but to `raw_lfx`/`raw_lfy` for the raw tier, so both name pairs must
    be checked. Returns None when neither is present -- there is no
    lateral force column to correct at all.
    """
    if 'lfx' in df.columns:
        return ('lfx', 'lfy')
    if 'raw_lfx' in df.columns:
        return ('raw_lfx', 'raw_lfy')
    return None


def is_lj_branch(lfx, sx) -> bool:
    """True when a pre-schema-2 sheet frame needs the unit correction.

    On the many-body branch the template defines ``xfrict`` AS ``v_sx``,
    so LAMMPS writes both columns from one value and they are byte-
    identical. Any difference means the LJ branch, whose ``compute
    group/group`` force is in eV/A while the load is already nN.

    Defined once and used by BOTH the reader (whole column) and
    ``audit_lf_units`` (a one-row sample), so the two cannot drift apart.
    """
    return not pd.Series(list(lfx)).equals(pd.Series(list(sx)))


def correct_legacy_lf_schema1(df, filepath, family: str):
    """Rescale a pre-``schema=2`` sheet frame's lateral force to nN.

    Frozen legacy path: every sheet-stack family (hetero and flake included)
    is corrected, as the old `'sheet'` traversal vocabulary did.
    """
    if df.attrs.get('lf_corrected'):
        # Re-entry guard: scaling twice gives 1.602^2.
        return df
    df.attrs['lf_corrected'] = False
    schema = df.attrs.get('lf_schema', 1)
    if family not in SHEET_LIKE or schema != 1:
        if schema > LF_SCHEMA_CURRENT or schema < 1:
            logger.warning(
                "%s: output schema %s is not one this reader understands "
                "(known: 1..%s). Lateral force left untouched; upgrade "
                "the reader rather than trusting this frame.",
                filepath, schema, LF_SCHEMA_CURRENT)
        return df
    cols = lateral_columns(df)
    if cols is None:
        return df                       # no lateral force to correct at all
    lfx_name, lfy_name = cols
    branch = branch_column(df)
    if branch is None:
        logger.warning(
            "%s: pre-schema-2 sheet file carries %s but no reaction-force "
            "column (react_x/v_sx), so the LJ/many-body branch cannot be "
            "identified. Lateral force left AS WRITTEN; if this run used "
            "an lj/cut interlayer potential its COF is a factor %s too "
            "small.", filepath, lfx_name, LF_SCHEMA1_CORRECTION)
        return df
    if not is_lj_branch(df[lfx_name], branch):
        return df                       # many-body branch: already nN
    df[lfx_name] = df[lfx_name] * LF_SCHEMA1_CORRECTION
    if lfy_name in df.columns:
        df[lfy_name] = df[lfy_name] * LF_SCHEMA1_CORRECTION
    df.attrs['lf_corrected'] = True
    logger.warning(
        "%s: pre-schema-2 sheet file -- lateral force was eV/A against a "
        "nN load, so COF was a factor %s too small. Multiplied %s/%s by "
        "%s. Re-run this simulation to retire the correction.",
        filepath, LF_SCHEMA1_CORRECTION, lfx_name, lfy_name,
        LF_SCHEMA1_CORRECTION)
    return df


def tier_of(filename: str) -> str:
    """`friction_`/`fc_ave_` -> window, `raw_` -> raw, `peratom_` -> peratom."""
    return ('raw' if filename.startswith('raw_')
            else 'peratom' if filename.startswith('peratom_')
            else 'window')


#: Slide-results filename patterns; `frictionsim2d.aiida.ingest` groups load
#: points with them (the reader itself uses `_RESULTS_RE`/`parse_results_name`).
FILE_PATTERN_TIP = re.compile(r'fc_ave_slide_(\d+\.?\d*)nN_(\d+)angle_(\d+)ms_l(\d+)')
# New AFM tip naming written by the current builder, e.g.
# ``friction_f30_a0.0_s2_layer2.txt``. Groups match FILE_PATTERN_TIP:
# (load, angle, speed, layer). Load is always a force (nN) for tip sims.
# The angle is signed: LAMMPS substitutes the scan angle verbatim into
# results_file_pattern, so a -15 deg scan writes friction_f5_a-15_s1_...
# An unsigned pattern skips those files in silence -- the run simply
# disappears from every analysis.
FILE_PATTERN_TIP_ALT = re.compile(
    r'friction_f(\d+\.?\d*)_a(-?\d+\.?\d*)_s(\d+\.?\d*)_layer(\d+)'
)
FILE_PATTERN_SHEET = re.compile(
    r'fc_ave_slide_(\d+\.?\d*)(GPa|nN)_(\d+\.?\d*)angle_(\d+\.?\d*)ms'
)
# Signed angle, as in FILE_PATTERN_TIP_ALT above.
FILE_PATTERN_SHEET_ALT = re.compile(
    r'friction_p(\d+\.?\d*)_a(-?\d+\.?\d*)_s(\d+\.?\d*)'
)


def read_results_file(filepath: Path, family: str,
                      n_layers: int = 2) -> pd.DataFrame:
    """Parse one LAMMPS results file into a labelled DataFrame.

    Header-first: the `# TimeStep v_...` line names the native columns and the
    observable registry's per-tier rename map turns them into channel names.
    The tier comes from the filename prefix. `n_layers` names the per-layer
    columns (`v_comz_L5` -> `comz_L5`); it defaults to 2 -- the pair REGISTRY
    carries -- so callers that do not know the layer count (the AiiDA store
    re-parsing a stored file) behave exactly as before. The map is
    `rename_map`'s own, never a second copy kept here. Raises `SchemaError` when a header-less
    file's width does not match `AFM_FILE_COLUMNS`.

    Sets `df.attrs['lf_schema']` and applies `correct_legacy_lf_schema1`, so a
    frame that comes out of here is directly comparable with one the DataReader
    produced from the same bytes -- which is what lets the store and the reader
    share one parser instead of drifting.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; known: {FAMILIES}")

    header_lines: list = ['', '']
    header_cols: list | None = None
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
            header_lines = [next(fh, ''), next(fh, '')]
        if len(header_lines) > 1 and header_lines[1].startswith('#'):
            tokens = header_lines[1].lstrip('#').strip().split()
            if tokens:
                header_cols = ['time' if tok == 'TimeStep' else tok
                               for tok in tokens]
    except OSError:
        pass

    tier = tier_of(Path(filepath).name)

    if header_cols:
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=header_cols,
                         skiprows=2)
        df.rename(columns=rename_map(family, tier, n_layers), inplace=True)
        df.attrs['lf_schema'] = parse_lf_schema(header_lines[0])
        return correct_legacy_lf_schema1(df, filepath, family)

    # LEGACY_2026 positional fallback -- AFM stack only. The sheet corpus was
    # surveyed on 2026-08-29: zero non-empty headerless files. Guessing is
    # worse than refusing.
    if family not in AFM_LIKE:
        raise SchemaError(
            f"{filepath}: no column header and no positional schema for "
            f"family {family!r}")
    try:
        probe = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=2,
                            nrows=1)
    except pd.errors.EmptyDataError as exc:
        raise SchemaError(f"{filepath}: no data rows") from exc
    if probe.shape[1] != len(AFM_FILE_COLUMNS):
        raise SchemaError(
            f"{filepath}: {probe.shape[1]} columns, LEGACY_2026 expects "
            f"{len(AFM_FILE_COLUMNS)} and no header is present")
    df = pd.read_csv(filepath, sep=r'\s+', header=None,
                     names=AFM_FILE_COLUMNS, skiprows=2)
    df.attrs['lf_schema'] = 1          # header-less files predate the marker
    return correct_legacy_lf_schema1(df, filepath, family)


@dataclass(frozen=True)
class LoadPoint:
    """One results FILE's identity: which load point and which tier it holds."""
    load_kind: str            # 'force' | 'pressure' | 'none'
    load: float
    angle: float
    speed: float
    tier: str
    layer: Optional[int]


#: Every results filename the campaign has ever written, in one pattern.
#: Group `kind` is 'f' (nN) or 'p' (GPa) for the current names; the two legacy
#: `fc_ave_slide_` forms carry their unit as a suffix instead. Angles are
#: SIGNED: LAMMPS substitutes the scan angle verbatim, so a -15 deg scan writes
#: `friction_f5_a-15_s1_...` and an unsigned pattern drops the run in silence.
#: `.txt` is optional: the sheet, hetero and flake builders write
#: `friction_p${pressure}_a${a}_s${speed}` with no extension.
_RESULTS_RE = re.compile(
    r'^(?:(?P<tier>friction|raw|peratom)_(?P<kind>[fp])(?P<load>-?\d+\.?\d*)'
    r'_a(?P<angle>-?\d+\.?\d*)_s(?P<speed>-?\d+\.?\d*)'
    r'(?:_layer(?P<layer>\d+))?'
    r'|fc_ave_slide_(?P<lload>\d+\.?\d*)(?P<lunit>nN|GPa)'
    r'_(?P<langle>-?\d+\.?\d*)angle_(?P<lspeed>\d+\.?\d*)ms'
    r'(?:_l(?P<llayer>\d+))?)(?:\.txt)?$')

_TIER_BY_PREFIX = {'friction': 'window', 'raw': 'raw', 'peratom': 'peratom'}

#: A PES scan's grid IS its window tier -- one grid per run, no load point to
#: speak of. The store files `pes_scan` under `'window'` the same way; two
#: readers, one rule.
_PES_GRID_NAME = 'pes_scan.csv'

_PES_POINT = LoadPoint(load_kind='none', load=0.0, angle=0.0, speed=0.0,
                       tier='window', layer=None)


def parse_results_name(name: str) -> Optional[LoadPoint]:
    """The load point a results filename names, or None if it is not one.

    `'none'` is a third `load_kind`: a PES scan is not run at a load at all,
    it is one grid, and calling its absent load a force of 0.0 nN would put a
    fictitious point on every load axis.
    """
    if name == _PES_GRID_NAME:
        return _PES_POINT
    match = _RESULTS_RE.match(name)
    if match is None:
        return None
    if match.group('tier'):
        layer = match.group('layer')
        return LoadPoint(
            load_kind='force' if match.group('kind') == 'f' else 'pressure',
            load=float(match.group('load')),
            angle=float(match.group('angle')),
            speed=float(match.group('speed')),
            tier=_TIER_BY_PREFIX[match.group('tier')],
            layer=int(layer) if layer else None)
    layer = match.group('llayer')
    return LoadPoint(
        load_kind='force' if match.group('lunit') == 'nN' else 'pressure',
        load=float(match.group('lload')),
        angle=float(match.group('langle')),
        speed=float(match.group('lspeed')),
        tier='window',
        layer=int(layer) if layer else None)


@dataclass
class RunFrames:
    """One load point of one run: its tiers, flat.

    `locator` is the `RunLocator` `discover_runs` built: it carries `data_dir`,
    `results_dir` and `provenance_file`, which `RunTokens` alone cannot.

    `status` is one of `run_status.STATUSES`; a point that never wrote a file
    (`no_output`/`not_reached`) or whose window failed to parse (`unreadable`)
    has `tiers == {}`. `expected_rows` is what the run's config says each tier
    writes over a finished slide (`{}` when unknown); `config` is the run's
    parsed `provenance/config.json`, or None.
    """
    locator: Any                        # frictionsim2d.run.tree.RunLocator
    load_kind: str                      # 'force' | 'pressure' | 'none'
    load: float
    angle: float
    speed: float
    n_layers: int
    tiers: Dict[str, pd.DataFrame]
    source_files: Dict[str, Path]
    post: Dict[str, float] = field(default_factory=dict)
    status: str = 'complete'
    status_detail: str = ''
    expected_rows: Dict[str, int] = field(default_factory=dict)
    config: Optional[dict] = None
    #: The stage-3 verdict, on every point when the check is on. It changes
    #: only a `complete` point's status; the others pass through.
    integrity: Optional[IntegrityVerdict] = None

    @property
    def n_rows(self) -> Dict[str, int]:
        """`{tier: rows}` actually read."""
        return {tier: len(frame) for tier, frame in sorted(self.tiers.items())}

    @property
    def tokens(self):
        """The run's `RunTokens`."""
        return self.locator.tokens

    @property
    def family(self) -> str:
        """The run's family token (`afm`, `sheetonsheet`, ...)."""
        return self.locator.tokens.family

    @property
    def material(self) -> str:
        """The downstream material key: `h_MoS2`, `h_MoS2+h_WS2`.

        Underscored, because every material filter and colour map in
        `postprocessing/` and the ML feature tables key on that spelling.
        """
        token = '+'.join(self.tokens.materials)
        if self.tokens.flake_material:
            token = f'{token}+flake-{self.tokens.flake_material}'
        return token.replace('-', '_')

    @property
    def size(self) -> str:
        """`100x100y` -- the per-size export key."""
        def fmt(v: float) -> str:
            return str(int(v)) if float(v).is_integer() else f'{v:g}'
        return f'{fmt(self.tokens.size_x)}x{fmt(self.tokens.size_y)}y'

    @property
    def window(self) -> pd.DataFrame:
        """The window tier, or an empty frame when the run has none."""
        return self.tiers.get('window', pd.DataFrame())

    def frames_by_tier(self) -> Dict[str, pd.DataFrame]:
        """What `summarise_run` consumes."""
        return dict(self.tiers)


#: Version of the export document. v1 (no 'schema' key) is not read by the
#: plot stack; regenerate with `postprocess read --export`.
EXPORT_SCHEMA = 3

#: Units of the columns this module derives rather than parses.
_DERIVED_UNITS = {'lateral_force': 'nN', 'cof': '1'}


def units_for_columns(columns, family: str) -> Dict[str, str]:
    """`{column: unit}` for the columns present, from the registry.

    `time` is a LAMMPS timestep index, which no Observable declares; it is
    named 'step' here so a consumer never has to special-case a missing key.
    Columns with no declared unit are omitted rather than guessed at.

    No family carries a unit CAVEAT any more. An earlier draft flagged the
    flake's `nf` as an interfacial adhesion rather than an applied load;
    flake slides now apply a real normal load (`fix force flake addforce`),
    so its `nf` and `cof` are the same quantities as the sheet's.
    """
    from ..core.observables import REGISTRY  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    by_name = {ob.name: ob.units for ob in REGISTRY.values()
               if family in ob.sim_types and ob.units}
    out: Dict[str, str] = {}
    for col in columns:
        if col == 'time':
            out['time'] = 'step'
        elif col in _DERIVED_UNITS:
            out[col] = _DERIVED_UNITS[col]
        elif col in by_name:
            out[col] = by_name[col]
    return out


#: The identity half of a summary row, in CSV order.
SUMMARY_IDENTITY_COLUMNS = (
    'model', 'family', 'material', 'materials', 'flake_material', 'size',
    'size_x', 'size_y', 'substrate', 'substrate_amorphous', 'tip_material',
    'tip_radius', 'temperature', 'layers', 'potential', 'rel_path',
    'load_kind', 'load', 'angle', 'speed', 'tiers', 'status',
    'status_detail')

#: The two alternative lateral-force definitions the sheet family carries.
#: NaN when the frame has no react_x/react_y (sx_sy) or drive_fx/drive_fy
#: (fx_fy) -- never silently aliased to mean_lf, which is what the db://
#: reader used to do.
SUMMARY_LF_COLUMNS = ('mean_lf_sx_sy', 'mean_lf_fx_fy')

#: The post-stream channels, all four, NaN where the family has none.
SUMMARY_POST_COLUMNS = ('gap_min', 'deflect_max', 'gap_min_il',
                        'corrugation_max')



#: Material-name prefix -> metadata bucket. `h_` hexagonal, `t_` T-phase,
#: `b_` buckled, `p_` phosphorene-like; anything else is 'other'.
_MATERIAL_TYPE_KEYS = ('b_type', 'h_type', 't_type', 'p_type', 'other')


def material_types(materials) -> Dict[str, List[str]]:
    """Group underscored material ids into the five plot-filter buckets."""
    out: Dict[str, List[str]] = {key: [] for key in _MATERIAL_TYPE_KEYS}
    for mat in materials:
        key = f"{str(mat).split('_', 1)[0]}_type"
        out[key if key in out else 'other'].append(str(mat))
    return {key: sorted(values) for key, values in out.items()}


def _config_sheet_layers(config: Optional[dict], family: str) -> Optional[int]:
    """The sheet layer count the run's `config.json` built, or None.

    hetero stacks `sheets[i].layers[0]` (`HeteroSheetOnSheetSimulation.
    n_layers`); sheetonsheet, flake and pes_sheet take `sheet.layers[0]`
    (the flake sits on top of a substrate of that many layers).
    """
    if not isinstance(config, dict):
        return None
    try:
        sheets = config.get('sheets') if family == 'hetero' else None
        if sheets:
            return sum(int(sheet['layers'][0]) for sheet in sheets)
        sheet = config.get('sheet')
        return int(sheet['layers'][0]) if isinstance(sheet, dict) else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def resolve_n_layers(locator, config: Optional[dict] = None) -> int:
    """How many sheet layers this run has.

    `RunTokens.layers` is the `L{n}` path level, and `output_dir_for` emits one
    only for the AFM-like models -- the AFM builders append it
    (`frictionsim2d/builders/afm/base.py:182`), the sheet-like ones never do. So for
    `sheetonsheet`, `hetero`, `flake` and `pes_sheet` the token is `None`, and
    defaulting it to 2 is exactly the defect P3 exists to fix, missed in the
    families that need it most: a `layers=[4]` sheet run's `v_comz_L3`/`L4` and
    its peratom `v_pe_sum_L3/L4`, `v_sigma_*_L3/L4` would stay un-renamed.

    The count IS on disk. `ProvenanceRecord.layer_type_map` is keyed
    `layer_1 ... layer_n` (`frictionsim2d/builders/sheetonsheet/base.py:62-83`,
    `frictionsim2d/builders/afm/base.py:961-969`), and `discover_runs` already opens that
    file to read `family`. It is `{}` for hetero and flake, whose layers are
    z-band regions rather than atom-type groups; those read the count off the
    run's `config` (`_config_sheet_layers`), and only without one log and fall
    back to 2, which is what they have always assumed.
    """
    if locator.tokens.layers is not None:
        return int(locator.tokens.layers)
    try:
        record = json.loads(locator.provenance_file.read_text(encoding='utf-8'))
        count = len([k for k in (record.get('layer_type_map') or {})
                     if str(k).startswith('layer_')])
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        logger.warning("Unreadable provenance for %s (%s); assuming 2 layers",
                       locator.tokens.rel_path, exc)
        return 2
    if count:
        return count
    configured = _config_sheet_layers(config, locator.tokens.family)
    if configured:
        return configured
    logger.info("%s has an empty layer_type_map (z-band layers); assuming 2",
                locator.tokens.rel_path)
    return 2


def _load_points(locator) -> Dict[Tuple[str, float, float, float],
                                  Dict[str, Tuple[Path, LoadPoint]]]:
    """Group a run's results files by load point, then by tier."""
    grouped: Dict[Tuple[str, float, float, float],
                  Dict[str, Tuple[Path, LoadPoint]]] = {}
    if not locator.results_dir.is_dir():
        return grouped
    for path in sorted(locator.results_dir.iterdir()):
        point = parse_results_name(path.name) if path.is_file() else None
        if point is None:
            continue
        key = (point.load_kind, point.load, point.angle, point.speed)
        grouped.setdefault(key, {})[point.tier] = (path, point)
    return grouped


def _join_detail(*parts: str) -> str:
    """Non-empty detail fragments, `'; '`-joined."""
    return '; '.join(part for part in parts if part)


def _grid_note(grid, config: Optional[dict], family: str) -> str:
    """Why a run's points are on-disk only; '' when the grid is known."""
    if grid is not None:
        return ''
    if family not in GRID_FAMILIES:
        return f'grid not modelled for {family}'
    if config is None:
        return 'grid unknown: no readable provenance/config.json'
    return (f'grid unknown: config has no general.'
            f'{load_kind_for_family(family)}')


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy and pandas types."""

    def default(self, o):  # pylint: disable=arguments-renamed
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            out = {'columns': o.columns.tolist(), 'data': o.values.tolist()}
            # Carry frame-level provenance -- notably lf_schema/lf_corrected.
            # Without this a unit-corrected frame serialises identically to a
            # stale pre-fix export, and since time_series_hash is computed
            # from (nf, lfx, lfy) the correction changes the hash, so the
            # dedup-by-hash check cannot recognise the re-ingested run.
            if o.attrs:
                out['attrs'] = dict(o.attrs)
            return out
        return super().default(o)


class DataReader:  # pylint: disable=too-many-instance-attributes
    """Reads and processes friction simulation data.

    Walks through a directory of simulation results, parses filenames and
    file paths to extract metadata, reads the time-series data from each
    valid file, calculates derived quantities (COF, lateral force), and
    stores it in a structured format.
    """

    def __init__(self, results_dir: str = 'results_110725_test', *,
                 skip_fraction: float = 0.2,
                 complete_only: bool = False, integrity: bool = True,
                 integrity_settings: Optional[IntegritySettings] = None
                 ) -> None:
        """Initialise the DataReader.

        Args:
            results_dir: A simulation root -- the directory holding `afm/`,
                `sheetonsheet/`, `indent/`, `pes_tip/`, `pes_sheet/`.
            skip_fraction: Transient fraction `summarise_run` discards.
            complete_only: Export and summarise only `complete` points (the
                old behaviour); by default every status is written. Also
                narrows `runs` to them.
            integrity: Run the stage-3 integrity check on every `complete`
                point (`postprocessing.integrity`); off leaves the file-level
                statuses alone.
            integrity_settings: Its thresholds; default `IntegritySettings()`.
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'outputs'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_fraction = skip_fraction
        self.complete_only = complete_only
        self.integrity_settings: Optional[IntegritySettings] = (
            (integrity_settings or IntegritySettings()) if integrity else None)
        self._run_blocks: Dict[Path, dict] = {}
        #: Why a line of `integrity_rulings.yaml` did not apply.
        self.ruling_warnings: List[str] = []

        #: Every expected load point, any status. `runs` keeps its old meaning
        #: -- the points with frames -- for `indent.py` and existing callers.
        self.points: List[RunFrames]
        chains: List[Tuple[List[PointKey], Dict[PointKey, RunFrames]]]
        self.points, chains = self._read_runs()
        self.runs: List[RunFrames] = [p for p in self.points if p.tiers]
        self._apply_legacy_truncation()
        for chain, by_key in chains:
            status = {key: point.status for key, point in by_key.items()}
            mark_not_reached(chain, status)
            for key, point in by_key.items():
                point.status = status[key]
        if self.integrity_settings is not None:
            self._check_integrity(self.integrity_settings)
            self.ruling_warnings = apply_rulings(
                self.points, load_rulings(self.results_dir), self.results_dir)
        if complete_only:
            self.runs = [run for run in self.runs if run.status == 'complete']
        self.ntimestep: int = max((len(r.window) for r in self.runs), default=0)
        self.time_series: List[int] = self._time_series()
        self.incomplete_files: Dict[str, List[str]] = {}
        self.incomplete_materials: Dict[str, set] = {}
        self._flag_incomplete()
        self.metadata: Dict[str, Any] = self._build_metadata()

    def _check_integrity(self, settings: IntegritySettings) -> None:
        """Stage 3 on every point; on a `complete` one the verdict becomes
        the status, every other status passes through (`verdict`).

        Runs after `mark_not_reached`, so a point that fails here never turns
        a later point of its chain `not_reached`: the chain did reach it.
        `multi_speed` is per run -- more than one distinct speed among its
        points. Whether the run writes an end snapshot is read once per run.
        """
        by_run: Dict[Path, List[RunFrames]] = {}
        for point in self.points:
            by_run.setdefault(Path(point.locator.run_dir), []).append(point)
        for points in by_run.values():
            locator = points[0].locator
            record = self._run_block(points[0])
            multi_speed = len({p.speed for p in points}) > 1
            writes = writes_end_snapshot(locator.lammps_dir, locator.data_dir)
            for point in points:
                before = point.status
                verdict = check_point(point, record, settings,
                                      multi_speed=multi_speed,
                                      writes_snapshot=writes)
                point.integrity = verdict
                point.status = verdict.status
                notes = [d for d in verdict.not_applicable if d in SKIP_DETAILS]
                # A non-complete verdict's reasons open with its own status.
                parts = ([r for r in verdict.reasons if r != before]
                         + verdict.quarantine + notes)
                if parts:
                    point.status_detail = _join_detail(
                        point.status_detail, 'integrity: ' + '; '.join(parts))

    def _time_series(self) -> List[int]:
        """The `time` column of the first full-length window frame."""
        for run in self.runs:
            if len(run.window) == self.ntimestep and 'time' in run.window:
                return [int(t) for t in run.window['time'].to_list()]
        return []

    def _apply_legacy_truncation(self) -> None:
        """The old rule (`run_status.legacy_truncated`), for points whose
        config gives no expected length: every existing tree whose
        `config.json` carries no `settings`. PES and indent are never judged.
        """
        runs = self.runs
        for i in legacy_truncated([(r.family, r.size, len(r.window),
                                    bool(r.expected_rows), r.status) for r in runs]):
            runs[i].status = 'truncated'
            runs[i].status_detail = _join_detail(runs[i].status_detail,
                                                 LEGACY_TRUNCATION_DETAIL)

    def _flag_incomplete(self) -> None:
        """Record every point whose status is not `complete`.

        One line per point: the window file when there is one, else the run
        directory and the point, suffixed with the status. Unlike the old
        reader nothing is dropped -- flagging and dropping are different
        decisions, and only the first is this class's to make.
        """
        for point in self.points:
            if point.status == 'complete':
                continue
            where = point.source_files.get('window')
            line = (str(where) if where is not None else
                    f'{point.locator.run_dir} {point.load_kind}='
                    f'{point.load:g} a={point.angle:g} s={point.speed:g}')
            self.incomplete_files.setdefault(point.size, []).append(
                f'{line} [{point.status}]')
            self.incomplete_materials.setdefault(point.size, set()).add(
                point.material)

    def points_by_size(self) -> Dict[str, List[RunFrames]]:
        """Exported points grouped by `size` (only `complete` ones when
        `complete_only`)."""
        grouped: Dict[str, List[RunFrames]] = {}
        for point in self.points:
            if self.complete_only and point.status != 'complete':
                continue
            grouped.setdefault(point.size, []).append(point)
        return grouped

    def _build_metadata(self, runs: Optional[List[RunFrames]] = None,
                        counted: Optional[List[RunFrames]] = None
                        ) -> Dict[str, Any]:
        """The `metadata` block for `entries` (default: every point).

        The v2 keys are computed over the entries that have data (non-empty
        tiers), so their v2 meaning is unchanged; `status_counts` and the
        pooled `provenance` cover every entry. `counted` widens only
        `status_counts` (e.g. to every point of the size under `complete_only`).
        """
        entries = self.points if runs is None else runs
        counted = entries if counted is None else counted
        runs = [run for run in entries if run.tiers]
        loads: Dict[str, Dict[str, set]] = {'force': {}, 'pressure': {}}
        for run in runs:
            loads.setdefault(run.load_kind, {}).setdefault(
                str(run.load), set()).add(run.angle)
        materials = sorted({run.material for run in runs})
        return {
            'families': sorted({run.family for run in runs}),
            'models': sorted({run.tokens.model for run in runs}),
            'materials': materials,
            'material_types': material_types(materials),
            'potentials': sorted({run.tokens.pot_token for run in runs
                                  if run.tokens.pot_token}),
            'temperatures': sorted({run.tokens.temperature for run in runs}),
            'sizes': sorted({run.size for run in runs}),
            'layers': sorted({run.tokens.layers for run in runs
                              if run.tokens.layers is not None}),
            'speeds': sorted({run.speed for run in runs}),
            'substrates': sorted({run.tokens.substrate for run in runs
                                  if run.tokens.substrate}),
            'tip_materials': sorted({run.tokens.tip_material for run in runs
                                     if run.tokens.tip_material}),
            'tip_radii': sorted({run.tokens.tip_radius for run in runs
                                 if run.tokens.tip_radius is not None}),
            'loads_by_kind': {kind: {load: sorted(angles)
                                     for load, angles in sorted(by_load.items())}
                              for kind, by_load in loads.items()},
            'tiers': sorted({tier for run in runs for tier in run.tiers}),
            'ntimestep': self.ntimestep,
            'time_series': self.time_series,
            'provenance': pool_provenance(
                [(run.tokens.rel_path, self._run_block(run))
                 for run in entries]),
            'status_counts': {
                status: sum(1 for run in counted if run.status == status)
                for status in STATUSES},
            'integrity_thresholds': (
                None if self.integrity_settings is None
                else self.integrity_settings.model_dump()),
            # Which COF definition produced the summary numbers beside this
            # export. v1 files carried no such statement and the definition
            # changed under readers with no way to tell; this is that way.
            'summary_definition': {
                'cof': 'ratio_of_means: mean(LF)/mean(NF)',
                'skip_fraction': self.skip_fraction,
                'source': 'frictionsim2d.data.models.summarise_run',
            },
        }

    def _read_runs(self) -> Tuple[List[RunFrames],
                                  List[Tuple[List[PointKey],
                                             Dict[PointKey, RunFrames]]]]:
        """One walk with `discover_runs`, one `RunFrames` per load point.

        The points are the configured grid (`expected_chains` over the run's
        `provenance/config.json`) plus whatever is on disk. Returns them
        sorted, and each chain with its points so `not_reached` can be set
        once the legacy truncation pass has run.
        """
        from ..run.tree import discover_runs  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        points: List[RunFrames] = []
        chains: List[Tuple[List[PointKey], Dict[PointKey, RunFrames]]] = []
        for locator in discover_runs(self.results_dir):
            family = locator.tokens.family
            config = read_run_config(locator)
            n_layers = resolve_n_layers(locator, config)
            expected = expected_tier_rows(config, family) if config else {}
            grid = expected_chains(config, family)
            halts = read_halt_registry(locator)
            on_disk = {point_key(*key): (key, tiers)
                       for key, tiers in _load_points(locator).items()}
            grid_keys = list(dict.fromkeys(
                key for chain in grid or [] for key in chain))
            in_grid = set(grid_keys)
            note = _grid_note(grid, config, family)
            by_key: Dict[PointKey, RunFrames] = {}
            for key in dict.fromkeys(grid_keys + list(on_disk)):
                raw_key, tiers = on_disk.get(key, (key, {}))
                point = self._read_point(locator, raw_key, tiers, n_layers,
                                         expected, config)
                halted = halt_detail(halts, key)
                if halted is not None:
                    # The frames record the blow-up, not friction: kept for
                    # inspection, never summarised (`summary_rows`).
                    point.status, point.status_detail = 'halted', halted
                if grid is not None and key not in in_grid:
                    point.status_detail = _join_detail(
                        'not in configured grid', point.status_detail)
                point.status_detail = _join_detail(note, point.status_detail)
                by_key[key] = point
                points.append(point)
            for chain in grid or []:
                chains.append((chain, {key: by_key[key] for key in chain}))
        points.sort(key=lambda r: (r.tokens.rel_path, r.load_kind, r.load,
                                   r.angle, r.speed))
        return points, chains

    @staticmethod
    def _read_point(locator, key: Tuple[str, float, float, float],
                    tiers: Dict[str, Tuple[Path, LoadPoint]], n_layers: int,
                    expected: Dict[str, int],
                    config: Optional[dict]) -> RunFrames:
        """Parse one load point's tier files and give it its first status."""
        kind, load, angle, speed = key
        family = locator.tokens.family
        frames: Dict[str, pd.DataFrame] = {}
        sources: Dict[str, Path] = {}
        failures: List[str] = []
        window_error: Optional[str] = None
        for tier, (path, _point) in sorted(tiers.items()):
            try:
                # A PES grid is a comma table with a `# family=` first
                # line, not a LAMMPS `ave/time` file, so it goes
                # through read_csv, as the store does for a stored `.csv`.
                frames[tier] = (
                    pd.read_csv(path, comment='#') if path.suffix == '.csv'
                    else read_results_file(path, family, n_layers))
                sources[tier] = path
            except (SchemaError, ValueError, OSError,
                    pd.errors.EmptyDataError) as exc:
                logger.warning("Could not process file %s: %s", path, exc)
                if tier == 'window':
                    window_error = str(exc)
                failures.append(f'{tier} unreadable: {exc}')
        point = RunFrames(locator=locator, load_kind=kind, load=load,
                          angle=angle, speed=speed, n_layers=n_layers,
                          tiers=frames, source_files=sources,
                          expected_rows=dict(expected), config=config)
        if not tiers:
            point.status = 'no_output'
        elif window_error is not None or not frames:
            # Discarding a good trace because a raw/peratom sidecar is corrupt
            # would lose data, so only a failed window (or nothing parsed at
            # all) makes the whole point unreadable.
            point.tiers, point.source_files = {}, {}
            point.status = 'unreadable'
            point.status_detail = (window_error if window_error is not None
                                   else '; '.join(failures))
        else:
            point.status, detail = classify(point.n_rows, expected)
            point.status_detail = _join_detail(*failures, detail)
            point.post = post_channels_for(locator, kind, load, angle)
        return point

    def _get_output_path(self, filename: str) -> Path:
        """Construct the full path for an output file."""
        return self.output_dir / filename

    _REGISTRY_SIM_TYPE = {'tip': 'afm', 'sheet': 'sheetonsheet'}

    @classmethod
    def _registry_sim_type(cls, sim_type: str, header_line1: str) -> str:
        """Registry sim_type for a results file.

        Prefers the `family=` key the builder stamps into title1: the file
        states its own provenance, so the parser never has to infer a family
        from the directory token -- which is ambiguous, because a hetero run
        WITH a flake carries both the '+' and the '+flake-' markers.
        """
        m = re.search(r'family=(\w+)', header_line1 or '')
        if m:
            family = m.group(1)
            if family not in FAMILIES:
                raise ValueError(
                    f"unknown family {family!r} in results header; known: "
                    f"{FAMILIES}")
            return family
        return cls._REGISTRY_SIM_TYPE.get(sim_type, 'afm')

    def _correct_legacy_lf_schema1(self, df, filepath, sim_type: str):
        """Delegate, mapping the traversal vocabulary onto the family one.

        Existing callers (and `tests/postprocessing/test_read_data_schema.py`)
        pass `'sheet'`/`'tip'`; `correct_legacy_lf_schema1` takes a family.
        """
        family = 'sheetonsheet' if sim_type == 'sheet' else 'afm'
        return correct_legacy_lf_schema1(df, filepath, family)

    def _read_dataframe(self, filepath: Path, sim_type: str,
                        n_layers: int = 2) -> pd.DataFrame:
        """Delegate to the module-level parser.

        The caller passes the traversal vocabulary (`'tip'`/`'sheet'`), so the
        family is resolved here first -- `_registry_sim_type` prefers the
        file's own `family=` header and falls back to `_REGISTRY_SIM_TYPE`.
        """
        header = ''
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                header = next(fh, '')
        except OSError:
            pass
        return read_results_file(filepath,
                                 self._registry_sim_type(sim_type, header),
                                 n_layers=n_layers)

    def _run_block(self, run: RunFrames) -> dict:
        """The run's provenance block, built once per run directory."""
        key = Path(run.locator.run_dir)
        if key not in self._run_blocks:
            self._run_blocks[key] = run_provenance(run.locator, run.config)
        return self._run_blocks[key]

    def run_payload(self, run: RunFrames) -> Dict[str, Any]:
        """One `runs[]` entry of the v2 document."""
        tokens = run.tokens
        tiers: Dict[str, Any] = {}
        for tier, frame in sorted(run.tiers.items()):
            out = frame
            if tier == 'window':
                out = frame.copy()
                if {'lfx', 'lfy', 'nf'} <= set(out.columns):
                    out['lateral_force'], out['cof'] = compute_derived_columns(
                        np.asarray(out['lfx'].values), np.asarray(out['lfy'].values),
                        np.asarray(out['nf'].values))
                out.attrs = dict(frame.attrs)
            tiers[tier] = {
                'columns': list(out.columns),
                'data': out.values.tolist(),
                'attrs': {'lf_schema': out.attrs.get('lf_schema', 2),
                          'lf_corrected': bool(out.attrs.get('lf_corrected',
                                                             False))},
            }
        block = self._run_block(run)
        detail = run.status_detail
        if 'error' in block:
            detail = _join_detail(detail, 'provenance unreadable')
        return {
            'model': tokens.model, 'family': run.family,
            'material': run.material, 'materials': list(tokens.materials),
            'flake_material': tokens.flake_material,
            'size': run.size, 'size_x': tokens.size_x, 'size_y': tokens.size_y,
            'substrate': tokens.substrate,
            'substrate_amorphous': tokens.substrate_amorphous,
            'tip_material': tokens.tip_material,
            'tip_radius': tokens.tip_radius,
            'temperature': tokens.temperature, 'layers': tokens.layers,
            'potential': tokens.pot_token, 'rel_path': tokens.rel_path,
            'load_kind': run.load_kind, 'load': run.load, 'angle': run.angle,
            'speed': run.speed, 'n_layers': run.n_layers,
            'tiers': tiers, 'post': dict(run.post),
            'status': run.status, 'status_detail': detail,
            'provenance': point_provenance(block, run.load),
            'n_rows': run.n_rows, 'expected_rows': dict(run.expected_rows),
            'integrity': (None if run.integrity is None
                          else run.integrity.as_dict()),
        }

    def export_payload(self, size: str, runs: List[RunFrames]) -> Dict[str, Any]:
        """The whole export document (schema 3) for one size."""
        counted = [p for p in self.points if p.size == size]
        payloads = [self.run_payload(run) for run in runs]
        units: Dict[str, str] = {}
        for run, payload in zip(runs, payloads):
            for tier in payload['tiers'].values():
                units.update(units_for_columns(tier['columns'], run.family))
        return {
            'schema': EXPORT_SCHEMA,
            'size': size,
            'generated_by': 'FrictionSim2D',
            'units': dict(sorted(units.items())),
            'metadata': self._build_metadata(runs, counted),
            'runs': payloads,
        }

    def summary_rows(self) -> List[Dict[str, Any]]:
        """One flat row per load point -- the rows the AiiDA store summarises.

        Every statistic comes from `summarise_run`, the single COF on the
        store path. Nothing here computes a mean of its own.
        """
        from ..data.models import summarise_run  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        rows: List[Dict[str, Any]] = []
        for run in self.points:
            if self.complete_only and run.status != 'complete':
                continue
            tokens = run.tokens
            # Only a complete point's frames describe friction: a halted one
            # records a blow-up, a failed or quarantined one a broken or
            # unjudged system. The others keep their row, with no statistics.
            usable = run.status == 'complete'
            summary = summarise_run(run.frames_by_tier() if usable else {},
                                    run.family,
                                    skip_fraction=self.skip_fraction,
                                    post=run.post if usable else {})
            row: Dict[str, Any] = {
                'model': tokens.model, 'family': run.family,
                'material': run.material,
                'materials': '+'.join(tokens.materials),
                'flake_material': tokens.flake_material,
                'size': run.size, 'size_x': tokens.size_x,
                'size_y': tokens.size_y, 'substrate': tokens.substrate,
                'substrate_amorphous': tokens.substrate_amorphous,
                'tip_material': tokens.tip_material,
                'tip_radius': tokens.tip_radius,
                'temperature': tokens.temperature,
                # The RESOLVED count, not `tokens.layers`: the sheet-like
                # models emit no `L{n}` path level, and `layers` is a headline
                # filter in the store and the GUI. NULL there for
                # every sheet, hetero, flake and pes_sheet row is worse than a
                # number read from provenance.
                'layers': run.n_layers,
                'potential': tokens.pot_token, 'rel_path': tokens.rel_path,
                'load_kind': run.load_kind, 'load': run.load,
                'angle': run.angle, 'speed': run.speed,
                'tiers': ';'.join(sorted(run.tiers)),
                'status': run.status,
                'status_detail': run.status_detail,
            }
            row.update(summary.as_dict())
            row['is_complete'] = (run.status == 'complete'
                                  and bool(summary.is_complete))
            for name in SUMMARY_LF_COLUMNS:  # from summarise_run; NaN where absent
                value = row.get(name) if usable else None
                row[name] = float('nan') if value is None else float(value)
            for name in SUMMARY_POST_COLUMNS:
                row[name] = (float(run.post.get(name, float('nan'))) if usable
                             else float('nan'))
            rows.append(row)
        return rows

    def export_summary_csv(self) -> List[Path]:
        """Write one `summary_<size>.csv` per size; return the paths."""
        by_size: Dict[str, List[Dict[str, Any]]] = {}
        for row in self.summary_rows():
            by_size.setdefault(row['size'], []).append(row)

        written: List[Path] = []
        for size, rows in sorted(by_size.items()):
            frame = pd.DataFrame(rows)
            ordered = [c for c in SUMMARY_IDENTITY_COLUMNS
                       if c in frame.columns]
            tail = list(SUMMARY_LF_COLUMNS) + list(SUMMARY_POST_COLUMNS)
            middle = [c for c in frame.columns
                      if c not in ordered and c not in tail]
            frame = frame[ordered + middle + tail]
            path = self._get_output_path(f'summary_{size}.csv')
            frame.to_csv(path, index=False)
            written.append(path)
            logger.info("Exported %d summary row(s) for size %s to %s",
                        len(rows), size, path)
        return written

    def export_full_data_to_json(self) -> List[Path]:
        """Write one `output_full_<size>.json` per size; return the paths."""
        written: List[Path] = []
        for size, runs in sorted(self.points_by_size().items()):
            path = self._get_output_path(f'output_full_{size}.json')
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(self.export_payload(size, runs), fh, cls=_NpEncoder,
                          indent=1, sort_keys=False)
            written.append(path)
            logger.info("Exported %d run(s) for size %s to %s",
                        len(runs), size, path)
        return written

    def export_issue_reports(self) -> None:
        """Export reports on incomplete files and materials to text files."""
        if self.incomplete_files:
            for size, files in self.incomplete_files.items():
                filepath = self._get_output_path(f'incomplete_files_{size}.txt')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sorted(files)))
                logger.info("Incomplete files for size %s saved to %s", size, filepath)

        if self.incomplete_materials:
            for size, materials in self.incomplete_materials.items():
                filepath = self._get_output_path(f'incomplete_materials_{size}.txt')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sorted(list(materials))))
                logger.info("Incomplete materials for size %s saved to %s", size, filepath)
