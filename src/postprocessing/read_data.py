"""Read and process friction simulation output data.

Walks through a directory of simulation results, parses filenames and
paths to extract metadata, reads time-series data, calculates derived
quantities (COF, lateral force) and stores everything in a structured
format that can be exported to JSON for downstream plotting.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.observables import FAMILIES, rename_map

from .post_channels import post_channels_for

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Results file layout does not match any known schema (spec §1.1).

    Raised instead of silently mis-aligning: a header-less file whose column
    count differs from the LEGACY_2026 positional list is corrupt or from an
    unknown template revision, and parsing it positionally would shift every
    value.
    """


#: Families whose results files follow the sheet-on-sheet schema. The legacy
#: lateral-force correction applies only to these -- the AFM stack never wrote
#: an eV/A lateral force.
SHEET_FAMILIES = ('sheetonsheet', 'hetero', 'flake', 'pes_sheet')
#: Families on the AFM stack. Only these have a positional LEGACY_2026 fallback.
AFM_FAMILIES = ('afm', 'indent', 'pes_tip')

#: eV/A -> nN. Moved out of DataReader; the class keeps an alias.
LF_SCHEMA1_CORRECTION = 1.602176565
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

    FROZEN legacy path, moved verbatim from
    `DataReader._correct_legacy_lf_schema1`. The ONLY change is the
    vocabulary: the old `if sim_type != 'sheet'` becomes
    `if family not in SHEET_FAMILIES`, so a hetero or flake file is corrected
    too -- under the old traversal vocabulary they arrived as `'sheet'` and
    were, which is the behaviour being preserved, not extended.
    """
    if df.attrs.get('lf_corrected'):
        # Re-entry guard: scaling twice gives 1.602^2.
        return df
    df.attrs['lf_corrected'] = False
    schema = df.attrs.get('lf_schema', 1)
    if family not in SHEET_FAMILIES or schema != 1:
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


def _tier_of(filename: str) -> str:
    """`friction_`/`fc_ave_` -> window, `raw_` -> raw, `peratom_` -> peratom."""
    return ('raw' if filename.startswith('raw_')
            else 'peratom' if filename.startswith('peratom_')
            else 'window')


#: Public alias. `src.aiida.ingest.load_points` imports this rather than
#: reaching across packages for a private name.
tier_of = _tier_of


#: Filename patterns (moved out of `DataReader`; the class below keeps
#: aliases, same as `parse_lf_schema`/`LF_SCHEMA1_CORRECTION` above). This is
#: the ONE parser for slide-results filenames (R29/U7): `src.aiida.ingest.
#: load_points` imports these rather than re-implementing them, so grouping
#: can never accept a filename the DataReader itself would refuse.
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

    tier = _tier_of(Path(filepath).name)

    if header_cols:
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=header_cols,
                         skiprows=2)
        df.rename(columns=rename_map(family, tier, n_layers), inplace=True)
        df.attrs['lf_schema'] = parse_lf_schema(header_lines[0])
        return correct_legacy_lf_schema1(df, filepath, family)

    # LEGACY_2026 positional fallback -- AFM stack only. The sheet corpus was
    # surveyed on 2026-08-29: zero non-empty headerless files. Guessing is
    # worse than refusing.
    if family not in AFM_FAMILIES:
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
_RESULTS_RE = re.compile(
    r'^(?:(?P<tier>friction|raw|peratom)_(?P<kind>[fp])(?P<load>-?\d+\.?\d*)'
    r'_a(?P<angle>-?\d+\.?\d*)_s(?P<speed>-?\d+\.?\d*)'
    r'(?:_layer(?P<layer>\d+))?'
    r'|fc_ave_slide_(?P<lload>\d+\.?\d*)(?P<lunit>nN|GPa)'
    r'_(?P<langle>-?\d+\.?\d*)angle_(?P<lspeed>\d+\.?\d*)ms'
    r'(?:_l(?P<llayer>\d+))?)\.txt$')

_TIER_BY_PREFIX = {'friction': 'window', 'raw': 'raw', 'peratom': 'peratom'}

#: A PES scan's grid IS its window tier -- one grid per run, no load point to
#: speak of. The same mapping `FrictionResultsData._TIER_PREFIX` makes when it
#: files `pes_scan` under `'window'`; two readers, one rule.
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

    Replaces the eight-deep `full_data_nested` dict, which had no dimension for
    the potential or the temperature -- two runs differing only in those
    collapsed onto one leaf and the last one walked won.

    `locator` is a `RunLocator`, not the `RunTokens` the spec's sketch named:
    tokens alone cannot find a file, and `post_channels_for` needs `data_dir`,
    the tier reader needs `results_dir` and `read_provenance` needs
    `provenance_file` -- all already on the locator `discover_runs` built.
    Carrying tokens and re-deriving the paths would be a second path parser.
    """
    locator: Any                        # src.run.tree.RunLocator
    load_kind: str                      # 'force' | 'pressure' | 'none'
    load: float
    angle: float
    speed: float
    n_layers: int
    tiers: Dict[str, pd.DataFrame]
    source_files: Dict[str, Path]
    post: Dict[str, float] = field(default_factory=dict)

    @property
    def tokens(self):
        """The run's `RunTokens`."""
        return self.locator.tokens

    @property
    def family(self) -> str:
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
EXPORT_SCHEMA = 2

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


def resolve_n_layers(locator) -> int:
    """How many sheet layers this run has.

    `RunTokens.layers` is the `L{n}` path level, and `output_dir_for` emits one
    only for the AFM-like models -- the AFM builders append it
    (`src/builders/afm/base.py:182`), the sheet-like ones never do. So for
    `sheetonsheet`, `hetero`, `flake` and `pes_sheet` the token is `None`, and
    defaulting it to 2 is exactly the defect P3 exists to fix, missed in the
    families that need it most: a `layers=[4]` sheet run's `v_comz_L3`/`L4` and
    its peratom `v_pe_sum_L3/L4`, `v_sigma_*_L3/L4` would stay un-renamed.

    The count IS on disk. `ProvenanceRecord.layer_type_map` is keyed
    `layer_1 ... layer_n` (`src/builders/sheetonsheet/base.py:62-83`,
    `src/builders/afm/base.py:961-969`), and `discover_runs` already opens that
    file to read `family`. It is `{}` for hetero and flake, whose layers are
    z-band regions rather than atom-type groups; those log and fall back to 2,
    which is what they have always assumed.
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

    # Aliases: the module-level definitions above are the source of truth.
    # Kept because audit_lf_units.py:30,42 and eight existing assertions reach
    # these through the class, and there is no reason to churn them.
    _LF_SCHEMA1_CORRECTION = LF_SCHEMA1_CORRECTION
    _LF_SCHEMA_CURRENT = LF_SCHEMA_CURRENT
    _AFM_FILE_COLUMNS = AFM_FILE_COLUMNS
    _parse_lf_schema = staticmethod(parse_lf_schema)
    _branch_column = staticmethod(branch_column)
    _lateral_columns = staticmethod(lateral_columns)
    _is_lj_branch = staticmethod(is_lj_branch)
    _FILE_PATTERN_TIP = FILE_PATTERN_TIP
    _FILE_PATTERN_TIP_ALT = FILE_PATTERN_TIP_ALT
    _FILE_PATTERN_SHEET = FILE_PATTERN_SHEET
    _FILE_PATTERN_SHEET_ALT = FILE_PATTERN_SHEET_ALT

    # Path patterns (compiled once at class level)
    # [\w-] (not \w alone) so the amorphous-substrate/tip token survives:
    # src.run.naming.output_dir_for writes 'amorph-<mat>' (Task 3b), and a
    # material's own structure prefix can also carry a hyphen (h-MoS2).
    _PATH_PATTERN_TIP = re.compile(r'(\d+x_\d+y)/sub_([\w-]+)_tip_([\w-]+)_(r\d+)')
    # pot_ segment consumed NON-capturing so group 3 keeps its legacy meaning
    # (substrate token). A repeated capturing group would capture only the
    # LAST repetition and record pot_... as substrate_material.
    #: ``+`` is in the material class deliberately: src.run.naming.material_token
    #: joins a heterostructure's materials with it (``h_MoS2+h_WS2``) and appends
    #: a ``+flake-<mat>`` suffix. Without it the match started AFTER the ``+``,
    #: so a hetero run was silently filed under its SECOND constituent and
    #: pooled with that material's homogeneous data -- misattributed, not
    #: skipped. tests/test_read_data_hetero_paths.py pins this.
    _PATH_PATTERN_SHEET = re.compile(
        r'(?:sheetvsheet/)?([\w\d\-+_]+)/(\d+x_\d+y)/'
        r'(?!pot_)([\w\d\-+_]+)?/?(?:pot_[^/]+/)?results'
    )

    def __init__(self, results_dir: str = 'results_110725_test', *,
                 skip_fraction: float = 0.2) -> None:
        """Initialise the DataReader.

        Args:
            results_dir: A simulation root -- the directory holding `afm/`,
                `sheetonsheet/`, `indent/`, `pes_tip/`, `pes_sheet/`.
            skip_fraction: Transient fraction `summarise_run` discards.
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'outputs'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_fraction = skip_fraction

        self.full_data_nested: dict = {}

        # The regex walk, kept ONLY for `full_data_nested` until Task 14
        # deletes it. Its four other return values are discarded: the
        # `discover_runs` walk below is the source of truth for the metadata,
        # the time series and the completeness reports, and merging the two
        # would mean two answers to the same question.
        self.read_data()

        self.runs: List[RunFrames] = self._read_runs()
        self.ntimestep: int = max((len(r.window) for r in self.runs), default=0)
        self.time_series: List[int] = self._time_series()
        self.incomplete_files: Dict[str, List[str]] = {}
        self.incomplete_materials: Dict[str, set] = {}
        self._flag_incomplete()
        self.metadata: Dict[str, Any] = self._build_metadata()

    def _time_series(self) -> List[int]:
        """The `time` column of the first full-length window frame."""
        for run in self.runs:
            if len(run.window) == self.ntimestep and 'time' in run.window:
                return [int(t) for t in run.window['time'].to_list()]
        return []

    def _flag_incomplete(self) -> None:
        """Record runs whose window frame is short of `ntimestep`.

        Three samples of slack, as the old two-pass reader used: a window tier
        can legitimately miss its final block. Unlike the old reader the run is
        KEPT -- flagging it and dropping it are different decisions, and only
        the first is this class's to make.
        """
        for run in self.runs:
            if self.ntimestep - len(run.window) <= 3:
                continue
            self.incomplete_files.setdefault(run.size, []).append(
                str(run.source_files.get('window', run.locator.run_dir)))
            self.incomplete_materials.setdefault(run.size, set()).add(run.material)

    def runs_by_size(self) -> Dict[str, List[RunFrames]]:
        """Runs grouped by their `size` token -- one export file per key."""
        grouped: Dict[str, List[RunFrames]] = {}
        for run in self.runs:
            grouped.setdefault(run.size, []).append(run)
        return grouped

    def _build_metadata(self,
                        runs: Optional[List[RunFrames]] = None) -> Dict[str, Any]:
        """The v2 `metadata` block for `runs` (default: every run)."""
        runs = self.runs if runs is None else runs
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
            # Which COF definition produced the summary numbers beside this
            # export. v1 files carried no such statement and the definition
            # changed under readers with no way to tell; this is that way.
            'summary_definition': {
                'cof': 'ratio_of_means: mean(LF)/mean(NF)',
                'skip_fraction': self.skip_fraction,
                'source': 'src.data.models.summarise_run',
            },
        }

    def _read_runs(self) -> List[RunFrames]:
        """One walk with `discover_runs`, one `RunFrames` per load point."""
        from ..run.tree import discover_runs  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        runs: List[RunFrames] = []
        for locator in discover_runs(self.results_dir):
            family = locator.tokens.family
            n_layers = resolve_n_layers(locator)
            for (kind, load, angle, speed), tiers in _load_points(locator).items():
                frames: Dict[str, pd.DataFrame] = {}
                sources: Dict[str, Path] = {}
                for tier, (path, _point) in sorted(tiers.items()):
                    try:
                        # A PES grid is a comma table with a `# family=` first
                        # line, not a LAMMPS `ave/time` file, so it goes
                        # through read_csv -- exactly as
                        # `FrictionResultsData.frame` does for a stored `.csv`.
                        frames[tier] = (
                            pd.read_csv(path, comment='#') if path.suffix == '.csv'
                            else read_results_file(path, family, n_layers))
                        sources[tier] = path
                    except (SchemaError, ValueError, OSError,
                            pd.errors.EmptyDataError) as exc:
                        logger.warning("Could not process file %s: %s", path, exc)
                if not frames:
                    continue
                runs.append(RunFrames(
                    locator=locator, load_kind=kind, load=load, angle=angle,
                    speed=speed, n_layers=n_layers, tiers=frames,
                    source_files=sources,
                    post=post_channels_for(locator, kind, load, angle)))
        return sorted(runs, key=lambda r: (r.tokens.rel_path, r.load_kind,
                                           r.load, r.angle, r.speed))

    def _calculate_derived_quantities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate lateral force and coefficient of friction.

        Delegates to :func:`~src.data.models.compute_derived_columns` to
        ensure consistent calculation across the codebase.

        Args:
            df: DataFrame with at least lfx, lfy, and nf columns.

        Returns:
            DataFrame with added lateral_force and cof columns.
        """
        if 'lfx' in df.columns and 'lfy' in df.columns and 'nf' in df.columns:
            from ..data.models import compute_derived_columns  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

            lateral_force, cof = compute_derived_columns(
                np.asarray(df['lfx'].values),
                np.asarray(df['lfy'].values),
                np.asarray(df['nf'].values),
            )
            df['lateral_force'] = lateral_force
            df['cof'] = cof

        return df

    def _get_output_path(self, filename: str) -> Path:
        """Construct the full path for an output file."""
        return self.output_dir / filename

    def _first_pass_ntimestep(self) -> int:
        """First directory walk to determine the expected timestep count."""
        ntimestep = 0
        logger.info("Starting first pass to determine ntimestep...")
        for root, _, files in os.walk(self.results_dir):
            is_tip = self._PATH_PATTERN_TIP.search(root)
            is_sheet = self._PATH_PATTERN_SHEET.search(root)
            if not (is_tip or is_sheet):
                continue
            file_patterns = (
                (self._FILE_PATTERN_TIP, self._FILE_PATTERN_TIP_ALT)
                if is_tip
                else (self._FILE_PATTERN_SHEET, self._FILE_PATTERN_SHEET_ALT)
            )
            for filename in files:
                if any(p.match(filename) for p in file_patterns):
                    filepath = Path(root) / filename
                    try:
                        df = pd.read_csv(
                            filepath, sep=r'\s+', header=None, usecols=[0], skiprows=2,
                        )
                        ntimestep = max(ntimestep, len(df))
                    except (pd.errors.EmptyDataError, IndexError):
                        continue
        return ntimestep

    def _resolve_path_info(
        self, root: str,
    ) -> tuple[str, str, str, str, str, str] | None:
        """Parse a directory path to extract simulation context.

        Returns:
            ``(sim_type, material, size, substrate, tip_material, tip_radius)``
            or ``None`` if the path does not match any known pattern.
        """
        path_match_tip = self._PATH_PATTERN_TIP.search(root)
        if path_match_tip:
            size, substrate_material, tip_material, tip_radius = path_match_tip.groups()
            try:
                search_path = str(self.results_dir / 'afm')
                start_dir = (
                    search_path
                    if os.path.commonpath([root, search_path]) == search_path
                    else str(self.results_dir)
                )
                material_path_full = root[:root.find(size)]
                material = str(Path(material_path_full).relative_to(start_dir)).strip('/')
            except (IndexError, ValueError):
                return None
            if not material or material == '.':
                return None
            return ('tip', material, size, substrate_material, tip_material, tip_radius)

        path_match_sheet = self._PATH_PATTERN_SHEET.search(root)
        if path_match_sheet:
            material, size, substrate_material = path_match_sheet.groups()
            substrate_material = (substrate_material or 'N/A').strip('/')
            return ('sheet', material, size, substrate_material, 'sheet', 'N/A')

        return None

    #: Traversal vocabulary ('tip'/'sheet') -> registry sim_type. The file's
    #: own `family=` marker overrides this when present; every retained file
    #: predates the marker and takes the fallback.
    _REGISTRY_SIM_TYPE = {'tip': 'afm', 'sheet': 'sheetonsheet'}
    _KNOWN_FAMILIES = FAMILIES

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
            if family not in cls._KNOWN_FAMILIES:
                raise ValueError(
                    f"unknown family {family!r} in results header; known: "
                    f"{cls._KNOWN_FAMILIES}")
            return family
        return cls._REGISTRY_SIM_TYPE.get(sim_type, 'afm')

    @staticmethod
    def _is_sheet_family(sim_type: str) -> bool:
        """True for the traversal vocabulary's sheet side.

        The filename patterns and parameter parsing are shared across the
        whole sheet family; only rename_map needs the precise family.
        """
        return sim_type == 'sheet'

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

    def _parse_filename_params(
        self,
        sim_type: str,
        filename: str,
        file_match: re.Match,
    ) -> tuple[float, int, int, int, bool]:
        """Extract load, angle, speed, layer and pressure flag from a regex match.

        Returns:
            ``(load_val, angle, speed, layer, is_pressure)``
        """
        if sim_type == 'sheet':
            if self._FILE_PATTERN_SHEET.match(filename):
                load_str, unit_str, angle_str, speed_str = file_match.groups()
                is_pressure = unit_str == 'GPa'
                load_val = float(load_str)
            else:
                load_str, angle_str, speed_str = file_match.groups()
                is_pressure = True
                load_val = float(load_str)
            layer = 2
        else:
            load_str, angle_str, speed_str, layer_str = file_match.groups()
            layer = int(layer_str)
            is_pressure = False
            load_val = float(load_str)
        return load_val, int(float(angle_str)), int(float(speed_str)), layer, is_pressure

    @staticmethod
    def _finalize_metadata(metadata: dict) -> dict:
        """Convert mutable sets in the metadata dict to sorted lists."""
        final: dict = {}
        for k, v in metadata.items():
            if k in ('forces_and_angles', 'pressures_and_angles'):
                final[k] = {load: sorted(list(angles)) for load, angles in v.items()}
            else:
                final[k] = sorted(list(v))
        # The module-level `material_types` is the one bucketing rule; this
        # method used to carry its own copy. Task 14 deletes the method.
        final['material_types'] = material_types(final.get('materials', []))
        return final

    def read_data(self) -> tuple:  # pylint: disable=too-many-locals
        """Walk through the results directory and read all simulation data.

        Uses a two-pass approach to dynamically determine the correct
        number of timesteps for a complete file.
        """
        ntimestep = self._first_pass_ntimestep()
        if ntimestep == 0:
            logger.warning("No valid data files found. Could not determine ntimestep.")
            return None, {}, {}, {}, 0
        logger.info("Determined ntimestep for a complete file to be: %d", ntimestep)

        time_series = None
        incomplete_files: dict = {}
        incomplete_materials: dict = {}
        metadata: dict = {
            'materials': set(), 'substrates': set(), 'tip_materials': set(),
            'tip_radii': set(), 'layers': set(), 'speeds': set(),
            'forces_and_angles': {}, 'pressures_and_angles': {},
        }

        for root, _, files in os.walk(self.results_dir):
            path_info = self._resolve_path_info(root)
            if path_info is None:
                continue
            sim_type, material, size, substrate_material, tip_material, tip_radius = path_info
            safe_material = material.replace('-', '_').replace('/', '__')
            size_key = size.replace('x_', 'x')

            for filename in files:
                if sim_type == 'sheet':
                    file_match = (
                        self._FILE_PATTERN_SHEET.match(filename)
                        or self._FILE_PATTERN_SHEET_ALT.match(filename)
                    )
                else:
                    file_match = (
                        self._FILE_PATTERN_TIP.match(filename)
                        or self._FILE_PATTERN_TIP_ALT.match(filename)
                    )
                if not file_match:
                    continue

                filepath = Path(root) / filename
                try:
                    df = self._read_dataframe(filepath, sim_type)

                    if ntimestep - len(df) > 3:
                        incomplete_files.setdefault(size_key, []).append(str(filepath))
                        incomplete_materials.setdefault(size_key, set()).add(material)
                        continue

                    if time_series is None:
                        time_series = df['time'].to_list()

                    load_val, angle, speed, layer, is_pressure = self._parse_filename_params(
                        sim_type, filename, file_match,
                    )
                    metadata['materials'].add(safe_material)
                    metadata['substrates'].add(substrate_material)
                    metadata['tip_materials'].add(tip_material)
                    metadata['tip_radii'].add(tip_radius)
                    metadata['speeds'].add(speed)
                    metadata['layers'].add(layer)
                    if is_pressure:
                        metadata['pressures_and_angles'].setdefault(load_val, set()).add(angle)
                    else:
                        metadata['forces_and_angles'].setdefault(load_val, set()).add(angle)

                    df_processed = df.drop(columns=['time'])
                    df_processed = self._calculate_derived_quantities(df_processed)

                    base_path = (
                        self.full_data_nested
                        .setdefault(safe_material, {})
                        .setdefault(size_key, {})
                        .setdefault(substrate_material, {})
                        .setdefault(tip_material, {})
                        .setdefault(tip_radius, {})
                        .setdefault(f'l{layer}', {})
                        .setdefault(f's{speed}', {})
                    )
                    load_key = f'p{load_val}' if is_pressure else f'f{load_val}'
                    base_path.setdefault(load_key, {})[f'a{angle}'] = df_processed

                except (pd.errors.EmptyDataError, IndexError, ValueError) as e:
                    logger.warning("Could not process file %s: %s", filepath, e)
                    incomplete_files.setdefault(size_key, []).append(str(filepath))
                    incomplete_materials.setdefault(size_key, set()).add(material)

        return (
            time_series,
            incomplete_files,
            incomplete_materials,
            self._finalize_metadata(metadata),
            ntimestep,
        )

    def run_payload(self, run: RunFrames) -> Dict[str, Any]:
        """One `runs[]` entry of the v2 document."""
        tokens = run.tokens
        tiers: Dict[str, Any] = {}
        for tier, frame in sorted(run.tiers.items()):
            out = frame
            if tier == 'window':
                out = self._calculate_derived_quantities(frame.copy())
                out.attrs = dict(frame.attrs)
            tiers[tier] = {
                'columns': list(out.columns),
                'data': out.values.tolist(),
                'attrs': {'lf_schema': out.attrs.get('lf_schema', 2),
                          'lf_corrected': bool(out.attrs.get('lf_corrected',
                                                             False))},
            }
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
        }

    def export_payload(self, size: str, runs: List[RunFrames]) -> Dict[str, Any]:
        """The whole v2 document for one size."""
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
            'metadata': self._build_metadata(runs),
            'runs': payloads,
        }

    def export_full_data_to_json(self) -> List[Path]:
        """Write one `output_full_<size>.json` per size; return the paths."""
        written: List[Path] = []
        for size, runs in sorted(self.runs_by_size().items()):
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
