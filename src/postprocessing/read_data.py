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
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.observables import FAMILIES, layer_channels, rename_map

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


def _rename_for(family: str, tier: str, n_layers: int) -> dict:
    """`rename_map` extended with this run's real layer count.

    REGISTRY only declares the bilayer `com{x,y,z}_L1/L2`, so a four-layer
    run's native `v_comx_L3` would keep its native name and the store and the
    reader would disagree about a column that exists in both.
    """
    rmap = dict(rename_map(family, tier))
    for observable in layer_channels(n_layers):
        if observable.tier == tier and observable.rename_to:
            rmap[observable.rename_to] = observable.name
    return rmap


def read_results_file(filepath: Path, family: str,
                      n_layers: int = 2) -> pd.DataFrame:
    """Parse one LAMMPS results file into a labelled DataFrame.

    Header-first: the `# TimeStep v_...` line names the native columns and the
    observable registry's per-tier rename map turns them into channel names.
    The tier comes from the filename prefix. `n_layers` extends the rename map
    to this run's real layer count. Raises `SchemaError` when a header-less
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
        df.rename(columns=_rename_for(family, tier, n_layers), inplace=True)
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

    # Filename patterns (compiled once at class level)
    _FILE_PATTERN_TIP = re.compile(r'fc_ave_slide_(\d+\.?\d*)nN_(\d+)angle_(\d+)ms_l(\d+)')
    # New AFM tip naming written by the current builder, e.g.
    # ``friction_f30_a0.0_s2_layer2.txt``. Groups match _FILE_PATTERN_TIP:
    # (load, angle, speed, layer). Load is always a force (nN) for tip sims.
    # The angle is signed: LAMMPS substitutes the scan angle verbatim into
    # results_file_pattern, so a -15 deg scan writes friction_f5_a-15_s1_...
    # An unsigned pattern skips those files in silence -- the run simply
    # disappears from every analysis.
    _FILE_PATTERN_TIP_ALT = re.compile(
        r'friction_f(\d+\.?\d*)_a(-?\d+\.?\d*)_s(\d+\.?\d*)_layer(\d+)'
    )
    _FILE_PATTERN_SHEET = re.compile(
        r'fc_ave_slide_(\d+\.?\d*)(GPa|nN)_(\d+\.?\d*)angle_(\d+\.?\d*)ms'
    )
    # Signed angle, as in _FILE_PATTERN_TIP_ALT above.
    _FILE_PATTERN_SHEET_ALT = re.compile(
        r'friction_p(\d+\.?\d*)_a(-?\d+\.?\d*)_s(\d+\.?\d*)'
    )

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

    def __init__(self, results_dir: str = 'results_110725_test') -> None:
        """Initialise the DataReader.

        Args:
            results_dir: Path to the directory containing simulation results.
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'outputs'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.full_data_nested: dict = {}

        # Read the data and populate the dictionaries and metadata
        (
            self.time_series,
            self.incomplete_files,
            self.incomplete_materials,
            self.metadata,
            self.ntimestep,
        ) = self.read_data()

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
        material_types: dict = {
            'b_type': [], 'h_type': [], 't_type': [], 'p_type': [], 'other': [],
        }
        for mat in final.get('materials', []):
            prefix = mat.split('_', 1)[0]
            type_key = f"{prefix}_type"
            material_types[type_key if type_key in material_types else 'other'].append(mat)
        final['material_types'] = material_types
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

    def export_full_data_to_json(self) -> None:
        """Export the full time-series data to JSON files, one per size."""
        data_by_size: dict = {}
        for material, mat_data in self.full_data_nested.items():
            for size, size_data in mat_data.items():
                data_by_size.setdefault(size, {})[material] = size_data

        for size_key, size_data in data_by_size.items():
            output_path = self._get_output_path(f'output_full_{size_key}.json')

            full_output_metadata = self.metadata.copy()
            full_output_metadata['time_series'] = self.time_series

            relevant_tip_radii: set = set()
            for mat_data in size_data.values():
                for sub_data in mat_data.values():
                    for tip_mat_data in sub_data.values():
                        relevant_tip_radii.update(tip_mat_data.keys())

            full_output_metadata['tip_radii'] = sorted(list(relevant_tip_radii))
            full_output_metadata['size'] = size_key

            output_with_metadata = {
                'metadata': full_output_metadata,
                'results': size_data,
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_with_metadata, f, cls=_NpEncoder)
            logger.info("Full time-series data for size %s exported to %s", size_key, output_path)

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
