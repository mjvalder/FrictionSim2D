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

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Results file layout does not match any known schema (spec §1.1).

    Raised instead of silently mis-aligning: a header-less file whose column
    count differs from the LEGACY_2026 positional list is corrupt or from an
    unknown template revision, and parsing it positionally would shift every
    value.
    """


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

    # Column names matching fix fc_ave output order in afm/slide.lmp.
    # 9 LAMMPS variables + timestep = 10 file columns.
    _AFM_FILE_COLUMNS = [
        'time', 'nf', 'lfx', 'lfy', 'comx', 'comy', 'comz',
        'tipx', 'tipy', 'tipz',
    ]

    # Column names matching fix fc_ave output order in sheetonsheet/slide.lmp.
    # 16 LAMMPS variables + timestep = 17 file columns.
    _SHEET_FILE_COLUMNS = [
        'time', 'v_xfrict', 'v_yfrict', 'v_sx', 'v_sy', 'v_sz',
        'v_fx', 'v_fy', 'v_fz', 'v_comx_top', 'v_comy_top',
        'v_comx_ctop', 'v_comy_ctop', 'v_comz_ctop',
        'v_comx_cbot', 'v_comy_cbot', 'v_comz_cbot',
    ]

    # eV/A -> nN. Pre-schema-2 sheet files emitted the LJ-branch lateral force
    # unscaled while the normal load was already nN, so their COF is a factor
    # 1.602176565 too SMALL. The correction is a multiplication.
    _LF_SCHEMA1_CORRECTION = 1.602176565

    # Output-schema version this reader understands. Bump BOTH this and the
    # templates' title1 together; _parse_lf_schema refuses to guess.
    _LF_SCHEMA_CURRENT = 2

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
    _PATH_PATTERN_TIP = re.compile(r'(\d+x_\d+y)/sub_(\w+)_tip_(\w+)_(r\d+)')
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

    def _read_dataframe(self, filepath: Path, sim_type: str) -> pd.DataFrame:
        """Read a LAMMPS output file and return a labelled DataFrame.

        Header-first for BOTH sim types: real campaign files carry the
        ``# TimeStep v_x v_y ...`` line LAMMPS writes, and the observable
        registry owns the native-name -> column-name map (per tier, chosen by
        filename prefix). The positional ``LEGACY_2026`` lists survive only as
        a strict fallback for header-less files: the width must match exactly,
        otherwise :class:`SchemaError` is raised. The old behaviour —
        ``pd.read_csv(names=...)`` with too few names — silently promoted the
        extra columns to an index AND shifted every value one column left,
        while keeping ``df.shape[1]`` equal to the expected count so the guard
        could never fire (spec §1.1 defect 1).
        """
        from src.core.observables import rename_map

        header_cols: list[str] | None = None
        header_lines: list[str] = ['', '']
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                header_lines = [next(fh, ''), next(fh, '')]
            if len(header_lines) > 1 and header_lines[1].startswith('#'):
                tokens = header_lines[1].lstrip('#').strip().split()
                if tokens:
                    header_cols = [
                        'time' if t == 'TimeStep' else t for t in tokens
                    ]
        except OSError:
            pass

        if header_cols:
            df = pd.read_csv(
                filepath, sep=r'\s+', header=None, names=header_cols,
                skiprows=2,
            )
            # tier from filename prefix: friction_/fc_ave_ -> window,
            # raw_ -> raw, peratom_ -> peratom (per-tier rename maps)
            stem = filepath.name
            tier = ('raw' if stem.startswith('raw_') else
                    'peratom' if stem.startswith('peratom_') else 'window')
            rmap = rename_map(
                'sheetonsheet' if sim_type == 'sheet' else 'afm', tier,
            )
            df.rename(columns=rmap, inplace=True)
            # Line 1 carries the output-schema marker. Absent means pre-fix:
            # the sheet's LJ-branch lateral force is in eV/A while the load it
            # is divided by is already nN, so its COF is a factor
            # 1.602176565 too SMALL (docs/notes/2026-08-27-sheet-cof-unit-
            # correction.md).
            df.attrs['lf_schema'] = self._parse_lf_schema(header_lines[0])
            return self._correct_legacy_lf_schema1(df, filepath, sim_type)

        # LEGACY_2026 positional fallback — width must match EXACTLY
        legacy = (self._SHEET_FILE_COLUMNS if sim_type == 'sheet'
                  else self._AFM_FILE_COLUMNS)
        try:
            probe = pd.read_csv(
                filepath, sep=r'\s+', header=None, skiprows=2, nrows=1,
            )
        except pd.errors.EmptyDataError as exc:
            raise SchemaError(f"{filepath}: no data rows") from exc
        if probe.shape[1] != len(legacy):
            raise SchemaError(
                f"{filepath}: {probe.shape[1]} columns, LEGACY_2026 expects "
                f"{len(legacy)} and no header is present"
            )
        df = pd.read_csv(
            filepath, sep=r'\s+', header=None, names=legacy, skiprows=2,
        )
        df.attrs['lf_schema'] = 1      # header-less files predate the marker
        return self._correct_legacy_lf_schema1(df, filepath, sim_type)

    @staticmethod
    def _branch_column(df):
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

    @staticmethod
    def _lateral_columns(df):
        """The (x, y) lateral-force column names present in the frame.

        Symmetric with `_branch_column`: the registry renames the sheet's
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

    @staticmethod
    def _is_lj_branch(lfx, sx) -> bool:
        """True when a pre-schema-2 sheet frame needs the unit correction.

        On the many-body branch the template defines ``xfrict`` AS ``v_sx``,
        so LAMMPS writes both columns from one value and they are byte-
        identical. Any difference means the LJ branch, whose ``compute
        group/group`` force is in eV/A while the load is already nN.

        Defined once and used by BOTH the reader (whole column) and
        ``audit_lf_units`` (a one-row sample), so the two cannot drift apart.
        """
        import pandas as _pd  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        return not _pd.Series(list(lfx)).equals(_pd.Series(list(sx)))

    @classmethod
    def _parse_lf_schema(cls, line1: str) -> int:
        """Output-schema version from a results file's first line.

        Parsed as an INTEGER, not matched as the substring 'schema=2'. A
        substring test treats every future version as schema 1 and silently
        applies the legacy unit correction to a file that never needed it --
        so bumping the marker would corrupt data rather than fail loudly.
        Absent marker means the file predates the marker entirely: schema 1.
        """
        m = re.search(r'schema=(\d+)', line1 or '')
        return int(m.group(1)) if m else 1

    def _correct_legacy_lf_schema1(self, df, filepath, sim_type: str):
        """Rescale a pre-``schema=2`` sheet frame's lateral force to nN.

        FROZEN legacy path. Delete it once the affected runs have been
        re-simulated -- it exists so old and new runs stay comparable during
        that window, not as long-term support.

        Branch detection: on the many-body branch the template defines
        ``xfrict`` AS ``v_sx``, so the two columns are bit-identical. Any
        difference means the LJ branch, whose ``compute group/group`` force is
        in eV/A while the load it is divided by is already nN. Verified over
        the retained tree: 7,171 of 7,669 files classify unambiguously.

        The discriminator is resolved by ``_branch_column``, which accepts
        either ``react_x`` (the registry's name, once Task 9 declares it) or
        ``v_sx`` (its native name, for files parsed before that declaration).
        The lateral-force column itself is resolved by ``_lateral_columns``,
        which likewise accepts ``lfx``/``lfy`` or ``raw_lfx``/``raw_lfy``.
        """
        if df.attrs.get('lf_corrected'):
            # Re-entry guard: scaling twice gives 1.602^2. Not reachable from
            # src/ today (each call site passes a freshly parsed frame, and a
            # fresh read_csv starts with empty attrs), but the function is
            # public enough to be called on a frame that already went through.
            return df
        df.attrs['lf_corrected'] = False
        schema = df.attrs.get('lf_schema', 1)
        if sim_type != 'sheet' or schema != 1:
            if schema > self._LF_SCHEMA_CURRENT or schema < 1:
                logger.warning(
                    "%s: output schema %s is not one this reader understands "
                    "(known: 1..%s). Lateral force left untouched; upgrade "
                    "the reader rather than trusting this frame.",
                    filepath, schema, self._LF_SCHEMA_CURRENT)
            return df
        lateral_cols = self._lateral_columns(df)
        if lateral_cols is None:
            return df                  # no lateral force to correct at all
        lfx_name, lfy_name = lateral_cols
        branch = self._branch_column(df)
        if branch is None:
            # There IS a lateral force but no react_x/v_sx to identify the
            # branch with, so it cannot be known whether it needs scaling.
            # This is the one exit that abandons a possibly-affected file:
            # say so.
            logger.warning(
                "%s: pre-schema-2 sheet file carries %s but no reaction-force "
                "column (react_x/v_sx), so the LJ/many-body branch cannot be "
                "identified. Lateral force left AS WRITTEN; if this run used "
                "an lj/cut interlayer potential its COF is a factor %s too "
                "small.",
                filepath, lfx_name, self._LF_SCHEMA1_CORRECTION)
            return df
        if not self._is_lj_branch(df[lfx_name], branch):
            return df                  # many-body branch: already nN
        df[lfx_name] = df[lfx_name] * self._LF_SCHEMA1_CORRECTION
        if lfy_name in df.columns:
            df[lfy_name] = df[lfy_name] * self._LF_SCHEMA1_CORRECTION
        df.attrs['lf_corrected'] = True
        logger.warning(
            "%s: pre-schema-2 sheet file -- lateral force was eV/A against a "
            "nN load, so COF was a factor %s too small. Multiplied %s/%s by "
            "%s. Re-run this simulation to retire the correction.",
            filepath, self._LF_SCHEMA1_CORRECTION, lfx_name, lfy_name,
            self._LF_SCHEMA1_CORRECTION)
        return df

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
