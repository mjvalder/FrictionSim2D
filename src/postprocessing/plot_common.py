"""Shared Plotter helpers for loading, filtering, fitting, and base drawing."""
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-return-statements,too-few-public-methods
# pylint: disable=missing-class-docstring,trailing-newlines

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_mixin_base import PlotterMixinBase
from .plot_derived import add_plot_derived_columns
from .plot_style import GEM24_COLORS

logger = logging.getLogger(__name__)


class PlotCommonMixin(PlotterMixinBase):
    def _display_dataset_label(self, label: str) -> str:
        """Return configured display label for a dataset key."""
        return self.dataset_display_labels.get(label, label)

    def _get_palette_color(self, color_idx: int) -> str | None:
        """Return a palette colour by index for the active palette."""
        palette = self.settings["colors"].get("palette", "gem12")
        if palette == "gem12":
            return GEM24_COLORS[color_idx % len(GEM24_COLORS)]
        return None

    def _initialize_series_color_map(self) -> None:
        """Build stable color mapping for datasets and named series."""
        self.next_series_color_index = len(self.series_color_map)
        for label in self.labels:
            color_key = self._display_dataset_label(label)
            if color_key in self.series_color_map:
                continue
            self.series_color_map[color_key] = self._get_palette_color(
                self.next_series_color_index,
            )
            self.next_series_color_index += 1

    def _get_consistent_color(
        self,
        dataset_label: str | None = None,
        series_name: str | None = None,
        fallback_index: int = 0,
    ) -> str | None:
        """Return a stable colour for datasets/series across plot types."""
        color_key = None
        if dataset_label is not None:
            color_key = self._display_dataset_label(dataset_label)
        elif series_name is not None:
            color_key = str(series_name)

        if color_key is None:
            return self._get_palette_color(fallback_index)

        if color_key not in self.series_color_map:
            self.series_color_map[color_key] = self._get_palette_color(
                self.next_series_color_index,
            )
            self.next_series_color_index += 1

        return self.series_color_map[color_key]

    @staticmethod
    def _should_use_dataset_colors(
        plot_config: dict, default: bool = False,
    ) -> bool:
        """Return whether dataset-mapped colours should be used."""
        if 'use_dataset_colors' not in plot_config:
            return default
        return bool(plot_config.get('use_dataset_colors'))

    def _resolve_aux_file_path(self, file_path: str) -> Path:
        """Resolve auxiliary data file path for external plotting inputs."""
        p = Path(file_path)
        if p.is_absolute() or p.exists():
            return p

        # Publication configs often keep companion files next to output dirs.
        out_parent_candidate = self.output_dir.parent / p
        if out_parent_candidate.exists():
            return out_parent_candidate

        out_dir_candidate = self.output_dir / p
        if out_dir_candidate.exists():
            return out_dir_candidate

        return p

    def _apply_series_transform(
        self,
        expr: str,
        values: list[float],
        *,
        x_values: list[float] | None = None,
    ) -> list[float] | None:
        """Apply a restricted lambda-style multiplicative transform.

        Supported examples:
        - ``lambda x: x * 98``
        - ``lambda x, y: x * 98 * y``
        """
        compact = ''.join(str(expr).split())

        if compact.startswith('lambdax:'):
            rhs = compact[len('lambdax:'):]
            tokens = rhs.split('*') if rhs else []
            if not tokens:
                return None
            out = []
            for xv in values:
                prod = 1.0
                for token in tokens:
                    if token == 'x':
                        prod *= float(xv)
                    else:
                        try:
                            prod *= float(token)
                        except ValueError:
                            return None
                out.append(prod)
            return out

        if compact.startswith('lambdax,y:'):
            if x_values is None or len(x_values) != len(values):
                return None
            rhs = compact[len('lambdax,y:'):]
            tokens = rhs.split('*') if rhs else []
            if not tokens:
                return None
            out = []
            for x_val, y_val in zip(x_values, values):
                prod = 1.0
                for token in tokens:
                    if token == 'x':
                        prod *= float(x_val)
                    elif token == 'y':
                        prod *= float(y_val)
                    else:
                        try:
                            prod *= float(token)
                        except ValueError:
                            return None
                out.append(prod)
            return out

        return None

    @staticmethod
    def _material_names_match(left: str, right: str) -> bool:
        """Return True when material identifiers should be considered equal."""
        left_s = str(left)
        right_s = str(right)
        if left_s in right_s or right_s in left_s:
            return True
        return left_s.replace('_', '') in right_s.replace('_', '')

    # =====================================================================
    # INITIALISATION HELPERS
    # =====================================================================

    def _deep_merge_dict(self, d1: dict, d2: dict) -> None:
        """Recursively merge *d2* into *d1*."""
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                self._deep_merge_dict(d1[k], v)
            elif k in d1 and isinstance(d1[k], list) and isinstance(v, list):
                d1[k].extend(v)
            else:
                d1[k] = v

    def _create_material_type_map(self) -> None:
        """Create a map from material_id to material_type from metadata."""
        material_types_dict = self.metadata.get('material_types')
        if isinstance(material_types_dict, dict):
            self.material_type_map = {
                material_id.strip(): type_name.strip()
                for type_name, material_list in material_types_dict.items()
                for material_id in material_list
            }
        else:
            logger.warning(
                "'material_types' not found in metadata. "
                "Plotting by type may fail."
            )

    def _discover_data_files(self) -> None:
        """Find all ``output_full_*.json`` files in each data directory.

        Entries whose ``data_dir`` starts with ``db://`` are DB-backed datasets
        and are skipped here — they are loaded by :meth:`_load_db_labels`.
        """
        for label, data_dir in zip(self.labels, self.data_dirs):
            if str(data_dir).startswith('db://'):
                logger.debug("Skipping file discovery for DB-backed label '%s'", label)
                continue

            search_dir = Path(data_dir) / 'outputs'

            if not search_dir.is_dir():
                logger.debug(
                    "'outputs' not found in %s, searching base directory.",
                    data_dir,
                )
                search_dir = Path(data_dir)

            if not search_dir.is_dir():
                logger.error(
                    "Data directory not found for label '%s': %s",
                    label, data_dir,
                )
                continue

            for entry in search_dir.iterdir():
                match = re.match(r'output_full_(.+)\.json', entry.name)
                if match:
                    file_key = match.group(1)
                    self.full_data_files[label][file_key] = str(entry)

            if not self.full_data_files[label]:
                logger.warning(
                    "No 'output_full_*.json' files found for label '%s'",
                    label,
                )

    # =====================================================================
    # DB-BACKED DATA SOURCE  (db:// URI scheme)
    # =====================================================================

    @staticmethod
    def _parse_db_uri(data_dir: str) -> dict | None:
        """Parse a ``db://`` URI into a dict of query parameters.

        URI format::

            db://profile?uploader=NAME&simulation_type=sheetonsheet&layers=2

        Supported query params: ``uploader``, ``simulation_type``, ``layers``,
        ``material``, ``limit``.  ``profile`` defaults to ``'local'``.

        Returns ``None`` if *data_dir* is not a ``db://`` URI.
        """
        if not str(data_dir).startswith('db://'):
            return None
        from urllib.parse import urlparse, parse_qs  # noqa: PLC0415
        parsed = urlparse(str(data_dir))
        params: dict = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        params['profile'] = parsed.netloc or 'local'
        if 'layers' in params:
            params['layers'] = int(params['layers'])
        if 'limit' in params:
            params['limit'] = int(params['limit'])
        return params

    @staticmethod
    def _db_df_to_summary_rows(label: str, db_df: 'pd.DataFrame') -> list[dict]:
        """Convert a :class:`~src.data.database.FrictionDB` query result to
        rows compatible with the summary DataFrame produced by
        :meth:`_calculate_summary_statistics`.

        Notes
        -----
        * ``lf`` is mapped from ``mean_lf`` (i.e. mean of sqrt(lfx²+lfy²)).
          For plots that use ``lf_source: "sx_sy"`` this is an approximation
          because the DB does not store the v_sx/v_sy-derived lateral force.
        * ``size`` is synthesised from ``size_x`` / ``size_y`` columns.
          Rows where both are NULL receive the key ``"unknownx"``; add
          a ``?size_x=100&size_y=100`` query parameter to the URI if needed.
        """
        rows = []
        for _, row in db_df.iterrows():
            sx = row.get('size_x')
            sy = row.get('size_y')
            file_key = (
                f"{int(sx)}x{int(sy)}y"
                if (sx is not None and sy is not None
                    and not (pd.isna(sx) or pd.isna(sy)))
                else 'unknownx'
            )
            mean_lf = row.get('mean_lf')
            record: dict = {
                'dataset_label': label,
                'file_key': file_key,
                'id': row.get('material', ''),
                'force': row.get('force_nN'),
                'pressure': row.get('pressure_gpa'),
                'angle': row.get('scan_angle', 0.0),
                'layer': row.get('layers', 1),
                'speed': row.get('scan_speed'),
                'potential_type': row.get('potential_type'),
                'size': file_key,
                # Columns used by plot x_axis / y_axis
                'nf': row.get('mean_nf'),
                'lf': mean_lf,
                'lfx': row.get('mean_lfx'),
                'lfy': row.get('mean_lfy'),
                # lf_sx_sy: DB stores v_xfrict-based lf; use as best available proxy
                'lf_sx_sy': mean_lf,
                'lf_as_was': mean_lf,
                'lf_fx_fy': mean_lf,
                'cof': row.get('mean_cof'),
                'mean_nf': row.get('mean_nf'),
                'mean_lf': mean_lf,
                'mean_cof': row.get('mean_cof'),
                'std_cof': row.get('std_cof'),
            }
            rows.append(record)
        return rows

    def _load_db_labels(self) -> None:
        """Query the DB for any label whose ``data_dir`` is a ``db://`` URI.

        Results are stored in ``self._db_summary_rows`` and merged into
        :attr:`summary_df_cache` on the next call to
        :meth:`_get_summary_data_df`.
        """
        for label, data_dir in zip(self.labels, self.data_dirs):
            db_params = self._parse_db_uri(str(data_dir))
            if db_params is None:
                continue

            profile = db_params.pop('profile', 'local')
            # Only pass params that FrictionDB.query accepts
            _QUERY_KEYS = {'material', 'simulation_type', 'layers',
                           'uploader', 'limit'}
            query_kwargs = {k: v for k, v in db_params.items()
                            if k in _QUERY_KEYS}

            try:
                from ..data.database import db_from_profile  # noqa: PLC0415
                db = db_from_profile(profile)
                db_df = db.query(**query_kwargs)
                rows = self._db_df_to_summary_rows(label, db_df)
                self._db_summary_rows.extend(rows)
                logger.info(
                    "Loaded %d rows for DB-backed label '%s' (profile=%s, filters=%s)",
                    len(rows), label, profile, query_kwargs,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "DB query failed for label '%s': %s", label, exc
                )

    def _load_all_metadata(self) -> None:
        """Load and merge metadata from all available data files."""
        for label in self.labels:
            if not self.full_data_files[label]:
                continue
            for file_key in self.full_data_files[label]:
                _, metadata = self._load_full_data(label, file_key)
                if metadata:
                    self._deep_merge_dict(self.metadata, metadata)

    def _load_full_data(
        self, label: str, file_key: str,
    ) -> tuple[dict | None, dict | None]:
        """Load a single data file and return ``(results, metadata)``."""
        file_path = self.full_data_files.get(label, {}).get(file_key)
        if not file_path:
            logger.debug(
                "No data file for label '%s', file_key '%s'",
                label, file_key,
            )
            return None, None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('results', {}), data.get('metadata', {})
        except (IOError, json.JSONDecodeError) as e:
            logger.error("Loading data from %s: %s", file_path, e)
            return None, None

    # =====================================================================
    # DATA EXTRACTION
    # =====================================================================

    def _extract_all_runs(self, label: str, file_key: str):
        """Yield a dict for each simulation run found in the data file."""
        results, _ = self._load_full_data(label, file_key)
        if not results:
            return

        def process_level(data_dict, params_so_far):
            if 'columns' in data_dict and 'data' in data_dict:
                df = pd.DataFrame(
                    data_dict['data'], columns=data_dict['columns'],
                )
                df = self._add_derived_columns(df)
                run_data = params_so_far.copy()
                run_data['df'] = df
                yield run_data
                return

            for key, value in data_dict.items():
                if not isinstance(value, dict):
                    continue
                new_params = params_so_far.copy()
                if 'id' not in new_params:
                    new_params['id'] = key.strip()
                else:
                    match_prefix = re.match(
                        r'([a-zA-Z]+)(\d+\.?\d*)', key,
                    )
                    if match_prefix:
                        prefix, val_str = match_prefix.groups()
                        val = float(val_str)
                        param_map = {
                            'f': 'force', 'a': 'angle',
                            'r': 'tip_radius', 'l': 'layer',
                            's': 'speed', 'p': 'pressure',
                        }
                        if prefix in param_map:
                            new_params[param_map[prefix]] = val
                    if m := re.match(r'(\d+\.?\d*)nN', key):
                        new_params['force'] = float(m.group(1))
                    if m := re.match(r'(\d+\.?\d*)deg', key):
                        new_params['angle'] = float(m.group(1))
                yield from process_level(value, new_params)

        yield from process_level(results, {})

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived quantities to a DataFrame."""
        return add_plot_derived_columns(df, self.time_step_fs)

    def _resolve_lf_axis(
        self, df: pd.DataFrame, y_col: str, plot_config: dict,
    ) -> str:
        """Resolve LF y-axis column based on optional ``lf_source`` setting."""
        if y_col != 'lf':
            return y_col

        lf_source = plot_config.get('lf_source')
        if not lf_source:
            return y_col

        lf_source_map = {
            'as_was': 'lf_as_was',
            'sx_sy': 'lf_sx_sy',
            'fx_fy': 'lf_fx_fy',
        }
        target_col = lf_source_map.get(lf_source, 'lf')

        if target_col in df.columns:
            return target_col
        if 'lf' in df.columns:
            logger.warning(
                "Requested lf_source='%s' but column '%s' is unavailable. "
                "Falling back to 'lf'.",
                lf_source,
                target_col,
            )
            return 'lf'
        return target_col

    # =====================================================================
    # SUMMARY DATA
    # =====================================================================

    def _get_summary_data_df(
        self,
        first_n: int | None = None,
        timestep_range: list[int] | None = None,
    ) -> pd.DataFrame:
        """Return summary DataFrame.

        When ``first_n`` or ``timestep_range`` is provided, a non-cached
        summary is computed for that window.
        """
        if first_n is not None or timestep_range is not None:
            return self._calculate_summary_statistics(
                first_n=first_n,
                timestep_range=timestep_range,
            )
        if self.summary_df_cache is None:
            self.summary_df_cache = self._calculate_summary_statistics()
        return self.summary_df_cache

    def _calculate_summary_statistics(
        self,
        first_n: int | None = None,
        timestep_range: list[int] | None = None,
    ) -> pd.DataFrame:
        """Calculate summary statistics for all runs."""
        logger.info("Calculating summary statistics...")
        all_records = []

        for label in self.full_data_files:
            for file_key in self.full_data_files[label]:
                for run_data in self._extract_all_runs(label, file_key):
                    df = run_data.pop('df')
                    original_len = len(df)

                    if timestep_range is not None and len(timestep_range) == 2:
                        df = df.iloc[timestep_range[0]:timestep_range[1]]
                        logger.info(
                            "Applied timestep range [%s:%s]: %d -> %d steps",
                            timestep_range[0],
                            timestep_range[1],
                            original_len,
                            len(df),
                        )
                    elif first_n is not None:
                        df = df.iloc[:first_n]
                        logger.info(
                            "Applied first %d timesteps: %d -> %d steps",
                            first_n,
                            original_len,
                            len(df),
                        )

                    summary_stats = df.mean().to_dict()
                    record = {
                        'dataset_label': label,
                        'file_key': file_key,
                        **run_data,
                        **summary_stats,
                    }
                    all_records.append(record)

        # Append rows from DB-backed datasets (loaded in _load_db_labels).
        # DB rows are pre-computed and not affected by first_n / timestep_range.
        if self._db_summary_rows and first_n is None and timestep_range is None:
            all_records.extend(self._db_summary_rows)

        summary_df = pd.DataFrame(all_records)

        if (
            not summary_df.empty
            and 'id' in summary_df.columns
        ):
            summary_df['material_type'] = (
                summary_df['id'].map(self.material_type_map)
            )
            if 'size' not in summary_df.columns:
                summary_df['size'] = (
                    summary_df['file_key']
                    .str.extract(r'(\d+x\d+y?)')[0]
                )

        if first_n is None and timestep_range is None:
            self.summary_df_cache = summary_df

        logger.debug(
            "Summary DataFrame shape %s", summary_df.shape,
        )
        return summary_df

    # =====================================================================
    # FILTERING
    # =====================================================================

    def _apply_default_filters(
        self, df: pd.DataFrame, plot_config: dict, x_col: str | None = None,
    ) -> dict:
        """Apply default filters based on data availability."""
        filters = {
            'angle': plot_config.get('angle'),
            'force': plot_config.get('force'),
            'size': plot_config.get('filter_size'),
            'layer': plot_config.get('filter_layer'),
            'speed': plot_config.get('filter_speed'),
            'tip_radius': plot_config.get('filter_tip_radius'),
        }

        if filters['layer'] is None and 'layer' in df.columns:
            unique_layers = df['layer'].dropna().unique()
            if 1 in unique_layers:
                logger.debug("Defaulting to layer 1")
                filters['layer'] = 1

        if x_col == 'force' and filters['angle'] is None:
            logger.debug("Defaulting to angle 0.0 for force plot")
            filters['angle'] = 0.0

        return filters

    def _apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """Apply filters to a DataFrame."""
        for key, value in filters.items():
            if value is not None and key in df.columns:
                original_len = len(df)
                if isinstance(value, list):
                    df = df[df[key].isin(value)]
                else:
                    df = df[df[key] == value]
                logger.debug(
                    "Filter '%s' == '%s': %d -> %d",
                    key, value, original_len, len(df),
                )
        return df

    def _apply_range_filters(
        self, df: pd.DataFrame, plot_config: dict,
    ) -> pd.DataFrame:
        """Apply range-based filters (e.g. filter_force_range)."""
        force_range = plot_config.get('filter_force_range')
        if force_range and len(force_range) == 2 and 'force' in df.columns:
            original_len = len(df)
            df = df[
                (df['force'] >= force_range[0])
                & (df['force'] <= force_range[1])
            ]
            logger.debug(
                "Force range filter [%s, %s]: %d -> %d",
                force_range[0], force_range[1], original_len, len(df),
            )

        nf_range = plot_config.get('filter_nf_range')
        if nf_range and len(nf_range) == 2 and 'nf' in df.columns:
            original_len = len(df)
            df = df[(df['nf'] >= nf_range[0]) & (df['nf'] <= nf_range[1])]
            logger.debug(
                "Normal force range filter [%s, %s]: %d -> %d",
                nf_range[0],
                nf_range[1],
                original_len,
                len(df),
            )
        return df

    def _apply_material_filter(
        self, df: pd.DataFrame, plot_config: dict, plot_by: str,
    ) -> pd.DataFrame:
        """Apply material/type filters based on plot_by mode."""
        filter_materials = plot_config.get('filter_materials')
        if not filter_materials:
            return df

        filter_values = [v.strip() for v in filter_materials]
        original_len = len(df)

        if plot_by in ('id', 'id_angle'):
            escaped = [re.escape(v) for v in filter_values]
            pattern = '|'.join(
                [f'(?:^|_){v}(?:_|$)' for v in escaped],
            )
            df = df[df['id'].str.contains(pattern, regex=True)]
        elif plot_by == 'material_type':
            ids_to_plot = [
                mid for mid, mtype in self.material_type_map.items()
                if mtype in filter_values
            ]
            df = df[df['id'].isin(ids_to_plot)]

        logger.debug("Material filter: %d -> %d", original_len, len(df))
        return df

    def _remove_outliers(
        self, df: pd.DataFrame, x_col: str, y_col: str,
        threshold: float = 10.0,
    ) -> pd.DataFrame:
        """Remove outliers based on magnitude relative to median."""
        if df.empty:
            return df

        initial_rows = len(df)

        def remove_magnitude_outliers(group):
            if len(group) < 3:
                return group
            median_y = group[y_col].median()
            if abs(median_y) < 1e-6:
                return group
            is_outlier = np.abs(group[y_col]) > threshold * np.abs(median_y)
            return group[~is_outlier]

        group_cols = [x_col]
        if 'dataset_label' in df.columns:
            group_cols = ['dataset_label', x_col]

        cleaned_groups = [
            remove_magnitude_outliers(group)
            for _, group in df.groupby(group_cols)
        ]
        cleaned_df = (
            pd.concat(cleaned_groups, ignore_index=True)
            if cleaned_groups else df.iloc[0:0].copy()
        )

        removed = initial_rows - len(cleaned_df)
        if removed > 0:
            logger.debug("Removed %d outlier points", removed)

        return cleaned_df

    # =====================================================================
    # LINEAR FIT
    # =====================================================================

    def _calculate_linear_fit(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_range: list | None = None,
        constraint: str = 'first_point',
    ) -> dict | None:
        """Calculate constrained linear regression through the first point.

        Returns fit parameters or ``None`` if insufficient data.
        """
        if len(x_data) < 2:
            return None

        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = np.array(x_data)[mask]
        y_clean = np.array(y_data)[mask]

        if x_range is not None and len(x_range) == 2:
            range_mask = (x_clean >= x_range[0]) & (x_clean <= x_range[1])
            x_clean, y_clean = x_clean[range_mask], y_clean[range_mask]

        if len(x_clean) < 2:
            return None

        x_origin = 0.0
        if constraint == 'origin':
            denom = np.sum(x_clean**2)
            if denom == 0:
                return None
            slope = np.sum(x_clean * y_clean) / denom
            intercept = 0.0
            x_shifted = x_clean
        elif constraint == 'none':
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            x_shifted = x_clean - np.mean(x_clean)
        else:
            x_origin, y_origin = x_clean[0], y_clean[0]
            x_shifted = x_clean - x_origin
            y_shifted = y_clean - y_origin

            denom = np.sum(x_shifted**2)
            if denom == 0:
                return None

            slope = np.sum(x_shifted * y_shifted) / denom
            intercept = y_origin - slope * x_origin

        y_pred = slope * x_clean + intercept
        residuals = y_clean - y_pred
        rmse = np.sqrt(np.mean(residuals**2))

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        n = len(x_clean)
        if n > 1:
            mse = ss_res / (n - 1)
            slope_stderr = np.sqrt(mse / np.sum(x_shifted**2))
        else:
            slope_stderr = 0

        if constraint == 'first_point':
            se_intercept = slope_stderr * np.abs(x_origin)
        else:
            se_intercept = 0

        return {
            'slope': slope,
            'intercept': intercept,
            'slope_stderr': slope_stderr,
            'intercept_stderr': se_intercept,
            'r_squared': r_squared,
            'rmse': rmse,
        }

    # =====================================================================
    # PLOTTING HELPERS
    # =====================================================================

    def _plot_series(
        self, ax, x_data, y_data, label, plot_style, add_fit,
        fit_x_range, std_data=None, *, show_error_bands=True,
        color_idx=0, line_style=None, marker_style=None,
        marker_face='filled', fit_only=False,
        fit_constraint='first_point', custom_color=None,
        extend_fit_to_origin=False,
    ):
        """Plot a single data series with optional fit and error bands.

        Returns the line colour used.
        """
        s = self.settings
        marker = marker_style if marker_style is not None else s["markers"]["style"]

        palette = s["colors"].get("palette", "gem12")
        if custom_color is not None:
            color = custom_color
        elif palette == "gem12":
            color = GEM24_COLORS[color_idx % len(GEM24_COLORS)]
        else:
            color = None

        marker_facecolor = color if marker_face == 'filled' else 'none'

        if plot_style == 'scatter':
            scatter = ax.scatter(
                x_data, y_data, label=label,
                s=s["markers"]["size"]**2, color=color,
                marker=marker, facecolors=marker_facecolor,
                edgecolors=color if color is not None else None,
            )
            if color is None:
                color = scatter.get_facecolors()[0]
        else:
            if fit_only:
                linestyle = 'None'
            elif line_style is not None:
                linestyle = line_style
            else:
                linestyle = '' if add_fit else '-'
            line = ax.plot(
                x_data, y_data, marker=marker, linestyle=linestyle,
                linewidth=s["lines"]["width"], label=label, color=color,
                markersize=s["markers"]["size"],
                markerfacecolor=marker_facecolor,
                markeredgecolor=color if color is not None else None,
            )
            if color is None:
                color = line[0].get_color()

        if std_data is not None and show_error_bands:
            lower_bound = np.maximum(y_data - std_data, 0)
            upper_bound = y_data + std_data
            ax.fill_between(
                x_data, lower_bound, upper_bound,
                alpha=s["error_bands"]["alpha"], color=color,
            )

        if add_fit:
            fit_params = self._calculate_linear_fit(
                np.array(x_data), np.array(y_data), fit_x_range,
                constraint=fit_constraint,
            )
            if fit_params:
                x_min = (
                    0 if extend_fit_to_origin
                    else (fit_x_range[0] if fit_x_range else x_data.min())
                )
                x_max = fit_x_range[1] if fit_x_range else x_data.max()
                x_fit = np.linspace(x_min, x_max, 100)
                y_fit = fit_params['slope'] * x_fit + fit_params['intercept']
                fit_linestyle = (
                    line_style if line_style is not None
                    else s["lines"]["fit_style"]
                )
                ax.plot(
                    x_fit, y_fit, linestyle=fit_linestyle,
                    alpha=s["lines"]["fit_alpha"],
                    linewidth=s["lines"]["width"], color=color,
                )

        return color

    def _finalize_plot(
        self, ax, plot_config: dict,
        x_col: str | None = None, y_col: str | None = None,
        *, x_label: str | None = None, y_label: str | None = None,
    ) -> None:
        """Apply final formatting to a plot (MATLAB-style)."""
        s = self.settings

        ax.set_xlabel(
            x_label or plot_config.get('x_label', x_col or ''),
            fontsize=s["fonts"]["axis_label"],
        )
        ax.set_ylabel(
            y_label or plot_config.get('y_label', y_col or ''),
            fontsize=s["fonts"]["axis_label"],
        )
        ax.tick_params(
            axis='both', which='major',
            labelsize=s["fonts"]["tick_label"],
        )

        axes_config = s.get("axes", {})
        use_sci = plot_config.get(
            'use_scientific_notation',
            axes_config.get('use_scientific_notation', False),
        )
        if use_sci:
            scilimits = axes_config.get('scilimits', [-2, 2])
            ax.ticklabel_format(
                style='sci', axis='both',
                scilimits=tuple(scilimits), useMathText=True,
            )
            ax.xaxis.get_offset_text().set_fontsize(s["fonts"]["tick_label"])
            ax.yaxis.get_offset_text().set_fontsize(s["fonts"]["tick_label"])

        title = plot_config.get('title')
        if title:
            ax.set_title(title, fontsize=s["fonts"]["title"])

        grid_config = s.get("grid", {})
        if grid_config.get("show", True):
            which = grid_config.get("which", "both")
            if which in ("major", "both"):
                ax.grid(
                    True, which='major',
                    linestyle=grid_config.get("major_style", "-"),
                    alpha=grid_config.get("major_alpha", 0.5),
                )
            if which in ("minor", "both"):
                ax.minorticks_on()
                ax.grid(
                    True, which='minor',
                    linestyle=grid_config.get("minor_style", ":"),
                    alpha=grid_config.get("minor_alpha", 0.3),
                )

        if ax.get_legend_handles_labels()[1]:
            legend_outside = plot_config.get('legend_outside', False)
            if legend_outside:
                ax.legend(
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=s["fonts"]["legend"],
                )
            else:
                legend_loc = plot_config.get(
                    'legend_location', s["legend"]["location"],
                )
                ax.legend(loc=legend_loc, fontsize=s["fonts"]["legend"])

    def _apply_axis_limits(self, ax, plot_config: dict) -> None:
        """Apply explicit x and y axis limits from plot config."""
        y_limits = plot_config.get('y_limits')
        if y_limits:
            if y_limits[0] is not None:
                ax.set_ylim(bottom=y_limits[0])
            if len(y_limits) > 1 and y_limits[1] is not None:
                ax.set_ylim(top=y_limits[1])

        x_limits = plot_config.get('x_limits')
        if x_limits:
            if x_limits[0] is not None:
                ax.set_xlim(left=x_limits[0])
            if len(x_limits) > 1 and x_limits[1] is not None:
                ax.set_xlim(right=x_limits[1])

    def _save_plot(
        self,
        fig,
        filename: str | None,
        tight_bbox: bool = True,
    ) -> None:
        """Save the plot to file(s) in configured formats."""
        if not filename:
            logger.warning("No filename specified. Plot not saved.")
            plt.close(fig)
            return

        export_config = self.settings.get("export", {})
        formats = export_config.get("formats", ["png"])
        transparent = export_config.get("transparent", False)

        base_name = Path(filename).stem
        original_ext = Path(filename).suffix.lstrip('.')

        if original_ext and original_ext not in formats:
            formats = [original_ext] + list(formats)

        try:
            if tight_bbox:
                fig.subplots_adjust(
                    left=0.13, bottom=0.13, right=0.97, top=0.97,
                )
            else:
                fig.subplots_adjust(
                    left=0.16, bottom=0.24, right=0.98, top=0.92,
                )
        except (TypeError, ValueError) as e:
            logger.warning("subplots_adjust failed: %s", e)

        for fmt in formats:
            output_path = self.output_dir / f"{base_name}.{fmt}"
            fig.savefig(
                output_path, dpi=self.settings["figure"]["dpi"],
                format=fmt, transparent=transparent,
                bbox_inches='tight' if tight_bbox else None,
                pad_inches=0.1 if tight_bbox else 0.0,
            )
            logger.info("Generated plot: %s", output_path)

        plt.close(fig)

    # =====================================================================
    # SUMMARY PLOT
    # =====================================================================

    def _load_external_json(
        self, source_config: dict,
    ) -> tuple[list[str] | None, np.ndarray | None, np.ndarray | None]:
        """Load data from an external JSON file with materials mapping."""
        file_path = source_config.get('file')
        if not file_path:
            return None, None, None

        resolved_path = self._resolve_aux_file_path(str(file_path))

        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            raw_materials = data.get(
                source_config.get('material_column', 'materials'), [],
            )
            materials = [str(m) for m in raw_materials]
            values = np.array(
                data.get(source_config.get('value_column', 'tribIndex'), []),
            )
            errors = np.array(
                data.get(source_config.get('error_column', 'dev'), []),
            )

            return materials, values, errors
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error("Loading external JSON file %s: %s", resolved_path, e)
            return None, None, None

    def _get_aggregated_source_data(
        self,
        summary_df: pd.DataFrame,
        source_config: dict,
        material_list: list[str] | None = None,
    ) -> tuple[list[str] | None, np.ndarray | None, np.ndarray | None]:
        """Extract data with optional force-range averaging.

        Returns ``(materials, values, errors)`` arrays.
        """
        dataset = source_config.get('dataset')
        metric = source_config.get('metric')

        if not dataset or not metric:
            logger.error("Source must specify 'dataset' and 'metric'")
            return None, None, None

        df = summary_df[summary_df['dataset_label'] == dataset].copy()

        for key in ('filter_layer', 'filter_size'):
            value = source_config.get(key)
            col = key.replace('filter_', '')
            if value is not None and col in df.columns:
                df = df[df[col] == value]

        angle = source_config.get('angle')
        if angle is not None and 'angle' in df.columns:
            if isinstance(angle, list):
                df = df[df['angle'].isin(angle)]
            else:
                df = df[df['angle'] == angle]

        if material_list is not None and 'id' in df.columns:
            matched_ids = []
            for mat in material_list:
                for df_id in df['id'].unique():
                    if self._material_names_match(str(mat), str(df_id)):
                        matched_ids.append(df_id)
            df = df[df['id'].isin(matched_ids)]

        if df.empty:
            logger.warning("No data found for source: %s", source_config)
            return None, None, None

        force_range = source_config.get('force_range')
        error_metric = source_config.get('error_metric', 'slope_stderr')

        materials_out: list[str] = []
        values_out: list = []
        errors_out: list = []

        for mat_id in df['id'].unique():
            mat_df = df[df['id'] == mat_id].copy()

            if force_range and len(force_range) == 2:
                force_col = 'force' if 'force' in mat_df.columns else 'nf' if 'nf' in mat_df.columns else None
                if force_col is None:
                    logger.warning(
                        "Skipping source %s for material %s: no force/nf column available",
                        source_config,
                        mat_id,
                    )
                    continue

                mat_df = mat_df[
                    (mat_df[force_col] >= force_range[0])
                    & (mat_df[force_col] <= force_range[1])
                ]

                if len(mat_df) < 2:
                    continue

                avg_value = mat_df[metric].mean()
                fit = self._calculate_linear_fit(
                    mat_df[force_col].values, mat_df[metric].values,
                )

                if fit:
                    error_map = {
                        'slope_stderr': fit['slope_stderr'],
                        'rmse': fit['rmse'],
                        'r_squared': 1 - fit['r_squared'],
                    }
                    error = error_map.get(error_metric, fit['slope_stderr'])
                else:
                    error = mat_df[metric].std()

                materials_out.append(mat_id)
                values_out.append(avg_value)
                errors_out.append(error)
            else:
                force = source_config.get('force')
                if force is not None:
                    mat_df = mat_df[mat_df['force'] == force]

                if mat_df.empty:
                    continue

                materials_out.append(mat_id)
                values_out.append(mat_df[metric].mean())
                errors_out.append(
                    mat_df[metric].std() if len(mat_df) > 1 else 0,
                )

        return materials_out, np.array(values_out), np.array(errors_out)

    @staticmethod
    def _iterative_outlier_removal(
        x: np.ndarray, y: np.ndarray,
        x_err: np.ndarray | None, y_err: np.ndarray | None,
        materials: list, num_remove: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Iteratively remove points with largest residuals from OLS fit."""
        from scipy import stats as scipy_stats

        x = np.array(x)
        y = np.array(y)
        x_err = np.array(x_err) if x_err is not None else np.zeros_like(x)
        y_err = np.array(y_err) if y_err is not None else np.zeros_like(y)
        materials = list(materials)
        removed = []

        for _ in range(min(num_remove, len(x) - 2)):
            if len(x) <= 2:
                break

            slope, intercept, _, _, _ = scipy_stats.linregress(x, y)
            y_pred = slope * x + intercept
            residuals = np.abs(y - y_pred)

            max_idx = np.argmax(residuals)
            removed.append(materials[max_idx])

            mask = np.ones(len(x), dtype=bool)
            mask[max_idx] = False
            x = x[mask]
            y = y[mask]
            x_err = x_err[mask]
            y_err = y_err[mask]
            materials = [m for i, m in enumerate(materials) if mask[i]]

        if removed:
            logger.info(
                "Outlier removal: removed %d points: %s",
                len(removed), removed,
            )

        return x, y, x_err, y_err, materials

    def _generate_scatter_comparison(self, plot_config: dict) -> None:
        """Generate a scatter plot comparing two data sources."""
        x_source = plot_config.get('x_source', {})
        y_source = plot_config.get('y_source', {})

        if not x_source or not y_source:
            logger.error(
                "scatter_comparison requires 'x_source' and 'y_source'",
            )
            return

        summary_df = self._get_summary_data_df()

        # Load data from both sources
        if 'file' in x_source:
            x_materials, x_data, x_errors = self._load_external_json(
                x_source,
            )
            if x_materials is None:
                return

            y_materials, y_data, y_errors = self._get_aggregated_source_data(
                summary_df, y_source, x_materials,
            )
            if y_materials is None:
                return

            # Match materials between x and y
            matched_x, matched_y = [], []
            matched_x_err, matched_y_err = [], []
            matched_materials = []

            for i, x_mat in enumerate(x_materials):
                for j, y_mat in enumerate(y_materials):
                    if self._material_names_match(str(x_mat), str(y_mat)):
                        if x_data is None or y_data is None:
                            continue
                        matched_x.append(x_data[i])
                        matched_y.append(y_data[j])
                        matched_x_err.append(
                            x_errors[i] if x_errors is not None and len(x_errors) > i else 0,
                        )
                        matched_y_err.append(
                            y_errors[j] if y_errors is not None and len(y_errors) > j else 0,
                        )
                        matched_materials.append(x_mat)
                        break

            x_data = np.array(matched_x)
            y_data = np.array(matched_y)
            x_errors = np.array(matched_x_err)
            y_errors = np.array(matched_y_err)
            materials = matched_materials
        else:
            x_materials, x_data, x_errors = self._get_aggregated_source_data(
                summary_df, x_source,
            )
            y_materials, y_data, y_errors = self._get_aggregated_source_data(
                summary_df, y_source, x_materials,
            )

            if x_data is None or y_data is None:
                logger.error(
                    "Could not extract data for scatter comparison",
                )
                return

            materials = x_materials if x_materials else []
            if x_errors is None:
                x_errors = np.zeros_like(x_data)
            if y_errors is None:
                y_errors = np.zeros_like(y_data)

        if len(x_data) == 0 or len(y_data) == 0:
            logger.error("No matched data points for scatter comparison")
            return

        if len(x_data) != len(y_data):
            logger.warning(
                "Data size mismatch: x=%d, y=%d. Using minimum.",
                len(x_data), len(y_data),
            )
            min_len = min(len(x_data), len(y_data))
            x_data = x_data[:min_len]
            y_data = y_data[:min_len]
            x_errors = x_errors[:min_len]
            y_errors = y_errors[:min_len]
            materials = materials[:min_len]

        num_outliers = plot_config.get('iterative_outlier_removal', 0)
        if num_outliers > 0:
            x_data, y_data, x_errors, y_errors, materials = (
                self._iterative_outlier_removal(
                    x_data, y_data, x_errors, y_errors,
                    materials, num_outliers,
                )
            )

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        palette = self.settings["colors"].get("palette", "gem12")
        show_error_bars = plot_config.get('show_error_bars', False)
        color_by_class = plot_config.get('color_by_material_class', False)

        if color_by_class and materials:
            self._plot_scatter_by_class(
                ax, x_data, y_data, x_errors, y_errors,
                materials, palette, show_error_bars,
            )
        else:
            point_color = GEM24_COLORS[0] if palette == "gem12" else None
            if (
                show_error_bars
                and (np.any(x_errors > 0) or np.any(y_errors > 0))
            ):
                ax.errorbar(
                    x_data, y_data,
                    xerr=x_errors if np.any(x_errors > 0) else None,
                    yerr=y_errors if np.any(y_errors > 0) else None,
                    fmt='o', color=point_color,
                    markersize=self.settings["markers"]["size"],
                    capsize=3, capthick=1, elinewidth=1,
                )
            else:
                ax.scatter(
                    x_data, y_data,
                    s=self.settings["markers"]["size"]**2,
                    color=point_color,
                )

        # Point labels
        if plot_config.get('show_point_labels', False) and materials:
            x_range = x_data.max() - x_data.min()
            dx = x_range * 0.01
            label_fontsize = plot_config.get('point_label_fontsize', 8)
            for i, mat in enumerate(materials):
                ax.text(
                    x_data[i] + dx, y_data[i], mat,
                    fontsize=label_fontsize,
                )

        # y=x reference line
        if plot_config.get('show_identity_line', False):
            lims = [
                min(x_data.min(), y_data.min()),
                max(x_data.max(), y_data.max()),
            ]
            ax.plot(lims, lims, '--', color='gray', alpha=0.5, label='y=x')

        # Linear fit
        fit = None
        if plot_config.get('add_linear_fit', False):
            fit = self._calculate_linear_fit(x_data, y_data)
            if fit:
                x_fit = np.linspace(
                    x_data.min() * 0.9, x_data.max() * 1.1, 100,
                )
                y_fit = fit['slope'] * x_fit + fit['intercept']
                fit_color = GEM24_COLORS[1] if palette == "gem12" else 'red'
                ax.plot(
                    x_fit, y_fit, '--', color=fit_color, alpha=0.8,
                    linewidth=self.settings["lines"]["width"],
                    label=f"Fit (R²={fit['r_squared']:.4f})",
                )

        # R² display
        if plot_config.get('show_r_squared', False):
            if fit is None:
                fit = self._calculate_linear_fit(x_data, y_data)
            if fit:
                r2_text = f"R² = {fit['r_squared']:.4f}"
                if num_outliers > 0:
                    r2_text += f" (Filtered {num_outliers})"
                ax.text(
                    0.05, 0.95, r2_text, transform=ax.transAxes,
                    fontsize=self.settings["fonts"]["legend"],
                    fontweight='bold', verticalalignment='top',
                )

        # Axis formatting via shared helpers
        x_label = plot_config.get(
            'x_label',
            f"{x_source.get('dataset', 'X')} {x_source.get('metric', '')}",
        )
        y_label = plot_config.get(
            'y_label',
            f"{y_source.get('dataset', 'Y')} {y_source.get('metric', '')}",
        )
        self._finalize_plot(
            ax, plot_config, x_label=x_label, y_label=y_label,
        )
        self._apply_axis_limits(ax, plot_config)
        self._save_plot(fig, plot_config.get('filename'))

    def _plot_scatter_by_class(
        self, ax, x_data, y_data, x_errors, y_errors,
        materials, palette, show_error_bars,
    ) -> None:
        """Plot scatter points coloured by material class."""
        prefixes = {
            'h_': 'hexagonal', 't_': 'trigonal',
            'p_': 'puckered', 'b_': 'buckled',
        }

        def get_class(mat_name):
            for prefix, class_name in prefixes.items():
                if mat_name.startswith(prefix):
                    return class_name
            return 'bi-buckled'

        class_indices: dict[str, list[int]] = {}
        for i, mat in enumerate(materials):
            class_indices.setdefault(get_class(mat), []).append(i)

        for class_idx, (class_name, indices) in enumerate(
            sorted(class_indices.items()),
        ):
            color = (
                GEM24_COLORS[class_idx % len(GEM24_COLORS)]
                if palette == "gem12" else None
            )

            c_x = x_data[indices]
            c_y = y_data[indices]
            c_x_err = x_errors[indices] if x_errors is not None else None
            c_y_err = y_errors[indices] if y_errors is not None else None

            has_err = (
                (c_x_err is not None and np.any(c_x_err > 0))
                or (c_y_err is not None and np.any(c_y_err > 0))
            )

            if show_error_bars and has_err:
                ax.errorbar(
                    c_x, c_y,
                    xerr=(
                        c_x_err
                        if c_x_err is not None and np.any(c_x_err > 0)
                        else None
                    ),
                    yerr=(
                        c_y_err
                        if c_y_err is not None and np.any(c_y_err > 0)
                        else None
                    ),
                    fmt='o', color=color, label=class_name,
                    markersize=self.settings["markers"]["size"],
                    capsize=3, capthick=1, elinewidth=1,
                )
            else:
                ax.scatter(
                    c_x, c_y,
                    s=self.settings["markers"]["size"]**2,
                    color=color, label=class_name,
                )

    # =====================================================================
    # TIMESERIES PLOT
    # =====================================================================

