"""Shared Plotter helpers for loading, filtering, fitting, and base drawing."""
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-few-public-methods

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.parse import parse_qsl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_mixin_base import PlotDataError, PlotterMixinBase
from .plot_derived import add_plot_derived_columns
from .plot_style import GEM24_COLORS
from .run_status import DELIVERED_STATUSES

logger = logging.getLogger(__name__)

#: Statuses the plot stack draws. A v2 record has no status and counts as
#: complete; truncated and missing points never reach a plot.
PLOT_STATUSES = DELIVERED_STATUSES


#: The plot source naming a stored dataset: ``aiida://dataset/<name>``.
DATASET_URI = 'aiida://dataset/'

#: The plot source naming a stored, summary-shaped analysis result
#: (``period_summary``): ``aiida://analysis/<name>/<dataset>``.
ANALYSIS_URI = 'aiida://analysis/'

#: summary_<size>.csv column -> the name the plot specs use.
SUMMARY_COLUMN_ALIASES = {
    'mean_nf': 'nf', 'mean_lf': 'lf', 'mean_lfx': 'lfx', 'mean_lfy': 'lfy',
    'mean_cof': 'cof', 'mean_lf_sx_sy': 'lf_sx_sy',
    'mean_lf_fx_fy': 'lf_fx_fy',
}


class PlotCommonMixin(PlotterMixinBase):
    """Shared helpers for every plot mixin: labels, filters, fits, saving."""

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

    @staticmethod
    def _filter_runs(all_runs: list[dict], plot_config: dict, *,
                     force=None, pressure=None) -> list[dict]:
        """The runs matching the plot's id/force/pressure/angle/layer/speed/tip_radius.

        ``id`` matches as a substring; a run lacking a filtered key is excluded.
        Layer defaults to 1 when present, angle to 0.
        """
        filters = {
            'id': plot_config.get('filter_materials'),
            'force': force,
            'pressure': pressure,
            'angle': plot_config.get('angle'),
            'layer': plot_config.get('filter_layer'),
            'speed': plot_config.get('filter_speed'),
            'tip_radius': plot_config.get('filter_tip_radius'),
        }
        if filters['layer'] is None:
            layers = {run.get('layer') for run in all_runs if 'layer' in run}
            if 1 in layers:
                filters['layer'] = 1
        if filters['angle'] is None:
            filters['angle'] = 0.0

        def matches(run: dict) -> bool:
            for key, value in filters.items():
                if value is None:
                    continue
                run_value = run.get(key)
                if run_value is None:
                    return False
                if key == 'id' and isinstance(value, list):
                    if not any(v in run_value for v in value):
                        return False
                elif isinstance(value, list):
                    if run_value not in value:
                        return False
                elif run_value != value:
                    return False
            return True

        return [run for run in all_runs if matches(run)]

    @staticmethod
    def _time_axis(run_time: np.ndarray, time_scale: float, *,
                   shift: bool = True) -> np.ndarray:
        """One run's own time column in display units, from 0 unless ``shift``
        is False (which keeps the LAMMPS clock).

        Per run, not per plot: a global `time_series` from the metadata forced
        every run onto the first one's length.
        """
        origin = run_time[0] if (shift and run_time.size) else 0.0
        return (run_time - origin) * time_scale

    def _transformed_xy(self, series: dict, x_data, y_data):
        """``(x, y)`` after the series' ``x_transform``/``y_transform``, if any.

        An unsupported expression is logged and the data left as it was.
        """
        orig_x = list(x_data)
        out = []
        for axis, values, kwargs in (('x', x_data, {}),
                                     ('y', y_data, {'x_values': orig_x})):
            expr = series.get(f'{axis}_transform')
            if expr:
                transformed = self._apply_series_transform(str(expr), values, **kwargs)
                if transformed is not None:
                    values = transformed
                else:
                    logger.warning("Unsupported %s_transform expression '%s'. Ignoring.",
                                   axis, expr)
            out.append(values)
        return out[0], out[1]

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

    def _merge_metadata(self, d1: dict, d2: dict) -> None:
        """Merge one dataset's metadata into the accumulated metadata.

        The OTHER merge: here a list is a SET OF VALUES SEEN and must be
        unioned, because `_load_all_metadata` folds every dataset's
        `materials`/`sizes`/`temperatures` into one. `merge_settings` used to
        serve both call sites, so making settings lists replace (which is the
        only right answer for `--settings`) would have made the second dataset
        silently overwrite the first's material list. Two call sites, two
        meanings, two functions.

        Sorted and de-duplicated, so the accumulated metadata does not depend
        on which dataset was read first.
        """
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                self._merge_metadata(d1[k], v)
            elif k in d1 and isinstance(d1[k], list) and isinstance(v, list):
                try:
                    d1[k] = sorted(set(d1[k]) | set(v))
                except TypeError:
                    # unhashable entries (a nested list): keep insertion order
                    d1[k] = d1[k] + [x for x in v if x not in d1[k]]
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

        Entries whose ``data_dir`` is a URI (``aiida://dataset/<name>``,
        ``aiida://analysis/<name>/<dataset>``, or the removed ``db://``) are
        skipped here; :meth:`_load_dataset_labels` reads or refuses them.
        """
        for label, data_dir in zip(self.labels, self.data_dirs):
            if str(data_dir).startswith(('db://', DATASET_URI, ANALYSIS_URI)):
                logger.debug("Skipping file discovery for store-backed label '%s'",
                             label)
                continue

            search_dir = Path(data_dir) / 'outputs'

            if not search_dir.is_dir():
                logger.debug(
                    "'outputs' not found in %s, searching base directory.",
                    data_dir,
                )
                search_dir = Path(data_dir)

            if not search_dir.is_dir():
                raise PlotDataError(
                    f"{label}: no output_full_*.json under {data_dir} "
                    f"(the directory does not exist). Generate one with "
                    f"`FrictionSim2D postprocess read <root> --export`.")

            for entry in search_dir.iterdir():
                match = re.match(r'output_full_(.+)\.json', entry.name)
                if match:
                    file_key = match.group(1)
                    self.full_data_files[label][file_key] = str(entry)
                summary_match = re.match(r'summary_(.+)\.csv', entry.name)
                if summary_match:
                    self.summary_files.setdefault(label, {})[
                        summary_match.group(1)] = entry

            if not self.full_data_files[label]:
                # Raised, not logged: before this, a mistyped data_dir gave an
                # empty summary frame, an early return in
                # `_generate_summary_plot`, and a user with no plot and no
                # reason. It is also what makes the plot worker's `ok: false`
                # path reachable for a bad document.
                raise PlotDataError(
                    f"{label}: no output_full_*.json under {data_dir}. "
                    f"Generate one with "
                    f"`FrictionSim2D postprocess read <root> --export`.")

    # =====================================================================
    # STORE-BACKED DATA SOURCE  (aiida://dataset/<name>)
    # =====================================================================

    @staticmethod
    def _dataset_to_summary_rows(label: str, rows: list[dict]) -> list[dict]:
        """`frictionsim2d.aiida.query.dataset_rows` rows as the summary rows the CSV
        path yields (`_summary_rows_from_csv`): `id`/`material` in the
        underscored spelling, `size`/`file_key` as `100x100y`, `force` or
        `pressure` by `load_kind`, `layer`, and the plot aliases (`cof` is the
        summary's `mean_cof`, the store's ratio of means).

        The three `lf_*` pair variants are NOT emitted: a summary carries one
        lateral force (`mean_lf`), and aliasing it would make `lf_source` a
        silent no-op. Without the columns `_resolve_lf_axis` warns and falls
        back, which is the honest answer.
        """
        from ..analysis.periods import summary_spelling  # pylint: disable=import-outside-toplevel

        out = []
        for record in rows:
            spelled = summary_spelling(
                record.get('materials'), record.get('flake_material'),
                record.get('size_x'), record.get('size_y'),
                record.get('load_kind'), record.get('load'))
            size = spelled['size']
            row = dict(record)
            row.update({
                'dataset_label': label, 'file_key': size, 'size': size,
                'id': spelled['id'], 'material': spelled['material'],
                'materials': spelled['materials'], 'potential': record.get('pot_token'),
                'layer': record.get('layers'), 'status': 'complete',
                'force': spelled['force'], 'pressure': spelled['pressure'],
            })
            for src, dst in SUMMARY_COLUMN_ALIASES.items():
                if src in record:
                    row[dst] = record[src]
            out.append(row)
        return out

    def _load_dataset_labels(self) -> None:
        """Load every label whose ``data_dir`` is ``aiida://dataset/<name>``.

        The rows are the dataset's (``build_dataset``): complete points with
        the summary the dataset fixed, read from the AiiDA profile in
        effect. ``dataset_units[label]`` keeps the dataset's units (family
        -> key -> unit); each row carries its own ``load_unit``,
        ``angle_unit`` and ``speed_unit``. An ``aiida://analysis/`` label is
        read by :meth:`_load_analysis_label`. A ``db://`` URI is refused: the
        central database was removed.
        """
        for label, data_dir in zip(self.labels, self.data_dirs):
            data_dir = str(data_dir)
            if data_dir.startswith('db://'):
                raise PlotDataError(
                    f"{label}: {data_dir} -- the central database was "
                    f"removed. Read the local AiiDA store with "
                    f"`{DATASET_URI}<name>` instead.")
            if data_dir.startswith(ANALYSIS_URI):
                self._load_analysis_label(label, data_dir)
                continue
            if not data_dir.startswith(DATASET_URI):
                continue
            name = data_dir[len(DATASET_URI):]
            if not name:
                raise PlotDataError(
                    f"{label}: {data_dir} names no dataset; "
                    f"use `{DATASET_URI}<name>`")
            rows, units = self._read_dataset(label, name)
            self._dataset_sources[label] = (name, rows)
            self.dataset_units[label] = units
            self._db_summary_rows.extend(
                self._dataset_to_summary_rows(label, rows))

    def _load_analysis_label(self, label: str, uri: str) -> None:
        """The summary rows of ``aiida://analysis/<name>/<dataset>[?k=v&...]``:
        the stored table of a summary-shaped analysis (``period_summary``)
        under exactly the query's params (none: the defaults), never
        computed here."""
        from .plot_tables import analysis_table  # pylint: disable=import-outside-toplevel
        path, _, query = uri[len(ANALYSIS_URI):].partition('?')
        name, sep, dataset = path.partition('/')
        if not (name and sep and dataset) or '/' in dataset:
            raise PlotDataError(
                f"{label}: {uri} -- expected `{ANALYSIS_URI}<name>/<dataset>[?k=v&...]`")
        try:
            params = dict(parse_qsl(query, keep_blank_values=True,
                                    strict_parsing=True)) if query else {}
            table = analysis_table(name, dataset, params=params)
        except ValueError as exc:
            raise PlotDataError(f"{label}: {uri} -- bad params: {exc}") from exc
        except PlotDataError as exc:
            raise PlotDataError(f"{label}: {exc}") from exc
        self._db_summary_rows.extend(
            self._analysis_to_summary_rows(label, table.to_dict('records')))

    @staticmethod
    def _analysis_to_summary_rows(label: str, rows: list[dict]) -> list[dict]:
        """A summary-shaped analysis table's rows as summary rows: it already
        spells ``id``/``material``/``size``/``layer``/``force``/``pressure``
        and the ``cof``/``lf``/``nf`` aliases (`periods.summary_identity`),
        so this adds what the table cannot know -- ``dataset_label`` and
        ``file_key`` -- and fills, as :meth:`_dataset_to_summary_rows` does,
        any spelling it lacks."""
        out = []
        for record in rows:
            row = dict(record)
            kind = record.get('load_kind')
            size = record.get('size') or 'unknownx'
            row.update({'dataset_label': label, 'file_key': size, 'size': size,
                        'status': 'complete'})
            row.setdefault('material', record.get('id'))
            row.setdefault('id', row['material'])
            if 'layer' not in row and 'layers' in record:
                row['layer'] = record['layers']
            for key in ('force', 'pressure'):
                if key not in row:
                    row[key] = record.get('load') if kind == key else float('nan')
            for src, dst in SUMMARY_COLUMN_ALIASES.items():
                if src in record and dst not in record:
                    row[dst] = record[src]
            out.append(row)
        return out

    @staticmethod
    def _read_dataset(label: str, name: str) -> tuple[list[dict], dict]:
        """``dataset_rows(name)`` from the AiiDA profile in effect."""
        # pylint: disable=import-outside-toplevel
        try:
            from ..aiida import load_aiida_profile
            from ..aiida.query import dataset_rows
        except ImportError as exc:  # pragma: no cover - aiida extra absent
            raise PlotDataError(
                f"{label}: {DATASET_URI}{name} needs aiida-core: "
                f"pip install -e '.[aiida]'") from exc
        load_aiida_profile()
        try:
            return dataset_rows(name)
        except LookupError as exc:
            raise PlotDataError(f"{label}: {exc}") from exc

    #: plot_config key(s) -> run key, for the filters a dataset's rows are cut
    #: down by before any frame is read. A tuple is read as the plots read
    #: it: the first key set wins (`filter_forces or force`).
    _DATASET_PREFILTERS = {'filter_materials': 'id', ('filter_forces', 'force'): 'force',
                           ('pressures', 'filter_pressures'): 'pressure', 'angle': 'angle',
                           'filter_layer': 'layer', 'filter_speed': 'speed',
                           'filter_tip_radius': 'tip_radius'}
    #: Run keys `plot_all_forces` lifts (plot_timeseries: every load is plotted).
    _ALL_FORCES_LIFTS = frozenset({'force', 'pressure'})

    @classmethod
    def _row_may_match(cls, row: dict, plot_config: dict) -> bool:
        """False only when a filter the plot sets explicitly rules the row out
        -- the same matching as the plots' own run filters, so the rows kept
        are always a superset of the runs the plot selects."""
        lifted = cls._ALL_FORCES_LIFTS if plot_config.get('plot_all_forces') else frozenset()
        for keys, run_key in cls._DATASET_PREFILTERS.items():
            if run_key in lifted:
                continue
            keys = keys if isinstance(keys, tuple) else (keys,)
            value = plot_config.get(keys[0])
            for key in keys[1:]:
                value = value or plot_config.get(key)
            if value is None:
                continue
            have = row.get(run_key)
            if have is None or (isinstance(have, float) and np.isnan(have)):
                return False
            if run_key == 'id':
                wanted = value if isinstance(value, list) else [value]
                if not any(str(v) in str(have) for v in wanted):
                    return False
            elif isinstance(value, list):
                if have not in value:
                    return False
            elif have != value:
                return False
        return True

    def _dataset_runs(self, label: str, file_key: str, tier: str,
                      plot_config: dict | None) -> list[dict]:
        """The runs of an ``aiida://dataset/`` label, read point by point from
        the store: only the rows of size ``file_key`` the plot's explicit
        filters keep, and only their ``tier``. Nothing is written to disk."""
        # pylint: disable=import-outside-toplevel
        from ..aiida import query  # noqa: PLC0415
        from ..data.models import compute_derived_columns  # noqa: PLC0415

        rows = [r for r in self._db_summary_rows
                if r.get('dataset_label') == label and r.get('size') == file_key
                and self._row_may_match(r, plot_config or {})]
        built: list[dict] = []
        for row in rows:
            cache_key = (label, tier, row.get('point_uuid'))
            if cache_key not in self.runs_cache:
                frame = query.point_frames(row['calc_uuid'], row['node_uuid']).get(tier)
                if frame is None:
                    self.runs_cache[cache_key] = []
                    continue
                # float64 throughout, as an export's rows come back from JSON
                frame = frame.reset_index(drop=True).astype(
                    {c: float for c in frame.columns
                     if pd.api.types.is_numeric_dtype(frame[c])})
                if {'lfx', 'lfy', 'nf'} <= set(frame.columns):
                    # what the DataReader adds before it exports
                    frame['lateral_force'], frame['cof'] = compute_derived_columns(
                        frame['lfx'].to_numpy(dtype=float),
                        frame['lfy'].to_numpy(dtype=float),
                        frame['nf'].to_numpy(dtype=float))
                df = self._add_derived_columns(frame)
                self.runs_cache[cache_key] = [{
                    'id': row['id'], 'material': row['material'],
                    'family': row.get('family'), 'model': row.get('model'),
                    'size': row['size'], 'temperature': row.get('temperature'),
                    'potential': row.get('potential'), 'rel_path': row.get('rel_path'),
                    'layer': row.get('layer'), 'speed': row.get('speed'),
                    'angle': row.get('angle'), 'tip_radius': row.get('tip_radius'),
                    'load_kind': row.get('load_kind'), 'load': row.get('load'),
                    'force': row.get('load') if row.get('load_kind') == 'force' else None,
                    'pressure': (row.get('load') if row.get('load_kind') == 'pressure'
                                 else None),
                    'tiers': {}, 'units': {}, 'post': {}, 'df': df,
                }]
            built += self.runs_cache[cache_key]
        if built:
            self.plot_family = built[0]['family'] or self.plot_family
        return built

    def _load_all_metadata(self) -> None:
        """Load and merge metadata from all available data files."""
        for label in self.labels:
            if not self.full_data_files[label]:
                continue
            for file_key in self.full_data_files[label]:
                _, metadata = self._load_full_data(label, file_key)
                if metadata:
                    self._merge_metadata(self.metadata, metadata)

    def _load_full_data(
        self, label: str, file_key: str,
    ) -> tuple[list | None, dict | None]:
        """Load one v2 export and return ``(runs, metadata)``.

        A v1 file has no ``schema`` key at all. It is refused rather than
        read: v1 has no family, no potential, no temperature and no tier, so
        every plot drawn from one is missing whichever sweep arm the old
        nested walk overwrote.
        """
        file_path = self.full_data_files.get(label, {}).get(file_key)
        if not file_path:
            logger.debug("No data file for label '%s', file_key '%s'",
                         label, file_key)
            return None, None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error("Loading data from %s: %s", file_path, e)
            return None, None

        schema = data.get('schema')
        if schema not in (2, 3):
            raise PlotDataError(
                f"{file_path}: export schema {schema!r}, expected 2 or 3. "
                f"Regenerate with "
                f"`FrictionSim2D postprocess read <root> --export`.")
        metadata = dict(data.get('metadata', {}))
        metadata['units'] = data.get('units', {})
        return data.get('runs', []), metadata

    # =====================================================================
    # DATA EXTRACTION
    # =====================================================================

    def _extract_all_runs(self, label: str, file_key: str,
                          tier: str = 'window', plot_config: dict | None = None):
        """Yield one dict per run in the v2 export, for the requested tier.

        The v1 version walked a nested dict and reconstructed parameters from
        key PREFIXES (`f30.0`, `a-15`, `l2`). Every one of those is now a
        typed field on the run record, so there is nothing left to parse --
        and nothing left to mis-parse.

        An ``aiida://dataset/`` label is read from the store instead
        (:meth:`_dataset_runs`); ``plot_config`` lets it read only the points
        the plot's explicit filters keep.
        """
        if label in self._dataset_sources:
            yield from self._dataset_runs(label, file_key, tier, plot_config)
            return
        runs, metadata = self._load_full_data(label, file_key)
        if not runs:
            return
        cache_key = (label, file_key, tier)
        if cache_key in self.runs_cache:
            yield from self.runs_cache[cache_key]
            return

        units = (metadata or {}).get('units', {})
        built: list[dict] = []
        for record in runs:
            if record.get('status', 'complete') not in PLOT_STATUSES:
                continue
            tier_data = record.get('tiers', {}).get(tier)
            if tier_data is None:
                continue
            df = pd.DataFrame(tier_data['data'], columns=tier_data['columns'])
            df.attrs.update(tier_data.get('attrs', {}))
            df = self._add_derived_columns(df)
            # A PES run's load_kind is 'none': neither key gets a value, so it
            # never lands on a force or a pressure axis as a phantom 0.0.
            kind = record['load_kind']
            built.append({
                'id': record['material'],
                'material': record['material'],
                'family': record['family'],
                'model': record['model'],
                'size': record['size'],
                'temperature': record['temperature'],
                'potential': record['potential'],
                'rel_path': record['rel_path'],
                'layer': record['layers'],
                'speed': record['speed'],
                'angle': record['angle'],
                'tip_radius': record['tip_radius'],
                'load_kind': record['load_kind'],
                'load': record['load'],
                'force': record['load'] if kind == 'force' else None,
                'pressure': record['load'] if kind == 'pressure' else None,
                'tiers': record.get('tiers', {}),
                'units': units,
                'post': record.get('post', {}),
                'df': df,
            })
        if built:
            self.plot_family = built[0]['family']
        self.runs_cache[cache_key] = built
        yield from built

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

    def _summary_rows_from_csv(self) -> list[dict]:
        """One row per load point, read from `summary_<size>.csv`.

        The CSV is `summarise_run`'s own output (`DataReader.summary_rows`),
        so `cof` here is the ratio-of-means the store carries. `df.mean()`
        over an element-wise cof column -- what this used
        to do -- is a different number, and putting both on one axis is how
        the same run reported two COFs.
        """
        rows: list[dict] = []
        for label in self.full_data_files:
            for file_key in self.full_data_files[label]:
                path = self.summary_files.get(label, {}).get(file_key)
                if path is None:
                    raise PlotDataError(
                        f"{label}: no summary_{file_key}.csv beside "
                        f"output_full_{file_key}.json. Regenerate with "
                        f"`FrictionSim2D postprocess read <root> --export`.")
                frame = pd.read_csv(path)
                for record in frame.to_dict('records'):
                    status = record.get('status')
                    if (isinstance(status, str)
                            and status not in PLOT_STATUSES):
                        continue
                    row = dict(record)
                    row['dataset_label'] = label
                    row['file_key'] = file_key
                    row['id'] = record['material']
                    kind = record['load_kind']
                    row['force'] = (record['load'] if kind == 'force'
                                    else float('nan'))
                    row['pressure'] = (record['load'] if kind == 'pressure'
                                       else float('nan'))
                    row['layer'] = record['layers']
                    for src, dst in SUMMARY_COLUMN_ALIASES.items():
                        if src in record:
                            row[dst] = record[src]
                    rows.append(row)
        return rows

    def _calculate_summary_statistics(
        self,
        first_n: int | None = None,
        timestep_range: list[int] | None = None,
    ) -> pd.DataFrame:
        """Summary statistics for every run.

        Default: read `summary_<size>.csv`, whose numbers are
        `summarise_run`'s. With `first_n` or `timestep_range` the statistics
        are recomputed over that window from the tier frames -- and `cof` is
        still `mean(lf)/mean(nf)`, never `mean(cof)`, so the two paths differ
        only in which samples they use.
        """
        logger.info("Calculating summary statistics...")
        if first_n is None and timestep_range is None:
            all_records = self._summary_rows_from_csv()
        else:
            all_records = []
            for label in self.full_data_files:
                for file_key in self.full_data_files[label]:
                    for run_data in self._extract_all_runs(label, file_key):
                        run_data = dict(run_data)
                        df = run_data.pop('df')
                        if (timestep_range is not None
                                and len(timestep_range) == 2):
                            df = df.iloc[timestep_range[0]:timestep_range[1]]
                        elif first_n is not None:
                            df = df.iloc[:first_n]
                        stats = df.mean(numeric_only=True).to_dict()
                        if 'lf' in df.columns and 'nf' in df.columns:
                            mean_nf = float(df['nf'].mean())
                            stats['cof'] = (float(df['lf'].mean()) / mean_nf
                                            if mean_nf else float('nan'))
                        run_data.pop('tiers', None)
                        run_data.pop('units', None)
                        all_records.append({
                            'dataset_label': label, 'file_key': file_key,
                            **run_data, **stats})

        # Stored datasets are pre-computed and carry no time series, so a
        # window argument cannot apply to them.
        if self._db_summary_rows and first_n is None and timestep_range is None:
            all_records.extend(self._db_summary_rows)

        summary_df = pd.DataFrame(all_records)
        if not summary_df.empty and 'id' in summary_df.columns:
            summary_df['material_type'] = summary_df['id'].map(
                self.material_type_map)
            if 'size' not in summary_df.columns:
                summary_df['size'] = (
                    summary_df['file_key'].str.extract(r'(\d+x\d+y?)')[0])

        if first_n is None and timestep_range is None:
            self.summary_df_cache = summary_df

        logger.debug("Summary DataFrame shape %s", summary_df.shape)
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
        """Constrained linear regression. Delegates to `plot_derived.linear_fit`.

        Kept as a method because every mixin calls it as one.
        """
        from .plot_derived import linear_fit  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        return linear_fit(x_data, y_data, x_range, constraint)

    # =====================================================================
    # PLOTTING HELPERS
    # =====================================================================

    def _plot_series(
        self, ax, x_data, y_data, label, plot_style, add_fit,
        fit_x_range, std_data=None, *, show_error_bands=True,
        color_idx=0, line_style=None, marker_style=None,
        marker_face='filled', fit_only=False,
        fit_constraint='first_point', custom_color=None,
        extend_fit_to_origin=False, clip_fit_to_data=False,
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
                if clip_fit_to_data and len(x_data):
                    x_max = min(x_max, float(np.max(x_data)))
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
        from .labels import label_for  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        s = self.settings

        # An explicit x_label/y_label still wins -- a paper figure sometimes
        # needs a shorter axis than the honest one. The registry is the
        # FALLBACK, in place of the bare column name this used to print.
        ax.set_xlabel(
            x_label or plot_config.get('x_label')
            or (label_for(x_col, self.plot_family) if x_col else ''),
            fontsize=s["fonts"]["axis_label"],
        )
        ax.set_ylabel(
            y_label or plot_config.get('y_label')
            or (label_for(y_col, self.plot_family) if y_col else ''),
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
            self._place_legend(ax, plot_config)

    def _place_legend(self, ax, plot_config: dict) -> None:
        """Draw the legend: ``legend_order`` first, then ``legend_outside``
        (True = right of the axes, ``'left'``, ``'top'``) or ``legend_location``."""
        s = self.settings
        handles, labels = ax.get_legend_handles_labels()
        legend_order = plot_config.get('legend_order')
        if legend_order:
            rank = {label: i for i, label in enumerate(legend_order)}
            pairs = sorted(zip(handles, labels), key=lambda p: rank.get(p[1], len(rank)))
            handles, labels = [p[0] for p in pairs], [p[1] for p in pairs]
        fontsize = s["fonts"]["legend"]
        legend_outside = plot_config.get('legend_outside', False)
        if legend_outside == 'left':
            # Left of the y-axis label; the tight-bbox export widens the canvas.
            anchor_x = plot_config.get('legend_outside_left_anchor') or -0.16
            ax.legend(handles, labels, loc='center right',
                      bbox_to_anchor=(anchor_x, 0.5), fontsize=fontsize)
        elif legend_outside == 'top':
            if plot_config.get('legend_top_align') == 'right':
                kwargs = {'loc': 'lower right', 'bbox_to_anchor': (1.0, 1.02),
                          'borderaxespad': 0}
            else:
                kwargs = {'loc': 'lower center', 'bbox_to_anchor': (0.5, 1.02)}
            ax.legend(handles, labels, ncol=plot_config.get('legend_ncol') or 5,
                      fontsize=fontsize, **kwargs)
        elif legend_outside:
            ax.legend(handles, labels, loc='center left',
                      bbox_to_anchor=(1.02, 0.5), fontsize=fontsize)
        else:
            legend_loc = plot_config.get('legend_location', s["legend"]["location"])
            ax.legend(handles, labels, loc=legend_loc, fontsize=fontsize)

    @staticmethod
    def _apply_y_limits(ax, plot_config: dict) -> None:
        """Apply explicit y axis limits from plot config."""
        y_limits = plot_config.get('y_limits')
        if y_limits:
            if y_limits[0] is not None:
                ax.set_ylim(bottom=y_limits[0])
            if len(y_limits) > 1 and y_limits[1] is not None:
                ax.set_ylim(top=y_limits[1])

    def _apply_axis_limits(self, ax, plot_config: dict) -> None:
        """Apply explicit x and y axis limits from plot config."""
        self._apply_y_limits(ax, plot_config)

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
        axes_margins_in: dict | None = None,
    ) -> None:
        """Save the plot to file(s) in configured formats.

        ``axes_margins_in`` (with ``tight_bbox``): any of ``left``/``right``/
        ``bottom`` -> inches between that side of the first axes and the image
        edge, so panels line up. It only ever widens the tight box, never clips.
        """
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

        # subplots_adjust on a constrained-layout figure warns and disables
        # the engine, so the fixed margins below would silently undo the
        # setting they were never meant to coexist with.
        if not getattr(self, 'constrained_layout', False):
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
                bbox_inches=(self._margin_bbox(fig, axes_margins_in)
                             if tight_bbox and axes_margins_in
                             else 'tight' if tight_bbox else None),
                pad_inches=0.1 if tight_bbox else 0.0,
            )
            logger.info("Generated plot: %s", output_path)
            self.written.append(output_path)

        plt.close(fig)

    @staticmethod
    def _margin_bbox(fig, axes_margins_in: dict):
        """The padded tight bbox, widened so each named side of the first axes
        sits that many inches from the image edge."""
        from matplotlib.transforms import Bbox  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
        axes_pos = fig.axes[0].get_position()
        fig_w, fig_h = fig.get_size_inches()
        x0, y0, x1 = tight.x0, tight.y0, tight.x1
        if 'left' in axes_margins_in:
            x0 = min(x0, axes_pos.x0 * fig_w - axes_margins_in['left'])
        if 'right' in axes_margins_in:
            x1 = max(x1, axes_pos.x1 * fig_w + axes_margins_in['right'])
        if 'bottom' in axes_margins_in:
            y0 = min(y0, axes_pos.y0 * fig_h - axes_margins_in['bottom'])
        return Bbox.from_extents(x0, y0, x1, tight.y1)

    # =====================================================================
    # SUMMARY PLOT
    # =====================================================================

    def _dataset_series_data(self, series: dict) -> dict:
        """``{x_key: [...], y_key: [...]}`` of the external series whose
        ``dataset`` names a plot label read from the store: the mean of each
        key over that label's summary rows at each load, by increasing load.
        None of the plot's filters apply (the label's rows are the series),
        as in ``provenance_scripts/make_cof_histogram_mlip.py`` (``load_points``).

        Raises:
            PlotDataError: the label has no summary rows, or they lack a key.
        """
        label = series['dataset']
        x_key, y_key = series.get('x_key'), series.get('y_key')
        rows = [r for r in self._db_summary_rows if r.get('dataset_label') == label]
        if not rows:
            raise PlotDataError(
                f"external series {series.get('label', label)!r}: no summary rows for "
                f"the label {label!r}; name a data_dirs label read from the store")
        frame = pd.DataFrame(rows)
        missing = [k for k in (x_key, y_key, 'load') if not k or k not in frame.columns]
        if missing:
            raise PlotDataError(
                f"external series {series.get('label', label)!r}: the rows of {label!r} "
                f"lack {missing}")
        means = frame.groupby('load', sort=True)[[x_key, y_key]].mean()
        return {x_key: means[x_key].astype(float).tolist(),
                y_key: means[y_key].astype(float).tolist()}

    def _external_series_payload(self, series: dict, cache: dict) -> dict | None:
        """The ``{key: values}`` mapping an external series reads its
        ``x_key``/``y_key`` from: its ``dataset``'s per-load means
        (:meth:`_dataset_series_data`), else the ``path`` into its JSON
        ``file`` (read once per file into ``cache``). None, logged, when the
        file or path is missing."""
        if series.get('dataset'):
            return self._dataset_series_data(series)
        file_path = series.get('file')
        if not file_path:
            logger.warning("External series names no 'file' or 'dataset'. Skipping.")
            return None
        resolved_path = self._resolve_aux_file_path(str(file_path))
        cache_key = str(resolved_path)
        if cache_key not in cache:
            try:
                with open(resolved_path, 'r', encoding='utf-8') as f:
                    cache[cache_key] = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logger.warning("Could not load external data from %s: %s", resolved_path, e)
                return None
        data = cache[cache_key]
        for key in series.get('path', []):
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                logger.warning("External data path not found: %s", series.get('path'))
                return None
        return data if isinstance(data, dict) else None

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

        # 'load_range' is an alias of 'force_range'; 'load_column' pins the
        # column it applies to ('force' nN or 'pressure' GPa) instead of the
        # force-then-pressure fallback below.
        force_range = source_config.get('load_range', source_config.get('force_range'))
        load_column = source_config.get('load_column')
        error_metric = source_config.get('error_metric', 'slope_stderr')
        expected_loads = None
        if (force_range and len(force_range) == 2 and load_column
                and source_config.get('require_all_loads', False)):
            in_range = df[load_column].between(force_range[0], force_range[1])
            expected_loads = set(df.loc[in_range, load_column].round(6))

        materials_out: list[str] = []
        values_out: list[float] = []
        errors_out: list[float] = []

        for mat_id in df['id'].unique():
            mat_df = df[df['id'] == mat_id].copy()

            if force_range and len(force_range) == 2 and load_column:
                filtered = mat_df[mat_df[load_column].between(force_range[0], force_range[1])]
                if len(filtered) < 2:
                    continue
                present = set(filtered[load_column].round(6))
                if expected_loads is not None and present != expected_loads:
                    logger.info("%s: skipped, missing loads %s", mat_id,
                                sorted(expected_loads - present))
                    continue
                materials_out.append(mat_id)
                values_out.append(float(filtered[metric].mean()))
                errors_out.append(self._source_error(filtered, load_column, metric,
                                                     error_metric))
                continue

            if force_range and len(force_range) == 2:
                candidate_cols = [
                    col for col in ('force', 'pressure', 'nf')
                    if col in mat_df.columns and mat_df[col].notna().any()
                ]
                if not candidate_cols:
                    logger.warning(
                        "Skipping source %s for material %s: no force/pressure/nf column available",
                        source_config,
                        mat_id,
                    )
                    continue

                # Prefer force when available, but fall back to pressure if force
                # does not provide enough points in the requested range.
                force_col = candidate_cols[0]
                filtered = mat_df[
                    (mat_df[force_col] >= force_range[0])
                    & (mat_df[force_col] <= force_range[1])
                ]
                if (
                    len(filtered) < 2
                    and force_col == 'force'
                    and 'pressure' in candidate_cols
                ):
                    force_col = 'pressure'
                    filtered = mat_df[
                        (mat_df[force_col] >= force_range[0])
                        & (mat_df[force_col] <= force_range[1])
                    ]

                if len(filtered) < 2:
                    continue

                materials_out.append(mat_id)
                values_out.append(float(filtered[metric].mean()))
                errors_out.append(self._source_error(filtered, force_col, metric,
                                                     error_metric))
                continue

            force = source_config.get('force')
            if force is not None and 'force' in mat_df.columns:
                mat_df = mat_df[mat_df['force'] == force]

            if mat_df.empty:
                continue

            materials_out.append(mat_id)
            values_out.append(float(mat_df[metric].mean()))
            errors_out.append(float(mat_df[metric].std()) if len(mat_df) > 1 else 0.0)

        return materials_out, np.array(values_out), np.array(errors_out)

    def _source_error(self, frame: pd.DataFrame, load_col: str, metric: str,
                      error_metric: str) -> float:
        """Error bar of a load-averaged source value: ``sem`` (SD across the
        loads / sqrt(n)), or from the metric-vs-load fit (``slope_stderr``,
        ``rmse``, ``r_squared`` as 1 - R^2); SD when the fit fails."""
        if error_metric == 'sem':
            return float(frame[metric].std() / np.sqrt(len(frame)))
        fit = self._calculate_linear_fit(frame[load_col].values, frame[metric].values)
        if not fit:
            return float(frame[metric].std())
        error_map = {'slope_stderr': fit['slope_stderr'], 'rmse': fit['rmse'],
                     'r_squared': 1 - fit['r_squared']}
        return float(error_map.get(error_metric, fit['slope_stderr']))

    @staticmethod
    def _iterative_outlier_removal(
        x: np.ndarray, y: np.ndarray,
        x_err: np.ndarray | None, y_err: np.ndarray | None,
        materials: list, num_remove: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Iteratively remove points with largest residuals from OLS fit."""
        from scipy import stats as scipy_stats  # pylint: disable=import-outside-toplevel

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
        fig, ax = self.new_figure()

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
