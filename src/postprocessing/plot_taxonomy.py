"""Run-taxonomy plotting mixin: the ML anomaly-mode figures for the article.

The taxonomy layer labels every tip-on-sheet run as clean, a physical anomaly
mode (ploughing, path deviation, ...), unassigned, or an integrity failure
(``t1``). These plots read its output tables and draw them in the same style
as every other ``Plotter`` figure. Four plot types:

``taxonomy_signature_heatmap``
    Median signed z per mode (rows) and channel or feature (columns).
``taxonomy_prevalence``
    Stacked mode fractions (or counts) per normal load, layer or scan angle.
``taxonomy_exemplar``
    Stacked time series of every channel for a mode's highest-severity runs.
``taxonomy_regime_map``
    Material x load grid coloured by each run's mode.

Inputs are files named in the plot config, never the taxonomy code itself, so
the plots survive changes to the taxonomy pipeline as long as the columns do:

``labels_file`` (parquet or csv)
    ``run_id, mode`` plus ``material, layer, force, angle`` (parsed from a
    ``material/l<layer>/f<force>/a<angle>`` run_id when absent). Optional:
    ``severity`` (exemplar selection), ``extrapolated`` (marks loads outside
    the reference fit).
``signatures_file`` (parquet or csv)
    ``run_id`` plus one ``<channel>__<statistic>`` column per signed feature.
``timeseries_file`` (parquet or csv, exemplars only)
    Long format ``run_id, step, <channels>``. Without it, exemplar runs are
    read from ``datasets[0]``/``filter_size`` like the ``timeseries`` plot.

Modes not in the colour map are pooled: ``<template>_<suffix>`` splits into
their template, everything else (e.g. ``emergent_family_3``) into ``other``.
Rows with mode ``missing`` (grid points with no run) are always dropped.
"""
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-few-public-methods,missing-class-docstring

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from .plot_mixin_base import PlotterMixinBase

logger = logging.getLogger(__name__)

# Buckets that are never a physical anomaly mode. ``t1`` is the integrity
# target and is kept visually apart from the physical modes.
TAXONOMY_NON_MODES = ('clean', 'unassigned', 't1', 'missing')

# Named modes take GEM palette colours; the three status buckets are neutral
# (clean, unassigned) or black (integrity failure).
TAXONOMY_MODE_COLORS: dict[str, str] = {
    'clean': '#BDBDBD',
    'path_deviation': '#0072BD',
    'ploughing': '#D95319',
    'marginal_contact_chatter': '#EDB120',
    'sheet_lift': '#7E2F8E',
    'stick': '#77AC30',
    'path_overrun': '#4DBEEE',
    'sheet_lift_recovery': '#A2142F',
    'poor_contact': '#FF007F',
    'delamination': '#00A3A3',
    'buckling': '#1B4F72',
    'other': '#CB845D',
    'unassigned': '#EFEFEF',
    't1': '#000000',
}

TAXONOMY_MODE_LABELS: dict[str, str] = {
    'clean': 'Clean',
    'path_deviation': 'Path deviation',
    'ploughing': 'Ploughing',
    'marginal_contact_chatter': 'Marginal-contact chatter',
    'sheet_lift': 'Sheet lift',
    'stick': 'Stick',
    'path_overrun': 'Path overrun',
    'sheet_lift_recovery': 'Sheet-lift recovery',
    'poor_contact': 'Poor contact',
    'delamination': 'Delamination',
    'buckling': 'Buckling',
    'other': 'Other',
    'unassigned': 'Unassigned',
    't1': 'Integrity failure',
}

TAXONOMY_CHANNEL_LABELS: dict[str, str] = {
    'nf': r'$F_N$',
    'lateral_force': r'$F_L$',
    'lfpar': r'$F_{L,\parallel}$',
    'lfperp': r'$F_{L,\perp}$',
    'tipz': r'$z_{\mathrm{tip}}$',
    'tippar': r'$x_{\mathrm{tip},\parallel}$',
    'tipperp': r'$x_{\mathrm{tip},\perp}$',
    'comz': r'$z_{\mathrm{sheet}}$',
    'compar': r'$x_{\mathrm{sheet},\parallel}$',
    'comperp': r'$x_{\mathrm{sheet},\perp}$',
    'gap': 'Gap',
    'path': 'Path',
    'nfcomz': r'$F_N$-$z_{\mathrm{sheet}}$',
}

TAXONOMY_SERIES_LABELS: dict[str, str] = {
    'nf': r'$F_N$ (nN)',
    'lateral_force': r'$F_L$ (nN)',
    'lf': r'$F_L$ (nN)',
    'lfx': r'$F_x$ (nN)',
    'lfy': r'$F_y$ (nN)',
    'cof': 'COF',
    'tipz': r'$\Delta z_{\mathrm{tip}}$ (Å)',
    'comz': r'$\Delta z_{\mathrm{sheet}}$ (Å)',
    'tipx': r'$\Delta x_{\mathrm{tip}}$ (Å)',
    'tipy': r'$\Delta y_{\mathrm{tip}}$ (Å)',
    'comx': r'$\Delta x_{\mathrm{sheet}}$ (Å)',
    'comy': r'$\Delta y_{\mathrm{sheet}}$ (Å)',
    'tip_sep': r'$z_{\mathrm{tip}} - z_{\mathrm{sheet}}$ (Å)',
}

_GROUP_AXIS_LABELS = {
    'force': 'Normal load (nN)',
    'layer': 'Layers',
    'angle': 'Scan angle (°)',
}

_RUN_ID_RE = re.compile(
    r'^(?P<material>[^/]+)/l(?P<layer>\d+)/f(?P<force>-?[\d.]+)/a(?P<angle>-?[\d.]+)$',
)


def parse_taxonomy_run_id(run_id: str) -> dict | None:
    """Split ``material/l<layer>/f<force>/a<angle>`` into its parts."""
    match = _RUN_ID_RE.match(str(run_id))
    if not match:
        return None
    return {
        'material': match['material'],
        'layer': int(match['layer']),
        'force': float(match['force']),
        'angle': float(match['angle']),
    }


def pool_taxonomy_mode(mode, known_modes) -> str:
    """Map a raw mode label onto a plotted bucket.

    Known modes and status buckets pass through; a template split such as
    ``ploughing_a`` pools into ``ploughing`` (longest template first, so
    ``sheet_lift_recovery_b`` is not taken by ``sheet_lift``); anything else
    pools into ``other``.
    """
    if not isinstance(mode, str):
        return 'other'
    if mode in TAXONOMY_NON_MODES or mode in known_modes:
        return mode
    for base in sorted(known_modes, key=len, reverse=True):
        if base not in TAXONOMY_NON_MODES and mode.startswith(base + '_'):
            return base
    return 'other'


def feature_channel(feature: str) -> str:
    """Channel part of a ``<channel>__<statistic>`` feature name."""
    return feature.split('__', 1)[0]


def aggregate_signature_by_channel(medians: pd.DataFrame) -> pd.DataFrame:
    """Collapse feature columns to channels, keeping the signed value of the
    feature with the largest magnitude in each channel (the channel's
    dominant deviation, not an average that could cancel opposite signs)."""
    channels: dict[str, list[str]] = {}
    for col in medians.columns:
        channels.setdefault(feature_channel(col), []).append(col)

    out = {}
    for channel, cols in channels.items():
        block = medians[cols].to_numpy(dtype=float)
        filled = np.where(np.isnan(block), 0.0, block)
        idx = np.abs(filled).argmax(axis=1)
        out[channel] = filled[np.arange(len(block)), idx]
    return pd.DataFrame(out, index=medians.index)


def format_material_name(name: str) -> str:
    """``h_MoS2`` -> ``h-MoS$_2$`` for publication labels."""
    pretty = re.sub(r'^([bhtp])_', r'\1-', str(name))
    pretty = pretty.replace('_', ' ')
    return re.sub(r'(?<=[A-Za-z])(\d+)', r'$_{\1}$', pretty)


class PlotTaxonomyMixin(PlotterMixinBase):
    # =====================================================================
    # SHARED LOADING AND VOCABULARY
    # =====================================================================

    def _read_taxonomy_table(self, file_path: str | None) -> pd.DataFrame | None:
        """Read a parquet or csv taxonomy table, cached per resolved path."""
        if not file_path:
            return None
        resolved = self._resolve_aux_file_path(str(file_path))
        cache = self.__dict__.setdefault('_taxonomy_table_cache', {})
        key = str(resolved)
        if key not in cache:
            if not Path(resolved).exists():
                logger.error("Taxonomy table not found: %s", resolved)
                return None
            if Path(resolved).suffix.lower() in ('.parquet', '.pq'):
                cache[key] = pd.read_parquet(resolved)
            else:
                cache[key] = pd.read_csv(resolved)
        return cache[key]

    def _taxonomy_mode_colors(self, plot_config: dict) -> dict[str, str]:
        colors = dict(TAXONOMY_MODE_COLORS)
        colors.update(plot_config.get('mode_colors') or {})
        return colors

    @staticmethod
    def _taxonomy_mode_label(mode: str, plot_config: dict) -> str:
        labels = dict(TAXONOMY_MODE_LABELS)
        labels.update(plot_config.get('mode_labels') or {})
        return labels.get(mode, str(mode).replace('_', ' ').capitalize())

    def _taxonomy_labels(self, plot_config: dict) -> pd.DataFrame | None:
        """Load the labels table, fill run coordinates, filter and pool."""
        table = self._read_taxonomy_table(plot_config.get('labels_file'))
        if table is None:
            logger.error("'labels_file' is required for taxonomy plots.")
            return None
        missing = {'run_id', 'mode'} - set(table.columns)
        if missing:
            logger.error("Labels table lacks columns: %s", sorted(missing))
            return None

        df = table[table['mode'] != 'missing'].copy()
        coords = ('material', 'layer', 'force', 'angle')
        if any(c not in df.columns for c in coords):
            parsed = df['run_id'].map(parse_taxonomy_run_id)
            unparsed = int(parsed.isna().sum())
            if unparsed:
                logger.warning(
                    "Dropping %d runs whose run_id is not material/l/f/a.",
                    unparsed,
                )
            df = df[parsed.notna()].copy()
            parsed_df = pd.DataFrame(list(parsed.dropna()), index=df.index)
            for col in coords:
                if col not in df.columns:
                    df[col] = parsed_df[col]

        filters = {
            'layer': plot_config.get('filter_layer'),
            'angle': plot_config.get('angle'),
            'force': plot_config.get('filter_forces'),
            'material': plot_config.get('filter_materials'),
        }
        for col, value in filters.items():
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            df = df[df[col].isin(values)]

        if plot_config.get('pool_unlisted_modes', True):
            known = self._taxonomy_mode_colors(plot_config)
            df['plot_mode'] = df['mode'].map(
                lambda m: pool_taxonomy_mode(m, known),
            )
        else:
            df['plot_mode'] = df['mode']
        return df

    def _taxonomy_mode_order(
        self, present, plot_config: dict,
    ) -> list[str]:
        """Order modes: the configured ``mode_order`` first, then clean,
        named modes by colour-map order, other, unassigned, t1."""
        present = list(dict.fromkeys(present))
        order = list(plot_config.get('mode_order') or [])
        default = ['clean'] + [
            m for m in self._taxonomy_mode_colors(plot_config)
            if m not in TAXONOMY_NON_MODES and m != 'other'
        ] + ['other', 'unassigned', 't1']
        for mode in default + sorted(present):
            if mode not in order:
                order.append(mode)
        return [m for m in order if m in present]

    @staticmethod
    def _taxonomy_extrapolated_groups(df: pd.DataFrame, group_col: str) -> set:
        """Group values where most runs sit outside the reference fit."""
        if 'extrapolated' not in df.columns or df.empty:
            return set()
        frac = (
            df['extrapolated'].astype('float64').fillna(0.0)
            .groupby(df[group_col]).mean()
        )
        return set(frac[frac > 0.5].index)

    @staticmethod
    def _taxonomy_tick_label(value, group_col: str) -> str:
        if group_col in ('force', 'layer', 'angle'):
            return f"{float(value):g}"
        return str(value)

    def _taxonomy_legend(
        self, target, handles, plot_config: dict, *, figure_level=False,
    ) -> None:
        """Place a legend in the Plotter's usual fonts and positions."""
        fontsize = self.settings["fonts"]["legend"]
        if plot_config.get('legend_outside', True):
            kwargs = {'loc': 'center left', 'bbox_to_anchor': (1.02, 0.5)}
            if figure_level:
                kwargs['bbox_to_anchor'] = (1.0, 0.5)
        else:
            kwargs = {
                'loc': plot_config.get(
                    'legend_location', self.settings["legend"]["location"],
                ),
            }
        target.legend(handles=handles, fontsize=fontsize, **kwargs)

    def _taxonomy_mode_patch(self, mode: str, plot_config: dict) -> Patch:
        colors = self._taxonomy_mode_colors(plot_config)
        return Patch(
            facecolor=colors.get(mode, colors['other']),
            edgecolor='#555555', linewidth=0.5,
            hatch='..' if mode == 'unassigned' else None,
            label=self._taxonomy_mode_label(mode, plot_config),
        )

    @staticmethod
    def _taxonomy_extrapolated_handle(style: str) -> Patch:
        if style == 'hatch':
            return Patch(
                facecolor='white', edgecolor='#333333', hatch='///',
                label='Extrapolated reference',
            )
        return Patch(
            facecolor='none', edgecolor='none',
            label='* Extrapolated reference',
        )

    # =====================================================================
    # (a) SIGNED SIGNATURE HEATMAP
    # =====================================================================

    def _generate_taxonomy_signature_heatmap(self, plot_config: dict) -> None:
        """Heatmap of each mode's median signed z per channel or feature."""
        labels = self._taxonomy_labels(plot_config)
        signatures = self._read_taxonomy_table(plot_config.get('signatures_file'))
        if labels is None or signatures is None:
            logger.error("Signature heatmap needs 'labels_file' and 'signatures_file'.")
            return

        mode_col = 'plot_mode' if plot_config.get('pool_modes_in_heatmap', False) else 'mode'
        anomalous = labels[~labels[mode_col].isin(TAXONOMY_NON_MODES)]
        counts = anomalous[mode_col].value_counts()
        modes = plot_config.get('modes')
        if modes:
            modes = [m for m in modes if m in counts.index]
        else:
            min_runs = int(plot_config.get('min_runs', 20))
            modes = list(counts[counts >= min_runs].index)
            modes = modes[:int(plot_config.get('top_n', 15))]
        if not modes:
            logger.warning("No anomaly modes to draw in the signature heatmap.")
            return

        feature_cols = [
            c for c in signatures.columns
            if '__' in c and pd.api.types.is_numeric_dtype(signatures[c])
        ]
        if plot_config.get('features'):
            feature_cols = [c for c in plot_config['features'] if c in feature_cols]
        merged = labels[['run_id', mode_col]].merge(
            signatures[['run_id'] + feature_cols], on='run_id', how='inner',
        )
        medians = (
            merged[merged[mode_col].isin(modes)]
            .groupby(mode_col)[feature_cols].median()
            .reindex(modes)
        )

        column_level = plot_config.get('column_level', 'channel')
        if column_level == 'channel':
            matrix = aggregate_signature_by_channel(medians)
            order = plot_config.get('channels') or list(TAXONOMY_CHANNEL_LABELS)
            ordered = [c for c in order if c in matrix.columns]
            matrix = matrix[ordered + [c for c in matrix.columns if c not in ordered]]
        else:
            matrix = medians

        threshold = float(plot_config.get('annotate_threshold', 1.0))
        if plot_config.get('drop_uninformative', True):
            matrix = matrix.loc[:, (matrix.abs() > threshold).any(axis=0)]
        if matrix.shape[1] == 0:
            logger.warning("Every signature column is below the annotation threshold.")
            return

        values = matrix.to_numpy(dtype=float)
        vlim = plot_config.get('vlim')
        if vlim is None:
            vlim = float(np.nanpercentile(
                np.abs(values), float(plot_config.get('vlim_percentile', 98)),
            )) or 1.0
        clipped = bool(np.nanmax(np.abs(values)) > vlim)

        fonts = self.settings["fonts"]
        fig, ax = plt.subplots(
            figsize=tuple(plot_config.get('figure_size', self.figure_size)),
        )
        image = ax.imshow(
            values, cmap=plot_config.get('colormap', 'RdBu_r'),
            vmin=-vlim, vmax=vlim, aspect='auto',
        )

        column_labels = dict(TAXONOMY_CHANNEL_LABELS)
        column_labels.update(plot_config.get('column_labels') or {})
        ax.set_xticks(range(matrix.shape[1]))
        rotation = int(plot_config.get('x_label_rotation', 45))
        ax.set_xticklabels(
            [column_labels.get(c, c.replace('__', ' ').replace('_', ' '))
             for c in matrix.columns],
            rotation=rotation, ha='right' if rotation else 'center',
            rotation_mode='anchor', fontsize=fonts["tick_label"],
        )
        row_labels = [self._taxonomy_mode_label(m, plot_config) for m in modes]
        if plot_config.get('show_counts', False):
            row_labels = [f"{lbl} ({counts[m]})" for lbl, m in zip(row_labels, modes)]
        ax.set_yticks(range(len(modes)))
        ax.set_yticklabels(row_labels, fontsize=fonts["tick_label"])
        ax.tick_params(axis='both', which='both', length=0)

        ax.set_xticks(np.arange(-0.5, matrix.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, len(modes)), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.grid(which='major', visible=False)

        if plot_config.get('annotate', True):
            decimals = int(plot_config.get('annotate_decimals', 1))
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    v = values[i, j]
                    if np.isfinite(v) and abs(v) > threshold:
                        ax.text(
                            j, i, f"{v:.{decimals}f}", ha='center', va='center',
                            fontsize=fonts["legend"],
                            color='white' if abs(v) > 0.6 * vlim else 'black',
                        )

        cbar = fig.colorbar(image, ax=ax, extend='both' if clipped else 'neither')
        cbar.set_label(
            plot_config.get('colorbar_label', 'Median signed z'),
            fontsize=fonts["axis_label"],
        )
        cbar.ax.tick_params(labelsize=fonts["tick_label"])

        if plot_config.get('x_label'):
            ax.set_xlabel(plot_config['x_label'], fontsize=fonts["axis_label"])
        if plot_config.get('y_label'):
            ax.set_ylabel(plot_config['y_label'], fontsize=fonts["axis_label"])
        if plot_config.get('title'):
            ax.set_title(plot_config['title'], fontsize=fonts["title"])

        self._save_plot(fig, plot_config.get('filename', 'taxonomy_signature_heatmap.png'))

    # =====================================================================
    # (b) MODE PREVALENCE (PHASE DIAGRAM)
    # =====================================================================

    def _generate_taxonomy_prevalence(self, plot_config: dict) -> None:
        """Stacked bars of mode prevalence per load (or layer, or angle).

        ``normalize`` (default) plots fractions with integrity failures
        removed from the denominator, so the physical-mode fractions are not
        diluted by the other target; ``normalize: false`` plots raw counts
        with ``t1`` as its own segment.
        """
        df = self._taxonomy_labels(plot_config)
        if df is None or df.empty:
            logger.warning("No taxonomy labels left for the prevalence plot.")
            return

        group_col = plot_config.get('group_by', 'force')
        normalize = plot_config.get('normalize', True)
        if normalize:
            df = df[df['plot_mode'] != 't1']

        table = df.groupby([group_col, 'plot_mode']).size().unstack(fill_value=0)
        modes = self._taxonomy_mode_order(table.columns, plot_config)
        table = table.reindex(columns=modes, fill_value=0).sort_index()
        if normalize:
            table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        extrapolated = (
            self._taxonomy_extrapolated_groups(df, group_col)
            if group_col == 'force' else set()
        )
        style = plot_config.get('extrapolated_style', 'tick')

        colors = self._taxonomy_mode_colors(plot_config)
        fonts = self.settings["fonts"]
        fig, ax = plt.subplots(
            figsize=tuple(plot_config.get('figure_size', self.figure_size)),
        )
        x_pos = np.arange(len(table.index))
        bar_width = float(plot_config.get('bar_width', 0.8))
        bottom = np.zeros(len(table.index))
        for mode in modes:
            heights = table[mode].to_numpy(dtype=float)
            ax.bar(
                x_pos, heights, bottom=bottom, width=bar_width,
                color=colors.get(mode, colors['other']),
                edgecolor='white', linewidth=0.8, zorder=2,
            )
            if mode == 'unassigned':
                ax.bar(
                    x_pos, heights, bottom=bottom, width=bar_width,
                    facecolor='none', edgecolor=(0.3, 0.3, 0.3, 0.6),
                    hatch='..', linewidth=0, zorder=3,
                )
            bottom += heights

        if style == 'hatch':
            for i, group in enumerate(table.index):
                if group in extrapolated:
                    ax.bar(
                        x_pos[i], bottom[i], width=bar_width, facecolor='none',
                        edgecolor=(0.2, 0.2, 0.2, 0.5), hatch='///',
                        linewidth=0, zorder=4,
                    )

        tick_labels = []
        for group in table.index:
            label = self._taxonomy_tick_label(group, group_col)
            if style == 'tick' and group in extrapolated:
                label += '*'
            tick_labels.append(label)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tick_labels, fontsize=fonts["tick_label"])
        ax.set_xlim(-0.5, len(table.index) - 0.5)
        if normalize:
            ax.set_ylim(0, 1)
        ax.tick_params(axis='both', which='major', labelsize=fonts["tick_label"])
        ax.set_xlabel(
            plot_config.get('x_label', _GROUP_AXIS_LABELS.get(group_col, group_col)),
            fontsize=fonts["axis_label"],
        )
        ax.set_ylabel(
            plot_config.get(
                'y_label', 'Fraction of runs' if normalize else 'Number of runs',
            ),
            fontsize=fonts["axis_label"],
        )
        self._apply_axis_limits(ax, plot_config)
        if plot_config.get('title'):
            ax.set_title(plot_config['title'], fontsize=fonts["title"])

        if plot_config.get('show_legend', True):
            handles = [
                self._taxonomy_mode_patch(m, plot_config) for m in reversed(modes)
            ]
            if extrapolated and style in ('tick', 'hatch'):
                handles.append(self._taxonomy_extrapolated_handle(style))
            self._taxonomy_legend(ax, handles, plot_config)

        self._save_plot(fig, plot_config.get('filename', 'taxonomy_prevalence.png'))

    # =====================================================================
    # (c) EXEMPLAR TIME SERIES
    # =====================================================================

    def _select_taxonomy_exemplars(self, plot_config: dict) -> pd.DataFrame | None:
        """Explicit ``run_ids``, or the mode's top-``n_runs`` by |severity|."""
        labels = self._taxonomy_labels(plot_config)
        if labels is None:
            return None

        run_ids = plot_config.get('run_ids')
        if run_ids:
            chosen = labels.set_index('run_id').reindex(run_ids)
            absent = chosen.index[chosen['mode'].isna()].tolist()
            if absent:
                logger.warning("Exemplar run_ids not in labels: %s", absent)
            return chosen.dropna(subset=['mode']).reset_index()

        mode = plot_config.get('mode')
        if not mode:
            logger.error("Exemplar plot needs 'mode' or 'run_ids'.")
            return None
        members = labels[(labels['mode'] == mode) | (labels['plot_mode'] == mode)]
        if members.empty:
            logger.warning("No runs labelled '%s'.", mode)
            return None
        if 'severity' in members.columns:
            members = members.assign(
                _rank=members['severity'].abs(),
            ).sort_values('_rank', ascending=False, na_position='last')
        return members.head(int(plot_config.get('n_runs', 3))).reset_index(drop=True)

    def _load_exemplar_series(
        self, exemplars: pd.DataFrame, plot_config: dict,
    ) -> tuple[dict[str, pd.DataFrame], np.ndarray | None]:
        """Return ``{run_id: frame}`` and the time axis (None = sample index)."""
        series: dict[str, pd.DataFrame] = {}
        timeseries_file = plot_config.get('timeseries_file')
        if timeseries_file:
            long_df = self._read_taxonomy_table(timeseries_file)
            if long_df is None:
                return series, None
            subset = long_df[long_df['run_id'].isin(exemplars['run_id'])]
            for run_id, group in subset.groupby('run_id'):
                frame = group.sort_values('step').reset_index(drop=True)
                series[run_id] = self._add_derived_columns(frame)
            interval = plot_config.get('sample_interval')
            if interval is None or not series:
                return series, None
            n_samples = max(len(f) for f in series.values())
            return series, np.arange(n_samples) * float(interval)

        datasets = plot_config.get('datasets') or self.labels[:1]
        file_key = plot_config.get('filter_size')
        if not datasets or not file_key:
            logger.error(
                "Exemplar plot needs 'timeseries_file' or 'datasets' + 'filter_size'.",
            )
            return series, None

        wanted = {
            (row['material'], int(row['layer']), float(row['force']),
             float(row['angle'])): row['run_id']
            for _, row in exemplars.iterrows()
        }
        label = datasets[0]
        for run in self._extract_all_runs(label, file_key):
            key = (
                run.get('id'), int(run.get('layer', 1)),
                float(run.get('force', np.nan)), float(run.get('angle', 0.0)),
            )
            if key in wanted:
                series[wanted[key]] = run['df']
                if len(series) == len(wanted):
                    break

        time_series = self.metadata.get('time_series')
        if time_series is None or not series:
            return series, None
        return series, np.asarray(time_series, dtype=float)

    def _generate_taxonomy_exemplar(self, plot_config: dict) -> None:
        """One panel per channel, the selected runs overlaid, shared time axis."""
        exemplars = self._select_taxonomy_exemplars(plot_config)
        if exemplars is None or exemplars.empty:
            return

        channels = list(plot_config.get(
            'channels', ['nf', 'lateral_force', 'tipz', 'comz'],
        ))
        series, time_axis = self._load_exemplar_series(exemplars, plot_config)
        if not series:
            logger.warning("No time series found for the exemplar runs.")
            return

        subtract_initial = set(plot_config.get('subtract_initial', ['tipz', 'comz']))
        time_scale = float(plot_config.get('time_scale', 0.001))
        shift_time = plot_config.get('shift_time_to_zero', True)
        series_labels = dict(TAXONOMY_SERIES_LABELS)
        series_labels.update(plot_config.get('channel_labels') or {})

        base_width = float(self.settings["lines"]["width"])
        line_width = float(plot_config.get('line_width', base_width * 1.25))
        n_channels = len(channels)
        fig_size = plot_config.get('figure_size', [self.figure_size[0], 3.0 * n_channels])
        fig, axes = plt.subplots(n_channels, 1, figsize=tuple(fig_size), sharex=True)
        axes = np.atleast_1d(axes)

        panel_config = {
            k: v for k, v in plot_config.items()
            if k not in ('title', 'x_label', 'y_label')
        }
        for color_idx, (_, row) in enumerate(exemplars.iterrows()):
            frame = series.get(row['run_id'])
            if frame is None:
                logger.warning("No time series for exemplar %s.", row['run_id'])
                continue
            n_points = len(frame)
            if time_axis is not None and len(time_axis) >= n_points:
                x_values = time_axis[:n_points].copy()
                if shift_time:
                    x_values = x_values - x_values[0]
                x_values = x_values * time_scale
            else:
                x_values = np.arange(n_points, dtype=float)

            run_label = f"{row['material']}, F={float(row['force']):.1f}nN"
            if float(row['angle']) != 0.0:
                run_label += f", {float(row['angle']):g}°"
            color = self._get_palette_color(color_idx)
            for panel, (channel, ax) in enumerate(zip(channels, axes)):
                if channel not in frame.columns:
                    logger.warning("Channel '%s' missing for %s.", channel, row['run_id'])
                    continue
                y_values = frame[channel].to_numpy(dtype=float)
                if channel in subtract_initial and len(y_values):
                    y_values = y_values - y_values[0]
                ax.plot(
                    x_values, y_values, color=color, linewidth=line_width,
                    alpha=float(plot_config.get('line_alpha', 0.95)),
                    label=run_label if panel == 0 else '_nolegend_',
                )

        x_label = plot_config.get(
            'x_label', 'Time (ps)' if time_axis is not None else 'Sample',
        )
        for panel, (channel, ax) in enumerate(zip(channels, axes)):
            self._finalize_plot(
                ax, panel_config,
                x_label=x_label if panel == n_channels - 1 else ' ',
                y_label=series_labels.get(channel, channel),
            )
            if panel != n_channels - 1:
                ax.set_xlabel('')
        # Every panel is full of data, so by default the run legend sits
        # above the stack instead of inside the first panel.
        if plot_config.get('legend_position', 'top') == 'top':
            handles, run_labels = axes[0].get_legend_handles_labels()
            if axes[0].get_legend() is not None:
                axes[0].get_legend().remove()
            if handles:
                axes[0].legend(
                    handles, run_labels, loc='lower center',
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=int(plot_config.get('legend_columns', min(len(handles), 2))),
                    fontsize=self.settings["fonts"]["legend"],
                )
        if plot_config.get('title'):
            fig.suptitle(plot_config['title'], fontsize=self.settings["fonts"]["title"])
        fig.align_ylabels(axes)

        filename = plot_config.get(
            'filename', f"taxonomy_exemplar_{plot_config.get('mode', 'runs')}.png",
        )
        self._save_plot(fig, filename)

    # =====================================================================
    # (d) MATERIAL x LOAD REGIME MAP
    # =====================================================================

    def _generate_taxonomy_regime_map(self, plot_config: dict) -> None:
        """Material x load grid coloured by mode, for one layer and angle."""
        config = dict(plot_config)
        config.setdefault('angle', 0.0)
        df = self._taxonomy_labels(config)
        if df is None or df.empty:
            logger.warning("No taxonomy labels left for the regime map.")
            return
        if df['layer'].nunique() > 1:
            logger.error(
                "Regime map spans layers %s; set 'filter_layer'.",
                sorted(df['layer'].unique()),
            )
            return

        grid = df.pivot_table(
            index='material', columns='force', values='plot_mode', aggfunc='first',
        )
        selection = config.get('material_selection', 'all')
        if selection == 'any_mode':
            # A material stays if any run carries a mode or an integrity
            # failure: t1 is the other target, but still worth seeing.
            has_mode = grid.apply(
                lambda row: row.dropna().map(
                    lambda m: m not in ('clean', 'unassigned'),
                ).any(),
                axis=1,
            )
            grid = grid[has_mode]
        if grid.empty:
            logger.warning("No materials selected for the regime map.")
            return

        sort_by = config.get('sort_materials', 'type')
        materials = list(grid.index)
        if config.get('filter_materials') and sort_by == 'config':
            materials = [m for m in config['filter_materials'] if m in grid.index]
        elif sort_by == 'mode_count':
            n_modes = grid.apply(
                lambda row: row.dropna().map(
                    lambda m: m not in TAXONOMY_NON_MODES,
                ).sum(),
                axis=1,
            )
            materials = sorted(materials, key=lambda m: (-n_modes[m], m))
        elif sort_by == 'type':
            def type_key(material):
                match = re.match(r'^([bhtp])_(.+)$', material)
                return (match[1], match[2]) if match else ('z', material)
            materials = sorted(materials, key=type_key)
        else:
            materials = sorted(materials)
        loads = sorted(grid.columns)
        grid = grid.reindex(index=materials, columns=loads)

        categories = self._taxonomy_mode_order(
            pd.unique(grid.to_numpy().ravel()[pd.notna(grid.to_numpy().ravel())]),
            config,
        )
        colors = self._taxonomy_mode_colors(config)
        cmap = ListedColormap([colors.get(c, colors['other']) for c in categories])
        cmap.set_bad('white')
        code_of = {c: i for i, c in enumerate(categories)}
        codes = grid.apply(lambda col: col.map(code_of)).to_numpy(dtype=float)

        extrapolated = self._taxonomy_extrapolated_groups(df, 'force')
        style = config.get('extrapolated_style', 'tick')

        n_columns = max(1, int(config.get('n_columns', 2)))
        rows_per_column = int(np.ceil(len(materials) / n_columns))
        fonts = self.settings["fonts"]
        fig_size = config.get(
            'figure_size',
            [self.figure_size[0], max(self.figure_size[1], 0.3 * rows_per_column)],
        )
        fig, axes = plt.subplots(1, n_columns, figsize=tuple(fig_size), squeeze=False)
        axes = axes[0]

        row_font = config.get('material_label_fontsize')
        if row_font is None:
            row_height_pt = fig_size[1] * 72.0 * 0.85 / max(rows_per_column, 1)
            row_font = min(fonts["tick_label"], max(4.0, 0.75 * row_height_pt))
        pretty = config.get('pretty_material_names', False)
        tick_labels = []
        for load in loads:
            label = f"{float(load):g}"
            if style == 'tick' and load in extrapolated:
                label += '*'
            tick_labels.append(label)

        unassigned_code = code_of.get('unassigned')
        for col_idx, ax in enumerate(axes):
            start = col_idx * rows_per_column
            block = codes[start:start + rows_per_column]
            names = materials[start:start + rows_per_column]
            if not names:
                ax.set_visible(False)
                continue
            ax.pcolormesh(
                np.ma.masked_invalid(block), cmap=cmap,
                vmin=-0.5, vmax=len(categories) - 0.5,
                edgecolors='white', linewidth=0.8,
            )
            if unassigned_code is not None:
                for r, c in zip(*np.where(block == unassigned_code)):
                    ax.add_patch(Rectangle(
                        (c, r), 1, 1, fill=False, hatch='..',
                        edgecolor=(0.3, 0.3, 0.3, 0.6), linewidth=0,
                    ))
            if style == 'hatch':
                for c, load in enumerate(loads):
                    if load in extrapolated:
                        ax.add_patch(Rectangle(
                            (c, 0), 1, len(names), fill=False, hatch='///',
                            edgecolor=(0.2, 0.2, 0.2, 0.5), linewidth=0,
                        ))
            ax.set_xlim(0, len(loads))
            ax.set_ylim(len(names), 0)
            ax.set_xticks(np.arange(len(loads)) + 0.5)
            ax.set_xticklabels(
                tick_labels, rotation=int(config.get('x_label_rotation', 90)),
                fontsize=fonts["tick_label"],
            )
            ax.set_yticks(np.arange(len(names)) + 0.5)
            ax.set_yticklabels(
                [format_material_name(n) if pretty else n for n in names],
                fontsize=row_font,
            )
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_xlabel(
                config.get('x_label', 'Normal load (nN)'),
                fontsize=fonts["axis_label"],
            )

        if config.get('title'):
            fig.suptitle(config['title'], fontsize=fonts["title"])
        if config.get('show_legend', True):
            handles = [self._taxonomy_mode_patch(c, config) for c in categories]
            if np.isnan(codes).any():
                handles.append(Patch(
                    facecolor='white', edgecolor='#555555', linewidth=0.5,
                    label='No run',
                ))
            if extrapolated and style in ('tick', 'hatch'):
                handles.append(self._taxonomy_extrapolated_handle(style))
            self._taxonomy_legend(fig, handles, config, figure_level=True)

        # _save_plot only sets the outer margins, so the gap that keeps one
        # column's material names off the previous grid survives it.
        fig.subplots_adjust(wspace=float(config.get('column_spacing', 0.8)))
        self._save_plot(
            fig, config.get('filename', 'taxonomy_regime_map.png'), tight_bbox=True,
        )
