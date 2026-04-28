"""Summary plotting mixin: summary lines, external series, and COF histogram."""
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-boolean-expressions,too-few-public-methods
# pylint: disable=missing-class-docstring,trailing-newlines

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .plot_mixin_base import PlotterMixinBase

logger = logging.getLogger(__name__)


class PlotSummaryMixin(PlotterMixinBase):
    def _plot_external_series(
        self,
        ax,
        external_series: list,
        plot_config: dict,
        fit_x_range_effective,
        color_idx: int,
    ) -> tuple[list[float], int]:
        """Plot external data series on an axis.

        Returns:
            ``(y_values_collected, next_color_idx)``
        """
        fit_only = plot_config.get('fit_only', False)
        fit_constraint = plot_config.get('fit_constraint', 'first_point')
        extend_fit_to_origin = plot_config.get('extend_fit_to_origin', False)
        shift_to_origin = plot_config.get(
            'shift_external_series_to_origin',
            plot_config.get('shift_series_to_origin', False),
        )
        slope_fit_constraint = plot_config.get('slope_fit_constraint', 'none')
        slope_label = plot_config.get('slope_label', 'COF')
        slope_decimals = plot_config.get('slope_decimals', 5)
        append_slope_to_legend = plot_config.get('append_slope_to_legend', False)
        external_y_values: list[float] = []
        data_cache: dict[str, dict] = {}

        for series in external_series:
            file_path = series.get('file')
            if not file_path:
                logger.warning("External series missing 'file' path. Skipping.")
                continue

            resolved_path = self._resolve_aux_file_path(str(file_path))
            cache_key = str(resolved_path)
            if cache_key not in data_cache:
                try:
                    with open(resolved_path, 'r', encoding='utf-8') as f:
                        data_cache[cache_key] = json.load(f)
                except (IOError, json.JSONDecodeError) as e:
                    logger.warning(
                        "Could not load external data from %s: %s",
                        resolved_path,
                        e,
                    )
                    continue

            data = data_cache[cache_key]
            for key in series.get('path', []):
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    logger.warning(
                        "External data path not found: %s",
                        series.get('path'),
                    )
                    data = None
                    break

            if not isinstance(data, dict):
                continue

            x_key = series.get('x_key')
            y_key = series.get('y_key')
            if not x_key or not y_key:
                logger.warning(
                    "External series missing 'x_key' or 'y_key'. Skipping.",
                )
                continue

            x_data = data.get(x_key)
            y_data = data.get(y_key)
            if not x_data or not y_data:
                logger.warning(
                    "External series data missing for keys: %s, %s",
                    x_key,
                    y_key,
                )
                continue

            x_transform = series.get('x_transform')
            y_transform = series.get('y_transform')
            if x_transform:
                transformed_x = self._apply_series_transform(str(x_transform), x_data)
                if transformed_x is not None:
                    x_data = transformed_x
                else:
                    logger.warning(
                        "Unsupported x_transform expression '%s'. Ignoring.",
                        x_transform,
                    )
            if y_transform:
                transformed_y = self._apply_series_transform(
                    str(y_transform), y_data, x_values=data.get(x_key),
                )
                if transformed_y is not None:
                    y_data = transformed_y
                else:
                    logger.warning(
                        "Unsupported y_transform expression '%s'. Ignoring.",
                        y_transform,
                    )

            try:
                external_y_values.extend([
                    float(v)
                    for v in y_data
                    if v is not None and not np.isnan(v)
                ])
            except (TypeError, ValueError):
                pass

            label = series.get('label', 'External')
            ext_plot_style = series.get('plot_style', 'line')
            line_style = series.get('line_style', '--')
            marker_style = series.get('marker_style')
            if marker_style is None:
                label_l = str(label).lower()
                if 'sw' in label_l and 'hu' in label_l:
                    marker_style = '^'
                elif 'reaxff' in label_l and 'serpini' in label_l:
                    marker_style = 's'
                elif 'exp.' in label_l and 'serpini' in label_l:
                    marker_style = 'D'
                else:
                    marker_style = 'o'
            marker_face = str(series.get('marker_face', 'none')).lower()
            color = self._get_consistent_color(series_name=label, fallback_index=color_idx)
            marker_facecolor = color if marker_face == 'filled' else 'none'

            x_arr = np.array(x_data, dtype=float)
            y_arr = np.array(y_data, dtype=float)

            display_label = label
            if len(x_arr) >= 2:
                slope_fit_params = self._calculate_linear_fit(
                    x_arr, y_arr, fit_x_range_effective,
                    constraint=slope_fit_constraint,
                )
                if slope_fit_params is not None:
                    logger.info(
                        "  [%s] COF (slope) = %.6f, intercept = %.4f, R² = %.4f",
                        label,
                        slope_fit_params['slope'],
                        slope_fit_params['intercept'],
                        slope_fit_params['r_squared'],
                    )
                    if append_slope_to_legend:
                        display_label = (
                            f"{label} {slope_label} = "
                            f"{slope_fit_params['slope']:.{slope_decimals}f}"
                        )

            if shift_to_origin and len(x_arr) >= 2:
                fit_params_shift = self._calculate_linear_fit(
                    x_arr, y_arr, fit_x_range_effective, constraint='none',
                )
                if fit_params_shift:
                    y_arr = y_arr - fit_params_shift['intercept']

            if ext_plot_style == 'scatter':
                ax.scatter(
                    x_arr, y_arr, label=display_label,
                    s=self.settings["markers"]["size"]**2, color=color,
                    marker=marker_style, facecolors=marker_facecolor, edgecolors=color,
                )
            else:
                ax.plot(
                    x_arr, y_arr, marker=marker_style,
                    linestyle='None' if fit_only else line_style,
                    linewidth=self.settings["lines"]["width"],
                    markersize=self.settings["markers"]["size"],
                    markerfacecolor=marker_facecolor,
                    markeredgecolor=color,
                    label=display_label,
                    color=color,
                )

                if fit_only:
                    fit_params_ext = self._calculate_linear_fit(
                        x_arr, y_arr, fit_x_range_effective,
                        constraint=fit_constraint,
                    )
                    if fit_params_ext:
                        x_min = (
                            0 if extend_fit_to_origin
                            else (
                                fit_x_range_effective[0]
                                if fit_x_range_effective else x_arr.min()
                            )
                        )
                        x_max = (
                            fit_x_range_effective[1]
                            if fit_x_range_effective else x_arr.max()
                        )
                        x_fit = np.linspace(x_min, x_max, 100)
                        y_fit = (
                            fit_params_ext['slope'] * x_fit
                            + fit_params_ext['intercept']
                        )
                        ax.plot(
                            x_fit, y_fit, linestyle='--',
                            linewidth=self.settings["lines"]["width"],
                            alpha=self.settings["lines"]["fit_alpha"],
                            color=color,
                        )

            color_idx += 1

        return external_y_values, color_idx

    def _finalize_summary_plot(
        self,
        fig,
        ax,
        zoom_df: pd.DataFrame,
        external_y_values: list[float],
        x_col: str,
        y_col: str,
        plot_config: dict,
    ) -> None:
        """Apply auto y-limits, axis limits, finalize, optional legend PNG, and save."""
        if not zoom_df.empty:
            min_y, max_y = zoom_df[y_col].min(), zoom_df[y_col].max()
            if external_y_values:
                min_y = min(min_y, *external_y_values)
                max_y = max(max_y, *external_y_values)
            padding = max((max_y - min_y) * 0.1, 0.1)
            ax.set_ylim(min_y - padding, max_y + padding)
        elif external_y_values:
            min_y = min(external_y_values)
            max_y = max(external_y_values)
            padding = max((max_y - min_y) * 0.1, 0.1)
            ax.set_ylim(min_y - padding, max_y + padding)

        self._apply_axis_limits(ax, plot_config)
        self._finalize_plot(ax, plot_config, x_col, y_col)

        if plot_config.get('legend_separate_png', False):
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                legend_fig = plt.figure(
                    figsize=plot_config.get('legend_figure_size', [8, 6]),
                )
                legend_ax = legend_fig.add_subplot(111)
                legend_ax.axis('off')
                legend_ax.legend(
                    handles,
                    labels,
                    loc='center',
                    frameon=False,
                    fontsize=self.settings["fonts"]["legend"],
                    ncol=plot_config.get('legend_ncol', 1),
                )
                legend_filename = plot_config.get('legend_filename')
                if not legend_filename:
                    base_name = Path(plot_config.get('filename', 'legend.png')).stem
                    legend_filename = f"{base_name}_legend.png"
                legend_path = self.output_dir / legend_filename
                legend_fig.savefig(
                    legend_path,
                    dpi=self.settings["figure"]["dpi"],
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0.05,
                )
                plt.close(legend_fig)

        self._save_plot(fig, plot_config.get('filename'))

    def _generate_summary_plot(self, plot_config: dict) -> None:
        """Generate a summary plot (main plot type)."""
        first_n = plot_config.get('average_first_n_timesteps')
        timestep_range = plot_config.get('average_timestep_range')
        if first_n is not None:
            logger.info("Using first %d timesteps for averaging", first_n)
        if timestep_range is not None:
            logger.info("Using timestep range %s for averaging", timestep_range)

        summary_df = self._get_summary_data_df(
            first_n=first_n,
            timestep_range=timestep_range,
        )
        if summary_df.empty:
            logger.warning("Summary data is empty. Skipping plot.")
            return

        title = plot_config.get('title', '(no title)')
        logger.debug("Generating plot: %s", title)

        df = summary_df.copy()
        datasets = plot_config.get('datasets')
        if datasets:
            df = df[df['dataset_label'].isin(datasets)]
            logger.debug("Dataset filter: %d -> %d", len(summary_df), len(df))

        plot_by = plot_config.get('plot_by', 'id')
        plot_style = plot_config.get('plot_style', 'line')
        x_col = plot_config['x_axis']
        y_col = plot_config['y_axis']
        add_fit = plot_config.get('add_linear_fit', False)
        fit_x_range = plot_config.get('fit_x_range')
        x_limits = plot_config.get('x_limits')
        use_visible_x_for_fit = plot_config.get('use_visible_xrange_for_fit', True)
        fit_x_range_effective = fit_x_range
        if (
            fit_x_range_effective is None
            and use_visible_x_for_fit and x_limits
            and len(x_limits) >= 2
            and x_limits[0] is not None
            and x_limits[1] is not None
        ):
            fit_x_range_effective = [x_limits[0], x_limits[1]]
        show_dataset = plot_config.get('show_dataset_in_legend', False)
        show_error_bands = plot_config.get('show_error_bands', True)
        external_series = plot_config.get('external_series', [])
        fit_only = plot_config.get('fit_only', False)
        shift_to_origin = plot_config.get('shift_series_to_origin', False)
        fit_constraint = plot_config.get('fit_constraint', 'first_point')
        extend_fit_to_origin = plot_config.get('extend_fit_to_origin', False)
        append_slope_to_legend = plot_config.get('append_slope_to_legend', False)
        slope_label = plot_config.get('slope_label', 'COF')
        slope_decimals = plot_config.get('slope_decimals', 5)
        slope_fit_constraint = plot_config.get('slope_fit_constraint', 'none')

        marker_by_potential = plot_config.get('marker_by_potential_type', False)
        potential_markers = plot_config.get('potential_markers', {
            'reaxff': 's',
            'rebomos': 'o',
            'sw': '^',
        })

        def classify_potential(name: str) -> str:
            lowered = str(name).lower()
            if 'reaxff' in lowered:
                return 'reaxff'
            if 'rebomos' in lowered:
                return 'rebomos'
            return 'sw'

        filters = self._apply_default_filters(df, plot_config, x_col)
        df = self._apply_filters(df, filters)
        df = self._apply_range_filters(df, plot_config)
        df = self._apply_material_filter(df, plot_config, plot_by)

        y_col = self._resolve_lf_axis(df, y_col, plot_config)

        external_only = False

        if df.empty:
            if not external_series:
                logger.warning("No data left after filtering. Skipping plot.")
                return
            external_only = True
            logger.info("No internal data after filtering; plotting external series only.")

        if not external_only and y_col not in df.columns:
            logger.error(
                "y-axis column '%s' not found in data. Skipping.", y_col,
            )
            return

        if not external_only:
            df = self._remove_outliers(df, x_col, y_col)
            if df.empty:
                if not external_series:
                    logger.warning("No data left after outlier removal. Skipping.")
                    return
                external_only = True
                logger.info(
                    "No internal data after outlier removal; "
                    "plotting external series only.",
                )

        plot_fig_size = tuple(plot_config.get('figure_size', list(self.figure_size)))
        fig, ax = plt.subplots(figsize=plot_fig_size)

        if plot_by == 'dataset_label':
            group_col = 'dataset_label'
            aggregate = True
        elif plot_by == 'material_type':
            group_col = 'material_type'
            aggregate = True
        elif plot_by == 'pressure':
            group_col = 'pressure'
            aggregate = False
        elif plot_by == 'id_angle':
            group_col = ['id', 'angle']
            aggregate = False
        else:
            group_col = 'id'
            aggregate = False

        color_idx = 0
        label_prefix = plot_config.get('label_prefix', '')
        label_suffix = plot_config.get('label_suffix', '')

        for group_name, group in (df.groupby(group_col) if not external_only else []):
            if plot_by == 'material_type':
                group_name_str = str(group_name)
                label = self.type_display_names.get(group_name_str, group_name_str)
            elif plot_by == 'pressure':
                label = f"{group_name:g} GPa"
            elif plot_by == 'id_angle':
                if isinstance(group_name, tuple) and len(group_name) == 2:
                    try:
                        angle_value = int(float(str(group_name[1])))
                        label = f"{group_name[0]}_{angle_value}"
                    except (TypeError, ValueError):
                        label = f"{group_name[0]}_{group_name[1]}"
                else:
                    label = str(group_name)
            elif show_dataset and 'dataset_label' in group.columns:
                dataset = group['dataset_label'].iloc[0]
                label = f"{group_name} ({self._display_dataset_label(dataset)})"
            else:
                if plot_by == 'dataset_label':
                    label = self._display_dataset_label(str(group_name))
                else:
                    label = str(group_name)

            if label_prefix:
                label = f"{label_prefix}{label}"
            if label_suffix:
                label = f"{label}{label_suffix}"

            marker_style = None
            if marker_by_potential:
                marker_style = potential_markers.get(
                    classify_potential(label),
                    self.settings["markers"]["style"],
                )

            group_x = None
            group_y = None
            group_std = None

            if aggregate:
                plot_data = (
                    group.groupby(x_col)[y_col]
                    .agg(['mean', 'std']).reset_index()
                )
                plot_data = plot_data.sort_values(by=x_col)
                group_x = plot_data[x_col].to_numpy()
                group_y = plot_data['mean'].to_numpy()
                group_std = plot_data['std'].to_numpy()
            else:
                group = group.sort_values(by=x_col)
                group_x = group[x_col].to_numpy()
                group_y = group[y_col].to_numpy()

            display_label = label
            if len(group_x) >= 2:
                slope_fit_params = self._calculate_linear_fit(
                    group_x,
                    group_y,
                    fit_x_range_effective,
                    constraint=slope_fit_constraint,
                )
                if slope_fit_params is not None:
                    logger.info(
                        "  [%s] COF (slope) = %.6f, intercept = %.4f nN, R² = %.4f",
                        label,
                        slope_fit_params['slope'],
                        slope_fit_params['intercept'],
                        slope_fit_params['r_squared'],
                    )
                    if append_slope_to_legend:
                        display_label = (
                            f"{label} {slope_label} = "
                            f"{slope_fit_params['slope']:.{slope_decimals}f}"
                        )

            if shift_to_origin and len(group_x) >= 2:
                fit_params_shift = self._calculate_linear_fit(
                    group_x,
                    group_y,
                    fit_x_range_effective,
                    constraint='none',
                )
                if fit_params_shift:
                    group_y = group_y - fit_params_shift['intercept']

            use_dataset_colors = self._should_use_dataset_colors(
                plot_config,
                default=(plot_by == 'dataset_label'),
            )

            series_color = None
            if (
                use_dataset_colors
                and 'dataset_label' in group.columns
                and group['dataset_label'].nunique() == 1
            ):
                series_color = self._get_consistent_color(
                    dataset_label=group['dataset_label'].iloc[0],
                    fallback_index=color_idx,
                )

            self._plot_series(
                ax,
                group_x,
                group_y,
                display_label,
                plot_style,
                add_fit or fit_only,
                fit_x_range_effective,
                group_std,
                show_error_bands=show_error_bands,
                color_idx=color_idx,
                line_style='-' if fit_only else None,
                marker_style=marker_style,
                marker_face='filled',
                fit_only=fit_only,
                fit_constraint=fit_constraint,
                custom_color=series_color,
                extend_fit_to_origin=extend_fit_to_origin,
            )

            logger.debug("Plotted %s (%d points)", group_name, len(group))
            color_idx += 1

        external_y_values, _ = self._plot_external_series(
            ax, external_series, plot_config, fit_x_range_effective, color_idx,
        )

        if y_col == 'cof' and x_col == 'force':
            zoom_df = df[df[x_col] > 10]
            if zoom_df.empty:
                zoom_df = df
        else:
            zoom_df = df
        self._finalize_summary_plot(
            fig, ax, zoom_df, external_y_values, x_col, y_col, plot_config,
        )

    # =====================================================================
    # SCATTER COMPARISON PLOT
    # =====================================================================

    def _generate_cof_histogram(self, plot_config: dict) -> None:
        """Bar chart comparing COF slope per dataset and external series."""
        first_n = plot_config.get('average_first_n_timesteps')
        timestep_range = plot_config.get('average_timestep_range')
        summary_df = self._get_summary_data_df(
            first_n=first_n,
            timestep_range=timestep_range,
        )

        x_col = plot_config.get('x_axis', 'nf')
        y_col_req = plot_config.get('y_axis', 'lf')
        fit_x_range = plot_config.get('fit_x_range')
        x_limits = plot_config.get('x_limits')
        use_visible_x_for_fit = plot_config.get('use_visible_xrange_for_fit', True)
        fit_x_range_effective = fit_x_range
        if (
            fit_x_range_effective is None
            and use_visible_x_for_fit and x_limits and len(x_limits) >= 2
            and x_limits[0] is not None and x_limits[1] is not None
        ):
            fit_x_range_effective = [x_limits[0], x_limits[1]]

        slope_fit_constraint = plot_config.get('slope_fit_constraint', 'none')
        slope_decimals = int(plot_config.get('slope_decimals', 5))

        bar_labels = []
        bar_cofs = []
        bar_errs = []
        bar_colors = []
        color_idx = 0

        df = summary_df.copy()
        datasets = plot_config.get('datasets')
        if datasets:
            df = df[df['dataset_label'].isin(datasets)]
        y_col = y_col_req
        if not df.empty:
            filters = self._apply_default_filters(df, plot_config, x_col)
            df = self._apply_filters(df, filters)
            df = self._apply_range_filters(df, plot_config)
            df = self._apply_material_filter(df, plot_config, 'dataset_label')
            y_col = self._resolve_lf_axis(df, y_col, plot_config)

        ordered_ds = [d for d in (datasets or []) if d in df['dataset_label'].unique()]
        ordered_ds += [d for d in df['dataset_label'].unique() if d not in ordered_ds]
        for ds_label in ordered_ds:
            color = self._get_consistent_color(
                dataset_label=ds_label,
                fallback_index=color_idx,
            )
            color_idx += 1
            group = df[df['dataset_label'] == ds_label]
            if group.empty:
                continue
            plot_data = (
                group.groupby(x_col)[y_col].agg(['mean'])
                .reset_index().sort_values(by=x_col)
            )
            gx = plot_data[x_col].to_numpy()
            gy = plot_data['mean'].to_numpy()
            fit = self._calculate_linear_fit(
                gx,
                gy,
                fit_x_range_effective,
                constraint=slope_fit_constraint,
            )
            if fit is None:
                continue
            label = self._display_dataset_label(ds_label)
            bar_labels.append(label)
            bar_cofs.append(fit['slope'])
            bar_errs.append(fit['slope_stderr'])
            bar_colors.append(color)

        data_cache: dict[str, dict] = {}
        for series in plot_config.get('external_series', []):
            file_path = series.get('file')
            if not file_path:
                color_idx += 1
                continue

            resolved_path = self._resolve_aux_file_path(str(file_path))
            cache_key = str(resolved_path)
            if cache_key not in data_cache:
                try:
                    with open(resolved_path, 'r', encoding='utf-8') as f:
                        data_cache[cache_key] = json.load(f)
                except (IOError, json.JSONDecodeError) as e:
                    logger.warning(
                        "Could not load external data '%s': %s",
                        resolved_path,
                        e,
                    )
                    color_idx += 1
                    continue

            data = data_cache[cache_key]
            for key in series.get('path', []):
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    data = None
                    break
            if not isinstance(data, dict):
                color_idx += 1
                continue

            x_key = series.get('x_key')
            y_key = series.get('y_key')
            if not x_key or not y_key:
                color_idx += 1
                continue

            x_data = list(data.get(x_key, []))
            y_data = list(data.get(y_key, []))
            if not x_data or not y_data:
                color_idx += 1
                continue

            orig_x = list(x_data)
            x_transform = series.get('x_transform')
            y_transform = series.get('y_transform')
            if x_transform:
                transformed_x = self._apply_series_transform(
                    str(x_transform),
                    x_data,
                )
                if transformed_x is not None:
                    x_data = transformed_x
                else:
                    logger.warning(
                        "Unsupported x_transform expression '%s'. Ignoring.",
                        x_transform,
                    )
            if y_transform:
                transformed_y = self._apply_series_transform(
                    str(y_transform),
                    y_data,
                    x_values=orig_x,
                )
                if transformed_y is not None:
                    y_data = transformed_y
                else:
                    logger.warning(
                        "Unsupported y_transform expression '%s'. Ignoring.",
                        y_transform,
                    )

            x_arr = np.array(x_data, dtype=float)
            y_arr = np.array(y_data, dtype=float)
            label = series.get('label', 'External')
            fit = self._calculate_linear_fit(
                x_arr,
                y_arr,
                fit_x_range_effective,
                constraint=slope_fit_constraint,
            )
            color = self._get_consistent_color(
                series_name=label,
                fallback_index=color_idx,
            )
            color_idx += 1
            if fit is None:
                continue

            bar_labels.append(label)
            bar_cofs.append(fit['slope'])
            bar_errs.append(fit['slope_stderr'])
            bar_colors.append(color)

        if not bar_labels:
            logger.warning("No datasets produced COF values for histogram. Skipping.")
            return

        fig, ax = plt.subplots(figsize=self.figure_size)
        x_pos = np.arange(len(bar_labels))
        bar_width = float(plot_config.get('bar_width', 0.6))
        bars_drawn = ax.bar(
            x_pos,
            bar_cofs,
            width=bar_width,
            color=bar_colors,
            yerr=bar_errs,
            capsize=4,
            error_kw={'elinewidth': 1.2, 'ecolor': 'k', 'alpha': 0.7},
            edgecolor='none',
        )

        ax.set_xlim(-0.5, len(bar_labels) - 0.5)
        ax.set_ylim(bottom=0)
        y_top = max(c + e for c, e in zip(bar_cofs, bar_errs)) * 1.35
        ax.set_ylim(top=y_top)

        internal_label_set = {
            self._display_dataset_label(ds_label) for ds_label in ordered_ds
        }
        external_outline_width = float(plot_config.get('external_outline_width', 10.0))
        outline_px = external_outline_width * fig.dpi / 72.0
        p0 = ax.transData.inverted().transform((0.0, 0.0))
        p_dx = ax.transData.inverted().transform((outline_px, 0.0))
        p_dy = ax.transData.inverted().transform((0.0, outline_px))
        inset_x = abs(p_dx[0] - p0[0])
        inset_y = abs(p_dy[1] - p0[1])
        background_color = ax.get_facecolor()
        for bar_obj, bar_label, bar_color in zip(
            bars_drawn,
            bar_labels,
            bar_colors,
        ):
            if bar_label not in internal_label_set:
                bar_obj.set_facecolor(bar_color)
                bar_obj.set_edgecolor('none')
                x0, y0 = bar_obj.get_x(), bar_obj.get_y()
                w0, h0 = bar_obj.get_width(), bar_obj.get_height()
                inset_x_eff = min(inset_x, 0.35 * w0)
                inset_y_eff = min(inset_y, 0.35 * h0)
                inner_w = max(w0 - 2.0 * inset_x_eff, 0.0)
                inner_h = max(h0 - 2.0 * inset_y_eff, 0.0)
                if inner_w > 0.0 and inner_h > 0.0:
                    cutout = Rectangle(
                        (x0 + inset_x_eff, y0 + inset_y_eff),
                        inner_w,
                        inner_h,
                        facecolor=background_color,
                        edgecolor='none',
                        zorder=bar_obj.get_zorder() + 0.2,
                    )
                    ax.add_patch(cutout)

        ax.set_xticks(x_pos)
        rotation = int(plot_config.get('x_label_rotation', 0))
        ha = 'right' if rotation != 0 else 'center'
        ax.set_xticklabels(
            bar_labels,
            rotation=rotation,
            ha=ha,
            fontsize=self.settings["fonts"]["tick_label"],
        )
        ax.set_ylabel(
            plot_config.get('y_label', 'COF'),
            fontsize=self.settings["fonts"]["axis_label"],
        )
        ax.tick_params(
            axis='y',
            which='major',
            labelsize=self.settings["fonts"]["tick_label"],
        )

        if plot_config.get('annotate_bars', True):
            err_pad = (max(bar_errs) if bar_errs else 0) * 0.3 + max(bar_cofs) * 0.01
            for bar_obj, cof_val in zip(bars_drawn, bar_cofs):
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2.0,
                    bar_obj.get_height() + err_pad,
                    f"{cof_val:.{slope_decimals}f}",
                    ha='center',
                    va='bottom',
                    fontsize=self.settings["fonts"]["legend"],
                )

        title = plot_config.get('title')
        if title:
            ax.set_title(title, fontsize=self.settings["fonts"]["title"])

        self._save_plot(fig, plot_config.get('filename'))

    # =====================================================================
    # CORRELATION PLOTS
    # =====================================================================

