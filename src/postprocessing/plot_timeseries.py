"""Timeseries plotting mixin."""
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-few-public-methods,missing-class-docstring
# pylint: disable=too-many-nested-blocks,duplicate-code

from __future__ import annotations

import logging

import matplotlib.pyplot as plt

from .plot_mixin_base import PlotterMixinBase

logger = logging.getLogger(__name__)


class PlotTimeseriesMixin(PlotterMixinBase):
    def _generate_timeseries_plot(self, plot_config: dict) -> None:
        """Generate a timeseries plot with support for multiple datasets."""
        datasets = plot_config.get('datasets', [self.labels[0]] if self.labels else [])
        if not datasets:
            logger.error("No dataset specified for timeseries plot.")
            return

        file_key = plot_config.get('filter_size')
        if not file_key:
            logger.error("'filter_size' is required for timeseries plots.")
            return

        time_series = None
        for label in datasets:
            _, local_metadata = self._load_full_data(label, file_key)
            if local_metadata:
                time_series = local_metadata.get('time_series')
                if time_series:
                    break
        if not time_series:
            logger.error("'time_series' not found in metadata for any dataset.")
            return

        all_runs_by_dataset = {}
        for label in datasets:
            all_runs = list(self._extract_all_runs(label, file_key))
            if all_runs:
                all_runs_by_dataset[label] = all_runs
            else:
                logger.warning(
                    "No runs found for label '%s' and file_key '%s'.",
                    label,
                    file_key,
                )

        if not all_runs_by_dataset:
            logger.warning("No runs found for any dataset.")
            return

        pressure_filter = None
        force_filter = None
        if not plot_config.get('plot_all_forces'):
            force_filter = plot_config.get('filter_forces') or plot_config.get('force')
            pressure_filter = (
                plot_config.get('pressures')
                or plot_config.get('filter_pressures')
            )

        def apply_filters_to_runs(all_runs) -> list[dict]:
            filters = {
                'id': plot_config.get('filter_materials'),
                'force': force_filter,
                'pressure': pressure_filter,
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

            filters = {
                k: v for k, v in filters.items()
                if k not in ['force', 'pressure'] or v is not None
            }

            filtered_runs = []
            for run in all_runs:
                match = True
                for key, value in filters.items():
                    if value is None:
                        continue
                    run_value = run.get(key)
                    if run_value is None:
                        match = False
                        break
                    if key == 'id' and isinstance(value, list):
                        if not any(v in run_value for v in value):
                            match = False
                            break
                    elif isinstance(value, list):
                        if run_value not in value:
                            match = False
                            break
                    elif run_value != value:
                        match = False
                        break
                if match:
                    filtered_runs.append(run)

            if force_filter and isinstance(force_filter, list):
                force_order = {
                    force: idx for idx, force in enumerate(force_filter)
                }
                force_default_idx = len(force_order)
                filtered_runs.sort(
                    key=lambda r: force_order.get(
                        r.get('force'), force_default_idx,
                    ),
                )
            elif pressure_filter and isinstance(pressure_filter, list):
                pressure_order = {
                    pressure: idx for idx, pressure in enumerate(pressure_filter)
                }
                pressure_default_idx = len(pressure_order)
                filtered_runs.sort(
                    key=lambda r: pressure_order.get(
                        r.get('pressure'), pressure_default_idx,
                    ),
                )

            return filtered_runs

        all_filtered_runs = {}
        for label, all_runs in all_runs_by_dataset.items():
            filtered_runs = apply_filters_to_runs(all_runs)
            if filtered_runs:
                all_filtered_runs[label] = filtered_runs

        if not all_filtered_runs:
            logger.warning("No runs matched filters for timeseries plot.")
            return

        total_filtered_runs = sum(
            len(filtered_runs) for filtered_runs in all_filtered_runs.values()
        )
        use_run_labels = bool(
            plot_config.get(
                'use_run_labels',
                total_filtered_runs > len(all_filtered_runs),
            ),
        )
        run_label_mode = str(plot_config.get('run_label_mode', 'id_force')).lower()

        y_col = plot_config.get('y_axis')
        if not y_col:
            logger.error("Missing y-axis for timeseries plot.")
            return

        fig, ax = plt.subplots(figsize=self.figure_size)

        time_scale = plot_config.get('time_scale', 1.0)
        shift_time_to_zero = plot_config.get('shift_time_to_zero', True)
        time_origin = time_series[0] if shift_time_to_zero and len(time_series) else 0.0
        scaled_time_series = [(t - time_origin) * time_scale for t in time_series]

        secondary_y_col = plot_config.get('secondary_y_axis')
        secondary_y_label = plot_config.get('secondary_y_label')
        secondary_y_scale = plot_config.get('secondary_y_scale', 1.0)
        primary_line_style = plot_config.get('primary_line_style', '-')
        secondary_line_style = plot_config.get('secondary_line_style', '--')
        primary_line_alpha = float(plot_config.get('primary_line_alpha', 0.95))
        secondary_line_alpha = float(plot_config.get('secondary_line_alpha', 0.5))
        base_width = float(self.settings["lines"]["width"])
        primary_line_width = float(
            plot_config.get('primary_line_width', base_width * 1.25),
        )
        secondary_line_width = float(
            plot_config.get('secondary_line_width', base_width * 0.9),
        )
        secondary_dash_pattern = plot_config.get('secondary_dash_pattern')
        ax2 = ax.twinx() if secondary_y_col else None

        use_dataset_colors = self._should_use_dataset_colors(
            plot_config,
            default=(len(all_filtered_runs) > 1),
        )

        color_idx = 0
        dataset_handles = []
        dataset_labels = []
        for label, filtered_runs in all_filtered_runs.items():
            for run in filtered_runs:
                df = run['df']
                run_y_col = self._resolve_lf_axis(df, y_col, plot_config)
                if run_y_col not in df.columns:
                    logger.warning(
                        "y-axis '%s' not found for run %s. Skipping.",
                        run_y_col,
                        run.get('id'),
                    )
                    continue

                if use_run_labels:
                    if run_label_mode == 'force_only':
                        if run.get('force') is not None:
                            primary_label = f"{float(run['force']):g} nN"
                        elif run.get('pressure') is not None:
                            primary_label = f"{float(run['pressure']):g}"
                        else:
                            primary_label = run.get(
                                'id', self._display_dataset_label(label),
                            )
                    else:
                        label_parts = [run.get('id', self._display_dataset_label(label))]
                        if run.get('force') is not None:
                            label_parts.append(f"F={float(run['force']):.1f}nN")
                        elif run.get('pressure') is not None:
                            label_parts.append(f"P={float(run['pressure']):.3g}")
                        primary_label = ", ".join(label_parts)
                else:
                    primary_label = self._display_dataset_label(label)

                color = (
                    self._get_consistent_color(
                        dataset_label=label,
                        fallback_index=color_idx,
                    )
                    if use_dataset_colors
                    else self._get_palette_color(color_idx)
                )
                primary_line, = ax.plot(
                    scaled_time_series,
                    df[run_y_col],
                    label=primary_label,
                    color=color,
                    linewidth=primary_line_width,
                    linestyle=primary_line_style,
                    alpha=primary_line_alpha,
                )
                if primary_label not in dataset_labels:
                    dataset_handles.append(primary_line)
                    dataset_labels.append(primary_label)

                if secondary_y_col:
                    if secondary_y_col not in df.columns:
                        logger.warning(
                            "Secondary y-axis '%s' not found for run %s. "
                            "Skipping overlay.",
                            secondary_y_col,
                            run.get('id'),
                        )
                    else:
                        assert ax2 is not None
                        secondary_line, = ax2.plot(
                            scaled_time_series,
                            df[secondary_y_col] * secondary_y_scale,
                            label='_nolegend_',
                            color=color,
                            linewidth=secondary_line_width,
                            linestyle=secondary_line_style,
                            alpha=secondary_line_alpha,
                        )
                        if secondary_dash_pattern is not None:
                            try:
                                secondary_line.set_dashes(
                                    tuple(
                                        float(v)
                                        for v in secondary_dash_pattern
                                    ),
                                )
                            except (TypeError, ValueError):
                                logger.warning(
                                    "Invalid secondary_dash_pattern; expected a list like [6, 3].",
                                )

                color_idx += 1

        self._finalize_plot(ax, plot_config, 'time', y_col)

        if ax2 and secondary_y_label:
            ax2.set_ylabel(
                secondary_y_label,
                fontsize=self.settings["fonts"]["axis_label"],
            )
            ax2.tick_params(
                axis='both', which='major',
                labelsize=self.settings["fonts"]["tick_label"],
            )

        if dataset_labels:
            dataset_legend = ax.legend(
                dataset_handles,
                dataset_labels,
                loc=plot_config.get(
                    'legend_location',
                    self.settings["legend"]["location"],
                ),
                fontsize=self.settings["fonts"]["legend"],
            )
            ax.add_artist(dataset_legend)

        self._save_plot(fig, plot_config.get('filename'))
