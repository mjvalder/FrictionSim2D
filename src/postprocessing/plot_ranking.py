"""Friction ranking export mixin."""
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-few-public-methods,missing-class-docstring
# pylint: disable=trailing-newlines

from __future__ import annotations

import json
import logging

from .plot_mixin_base import PlotterMixinBase

logger = logging.getLogger(__name__)


class PlotRankingMixin(PlotterMixinBase):
    def rank_friction(self, plot_config: dict) -> None:
        """Rank materials by friction and export to JSON."""
        logger.info("Generating friction rankings...")

        summary_df = self._get_summary_data_df()
        if summary_df.empty:
            logger.warning("Summary data is empty.")
            return

        metrics = plot_config.get('rank_by', ['lf'])
        if not isinstance(metrics, list):
            metrics = [metrics]

        df = summary_df.copy()
        layer = plot_config.get('filter_layer')
        if layer:
            df = df[df['layer'] == layer]

        angle = plot_config.get('angle', 0)
        df = df[df['angle'] == angle]

        force = plot_config.get('force')
        if force is not None:
            df = df[df['force'] == force]

        force_range = plot_config.get('force_range')
        if force_range is None:
            force_range = plot_config.get('filter_force_range')

        if df.empty:
            logger.warning("No data available for ranking.")
            return

        fit_x_range = plot_config.get('fit_x_range')
        aggregate_over_force_range = bool(
            plot_config.get('aggregate_over_force_range', False),
        )
        rank_order = str(plot_config.get('rank_order', 'auto')).lower()

        for metric in metrics:
            if metric not in df.columns:
                logger.warning("Metric '%s' not found. Skipping.", metric)
                continue

            rank_df = df[df[metric] > 0].copy()
            if rank_df.empty:
                continue

            if force_range and len(force_range) == 2:
                lo, hi = float(force_range[0]), float(force_range[1])
                rank_df = rank_df[
                    (rank_df['force'] >= lo) & (rank_df['force'] <= hi)
                ]
                if rank_df.empty:
                    logger.warning(
                        "No data for metric '%s' in force range [%s, %s].",
                        metric,
                        lo,
                        hi,
                    )
                    continue

            if rank_order == 'auto':
                ascending = metric == 'cof'
            else:
                ascending = rank_order in [
                    'asc', 'ascending', 'low_to_high', 'lowest_first',
                ]

            if aggregate_over_force_range:
                agg_df = (
                    rank_df.groupby(['size', 'id'])
                    .agg(
                        mean_value=(metric, 'mean'),
                        n_points=(metric, 'count'),
                        min_force=('force', 'min'),
                        max_force=('force', 'max'),
                    )
                    .reset_index()
                )

                for size, group in agg_df.groupby('size'):
                    ranked = group.sort_values(
                        'mean_value',
                        ascending=ascending,
                    ).reset_index(drop=True)
                    ranked['rank'] = ranked.index + 1

                    materials = []
                    for _, row in ranked.iterrows():
                        materials.append({
                            'material': row['id'],
                            'rank': int(row['rank']),
                            'metric': metric,
                            'mean_value': float(row['mean_value']),
                            'n_points': int(row['n_points']),
                            'min_force': float(row['min_force']),
                            'max_force': float(row['max_force']),
                        })

                    if force_range and len(force_range) == 2:
                        range_key = (
                            f"average_f{float(force_range[0]):g}"
                            f"_to_f{float(force_range[1]):g}"
                        )
                    else:
                        range_key = "average_over_available_forces"

                    output_data = {
                        'ranking_mode': 'average_over_force_range',
                        'rank_order': (
                            'ascending' if ascending else 'descending'
                        ),
                        range_key: materials,
                    }

                    if plot_config.get('filename'):
                        filename = plot_config['filename']
                    else:
                        layer_str = f"_layer{layer}" if layer else ""
                        range_suffix = ""
                        if force_range and len(force_range) == 2:
                            range_suffix = (
                                f"_f{float(force_range[0]):g}"
                                f"to{float(force_range[1]):g}"
                            )
                        filename = (
                            f'friction_ranking_{metric}_{size}'
                            f'{range_suffix}{layer_str}.json'
                        )

                    output_path = self.output_dir / filename
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=4)
                    logger.info(
                        "Exported range-averaged ranking to %s",
                        output_path,
                    )

                continue

            agg_df = (
                rank_df.groupby(['size', 'force', 'id'])[metric]
                .mean().reset_index()
            )

            for size, group in agg_df.groupby('size'):
                fits = {}
                for mat_id, mat_group in group.groupby('id'):
                    fit = self._calculate_linear_fit(
                        mat_group['force'].values,
                        mat_group[metric].values,
                        fit_x_range,
                    )
                    fits[mat_id] = fit

                ranks_by_force = {}
                for f, f_group in group.groupby('force'):
                    ranked = f_group.sort_values(
                        metric, ascending=ascending,
                    ).reset_index()
                    ranked['rank'] = ranked.index + 1

                    materials = []
                    for _, row in ranked.iterrows():
                        record = {
                            'material': row['id'],
                            'rank': row['rank'],
                            'metric': metric,
                            'mean_value': row[metric],
                        }
                        if fits.get(row['id']):
                            record.update({
                                'fit_slope_stderr': fits[row['id']]['slope_stderr'],
                                'fit_r_squared': fits[row['id']]['r_squared'],
                                'fit_rmse': fits[row['id']]['rmse'],
                            })
                        materials.append(record)

                    ranks_by_force[f"f{f}"] = materials

                if plot_config.get('filename'):
                    filename = plot_config['filename']
                else:
                    layer_str = f"_layer{layer}" if layer else ""
                    filename = (
                        f'friction_ranking_{metric}_{size}{layer_str}.json'
                    )

                output_path = self.output_dir / filename
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(ranks_by_force, f, indent=4)
                logger.info("Exported ranking to %s", output_path)

    # =====================================================================
    # MAIN DISPATCHER
    # =====================================================================

