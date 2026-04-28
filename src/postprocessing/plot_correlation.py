"""Correlation plotting mixin."""
# pylint: disable=too-many-locals,too-few-public-methods
# pylint: disable=missing-class-docstring,trailing-newlines

from __future__ import annotations

import json
import logging
import re

import matplotlib.pyplot as plt
import pandas as pd

from .plot_mixin_base import PlotterMixinBase

logger = logging.getLogger(__name__)


class PlotCorrelationMixin(PlotterMixinBase):
    @staticmethod
    def _seaborn():
        """Import seaborn on demand for correlation heatmaps."""
        import seaborn as sns
        return sns

    def _generate_correlation_plots(self, plot_config: dict) -> None:
        """Generate correlation heatmaps from friction ranking files."""
        logger.info("Generating correlation plots...")

        ranking_files = list(
            self.output_dir.glob('friction_ranking_*.json'),
        )
        if not ranking_files:
            logger.error("No friction_ranking_*.json files found.")
            return

        all_ranks = []
        for f_path in ranking_files:
            size_match = re.search(
                r'friction_ranking_(.+)\.json', f_path.name,
            )
            if not size_match:
                continue
            size = size_match.group(1)

            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for force_key, material_list in data.items():
                    force = float(force_key[1:])
                    for rank, material_data in enumerate(material_list, 1):
                        all_ranks.append({
                            'size': size, 'force': force,
                            'material': material_data['material'],
                            'rank': rank,
                        })

        if not all_ranks:
            logger.error("Could not parse ranking data.")
            return

        rank_df = pd.DataFrame(all_ranks)
        correlate_by = plot_config.get('correlate_by')

        if correlate_by == 'size':
            self._plot_correlation_heatmap(rank_df, 'size', plot_config)
        elif correlate_by == 'force':
            self._plot_correlation_heatmap(rank_df, 'force', plot_config)
        elif correlate_by == 'pairwise':
            self._plot_pairwise_correlation(rank_df, plot_config)
        else:
            logger.error("Unknown correlation type '%s'", correlate_by)

    def _plot_correlation_heatmap(
        self, rank_df: pd.DataFrame, correlate_by: str, plot_config: dict,
    ) -> None:
        """Generic correlation heatmap plotter."""

        if correlate_by == 'size':
            force = plot_config.get('correlation_force', 30)
            df = rank_df[rank_df['force'] == force]
            pivot_df = df.pivot_table(
                index='material', columns='size',
                values='rank', aggfunc='mean',
            )
            title = f'Rank Correlation Across Sizes (Force={force}nN)'
            filename = plot_config.get(
                'filename', f'rank_corr_by_size_f{force}.png',
            )
        else:
            for size, group in rank_df.groupby('size'):
                pivot_df = group.pivot_table(
                    index='material', columns='force',
                    values='rank', aggfunc='mean',
                )
                pivot_df.dropna(inplace=True)

                if len(pivot_df) < 2 or len(pivot_df.columns) < 2:
                    continue

                corr = pivot_df.corr(method='spearman')

                fig, ax = plt.subplots(figsize=(10, 8))
                sns = self._seaborn()
                sns.heatmap(corr, annot=True, cmap='crest', fmt='.2f', ax=ax)
                ax.set_title(
                    f'Rank Correlation Across Forces (Size={size})',
                )

                filename = (
                    plot_config.get('filename_prefix', 'rank_corr_by_force')
                    + f'_{size}.png'
                )
                self._save_plot(fig, filename)
            return

        pivot_df.dropna(inplace=True)
        if len(pivot_df) < 2:
            logger.error("Not enough data for correlation matrix.")
            return

        corr = pivot_df.corr(method='spearman')

        fig, ax = plt.subplots(figsize=(8, 6))
        sns = self._seaborn()
        sns.heatmap(corr, annot=True, cmap='crest', fmt='.2f', ax=ax)
        ax.set_title(title)

        self._save_plot(fig, filename)

    def _plot_pairwise_correlation(
        self, rank_df: pd.DataFrame, plot_config: dict,
    ) -> None:
        """Plot force-vs-force correlation between two sizes."""

        sizes = plot_config.get('sizes_to_compare')
        if not sizes or len(sizes) != 2:
            logger.error(
                "'pairwise' requires 'sizes_to_compare' with two sizes.",
            )
            return

        size1, size2 = sizes
        df1 = rank_df[rank_df['size'] == size1]
        df2 = rank_df[rank_df['size'] == size2]

        forces1 = sorted(df1['force'].unique())
        forces2 = sorted(df2['force'].unique())

        if not forces1 or not forces2:
            logger.error("No force data for sizes: %s, %s", size1, size2)
            return

        corr_matrix = pd.DataFrame(
            index=forces2, columns=forces1, dtype=float,
        )

        for f1 in forces1:
            for f2 in forces2:
                ranks1 = (
                    df1[df1['force'] == f1]
                    .groupby('material')['rank'].mean()
                )
                ranks2 = (
                    df2[df2['force'] == f2]
                    .groupby('material')['rank'].mean()
                )
                combined = pd.DataFrame(
                    {'r1': ranks1, 'r2': ranks2},
                ).dropna()

                if len(combined) > 1:
                    corr_matrix.loc[f2, f1] = combined['r1'].corr(
                        combined['r2'], method='spearman',
                    )

        fig, ax = plt.subplots(figsize=(12, 10))
        sns = self._seaborn()
        sns.heatmap(
            corr_matrix, annot=True, fmt='.2f', cmap='crest',
            cbar_kws={'label': "Spearman's Correlation"},
            linewidths=.5, ax=ax,
        )

        ax.set_title(f'Force vs Force Correlation ({size1} vs {size2})')
        ax.set_xlabel(f'Force (nN) for {size1}')
        ax.set_ylabel(f'Force (nN) for {size2}')

        filename = plot_config.get(
            'filename',
            f'force_vs_force_corr_{size1}_vs_{size2}.png',
        )
        self._save_plot(fig, filename)

    # =====================================================================
    # FRICTION RANKING
    # =====================================================================

