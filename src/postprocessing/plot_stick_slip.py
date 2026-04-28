"""Stick-slip analysis plotting mixin."""
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-return-statements,too-few-public-methods
# pylint: disable=missing-class-docstring,too-many-function-args,trailing-newlines
# pylint: disable=duplicate-code

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from .plot_mixin_base import PlotterMixinBase
from .plot_style import StickSlipPSD

logger = logging.getLogger(__name__)


class PlotStickSlipMixin(PlotterMixinBase):
    @staticmethod
    def _find_peaks(*args, **kwargs):
        """Import scipy peak finder only when stick-slip routines are used."""
        from scipy.signal import find_peaks
        return find_peaks(*args, **kwargs)

    @staticmethod
    def _apply_consistent_y_formatter(ax) -> None:
        """Set y formatter so all tick labels share decimal precision."""
        ticks = ax.get_yticks()
        ymin, ymax = ax.get_ylim()
        visible = [t for t in ticks if ymin - 1e-12 <= t <= ymax + 1e-12]
        if len(visible) < 2:
            return
        step = abs(visible[1] - visible[0])
        if step <= 0:
            return
        decimals = max(0, -int(math.floor(math.log10(step))))
        ax.yaxis.set_major_formatter(
            ticker.FormatStrFormatter(f'%.{decimals}f'),
        )

    def _load_stick_slip_run(self, plot_config: dict) -> dict | None:
        """Load and filter one run for stick-slip analysis.

        Returns a dict with keys ``run``, ``t``, ``x``, ``label``,
        ``file_key``, ``displacement_col``, ``time_scale``,
        ``displacement_scale``, ``time_unit``, ``disp_unit``, or
        ``None`` when loading or filtering fails.
        """
        datasets = plot_config.get('datasets', [self.labels[0]] if self.labels else [])
        if not datasets:
            logger.error("No dataset specified for stick_slip_analysis.")
            return None

        label = datasets[0]
        if len(datasets) > 1:
            logger.warning(
                "stick_slip_analysis currently uses one dataset. Using '%s'.",
                label,
            )

        file_key = plot_config.get('filter_size')
        if not file_key:
            logger.error("'filter_size' is required for stick_slip_analysis.")
            return None

        _, local_metadata = self._load_full_data(label, file_key)
        if not local_metadata:
            logger.error(
                "Could not load metadata for label '%s' and file '%s'.",
                label,
                file_key,
            )
            return None

        time_series = local_metadata.get('time_series')
        if not time_series:
            logger.error(
                "'time_series' missing in metadata for label '%s' and file '%s'.",
                label,
                file_key,
            )
            return None

        all_runs = list(self._extract_all_runs(label, file_key))
        if not all_runs:
            logger.error(
                "No runs found for label '%s' and file_key '%s'.",
                label,
                file_key,
            )
            return None

        filters = {
            'id': plot_config.get('filter_materials'),
            'force': plot_config.get('filter_forces') or plot_config.get('force'),
            'pressure': plot_config.get('pressures') or plot_config.get('filter_pressures'),
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

        if not filtered_runs:
            logger.error("No runs matched filters for stick_slip_analysis.")
            return None

        run_index = int(plot_config.get('run_index', 0))
        if run_index < 0 or run_index >= len(filtered_runs):
            logger.warning("run_index %d out of range. Using 0.", run_index)
            run_index = 0

        run = filtered_runs[run_index]
        df = run['df']

        displacement_col = plot_config.get(
            'displacement_axis',
            plot_config.get('secondary_y_axis', 'tip_pos'),
        )
        if displacement_col not in df.columns:
            logger.error(
                "Displacement column '%s' not found. Available: %s",
                displacement_col,
                df.columns.tolist(),
            )
            return None

        time_scale = float(plot_config.get('time_scale', 1.0))
        displacement_scale = float(
            plot_config.get(
                'displacement_scale',
                plot_config.get('secondary_y_scale', 1.0),
            ),
        )

        n = min(len(time_series), len(df[displacement_col]))
        if n < 8:
            logger.error("Not enough points for stick-slip analysis.")
            return None

        t = np.asarray(time_series[:n], dtype=float) * time_scale
        x = np.asarray(df[displacement_col].iloc[:n], dtype=float) * displacement_scale

        finite_mask = np.isfinite(t) & np.isfinite(x)
        t = t[finite_mask]
        x = x[finite_mask]
        if len(t) < 8:
            logger.error("Not enough finite points after filtering.")
            return None

        return {
            'run': run,
            't': t,
            'x': x,
            'label': label,
            'file_key': file_key,
            'displacement_col': displacement_col,
            'time_scale': time_scale,
            'displacement_scale': displacement_scale,
            'time_unit': plot_config.get('time_unit', 'ns'),
            'disp_unit': plot_config.get('displacement_unit', 'nm'),
        }

    def _compute_stick_slip_psd(
        self, t: np.ndarray, detrended: np.ndarray, plot_config: dict,
    ) -> StickSlipPSD | None:
        """Compute FFT-based PSD for stick-slip analysis.

        Returns a :class:`StickSlipPSD` dataclass, or ``None`` when the
        computation cannot proceed.
        """
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            logger.error("Non-positive time step detected.")
            return None

        signal = detrended - np.mean(detrended)
        if plot_config.get('apply_hann_window', True):
            signal = signal * np.hanning(len(signal))

        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=dt)
        psd = (np.abs(fft_vals) ** 2) / max(len(signal), 1)

        valid = freqs > 0
        freq_min = plot_config.get('freq_min')
        if freq_min is not None:
            valid &= freqs >= float(freq_min)
        freq_max = plot_config.get('freq_max')
        if freq_max is not None:
            valid &= freqs <= float(freq_max)
        if not np.any(valid):
            logger.error("No valid frequencies in requested range.")
            return None

        valid_idx = np.where(valid)[0]
        valid_psd = psd[valid]
        fft_peak_prominence = plot_config.get('fft_peak_min_prominence')
        if fft_peak_prominence is None:
            fft_peak_prominence = (
                0.02 * float(np.max(valid_psd)) if len(valid_psd) else 0.0
            )
        fft_peak_distance = max(
            int(plot_config.get('fft_peak_min_distance_points', 1)), 1,
        )
        fft_peak_local_idx, _ = self._find_peaks(
            valid_psd,
            prominence=float(fft_peak_prominence),
            distance=fft_peak_distance,
        )
        if len(fft_peak_local_idx) == 0:
            fft_peak_local_idx = np.array([int(np.argmax(valid_psd))], dtype=int)

        fft_peaks_to_report = max(int(plot_config.get('fft_peaks_to_report', 2)), 1)
        ranked_local_idx = sorted(
            fft_peak_local_idx.tolist(),
            key=lambda idx: valid_psd[idx],
            reverse=True,
        )[:fft_peaks_to_report]

        top_two_local_idx = sorted(
            ranked_local_idx[:2],
            key=lambda idx: float(freqs[int(valid_idx[idx])]),
        )

        fft_peak_rank = max(int(plot_config.get('fft_peak_rank', 1)), 1)
        peak_offset = min(fft_peak_rank - 1, len(ranked_local_idx) - 1)
        peak_idx = int(valid_idx[int(ranked_local_idx[peak_offset])])
        f_slip = float(freqs[peak_idx])
        if f_slip <= 0:
            logger.error("Dominant frequency is non-positive.")
            return None

        peak_labels: list[dict] = []
        for i, local_idx in enumerate(top_two_local_idx, start=1):
            global_idx = int(valid_idx[local_idx])
            f_val = float(freqs[global_idx])
            if f_val <= 0:
                continue
            peak_labels.append({
                'name': f"f{i}",
                'freq': f_val,
                'psd': float(psd[global_idx]),
                'period': float(1.0 / f_val),
            })

        return StickSlipPSD(
            freqs=freqs,
            psd=psd,
            valid=valid,
            peak_labels=peak_labels,
            f_slip=f_slip,
            period=float(1.0 / f_slip),
            fft_peak_rank=fft_peak_rank,
        )

    def _generate_stick_slip_analysis_plot(self, plot_config: dict) -> None:
        """Run stick-slip diagnostics and export displacement/PSD plots."""
        run_data = self._load_stick_slip_run(plot_config)
        if run_data is None:
            return

        run = run_data['run']
        t = run_data['t']
        x = run_data['x']
        time_unit = run_data['time_unit']
        disp_unit = run_data['disp_unit']
        label = run_data['label']
        file_key = run_data['file_key']
        displacement_col = run_data['displacement_col']
        time_scale = run_data['time_scale']
        displacement_scale = run_data['displacement_scale']

        coeff = np.polyfit(t, x, 1)
        slope = float(coeff[0])
        intercept = float(coeff[1])
        trend = slope * t + intercept
        detrended = x - trend

        extrema_signal = detrended
        if plot_config.get('smooth_for_extrema', False):
            window = int(plot_config.get('extrema_smooth_window', 5))
            if window > 1 and window % 2 == 1 and window < len(detrended):  # pylint: disable=chained-comparison
                kernel = np.ones(window) / window
                extrema_signal = np.convolve(detrended, kernel, mode='same')

        prominence = plot_config.get('extrema_min_prominence')
        if prominence is None:
            prominence = 0.8 * np.std(extrema_signal) if len(extrema_signal) > 0 else 0.0
        min_distance = max(int(plot_config.get('extrema_min_distance_points', 5)), 1)
        maxima_idx, _ = self._find_peaks(
            extrema_signal,
            prominence=float(prominence),
            distance=min_distance,
        )
        minima_idx, _ = self._find_peaks(
            -extrema_signal,
            prominence=float(prominence),
            distance=min_distance,
        )

        psd_result = self._compute_stick_slip_psd(t, detrended, plot_config)
        if psd_result is None:
            return

        f_slip = psd_result.f_slip
        T = psd_result.period  # pylint: disable=invalid-name
        freqs = psd_result.freqs
        psd = psd_result.psd
        valid = psd_result.valid
        peak_labels = psd_result.peak_labels
        fft_peak_rank = psd_result.fft_peak_rank

        logger.info("--- Stick-Slip Analysis ---")
        logger.info(
            "Run: id=%s force=%s angle=%s layer=%s",
            run.get('id'),
            run.get('force'),
            run.get('angle'),
            run.get('layer'),
        )
        logger.info("v (slope) = %.8g (%s/%s)", slope, disp_unit, time_unit)
        logger.info("f_slip = %.8g 1/%s", f_slip, time_unit)
        logger.info("T = %.8g %s", T, time_unit)

        displacement_filename = plot_config.get('filename', 'stick_slip_analysis.png')
        displacement_root = Path(displacement_filename).stem
        displacement_ext = Path(displacement_filename).suffix or '.png'
        spectrum_filename = plot_config.get(
            'spectrum_filename',
            f"{displacement_root}_spectrum{displacement_ext}",
        )
        dataset_color = self._get_consistent_color(
            dataset_label=label,
            fallback_index=0,
        )

        base_figure_size = tuple(plot_config.get('figure_size', [10, 11]))
        single_panel_figure_size = tuple(
            plot_config.get(
                'single_panel_figure_size',
                [base_figure_size[0], max(base_figure_size[1] / 3.0, 3.2)],
            ),
        )
        max_intervals_to_draw = int(plot_config.get('max_intervals_to_draw', 40))

        display_time_window = plot_config.get('display_time_window')
        if display_time_window and len(display_time_window) == 2:
            display_start = float(display_time_window[0])
            display_end = float(display_time_window[1])
        else:
            display_periods = plot_config.get('display_periods')
            display_start = float(plot_config.get('display_time_start', t[0]))
            if display_periods is not None:
                display_end = min(t[-1], display_start + float(display_periods) * T)
            else:
                display_end = t[-1]

        display_mask = (t >= display_start) & (t <= display_end)
        if np.count_nonzero(display_mask) < 8:
            display_mask = np.ones_like(t, dtype=bool)

        plot_t = t[display_mask]
        plot_detrended = detrended[display_mask]

        interval_pairs = []
        for p_idx in maxima_idx:
            following_troughs = minima_idx[minima_idx > p_idx]
            if len(following_troughs) == 0:
                continue
            q_idx = int(following_troughs[0])
            interval_pairs.append((int(p_idx), q_idx))
            if len(interval_pairs) >= max_intervals_to_draw:
                break
        displayed_interval_pairs = [
            (p_idx, q_idx)
            for p_idx, q_idx in interval_pairs
            if display_mask[p_idx] and display_mask[q_idx]
        ]

        freq_plot_mask = valid.copy()
        freq_plot_min = plot_config.get('freq_plot_min')
        if freq_plot_min is not None:
            freq_plot_mask &= freqs >= float(freq_plot_min)
        freq_plot_max = plot_config.get('freq_plot_max')
        if freq_plot_max is not None:
            freq_plot_mask &= freqs <= float(freq_plot_max)
        if not np.any(freq_plot_mask):
            freq_plot_mask = valid.copy()

        fig_disp, ax_disp = plt.subplots(1, 1, figsize=single_panel_figure_size)
        ax_disp.plot(plot_t, plot_detrended, color=dataset_color, linewidth=1.2)
        ax_disp.axhline(0.0, color='gray', linestyle='--', linewidth=0.8)

        if plot_config.get('show_peak_trough_intervals', False):
            for p_idx, q_idx in displayed_interval_pairs:
                ax_disp.plot(
                    [t[p_idx], t[q_idx]],
                    [detrended[p_idx], detrended[q_idx]],
                    color='tab:red',
                    alpha=0.35,
                    linewidth=1.0,
                )

        if peak_labels:
            y_span = float(max(np.ptp(plot_detrended), 1e-6))
            t_span = max(plot_t[-1] - plot_t[0], 1e-9)
            y_base = float(np.max(plot_detrended) + 0.10 * y_span)

            extrema_for_intervals = [int(i) for i in maxima_idx if display_mask[i]]
            if len(extrema_for_intervals) < 2:
                extrema_for_intervals = [int(i) for i in minima_idx if display_mask[i]]
            extrema_for_intervals = sorted(extrema_for_intervals)

            def _pick_extrema_pair(target_period, side):
                if target_period <= 0.0 or len(extrema_for_intervals) < 2:
                    return None
                candidates = []
                for left in range(len(extrema_for_intervals) - 1):
                    i0 = extrema_for_intervals[left]
                    for right in range(left + 1, len(extrema_for_intervals)):
                        i1 = extrema_for_intervals[right]
                        dt_pair = float(t[i1] - t[i0])
                        if dt_pair <= 0.0:
                            continue
                        if dt_pair > 1.6 * target_period:
                            break
                        fit_err = abs(dt_pair - target_period) / target_period
                        if side == 'start':
                            edge_penalty = (t[i0] - plot_t[0]) / t_span
                        else:
                            edge_penalty = (plot_t[-1] - t[i1]) / t_span
                        score = fit_err + 0.5 * edge_penalty
                        candidates.append((score, i0, i1))
                if not candidates:
                    return None
                candidates.sort(key=lambda row: row[0])
                return candidates[0][1], candidates[0][2]

            for pos, item in zip(('start', 'end'), peak_labels[:2]):
                period = float(item['period'])
                if period <= 0.0 or period > 0.95 * t_span:
                    continue
                extrema_pair = _pick_extrema_pair(period, pos)
                if extrema_pair is None:
                    continue
                i0, i1 = extrema_pair
                x0 = float(t[i0])
                x1 = float(t[i1])
                ax_disp.annotate(
                    '',
                    xy=(x0, y_base),
                    xytext=(x1, y_base),
                    arrowprops={'arrowstyle': '<->', 'color': 'tab:red', 'lw': 1.2, 'alpha': 0.75},
                    zorder=8,
                )
                ax_disp.plot(
                    [x0, x0], [float(detrended[i0]), y_base],
                    color='tab:red', linewidth=0.9, alpha=0.45, zorder=7,
                )
                ax_disp.plot(
                    [x1, x1], [float(detrended[i1]), y_base],
                    color='tab:red', linewidth=0.9, alpha=0.45, zorder=7,
                )
                ax_disp.text(
                    0.5 * (x0 + x1),
                    y_base + 0.025 * y_span,
                    f"T({item['name']})",
                    ha='center',
                    va='bottom',
                    fontsize=self.settings["fonts"]["legend"],
                    color='tab:red',
                    bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'},
                    zorder=6,
                )

            y_top_needed = float(y_base + 0.25 * y_span)
            y_min_current, y_max_current = ax_disp.get_ylim()
            if y_top_needed > y_max_current:
                ax_disp.set_ylim(y_min_current, y_top_needed)

        ax_disp.set_xlabel(
            plot_config.get('x_label', f"Time ({time_unit})"),
            fontsize=self.settings["fonts"]["axis_label"],
        )
        ax_disp.set_ylabel(
            plot_config.get('top_ylabel', f"Displacement ({disp_unit})"),
            fontsize=self.settings["fonts"]["axis_label"],
        )
        ax_disp.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings["fonts"]["tick_label"],
        )
        ax_disp.grid(True, alpha=0.3)
        self._apply_consistent_y_formatter(ax_disp)
        fig_disp.tight_layout()
        self._save_plot(fig_disp, displacement_filename, tight_bbox=False)

        fig_psd, ax_psd = plt.subplots(1, 1, figsize=single_panel_figure_size)
        ax_psd.plot(freqs[freq_plot_mask], psd[freq_plot_mask], color=dataset_color, linewidth=1.2)

        peak_freqs = [
            peak['freq']
            for peak in peak_labels
            if freq_plot_mask[np.argmin(np.abs(freqs - peak['freq']))]
        ]
        peak_psd_values = [
            peak['psd']
            for peak in peak_labels
            if freq_plot_mask[np.argmin(np.abs(freqs - peak['freq']))]
        ]
        if peak_freqs:
            ax_psd.scatter(peak_freqs, peak_psd_values, color=dataset_color, s=28, zorder=3)

        for peak in peak_labels:
            if not freq_plot_mask[np.argmin(np.abs(freqs - peak['freq']))]:
                continue
            ax_psd.scatter(
                [peak['freq']],
                [peak['psd']],
                color=dataset_color,
                s=40,
                zorder=4,
            )
            ax_psd.text(
                peak['freq'],
                peak['psd'] * 1.05 if peak['psd'] > 0 else peak['psd'] + 1e-6,
                peak['name'],
                color='tab:red',
                fontsize=self.settings["fonts"]["legend"],
                ha='center',
                va='bottom',
            )

        ax_psd.set_xlabel(
            plot_config.get('freq_xlabel', f"Frequency (1/{time_unit})"),
            fontsize=self.settings["fonts"]["axis_label"],
        )
        ax_psd.set_ylabel(
            plot_config.get('bottom_ylabel', 'PSD'),
            fontsize=self.settings["fonts"]["axis_label"],
        )
        ax_psd.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings["fonts"]["tick_label"],
        )
        ax_psd.grid(True, alpha=0.3)
        self._apply_consistent_y_formatter(ax_psd)
        fig_psd.tight_layout()
        self._save_plot(fig_psd, spectrum_filename, tight_bbox=False)

        results_filename = plot_config.get('results_filename')
        if results_filename:
            results_path = self.output_dir / results_filename
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'dataset': label,
                    'file_key': file_key,
                    'run': {
                        'id': run.get('id'),
                        'force': run.get('force'),
                        'angle': run.get('angle'),
                        'layer': run.get('layer'),
                        'speed': run.get('speed'),
                        'tip_radius': run.get('tip_radius'),
                    },
                    'displacement_column': displacement_col,
                    'time_scale': time_scale,
                    'displacement_scale': displacement_scale,
                    'frequency_analysis': {
                        'selected_primary_frequency': f_slip,
                        'selected_peak_rank': fft_peak_rank,
                    },
                }, f, indent=2)
            logger.info("Saved stick-slip analysis results: %s", results_path)

