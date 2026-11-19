"""Analysis-figure plot types: they read tables and snapshots, not run exports.

Four plot types ported from the publication plotter, which reproduced figures
first drawn by ``ML_taxonomy`` (``scripts/figs/fig_fs2d_article.py``: the period
histogram, contact scatters and sheet profile; ``build_round13_integrity_audit.py``:
the integrity panels). The drawing follows the originals line for line so the
images reproduce. Tables are read with ``float_precision='round_trip'``: the
default CSV parser can be 1 ulp off the written value.
"""
# pylint: disable=too-many-locals,too-many-statements,too-many-branches

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .plot_mixin_base import PlotDataError, PlotterMixinBase
from .plot_style import GEM12_COLORS
from .snapshot_geometry import radial_profile, read_type_groups

logger = logging.getLogger(__name__)

_STATUS_TEXT = {"halted": "halted (T guard)", "truncated": "truncated", "no_output": "crashed",
                "not_reached": "not reached", "unreadable": "unreadable"}


def _source_path(source: Any) -> Path:
    """The file of a ``TableSource`` given as a model, its dump, or a bare path."""
    if isinstance(source, dict):
        source = source.get('file')
    elif hasattr(source, 'file'):
        source = source.file
    if not source:
        raise PlotDataError('table source names no file')
    return Path(source)


def load_table(source: Any, base_dir: Path, required: Sequence[str] = ()) -> pd.DataFrame:
    """Read a table source; raise naming the table and any missing column.

    A relative path is tried as given, then under ``base_dir``.
    """
    path = _source_path(source)
    if not path.is_absolute() and not path.exists():
        path = Path(base_dir) / path
    if not path.is_file():
        raise PlotDataError(f"{path}: table not found")
    table = pd.read_csv(path, float_precision='round_trip')
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise PlotDataError(f"{path.name}: missing column(s) {missing}")
    return table


class PlotTablesMixin(PlotterMixinBase):
    """stickslip_period_histogram, contact_cof_scatter, sheet_profile, integrity_panel."""

    @staticmethod
    def _required(plot_config: dict, key: str) -> Any:
        """``plot_config[key]``, or a PlotDataError naming the plot type and key
        (a document being edited may leave them empty; one being rendered may not)."""
        value = plot_config.get(key)
        if value in (None, '', {}):
            raise PlotDataError(f"{plot_config.get('plot_type')} needs {key}")
        return value

    def _require_all(self, plot_config: dict, *keys: str) -> None:
        """:meth:`_required` for each key, before any file is opened."""
        for key in keys:
            self._required(plot_config, key)

    def _table(self, source: Any, required: Sequence[str] = ()) -> pd.DataFrame:
        resolved = self._resolve_aux_file_path(str(_source_path(source)))
        return load_table(resolved, self.output_dir.parent, required)

    @staticmethod
    def _load_colors(loads, plot_config: dict) -> list:
        """One colour per load: ``load_colors`` from the spec, else the palette in order."""
        colors = plot_config.get('load_colors') or GEM12_COLORS
        return [colors[i % len(colors)] for i in range(len(loads))]

    def _eligible_materials(self, ranking_source: Any) -> pd.DataFrame:
        """Rows of the COF ranking whose ``eligible`` flag is set, indexed by material."""
        r = self._table(ranking_source, ['material', 'eligible'])
        return r[r['eligible'].astype(bool)].set_index('material')

    def _generate_stickslip_period_histogram(self, plot_config: dict) -> None:
        """Lateral-force period / lattice period, one step line per load."""
        self._require_all(plot_config, 'stickslip_file', 'ranking_file')
        s = self._table(self._required(plot_config, 'stickslip_file'),
                        ['material', 'load_nN', 'lf_period_A', 'a_x_A', 'lf_period_clear'])
        eligible = self._eligible_materials(self._required(plot_config, 'ranking_file'))
        s = s[s['material'].isin(eligible.index) & s['lf_period_clear'].astype(bool)]
        loads = plot_config.get('loads', [10, 20, 30, 50])
        colors = self._load_colors(loads, plot_config)
        bins = np.arange(*plot_config.get('bins_arange', [0.4, 2.6, 0.1]))

        fig, ax = self.new_figure()
        for load, color in zip(loads, colors):
            g = s[s['load_nN'] == load]
            ax.hist(g['lf_period_A'] / g['a_x_A'], bins=bins, histtype='step',
                    lw=plot_config.get('line_width', 2.0), color=color,
                    label=f"{load:g} nN")
        ref = plot_config.get('reference_line', 1)
        if ref is not None:
            ax.axvline(ref, color='k', ls='--', lw=1.2)
        self._finalize_plot(ax, plot_config, 'period_ratio', 'runs')
        self._save_plot(fig, plot_config.get('filename'))

    def _generate_contact_cof_scatter(self, plot_config: dict) -> None:
        """COF vs a tip-contact measure at one state, one colour per load."""
        self._require_all(plot_config, 'contact_file', 'ranking_file', 'x_column')
        from matplotlib import ticker  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        x_col = self._required(plot_config, 'x_column')
        c = self._table(self._required(plot_config, 'contact_file'), ['material', 'load_nN', 'state', x_col])
        eligible = self._eligible_materials(self._required(plot_config, 'ranking_file'))
        c = c[(c['state'] == plot_config.get('state', 'load'))
              & c['material'].isin(eligible.index)].copy()
        c['cof'] = [eligible.loc[m, f"cof_{load:g}nN"]
                    for m, load in zip(c['material'], c['load_nN'])]
        loads = plot_config.get('loads', [10, 20, 30, 50])
        colors = self._load_colors(loads, plot_config)
        marker_area = self.settings["markers"]["size"] ** 2 * plot_config.get('marker_scale', 0.8)

        fig, ax = self.new_figure()
        for load, color in zip(loads, colors):
            g = c[c['load_nN'] == load]
            ax.scatter(g[x_col], g['cof'], s=marker_area, color=color,
                       alpha=plot_config.get('alpha', 0.75), edgecolor='none',
                       label=f"{load:g} nN")
        if plot_config.get('y_scale', 'log') == 'log':
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        if plot_config.get('y_ticks'):
            ax.set_yticks(plot_config['y_ticks'])
        self._finalize_plot(ax, plot_config, x_col, 'cof')
        self._save_plot(fig, plot_config.get('filename'))

    def _generate_sheet_profile(self, plot_config: dict) -> None:
        """Height of the top surface and bottom layer vs distance from the tip axis.

        Reads ``<run_dir>/<layer_dir>/data/load_<L>N.data`` (the loaded state
        before sliding) and the groups of ``.../lammps/system.in.settings``. For
        each layer only its upper sublayer is profiled (types whose far-field
        median z is within ``sublayer_dz`` of the highest), relative to that
        sublayer's far-field median.
        """
        self._require_all(plot_config, 'run_dir')
        from .integrity import read_snapshot  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        run = Path(self._required(plot_config, 'run_dir')) / plot_config.get('layer_dir', 'L2')
        groups = read_type_groups(run / 'lammps' / 'system.in.settings')
        top = groups.get('layer_2') or groups.get('layer_1', set())
        bottom = groups.get('layer_1', set()) - top
        loads = plot_config.get('loads', [10, 20, 30, 50])
        colors = self._load_colors(loads, plot_config)
        edges = np.arange(*plot_config.get('radial_bins_arange', [0, 42, 1.5]))
        tip_types = plot_config.get('tip_types', [4, 5, 6])
        apex_depth = plot_config.get('tip_apex_depth', 2.0)
        far_radius = plot_config.get('far_field_radius', 30)
        sublayer_dz = plot_config.get('sublayer_dz', 0.5)
        lw = plot_config.get('line_width', 2.0)

        fig, ax = self.new_figure()
        for load, color in zip(loads, colors):
            typ, xyz, _ = read_snapshot(run / 'data' / f"load_{load:g}N.data")
            tip = xyz[np.isin(typ, tip_types)]
            axis = tip[tip[:, 2] < tip[:, 2].min() + apex_depth][:, :2].mean(axis=0)
            for group, ls in ((top, "-"), (bottom, "--")):
                m = np.isin(typ, list(group))
                pts, pt = xyz[m], typ[m]
                far = np.linalg.norm(pts[:, :2] - axis, axis=1) > far_radius
                zt = {t: np.median(pts[far & (pt == t), 2])
                      for t in np.unique(pt) if (far & (pt == t)).any()}
                up = np.isin(pt, [t for t, z in zt.items() if z >= max(zt.values()) - sublayer_dz])
                zf = np.median(pts[up & far, 2])
                x, y = radial_profile(pts[up, :2], pts[up, 2], axis, zf, edges)
                ax.plot(x, y, ls, color=color, lw=lw,
                        label=f"{load:g} nN" if ls == "-" else None)
        ax.axhline(0, color="k", lw=0.8)
        ax.plot([], [], "k-", label=plot_config.get('top_label', 'top surface'))
        ax.plot([], [], "k--", label=plot_config.get('bottom_label', 'bottom layer'))
        self._finalize_plot(ax, plot_config, 'r', 'dz')
        self._save_plot(fig, plot_config.get('filename'))

    def _generate_integrity_panel(self, plot_config: dict) -> None:
        """End-of-slide integrity panel of one material and layer, a column per load.

        Rows: side view (x-z; grey tip, blue lower, orange upper layer), top
        view of each layer (wrapped into the box), sheet temperature over the
        slide and, for bilayers, the interlayer gap (upper minus lower layer COM
        z) on one y scale across loads. Snapshots are
        ``<run_dir>/L<layer>/data/final_f<F>N_a0.data``; run ids are
        ``<material>/l<layer>/f<F>/a0``. Drawn in plain matplotlib at its own
        size and dpi, not the shared figure settings.
        """
        self._require_all(plot_config, 'material', 'run_dir', 'traces_file', 'filename')
        import matplotlib.pyplot as plt  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        from .integrity import read_snapshot  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        material = self._required(plot_config, 'material')
        layer = int(plot_config.get('layer', 2))
        loads = plot_config.get('loads', [2, 5, 10, 20, 30, 50, 70, 85, 100])
        lay = Path(self._required(plot_config, 'run_dir')) / f"L{layer}"
        prefix = f"{material}/l{layer}/"

        status = {}
        if plot_config.get('status_file'):
            st = self._table(plot_config['status_file'], ['run_id', 'force', 'status'])
            st = st[st['run_id'].str.startswith(prefix)]
            status = dict(zip(st['force'].astype(int), st['status']))
        ex = self._table(self._required(plot_config, 'traces_file'), ['run_id', 'step', 'temp_sheet_3d'])
        ex = ex[ex['run_id'].str.startswith(prefix)].sort_values(["run_id", "step"])
        temps = {r: g["temp_sheet_3d"].to_numpy() for r, g in ex.groupby("run_id")}
        gaps = {}
        if layer == 2:
            ex = ex.assign(gap=ex["comz_L2"] - ex["comz_L1"])
            gaps = {r: g["gap"].to_numpy() for r, g in ex.groupby("run_id")}

        grp = {n: v for n, v in read_type_groups(lay / 'lammps' / 'system.in.settings').items()
               if n in ("layer_1", "layer_2", "tip_all")}
        layers = [n for n in ("layer_2", "layer_1") if n in grp]   # upper first
        rows = ["side"] + layers + ["T"] + (["gap"] if layer == 2 else [])
        ratios = {'side': 1.6, 'layer_2': 2.3, 'layer_1': 2.3, 'T': 0.9, 'gap': 0.9}
        fig, ax = plt.subplots(len(rows), len(loads),
                               figsize=(2.5 * len(loads), 0.4 + sum(ratios[r] for r in rows)),
                               gridspec_kw={'height_ratios': [ratios[r] for r in rows]},
                               squeeze=False)
        colour = {'layer_1': "tab:blue", 'layer_2': "tab:orange"}
        for j, force in enumerate(loads):
            col = {r: ax[i, j] for i, r in enumerate(rows)}
            st = status.get(int(force), "complete" if not status else "not_reached")
            col["side"].set_title(
                f"{force:g} nN" + ("" if st == "complete" else f"\n{_STATUS_TEXT.get(st, st)}"),
                fontsize=10, color="k" if st == "complete" else "firebrick")
            snap = lay / 'data' / f"final_f{force:g}N_a0.data"
            if snap.exists():
                typ, xyz, box = read_snapshot(snap)
                sheet = np.isin(typ, list(set().union(*(grp[n] for n in layers))))
                tip = np.isin(typ, list(grp.get("tip_all", ())))
                side = col["side"]
                side.scatter(xyz[tip, 0], xyz[tip, 2], s=0.3, c="0.75", lw=0, rasterized=True)
                for n in layers:
                    m = np.isin(typ, list(grp[n]))
                    side.scatter(xyz[m, 0], xyz[m, 2], s=0.5, c=colour[n], lw=0, rasterized=True)
                    t = col[n]
                    lx, ly = box["x"][1] - box["x"][0], box["y"][1] - box["y"][0]
                    t.scatter((xyz[m, 0] - box["x"][0]) % lx, (xyz[m, 1] - box["y"][0]) % ly,
                              s=1.2, c=colour[n], lw=0, rasterized=True)
                    t.set_xlim(0, lx)
                    t.set_ylim(0, ly)
                    t.set_aspect("equal")
                    t.tick_params(labelsize=6)
                zc = np.median(xyz[sheet, 2])
                side.set_ylim(zc - 25, zc + 40)
                side.tick_params(labelsize=7)
            else:
                for r in ["side"] + layers:
                    col[r].text(0.5, 0.5, "no snapshot", ha="center", va="center",
                                transform=col[r].transAxes)
                    col[r].set_axis_off()
            rid = f"{prefix}f{force:g}/a0"
            if rid in temps:
                col["T"].plot(temps[rid], lw=0.7, c="k")
            col["T"].axhline(300, c="0.6", lw=0.5, ls="--")
            col["T"].set_ylim(270, 400)
            col["T"].tick_params(labelsize=7)
            if layer == 2:
                if rid in gaps:
                    col["gap"].plot(gaps[rid], lw=0.7, c="k")
                col["gap"].tick_params(labelsize=7)
            if j == 0:
                col["side"].set_ylabel("side: z (Å)", fontsize=8)
                for n in layers:
                    col[n].set_ylabel(
                        ("top view, upper layer" if n == "layer_2" else "top view, lower layer")
                        + "\ny (Å)" if layer == 2 else "top view\ny (Å)", fontsize=8)
                col["T"].set_ylabel("sheet T (K)", fontsize=8)
                if layer == 2:
                    col["gap"].set_ylabel("gap (Å)", fontsize=8)
        if layer == 2:   # one gap scale per material, so a rise compares across loads
            lim = [ax[-1, j].get_ylim() for j in range(len(loads)) if ax[-1, j].lines]
            if lim:
                lo, hi = min(v[0] for v in lim), max(v[1] for v in lim)
                for j in range(len(loads)):
                    ax[-1, j].set_ylim(lo, hi)
        kind = "bilayer" if layer == 2 else "monolayer"
        fig.suptitle(plot_config.get(
            'title',
            f"{material} {kind} (L{layer}): end of slide; grey tip, blue lower, orange upper layer"),
            fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        output_path = self.output_dir / self._required(plot_config, 'filename')
        fig.savefig(output_path, dpi=plot_config.get('dpi', 100))
        plt.close(fig)
        self.written.append(output_path)
        logger.info("Generated plot: %s", output_path)
