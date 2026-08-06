"""Static potential-energy-surface (PES) scan builders.

These builders reuse the existing sliding-simulation geometry but replace the
dynamic sliding run with a cheap, *static* energy-vs-lateral-position map over
one surface unit cell — the corrugation-barrier descriptor the friction
literature relies on (Prandtl–Tomlinson). See ``PES_SCAN_RECIPE.md``.

Two flavours:

    - :class:`PESSheetSimulation` — the sheet-on-sheet interlayer PES. A frozen
      bilayer whose top layer is grid-scanned laterally, relaxing only the
      out-of-plane coordinate at each grid point.
    - :class:`PESTipSimulation` — the tip-on-sheet (AFM) surface PES. The AFM
      stack is built and indented to a low load exactly as the sliding sim, then
      the rigid tip is grid-scanned laterally while the sheet + substrate relax;
      the lateral force on the tip is recorded alongside the energy.

Both emit their scan as the run-phase script ``lammps/slide.in`` (the framework
production-run slot the recipe swaps the sliding out of), so all existing HPC
array/combined job generation applies unchanged. The scan writes its grid to
``results/pes_scan.csv``; :mod:`src.postprocessing.pes_scan` reduces the grids
to per-material descriptors.

Both are MRO-composed with the heterostructure builders to give two more
flavours, each scanning the *matched interface's coincidence cell* rather than
a single material's own unit cell:

    - :class:`HeteroPESTipSimulation` — the tip scan over a two-material
      heterostructure sheet built by
      :class:`~src.builders.hetero_afm.HeteroAFMSimulation`.
    - :class:`HeteroPESSheetSimulation` — the interlayer scan over a 1+1
      heterostructure bilayer built by
      :class:`~src.builders.hetero_slide.HeteroSheetOnSheetSimulation`.

The sheet flavours share their grid/scan-script logic through
:class:`PESGridMixin`, whose hooks isolate the four places the homogeneous and
hetero builders genuinely differ (template/layer selection, path convention,
scan period, energy normalization).
"""

import logging
from typing import Dict, List, Optional, Union

from ..core.potential_manager import PotentialManager
from ..core.utils import normalize_potential_type
from ..data.models import EV_A_TO_NN
from .afm import AFMSimulation
from .hetero import _detect_atom_style
from .hetero_afm import HeteroAFMSimulation
from .hetero_slide import HeteroSheetOnSheetSimulation
from .sheetonsheet import SheetOnSheetSimulation

logger = logging.getLogger(__name__)

#: Default lateral grid resolution (N in the N×N scan) when a config omits it.
DEFAULT_GRID_N = 12

#: Force-converged minimization for each PES grid point. The sliding-sim default
#: (``minimize 1e-4 1e-8 …``) stops on a *relative* energy tolerance that, on a
#: many-hundred-eV system, is far coarser than the sub-meV corrugation we are
#: resolving — so it quits after 1–2 iterations and buries the signal in noise.
#: etol=0 forces convergence on an absolute force tolerance (size-independent),
#: giving clean per-point energies and lateral forces.
PES_MINIMIZE_COMMAND = "minimize 0.0 1.0e-6 10000 100000"

#: Default contact gap (Å) between the lowest tip atom and the highest sheet atom for
#: the tip scan. The calibration study showed this fixed gap sits in the *attractive*
#: well (fz ≈ −2 to −6 nN) for every material, so on its own it is an adhesion, not a
#: load, descriptor. A constant-load campaign (see gap_load_calibration/) overrides
#: it per material with the gap whose grid-mean load hits ``pes.tip_load``.
PES_TIP_CONTACT_GAP = 3.5

#: Gap search bracket (Å) the constant-load campaign runner uses to calibrate the
#: per-material gap. The lower bound is the firmest gap that stays physical for the
#: bilayer (gaps < 2.5 Å drive some grid points into the atom-crash regime); the
#: upper bound is safely in the attractive tail. Loads of ~2–12 nN sit inside it.
PES_LOAD_GAP_BRACKET = (2.5, 3.75)

#: The static PES bilayer needs exactly two layers: a frozen bottom and a rigid,
#: laterally-scanned top.
PES_SHEET_LAYERS = 2


def _as_float_list(value: Optional[Union[float, List[float]]]) -> List[float]:
    """Normalize a scalar/list sweep value to a list of floats."""
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def _cell_period(unit_cell: Dict[str, float], axis: str) -> float:
    """Return the orthogonalized surface-unit-cell period along ``axis`` (Å)."""
    lo = unit_cell.get(f"{axis}lo")
    hi = unit_cell.get(f"{axis}hi")
    if lo is None or hi is None:
        raise ValueError(
            "Surface unit-cell dimensions unavailable; cannot size the PES grid. "
            "Ensure build_sheet was called with unit_cell_out."
        )
    period = float(hi) - float(lo)
    if period <= 0.0:
        raise ValueError(f"Non-positive surface cell period along {axis}: {period}.")
    return period


def _tile_count(full_dims: Dict[str, float], unit_cell: Dict[str, float]) -> int:
    """Number of surface unit cells tiling the built sheet footprint (x·y)."""
    nx = max(1, round(
        (float(full_dims["xhi"]) - float(full_dims["xlo"])) / _cell_period(unit_cell, "x")
    ))
    ny = max(1, round(
        (float(full_dims["yhi"]) - float(full_dims["ylo"])) / _cell_period(unit_cell, "y")
    ))
    return int(nx * ny)


class PESGridMixin:
    """The sheet-PES grid/scan-script logic, independent of how the bilayer was built.

    Extracted from :class:`PESSheetSimulation` so the same static lateral grid
    can be written over a *heterostructure* bilayer
    (:class:`HeteroPESSheetSimulation`), whose geometry builder
    (:class:`~src.builders.hetero_slide.HeteroSheetOnSheetSimulation`) differs
    from the homogeneous one in four places. This is deliberately NOT a
    drop-in: the mixin owns only what the two flavours share (grid resolution,
    the base/refine script variants, the render-and-write loop, the job-name
    shape, the bilayer layer-count gate) and delegates every point of
    divergence to a hook the concrete class implements.

    **Step-1 hook list** (each verified against both builders, not inferred):

    * :meth:`_scan_template` — the LAMMPS template. The homogeneous
      ``sheetonsheet/pes_scan.lmp`` selects ``layer_1``/``layer_2`` by ATOM
      TYPE (the groups are written into ``system.in.settings`` by
      :class:`~src.core.potential_manager.PotentialManager`). Heterostructure
      layers share atom types, so they can only be selected by z-band REGION —
      the very reason ``hetero_slide`` exists — hence the separate
      ``hetero/pes_scan.lmp``, which emits the same region/group convention as
      ``hetero/slide.lmp`` before the (deliberately duplicated, per this
      package's one-template-per-family convention — cf. ``afm/pes_scan.lmp``
      vs ``afm_freestanding/pes_scan.lmp``) scan loop.
    * :meth:`_scan_data_file` / :meth:`_scan_potential_file` /
      :meth:`_scan_results_path` — the path convention. The homogeneous builder
      is SIM-ROOT-relative (``{relative_run_dir}/build/<sheet>.lmp``, LAMMPS run
      from the simulation root); the hetero builder is RUN-DIR-relative
      (``hetero.data``, ``lammps/system.in.settings``, ``results/...``, LAMMPS
      run with ``cwd = output_dir`` so the settings file's staged
      ``provenance/potentials/...`` prefix resolves). Mixing them yields a
      script LAMMPS cannot open, so neither convention is hardcoded here.
    * :meth:`_scan_periods` — the scan period, i.e. the surface cell the top
      layer is rastered over. Homogeneous: the material's own orthogonalized
      unit cell, filled by ``components.build_sheet(..., unit_cell_out=...)``
      into ``self.sheet_unit_cell``. Hetero: the **coincidence cell** of the
      matched interface (both lattices were strained onto it), which is the
      assembled stack's own periodic box — using either constituent's unit cell
      there would give a plausible-looking but physically wrong PES.
    * :meth:`_scan_n_cells` — the per-surface-cell energy normalization.
      Homogeneous: ``_tile_count(sheet_dims, sheet_unit_cell)``, the number of
      unit cells tiling the built footprint. Hetero: 1 — the box IS one
      coincidence cell (no in-plane replication), and that cell is the
      interface's repeat unit.
    * :meth:`_scan_atom_style` / :meth:`_scan_pot_type` — ``atom_style`` comes
      from the sheet's potential type homogeneously, and from the assembled
      data file (``_detect_atom_style``) for hetero.
    * :meth:`_scan_extra_context` — extra template variables; ``{}``
      homogeneously, the box + z-band layer list for hetero.
    * :meth:`_material_id` — the job-name token (``pes_sheet_<mat>`` vs
      ``pes_sheet_<mat1>+<mat2>``). Defined on
      :class:`~src.builders.sheetonsheet.SheetOnSheetSimulation` and overridden
      by the hetero builder, mirroring the Task-3 AFM-family hook.

    **State the mixin may touch**: only ``self.config`` (``.pes``,
    ``.settings.simulation``) and the :class:`~src.core.simulation_base.SimulationBase`
    services ``render_template`` / ``write_file`` / ``output_dir``. It never
    reads ``self.pm``, ``self.sheet_dims``, ``self.structure_paths``,
    ``self.relative_run_dir`` or any hetero stack attribute directly — those
    are reachable only through the hooks above, which is what keeps it valid on
    both builders (the hetero builder has no ``pm``/``sheet_dims``/
    ``structure_paths`` at all).

    ``SPLIT_OUTER_LOOPS`` is declared here (and so inherited by both concrete
    classes, ahead of any base's value in MRO order): the scan replaces the run
    phase outright and never reads ``outer_loop``, so nothing is ever split.
    The grid variants below are counted separately by
    ``src.core.run._run_scripts_per_build`` (keyed on the ``pes_sheet`` model,
    so the refinement script is predicted for the hetero variant too).
    """

    #: The run phase is a static scan that never reads outer_loop.
    SPLIT_OUTER_LOOPS = frozenset()

    #: A bilayer is enough (and cheapest) for a static PES.
    MIN_LAYERS: int = PES_SHEET_LAYERS

    # ---------------------------------------------------------------- hooks --

    def _scan_template(self) -> str:
        raise NotImplementedError

    def _scan_data_file(self) -> str:
        raise NotImplementedError

    def _scan_potential_file(self) -> str:
        raise NotImplementedError

    def _scan_results_path(self, csv_name: str) -> str:
        raise NotImplementedError

    def _scan_periods(self) -> tuple:
        raise NotImplementedError

    def _scan_n_cells(self) -> int:
        raise NotImplementedError

    def _scan_pot_type(self) -> str:
        raise NotImplementedError

    def _scan_atom_style(self) -> str:
        raise NotImplementedError

    def _scan_extra_context(self) -> Dict:
        """Template variables beyond the shared scan set (default: none)."""
        return {}

    # ------------------------------------------------------------ shared --

    def _layer_count_hint(self) -> str:
        """How the user sets the two layers, for the bilayer-gate message."""
        return f"set [2D] layers = [{PES_SHEET_LAYERS}]"

    def _require_pes_bilayer(self, n_layers: int) -> None:
        """Reject anything but a frozen-bottom + scanned-top bilayer."""
        if n_layers != PES_SHEET_LAYERS:
            raise ValueError(
                "PES sheet scan requires exactly 2 layers (a frozen bottom + a "
                f"scanned top); {self._layer_count_hint()}, got {n_layers}."
            )

    def _get_hpc_job_name(self) -> str:
        """``pes_sheet_<material token>`` (a pair token for the hetero variant)."""
        return f"pes_sheet_{self._material_id()}"

    def _grid_n(self) -> int:
        pes = getattr(self.config, 'pes', None)
        return int(pes.grid_n) if pes is not None else DEFAULT_GRID_N

    def _z_relax(self) -> bool:
        pes = getattr(self.config, 'pes', None)
        return bool(pes.z_relax) if pes is not None else True

    def _grid_variants(self) -> List[tuple]:
        """Return (script_name, csv_name, grid_n) pairs to emit.

        Always emits the base ``slide.in`` at ``grid_n``. When ``pes.grid_n_refine``
        is set, also emits ``slide_refine.in`` at the finer resolution so a
        grid-convergence check (recipe §6) runs in the same job array.
        """
        variants = [("lammps/slide.in", "pes_scan.csv", self._grid_n())]
        pes = getattr(self.config, 'pes', None)
        if pes is not None and pes.grid_n_refine:
            variants.append(
                ("lammps/slide_refine.in", "pes_scan_refine.csv", int(pes.grid_n_refine))
            )
        return variants

    def write_scan_scripts(self) -> None:
        """Render the static interlayer PES scan script(s)."""
        logger.info("Writing sheet PES scan inputs...")

        sim = self.config.settings.simulation
        period_x, period_y = self._scan_periods()
        n_cells = self._scan_n_cells()

        base_context = {
            'atom_style': self._scan_atom_style(),
            'pot_type': self._scan_pot_type(),
            'neighbor_list': sim.neighbor_list,
            'neigh_modify_command': sim.neigh_modify_command,
            'data_file': self._scan_data_file(),
            'potential_file': self._scan_potential_file(),
            'min_style': sim.min_style,
            'minimization_command': PES_MINIMIZE_COMMAND,
            'timestep': sim.timestep,
            'thermo': sim.thermo,
            'period_x': period_x,
            'period_y': period_y,
            'n_cells': n_cells,
            'z_relax': self._z_relax(),
        }
        base_context.update(self._scan_extra_context())

        template = self._scan_template()
        for script_name, csv_name, grid_n in self._grid_variants():
            context = dict(base_context)
            context['grid_n'] = grid_n
            context['results_csv'] = self._scan_results_path(csv_name)
            script = self.render_template(template, context)
            self.write_file(script_name, script)

        logger.info(
            "Sheet PES scan written to %s/lammps/ (grid periods %.3f×%.3f Å, %d cells)",
            self.output_dir, period_x, period_y, n_cells,
        )

    def write_inputs(self) -> None:
        """Run-phase entry point for the homogeneous builder's ``build``."""
        self.write_scan_scripts()

    def write_run_scripts(self) -> None:
        """Run-phase entry point for the hetero builder's ``build`` seam."""
        self.write_scan_scripts()


class PESSheetSimulation(PESGridMixin, SheetOnSheetSimulation):
    """Sheet-on-sheet interlayer PES scan (frozen bilayer, static lateral grid).

    Reuses the sheet-on-sheet geometry/potential setup with a two-layer stack and
    plain real interlayer LJ (no bonds/springs, no driving atom), then writes a
    static grid-scan script instead of the sliding one (:class:`PESGridMixin`,
    whose hooks this class implements with the homogeneous conventions:
    sim-root-relative paths, atom-type layer groups, the material's own surface
    unit cell as the scan period).
    """

    def build(self) -> None:
        self._require_pes_bilayer(self.n_layers)
        # A static PES has no interlayer bonds/springs — just the real LJ that
        # governs the corrugation. Force 'none' regardless of the run settings.
        self.config.settings.simulation.constraint_mode = 'none'
        super().build()

    def _register_drive_atom(self, pm: PotentialManager) -> None:
        """No driving atom in a static scan (keeps types aligned with the data)."""

    # -- PESGridMixin hooks: the homogeneous conventions ---------------------

    def _scan_template(self) -> str:
        return "sheetonsheet/pes_scan.lmp"

    def _scan_data_file(self) -> str:
        return f"{self.relative_run_dir}/build/{self.structure_paths['sheet'].name}"

    def _scan_potential_file(self) -> str:
        return f"{self.relative_run_dir}/lammps/system.in.settings"

    def _scan_results_path(self, csv_name: str) -> str:
        return f"{self.relative_run_dir}/results/{csv_name}"

    def _scan_periods(self) -> tuple:
        return (_cell_period(self.sheet_unit_cell, "x"),
                _cell_period(self.sheet_unit_cell, "y"))

    def _scan_n_cells(self) -> int:
        assert self.sheet_dims is not None
        return _tile_count(self.sheet_dims, self.sheet_unit_cell)

    def _scan_pot_type(self) -> str:
        return normalize_potential_type(self.config.sheet.pot_type)

    def _scan_atom_style(self) -> str:
        return 'charge' if self._scan_pot_type() in ('reaxff', 'reax/c') else 'atomic'


class PESTipSimulation(AFMSimulation):
    """Tip-on-sheet (AFM) surface PES scan (rigid tip, static lateral grid).

    Reuses the full AFM build + indentation (``system.in`` → ``load_<f>N.data``),
    then writes a static grid-scan script as the run phase: the rigid tip is
    held at each grid point while the sheet + substrate relax, recording the
    total energy and the lateral force on the tip.
    """

    #: write_inputs below replaces the AFM run phase with a single
    #: self-contained scan script and never reads outer_loop -- nothing splits.
    SPLIT_OUTER_LOOPS = frozenset()

    def _get_hpc_job_name(self) -> str:
        return f"pes_tip_{self.config.sheet.mat}"

    def _grid_n(self) -> int:
        pes = getattr(self.config, 'pes', None)
        return int(pes.grid_n) if pes is not None else DEFAULT_GRID_N

    def _z_relax(self) -> bool:
        pes = getattr(self.config, 'pes', None)
        return bool(pes.z_relax) if pes is not None else True

    def _scan_cells(self) -> int:
        pes = getattr(self.config, 'pes', None)
        return int(pes.scan_cells) if pes is not None else 1

    def _warn_if_footprint_too_small(self, m: int, period_x: float, period_y: float) -> None:
        """Warn if the tip would overlap its own periodic image over an M-cell scan.

        The rigid tip (diameter 2·r) is rastered by up to M·period from its start; for
        its far edge not to reach the tip's periodic image the box must exceed the tip
        diameter plus the scan travel plus one interaction cutoff. On a too-small box the
        multi-cell PES is contaminated by tip–image forces, not the surface corrugation.
        """
        if m <= 1:
            return
        try:
            box_x = float(self.box_dims['xhi'] - self.box_dims['xlo'])
            box_y = float(self.box_dims['yhi'] - self.box_dims['ylo'])
            r = float(self.config.tip.r)
        except (AttributeError, KeyError, TypeError):
            return
        cutoff = 12.0  # ~LJ interaction range (Å); a conservative periodic-image buffer
        need_x = 2.0 * r + m * period_x + cutoff
        need_y = 2.0 * r + m * period_y + cutoff
        if box_x < need_x or box_y < need_y:
            logger.warning(
                "Tip PES scan_cells=%d may overlap the tip's periodic image: box "
                "%.1f×%.1f Å vs needed ~%.1f×%.1f Å (2·r=%.1f + %d cells + %.0f Å buffer). "
                "Enlarge [2D] x,y or reduce scan_cells.",
                m, box_x, box_y, need_x, need_y, 2.0 * r, m, cutoff,
            )

    def write_inputs(self, n_layers: int) -> None:
        """Render the self-contained static tip PES scan (single ``slide.in``).

        The scan assembles the tip + sheet + substrate itself; no separate
        ``system.in`` indentation phase is generated. Tip placement has two modes:

        - ``pes.tip_load`` set (preferred): the tip self-calibrates its gap by
          bisection until the reaction normal force hits the target load (nN), so
          the scan runs at a firm, repulsive contact in the friction regime. The
          gap→load calibration (``gap_load_calibration``) showed the fixed gap below
          sits in the *attractive* well for every material, giving an adhesion — not
          load — descriptor; the load setpoint is the deployable one.
        - ``pes.tip_load`` unset: fall back to the fixed ``PES_TIP_CONTACT_GAP``.

        Either way the recorded ``fz`` column reports the actually-achieved load
        (compliant materials that cannot reach the target cap at the firmest gap).
        """
        logger.info("Writing tip PES scan inputs...")

        context = self.build_render_context(n_layers)
        rel_layer = str(self.relative_run_dir_layer[n_layers])
        m = self._scan_cells()
        period_x = _cell_period(self.sheet_unit_cell, "x")
        period_y = _cell_period(self.sheet_unit_cell, "y")
        self._warn_if_footprint_too_small(m, period_x, period_y)
        context.update({
            # M×M unit cells at grid_n points *per cell*: multiply both the scanned
            # extent and the grid resolution by M (M=1 → unchanged single-cell scan).
            'grid_n': self._grid_n() * m,
            'period_x': period_x * m,
            'period_y': period_y * m,
            'z_relax': self._z_relax(),
            # Contact gap (Å). The constant-load campaign runner overrides this per
            # material by patching the rendered script's approach line to the gap that
            # lands the grid-mean load on target (see gap_load_calibration/).
            'contact_gap': PES_TIP_CONTACT_GAP,
            'results_csv': f"{rel_layer}/results/pes_scan.csv",
            'ev_a_to_nn': EV_A_TO_NN,
            # Force-converged minimize so per-point energies/forces are clean; the
            # sliding default's relative energy tol is too coarse for the PES.
            'minimization_command': PES_MINIMIZE_COMMAND,
        })

        out_dir = self.output_dir_layer[n_layers]
        scan_script = self.render_template("afm/pes_scan.lmp", context)
        self.write_file("lammps/slide.in", scan_script, out_dir)

        logger.info("Tip PES scan written to %s/lammps/", out_dir)


class HeteroPESTipSimulation(PESTipSimulation, HeteroAFMSimulation):
    """Tip-on-heterostructure static PES scan (two-material coincidence cell).

    Pure MRO composition, mirroring
    :class:`~src.builders.hetero_afm.HeteroIndentSimulation`:
    :class:`PESTipSimulation` supplies the run-phase identity (the
    self-contained static-scan ``write_inputs``, ``SPLIT_OUTER_LOOPS``),
    :class:`~src.builders.hetero_afm.HeteroAFMSimulation` supplies the
    component build (hetero stack -> in-plane replicate, ``sheet_1``/
    ``sheet_2`` potential split). Only the scan-period population and the
    job-name token are added here.

    **Step-1 state-contract audit** (verified by printing/asserting in
    ``tests/test_hetero_pes_tip.py``, not inferred):

    - MRO: ``HeteroPESTipSimulation -> PESTipSimulation -> HeteroAFMSimulation
      -> AFMSimulation -> SimulationBase -> ABC -> object``.
      ``PESTipSimulation`` resolves before ``HeteroAFMSimulation``, so
      ``write_inputs`` is its self-contained static-scan version (not
      ``AFMSimulation``'s indentation + slide), and
      ``SPLIT_OUTER_LOOPS == frozenset()`` (``PESTipSimulation`` declares that
      class attribute directly, ahead of ``HeteroAFMSimulation``'s *inherited*
      ``AFMSimulation.SPLIT_OUTER_LOOPS == {'force', 'scan_angle'}`` in MRO
      order) — correct here too: a static scan replaces the run phase outright
      and never reads ``outer_loop``, hetero or not. Pinned by
      ``test_hetero_pes_tip_mro_and_split_outer_loops`` so a future
      MRO/base-class change cannot silently flip it and make the GUI promise a
      job count the build never produces (the exact bug class fixed elsewhere
      on this branch).
    - Every attribute ``PESTipSimulation.write_inputs``/its helpers read,
      and where the hetero build populates each:

        * ``self.sheet_unit_cell`` (via ``_cell_period``, the scan period) —
          the homogeneous path fills it from ``build_sheet(...,
          unit_cell_out=...)``: one material's own orthogonalized unit cell.
          For a matched heterostructure interface that would be *physically
          wrong* — the scan must repeat over the **coincidence cell** of the
          interface (the cell both lattices were strained onto to match), not
          either constituent's own cell; using a constituent's cell would
          produce a plausible-looking but wrong PES. ``HeteroAFMSimulation.
          _build_hetero_sheet`` already computes this cell (the *unreplicated*
          assembled-stack box, before in-plane tiling) and now exposes it as
          ``self._hetero_coincidence_cell``. Populated here by overriding
          :meth:`_build_sheet_for_layers` to call ``super()`` (dispatches to
          ``HeteroAFMSimulation``'s override, which sets
          ``self._hetero_coincidence_cell``) and then filling
          ``self.sheet_unit_cell = {'xlo': 0.0, 'xhi': lx_cell, 'ylo': 0.0,
          'yhi': ly_cell}`` — the same ``xlo``/``xhi``/``ylo``/``yhi`` shape
          ``_cell_period`` expects, matching the homogeneous
          ``unit_cell_out`` contract exactly.
        * ``self.sheet_dims`` (the built footprint; read by ``build_sheet``'s
          sibling ``PESSheetSimulation._tile_count``, not by
          ``PESTipSimulation.write_inputs`` itself, but by
          ``build_render_context`` for tip centring) — already populated by
          ``_build_hetero_sheet`` from the *replicated* box (distinct from the
          unreplicated coincidence cell above); no override needed.
        * ``self.box_dims`` (read by ``_warn_if_footprint_too_small`` for the
          tip-periodic-image guard) — populated by the inherited
          ``AFMSimulation.build()`` calling ``self._get_substrate_box_dims()``
          after ``_build_components``; neither is overridden by
          ``HeteroAFMSimulation``, so it already works off the hetero
          ``self.sheet_dims``; no override needed.
        * ``self.config.tip.r`` (also read by ``_warn_if_footprint_too_small``)
          — plain config, unaffected by hetero.
        * ``build_render_context`` (called by ``write_inputs``) reads
          ``self.pm[n]``, ``self.sheet_offset``, ``self.lat_c``,
          ``self.z_positions[n]``, ``self.groups[n]``, ``self.sheet_paths[n]``,
          ``self.tip_path``, ``self.sub_path``, ``self.sheet_dims`` — all
          already populated by ``HeteroAFMSimulation``'s overridden
          ``_build_components`` / ``_generate_potentials`` /
          ``_calculate_z_positions`` (documented on that class); no additional
          override needed for the build/render path itself.
    - Job name: ``PESTipSimulation._get_hpc_job_name`` hardcodes
      ``f"pes_tip_{self.config.sheet.mat}"`` (sheet 1's material only — it
      does NOT call ``_material_id()``), so MRO alone would collide two hetero
      pairs sharing material 1 (e.g. ``h-MoS2+h-WS2`` vs ``h-MoS2+h-WSe2``)
      on the identical job name ``pes_tip_h-MoS2``. Overridden below to
      compose ``pes_tip_<pair>`` from
      :meth:`~src.builders.afm.AFMSimulation._material_id` (which
      ``HeteroAFMSimulation`` overrides to the numeric-order pair token) —
      reusing the Task-3 hook built for exactly this, rather than duplicating
      its pair-token logic.
    """

    def _build_sheet_for_layers(self, build_dir, n_layers):
        """Populate ``sheet_unit_cell`` from the coincidence cell after the
        inherited hetero build. See the class docstring's state-contract
        audit for why the coincidence cell (not either constituent's own
        unit cell) is the physically correct scan period."""
        super()._build_sheet_for_layers(build_dir, n_layers)
        lx_cell, ly_cell = self._hetero_coincidence_cell
        self.sheet_unit_cell = {'xlo': 0.0, 'xhi': lx_cell, 'ylo': 0.0, 'yhi': ly_cell}

    def _get_hpc_job_name(self) -> str:
        """``pes_tip_<pair>`` via :meth:`_material_id`, not sheet 1 alone."""
        return f"pes_tip_{self._material_id()}"


class HeteroPESSheetSimulation(PESGridMixin, HeteroSheetOnSheetSimulation):
    """Sheet-on-sheet interlayer PES scan over a heterostructure bilayer.

    A 1+1 stack of two matched materials — the bottom layer frozen, the top
    laterally grid-scanned — assembled by
    :class:`~src.builders.hetero_slide.HeteroSheetOnSheetSimulation` (Zur-McGill
    coincidence match, symmetric strain onto the shared mean cell, real-
    minimization interface spacing) and written as a static scan by
    :class:`PESGridMixin` instead of the dynamic slide.

    This is the composition the mixin exists for; the four seams it crosses
    (each implemented as a hook below, and each a real difference, not a
    naming detail):

    1. **Layer selection.** Heterostructure layers share atom types, so the
       homogeneous template's type-based ``layer_1``/``layer_2`` groups cannot
       exist. ``hetero/pes_scan.lmp`` selects them by z-band REGION, using the
       same ``compute_layer_zbands`` bands and the same
       ``region``/``group ... region`` convention as ``hetero/slide.lmp``.
    2. **Path convention.** Every path is RUN-DIR-RELATIVE (``hetero.data``,
       ``lammps/system.in.settings``, ``results/...``), matching the hetero
       slide — the stack is built into the run dir precisely so the settings
       file's staged ``provenance/potentials/...`` prefix resolves with
       ``cwd = output_dir``. The homogeneous sim-root-relative paths would not
       open.
    3. **Scan period.** The matched interface's COINCIDENCE cell, i.e. the
       assembled stack's own in-plane box (the hetero sheet-on-sheet stack is
       not replicated in-plane, so box == cell). Either constituent's own unit
       cell is NOT a periodicity of the interface — scanning it would give a
       plausible-looking but physically wrong PES. Correspondingly the energy
       is normalized per coincidence cell (``n_cells == 1``).
    4. **Layer gate.** The dynamic slide needs >= 4 layers (fixed bottom,
       thermostatted middle, driven top across two materials); a static PES
       needs exactly 2 (one per material). ``MIN_LAYERS`` is relaxed HERE only
       — :class:`~src.builders.hetero_slide.HeteroSheetOnSheetSimulation` keeps
       its 4-layer gate and its message, so a 1+1 *dynamic* slide is still
       refused.

    ``SPLIT_OUTER_LOOPS`` resolves to ``frozenset()`` from
    :class:`PESGridMixin` (which precedes the hetero builder in MRO); both
    candidates are ``frozenset()`` anyway, and the static scan never reads
    ``outer_loop``. Pinned by ``tests/test_hetero_pes_sheet.py``.
    """

    #: Exactly one layer per material: frozen bottom + scanned top.
    MIN_LAYERS: int = PES_SHEET_LAYERS

    def build(self) -> None:
        self._require_pes_bilayer(self.n_layers)
        # A static PES has no interlayer bonds/springs — just the real (hybrid)
        # interaction that governs the corrugation. The hetero slide already
        # renders constraint_mode 'none'; force it on the config too so any
        # settings-driven branch below sees the static intent.
        self.config.settings.simulation.constraint_mode = 'none'
        super().build()

    def _layer_count_hint(self) -> str:
        return "set each material's [2D-N] layers = [1]"

    def _reserve_drive_virtual_type(self) -> bool:
        """A static scan has no drive, so no virtual atom type is reserved."""
        return False

    # -- PESGridMixin hooks: the hetero conventions --------------------------

    def _scan_template(self) -> str:
        return "hetero/pes_scan.lmp"

    def _scan_data_file(self) -> str:
        """Run-dir-relative, like the hetero slide (LAMMPS runs from the run dir)."""
        return "hetero.data"

    def _scan_potential_file(self) -> str:
        return "lammps/system.in.settings"

    def _scan_results_path(self, csv_name: str) -> str:
        return f"results/{csv_name}"

    def _scan_periods(self) -> tuple:
        """The matched interface's coincidence cell (== the assembled box).

        Set by :meth:`HeteroSheetOnSheetSimulation.build` from the assembled
        ``hetero.data``; unreplicated, so it is the interface's true in-plane
        periodicity. A triclinic cell is refused: the scan grid steps along
        +x/+y, so a tilted cell's grid points would not be lattice
        translations of the interface.
        """
        if self._hetero_coincidence_cell is None:
            raise ValueError(
                "Hetero PES scan period unavailable: the coincidence cell is "
                "populated by the hetero stack assembly in build(); "
                "write_run_scripts() cannot be called before it."
            )
        lx, ly = self._hetero_coincidence_cell
        if abs(self._hetero_cell_xy_tilt) > 1e-6:
            raise ValueError(
                f"Hetero coincidence cell is triclinic (xy tilt="
                f"{self._hetero_cell_xy_tilt:.4f} A); an axis-aligned (dx, dy) "
                "PES grid over it would not step by lattice translations. Use a "
                "pair whose coincidence cell canonicalises orthogonal."
            )
        if lx <= 0.0 or ly <= 0.0:
            raise ValueError(
                f"Non-positive hetero coincidence cell: {lx} x {ly} A."
            )
        return (lx, ly)

    def _scan_n_cells(self) -> int:
        """One: the box holds exactly one coincidence cell.

        The energy is therefore reported per interface repeat unit. Dividing by
        a constituent's unit-cell count instead would mix two different
        periodicities and mislabel the barrier.
        """
        return 1

    def _scan_pot_type(self) -> str:
        """``sw`` — the hetero spine is a ``pair_style hybrid`` of ``sw`` blocks
        plus UFF-mixed ``lj/cut`` cross-terms, exactly as ``hetero/slide.lmp``
        hardcodes. Only used to gate the ReaxFF ``qeq`` fix, which does not
        apply here."""
        return "sw"

    def _scan_atom_style(self) -> str:
        """Detected from the assembled data file, like the hetero slide's."""
        assert self._stack is not None
        return _detect_atom_style(self._stack.data_path)

    def _scan_extra_context(self) -> Dict:
        """Box + z-band layer groups (hetero layers share atom types)."""
        context = dict(self._hetero_box)
        context['layers'] = self._layer_bands
        return context
