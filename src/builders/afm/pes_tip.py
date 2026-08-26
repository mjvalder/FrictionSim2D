"""Static PES scan with an AFM tip (AFM family).

Split out of the former ``builders/pes_scan.py``. The sheet-family scan lives in
``sheetonsheet/pes_sheet.py``; the two shared constants and ``_cell_period`` are
in ``builders/pes_common.py``.
"""
import logging
from typing import Dict, List, Optional, Union

from ...potentials.manager import PotentialManager
from ...potentials.utils import normalize_potential_type
from ...data.models import EV_A_TO_NN
from ..construction.hetero import detect_inplane_repeat
from ..utils.pes_common import DEFAULT_GRID_N, PES_MINIMIZE_COMMAND, _cell_period
from .base import AFMSimulation
from .hetero import HeteroAFMSimulation

logger = logging.getLogger(__name__)

PES_TIP_CONTACT_GAP = 3.5

PES_LOAD_GAP_BRACKET = (2.5, 3.75)


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

        # A tip PES scan is a build like any other and needs the same audit
        # record. It was missing only because this method replaces
        # AFMSimulation.write_inputs, which is where the single call site
        # lived -- not because a scan has nothing to record. The context is
        # the one build_render_context already produced above.
        self._write_provenance(n_layers, context, out_dir)

        logger.info("Tip PES scan written to %s/lammps/", out_dir)


class HeteroPESTipSimulation(PESTipSimulation, HeteroAFMSimulation):
    """Tip-on-heterostructure static PES scan (matched interface's repeat unit).

    Pure MRO composition, mirroring
    :class:`~src.builders.afm.hetero.HeteroIndentSimulation`:
    :class:`PESTipSimulation` supplies the run-phase identity (the
    self-contained static-scan ``write_inputs``, ``SPLIT_OUTER_LOOPS``),
    :class:`~src.builders.afm.hetero.HeteroAFMSimulation` supplies the
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
          unit_cell_out=...)``: one material's own orthogonalized unit cell,
          i.e. the smallest lateral displacement that returns the tip to an
          equivalent registry. The hetero analogue is the assembled stack's
          MINIMAL in-plane repeat, measured off the *unreplicated* stack by
          :func:`~src.builders.construction.hetero.detect_inplane_repeat` and filled here by
          overriding :meth:`_build_sheet_for_layers` (which calls ``super()``,
          dispatching to ``HeteroAFMSimulation``'s override) into the same
          ``xlo``/``xhi``/``ylo``/``yhi`` shape ``_cell_period`` expects.

          An earlier revision used ``self._hetero_coincidence_cell`` (the whole
          unreplicated assembled box) here on the premise that the coincidence
          cell is the interface's periodicity. Measurement refutes that for the
          shipped fixture: each material is tiled to the configured x/y
          footprint *before* the match runs, so the pair matches with
          (near-)identity matrices and the box is a 4x2 superlattice of the
          real repeat (3.185 x 5.517 Å). Scanning the box wasted 4x2 of the
          grid re-visiting registries it had already sampled. Unlike the sheet
          flavour there is no energy-normalization consequence here:
          ``afm/pes_scan.lmp`` records the total energy and the lateral force
          on the tip, with no per-cell divisor.
        * ``self.sheet_dims`` (the built footprint; read by ``build_sheet``'s
          sibling ``PESSheetSimulation._tile_count``, not by
          ``PESTipSimulation.write_inputs`` itself, but by
          ``build_render_context`` for tip centring) — already populated by
          ``_build_hetero_sheet`` from the *replicated* box (distinct from both
          the coincidence cell and the repeat unit above); no override needed.
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
        """Populate ``sheet_unit_cell`` with the hetero surface's MINIMAL repeat.

        The homogeneous path fills ``sheet_unit_cell`` from ``build_sheet(...,
        unit_cell_out=...)`` with the material's own orthogonalized unit cell —
        the smallest lateral displacement that returns the tip to an equivalent
        registry. The hetero analogue is the assembled stack's minimal in-plane
        repeat, which is generally a SUB-cell of the coincidence cell (see the
        class docstring and
        :func:`~src.builders.construction.hetero.detect_inplane_repeat`). Detection runs on
        the UNREPLICATED stack: in-plane replication cannot change the repeat
        unit, and the unreplicated file is much the smaller of the two.
        """
        super()._build_sheet_for_layers(build_dir, n_layers)
        (period_x, period_y), _ = detect_inplane_repeat(self._hetero_stack.data_path)
        self.sheet_unit_cell = {'xlo': 0.0, 'xhi': period_x,
                                'ylo': 0.0, 'yhi': period_y}

    def _get_hpc_job_name(self) -> str:
        """``pes_tip_<pair>`` via :meth:`_material_id`, not sheet 1 alone."""
        return f"pes_tip_{self._material_id()}"
