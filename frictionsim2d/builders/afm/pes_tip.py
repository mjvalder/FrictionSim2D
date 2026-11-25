"""Static PES scan with an AFM tip (AFM family).

Split out of the former ``builders/pes_scan.py``. The sheet-family scan lives in
``sheetonsheet/pes_sheet.py``; the two shared constants and ``cell_period`` are
in ``builders/pes_common.py``.
"""
import logging

from ...data.models import EV_A_TO_NN
from ..construction.hetero_stack import detect_inplane_repeat
from ..utils.pes_common import DEFAULT_GRID_N, PES_MINIMIZE_COMMAND, cell_period
from .base import AFMSimulation
from .hetero import HeteroAFMSimulation

logger = logging.getLogger(__name__)

PES_TIP_CONTACT_GAP = 3.5


class PESTipSimulation(AFMSimulation):
    """Tip-on-sheet (AFM) surface PES scan (rigid tip, static lateral grid).

    Reuses the full AFM build + indentation (``system.in`` → ``load_<f>N.data``),
    then writes a static grid-scan script as the run phase: the rigid tip is
    held at each grid point while the sheet + substrate relax, recording the
    total energy and the lateral force on the tip.
    """

    FAMILY = 'pes_tip'
    HPC_JOB_PREFIX = 'pes_tip'

    #: write_inputs below replaces the AFM run phase with a single
    #: self-contained scan script and never reads outer_loop -- nothing splits.
    SPLIT_OUTER_LOOPS = frozenset()

    #: The scan templates this family renders, named in the refusal below so the
    #: message points at the file a reader can check the claim against.
    _SCAN_TEMPLATES_WITHOUT_A_FLAKE = "afm/pes_scan.lmp, afm_freestanding/pes_scan.lmp"

    def __init__(self, *args, **kwargs) -> None:
        """Refuse a ``[flake]`` config before anything is built.

        Neither static tip-scan template reads a flake: ``afm/pes_scan.lmp`` and
        ``afm_freestanding/pes_scan.lmp`` have only sub/tip/sheet ``read_data``
        lines. The AFM build in front of them, however, is the full one -- it
        builds the flake component, registers its atom types and writes its
        masses and ``pair_coeff`` into ``system.in.settings`` -- so a
        ``[flake]`` config renders a scan whose box is sized for the flake's
        types and whose potential block stages them, while the flake's atoms are
        never read in. The scan then succeeds and reports the corrugation of the
        BARE surface: a silent wrong result, not a crash.

        Refusing rather than wiring is deliberate. A flake dragged under a
        rastered rigid tip is a physics question nobody has validated here, and
        the honest options are to answer it or to say no; inventing a flake
        block in the scan template would just move the silence.

        Raised in ``__init__`` so it fires the moment ``run/simulate.py``
        selects this builder -- before any directory, structure, LAMMPS call or
        rendered script exists -- and so one definition covers every subclass
        (:class:`HeteroPESTipSimulation`,
        :class:`~frictionsim2d.builders.afm.freestanding.FreestandingPESTipSimulation`),
        both of which resolve ``PESTipSimulation`` first in MRO order.
        """
        super().__init__(*args, **kwargs)
        if getattr(self.config, 'flake', None) is not None:
            raise ValueError(
                f"{type(self).__name__} cannot run a config with a [flake] "
                "section: the static tip PES scan templates "
                f"({self._SCAN_TEMPLATES_WITHOUT_A_FLAKE}) read only the "
                "substrate, tip and sheet. The flake would still be built and "
                "its atom types staged into the potential block, but its atoms "
                "would never enter the system -- the scan would silently "
                "measure the bare surface instead. Remove the [flake] section "
                "from this PES-tip config, or drive the flake with `run afm` / "
                "`run indent`, whose templates do read it."
            )

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

    def _pes_scan_context(self, n_layers: int):
        """The render context and output dir every tip PES scan shares.

        Extracted from this class's and
        :class:`~frictionsim2d.builders.afm.freestanding.FreestandingPESTipSimulation`'s
        ``write_inputs``, which had copy-pasted the whole setup even though the
        latter already subclasses this one -- one of the ``duplicate-code``
        blocks ``docs/superpowers/reviews/builders-afm.md``'s decision row
        AFM-2 tabulated, and the one it called the most clearly avoidable.

        Only the keys BOTH scans set live here. The two genuinely differ
        afterwards, and those differences stay at the call sites: the plain tip
        scan adds ``z_relax``; the freestanding one adds ``eval_mode`` and
        ``md_steps`` and renders a different template. Adding either family's
        extra key here would change the other's rendered script.

        Returns ``(context, out_dir)``.
        """
        context = self.build_render_context(n_layers)
        m = self._scan_cells()
        period_x = cell_period(self.sheet_unit_cell, "x")
        period_y = cell_period(self.sheet_unit_cell, "y")
        self._warn_if_footprint_too_small(m, period_x, period_y)
        context.update({
            # M×M unit cells at grid_n points *per cell*: multiply both the scanned
            # extent and the grid resolution by M (M=1 → unchanged single-cell scan).
            'grid_n': self._grid_n() * m,
            'period_x': period_x * m,
            'period_y': period_y * m,
            # Contact gap (Å). The constant-load campaign runner overrides this per
            # material by patching the rendered script's approach line to the gap that
            # lands the grid-mean load on target (see gap_load_calibration/).
            'contact_gap': PES_TIP_CONTACT_GAP,
            'results_csv': self._layer_path(n_layers, "results", "pes_scan.csv"),
            'ev_a_to_nn': EV_A_TO_NN,
            # Force-converged minimize so per-point energies/forces are clean; the
            # sliding default's relative energy tol is too coarse for the PES.
            'minimization_command': PES_MINIMIZE_COMMAND,
        })
        return context, self.output_dir_layer[n_layers]

    def write_inputs(self, n_layers: int) -> None:
        """Render the self-contained static tip PES scan (single ``slide.in``).

        The scan assembles the tip + sheet + substrate itself; no separate
        ``system.in`` indentation phase is generated. The tip sits at the fixed
        ``PES_TIP_CONTACT_GAP`` (``contact_gap`` in the template); there is no
        in-deck gap calibration. ``pes.tip_load`` is only the target for an
        external gap→load calibration step (see ``documentation/pes_scan.md``).
        The gap→load calibration (``gap_load_calibration``) showed the fixed gap
        sits in the *attractive* well for every material, giving an adhesion, not
        load, descriptor.

        The recorded ``fz`` column reports the actually-achieved load
        (compliant materials that cannot reach the target cap at the firmest gap).
        """
        logger.info("Writing tip PES scan inputs...")

        context, out_dir = self._pes_scan_context(n_layers)
        context['z_relax'] = self._z_relax()

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
    :class:`~frictionsim2d.builders.afm.hetero.HeteroIndentSimulation`:
    :class:`PESTipSimulation` supplies the run-phase identity (the
    self-contained static-scan ``write_inputs``, ``SPLIT_OUTER_LOOPS``),
    :class:`~frictionsim2d.builders.afm.hetero.HeteroAFMSimulation` supplies the
    component build (hetero stack -> in-plane replicate, ``sheet_1``/
    ``sheet_2`` potential split). Only the scan-period population and the
    job-name token are added here.

    MRO: ``HeteroPESTipSimulation -> PESTipSimulation -> HeteroAFMSimulation ->
    AFMSimulation``, so ``write_inputs`` is the static-scan version and
    ``SPLIT_OUTER_LOOPS == frozenset()`` (pinned by
    ``test_hetero_pes_tip_mro_and_split_outer_loops``).

    The scan period is the stack's MINIMAL in-plane repeat, measured off the
    unreplicated stack by
    :func:`~frictionsim2d.builders.construction.hetero_stack.detect_inplane_repeat`
    and stored in ``sheet_unit_cell`` as ``cell_period`` expects. The assembled
    box is not the repeat: each material is tiled to the configured footprint
    before the match runs, so the box is a superlattice of it and scanning the
    box would re-visit registries already sampled. The job name is
    ``pes_tip_<pair>`` through ``_material_id``.
    """

    FAMILY = 'pes_tip'

    def _build_sheet_for_layers(self, build_dir, n_layers):
        """Populate ``sheet_unit_cell`` with the hetero surface's MINIMAL repeat.

        The homogeneous path fills ``sheet_unit_cell`` from ``build_sheet(...,
        unit_cell_out=...)`` with the material's own orthogonalized unit cell —
        the smallest lateral displacement that returns the tip to an equivalent
        registry. The hetero analogue is the assembled stack's minimal in-plane
        repeat, which is generally a SUB-cell of the coincidence cell (see the
        class docstring and
        :func:`~frictionsim2d.builders.construction.hetero_stack.detect_inplane_repeat`). Detection
        runs on
        the UNREPLICATED stack: in-plane replication cannot change the repeat
        unit, and the unreplicated file is much the smaller of the two.
        """
        super()._build_sheet_for_layers(build_dir, n_layers)
        (period_x, period_y), _ = detect_inplane_repeat(self._hetero_stack.data_path)
        self.sheet_unit_cell = {'xlo': 0.0, 'xhi': period_x,
                                'ylo': 0.0, 'yhi': period_y}
