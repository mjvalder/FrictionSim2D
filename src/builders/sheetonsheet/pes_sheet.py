"""Static PES scan over a sheet stack (sheet family).

Split out of the former ``builders/pes_scan.py``. The tip scan lives in
``afm/pes_tip.py``; the two shared constants and ``_cell_period`` are in
``builders/pes_common.py``.

``PESGridMixin`` is here rather than shared because only the two sheet-family
scans use it -- the tip scan never did.
"""
import logging
from typing import Dict, List

from ...potentials.manager import PotentialManager
from ...potentials.utils import normalize_potential_type
from ..construction.hetero_stack import detect_inplane_repeat
from ..utils.pes_common import DEFAULT_GRID_N, PES_MINIMIZE_COMMAND, _cell_period
from .base import (CUTOFF_NOT_APPLICABLE, SheetOnSheetSimulation,
                   write_sheet_provenance)
from .hetero import HeteroSheetOnSheetSimulation

logger = logging.getLogger(__name__)

PES_SHEET_LAYERS = 2


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
    (:class:`~src.builders.sheetonsheet.hetero.HeteroSheetOnSheetSimulation`) differs
    from the homogeneous one in four places. This is deliberately NOT a
    drop-in: the mixin owns only what the two flavours share (grid resolution,
    the base/refine script variants, the render-and-write loop, the job-name
    shape, the bilayer layer-count gate) and delegates every point of
    divergence to a hook the concrete class implements.

    **Step-1 hook list** (each verified against both builders, not inferred):

    * :meth:`_scan_template` — the LAMMPS template. The homogeneous
      ``sheetonsheet/pes_scan.lmp`` selects ``layer_1``/``layer_2`` by ATOM
      TYPE (the groups are written into ``system.in.settings`` by
      :class:`~src.potentials.manager.PotentialManager`). Heterostructure
      layers share atom types, so they can only be selected by z-band REGION —
      the very reason ``hetero_slide`` exists — hence the separate
      ``hetero/pes_scan.lmp``, which emits the same region/group convention as
      ``hetero/slide.lmp`` before the (deliberately duplicated, per this
      package's one-template-per-family convention — cf. ``afm/pes_scan.lmp``
      vs ``afm_freestanding/pes_scan.lmp``) scan loop.
    * :meth:`_scan_data_file` / :meth:`_scan_potential_file` /
      :meth:`_scan_results_path` — *which* files, not which convention. Both
      families are SIM-ROOT-relative (``{relative_run_dir}/...``, LAMMPS run
      from the simulation root by the generated job scripts), but they name
      different artefacts: the homogeneous builder's ``build/<sheet>.lmp``
      versus the hetero builder's assembled ``hetero.data``. Hardcoding either
      set yields a script LAMMPS cannot open, so neither is hardcoded here.
    * :meth:`_scan_periods` — the scan period, i.e. the surface cell the top
      layer is rastered over. Homogeneous: the material's own orthogonalized
      unit cell, filled by ``components.build_sheet(..., unit_cell_out=...)``
      into ``self.sheet_unit_cell``. Hetero: the matched interface's **minimal
      in-plane repeat unit**, measured off the assembled stack by
      :func:`~src.builders.construction.hetero_stack.detect_inplane_repeat` — which is generally a
      SUB-cell of the coincidence cell, because each material is tiled to the
      configured footprint before the match runs.
    * :meth:`_scan_n_cells` — the per-surface-cell energy normalization.
      Homogeneous: ``_tile_count(sheet_dims, sheet_unit_cell)``, the number of
      unit cells tiling the built footprint. Hetero: the number of interface
      repeat units tiling the assembled box, from the same detection. Both
      report per surface unit cell, so the two families' barriers are
      comparable and neither depends on the configured footprint.
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
    ``.settings.simulation``) and the :class:`~src.builders.simulation_base.SimulationBase`
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
    ``src.run.expand._run_scripts_per_build`` (keyed on the ``pes_sheet`` model,
    so the refinement script is predicted for the hetero variant too).
    """

    #: The run phase is a static scan that never reads outer_loop.
    SPLIT_OUTER_LOOPS = frozenset()

    #: A bilayer is enough (and cheapest) for a static PES.
    MIN_LAYERS: int = PES_SHEET_LAYERS

    #: The two sheet scan templates, named in the refusal below so the claim is
    #: checkable rather than asserted: both read ONE pre-assembled data file.
    _SCAN_TEMPLATES_WITHOUT_A_FLAKE = (
        "sheetonsheet/pes_scan.lmp, hetero/pes_scan.lmp"
    )

    def __init__(self, *args, **kwargs) -> None:
        """Refuse a ``[Flake]`` config before anything is built.

        The sheet-family counterpart of
        :meth:`~src.builders.afm.pes_tip.PESTipSimulation.__init__`'s guard, and
        the same defect one model over: ``src/run/simulate.py`` routes on
        ``model == 'pes_sheet'`` BEFORE it reaches the ``[Flake]`` check, which
        is guarded on ``model == 'sheetonsheet'`` and so never sees a pes-sheet
        config. The identical config therefore builds a flake slide under
        ``run sheetonsheet`` and a flakeless bilayer scan under
        ``run pes-sheet``.

        It is dropped even EARLIER than on the tip path, and that makes it
        harder to notice rather than easier: neither
        :class:`~src.builders.sheetonsheet.SheetOnSheetSimulation` nor this
        module has a single ``flake`` reference, so nothing is built, nothing is
        staged into the potential block, and no unread artefact is left in the
        tree. The scan simply reports the corrugation of the plain bilayer.

        Refusing rather than wiring, for the reason the tip path settled on:
        both scan templates read one pre-assembled ``{{ data_file }}``, so
        supporting a flake means answering what a flake ON a statically scanned
        bilayer even means -- unvalidated physics. Adding a flake block to the
        templates would move the silence, not remove it.

        Defined on the mixin so one definition covers both concrete classes:
        :class:`PESGridMixin` precedes ``SheetOnSheetSimulation`` and
        ``HeteroSheetOnSheetSimulation`` in their MROs (pinned by
        ``test_hetero_pes_sheet.py::test_hetero_pes_sheet_mro_and_split_outer_loops``),
        so ``super().__init__`` continues to whichever base the concrete class
        has. It fires the moment ``run/simulate.py`` selects the builder --
        before any directory, structure, LAMMPS call or rendered script exists.
        """
        super().__init__(*args, **kwargs)
        if getattr(self.config, 'flake', None) is not None or getattr(
                self.config, 'flake_sheets', None):
            raise ValueError(
                f"{type(self).__name__} cannot run a config with a [Flake] "
                f"section: the sheet PES scan templates "
                f"({self._SCAN_TEMPLATES_WITHOUT_A_FLAKE}) read a single "
                "pre-assembled bilayer, and no sheet-family builder reads a "
                "flake at all -- the flake would be dropped without trace and "
                "the scan would silently report the corrugation of the plain "
                "bilayer instead. Remove the [Flake] section from this "
                "pes-sheet config, or drive the flake with `run sheetonsheet`, "
                "whose builder does read it."
            )

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

        # One record per build, not per grid variant: slide.in and
        # slide_refine.in are two resolutions of the same scan, in the same run
        # directory, from the same config.
        #
        # A static scan has no thermostat and no contact cutoff -- it minimises
        # at each grid point and computes no coordination -- but
        # ProvenanceRecord requires a float for contact_cutoff, so it gets 0.0
        # and a source that says the number is meaningless rather than a
        # plausible-looking one a reader would trust.
        write_sheet_provenance(
            self,
            contact_cutoff=0.0,
            cutoff_source=CUTOFF_NOT_APPLICABLE,
            thermostat_scheme='none_static_scan',
        )

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

    FAMILY = 'pes_sheet'

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


class HeteroPESSheetSimulation(PESGridMixin, HeteroSheetOnSheetSimulation):
    """Sheet-on-sheet interlayer PES scan over a heterostructure bilayer.

    A 1+1 stack of two matched materials — the bottom layer frozen, the top
    laterally grid-scanned — assembled by
    :class:`~src.builders.sheetonsheet.hetero.HeteroSheetOnSheetSimulation` (Zur-McGill
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
    3. **Scan period.** The interface's MINIMAL in-plane repeat unit, measured
       off the assembled ``hetero.data`` by
       :func:`~src.builders.construction.hetero_stack.detect_inplane_repeat`, with the energy
       normalized per repeat unit (``n_cells`` = how many tile the box).

       The coincidence cell is *not* that repeat unit in general, and on the
       shipped fixtures it is not one in practice either: ``build_monolayer``
       tiles each material to the configured x/y footprint before the match
       runs, so a near-lattice-matched pair (MoS2/WS2 here) matches with
       (near-)identity matrices on already-tiled cells and the assembled box
       comes out a 4x2 (footprint 12 Å) or 4x3 (footprint 14 Å) superlattice of
       the true repeat. Both constituents' strained unit cells ARE therefore a
       periodicity of this interface — an earlier revision of this docstring
       asserted the opposite, and the measurement refutes it: the assembled
       stack is invariant under ``(Lx/4, 0)`` and ``(0, Ly/2)`` to within
       2e-7 Å. Treating the box as the cell reported a per-BOX energy (8x/12x
       too large) that changed with the configured footprint, which no physical
       barrier may do.
    4. **Layer gate.** The dynamic slide needs >= 4 layers (fixed bottom,
       thermostatted middle, driven top across two materials); a static PES
       needs exactly 2 (one per material). ``MIN_LAYERS`` is relaxed HERE only
       — :class:`~src.builders.sheetonsheet.hetero.HeteroSheetOnSheetSimulation` keeps
       its 4-layer gate and its message, so a 1+1 *dynamic* slide is still
       refused.

    ``SPLIT_OUTER_LOOPS`` resolves to ``frozenset()`` from
    :class:`PESGridMixin` (which precedes the hetero builder in MRO); both
    candidates are ``frozenset()`` anyway, and the static scan never reads
    ``outer_loop``. Pinned by ``tests/builders/test_hetero_pes_sheet.py``.
    """

    #: 'pes_sheet', not 'hetero' -- same reasoning as ``PESSheetSimulation``:
    #: this class's results files follow the PES-scan schema, not the dynamic
    #: hetero slide's, regardless of the hetero geometry underneath.
    FAMILY = 'pes_sheet'

    #: Exactly one layer per material: frozen bottom + scanned top.
    MIN_LAYERS: int = PES_SHEET_LAYERS

    #: Memo for :meth:`_interface_repeat` (the scan writes up to two grid
    #: variants and must not re-measure the stack for each). Declared here
    #: rather than in an ``__init__``: this class defines none, and ``None``
    #: is exactly the "not measured yet" sentinel the reader tests for.
    _interface_repeat_cache = None

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
        """Sim-root-relative, like the hetero slide this builder inherits from.

        The assembled stack lives in the run dir, but the generated HPC job
        scripts run LAMMPS from the simulation root, so the script names it
        from there (:meth:`HeteroSheetOnSheetSimulation._run_dir_path`).
        """
        return self._run_dir_path("hetero.data")

    def _scan_potential_file(self) -> str:
        return self._run_dir_path("lammps", "system.in.settings")

    def _scan_results_path(self, csv_name: str) -> str:
        return self._run_dir_path("results", csv_name)

    def _interface_repeat(self) -> tuple:
        """``((period_x, period_y), (nx, ny))`` for the assembled interface.

        The coincidence cell is NOT in general the interface's minimal repeat.
        ``build_monolayer`` tiles each material to the configured x/y footprint
        *before* the match runs, so a near-lattice-matched pair matches with
        (near-)identity matrices on cells that are already an n x m tiling of
        their own primitive — leaving the assembled box a superlattice of the
        actual repeat unit. Measured directly off ``hetero.data`` by
        :func:`~src.builders.construction.hetero_stack.detect_inplane_repeat` (see its docstring
        for the tolerance and why detection beats the algebraic route), cached
        because the scan writes up to two grid variants.
        """
        if getattr(self, "_interface_repeat_cache", None) is None:
            assert self._stack is not None
            self._interface_repeat_cache = detect_inplane_repeat(self._stack.data_path)
        return self._interface_repeat_cache

    def _guard_scannable_cell(self) -> None:
        """Refuse a cell an axis-aligned (dx, dy) grid cannot step over."""
        if self._hetero_coincidence_cell is None:
            raise ValueError(
                "Hetero PES scan period unavailable: the coincidence cell is "
                "populated by the hetero stack assembly in build(); "
                "write_run_scripts() cannot be called before it."
            )
        if abs(self._hetero_cell_xy_tilt) > 1e-6:
            raise ValueError(
                f"Hetero coincidence cell is triclinic (xy tilt="
                f"{self._hetero_cell_xy_tilt:.4f} A); an axis-aligned (dx, dy) "
                "PES grid over it would not step by lattice translations. Use a "
                "pair whose coincidence cell canonicalises orthogonal."
            )
        lx, ly = self._hetero_coincidence_cell
        if lx <= 0.0 or ly <= 0.0:
            raise ValueError(
                f"Non-positive hetero coincidence cell: {lx} x {ly} A."
            )

    def _scan_periods(self) -> tuple:
        """The interface's MINIMAL in-plane repeat, not the assembled box.

        Scanning the whole box re-measures the same registries n x m times over
        and resolves only ``grid_n / n`` distinct x-registries; scanning the
        repeat unit spends every grid point on a distinct registry. A triclinic
        cell is refused: the grid steps along +x/+y, so a tilted cell's grid
        points would not be lattice translations of the interface.
        """
        self._guard_scannable_cell()
        (period_x, period_y), _ = self._interface_repeat()
        return (period_x, period_y)

    def _scan_n_cells(self) -> int:
        """How many interface repeat units tile the assembled box.

        The template divides the box's total ``pe`` by this, so the CSV carries
        an energy PER INTERFACE REPEAT UNIT — the same normalization the
        homogeneous path gets from ``_tile_count``, and the only one that is
        footprint-independent. The old hardcoded ``1`` reported a per-box
        energy: on the shipped fixture that is 8x (footprint 12 Å) or 12x
        (footprint 14 Å) too large, so the *same* physical pair produced
        barriers differing by 1.5x purely from the configured footprint, and
        every hetero barrier landed ~an order of magnitude above the
        homogeneous ones ``src.postprocessing.pes_scan`` compares them with.
        """
        _, (nx, ny) = self._interface_repeat()
        return int(nx * ny)

    def _scan_pot_type(self) -> str:
        """``sw`` — the hetero spine is a ``pair_style hybrid`` of ``sw`` blocks
        plus UFF-mixed ``lj/cut`` cross-terms, exactly as ``hetero/slide.lmp``
        hardcodes. Only used to gate the ReaxFF ``qeq`` fix, which does not
        apply here."""
        return "sw"

    def _scan_atom_style(self) -> str:
        """The style ``build`` already detected from the assembled data file.

        ``HeteroSheetOnSheetSimulation.build`` runs ``_detect_atom_style`` on
        ``hetero.data`` and stores the result in ``self._hetero_atom_style``
        before handing over to the run-phase seam, so re-reading the file here
        would only re-derive the identical value — a second source of truth
        that could drift, plus a needless file read.
        """
        return self._hetero_atom_style

    def _scan_extra_context(self) -> Dict:
        """Box + z-band layer groups (hetero layers share atom types)."""
        context = dict(self._hetero_box)
        context['layers'] = self._layer_bands
        return context
