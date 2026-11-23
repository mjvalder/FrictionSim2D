"""Hetero sheet-on-sheet slide builder (consumes construction/hetero_stack.py)."""
import logging
from pathlib import Path
from typing import List, Dict, Optional

from ...core.config import SheetOnSheetSimulationConfig
from ..simulation_base import (SimulationBase, drive_spring, halt_temperature,
                               mlip_template_context)
from .base import (sheet_obs_ctx, sheet_thermostat_scheme,
                   write_sheet_provenance)
from ...data.models import EV_A_TO_NN, EV_A3_TO_GPA, NM_TO_EV_A2
from ...interfaces.ase import read_lammps_data
from ..utils.z_vacuum import SHEET_VIRTUAL_ATOM_HEIGHT, VIRTUAL_ATOM_ROOM, ZBox
from ..utils.interlayer import (check_interlayer_box, interlayer_style,
                                interlayer_template_context)
from ..construction.hetero_stack import (
    _detect_atom_style,
    build_hetero_structure,
    register_hetero_potentials,
)

logger = logging.getLogger(__name__)

PAD = 1.0  # Å pad above/below each layer's atom z-range


def compute_layer_zbands(data_path, layers) -> List[Dict]:
    """Compute per-layer z-bands from each layer's KNOWN placement.

    Each layer in ``layers`` (a
    :class:`~frictionsim2d.builders.construction.hetero_stack.HeteroStackLayer`)
    records ``.source`` (the per-material supercell data file it was read
    from) and ``.z`` (the z-shift applied via
    ``read_data ... shift 0 0 z``, see templates/hetero/stack_hetero.lmp).
    So the placed atoms of a layer occupy exactly
    ``[native_zmin + z, native_zmax + z]``, where ``native_z`` comes from
    reading ``.source``. This is exact and independent of gap heuristics, so
    (unlike clustering on the (n-1) largest z-gaps) it cannot be fooled by a
    puckered layer whose internal sub-plane gap exceeds an inter-layer gap
    (e.g. black phosphorus, GeS).

    Multiple layers of the SAME material share one ``.source`` file but have
    DIFFERENT ``.z``, so their bands are still distinct.

    Returns one ``{'idx', 'zlo', 'zhi'}`` per layer (idx 1..n bottom->top).

    Raises:
        ValueError: If ``data_path``'s actual atom z-coordinates disagree with
            the bands derived from known placement (i.e. some atom falls in
            zero or more than one band). This is a self-check that makes a
            silent mis-grouping impossible -- it fails loud instead.
    """
    native_zrange: Dict[str, tuple] = {}
    placed = []
    for layer in layers:
        key = str(layer.source)
        if key not in native_zrange:
            zz = read_lammps_data(key).positions[:, 2]
            native_zrange[key] = (float(zz.min()), float(zz.max()))
        zmin, zmax = native_zrange[key]
        zlo, zhi = zmin + layer.z - PAD, zmax + layer.z + PAD
        center = 0.5 * (zmin + zmax) + layer.z
        placed.append((center, zlo, zhi))

    placed.sort(key=lambda t: t[0])
    bands = [{"idx": i + 1, "zlo": zlo, "zhi": zhi} for i, (_, zlo, zhi) in enumerate(placed)]

    # Self-check (fail loud): every atom in the actual assembled data file
    # must fall in exactly one band. Catches any placement/write discrepancy
    # instead of silently mis-grouping.
    z = read_lammps_data(data_path).positions[:, 2]
    for zi in z:
        hits = [b for b in bands if b["zlo"] <= zi <= b["zhi"]]
        if len(hits) != 1:
            raise ValueError(
                f"compute_layer_zbands: atom at z={zi:.4f} falls in {len(hits)} "
                f"bands (expected exactly 1); bands={bands}. The assembled "
                f"{data_path} disagrees with the known layer placement "
                f"(source native z-range + z-shift) -- check read_data shift "
                f"semantics or PAD={PAD}."
            )
    return bands


class HeteroSheetOnSheetSimulation(SimulationBase):
    """Builder for a periodic heterostructure sheet-on-sheet slide.

    Turns a two-material :class:`SheetOnSheetSimulationConfig` into a runnable
    periodic hetero slide by combining the Phase-B structure builder
    (:func:`~frictionsim2d.builders.construction.hetero_stack.build_hetero_structure`, which
    assembles
    ``hetero.data`` + ``lammps/system.in.settings`` and stages the potentials)
    with the D1 slide template (``templates/hetero/slide.lmp``).

    The assembled N-layer stack (N = sum of each material's layer count, >= 4)
    is a fixed bottom / thermostatted middle / driven top slide, exactly like the
    homogeneous :class:`~frictionsim2d.builders.sheetonsheet.SheetOnSheetSimulation`, but
    the layer groups are defined by z-slab region (hetero layers share atom
    types, so type-based groups are impossible) and the interlayer friction is
    read from the bottom-layer reaction force (many-body ``sw`` hybrid, no
    ``compute group/group``).

    Every script path is root-relative, as for every builder (see
    :attr:`~frictionsim2d.builders.simulation_base.SimulationBase.relative_run_dir`).
    """

    FAMILY = 'hetero'

    #: A dynamic hetero slide needs a fixed bottom, a thermostatted middle and a
    #: driven top across TWO materials -> at least 2 + 2 layers.
    MIN_LAYERS: int = 4

    #: This builder writes a single lammps/slide.in and never reads outer_loop
    #: (pressure/scan_speed/scan_angle all stay in-script), so a config that
    #: sets outer_loop produces one job here, not one per value. Explicit (the
    #: base default is the same) because a twist or a second material silently
    #: reroutes a sheet-on-sheet config here.
    SPLIT_OUTER_LOOPS = frozenset()

    #: The slide is loaded by ``[general] pressure``; with none it would run at
    #: 0 GPa and report a meaningless COF, so ``frictionsim2d.run.expand.missing_pressure``
    #: refuses such a config before anything is built.
    REQUIRES_PRESSURE: bool = True

    def __init__(self, config: SheetOnSheetSimulationConfig, output_dir: str,
                 config_path: Optional[str] = None):
        super().__init__(config, output_dir, config_path=config_path)
        self.config: SheetOnSheetSimulationConfig = config
        # Assembled-stack state, populated by build() and read by write_run_scripts().
        self._stack = None
        self._layer_bands: List[Dict] = []
        self._hetero_box: Dict[str, float] = {}
        #: (lx, ly) of the assembled stack's box. The stack is not replicated
        #: in-plane, so this is the matched interface's coincidence cell: the
        #: static PES scan's period.
        self._hetero_coincidence_cell: Optional[tuple] = None
        #: xy tilt of the assembled box; a triclinic cell cannot be scanned by
        #: an axis-aligned (dx, dy) grid, so the PES variant refuses it.
        self._hetero_cell_xy_tilt: float = 0.0
        self._hetero_atom_style: str = "atomic"
        self._hetero_lat_c: Optional[float] = None
        self._hetero_num_atom_types: int = 0
        self._hetero_virtual_type_reserved: bool = False

    def _material_id(self) -> str:
        """Every configured material, numeric ``[2D-N]`` order, joined with ``+``.

        Same convention as :func:`frictionsim2d.run.naming.material_token`.
        """
        return '+'.join(sheet.mat for sheet in self.config.sheets)

    @property
    def n_layers(self) -> int:
        """Total layer count = sum of each material's single layer count.

        Each material's ``layers`` is a single-element list (e.g. ``[2]``); a
        2+2 stack is 4 layers. Mirrors
        :attr:`SheetOnSheetSimulation.n_layers`' single-count requirement.
        """
        for sheet in self.config.sheets:
            if len(sheet.layers) != 1:
                raise ValueError(
                    "Hetero sheet-on-sheet currently requires exactly one value "
                    "in each material's 2D.layers (e.g., layers=[2])."
                )
        return sum(int(sheet.layers[0]) for sheet in self.config.sheets)

    def _provenance_components(self):
        """One provenance component per stacked material."""
        return [(f"sheet_{i + 1}", sheet) for i, sheet in enumerate(self.config.sheets)]

    def build(self) -> None:
        """Assemble the hetero stack and write the run phase (``lammps/slide.in``).

        The run phase itself is delegated to :meth:`write_run_scripts` (the
        seam a subclass overrides to emit something other than the dynamic
        slide); everything here is stack assembly shared by both.
        """
        n = self.n_layers
        if n < self.MIN_LAYERS:
            raise ValueError(
                f"Hetero sheet-on-sheet requires at least {self.MIN_LAYERS} "
                f"layers, got {n}"
            )

        logger.info("Building %d-layer heterostructure sheet-on-sheet slide...", n)
        self._create_directories()
        self._init_provenance()

        # Build the stack INTO the run dir (output_dir/hetero.data,
        # output_dir/lammps/system.in.settings, output_dir/provenance/), but
        # have the settings NAME the staged potentials sim-root-relatively --
        # that is the directory the HPC job scripts run LAMMPS from.
        stack = build_hetero_structure(
            self.config, self.config.settings, workdir=self.output_dir,
            potentials_prefix=self._potentials_prefix(),
        )

        bands = compute_layer_zbands(stack.data_path, stack.layers)

        # Box from the assembled hetero.data: the template sets the z extent
        # with change_box, from the atom z-range (self.z_box, set below once
        # the potentials are final).
        at = read_lammps_data(stack.data_path)
        cell = at.cell
        xlo, xhi = 0.0, float(cell[0, 0])
        ylo, yhi = 0.0, float(cell[1, 1])
        zmin = float(at.positions[:, 2].min())
        zhi = float(at.positions[:, 2].max())

        settings = self.config.settings
        check_interlayer_box(self.config.general, lx=xhi - xlo, ly=yhi - ylo,
                             xy=float(cell[1, 0]),
                             skin=settings.simulation.neighbor_list)
        self._check_d3_fits({'xlo': xlo, 'xhi': xhi, 'ylo': ylo, 'yhi': yhi})

        atom_style = _detect_atom_style(stack.data_path)  # 'atomic' for sw
        lat_c = stack.interface_spacing or settings.geometry.lat_c_default

        # ------------------------------------------------------------------ #
        # virtual_atom drive: reserve a DEDICATED non-interacting virtual type.
        #
        # The reused drive block does ``create_atoms {{num_atom_types}}`` +
        # ``group virtual type {{num_atom_types}}`` + ``fix move``. If
        # ``num_atom_types`` were the highest REAL type (``stack.total_types``),
        # that group would capture the ENTIRE top-type sublattice -- spread
        # across the driven top AND the thermostatted center layer -- and
        # rigidly drag those REAL atoms (wrong friction, silent). So, exactly
        # like the homogeneous path
        # (``SheetOnSheetSimulation._register_drive_atom`` ->
        # ``PotentialManager.register_virtual_atom``), add one extra
        # non-interacting type and make the box reserve room for it.
        reserve_virtual_type = self._reserve_drive_virtual_type()
        num_atom_types = stack.total_types
        if reserve_virtual_type:
            # Re-register the (geometry-independent, deterministic) hetero
            # potential spine, this time reserving a DEDICATED non-interacting
            # virtual type. register_hetero_potentials adds it BEFORE the
            # many-body self-interactions -- the same order the homogeneous path
            # uses -- so every ``pair_coeff * * sw ...`` element list is sized to
            # the full box type count (real types + trailing NULL for the virtual
            # type) the first time, and write_file emits the virtual type's
            # ``mass N 1.0`` + zero-LJ ``pair_coeff * N lj/cut 1e-100``. This
            # re-registration OVERWRITES lammps/system.in.settings (which
            # build_hetero_structure wrote with real types only); hetero.data
            # (real types unchanged) stays consistent with the new settings.
            hp = register_hetero_potentials(
                self.config.sheets,
                self.config.settings,
                self.output_dir,
                reserve_virtual_type=True,
                potentials_prefix=self._potentials_prefix(),
                interlayer_general=self.config.general,
            )
            # Fail loud on any real-type-layout divergence: never ship an
            # inconsistent settings/data pairing. type_ids covers the REAL
            # materials only, so their total must equal the assembled stack's
            # real-type count; pm.types additionally holds the virtual type.
            real_type_count = sum(len(ids) for ids in hp.type_ids)
            if real_type_count != stack.total_types:
                raise RuntimeError(
                    "Re-registered hetero potentials do not match the assembled "
                    f"stack: {real_type_count} real types vs "
                    f"stack.total_types={stack.total_types}. Refusing to write an "
                    "inconsistent settings/hetero.data pairing for the "
                    "virtual_atom drive."
                )
            if len(hp.pm.types) != stack.total_types + 1:
                raise RuntimeError(
                    "Re-registered hetero potentials must hold exactly one extra "
                    f"(virtual) type: len(pm.types)={len(hp.pm.types)} vs "
                    f"stack.total_types + 1={stack.total_types + 1}."
                )
            expected_settings = self.output_dir / "lammps" / "system.in.settings"
            if Path(hp.settings_path).resolve() != expected_settings.resolve():
                raise RuntimeError(
                    f"Re-registered settings path {hp.settings_path} != the "
                    f"settings the slide includes ({expected_settings})."
                )
            # register_hetero_potentials already wrote the settings (with the
            # virtual type) to hp.settings_path -- no second write needed.
            num_atom_types = len(hp.pm.types)  # == stack.total_types + 1
            stack.pm = hp.pm

        # Hand the assembled-stack state to the run-phase seam. Everything
        # below this point that a subclass might want to replace lives in
        # write_run_scripts(); everything above (the stack assembly, the layer
        # bands, the type/virtual-atom spine) is shared.
        self._stack = stack
        self._layer_bands = bands
        cutoff = (stack.pm.max_pair_cutoff() if stack.pm is not None
                  else float(settings.potential.lj_cutoff))
        self.z_box = ZBox.for_settings(
            settings, (zmin, zhi), cutoff,
            top_clearance=(SHEET_VIRTUAL_ATOM_HEIGHT + VIRTUAL_ATOM_ROOM
                           if reserve_virtual_type else 0.0))
        self._hetero_box = {"xlo": xlo, "xhi": xhi, "ylo": ylo, "yhi": yhi, "zhi": zhi,
                            **self.z_box.context()}
        self._hetero_coincidence_cell = (xhi - xlo, yhi - ylo)
        self._hetero_cell_xy_tilt = float(cell[1, 0])
        self._hetero_atom_style = atom_style
        self._hetero_lat_c = lat_c
        self._hetero_num_atom_types = num_atom_types
        self._hetero_virtual_type_reserved = reserve_virtual_type

        self.write_run_scripts()

        # D3: record the achieved twist angle + moiré period into the run
        # manifest (achieved != target for discrete commensurate angles; always
        # log the achieved value). Inert for the non-twist hetero slide
        # (twist_angle_achieved is None), so the aligned path is unaffected.
        if stack.twist_angle_achieved is not None:
            self._record_twist_manifest(stack, self.config.general.twist_angle)

        # The homogeneous HPC scaffolding, used unmodified: this builder renders
        # sim-root-relative paths (see the class docstring), which is the
        # convention that scaffolding — and the job scripts it writes — assume.
        # Called bare, like SheetOnSheetSimulation.build and AFMSimulation.build:
        # a failure here (e.g. settings.hpc.modules unset) is a real, reportable
        # misconfiguration, and swallowing it produced a tree with no job scripts
        # and nothing but a log line to say why.
        self._generate_hpc_scripts()

    def _reserve_drive_virtual_type(self) -> bool:
        """Whether the run phase needs a dedicated non-interacting drive atom.

        Seam (not inlined in :meth:`build`) because it is a property of the RUN
        PHASE, not of the stack: a static PES scan has no drive at all, so
        :class:`~frictionsim2d.builders.sheetonsheet.pes_sheet.HeteroPESSheetSimulation` returns
        False
        and the extra virtual type is never registered — mirroring
        :meth:`PESSheetSimulation._register_drive_atom`'s no-op on the
        homogeneous side, which likewise keeps the type count aligned with the
        real atoms in the data file.
        """
        return self.config.settings.simulation.drive_method == "virtual_atom"

    def _build_obs_ctx(self, n_layers: int, results_file_pattern: str,
                       interlayer_energy: bool = False) -> Dict[str, object]:
        """Observable context for the hetero family.

        Mirrors :meth:`SheetOnSheetSimulation._build_obs_ctx`. This class
        always renders the ``has_internal_lj`` branch of the template (the
        hybrid-sw reaction-force friction proxy -- see the ``has_internal_lj``
        context key in :meth:`write_run_scripts`), so `extra_filter` always
        drops the interlayer channels here. That leaves hetero's peratom
        varlist empty -- ``E_int_interlayer`` is the family's only peratom
        channel -- and `build_obs_ctx`'s emptiness gate correctly suppresses
        `fc_pa` as a result. That is the intended outcome of this task: hetero
        gains the raw tier, not a populated peratom tier. A real hetero
        peratom channel is a later opt-in addition, not something to force
        here by relaxing this filter.

        ``interlayer_energy`` requests ``E_int_interlayer_all`` instead: set
        when the run script computes ``v_e_int_il_all`` from an ILP/KC
        sub-style (``interlayer_energy_substyle``; spec 2026-11-11 §7). Its
        value is the interlayer energy summed over EVERY layer pair, not the
        shearing interface ``E_int_interlayer`` names, hence its own channel.
        """
        # `contact_cutoff` is left to sheet_obs_ctx's fallback default:
        # its only consumer in the registry is n_contact_il, and the
        # extra_filter below strips n_contact_il UNCONDITIONALLY
        # (has_internal_lj is not threaded through here the way
        # SheetOnSheetSimulation's is -- see the docstring above), so it never
        # reaches a rendered compute regardless of optional_channels. Not a
        # placeholder awaiting a real derivation: this class extends
        # SimulationBase, not SheetOnSheetSimulation, so it could not inherit
        # SheetOnSheetSimulation._contact_cutoff() even if it wanted to --
        # that method reads self.config.sheet, which this class does not have.
        # A real derivation would only be worth adding if a future task made
        # this extra_filter conditional on has_internal_lj (as
        # SheetOnSheetSimulation's already is), making n_contact_il reachable.
        interlayer = {'E_int_interlayer', 'n_contact_il'}
        if not interlayer_energy:
            interlayer.add('E_int_interlayer_all')
        return sheet_obs_ctx(
            self.config.settings,
            sim_type=self.FAMILY,
            n_layers=n_layers,
            results_file_pattern=results_file_pattern,
            extra_filter=lambda ob: ob.name not in interlayer,
            requested_channels=('E_int_interlayer_all',) if interlayer_energy else (),
        )

    def _interlayer_context(self, pm, bands: List[Dict]) -> Dict[str, object]:
        """Friction-branch and molecule-ID keys for ``hetero/slide.lmp``.

        The hetero slide reads friction from the bottom-layer reaction force
        under every interlayer style -- it never had a ``compute group/group``
        branch -- so ``use_group_group`` is always False and comparison mode
        changes nothing here. ``lj`` keeps today's ``has_internal_lj`` branch
        (its "many-body potential" comment included) so its script is
        unchanged; other styles take the interlayer-style branch, with
        molecule IDs per z-band layer (``layer_N``) for ILP/KC and, there, the
        interlayer energy from the ILP/KC sub-style.
        """
        general = self.config.general
        legacy_lj = interlayer_style(general) == 'lj'
        ctx = interlayer_template_context(
            general, has_internal_lj=legacy_lj, pm=pm,
            molecule_groups=[f"layer_{band['idx']}" for band in bands])
        ctx['use_group_group'] = False
        return {'has_internal_lj': legacy_lj, 'interlayer_style': interlayer_style(general),
                **ctx}

    def write_run_scripts(self) -> None:
        """Render and write the run phase (``lammps/slide.in``): the dynamic slide.

        The run-script seam of :meth:`build`. Everything it needs was placed on
        ``self`` by ``build`` (``_stack``, ``_layer_bands``, ``_hetero_box``,
        ``_hetero_atom_style``, ``_hetero_lat_c``, ``_hetero_num_atom_types``,
        ``_hetero_virtual_type_reserved``), so a subclass can swap the whole
        run phase — e.g. a static PES grid scan instead of a slide — without
        reimplementing the stack assembly above it.
        """
        stack = self._stack
        bands = self._layer_bands
        box = self._hetero_box
        n = self.n_layers
        settings = self.config.settings
        sim = settings.simulation
        out = settings.output
        general = self.config.general

        results_file_pattern = self._run_dir_path(
            "results", "friction_p${pressure}_a${a}_s${speed}"
        )
        interlayer_ctx = self._interlayer_context(getattr(stack, "pm", None), bands)

        # Same key set as SheetOnSheetSimulation.write_inputs' base_context (only
        # the keys the template actually references), with hetero-specific values
        # and the same SIM-ROOT-RELATIVE paths that builder uses.
        context = {
            "temp": general.temp,
            "halt_temperature": halt_temperature(settings, general.temp),
            "data_dir": self._run_dir_path("data"),   # final-state retention
            # Points the guard halted (common/halt.lmp); run_status reads it.
            "halt_record": self._run_dir_path("results", "halted.txt"),
            # One seed per channel, drawn in the builder like every other
            # family -- see SimulationBase._run_seed for why, and note a
            # template-side `| random` re-draws per occurrence.
            "seed_sheet_center": self._run_seed('sheet_center'),
            "xlo": box["xlo"],
            "xhi": box["xhi"],
            "ylo": box["ylo"],
            "yhi": box["yhi"],
            "zhi": box["zhi"],
            "box_zlo": box["box_zlo"],
            "box_zhi": box["box_zhi"],
            "data_file": self._run_dir_path("hetero.data"),
            "potential_file": self._run_dir_path("lammps", "system.in.settings"),
            "num_atom_types": self._hetero_num_atom_types,
            "reserve_virtual_type": self._hetero_virtual_type_reserved,
            "ngroups": stack.total_types,
            "n_layers": n,
            "constraint_mode": "none",
            "n_bond_types": 0,
            "atom_style": self._hetero_atom_style,
            "pot_type": "sw",
            # The cell's MLIP keys, from the first sheet: a cell is uniform
            # by construction, and ``stack.pm`` exists only on some branches.
            **mlip_template_context(self.config.sheets[0].pot_type
                                    if self.config.sheets else None),
            **interlayer_ctx,
            # ``run_simulations`` refuses a slide with no pressure
            # (REQUIRES_PRESSURE, frictionsim2d.run.expand.missing_pressure). The 0.0
            # fallback only keeps a raw None out of the template (``variable
            # pressure equal None`` is a LAMMPS error) for a builder driven
            # directly, and for the hetero PES scan subclass, which reads no
            # pressure at all.
            "pressures": general.pressure if general.pressure is not None else 0.0,
            "scan_speed_config": general.scan_speed,
            "scan_angle_config": general.scan_angle,
            "scan_angle_force": general.scan_angle_force,
            "drive_method": sim.drive_method,
            "thermostat_type": settings.thermostat.type,
            "timestep": sim.timestep,
            "thermo": sim.thermo,
            "neighbor_list": sim.neighbor_list,
            "neigh_modify_command": sim.neigh_modify_command,
            "run_steps": sim.slide_run_steps,
            "min_style": sim.min_style,
            "minimization_command": sim.minimization_command,
            "results_freq": out.results_frequency,
            "dump_freq": out.dump_frequency.get("slide", 1000),
            "dump_enabled": out.dump.get("slide", False),
            "results_file_pattern": results_file_pattern,
            "obs_ctx": self._build_obs_ctx(
                n, results_file_pattern,
                interlayer_energy=bool(interlayer_ctx["interlayer_energy_substyle"])),
            "dump_file_pattern": self._run_dir_path(
                "visuals", "slide_p${pressure}_a${a}_s${speed}.lammpstrj"
            ),
            "driving_spring_ev": (general.driving_spring or 50.0) / NM_TO_EV_A2,
            "bond_spring_ev": (general.bond_spring or 80.0) / NM_TO_EV_A2,
            "lat_c": self._hetero_lat_c,
            "ev_a_to_nn": EV_A_TO_NN,
            "ev_a3_to_gpa": EV_A3_TO_GPA,
            "layers": bands,
        }

        script = self.render_template("hetero/slide.lmp", context)
        self.write_file("lammps/slide.in", script)

        # cutoff_source is stated here rather than read off the instance: this
        # class never calls SheetOnSheetSimulation._contact_cutoff (it cannot --
        # see _build_obs_ctx), so _contact_cutoff_source is genuinely absent and
        # sheet_obs_ctx's documented fallback is what the script uses.
        drive_k, drive_rest_length = drive_spring(
            context["drive_method"], context["driving_spring_ev"], 0.0)
        write_sheet_provenance(
            self,
            contact_cutoff=context["obs_ctx"]["contact_cutoff"],
            cutoff_source="fallback_default",
            thermostat_scheme=sheet_thermostat_scheme(self.config.settings),
            drive_k=drive_k,
            drive_rest_length=drive_rest_length,
            pm=getattr(stack, "pm", None),
        )
        logger.info("Wrote hetero slide script to %s/lammps/slide.in", self.output_dir)

    def _record_twist_manifest(self, stack, twist_angle_target) -> None:
        """Write the achieved twist angle + moiré period into the run manifest.

        Augments the provenance manifest (``provenance/manifest.json``, created
        by :meth:`SimulationBase.write_file` during provenance init) with a
        ``twist`` block recording the user-selected target, the ACHIEVED
        commensurate angle (the two differ -- commensurate angles are discrete),
        and the moiré period. Downstream (postprocessing/AiiDA) can read the
        realised twist straight from provenance.
        """
        data = self._load_or_init_manifest()
        data["twist"] = {
            "twist_angle_target_deg": twist_angle_target,
            "twist_angle_achieved_deg": stack.twist_angle_achieved,
            "moire_period_ang": stack.moire_period,
        }
        manifest_path = self._write_manifest(data)
        logger.info(
            "Twisted moiré: target %.3f deg -> ACHIEVED %.3f deg, moiré period "
            "%.3f A (recorded in %s)",
            twist_angle_target, stack.twist_angle_achieved, stack.moire_period,
            manifest_path,
        )
