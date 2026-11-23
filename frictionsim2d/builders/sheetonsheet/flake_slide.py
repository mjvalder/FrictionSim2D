"""D4 flake-driven slide: a finite flake driven across a substrate sheet.

A finite flake (single-material or a hetero bilayer) is placed on a substrate
sheet at the minimized flake--sheet gap, pressed onto it by the configured
normal pressure (applied over the flake's projected area, see
:func:`flake_projected_area`) and driven laterally by a virtual-atom / COM
spring (no tip, no substrate slab). The substrate sheet's bottom band is fixed;
its interior + the flake are thermostatted; lateral friction is read from the
bottom-sheet reaction force (the many-body ``sw`` observable) and the normal
load from the force the sheet exerts on the flake (``v_fz``), so COF is
lateral force / normal load exactly as for sheet-on-sheet.

This reuses the D1 slide body (``templates/hetero/slide.lmp``) via a dedicated
``templates/flake/slide.lmp`` in which the driven and loaded group is the flake
(by type), the tip block is dropped, and the substrate-sheet layer groups are
the z-region bands.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ...core.config import SheetOnSheetSimulationConfig
from ...core.mlip import validate_mlip_cell
from ...potentials.manager import PotentialManager
from ..simulation_base import (SimulationBase, drive_spring, halt_temperature,
                               mlip_template_context)
from .base import (sheet_obs_ctx, sheet_thermostat_scheme,
                   write_sheet_provenance)
from ...data.models import EV_A_TO_NN, EV_A3_TO_GPA, NM_TO_EV_A2
from ..construction import components
from ..utils.interlayer import (check_interlayer_box_from_data, interlayer_style,
                                interlayer_template_context, set_interlayer_from_config)
from ..utils.z_vacuum import VIRTUAL_ATOM_ROOM, ZBox
from ..construction.hetero_stack import _box_region_cmd, _detect_atom_style, _run_lammps
from ...interfaces.ase import read_lammps_data

logger = logging.getLogger(__name__)

PAD = 1.0  # Å pad above/below each sheet layer's atom z-range (mirrors hetero_slide.PAD)


def flake_projected_area(shape: str, edge_length: float) -> float:
    """Projected in-plane area (Å²) of the outline ``build_flake`` carves.

    The flake-slide normal load is a pressure over this area (the sheet slide
    uses the periodic box area instead; a finite flake's contact is its own
    footprint). ``edge_length`` means what :class:`FlakeConfig` says it means:

    * ``triangle`` -- equilateral, side ``a``: ``A = sqrt(3)/4 * a**2``
    * ``square``   -- side ``a``: ``A = a**2``
    * ``hexagon``  -- circumradius ``R`` (= side): the regular hexagon,
      ``A = 3*sqrt(3)/2 * R**2``

    Each is the area of the outline ``components._in_*`` actually carves;
    ``test_flake_projected_area_matches_the_carving_predicate`` integrates the
    predicates numerically and fails the moment either side drifts.

    ``rotation_deg`` and the D3b twist are rigid in-plane rotations, so they
    leave the area unchanged. The value is the geometric cut outline, not an
    atom count: it differs from ``N_atoms * area_per_atom`` only by the edge
    discretisation of the carved lattice.
    """
    a = float(edge_length)
    shape_lower = shape.lower()
    if shape_lower == "triangle":
        return float(np.sqrt(3.0) / 4.0 * a * a)
    if shape_lower == "square":
        return a * a
    if shape_lower == "hexagon":
        return float(1.5 * np.sqrt(3.0) * a * a)
    raise ValueError(
        f"No projected-area formula for flake shape {shape!r} "
        "(supported: triangle, square, hexagon)."
    )


def compute_sheet_zbands(sheet_path, n_layers: int, pad: float = PAD) -> List[Dict]:
    """Split a stacked substrate sheet's atoms into ``n_layers`` z-region bands.

    The PotentialManager already writes a per-layer ``group layer_N type ...``
    for the stacked sheet; the template ALSO adds each z-slab (region) to that
    group, like the D1 hetero slide, which is a no-op when the two agree.
    Unlike the hetero path there is no per-layer ``.source``/``.z`` placement
    record (``build_sheet`` writes one stacked file), so the bands are recovered
    by cutting at the ``n_layers - 1`` largest inter-atom z-gaps of the sorted
    z-coordinates. For a well-separated 2D stack (interlayer gap > the largest
    intra-layer sub-plane gap, true for MoS2/TMDs) this reproduces the physical
    layers. (A puckered layer whose internal sub-plane gap exceeds the interlayer
    gap -- e.g. black phosphorus -- would need the placement-based
    ``hetero_slide.compute_layer_zbands`` instead; out of scope for D4's TMD
    substrates.)

    Returns one ``{'idx', 'zlo', 'zhi'}`` per band (idx 1..n bottom->top).
    """
    z = np.sort(read_lammps_data(sheet_path).positions[:, 2])
    if n_layers <= 1:
        return [{"idx": 1, "zlo": float(z.min()) - pad, "zhi": float(z.max()) + pad}]

    gaps = np.diff(z)
    # Indices (into z) after which the n_layers-1 largest gaps fall.
    cut_after = np.sort(np.argsort(gaps)[-(n_layers - 1):])
    bands: List[Dict] = []
    start = 0
    for band_idx, cut in enumerate(list(cut_after) + [len(z) - 1]):
        seg = z[start:cut + 1]
        bands.append({
            "idx": band_idx + 1,
            "zlo": float(seg.min()) - pad,
            "zhi": float(seg.max()) + pad,
        })
        start = cut + 1
    return bands


class FlakeSlideSimulation(SimulationBase):
    """Builder for a finite flake driven across a substrate sheet (D4, no tip).

    Assembles a single-material (or hetero-bilayer) flake sitting on a
    single-material substrate sheet at the minimized flake--sheet gap, loads the
    flake with ``pressure`` x its projected area, and drives the flake laterally
    by a virtual-atom / COM spring. The substrate sheet's bottom band is fixed;
    its interior + the flake are thermostatted; lateral friction is read from
    the bottom-sheet reaction force (the many-body ``sw`` observable), the
    normal load from ``v_fz``. Routed from ``run sheetonsheet`` when a
    ``[Flake]`` section is present.

    Mirrors :class:`~frictionsim2d.builders.sheetonsheet.hetero.HeteroSheetOnSheetSimulation`'s
    run-dir-relative convention: everything (``flake_slide.data``,
    ``lammps/system.in.settings``, ``provenance/potentials/...``, ``results/...``)
    is written into ``output_dir`` and referenced run-dir-relatively, so LAMMPS
    is launched with ``cwd = output_dir``.
    """

    FAMILY = 'flake'

    #: A flake slide needs a fixed bottom band + at least one thermostatted
    #: interior band -> the substrate sheet must have >= 2 layers.
    MIN_SHEET_LAYERS: int = 2

    #: Like the hetero slide, this builder writes a single lammps/slide.in and
    #: never reads outer_loop, so a [Flake] section collapses any configured
    #: split to one job. Explicit (the base default is the same) because the
    #: reroute from sheet-on-sheet is silent.
    SPLIT_OUTER_LOOPS = frozenset()

    #: The slide is loaded by ``[general] pressure``; with none it would run at
    #: 0 GPa and report a meaningless COF, so ``frictionsim2d.run.expand.missing_pressure``
    #: refuses such a config before anything is built.
    REQUIRES_PRESSURE: bool = True

    def __init__(self, config: SheetOnSheetSimulationConfig, output_dir: str,
                 config_path: Optional[str] = None):
        super().__init__(config, output_dir, config_path=config_path)
        self.config: SheetOnSheetSimulationConfig = config

    def _init_provenance(self) -> None:
        """Create the provenance folder and record the substrate + flake inputs."""
        prov_dir = self.output_dir / "provenance"
        prov_dir.mkdir(parents=True, exist_ok=True)
        self._add_component_files_to_provenance("sheet", self.config.sheet)
        if self.config.flake_sheets:
            for i, fs in enumerate(self.config.flake_sheets):
                self._add_component_files_to_provenance(f"flake_{i + 1}", fs)
        elif self.config.flake is not None:
            self._add_component_files_to_provenance("flake", self.config.flake)

    def _filled_flake_config(self):
        """Return the [Flake] config with its material fields ensured.

        A single-material flake may omit ``mat``/``cif_path``/``pot_path``/
        ``pot_type`` and reuse the substrate ``[2D]`` material (the plan's "the
        flake reuses [2D]"). Fill any missing field from the substrate sheet.
        """
        flake = self.config.flake
        sheet = self.config.sheet
        updates = {}
        if flake.mat is None:
            updates["mat"] = sheet.mat
        if flake.cif_path is None:
            updates["cif_path"] = sheet.cif_path
        if flake.pot_path is None:
            updates["pot_path"] = sheet.pot_path
        if flake.pot_type is None:
            updates["pot_type"] = sheet.pot_type
        return flake.model_copy(update=updates) if updates else flake

    def _build_obs_ctx(self, n_layers: int,
                       results_file_pattern: str) -> Dict[str, object]:
        """Observable context for the flake family.

        Mirrors :meth:`HeteroSheetOnSheetSimulation._build_obs_ctx`. Unlike
        hetero, ``has_internal_lj`` is not threaded through as a filter here:
        the flake model declares no interlayer channel at all (its friction
        proxy is the bottom-sheet reaction force `react_{x,y,z}`, not a
        group/group interlayer energy), so there is nothing for an
        ``extra_filter`` to drop -- ``channels_for('flake', 'peratom')`` is
        already empty. That leaves the peratom tier empty and
        `build_obs_ctx`'s emptiness gate correctly suppresses `fc_pa`, same
        outcome as hetero but for a different reason (no channel exists,
        rather than one being filtered out).

        ``layer_groups_available=True`` (the default): `flake/slide.lmp`
        defines `group layer_{{ layer.idx }} region reg_layer_{{ layer.idx
        }}` for every substrate layer, so `layer_channels(n_layers)`'s
        `compute COM_L{i} layer_{i} com` resolves against a real group. Those
        layer COMs describe the SUBSTRATE stack -- the flake's own COM is
        `drivex@flake`/`drivey@flake`/`drivez@flake`, a template-owned
        channel with no `lammps_compute` of its own (see the "Flake COM"
        comment in the template).
        """
        # `contact_cutoff` is left to sheet_obs_ctx's fallback default: its
        # only consumer in the registry is n_contact_il, whose sim_types are
        # ('sheetonsheet', 'hetero') -- 'flake' is not among them, so it never
        # reaches a rendered compute here regardless of optional_channels. Not
        # a placeholder awaiting a real derivation: this class extends
        # SimulationBase, not SheetOnSheetSimulation, so it could not inherit
        # SheetOnSheetSimulation._contact_cutoff() even if it wanted to --
        # that method reads self.config.sheet, which this class does not have.
        # A real derivation would only be worth adding if a future task
        # declared a flake-specific channel that consumes it.
        return sheet_obs_ctx(
            self.config.settings,
            sim_type=self.FAMILY,
            n_layers=n_layers,
            results_file_pattern=results_file_pattern,
        )

    def _normal_load_context(self) -> Dict[str, object]:
        """Render keys for the flake normal-load block.

        The ``[Flake]`` section is the shape spec for both a single-material
        and a hetero (``[Flake-1]``/``[Flake-2]``) flake, so its
        ``shape``/``edge_length`` give the contact outline either way.
        """
        flake = self.config.flake
        return {
            "flake_shape": flake.shape,
            "flake_edge_length": float(flake.edge_length),
            "flake_area": float(flake_projected_area(flake.shape, flake.edge_length)),
        }

    def build(self) -> None:
        """Assemble the flake-on-sheet system and write ``lammps/slide.in``."""
        if len(self.config.sheets) > 1:
            raise NotImplementedError(
                "FlakeSlideSimulation currently supports a single-material [2D] "
                "substrate sheet; a hetero substrate ([2D-1]/[2D-2]) under a flake "
                "is out of scope for D4."
            )
        if self.config.flake is None:
            raise ValueError("FlakeSlideSimulation requires a [Flake] section.")
        n_flake_sections = len(self.config.flake_sheets)
        if n_flake_sections not in (0, 2):
            raise ValueError(
                "A hetero flake is carved from exactly two materials (a "
                f"bilayer); this config has {n_flake_sections} [Flake-N] "
                "section(s). Before this check any other count was neither an "
                "error nor built: the hetero-flake branch below is selected by "
                "`len(flake_sheets) == 2`, so 1 or 3+ fell through to the "
                "single-material path and EVERY [Flake-N] section was ignored "
                "-- while frictionsim2d.run.naming.material_token still named all of "
                "them in the output directory, so the run built one thing and "
                "was filed under the name of another. Use exactly [Flake-1] "
                "and [Flake-2] for a hetero flake, or drop them for a "
                "single-material [Flake]."
            )

        settings = self.config.settings
        sheet_cfg = self.config.sheet
        n_sheet_layers = int(sheet_cfg.layers[0]) if sheet_cfg.layers else 1
        if n_sheet_layers < self.MIN_SHEET_LAYERS:
            raise ValueError(
                f"Flake-slide substrate sheet needs at least {self.MIN_SHEET_LAYERS} "
                f"layers (a fixed bottom band + a thermostatted interior), got "
                f"{n_sheet_layers}."
            )

        logger.info("Building flake-on-sheet slide (substrate %d layers)...", n_sheet_layers)
        self._create_directories()
        self._init_provenance()
        build_dir = self.output_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        # 1. Substrate sheet (homogeneous, stacked >=2 layers). The stacked file
        #    carries PER-LAYER atom types (stack_multilayer_sheet renumbers them
        #    element-first: Mo_L1, Mo_L2, S_L1, S_L2 for a MoS2 bilayer), so the
        #    PotentialManager below must register the sheet with the same layer
        #    count -- see step 3.
        sheet_path, _sheet_dims, lat_c = components.build_sheet(
            sheet_cfg, build_dir, stack_if_multi=True, settings=settings,
            n_layers_override=n_sheet_layers,
            stacking_type=getattr(sheet_cfg, "stack_type", "AA"),
            interlayer_general=self.config.general,
        )
        check_interlayer_box_from_data(self.config.general, sheet_path,
                                       skin=settings.simulation.neighbor_list)

        # 2. Flake (single-material reuses [2D]; hetero flake from [Flake-1/2]).
        hetero_flake = len(self.config.flake_sheets) == 2
        if hetero_flake:
            flake_path, _flake_dims = components.build_hetero_flake_component(
                self.config.flake_sheets, settings, self.config.flake, build_dir,
                hetero_stacking=getattr(self.config.general, "hetero_stacking", "grouped"),
                interlayer_general=self.config.general,
            )
            flake_names = ["flake_1", "flake_2"]
            flake_components = list(self.config.flake_sheets)
        else:
            filled_flake = self._filled_flake_config()
            flake_path, _flake_dims = components.build_flake_component(
                filled_flake, build_dir,
                interlayer=interlayer_style(self.config.general),
            )
            flake_names = ["flake"]
            flake_components = [filled_flake]

        # 2b. D3b: incommensurate twist of the FINITE flake. Rotate the flake's
        #     atoms about z in the fixed box frame (a genuine lattice twist vs the
        #     periodic bottom sheet -- the flake is finite, so any angle is legal
        #     and no commensurate snapping applies). Inert when twist_angle is
        #     None -> the D4 flake path stays byte-identical.
        twist = getattr(self.config.general, "twist_angle", None)
        if twist is not None:
            # Deferred because this whole branch is optional: only a config
            # with `twist_angle` set ever needs the lattice package (~1.7 ms
            # for `frictionsim2d.lattice.apply` + `frictionsim2d.lattice.matching`, measured with
            # `python -X importtime`). The comment this replaces said "avoid
            # cycle", which is not the reason: `frictionsim2d/lattice/` imports nothing
            # from `frictionsim2d/builders/`, and `frictionsim2d.builders.afm.base` imports this
            # very function at module scope.
            # pylint: disable-next=import-outside-toplevel
            from ...lattice.apply import rotate_flake_in_plane
            rotate_flake_in_plane(flake_path, float(twist))   # in-place twist, box unchanged
            self._record_twist_manifest(float(twist))
            logger.info("Twisted flake by %.3f deg (D3b incommensurate)", float(twist))

        # 2c. The flake is finite and the substrate sheet is periodic: a flake
        #     too wide for the cell slides against a copy of itself. Checked
        #     after the twist, which changes the span.
        components.check_flake_pbc_clearance(
            components.flake_atom_span(flake_path),
            (float(_sheet_dims['xhi'] - _sheet_dims['xlo']),
             float(_sheet_dims['yhi'] - _sheet_dims['ylo'])),
            settings.potential.lj_cutoff,
        )

        # 3. ONE PotentialManager: sheet -> flake(s) -> reserved virtual, so global
        #    atom types are sheet 1..S, flake S+1..S+F, virtual S+F+1.
        #
        #    The RUN-DIR-relative prefix here, always: assembly (step 4) runs
        #    LAMMPS with cwd = output_dir, so that is the only prefix its
        #    minimization can open. The run scripts start from the simulation
        #    ROOT instead, so the settings are re-registered with the sim-root
        #    prefix once assembly is done (step 4b) -- the same two-phase dance
        #    build_hetero_structure does.
        reserve_virtual_type = settings.simulation.drive_method == "virtual_atom"
        settings_path = self.output_dir / "lammps" / "system.in.settings"

        # Capability-matrix rules across the whole cell (spec §1.1/§3.4).
        validate_mlip_cell(
            sheet_cfg,
            components=list(zip(flake_names, flake_components)),
            constraint_mode=settings.simulation.constraint_mode,
        )

        def _register(prefix: str) -> PotentialManager:
            return self._register_potentials(
                prefix, sheet_cfg=sheet_cfg, n_sheet_layers=n_sheet_layers,
                flake_names=flake_names, flake_components=flake_components,
                reserve_virtual_type=reserve_virtual_type,
                settings_path=settings_path)

        pm = _register("provenance/potentials")
        n_sheet_types = len(pm.types.ids_by_component("sheet"))

        # 3b. The assembly run 0 (step 4) gets the same type layout -- ilp/tmd's
        #     chalcogen split included -- but today's lj interactions, in a
        #     settings file of its own: it runs the lmp_serial binary, which
        #     need not carry the INTERLAYER package. lj assembles on the run
        #     settings exactly as before.
        assembly_pm, assembly_settings = pm, settings_path
        if interlayer_style(self.config.general) != 'lj':
            assembly_settings = build_dir / "assembly.in.settings"
            assembly_pm = self._register_potentials(
                "provenance/potentials", sheet_cfg=sheet_cfg,
                n_sheet_layers=n_sheet_layers, flake_names=flake_names,
                flake_components=flake_components,
                reserve_virtual_type=reserve_virtual_type,
                settings_path=assembly_settings, interlayer_terms=False)

        # Flake type-groups (combined) + corner type-ids, read back from the PM.
        flake_group_ids: List[int] = []
        flake_corner_ids: List[int] = []
        for name in flake_names:
            flake_group_ids += pm.types.ids_by_component(name)
            flake_corner_ids += pm.types.ids_by_component_region(name, "corner")
        flake_group_ids = sorted(set(flake_group_ids))
        flake_corner_ids = sorted(set(flake_corner_ids))

        # 4. Assemble sheet + flake into ONE data file (flake types offset by S,
        #    flake placed centered in xy at the minimized flake-sheet gap).
        combined_path, zhi = self._assemble_flake_on_sheet(
            sheet_path=sheet_path,
            flake_path=flake_path,
            n_sheet_types=n_sheet_types,
            pm=assembly_pm,
            settings_path=assembly_settings,
            bottom_flake_component=flake_names[0],
        )

        # 4b. Retarget the settings for the RUN scripts. Assembly above is done,
        #     so the pair_coeff lines may now name the staged potentials
        #     sim-root-relatively -- which is what the generated HPC job scripts
        #     need, because they run LAMMPS from the simulation root, not from
        #     this run dir. Left run-dir-relative, every submitted flake job dies
        #     with "cannot open sw potential file". Re-registering (rather than
        #     mutating the prefix) is required: pair_coeff lines are built at
        #     registration time and only re-emitted by write_file. Standalone,
        #     relative_run_dir is the output dir itself, so this is a no-op.
        pm = _register(self._potentials_prefix())

        # 5. Substrate-sheet z-region bands (from the standalone, unshifted sheet
        #    file -- it is placed at shift 0 0 0 in the combined data).
        sheet_layers = compute_sheet_zbands(sheet_path, n_sheet_layers)

        # 6. Render context + slide.in.
        at = read_lammps_data(combined_path)
        cell = at.cell
        self.z_box = ZBox.for_settings(
            settings, (float(at.positions[:, 2].min()), float(at.positions[:, 2].max())),
            pm.max_pair_cutoff(),
            # flake/slide.lmp creates the virtual atom lat_c_default above the
            # flake's centre of mass.
            top_clearance=(settings.geometry.lat_c_default + VIRTUAL_ATOM_ROOM
                           if reserve_virtual_type else 0.0))
        general = self.config.general
        sim = settings.simulation
        out = settings.output
        flake_cfg = self.config.flake
        results_file_pattern = self._run_dir_path(
            "results", "friction_p${pressure}_a${a}_s${speed}")

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
            "xlo": 0.0,
            "xhi": float(cell[0, 0]),
            "ylo": 0.0,
            "yhi": float(cell[1, 1]),
            "zhi": zhi,
            **self.z_box.context(),
            "data_file": self._run_dir_path(combined_path.name),
            "potential_file": self._run_dir_path("lammps", "system.in.settings"),
            "num_atom_types": len(pm.types),
            # The reserved virtual type is baked into the assembled data file
            # (an extra type with zero atoms), so the slide read_data needs no
            # extra/atom/types; create_atoms fills it at drive time.
            "reserve_virtual_type": False,
            "ngroups": len(pm.types),
            "n_layers": n_sheet_layers,
            "constraint_mode": sim.constraint_mode,
            "atom_style": _detect_atom_style(combined_path),
            "pot_type": "sw",
            # LOAD-BEARING (see mlip_template_context): this builder
            # subclasses SimulationBase directly, so it inherits neither the
            # AFM nor the sheet-on-sheet wiring. The cell is uniform, so the
            # substrate sheet names its framework.
            **mlip_template_context(sheet_cfg.pot_type),
            "has_internal_lj": True,
            # has_internal_lj is pinned True above, so use_group_group is
            # False under every style: the flake always reads the reaction
            # force. The flake registers no E_int_interlayer channel, so no
            # interlayer energy compute either.
            **{**interlayer_template_context(
                general, has_internal_lj=True, pm=pm,
                molecule_groups=self._molecule_groups(
                    pm, n_sheet_layers, flake_names, in_slide=True)),
               "interlayer_energy_substyle": None},
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
            "obs_ctx": self._build_obs_ctx(n_sheet_layers, results_file_pattern),
            "dump_file_pattern": self._run_dir_path(
                "visuals", "slide_p${pressure}_a${a}_s${speed}.lammpstrj"),
            "driving_spring_ev": (general.driving_spring or 50.0) / NM_TO_EV_A2,
            "bond_spring_ev": (general.bond_spring or 80.0) / NM_TO_EV_A2,
            "lat_c": lat_c if lat_c is not None else settings.geometry.lat_c_default,
            "ev_a_to_nn": EV_A_TO_NN,
            "ev_a3_to_gpa": EV_A3_TO_GPA,
            "sheet_layers": sheet_layers,
            "flake_group_types": " ".join(map(str, flake_group_ids)),
            "flake_corner_types": " ".join(map(str, flake_corner_ids)),
            "flake_corner_k": flake_cfg.corner_spring_k,
            "corner_hold": flake_cfg.corner_hold,
            "virtual_offset": settings.geometry.lat_c_default,
            **self._normal_load_context(),
        }

        script = self.render_template("flake/slide.lmp", context)
        self.write_file("lammps/slide.in", script)

        # See the note on the hetero builder's call: this class does not have
        # SheetOnSheetSimulation._contact_cutoff either, so the fallback is what
        # its obs_ctx used and what the script carries.
        drive_k, drive_rest_length = drive_spring(
            context["drive_method"], context["driving_spring_ev"], 0.0)
        write_sheet_provenance(
            self,
            contact_cutoff=context["obs_ctx"]["contact_cutoff"],
            cutoff_source="fallback_default",
            thermostat_scheme=sheet_thermostat_scheme(self.config.settings),
            drive_k=drive_k,
            drive_rest_length=drive_rest_length,
            pm=pm,
        )
        logger.info("Wrote flake slide script to %s/lammps/slide.in", self.output_dir)

        self._generate_hpc_scripts()

    def _register_potentials(self, prefix: str, *, sheet_cfg, n_sheet_layers: int,
                             flake_names: List[str], flake_components: list,
                             reserve_virtual_type: bool,
                             settings_path: Path,
                             interlayer_terms: bool = True) -> PotentialManager:
        """Register sheet -> flake(s) -> virtual and write ``settings_path``.

        ``lj`` (the default) emits exactly what it always has. Any other
        ``interlayer_style`` acts between every 2D layer of the system -- the
        substrate sheet's layers AND the flake(s), whose interface is the
        sliding one. ``ilp``/``kc`` therefore replace the sheet-flake (and
        flake-flake) LJ with their many-body term; ``lj_g`` keeps that LJ and
        adds the Gaussian over all of them.

        ``interlayer_terms=False`` keeps the style's type layout (the ilp/tmd
        chalcogen split) but emits the ``lj`` interactions -- the settings the
        assembly run uses.
        """
        settings = self.config.settings
        style = interlayer_style(self.config.general)
        pm = PotentialManager(
            settings,
            potentials_dir=self.output_dir / "provenance" / "potentials",
            potentials_prefix=prefix,
        )
        pm.set_lj_overrides(self.config.lj_override)
        if style != 'lj':
            # Before registration: ilp/tmd's chalcogen split is decided there.
            set_interlayer_from_config(
                pm, self.config.general,
                dict(zip(["sheet"] + flake_names, [sheet_cfg] + flake_components)))
        # n_layers MUST match the stacked sheet file's per-layer types.
        # Registering it as one layer gave the sheet 2 types while the file
        # held 2*N: the flake's type offset then landed on the sheet's own
        # upper types, so `group flake` swallowed the sheet's S atoms and
        # the load, drive and v_fz acted on sheet + flake together.
        pm.register_component("sheet", sheet_cfg, n_layers=n_sheet_layers)
        for name, comp in zip(flake_names, flake_components):
            pm.register_component(name, comp, regions=[None, "corner"])
        if reserve_virtual_type:
            pm.register_virtual_atom()
        pm.add_self_interaction("sheet")
        lj_terms = style == 'lj' or not interlayer_terms
        if lj_terms:
            if pm.is_sheet_lj(sheet_cfg.pot_type):
                # One many-body instance per layer, so the layers need the
                # interlayer LJ explicitly (sheetonsheet/base.py does the same).
                pm.add_interlayer_interaction("sheet")
        for name in flake_names:
            pm.add_self_interaction(name)
        if lj_terms or style not in ('ilp', 'kc'):
            for name in flake_names:
                pm.add_cross_interaction("sheet", name)
            for i, name_i in enumerate(flake_names):
                for j in range(i + 1, len(flake_names)):
                    pm.add_cross_interaction(name_i, flake_names[j])
        if not lj_terms:
            pm.add_interlayer(["sheet"] + flake_names)
        pm.write_file(settings_path)
        return pm

    @staticmethod
    def _molecule_groups(pm: PotentialManager, n_sheet_layers: int,
                         flake_names: List[str], *, in_slide: bool) -> list:
        """One molecule per substrate layer and per flake component.

        In the slide script the substrate layers are the z-band groups
        ``layer_N`` and a single flake is the type group ``flake``. The
        assembly run has no such groups, so there (and for each component of
        a hetero flake) the entry carries its type ids for the macro to group.
        """
        groups: list = []
        for layer in range(n_sheet_layers):
            if in_slide:
                groups.append(f"layer_{layer + 1}")
            else:
                groups.append({"name": f"mol_layer_{layer + 1}",
                               "types": pm.types.get_layer_group_string("sheet", layer)})
        if in_slide and len(flake_names) == 1:
            groups.append("flake")
        else:
            for name in flake_names:
                groups.append({"name": f"mol_{name}",
                               "types": " ".join(map(str, pm.types.ids_by_component(name)))})
        return groups

    def run_dir(self) -> Path:
        """Return the run directory (where LAMMPS is launched: cwd = output_dir)."""
        return self.output_dir

    def _get_hpc_job_name(self) -> str:
        """Job name keyed by the substrate AND the flake material(s).

        Named off the substrate ``[2D]`` material alone before this fix -- the
        same collision as ``output_dir_for``/``material_token`` (see
        :func:`frictionsim2d.run.naming.flake_material_token`): two flake configs sharing
        the substrate but differing only in flake material produced identical
        HPC job identities. Mirrors that helper's rule exactly, so the two
        stay in lockstep: a hetero flake (``flake_sheets`` has 2 entries, each
        with a required ``mat``) always contributes; a single ``[Flake]``
        section only contributes when it sets an EXPLICIT ``mat`` of its own
        (``self.config.flake.mat`` stays ``None`` otherwise --
        ``_filled_flake_config`` returns a copy and never mutates
        ``self.config.flake``) -- the common single-material case (the flake
        reuses the substrate's material) stays byte-identical to before this
        fix.
        """
        sheet_mat = self.config.sheet.mat
        if len(self.config.flake_sheets) == 2:
            flake_token = '+'.join(fs.mat for fs in self.config.flake_sheets)
        elif self.config.flake is not None and self.config.flake.mat is not None:
            flake_token = self.config.flake.mat
        else:
            flake_token = None
        if flake_token is not None:
            return f"flake_{sheet_mat}+flake-{flake_token}"
        return f"flake_{sheet_mat}"

    def _record_twist_manifest(self, twist_angle_applied: float) -> None:
        """Record the applied flake twist into the run manifest (D3b).

        Augments the provenance manifest (``provenance/manifest.json``, created
        during provenance init) with a ``twist`` block. The flake is finite, so
        the ACHIEVED angle equals the requested angle (no commensurate snapping):
        both are logged as the same value so downstream (postprocessing/AiiDA)
        can read the realised twist straight from provenance.
        """
        data = self._load_or_init_manifest()
        data["twist"] = {
            "twist_angle_target_deg": twist_angle_applied,
            "twist_angle_applied_deg": twist_angle_applied,
        }
        manifest_path = self._write_manifest(data)
        logger.info(
            "Flake twist applied %.3f deg (finite flake: achieved == requested; "
            "recorded in %s)", twist_angle_applied, manifest_path,
        )

    def _assemble_flake_on_sheet(
        self,
        *,
        sheet_path: Path,
        flake_path: Path,
        n_sheet_types: int,
        pm: PotentialManager,
        settings_path: Path,
        bottom_flake_component: str,
    ):
        """Merge the sheet + flake into one data file via a LAMMPS assembly run.

        The substrate sheet is read unshifted (global types 1..S); the flake is
        read with an ``offset`` of ``n_sheet_types`` (so its types become
        S+1..S+F), shifted in xy to center over the sheet and in z to sit at the
        minimized flake--sheet gap above the sheet top. The reserved virtual type
        (if any) is created by the box (``create_box`` sizes to every registered
        type) and written as an atom-less extra type. Runs with ``cwd =
        output_dir`` so the settings' run-dir-relative potential paths resolve.

        Returns ``(combined_path, zhi)`` where ``zhi`` is the topmost atom z
        (the template adds the z-vacuum itself via ``change_box``).
        """
        sheet_at = read_lammps_data(sheet_path)
        cell = sheet_at.cell
        lx, ly, xy = float(cell[0, 0]), float(cell[1, 1]), float(cell[1, 0])
        s_pos = sheet_at.positions
        sheet_bot, sheet_top = float(s_pos[:, 2].min()), float(s_pos[:, 2].max())

        flake_at = read_lammps_data(flake_path)
        f_pos = flake_at.positions
        fxc = 0.5 * (float(f_pos[:, 0].min()) + float(f_pos[:, 0].max()))
        fyc = 0.5 * (float(f_pos[:, 1].min()) + float(f_pos[:, 1].max()))
        f_zmin, f_zmax = float(f_pos[:, 2].min()), float(f_pos[:, 2].max())

        # Guard against a flake wider than the (periodic) substrate footprint.
        f_span_x = float(f_pos[:, 0].max() - f_pos[:, 0].min())
        f_span_y = float(f_pos[:, 1].max() - f_pos[:, 1].min())
        if f_span_x >= lx or f_span_y >= ly:
            raise ValueError(
                f"Flake footprint ({f_span_x:.1f} x {f_span_y:.1f} A) does not fit "
                f"inside the substrate box ({lx:.1f} x {ly:.1f} A). Use a larger "
                f"[2D] x/y or a smaller flake edge_length."
            )

        gap = pm.calculate_gap("sheet", bottom_flake_component, buffer=0.5)
        dx = lx / 2.0 - fxc
        dy = ly / 2.0 - fyc
        dz = sheet_top + gap - f_zmin

        placed_zmax = f_zmax + dz
        zlo = min(sheet_bot, f_zmin + dz) - 15.0
        zhi_box = max(sheet_top, placed_zmax) + 15.0
        box_region = _box_region_cmd(lx, ly, xy, zlo, zhi_box)
        # `read_data ... add append` GROWS the box to take in the flake file's
        # own (larger, carving) box. Pin it back to the sheet's periodic cell,
        # or the substrate is a finite island with vacuum seams at the box edge.
        tilt = f" xy final {xy:.6f}" if abs(xy) > 1e-9 else ""
        restore_box = (
            f"change_box      all x final 0.0 {lx:.6f} y final 0.0 {ly:.6f} "
            f"z final {zlo:.6f} {zhi_box:.6f}{tilt} units box"
        )
        atom_style = _detect_atom_style(sheet_path)
        total_types = len(pm.types)
        masses_block = pm.get_masses_string()
        combined_path = self.output_dir / "flake_slide.data"
        mol_setup, mol_nofix = "", ""
        if pm.needs_molecule_ids:
            # ILP/KC need molecule IDs for this run 0 too; the combined file is
            # read back by a plain read_data, so write_data drops the fix's
            # Molecules section (`nofix`, see common/stack_layers.lmp).
            mol_setup = self.jinja_env.from_string(
                "{% import 'common/molecule_ids.lmp' as m %}"
                "{{ m.emit_molecule_ids(groups) }}").render(
                    groups=self._molecule_groups(
                        pm, pm.components["sheet"]["n_layers"],
                        [n for n in pm.components if n.startswith("flake")],
                        in_slide=False))
            mol_nofix = " nofix"

        # This run only places atoms (read_data + shift) and writes them back:
        # `run 0` computes nothing the result depends on. For an MLIP cell the
        # real settings would demand an MLIP-enabled LAMMPS just to *generate*
        # inputs, so include the zero surrogate instead (spec §3.3).
        if pm.has_mlip():
            settings_path = pm.write_surrogate_file(
                self.output_dir / "assemble_flake_slide.in.settings")

        script = f"""clear
units           metal
atom_style      {atom_style}
boundary        p p p

{box_region}
create_box      {total_types} box

read_data       {sheet_path} add append shift 0.0 0.0 0.0 group g_sheet offset 0 0 0 0 0 nocoeff
read_data       {flake_path} add append shift {dx:.6f} {dy:.6f} {dz:.6f} group g_flake offset {n_sheet_types} 0 0 0 0 nocoeff
{restore_box}

{masses_block}
include         {settings_path}
{mol_setup}run             0
write_data      {combined_path}{mol_nofix}
"""
        script_path = self.output_dir / "assemble_flake_slide.lmp"
        script_path.write_text(script, encoding="utf-8")
        _run_lammps(self.output_dir, script_path)

        if not combined_path.exists():
            raise RuntimeError(
                f"_assemble_flake_on_sheet: LAMMPS did not produce {combined_path}."
            )

        placed = read_lammps_data(combined_path).positions
        return combined_path, float(placed[:, 2].max())
