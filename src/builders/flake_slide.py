"""D4 flake-driven slide: a finite flake driven across a substrate sheet.

A finite flake (single-material or a hetero bilayer) sits on a substrate sheet
at the minimized flake--sheet gap and is driven laterally by a virtual-atom /
COM spring (no tip, no substrate slab). The substrate sheet's bottom band is
fixed; its interior + the flake are thermostatted; lateral friction is read
from the bottom-sheet reaction force (the many-body ``sw`` observable).

This reuses the D1 slide body (``templates/hetero/slide.lmp``) via a dedicated
``templates/flake/slide.lmp`` in which the driven group is the flake (by type),
the tip/load blocks are dropped, and the substrate-sheet layer groups are the
z-region bands.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ase.io import read
from jinja2 import Environment

from ..core.config import SheetOnSheetSimulationConfig
from ..core.mlip import validate_mlip_cell
from ..core.potential_manager import PotentialManager
from ..core.simulation_base import SimulationBase
from ..data.models import EV_A_TO_NN, EV_A3_TO_GPA, NM_TO_EV_A2
from ..interfaces.jinja import PackageLoader
from . import components
from .hetero import _box_region_cmd, _detect_atom_style, _run_lammps

logger = logging.getLogger(__name__)

PAD = 1.0  # Å pad above/below each sheet layer's atom z-range (mirrors hetero_slide.PAD)

# Same Jinja environment construction as SimulationBase.__init__ and
# hetero_slide.render_hetero_slide: package-resource loader, trimmed blocks.
_ENV = Environment(
    loader=PackageLoader('src.templates'),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_flake_slide(context: Dict) -> str:
    """Render ``flake/slide.lmp`` with the given context.

    Thin module-level helper mirroring :func:`hetero_slide.render_hetero_slide`:
    renders the flake-driven slide template without needing a full builder
    instance (used by the render unit tests).
    """
    return _ENV.get_template("flake/slide.lmp").render(context)


def compute_sheet_zbands(sheet_path, n_layers: int, pad: float = PAD) -> List[Dict]:
    """Split a stacked substrate sheet's atoms into ``n_layers`` z-region bands.

    The substrate sheet's layers share atom types (many-body ``sw``), so the
    template groups them by z-slab (region), exactly like the D1 hetero slide.
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
    z = np.sort(read(str(sheet_path), format="lammps-data").get_positions()[:, 2])
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
    single-material substrate sheet at the minimized flake--sheet gap, and drives
    the flake laterally by a virtual-atom / COM spring. The substrate sheet's
    bottom band is fixed; its interior + the flake are thermostatted; lateral
    friction is read from the bottom-sheet reaction force (the many-body ``sw``
    observable). Routed from ``run sheetonsheet`` when a ``[Flake]`` section is
    present.

    Mirrors :class:`~src.builders.hetero_slide.HeteroSheetOnSheetSimulation`'s
    run-dir-relative convention: everything (``flake_slide.data``,
    ``lammps/system.in.settings``, ``provenance/potentials/...``, ``results/...``)
    is written into ``output_dir`` and referenced run-dir-relatively, so LAMMPS
    is launched with ``cwd = output_dir``.
    """

    #: A flake slide needs a fixed bottom band + at least one thermostatted
    #: interior band -> the substrate sheet must have >= 2 layers.
    MIN_SHEET_LAYERS: int = 2

    #: Like the hetero slide, this builder writes a single lammps/slide.in and
    #: never reads outer_loop, so a [Flake] section collapses any configured
    #: split to one job. Explicit (the base default is the same) because the
    #: reroute from sheet-on-sheet is silent.
    SPLIT_OUTER_LOOPS = frozenset()

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

        # 1. Substrate sheet (homogeneous, stacked >=2 layers). Layer groups are
        #    z-region bands (shared sw types), so register the sheet as ONE
        #    component (n_layers=1) -- the D1 hetero convention.
        sheet_path, _sheet_dims, lat_c = components.build_sheet(
            sheet_cfg, build_dir, stack_if_multi=True, settings=settings,
            n_layers_override=n_sheet_layers,
            stacking_type=getattr(sheet_cfg, "stack_type", "AA"),
        )

        # 2. Flake (single-material reuses [2D]; hetero flake from [Flake-1/2]).
        hetero_flake = len(self.config.flake_sheets) == 2
        if hetero_flake:
            flake_path, _flake_dims = components.build_hetero_flake_component(
                self.config.flake_sheets, settings, self.config.flake, build_dir,
                hetero_stacking=getattr(self.config.general, "hetero_stacking", "grouped"),
            )
            flake_names = ["flake_1", "flake_2"]
            flake_components = list(self.config.flake_sheets)
        else:
            filled_flake = self._filled_flake_config()
            flake_path, _flake_dims = components.build_flake_component(
                filled_flake, build_dir
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
            from ..core.lattice_apply import rotate_flake_in_plane  # lazy: avoid cycle
            rotate_flake_in_plane(flake_path, float(twist))   # in-place twist, box unchanged
            self._record_twist_manifest(float(twist))
            logger.info("Twisted flake by %.3f deg (D3b incommensurate)", float(twist))

        # 3. ONE PotentialManager: sheet -> flake(s) -> reserved virtual, so global
        #    atom types are sheet 1..S, flake S+1..S+F, virtual S+F+1. Run-dir-
        #    relative potential staging (LAMMPS runs with cwd = output_dir).
        # Capability-matrix rules across the whole cell (spec §1.1/§3.4).
        validate_mlip_cell(
            sheet_cfg,
            components=[(name, comp)
                        for name, comp in zip(flake_names, flake_components)],
            constraint_mode=settings.simulation.constraint_mode,
        )

        pm = PotentialManager(
            settings,
            potentials_dir=self.output_dir / "provenance" / "potentials",
            potentials_prefix="provenance/potentials",
        )
        pm.set_lj_overrides(self.config.lj_override)
        pm.register_component("sheet", sheet_cfg, n_layers=1)
        n_sheet_types = len(pm.types)
        for name, comp in zip(flake_names, flake_components):
            pm.register_component(name, comp, regions=[None, "corner"])

        reserve_virtual_type = settings.simulation.drive_method == "virtual_atom"
        if reserve_virtual_type:
            pm.register_virtual_atom()

        pm.add_self_interaction("sheet")
        for name in flake_names:
            pm.add_self_interaction(name)
        for name in flake_names:
            pm.add_cross_interaction("sheet", name)
        for i in range(len(flake_names)):
            for j in range(i + 1, len(flake_names)):
                pm.add_cross_interaction(flake_names[i], flake_names[j])

        settings_path = self.output_dir / "lammps" / "system.in.settings"
        pm.write_file(settings_path)

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
            pm=pm,
            settings_path=settings_path,
            bottom_flake_component=flake_names[0],
        )

        # 5. Substrate-sheet z-region bands (from the standalone, unshifted sheet
        #    file -- it is placed at shift 0 0 0 in the combined data).
        sheet_layers = compute_sheet_zbands(sheet_path, n_sheet_layers)

        # 6. Render context + slide.in.
        at = read(str(combined_path), format="lammps-data")
        cell = np.array(at.cell)
        general = self.config.general
        sim = settings.simulation
        out = settings.output
        flake_cfg = self.config.flake

        context = {
            "temp": general.temp,
            "xlo": 0.0,
            "xhi": float(cell[0, 0]),
            "ylo": 0.0,
            "yhi": float(cell[1, 1]),
            "zhi": zhi,
            "data_file": combined_path.name,
            "potential_file": "lammps/system.in.settings",
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
            # LOAD-BEARING, both keys. The templates render them behind
            # `| default(...)`, which does NOT raise when a key is absent -- it
            # silently renders nothing (directives) or the classical branch
            # (mlip_sheet), the build still reports success, and the omission
            # surfaces only when LAMMPS aborts or balloons its memory. This
            # builder subclasses SimulationBase directly, so it inherits
            # neither the AFM nor the sheet-on-sheet wiring.
            "mlip_sheet": pm.has_mlip(),
            "mlip_deck_directives": pm.deck_directives(),
            "has_internal_lj": True,
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
            "results_file_pattern": "results/friction_p${pressure}_a${a}_s${speed}",
            "dump_file_pattern": "visuals/slide_p${pressure}_a${a}_s${speed}.lammpstrj",
            "driving_spring_ev": (general.driving_spring or 50.0) / NM_TO_EV_A2,
            "bond_spring_ev": (general.bond_spring or 80.0) / NM_TO_EV_A2,
            "lat_c": lat_c if lat_c is not None else settings.geometry.lat_c_default,
            "ev_a_to_nn": EV_A_TO_NN,
            "ev_a3_to_gpa": EV_A3_TO_GPA,
            "sheet_layers": sheet_layers,
            "flake_group_types": " ".join(map(str, flake_group_ids)),
            "flake_corner_types": " ".join(map(str, flake_corner_ids)),
            "flake_corner_k": flake_cfg.corner_spring_k,
            "corner_hold": getattr(flake_cfg, "corner_hold", True),
            "virtual_offset": settings.geometry.lat_c_default,
        }

        script = self.render_template("flake/slide.lmp", context)
        self.write_file("lammps/slide.in", script)
        logger.info("Wrote flake slide script to %s/lammps/slide.in", self.output_dir)

        try:
            self._generate_hpc_scripts()
        except Exception as exc:  # pragma: no cover - defensive, non-fatal
            logger.warning(
                "HPC script generation skipped for flake slide (%s); the "
                "homogeneous HPC path assumes sim-root-relative paths while the "
                "flake slide is run-dir-relative.", exc,
            )

    def run_dir(self) -> Path:
        """Return the run directory (where LAMMPS is launched: cwd = output_dir)."""
        return self.output_dir

    def _get_hpc_job_name(self) -> str:
        return f"flake_{self.config.sheet.mat}"

    def _record_twist_manifest(self, twist_angle_applied: float) -> None:
        """Record the applied flake twist into the run manifest (D3b).

        Augments the provenance manifest (``provenance/manifest.json``, created
        during provenance init) with a ``twist`` block. The flake is finite, so
        the ACHIEVED angle equals the requested angle (no commensurate snapping):
        both are logged as the same value so downstream (postprocessing/AiiDA)
        can read the realised twist straight from provenance.
        """
        import json  # pylint: disable=import-outside-toplevel
        manifest_path = self.output_dir / "provenance" / "manifest.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"version": "1.0", "files": []}
        data["twist"] = {
            "twist_angle_target_deg": twist_angle_applied,
            "twist_angle_applied_deg": twist_angle_applied,
        }
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
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
        sheet_at = read(str(sheet_path), format="lammps-data")
        cell = np.array(sheet_at.cell)
        lx, ly, xy = float(cell[0, 0]), float(cell[1, 1]), float(cell[1, 0])
        s_pos = sheet_at.get_positions()
        sheet_bot, sheet_top = float(s_pos[:, 2].min()), float(s_pos[:, 2].max())

        flake_at = read(str(flake_path), format="lammps-data")
        f_pos = flake_at.get_positions()
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
        atom_style = _detect_atom_style(sheet_path)
        total_types = len(pm.types)
        masses_block = pm.get_masses_string()
        combined_path = self.output_dir / "flake_slide.data"

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

{masses_block}
include         {settings_path}
run             0
write_data      {combined_path}
"""
        script_path = self.output_dir / "assemble_flake_slide.lmp"
        script_path.write_text(script, encoding="utf-8")
        _run_lammps(self.output_dir, script_path)

        if not combined_path.exists():
            raise RuntimeError(
                f"_assemble_flake_on_sheet: LAMMPS did not produce {combined_path}."
            )

        placed = read(str(combined_path), format="lammps-data").get_positions()
        return combined_path, float(placed[:, 2].max())
