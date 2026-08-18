"""AFM Simulation Builder.

This module orchestrates the setup of a complete Atomic Force Microscopy (AFM)
simulation. It coordinates the construction of the Tip, Substrate, and Sheet,
generates the necessary potentials, and writes the LAMMPS input scripts.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..core.simulation_base import SimulationBase
from ..core.config import AFMSimulationConfig
from ..core.potential_manager import PotentialManager
from ..core.utils import get_model_dimensions, format_numeric_token, normalize_potential_type
from ..data.models import EV_A_TO_NN
from . import components

logger = logging.getLogger(__name__)

# Contact-cutoff rule (spec §5 F3): K_CONTACT * max UFF sigma over the
# tip-sheet element pairs, recorded in provenance with its source.
K_CONTACT = 1.5
_CONTACT_CUTOFF_FALLBACK = 5.0  # Angstrom; used only when the CIF is unreadable


class AFMSimulation(SimulationBase):
    """Builder for AFM simulations (Tip + Sheet + Substrate).

    Handles layer sweeps internally - when config.sheet.layers is a list,
    builds common components once and iterates over layer counts.
    """

    #: Run-phase (post-indentation) LAMMPS template. Subclasses swap this to
    #: reuse the identical build + system.in indentation with a different
    #: production run (e.g. the indent hold+retract replaces the slide).
    SLIDE_TEMPLATE: str = "afm/slide.lmp"

    #: The AFM run phase splits per load and per scan angle. 'scan_angle' is
    #: here because afm/slide.lmp really loops over the angle in-script (its
    #: scan_angle_is_sequence switch), so there is something to split.
    #: Subclasses whose SLIDE_TEMPLATE doesn't use the angle at all (e.g.
    #: IndentSimulation's afm/indent.lmp) must drop it, else the split would
    #: emit N physically-identical scripts.
    SPLIT_OUTER_LOOPS = frozenset({'force', 'scan_angle'})

    def __init__(self, config: AFMSimulationConfig, output_dir: str,
                    config_path: Optional[str] = None):
        super().__init__(config, output_dir, config_path=config_path)
        self.config: AFMSimulationConfig = config
        self.sheet_paths: Dict[int, Path] = {}
        self.tip_path: Path
        self.sub_path: Optional[Path]
        self.flake_path: Optional[Path] = None
        self.flake_dims: Dict[str, float] = {}
        self.z_positions: Dict[int, Dict[str, float]] = {}
        self.groups: Dict[int, Dict[str, str]] = {}
        self.pm: Dict[int, PotentialManager] = {}
        self.lat_c: Optional[float] = None
        self.sheet_dims: Dict[str, float] = {}
        self.sheet_unit_cell: Dict[str, float] = {}
        self.box_dims: Dict[str, float] = {}
        self.sheet_offset: Dict[str, float] = {'x': 0.0, 'y': 0.0}
        self.output_dir_layer: Dict[int, Path] = {}
        self.relative_run_dir_layer: Dict[int, Path] = {}

    def build(self) -> None:
        """Constructs the atomic systems and layout.
        
        If config.sheet.layers is a list, iterates over layer counts.
        """
        logger.info("Starting AFM Simulation Build...")

        build_dir = self.output_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        self._init_provenance()

        for n_layers in self.config.sheet.layers:
            logger.info("--- Building for %s layer(s) ---", n_layers)

            self.output_dir_layer[n_layers] = self.output_dir / f"L{n_layers}"
            self.relative_run_dir_layer[n_layers] = self.relative_run_dir / f"L{n_layers}"
            self._create_directories(self.output_dir_layer[n_layers])

            self._build_sheet_for_layers(build_dir, n_layers)

        if not self.sheet_dims:
            raise ValueError("No sheet dimensions were generated during AFM build.")

        self.tip_path, tip_radius, self.sub_path = self._build_components(build_dir)
        self.box_dims = self._get_substrate_box_dims()
        self._validate_final_substrate_dims()
        self.sheet_offset = self._compute_sheet_offset()

        for n_layers in self.config.sheet.layers:
            self.pm[n_layers] = self._generate_potentials(n_layers)
            self._calculate_z_positions(n_layers, tip_radius)
            self.write_inputs(n_layers)

        self._generate_hpc_scripts()
        logger.info("Build complete for all layer configurations.")

    def _build_sheet_for_layers(self, build_dir: Path, n_layers: int) -> None:
        """Build the sheet for one layer count and record its path/dims/lat_c.

        Extracted (behavior-preserving) from the :meth:`build` loop so subclasses
        can substitute a different sheet source (e.g. the D2 hetero stack) without
        overriding the whole ``build`` orchestration.
        """
        stacking_type = getattr(self.config.sheet, 'stack_type', 'AB')
        sheet_path, sheet_dims, lat_c = components.build_sheet(
            self.config.sheet, build_dir,
            stack_if_multi=True, settings=self.config.settings,
                n_layers_override=n_layers,
                stacking_type=stacking_type,
                edge_mode=(
                    self.config.settings.geometry.finite_sheet_afm_edge_mode
                    if self.config.general.finite_sheet else 'none'
                ),
                edge_width=self.config.settings.geometry.finite_sheet_edge_width,
                unit_cell_out=self.sheet_unit_cell,
        )
        self.sheet_paths[n_layers] = sheet_path
        if self.lat_c is None:
            self.lat_c = lat_c
            self.sheet_dims = dict(sheet_dims)

    def _material_id(self) -> str:
        """Material token used in the HPC job name.

        Single-material AFM: the sheet's material name. Overridden by
        :class:`~src.builders.hetero_afm.HeteroAFMSimulation` (inherited by
        :class:`~src.builders.hetero_afm.FiniteHeteroAFMSimulation`) to return
        the numeric-order ``+``-joined pair token instead, so two hetero
        configs sharing material 1 don't collide on HPC job identity (mirrors
        :meth:`~src.builders.indent.IndentSimulation._material_id`).

        Includes :meth:`_flake_id_suffix` so an AFM ``[flake]`` also keys the
        job name -- see that method's docstring for the collision this closes.
        """
        return self.config.sheet.mat + self._flake_id_suffix()

    def _flake_id_suffix(self) -> str:
        """``'+flake-<mat...>'`` when a flake section names a flake material
        that changes what gets built, else ``''``.

        Mirrors :func:`src.core.run.flake_material_token`'s rule exactly (kept
        in lockstep so the job name and the output directory always agree on
        whether a flake changes a config's identity): a hetero flake
        (``flake_sheets`` has 2 entries, ``mat`` required on each) always
        contributes; a single ``[flake]`` section contributes whenever it
        names a material (``self.config.flake.mat``) -- AFM's own flake,
        unlike the sheet-on-sheet flake slide's, has no substrate fallback
        (``components.build_flake_component`` needs its own
        ``mat``/``cif_path``/``pot_path`` to build the flake at all), so a
        configured AFM flake always contributes. No ``[flake]``/
        ``[flake-N]`` section at all -> ``''``, so the job name stays
        byte-identical to before this method existed.

        Shared (not overridden) by every ``_material_id`` in this family --
        :class:`~src.builders.indent.IndentSimulation` and
        :class:`~src.builders.hetero_afm.HeteroAFMSimulation`/
        :class:`~src.builders.hetero_afm.HeteroIndentSimulation` all inherit
        it unmodified and only append it to their own material token.
        """
        if len(self.config.flake_sheets) == 2:
            return '+flake-' + '+'.join(fs.mat for fs in self.config.flake_sheets)
        if self.config.flake is not None and self.config.flake.mat is not None:
            return f'+flake-{self.config.flake.mat}'
        return ''

    def _get_hpc_job_name(self) -> str:
        """Get AFM-specific job name."""
        return f"afm_{self._material_id()}"

    def _velocity_seed(self, n_layers: int) -> Optional[int]:
        """Explicit ``velocity create`` seed for system.in, or None for random.

        With ``[general] campaign_seed`` set, the seed derives deterministically
        from the run identity (spec §5 F5) so a rebuild is byte-identical.
        Without it, AFM keeps the legacy fresh-random-per-render behaviour. The
        indent builder overrides this with its own recorded per-(material,
        layer) seed manifest.
        """
        return self._derived_seed(n_layers, 'velocity')

    def _derived_seed(self, n_layers: int, channel: str) -> Optional[int]:
        """Deterministic per-(material, layer, channel) seed, or None.

        Loads and angles are LAMMPS in-file loops for the AFM scripts, so the
        per-run identity here is (material, layer, seed_index); load/angle are
        passed as None. None return (no campaign_seed) keeps the template's
        legacy Jinja-random fallback.
        """
        base = self.config.general.campaign_seed
        if base is None:
            return None
        from ..core.seeds import derive_seed
        return derive_seed(
            base,
            material=self.config.sheet.mat,
            layer=n_layers,
            load=None,
            angle=None,
            seed_index=self.config.general.seed_index,
            channel=channel,
        )

    def _init_provenance(self) -> None:
        """Initialize provenance folder and collect input files.

        Records every configured sheet, not just sheet 1: ``self.config.sheets``
        is the per-material stack folded from ``[2D-1]``, ``[2D-2]``, ... (or a
        one-element wrap of the single ``[2D]``), so a heterostructure build
        (``HeteroAFMSimulation``/``HeteroIndentSimulation``, which inherit this
        method unmodified) needs every material's cif/potential copied for
        portability, not just the first. The single-material path keeps the
        plain ``'sheet'`` component name (byte-identical to before this fix);
        only a genuine multi-material stack uses the numbered ``sheet_1``/
        ``sheet_2`` names already used elsewhere (e.g.
        :class:`~src.builders.hetero_slide.HeteroSheetOnSheetSimulation`'s own
        ``_init_provenance``, the potential registration's component names).
        """
        prov_dir = self.output_dir / 'provenance'
        prov_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Provenance folder initialized at: %s", prov_dir)

        sheets = self.config.sheets or [self.config.sheet]
        if len(sheets) > 1:
            provenance_components = [
                (f'sheet_{i + 1}', sheet) for i, sheet in enumerate(sheets)
            ]
        else:
            provenance_components = [('sheet', self.config.sheet)]
        provenance_components.append(('tip', self.config.tip))
        if self.config.sub is not None:
            provenance_components.append(('sub', self.config.sub))
        if self.config.flake is not None:
            provenance_components.append(('flake', self.config.flake))
        for component_name, config in provenance_components:
            self._add_component_files_to_provenance(component_name, config)

        logger.info("Initialized provenance folder: %s", prov_dir)

    def _maybe_twist_flake(self) -> None:
        """D3b: twist the built AFM flake in place when ``twist_angle`` is set.

        Rotates the finite flake's atoms about z in the fixed box frame (a
        genuine lattice twist vs the sheet below -- the flake is finite, so any
        angle is legal and no commensurate snapping applies). Shared by
        :class:`AFMSimulation` and :class:`~src.builders.hetero_afm.HeteroAFMSimulation`.
        Inert when ``twist_angle is None`` -> the AFM flake path stays
        byte-identical.
        """
        twist = getattr(self.config.general, "twist_angle", None)
        if twist is None:
            return
        from ..core.lattice_apply import rotate_flake_in_plane  # lazy: avoid cycle
        rotate_flake_in_plane(self.flake_path, float(twist))
        logger.info("Twisted AFM flake by %.3f deg (D3b)", float(twist))

    def _build_components(self, build_dir: Path) -> Tuple[Path, float, Path]:
        """Builds tip and substrate (shared across layer configurations).
        
        Sheet is built separately for each layer count.
        
        Returns:
            Tuple of (tip_path, tip_radius, sub_path); sub_path is None when config.sub is None (freestanding).
        """
        tip_path, tip_radius = components.build_tip(
            self.config.tip, build_dir, self.config.settings
        )
        logger.info("Built tip: %s", tip_path.name)

        if self.config.flake is not None:
            self.flake_path, self.flake_dims = components.build_flake_component(
                self.config.flake, build_dir
            )
            logger.info("Built flake: %s", self.flake_path.name)
            self._maybe_twist_flake()

        sub_path = None
        if self.config.sub is not None:
            sub_path = components.build_substrate(
                self.config.sub,
                build_dir,
                self.sheet_dims,
                settings=self.config.settings,
                target_x=self.config.sub.x if self.config.general.finite_sheet else None,
                target_y=self.config.sub.y if self.config.general.finite_sheet else None,
            )
            logger.info("Built substrate: %s", sub_path.name)

        if self.config.settings.thermostat.type == 'langevin':
            tip_height = tip_radius / self.config.settings.geometry.tip_reduction_factor

            components.apply_langevin_regions(
                component_name='tip',
                component_path=tip_path,
                config=self.config.tip,
                settings=self.config.settings,
                component_height=tip_height
            )
            logger.info("Applied Langevin regions to tip")

            if self.config.sub is not None:
                components.apply_langevin_regions(
                    component_name='sub',
                    component_path=sub_path,
                    config=self.config.sub,
                    settings=self.config.settings,
                    component_height=self.config.sub.thickness,
                    substrate_thickness=self.config.sub.thickness
                )
                logger.info("Applied Langevin regions to substrate")

        return tip_path, tip_radius, sub_path

    def _get_substrate_box_dims(self) -> Dict[str, float]:
        """Box dimensions: from the substrate .lmp, or from the sheet when freestanding."""
        if self.config.sub is None:
            sd = self.sheet_dims
            return {
                'xlo': float(sd['xlo']), 'xhi': float(sd['xhi']),
                'ylo': float(sd['ylo']), 'yhi': float(sd['yhi']),
                'zlo': 0.0, 'zhi': 0.0,
            }
        dims = get_model_dimensions(self.sub_path)
        required = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi']
        return {key: float(dims[key]) for key in required}

    def _validate_final_substrate_dims(self) -> None:
        """Validate finite-sheet constraints against final substrate dimensions."""
        if not self.config.general.finite_sheet:
            return

        sheet_x = float(self.sheet_dims['xhi'] - self.sheet_dims['xlo'])
        sheet_y = float(self.sheet_dims['yhi'] - self.sheet_dims['ylo'])
        sub_x = float(self.box_dims['xhi'] - self.box_dims['xlo'])
        sub_y = float(self.box_dims['yhi'] - self.box_dims['ylo'])

        if sub_x < sheet_x:
            raise ValueError(
                "Final substrate x from built .lmp is smaller than sheet x: "
                f"sub_x={sub_x}, sheet_x={sheet_x}."
            )
        if sub_y < sheet_y:
            raise ValueError(
                "Final substrate y from built .lmp is smaller than sheet y: "
                f"sub_y={sub_y}, sheet_y={sheet_y}."
            )

        lj_cutoff = float(self.config.settings.potential.lj_cutoff)
        margin_x = sub_x - sheet_x
        margin_y = sub_y - sheet_y
        if margin_x <= lj_cutoff:
            raise ValueError(
                "Final built substrate x-span delta must be greater than LJ cutoff. "
                f"Got margin_x={margin_x}, LJ cutoff={lj_cutoff}."
            )
        if margin_y <= lj_cutoff:
            raise ValueError(
                "Final built substrate y-span delta must be greater than LJ cutoff. "
                f"Got margin_y={margin_y}, LJ cutoff={lj_cutoff}."
            )

    def _compute_sheet_offset(self) -> Dict[str, float]:
        """Center the sheet inside the substrate footprint when finite-sheet is enabled."""
        if not self.config.general.finite_sheet:
            return {'x': 0.0, 'y': 0.0}

        sheet_center_x = (self.sheet_dims['xlo'] + self.sheet_dims['xhi']) / 2.0
        sheet_center_y = (self.sheet_dims['ylo'] + self.sheet_dims['yhi']) / 2.0
        box_center_x = (self.box_dims['xlo'] + self.box_dims['xhi']) / 2.0
        box_center_y = (self.box_dims['ylo'] + self.box_dims['yhi']) / 2.0
        return {
            'x': box_center_x - sheet_center_x,
            'y': box_center_y - sheet_center_y,
        }

    def _calculate_z_positions(self, n_layers: int, tip_radius: float) -> None:
        """Calculates vertical positions for all components.
        
        Args:
            n_layers: Number of sheet layers.
            tip_radius: Radius of the tip.
        """
        pm = self.pm[n_layers]
        lat_c = self.lat_c

        self.z_positions[n_layers] = {}
        if self.config.sub is None:
            # Freestanding: bottom sheet layer sits at a small vacuum pad; no substrate.
            gap_sub_sheet = 0.0
            self.z_positions[n_layers]['sub'] = 0.0
            sheet_base_z = 5.0
        else:
            gap_sub_sheet = pm.calculate_gap('sub', 'sheet', buffer=0.5)
            sub_thickness = self.config.sub.thickness
            self.z_positions[n_layers]['sub'] = 0.0
            sheet_base_z = sub_thickness + gap_sub_sheet
        self.z_positions[n_layers]['sheet'] = sheet_base_z

        lat_c = (self.config.sheet.lat_c or 6.0) if lat_c is None else lat_c
        sheet_stack_height = (n_layers - 1) * lat_c
        sheet_top_z = sheet_base_z + sheet_stack_height

        # The tip is placed a fixed clearance ABOVE its intended contact gap so it
        # equilibrates clear of the surface; the indentation script then lowers it by
        # a live-measured amount to sit exactly ``contact_gap`` above the surface it
        # indents (the flake if present, else the top sheet). ``contact_gap`` is the
        # material-derived gap, so "how close the tip lands" is computed, not hardcoded.
        start_clearance = self.config.settings.geometry.tip_start_clearance

        if self.config.flake is not None:
            # Flake sits above the top sheet; the tip then sits above the flake.
            gap_sheet_flake = pm.calculate_gap('sheet', 'flake', buffer=0.5)
            gap_flake_tip = pm.calculate_gap('flake', 'tip', buffer=0.5)
            flake_z = sheet_top_z + gap_sheet_flake
            self.z_positions[n_layers]['flake'] = flake_z
            # Physical flake thickness (strip the ±1 Å z-padding build_flake adds).
            flake_thickness = max(
                0.0,
                (self.flake_dims.get('zhi', 0.0) - self.flake_dims.get('zlo', 0.0)) - 2.0,
            )
            contact_gap = gap_flake_tip
            tip_z = flake_z + flake_thickness + gap_flake_tip + tip_radius + start_clearance
            logger.info(
                "Calculated gaps: Sub-Sheet=%.2fA, Sheet-Flake=%.2fA, Flake-Tip=%.2fA "
                "(tip start clearance %.1fA)",
                gap_sub_sheet, gap_sheet_flake, gap_flake_tip, start_clearance,
            )
        else:
            gap_sheet_tip = pm.calculate_gap('sheet', 'tip', buffer=0.5)
            contact_gap = gap_sheet_tip
            tip_z = sheet_top_z + gap_sheet_tip + tip_radius + start_clearance
            logger.info("Calculated gaps: Sub-Sheet=%.2fA, Sheet-Tip=%.2fA "
                        "(tip start clearance %.1fA)",
                        gap_sub_sheet, gap_sheet_tip, start_clearance)

        self.z_positions[n_layers]['tip'] = tip_z
        self.z_positions[n_layers]['tip_contact_gap'] = contact_gap

    def _generate_potentials(
        self,
        n_sheet_layers: int,
    ) -> PotentialManager:
        """Configures and writes the potential file using PotentialManager.
        
        Args:
            n_sheet_layers: Number of 2D material layers.
        
        Returns:
            Configured PotentialManager instance.
        """
        pm = PotentialManager(
            self.config.settings,
            potentials_dir=self.output_dir / "provenance" / "potentials",
            potentials_prefix=str(self.relative_run_dir / "provenance" / "potentials"),
        )
        pm.set_lj_overrides(self.config.lj_override)

        if self.config.sub is not None:
            pm.register_component('sub', self.config.sub)
        pm.register_component('tip', self.config.tip)

        sheet_needs_layer_types = (
            n_sheet_layers > 1 and
            pm.is_sheet_lj(self.config.sheet.pot_type)
        )
        sheet_edge_regions = [None, 'edge'] if self.config.general.finite_sheet and self.config.settings.geometry.finite_sheet_afm_edge_mode != 'none' else None
        pm.register_component(
            'sheet',
            self.config.sheet,
            n_layers=n_sheet_layers if sheet_needs_layer_types else 1,
            regions=sheet_edge_regions,
        )

        flake_cfg = self.config.flake
        has_flake = flake_cfg is not None
        if flake_cfg is not None:
            # regions=[None, 'corner'] interleaves core/corner types to match the
            # type splitting done by components.apply_flake_corner_types.
            pm.register_component('flake', flake_cfg, regions=[None, 'corner'])

        if self.config.settings.simulation.drive_method == 'virtual_atom':
            pm.register_virtual_atom()

        if self.config.sub is not None:
            pm.add_self_interaction('sub')
        pm.add_self_interaction('tip')
        pm.add_self_interaction('sheet')

        if self.config.sub is not None:
            pm.add_cross_interaction('sub', 'tip')
            pm.add_cross_interaction('sub', 'sheet')
        pm.add_cross_interaction('tip', 'sheet')

        if has_flake:
            pm.add_self_interaction('flake')
            if self.config.sub is not None:
                pm.add_cross_interaction('sub', 'flake')
            pm.add_cross_interaction('tip', 'flake')
            pm.add_cross_interaction('sheet', 'flake')

        if sheet_needs_layer_types and n_sheet_layers > 1:
            pm.add_interlayer_interaction('sheet')

        settings_path = self.output_dir_layer[n_sheet_layers] / "lammps" / "system.in.settings"
        pm.write_file(settings_path)

        self.groups[n_sheet_layers] = {}
        if self.config.sub is not None:
            self.groups[n_sheet_layers]['sub_types'] = pm.types.get_group_string('sub')
        self.groups[n_sheet_layers]['tip_types'] = pm.types.get_group_string('tip')
        self.groups[n_sheet_layers]['sheet_types'] = pm.types.get_group_string('sheet')

        if has_flake:
            self.groups[n_sheet_layers]['flake_types'] = pm.types.get_group_string('flake')
            corner_ids = pm.types.ids_by_component_region('flake', 'corner')
            self.groups[n_sheet_layers]['flake_corner_types'] = ' '.join(map(str, corner_ids))

        if sheet_needs_layer_types:
            for layer in range(n_sheet_layers):
                layer_key = f'sheet_l{layer+1}_types'
                self.groups[n_sheet_layers][layer_key] = pm.types.get_layer_group_string('sheet', layer)

        return pm

    def build_render_context(self, n_layers: int) -> Dict[str, object]:
        """Assemble the Jinja render context for one layer configuration.

        Split out from :meth:`write_inputs` so alternative scripts (e.g. the
        static PES tip scan) can reuse the identical geometry/potential context
        and only swap the rendered template.

        Args:
            n_layers: Number of sheet layers for this configuration.

        Returns:
            The template context dict shared by ``system_init``/``slide`` scripts.
        """
        pm = self.pm[n_layers]
        box_dims = self.box_dims
        sheet_offset = self.sheet_offset
        lat_c = self.lat_c
        z_positions = self.z_positions[n_layers]
        groups = self.groups[n_layers]
        sheet_path = self.sheet_paths[n_layers]
        tip_path = self.tip_path
        sub_path = self.sub_path

        total_types = len(pm.types)

        sim = self.config.settings.simulation
        out = self.config.settings.output

        reaxff_types = {'reaxff', 'reax/c'}
        uses_reaxff = (
            normalize_potential_type(self.config.sheet.pot_type) in reaxff_types or
            normalize_potential_type(self.config.tip.pot_type) in reaxff_types or
            (self.config.sub is not None and
             normalize_potential_type(self.config.sub.pot_type) in reaxff_types)
        )
        atom_style = 'charge' if uses_reaxff else 'atomic'

        if self.config.general.scan_speed is None:
            raise ValueError("scan_speed must be specified in [general] section")

        xlo, xhi = box_dims['xlo'], box_dims['xhi']
        ylo, yhi = box_dims['ylo'], box_dims['yhi']
        zhi_box = z_positions['tip'] + 50.0

        default_tip_x = ((self.sheet_dims['xlo'] + self.sheet_dims['xhi']) / 2.0) + sheet_offset['x']
        default_tip_y = ((self.sheet_dims['ylo'] + self.sheet_dims['yhi']) / 2.0) + sheet_offset['y']
        if self.config.tip.tip_x is not None:
            tip_x = float(self.config.tip.tip_x)
        else:
            tip_x = default_tip_x + float(self.config.tip.tip_x_offset)

        if self.config.tip.tip_y is not None:
            tip_y = float(self.config.tip.tip_y)
        else:
            tip_y = default_tip_y + float(self.config.tip.tip_y_offset)
        tip_z = z_positions['tip']

        sub_natypes = len(groups['sub_types'].split()) if 'sub_types' in groups else 0
        tip_natypes = len(groups['tip_types'].split())
        offset_2d = sub_natypes + tip_natypes

        # Flake placement (centered under the tip in xy, above the sheet in z).
        flake_cfg = self.config.flake
        flake_enabled = flake_cfg is not None
        flake_ctx: Dict[str, object] = {'flake_enabled': flake_enabled}
        if flake_cfg is not None and self.flake_path is not None:
            sheet_natypes = len(groups['sheet_types'].split())
            flake_center_x = (self.flake_dims['xlo'] + self.flake_dims['xhi']) / 2.0
            flake_center_y = (self.flake_dims['ylo'] + self.flake_dims['yhi']) / 2.0
            flake_ctx.update({
                'flake_file': f"{self.relative_run_dir}/build/{self.flake_path.name}",
                'flake_shift_x': tip_x - flake_center_x,
                'flake_shift_y': tip_y - flake_center_y,
                'flake_z': z_positions['flake'],
                'offset_flake': offset_2d + sheet_natypes,
                'flake_types': groups['flake_types'],
                'flake_corner_types': groups['flake_corner_types'],
                'flake_corner_k': flake_cfg.corner_spring_k,
            })

        # Tip viscous damping: an explicit dspring (including 0.0) overrides; otherwise the
        # viscous coefficient is auto-computed in the LAMMPS script from damping_ratio, the
        # driving spring constant, and the actual tip mass/atom count (see slide template).
        if self.config.tip.dspring is not None:
            damp_ev = self.config.tip.dspring / 0.016
            damping_ratio = None
        else:
            damp_ev = None
            damping_ratio = (
                self.config.tip.damping_ratio
                if self.config.tip.damping_ratio is not None
                else 0.1
            )

        context = {
            'temp': self.config.general.temp,
            'forces': self.config.general.force,
            'scan_angle_config': self.config.general.scan_angle,
            'scan_angle_force': self.config.general.scan_angle_force,
            'scan_speed_config': self.config.general.scan_speed,
            'xlo': xlo,
            'xhi': xhi,
            'ylo': ylo,
            'yhi': yhi,
            'zhi_box': zhi_box,
            'potential_file': f"{self.relative_run_dir_layer[n_layers]}/lammps/system.in.settings",
            'sub_file': f"{self.relative_run_dir}/build/{sub_path.name}" if sub_path is not None else None,
            'tip_file': f"{self.relative_run_dir}/build/{tip_path.name}",
            'sheet_file': f"{self.relative_run_dir}/build/{sheet_path.name}",
            'path_sub': f"{self.relative_run_dir}/build/{sub_path.name}" if sub_path is not None else None,
            'path_tip': f"{self.relative_run_dir}/build/{tip_path.name}",
            'path_sheet': f"{self.relative_run_dir}/build/{sheet_path.name}",
            'tip_x': tip_x,
            'tip_y': tip_y,
            'tip_z': tip_z,
            'contact_gap': z_positions['tip_contact_gap'],
            'sheet_shift_x': sheet_offset['x'],
            'sheet_shift_y': sheet_offset['y'],
            'sheet_z': z_positions['sheet'],
            'offset_2d': offset_2d,
            'results_file_pattern': (f"{self.relative_run_dir_layer[n_layers]}/results/"
                                    f"friction_f$(v_find)_a${{a}}_s${{speed}}_layer{n_layers}.txt"),
            'dump_file_pattern': (f"{self.relative_run_dir_layer[n_layers]}/visuals/"
                                    f"slide_f$(v_find)_a${{a}}_s${{speed}}.lammpstrj"),
            'dump_enabled': out.dump.get('slide', False),
            'z_sub': z_positions['sub'],
            'z_sheet': z_positions['sheet'],
            'z_tip': z_positions['tip'],
            'sub_types': groups.get('sub_types', ''),
            'tip_types': groups['tip_types'],
            'sheet_types': groups['sheet_types'],
            'ngroups': total_types,
            'sub_natypes': sub_natypes,
            'tip_natypes': tip_natypes,
            'timestep': sim.timestep,
            'thermo': sim.thermo,
            'neighbor_list': sim.neighbor_list,
            'neigh_modify_command': sim.neigh_modify_command,
            'run_steps': sim.slide_run_steps,
            'drive_method': sim.drive_method,
            'damp_ev': damp_ev,
            'damping_ratio': damping_ratio,
            'spring_ev': (self.config.general.driving_spring or 8.0) / 16.02,
            'virtual_offset': self.config.tip.r * 1.5, 
            'results_freq': out.results_frequency,
            'dump_freq': out.dump_frequency.get('slide', 1000),
            'tip_fix_group': 'tip_all' if self.config.settings.geometry.rigid_tip else 'tip_fix',
            'layer_group': '2D_all',
            'n_sheet_layers': n_layers,
            'lat_c': lat_c,
            'tip_radius': self.config.tip.r,
            'box_dims': box_dims,
            'thermostat_type': self.config.settings.thermostat.type,
            'use_langevin': self.config.settings.thermostat.type == 'langevin',
            'min_style': sim.min_style,
            'minimization_command': sim.minimization_command,
            'output_dir': f"{self.relative_run_dir_layer[n_layers]}/data",
            'dump_file': f"{self.relative_run_dir_layer[n_layers]}/visuals/system.lammpstrj",
            'atom_style': atom_style,
            'ev_a_to_nn': EV_A_TO_NN,
            'finite_sheet_enabled': bool(self.config.general.finite_sheet),
            'finite_sheet_edge_mode': self.config.settings.geometry.finite_sheet_afm_edge_mode,
            'finite_sheet_edge_width': self.config.settings.geometry.finite_sheet_edge_width,
            'finite_sheet_edge_spring_k': self.config.settings.geometry.finite_sheet_edge_spring_k,
            'velocity_seed': self._velocity_seed(n_layers),
            'seed_thermal_tip': self._derived_seed(n_layers, 'thermal_tip'),
            'seed_thermal_sub': self._derived_seed(n_layers, 'thermal_sub'),
        }
        context.update(flake_ctx)
        context['obs_ctx'] = self._build_obs_ctx(
            n_layers, context['results_file_pattern'],
            use_langevin=context['use_langevin'])
        return context

    def _build_obs_ctx(self, n_layers: int, results_file_pattern: str,
                       use_langevin: bool) -> Dict[str, object]:
        """Assemble the observable-registry context for the macro library.

        A single dict passed under the ``obs_ctx`` key — the macros receive it
        explicitly (the Jinja env has no StrictUndefined, so a missing key
        would silently render empty; the contract test's varlist assertions
        are the guard). Thermostat-work channels are filtered out for
        non-Langevin runs: their f_lang_* references would not exist.
        """
        from ..core.observables import channels_for, layer_channels

        static_layer_names = {f'com{ax}_L{lyr}'
                              for lyr in (1, 2) for ax in ('x', 'y', 'z')}
        win = [ob for ob in channels_for('afm', 'window')
               if ob.name not in static_layer_names]
        win += [ob for ob in layer_channels(n_layers) if ob.tier == 'window']
        if not use_langevin:
            win = [ob for ob in win if not ob.name.startswith('w_thermostat')]
        pa = [ob for ob in channels_for('afm', 'peratom')
              if ob.stream == 'results']

        def native(ob):
            return ob.rename_to or f'v_{ob.name}'

        contact_cutoff = round(self._contact_cutoff(), 4)

        # Dedupe compute lines: sibling channels (comx/comy/comz_L1) share one
        # compute; emitting it per channel would redefine it. Order-preserving.
        compute_lines: list = []
        variable_lines: list = []
        for ob in [*win, *pa]:
            if ob.lammps_compute:
                line = ob.lammps_compute.replace(
                    '{contact_cutoff}', str(contact_cutoff))
                if line not in compute_lines:
                    compute_lines.append(line)
            if ob.lammps_variable:
                variable_lines.append(ob.lammps_variable)

        return {
            'compute_lines': compute_lines,
            'variable_lines': variable_lines,
            'window_varlist': ' '.join(native(ob) for ob in win),
            'raw_varlist': ('v_fx_spr v_fy_spr v_fz_tip '
                            'v_comx_tip v_comy_tip v_comz_tip'),
            'peratom_varlist': ' '.join(native(ob) for ob in pa),
            'contact_cutoff': contact_cutoff,
            'results_file_pattern': results_file_pattern,
            'results_file_raw':
                results_file_pattern.replace('friction_', 'raw_'),
            'results_file_peratom':
                results_file_pattern.replace('friction_', 'peratom_'),
        }

    def _contact_cutoff(self) -> float:
        """Per-material contact cutoff from the tip-sheet UFF LJ sigma.

        The SW potential does not act between tip and sheet — the cross
        coupling is lj/cut with UFF element parameters — so the contact
        criterion is K_CONTACT * max(sigma_ij) over the Si-(sheet element)
        pairs (spec §5 F3). A fixed absolute cutoff would conflate "more atoms
        in contact" with "larger atoms". Capped below the pair cutoff (LAMMPS
        errors if coord/atom's cutoff exceeds it). Falls back to a documented
        default when the sheet elements cannot be read (dummy-CIF test
        configs); provenance records which route was taken.
        """
        from itertools import product
        from ..core.utils import lj_params, cifread

        lj_cut = getattr(self.config.settings.potential, 'lj_cutoff', 11.0)
        try:
            sheet_els = cifread(self.config.sheet.cif_path)['elements']
            tip_els = [self.config.tip.mat]
            sigma_max = max(lj_params(a, b)[1]
                            for a, b in product(tip_els, sheet_els))
            self._contact_cutoff_source = f'uff_sigma_k{K_CONTACT}'
            cut = K_CONTACT * sigma_max
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Contact cutoff fallback: could not derive UFF sigma for "
                "sheet '%s'; using %.1f A.",
                self.config.sheet.mat, _CONTACT_CUTOFF_FALLBACK)
            self._contact_cutoff_source = 'fallback_default'
            cut = _CONTACT_CUTOFF_FALLBACK
        return min(cut, 0.95 * lj_cut)

    def write_inputs(self, n_layers: int) -> None:
        """Generates the LAMMPS input scripts.

        Args:
            n_layers: Number of sheet layers for this configuration.
        """
        logger.info("Writing LAMMPS inputs...")

        context = self.build_render_context(n_layers)
        output_root = self.output_dir_layer[n_layers]

        # The single system.in always keeps the full force list so it writes
        # every load_<X>N.data file the slide phase(s) read.
        init_script = self.render_template("afm/system_init.lmp", context)
        self.write_file("lammps/system.in", init_script, output_root)

        self._write_slide_scripts(context, output_root)

        logger.info("Inputs written to %s/lammps/", output_root)

    def _write_slide_scripts(self, context: Dict, output_root: Path) -> None:
        """Write the slide phase script(s).

        With ``outer_loop = force`` (and more than one load) each load is emitted
        as its own ``slide_f<load>N.in`` reading its shared ``load_<load>N.data``,
        mirroring the sheet-on-sheet ``outer_loop`` split. With ``outer_loop =
        scan_angle`` (and more than one angle) each angle is emitted as its own
        ``slide_a<angle>.in``, passing a scalar ``scan_angle_config`` so the
        template's own ``scan_angle_is_sequence`` switch (afm/slide.lmp) collapses
        the in-script angle loop -- no parallel code path needed. Both split
        modes become independent HPC jobs (the two-phase generator makes them
        depend on the shared system.in phase). Otherwise a single ``slide.in``
        loops over the loads/angles in-script, as before.

        The angle split does not carry the ``scan_angle_force`` selector: that
        option restricts which force value(s) the scan_angle list applies to
        (afm/slide.lmp's angle_loop gate), but the split applies its one fixed
        angle to every force in every per-angle script -- so the selector would
        be silently dropped, changing what a force-gated scan means. Reject the
        combination instead. :func:`src.core.run.outer_loop_conflict` rejects it
        first (before the build, and only for the builders that really split),
        so this raise is the backstop for builds driven directly.

        Only applies when ``'scan_angle' in self.SPLIT_OUTER_LOOPS`` (the
        default). :class:`IndentSimulation` drops it because its
        ``afm/indent.lmp`` template declares the angle variable but never
        uses it (no lateral drive) -- splitting there would just emit N
        physically-identical scripts.
        """
        outer_loop = getattr(self.config.general, 'outer_loop', None)
        force = self.config.general.force
        forces = force if isinstance(force, list) else ([] if force is None else [force])

        # Keep a single load on slide.in so the combined system+slide HPC path
        # (which references the literal name slide.in) still applies.
        if (outer_loop == 'force' and len(forces) > 1
                and 'force' in self.SPLIT_OUTER_LOOPS):
            for load in forces:
                slide_context = dict(context)
                slide_context['forces'] = load
                script = self.render_template(self.SLIDE_TEMPLATE, slide_context)
                script_name = f"slide_f{format_numeric_token(float(load))}N.in"
                self.write_file(f"lammps/{script_name}", script, output_root)
            logger.info("Split slide phase into %d per-load scripts.", len(forces))
            return

        scan_angle = self.config.general.scan_angle
        angles = scan_angle if isinstance(scan_angle, list) else (
            [] if scan_angle is None else [scan_angle]
        )
        if (outer_loop == 'scan_angle' and len(angles) > 1
                and 'scan_angle' in self.SPLIT_OUTER_LOOPS):
            if self.config.general.scan_angle_force is not None:
                raise ValueError(
                    "outer_loop='scan_angle' cannot be combined with "
                    "scan_angle_force: splitting into one script per angle "
                    "applies each angle to every force value, silently "
                    "dropping the force-selective filter. Remove "
                    "scan_angle_force, or drop outer_loop to keep the "
                    "in-script angle loop (which honors it)."
                )
            for angle in angles:
                slide_context = dict(context)
                slide_context['scan_angle_config'] = float(angle)
                script = self.render_template(self.SLIDE_TEMPLATE, slide_context)
                script_name = f"slide_a{format_numeric_token(float(angle))}.in"
                self.write_file(f"lammps/{script_name}", script, output_root)
            logger.info("Split slide phase into %d per-angle scripts.", len(angles))
            return

        slide_script = self.render_template(self.SLIDE_TEMPLATE, context)
        self.write_file("lammps/slide.in", slide_script, output_root)
