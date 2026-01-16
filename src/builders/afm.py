"""AFM Simulation Builder.

This module orchestrates the setup of a complete Atomic Force Microscopy (AFM)
simulation. It coordinates the construction of the Tip, Substrate, and Sheet,
generates the necessary potentials, and writes the LAMMPS input scripts.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

from src.core.simulation_base import SimulationBase
from src.core.config import AFMSimulationConfig
from src.core.potential_manager import PotentialManager
from src.builders import components

logger = logging.getLogger(__name__)


class AFMSimulation(SimulationBase):
    """Builder for AFM simulations (Tip + Sheet + Substrate).
    
    Handles layer sweeps internally - when config.sheet.layers is a list,
    builds common components once and iterates over layer counts.
    """

    def __init__(self, config: AFMSimulationConfig, output_dir: str):
        super().__init__(config, output_dir)
        self.config: AFMSimulationConfig = config  # Type hinting alias
        
        # Base output directory (layer subdirs created within)
        self.base_output_dir = self.output_dir

        # State to track build artifacts (reset per layer)
        self.structure_paths: Dict[str, Path] = {}
        self.z_positions: Dict[str, float] = {}
        self.groups: Dict[str, str] = {}
        self.pm: Optional[PotentialManager] = None
        
        # Shared components (built once, reused across layers)
        self._tip_path: Optional[Path] = None
        self._tip_radius: Optional[float] = None
        self._sub_path: Optional[Path] = None
        self._monolayer_path: Optional[Path] = None
        self._monolayer_dims: Optional[dict] = None
        self._base_lat_c: Optional[float] = None
        self._pot_counts: Optional[dict] = None
        self._total_pot_types: Optional[int] = None

    def build(self) -> None:
        """Constructs the atomic systems and layout.
        
        If config.sheet.layers is a list, iterates over layer counts.
        """
        logger.info("Starting AFM Simulation Build...")
        
        # Normalize layers to list
        layers = self.config.sheet.layers
        if isinstance(layers, int):
            layers = [layers]
        elif not layers:
            layers = [1]
        
        # Initialize provenance folder
        self._init_provenance()
        
        # 1. Iterate over layer counts (build per layer)
        for n_layers in layers:
            logger.info(f"--- Building for {n_layers} layer(s) ---")
            
            # Set output directory for this layer count
            if len(layers) > 1:
                self.output_dir = self.base_output_dir / f"L{n_layers}"
            else:
                self.output_dir = self.base_output_dir
            
            self._create_directories()
            
            # Build directory scoped to this layer
            layer_build_dir = self.output_dir / "build"
            if layer_build_dir.exists():
                shutil.rmtree(layer_build_dir)
            layer_build_dir.mkdir(parents=True, exist_ok=True)
            
            # Build common components inside the layer build dir
            self._build_common_components(layer_build_dir)
            
            # Build sheet for this layer count
            self._build_sheet_for_layers(n_layers, layer_build_dir)
            
            # Use paths from the layer build directory
            self.structure_paths['tip'] = layer_build_dir / self._tip_path.name
            self.structure_paths['sub'] = layer_build_dir / self._sub_path.name
            
            # Generate potentials for this layer count
            self.pm = self._generate_potentials(n_sheet_layers=n_layers)
            
            # Calculate vertical layout
            self._calculate_z_positions(n_layers)
            
            # Write inputs for this layer
            self.write_inputs()
        
        logger.info("Build complete for all layer configurations.")

    def _init_provenance(self) -> None:
        """Initialize provenance folder and collect input files."""
        from src.core.utils import get_material_path, get_potential_path
        
        # Initialize the provenance folder
        prov_dir = self.base_output_dir / 'provenance'
        prov_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect CIF files for all components
        for component_name, config in [
            ('sheet', self.config.sheet),
            ('tip', self.config.tip),
            ('sub', self.config.sub)
        ]:
            if hasattr(config, 'cif_path') and config.cif_path:
                self.add_to_provenance(config.cif_path, 'cif')
            elif hasattr(config, 'mat') and config.mat:
                # Try to find CIF file for this material
                try:
                    cif_path = get_material_path(config.mat, 'cif')
                    if cif_path:
                        self.add_to_provenance(cif_path, 'cif')
                except Exception:
                    pass
        
        # Collect potential files for all components
        for component_name, config in [
            ('sheet', self.config.sheet),
            ('tip', self.config.tip),
            ('sub', self.config.sub)
        ]:
            if hasattr(config, 'pot_path') and config.pot_path:
                self.add_to_provenance(config.pot_path, 'potential')
            elif hasattr(config, 'pot') and config.pot:
                try:
                    pot_path = get_potential_path(config.pot)
                    if pot_path:
                        self.add_to_provenance(pot_path, 'potential')
                except Exception:
                    pass
        
        logger.info(f"Initialized provenance folder: {prov_dir}")

    def _build_common_components(self, build_dir: Path) -> None:
        """Builds tip, substrate, and monolayer (components shared across layers)."""
        
        # Build Tip (once)
        self._tip_path, self._tip_radius = components.build_tip(
            self.config.tip, self.atomsk, build_dir, self.config.settings
        )
        logger.info(f"Built tip: {self._tip_path.name}")
        
        # Build Monolayer (once) - don't stack yet
        (self._monolayer_path, self._monolayer_dims, 
         self._base_lat_c, self._pot_counts, self._total_pot_types) = components.build_monolayer(
            self.config.sheet, self.atomsk, build_dir, self.config.settings
        )
        logger.info(f"Built monolayer: {self._monolayer_path.name}")
        
        # Build Substrate (once) - uses monolayer dims
        self._sub_path = components.build_substrate(
            self.config.sub, self.atomsk, build_dir, self._monolayer_dims,
            settings=self.config.settings
        )
        logger.info(f"Built substrate: {self._sub_path.name}")

    def _build_sheet_for_layers(self, n_layers: int, build_dir: Path) -> None:
        """Stacks the monolayer to create n-layer sheet."""
        
        if n_layers == 1:
            # Just copy the monolayer
            sheet_path = build_dir / f"{self.config.sheet.mat}_1.lmp"
            if sheet_path != self._monolayer_path:
                shutil.copy(self._monolayer_path, sheet_path)
            self.structure_paths['sheet'] = sheet_path
            self.lat_c = self._base_lat_c
            self.sheet_dims = self._monolayer_dims
        else:
            # Stack multiple layers
            stacked_path = build_dir / f"{self.config.sheet.mat}_{n_layers}.lmp"
            stacked_path, lat_c = components.stack_multilayer_sheet(
                base_layer_path=self._monolayer_path,
                config=self.config.sheet,
                output_path=stacked_path,
                box_dims=self._monolayer_dims,
                n_layers=n_layers,
                types_per_layer=self._total_pot_types,
                pot_counts=self._pot_counts,
                lat_c=self._base_lat_c,
                settings=self.config.settings
            )
            self.structure_paths['sheet'] = stacked_path
            self.lat_c = lat_c
            self.sheet_dims = self._monolayer_dims

    def _calculate_z_positions(self, n_layers: int) -> None:
        """Calculates vertical positions for all components."""
        
        gap_sub_sheet = self.pm.calculate_gap('sub', 'sheet', buffer=0.5)
        gap_sheet_tip = self.pm.calculate_gap('sheet', 'tip', buffer=0.5)
        
        logger.info(f"Calculated gaps: Sub-Sheet={gap_sub_sheet:.2f}A, Sheet-Tip={gap_sheet_tip:.2f}A")
        
        sub_thickness = self.config.sub.thickness
        
        # Position 1: Substrate (Base)
        self.z_positions['sub'] = 0.0
        
        # Position 2: Sheet (Above Substrate)
        sheet_base_z = sub_thickness + gap_sub_sheet
        self.z_positions['sheet'] = sheet_base_z
        
        # Position 3: Tip (Above Sheet)
        sheet_stack_height = (n_layers - 1) * self.lat_c
        tip_z = sheet_base_z + sheet_stack_height + gap_sheet_tip + self._tip_radius
        self.z_positions['tip'] = tip_z

    def _generate_potentials(
        self, 
        n_sheet_layers: int = 1
    ) -> PotentialManager:
        """Configures and writes the potential file using PotentialManager.
        
        Args:
            n_sheet_layers: Number of 2D material layers.
            
        Returns:
            Configured PotentialManager instance.
        """
        pm = PotentialManager(self.config.settings)

        # Register components (PM internally knows if Langevin is used)
        pm.register_component('sub', self.config.sub)
        pm.register_component('tip', self.config.tip)
        
        # Sheet with layer-specific types if multiple layers and requires LJ
        sheet_needs_layer_types = (
            n_sheet_layers > 1 and 
            pm.is_sheet_lj(self.config.sheet.pot_type)
        )
        pm.register_component(
            'sheet', 
            self.config.sheet, 
            n_layers=n_sheet_layers if sheet_needs_layer_types else 1
        )

        # Define Self-Interactions (many-body potentials)
        pm.add_self_interaction('sub')
        pm.add_self_interaction('tip')
        pm.add_self_interaction('sheet')

        # Cross Interactions (LJ Mixing between components)
        pm.add_cross_interaction('sub', 'tip')
        pm.add_cross_interaction('sub', 'sheet')
        pm.add_cross_interaction('tip', 'sheet')
        
        # Interlayer interactions for multi-layer sheets
        if sheet_needs_layer_types and n_sheet_layers > 1:
            pm.add_interlayer_interaction('sheet')

        # Write the potential file
        pm.write_file(self.output_dir / "lammps" / "system.in.settings")

        # Store group ID strings for LAMMPS grouping
        self.groups['sub_types'] = pm.get_group_string('sub')
        self.groups['tip_types'] = pm.get_group_string('tip')
        self.groups['sheet_types'] = pm.get_group_string('sheet')
        
        # Store layer-specific groups if applicable
        if sheet_needs_layer_types:
            for layer in range(n_sheet_layers):
                self.groups[f'sheet_l{layer+1}_types'] = pm.get_layer_group_string('sheet', layer)

        return pm

    def write_inputs(self) -> None:
        """Generates the LAMMPS input scripts."""
        logger.info("Writing LAMMPS inputs...")

        # Get total types from PotentialManager
        total_types = self.pm.get_total_types() if self.pm else len(set(
            t for s in self.groups.values() for t in s.split()
        ))
        
        sim = self.config.settings.simulation
        out = self.config.settings.output

        # Calculate box dimensions from sheet dims
        xlo, xhi = self.sheet_dims['xlo'], self.sheet_dims['xhi']
        ylo, yhi = self.sheet_dims['ylo'], self.sheet_dims['yhi']
        zhi_box = self.z_positions['tip'] + 50.0  # Extra space above tip
        
        # Tip centered on sheet
        tip_x = (xlo + xhi) / 2.0
        tip_y = (ylo + yhi) / 2.0
        tip_z = self.z_positions['tip']
        
        # Type offsets for read_data append
        sub_natypes = len(self.groups['sub_types'].split())
        tip_natypes = len(self.groups['tip_types'].split())
        offset_2d = sub_natypes + tip_natypes

        context = {
            # Temperature and forces/angles for LAMMPS loops
            'temp': self.config.general.temp,
            'forces': self.config.general.force,  # List for LAMMPS index variable
            'angles': self.config.general.scan_angle,  # List for LAMMPS index variable
            'speed': self.config.tip.s,

            # Box dimensions
            'xlo': xlo,
            'xhi': xhi,
            'ylo': ylo,
            'yhi': yhi,
            'zhi_box': zhi_box,
            
            # File paths for LAMMPS read_data
            # Relative paths from the run directory (where python script is executed)
            'data_file': str(self.output_dir / "build" / self.structure_paths['sheet'].name),
            'potential_file': str(self.output_dir / "lammps" / "system.in.settings"),
            'sub_file': str(self.output_dir / "build" / self.structure_paths['sub'].name),
            'tip_file': str(self.output_dir / "build" / self.structure_paths['tip'].name),
            'sheet_file': str(self.output_dir / "build" / self.structure_paths['sheet'].name),
            
            # Legacy paths (keep for compatibility)
            'path_sub': str(self.output_dir / "build" / self.structure_paths['sub'].name),
            'path_tip': str(self.output_dir / "build" / self.structure_paths['tip'].name),
            'path_sheet': str(self.output_dir / "build" / self.structure_paths['sheet'].name),
            
            # Tip position for read_data shift
            'tip_x': tip_x,
            'tip_y': tip_y,
            'tip_z': tip_z,
            
            # Sheet z position
            'sheet_z': self.z_positions['sheet'],
            
            # Type offsets
            'offset_2d': offset_2d,
            
            # Results output patterns (relative to run directory)
            'results_file_pattern': str(self.output_dir / 'results' / 'friction_f${find}_a${a}.dat'),
            'dump_file_pattern': str(self.output_dir / 'visuals' / 'slide_f${find}_a${a}.*.dump'),
            'dump_enabled': out.dump.get('slide', False),

            # Z positions
            'z_sub': self.z_positions['sub'],
            'z_sheet': self.z_positions['sheet'],
            'z_tip': self.z_positions['tip'],

            # Group type IDs
            'sub_types': self.groups['sub_types'],
            'tip_types': self.groups['tip_types'],
            'sheet_types': self.groups['sheet_types'],
            'ngroups': total_types,
            'extra_atom_types': 1,  # For virtual atom if using that drive method

            'sub_natypes': len(self.groups['sub_types'].split()),
            'tip_natypes': len(self.groups['tip_types'].split()),

            # Simulation settings
            'timestep': sim.timestep,
            'thermo': sim.thermo,
            'neighbor_list': sim.neighbor_list,
            'neigh_modify_command': sim.neigh_modify_command,
            'run_steps': sim.slide_run_steps,
            'drive_method': sim.drive_method,
            
            # Drive mechanism parameters
            'damp_ev': self.config.tip.dspring / 0.016,
            'spring_ev': (self.config.general.driving_spring or 8.0) / 16.02,  # N/m to eV/Å²
            'tipps': self.config.tip.s / 100,
            'virtual_offset': 10.0, 
            'virtual_atom_type': total_types + 1,
            
            # Output frequencies
            'results_freq': out.results_frequency,
            'dump_freq': out.dump_frequency.get('slide', 1000),
            
            # Group names for LAMMPS fixes
            'tip_fix_group': 'tip',
            'layer_group': 'sheet',
            
            # Multi-layer sheet context
            'n_sheet_layers': max(self.config.sheet.layers) if self.config.sheet.layers else 1,
            'lat_c': self.lat_c,
            'tip_radius': self.config.tip.r,
            'sheet_dims': self.sheet_dims,
            
            # Thermostat settings
            'thermostat_type': self.config.settings.thermostat.type,
            'use_langevin': self.config.settings.thermostat.type == 'langevin',
            
            # Minimization settings
            'min_style': sim.min_style,
            'minimization_command': sim.minimization_command,
            
            # Output paths
            'output_dir': '../results',
            'dump_file': '../visuals/system.*.dump',
        }

        init_script = self.render_template("afm/system_init.lmp", context)
        self.write_file("lammps/system.in", init_script)

        slide_script = self.render_template("afm/slide.lmp", context)
        self.write_file("lammps/slide.in", slide_script)

        logger.info(f"Inputs written to {self.output_dir}/lammps/")