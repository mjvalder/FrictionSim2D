"""Sheet-on-Sheet Simulation Builder.

This module orchestrates the setup of a friction simulation between two
2D material sheets. The standard model is a 4-layer stack:
  - Layer 1: Fixed (bottom)
  - Layer 2-3: Mobile with Langevin thermostat (friction interface)
  - Layer 4: Rigid, driven by virtual atom (top)

Interlayer interactions:
  - Adjacent layers (1-2, 2-3, 3-4): Real LJ interactions
  - Non-adjacent layers (1-3, 1-4, 2-4): Ghost interactions (prevent interpenetration)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from FrictionSim2D.core.simulation_base import SimulationBase
from FrictionSim2D.core.config import SheetOnSheetSimulationConfig
from FrictionSim2D.core.potential_manager import PotentialManager
from FrictionSim2D.builders import components

logger = logging.getLogger(__name__)

# Standard 4-layer model configuration
N_LAYERS = 4


class SheetOnSheetSimulation(SimulationBase):
    """Builder for Sheet-on-Sheet friction simulations.
    
    Creates a 4-layer stack of the same 2D material:
      - Layer 1: Fixed bottom layer
      - Layer 2: Mobile (Langevin thermostat)
      - Layer 3: Mobile (Langevin thermostat)  
      - Layer 4: Driven top layer (rigid body)
    """

    def __init__(self, config: SheetOnSheetSimulationConfig, output_dir: str):
        super().__init__(config, output_dir)
        self.config: SheetOnSheetSimulationConfig = config
        
        # State
        self.structure_paths: Dict[str, Path] = {}
        self.z_positions: Dict[str, float] = {}
        self.groups: Dict[str, str] = {}
        self.pm: Optional[PotentialManager] = None

    def build(self) -> None:
        """Constructs the 4-layer sheet stack."""
        logger.info("Starting Sheet-vs-Sheet Build (4-layer model)...")
        self._create_directories()
        build_dir = self.output_dir / "build"
        
        # Build the 4-layer sheet stack
        logger.info(f"Building {N_LAYERS}-layer sheet stack...")
        sheet_path, dims, lat_c = components.build_sheet(
            self.config.sheet, self.atomsk, build_dir,
            stack_if_multi=True, settings=self.config.settings,
            n_layers_override=N_LAYERS
        )
        self.structure_paths['sheet'] = sheet_path

        # Generate Potentials
        self.pm = self._generate_potentials()

        # Calculate Z positions for each layer
        self.z_positions['layer_1'] = 0.0
        self.z_positions['layer_2'] = lat_c
        self.z_positions['layer_3'] = 2 * lat_c
        self.z_positions['layer_4'] = 3 * lat_c
        
        # Store calculated values
        self.lat_c = lat_c
        self.sheet_dims = dims
        
        # Write LAMMPS inputs
        self.write_inputs()
        
        logger.info("Build complete.")

    def _generate_potentials(self) -> PotentialManager:
        """Configures potential file for 4-layer sheet-on-sheet simulation.
        
        Returns:
            Configured PotentialManager instance.
        """
        pm = PotentialManager(self.config.settings)
        
        # Register single sheet component with 4 layers
        # Each layer gets its own potential instance (for SW, Tersoff, etc.)
        pm.register_component('sheet', self.config.sheet, n_layers=N_LAYERS)
        
        # Self-Interactions: each layer gets its own many-body potential
        pm.add_self_interaction('sheet')
        
        # Interlayer Interactions:
        # - Adjacent layers (distance=1): Real LJ
        # - Non-adjacent layers (distance>1): Ghost LJ (prevent interpenetration)
        pm.add_interlayer_lj_by_distance('sheet', max_real_distance=1)
        
        pm.write_file(self.output_dir / "lammps" / "system.in.settings")
        
        # Store layer group strings
        for layer in range(N_LAYERS):
            layer_num = layer + 1
            self.groups[f'layer_{layer_num}'] = pm.get_layer_group_string('sheet', layer)
        
        # Convenience groups
        self.groups['center'] = f"{self.groups['layer_2']} {self.groups['layer_3']}"
        self.groups['all_types'] = pm.get_group_string('sheet')
        
        return pm

    def write_inputs(self) -> None:
        """Generates LAMMPS scripts."""
        logger.info("Writing LAMMPS inputs...")
        
        total_types = self.pm.get_total_types() if self.pm else 0
        virtual_atom_type = total_types + 1
        
        sim = self.config.settings.simulation
        out = self.config.settings.output

        context = {
            # General settings
            'temp': self.config.general.temp,
            'pressures': self.config.general.pressure,  # For LAMMPS index variable
            'angles': self.config.general.scan_angle,   # For LAMMPS index variable
            'scan_speed_norm': self.config.general.scan_speed / 100,  # Normalized for LAMMPS
            
            # Box dimensions (actual from build)
            'xlo': self.sheet_dims.get('xlo', 0.0),
            'xhi': self.sheet_dims.get('xhi', 100.0),
            'ylo': self.sheet_dims.get('ylo', 0.0),
            'yhi': self.sheet_dims.get('yhi', 100.0),
            
            # File paths
            'data_file': f"../build/{self.structure_paths['sheet'].name}",
            'potential_file': 'system.in.settings',
            
            # Atom types
            'num_atom_types': total_types,
            'ngroups': total_types,
            
            # Layer groups
            'layer_1_types': self.groups['layer_1'],
            'layer_2_types': self.groups['layer_2'],
            'layer_3_types': self.groups['layer_3'],
            'layer_4_types': self.groups['layer_4'],
            'center_types': self.groups['center'],
            
            # Geometry
            'n_layers': N_LAYERS,
            'lat_c': self.lat_c,
            'sheet_dims': self.sheet_dims,
            
            # Bond settings for interlayer springs
            'bond_spring_ev': (self.config.general.bond_spring or 5.0) / 16.02,
            'bond_min': self.lat_c * 0.8,  # 80% of interlayer spacing
            'bond_max': self.lat_c * 1.2,  # 120% of interlayer spacing
            
            # Driving spring (convert N/m to eV/Å²: divide by 16.02)
            'driving_spring_ev': (self.config.general.driving_spring or 50.0) / 16.02,
            
            # Simulation settings
            'timestep': sim.timestep,
            'thermo': sim.thermo,
            'neighbor_list': sim.neighbor_list,
            'neigh_modify_command': sim.neigh_modify_command,
            'run_steps': sim.slide_run_steps,
            
            # Minimization settings
            'min_style': sim.min_style,
            'minimization_command': sim.minimization_command,
            
            # Output settings
            'results_freq': out.results_frequency,
            'dump_freq': out.dump_frequency.get('slide', 1000),
            'dump_enabled': out.dump.get('slide', False),
            'results_file_pattern': '../results/friction_p${pressure}_a${a}.dat',
            'dump_file_pattern': '../visuals/slide_p${pressure}_a${a}.*.dump',
            
            # Virtual atom for driving
            'virtual_atom_type': virtual_atom_type,
            'drive_method': sim.drive_method,
            
            # Thermostat
            'thermostat_type': self.config.settings.thermostat.type,
        }
        
        # Render Templates
        script = self.render_template("sheetonsheet/slide.lmp", context)
        self.write_file("lammps/slide.in", script)
        
        logger.info(f"Inputs written to {self.output_dir}/lammps/")