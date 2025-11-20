"""Sheet-on-Sheet Simulation Builder.

This module orchestrates the setup of a friction simulation between two
2D material sheets. It coordinates the construction of the top and bottom
sheets, manages their "ghost" interactions, and generates LAMMPS scripts.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from FrictionSim2D.core.base_builder import BaseBuilder
from FrictionSim2D.core.config import AFMSimulationConfig
from FrictionSim2D.core.potential_manager import PotentialManager
from FrictionSim2D.builders import components

logger = logging.getLogger(__name__)

class SheetVsSheetSimulation(BaseBuilder):
    """Builder for Sheet-on-Sheet friction simulations."""

    def __init__(self, config: AFMSimulationConfig, output_dir: str):
        super().__init__(config, output_dir)
        self.config: AFMSimulationConfig = config
        
        # State
        self.structure_paths: Dict[str, Path] = {}
        self.z_positions: Dict[str, float] = {}
        self.groups: Dict[str, str] = {}

    def build(self) -> None:
        """Constructs the dual-sheet system."""
        logger.info("Starting Sheet-vs-Sheet Build...")
        self._create_directories()
        build_dir = self.output_dir / "build"

        # 1. Build Components
        # Bottom Sheet (Acts as substrate)
        # Note: Config currently has one 'sheet' entry. 
        # Ideally, we'd have 'sheet_top' and 'sheet_bottom' configs.
        # Assuming for now we use the SAME material for both, just stacked differently.
        
        logger.info("Building Bottom Sheet...")
        bot_path, dims, lat_c = components.build_sheet(
            self.config.sheet, self.atomsk, build_dir, stack_if_multi=True
        )
        self.structure_paths['bottom'] = bot_path
        
        logger.info("Building Top Sheet...")
        top_path, _, _ = components.build_sheet(
            self.config.sheet, self.atomsk, build_dir, stack_if_multi=True
        )
        self.structure_paths['top'] = top_path

        # 2. Calculate Layout
        # Bottom sheet starts at 0
        self.z_positions['bottom'] = 0.0
        
        # Top sheet starts above bottom sheet
        # Gap calculation
        # Since they are likely same material, use self-interaction sigma or standard calculation
        gap = self._calculate_gap(self.config.sheet, self.config.sheet, buffer=0.5)
        
        n_layers = max(self.config.sheet.layers) if self.config.sheet.layers else 1
        bot_height = (n_layers - 1) * lat_c
        
        self.z_positions['top'] = bot_height + gap
        
        # 3. Generate Potentials
        self._generate_potentials()
        
        logger.info("Build complete.")

    def _calculate_gap(self, comp1, comp2, buffer=0.5):
        """Calculates gap based on max sigma."""
        # Helper to calculate gap (could also reuse from PM or base if moved there)
        # For identical sheets, this is just the material's VdW gap proxy
        pm = PotentialManager() # Temp instance just for calc
        # We can just use utils directly if configs are available
        # But since we have the logic in afm.py, we could move this to BaseBuilder to avoid duplication.
        # For now, replicating the logic from afm.py for independence.
        from FrictionSim2D.core.utils import cifread, lj_params
        
        elems1 = cifread(comp1.cif_path)['elements']
        elems2 = cifread(comp2.cif_path)['elements']
        max_sigma = 0.0
        for e1 in set(elems1):
            for e2 in set(elems2):
                _, sigma = lj_params(e1, e2)
                max_sigma = max(max_sigma, sigma)
        return max_sigma + buffer

    def _generate_potentials(self) -> None:
        """Configures potential file with Ghost interactions."""
        pm = PotentialManager()
        
        # Register
        pm.register_component('bottom', self.config.sheet)
        pm.register_component('top', self.config.sheet)
        
        # Interactions
        pm.add_self_interaction('bottom')
        pm.add_self_interaction('top')
        
        # Cross Interaction: Ghost / Weak LJ
        # We want them to interact, but perhaps with specific settings defined in config
        # or standard LJ. Original code used 'hybrid' and specific coeffs.
        # Defaulting to standard LJ mixing for physical friction.
        # If "ghost" behavior (no interaction) is desired, epsilon can be set to ~0.
        pm.add_cross_interaction('bottom', 'top', interaction_type='lj/cut')
        
        pm.write_file(self.output_dir / "lammps" / "system.in.settings")
        
        self.groups['bottom_types'] = pm.get_group_string('bottom')
        self.groups['top_types'] = pm.get_group_string('top')

    def write_inputs(self) -> None:
        """Generates LAMMPS scripts."""
        logger.info("Writing LAMMPS inputs...")
        
        all_types = set()
        for s in self.groups.values():
            all_types.update(s.split())
        total_types = len(all_types)

        context = {
            'temp': self.config.general.temp,
            'pressure': self.config.general.pressure,
            'angle': self.config.general.scan_angle,
            'speed': self.config.general.scan_speed, # Sheet-on-sheet often uses scan_speed
            'settings': self.config.settings.simulation,
            
            'path_bot': f"../build/{self.structure_paths['bottom'].name}",
            'path_top': f"../build/{self.structure_paths['top'].name}",
            
            'z_bot': self.z_positions['bottom'],
            'z_top': self.z_positions['top'],
            
            'bot_types': self.groups['bottom_types'],
            'top_types': self.groups['top_types'],
            'ngroups': total_types,
            
            'results_freq': self.config.settings.output.results_frequency,
            'dump_freq': self.config.settings.output.dump_frequency['slide']
        }
        
        # Render Template
        # Note: Sheet-on-sheet typically skips 'system_init' and goes straight to slide/equilibration
        # or combines them. Assuming 'slide.lmp' covers the physics.
        script = self.render_template("sheet_vs_sheet/slide.lmp", context)
        self.write_file("lammps/slide.in", script)
        
        logger.info(f"Inputs written to {self.output_dir}/lammps/")