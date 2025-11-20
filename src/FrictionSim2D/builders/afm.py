"""AFM Simulation Builder.

This module orchestrates the setup of a complete Atomic Force Microscopy (AFM)
simulation. It coordinates the construction of the Tip, Substrate, and Sheet,
generates the necessary potentials, and writes the LAMMPS input scripts.
"""

import logging
from pathlib import Path
from typing import Dict

from FrictionSim2D.core.base_builder import BaseBuilder
from FrictionSim2D.core.config import AFMSimulationConfig
from FrictionSim2D.core.potential_manager import PotentialManager
from FrictionSim2D.builders import components

logger = logging.getLogger(__name__)

class AFMSimulation(BaseBuilder):
    """Builder for AFM simulations (Tip + Sheet + Substrate)."""

    def __init__(self, config: AFMSimulationConfig, output_dir: str):
        super().__init__(config, output_dir)
        self.config: AFMSimulationConfig = config  # Type hinting alias

        # State to track build artifacts
        self.structure_paths: Dict[str, Path] = {}
        self.z_positions: Dict[str, float] = {}
        self.groups: Dict[str, str] = {} # Component name -> Atom type IDs string

    def build(self) -> None:
        """Constructs the atomic systems and layout."""
        logger.info("Starting AFM Simulation Build...")
        self._create_directories()
        build_dir = self.output_dir / "build"

        # 1. Build Physical Components
        # Tip
        tip_path, tip_radius = components.build_tip(
            self.config.tip, self.atomsk, build_dir, self.config.settings
        )
        self.structure_paths['tip'] = tip_path

        # Sheet
        sheet_path, sheet_dims, lat_c = components.build_sheet(
            self.config.sheet, self.atomsk, build_dir, stack_if_multi=True
        )
        self.structure_paths['sheet'] = sheet_path

        # Substrate
        sub_path = components.build_substrate(
            self.config.sub, self.atomsk, build_dir, sheet_dims
        )
        self.structure_paths['sub'] = sub_path

        # 2. Generate Potentials & Calculate Gaps
        # We need to register components first to calculate gaps using PM
        pm = self._generate_potentials()

        # Calculate Vertical Layout (Z-Offsets)
        # Use the PM to calculate gaps based on max sigma + buffer
        gap_sub_sheet = pm.calculate_gap('sub', 'sheet', buffer=0.5)
        gap_sheet_tip = pm.calculate_gap('sheet', 'tip', buffer=0.5)

        logger.info(f"Calculated gaps: Sub-Sheet={gap_sub_sheet:.2f}A, Sheet-Tip={gap_sheet_tip:.2f}A")

        sub_thickness = self.config.sub.thickness
        
        # Position 1: Substrate (Base)
        self.z_positions['sub'] = 0.0

        # Position 2: Sheet (Above Substrate)
        sheet_base_z = sub_thickness + gap_sub_sheet
        self.z_positions['sheet'] = sheet_base_z

        # Position 3: Tip (Above Sheet)
        n_layers = max(self.config.sheet.layers) if self.config.sheet.layers else 1
        sheet_stack_height = (n_layers - 1) * lat_c

        # Tip Z is usually center of sphere, so add radius
        tip_z = sheet_base_z + sheet_stack_height + gap_sheet_tip + tip_radius
        self.z_positions['tip'] = tip_z

        logger.info("Build complete.")

    def _generate_potentials(self) -> PotentialManager:
        """Configures and writes the potential file using PotentialManager."""
        pm = PotentialManager()

        # Register components
        pm.register_component('sub', self.config.sub)
        pm.register_component('tip', self.config.tip)
        pm.register_component('sheet', self.config.sheet)

        # Define Interactions
        pm.add_self_interaction('sub')
        pm.add_self_interaction('tip')
        pm.add_self_interaction('sheet')

        # Cross Interactions (LJ Mixing)
        pm.add_cross_interaction('sub', 'tip', interaction_type='lj/cut')
        pm.add_cross_interaction('sub', 'sheet', interaction_type='lj/cut')
        pm.add_cross_interaction('tip', 'sheet', interaction_type='lj/cut')

        pm.write_file(self.output_dir / "lammps" / "system.in.settings")

        # Store group ID strings
        self.groups['sub_types'] = pm.get_group_string('sub')
        self.groups['tip_types'] = pm.get_group_string('tip')
        self.groups['sheet_types'] = pm.get_group_string('sheet')

        return pm

    def write_inputs(self) -> None:
        """Generates the LAMMPS input scripts."""
        logger.info("Writing LAMMPS inputs...")

        all_types = set()
        for s in self.groups.values():
            all_types.update(s.split())
        total_types = len(all_types)

        context = {
            'temp': self.config.general.temp,
            'force': self.config.general.force,
            'angle': self.config.general.scan_angle,
            'speed': self.config.tip.s,
            'settings': self.config.settings.simulation,

            'path_sub': f"../build/{self.structure_paths['sub'].name}",
            'path_tip': f"../build/{self.structure_paths['tip'].name}",
            'path_sheet': f"../build/{self.structure_paths['sheet'].name}",

            'z_sub': self.z_positions['sub'],
            'z_sheet': self.z_positions['sheet'],
            'z_tip': self.z_positions['tip'],

            'sub_types': self.groups['sub_types'],
            'tip_types': self.groups['tip_types'],
            'sheet_types': self.groups['sheet_types'],
            'ngroups': total_types,

            'sub_natypes': len(self.groups['sub_types'].split()),
            'tip_natypes': len(self.groups['tip_types'].split()),

            'damp_ev': self.config.tip.dspring / 0.016,
            'spring_ev': self.config.tip.cspring / 16.02,
            'tipps': self.config.tip.s / 100,
            'drive_method': self.config.settings.simulation.drive_method,
            'virtual_offset': 10.0, 
            'virtual_atom_type': total_types + 1,
            'run_steps': self.config.settings.simulation.slide_run_steps,
            'results_freq': self.config.settings.output.results_frequency,
            'dump_freq': self.config.settings.output.dump_frequency['slide']
        }

        init_script = self.render_template("afm/system_init.lmp", context)
        self.write_file("lammps/system.in", init_script)

        slide_script = self.render_template("afm/slide.lmp", context)
        self.write_file("lammps/slide.in", slide_script)

        logger.info(f"Inputs written to {self.output_dir}/lammps/")