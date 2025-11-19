"""Builder for Sheet-vs-Sheet Simulations."""

import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any

from ase import data as ase_data

from FrictionSim2D.core.base_builder import BaseBuilder
from FrictionSim2D.builders import components
from FrictionSim2D.core.utils import count_atomtypes, lj_params, cifread
logger = logging.getLogger(__name__)

class SheetVsSheetSimulation(BaseBuilder):
    """Builder for Sheet-vs-Sheet friction simulations."""

    def __init__(self, config, output_dir=None):
        super().__init__(config, output_dir)
        self.meta_data = {}
        self.pot_data = {}
        self.atom_groups = {}
        self.group_definitions = {}

    def build(self) -> None:
        self.setup_directories(["visuals", "results", "build", "potentials", "lammps"])
        self._initialize_component_metadata('sheet', self.config.sheet)
        
        original_layers = self.config.sheet.layers
        if not original_layers or max(original_layers) < 4:
            self.config.sheet.layers = [1, 2, 3, 4]
            
        # Build base layer
        sheet_path, sheet_dims, lat_c = components.build_sheet(
            self.config.sheet, self.atomsk, self.work_dir / "build"
        )
        
        # Stack for friction simulation (4 layers)
        final_stack_path = self._create_friction_stack(sheet_path, sheet_dims, lat_c, layers=4)

        self._write_potential_file("system.in.settings", layers=4)
        self.write_inputs(sheet_dims, lat_c, final_stack_path)
        
        self.config.sheet.layers = original_layers

    def _create_friction_stack(self, base_layer_path: Path, dims: Dict, lat_c: float, layers: int) -> Path:
        """Creates the specific 4-layer stack with unique types per layer."""
        output_path = self.work_dir / "build" / f"{self.config.sheet.mat}_4.lmp"
        
        # Use modular shift calculation
        sx, sy = components.calculate_layer_shifts(self.config.sheet.mat, dims)
        
        natypes = self.pot_data['sheet']['natype']
        total_types = natypes * layers # We want unique IDs for every layer's atoms
        
        cmds = [
            "clear",
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            f"region box block {dims['xlo']} {dims['xhi']} {dims['ylo']} {dims['yhi']} -5 {dims['yhi']+6*layers}",
            f"create_box {total_types} box",
        ]
        
        # Build stack commands with specific friction-simulation shifts
        # Layers 1,2 (Bottom) vs 3,4 (Top)
        # Top layers usually shifted to align/misalign
        
        # Layer 1
        cmds.append(f"read_data {base_layer_path} add append group layer_1")
        
        # Layer 2
        cmds.append(f"read_data {base_layer_path} add append shift 0 0 {lat_c} group layer_2")
        cmds.append(f"displace_atoms layer_2 move {sx} {sy} 0 units box")
        
        # Layer 3 (Start of Top Block)
        cmds.append(f"read_data {base_layer_path} add append shift 0 0 {lat_c*2} group layer_3")
        cmds.append(f"displace_atoms layer_3 move {sx} {sy} 0 units box")
        
        # Layer 4
        cmds.append(f"read_data {base_layer_path} add append shift 0 0 {lat_c*3} group layer_4")
        cmds.append(f"displace_atoms layer_4 move {sx} {sy} 0 units box")

        # Renumber types: Map original types to Layer-Specific types
        # E.g. Layer 2 Carbon (Type 1 in file) -> Global Type (natypes + 1)
        current_new_type = 1
        for l in range(1, layers + 1):
            for t in range(1, natypes + 1):
                cmds.append(f"group t_{t} type {t}")
                # Intersect atoms that are in this Layer AND are this original Element
                cmds.append(f"group target intersect layer_{l} t_{t}")
                cmds.append(f"set group target type {current_new_type}")
                cmds.append(f"group t_{t} delete")
                cmds.append(f"group target delete")
                current_new_type += 1

        cmds.append(f"write_data {output_path}")
        
        components.run_lammps_commands(cmds)
        return output_path

    def _initialize_component_metadata(self, name: str, component_config: Any) -> None:
        """Reads CIF/Potential data."""
        cif_path = Path(component_config.cif_path)
        if not cif_path.exists():
            cif_path = components._get_material_path(component_config.cif_path, 'cif')
            
        data = cifread(cif_path)
        pot_path = Path(component_config.pot_path)
        counts = count_atomtypes(pot_path, data['elements'])
        
        dest_pot = self.work_dir / "potentials" / pot_path.name
        if pot_path.exists():
            shutil.copy(pot_path, dest_pot)
            
        self.meta_data[name] = data
        self.pot_data[name] = {
            'path': dest_pot.name,
            'counts': counts,
            'natype': sum(counts.values())
        }

    def _write_potential_file(self, filename: str, layers: int = 4) -> None:
        # ... (Same logic as previous step, omitted for brevity unless requested) ...
        # This function generates the pair_coeffs based on the unique types we just created
        pass

    def write_inputs(self, dims: Dict, lat_c: float, data_file: Path) -> None:
        # ... (Same logic as previous step) ...
        pass