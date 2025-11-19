"""Builder for AFM Simulations.

This module orchestrates the setup of an AFM friction simulation.
It builds the necessary components (tip, substrate, sheet), manages
potential files, and generates the LAMMPS input scripts.
"""
from pathlib import Path
from typing import List, Dict, Any
import logging
import shutil
# import numpy as np
from ase import data as ase_data

from FrictionSim2D.core.base_builder import BaseBuilder
from FrictionSim2D.builders import components
from FrictionSim2D.core.utils import count_atomtypes, lj_params, cifread

logger = logging.getLogger(__name__)

class AFMSimulation(BaseBuilder):
    """Builder for Atomic Force Microscopy (AFM) simulations."""

    def __init__(self, config, output_dir=None):
        super().__init__(config, output_dir)
        # Metadata storage to replace the old self.data/self.potentials dictionaries
        self.meta_data = {} 
        self.pot_data = {}
        self.atom_groups = {} # Tracks atom type IDs
        self.group_definitions = {} # Tracks LAMMPS group names

    def build(self) -> None:
        """Main execution method to build the simulation environment."""
        
        # 1. Setup Directory Structure
        self.setup_directories(["visuals", "results", "build", "potentials"])

        # 2. Gather Metadata & Copy Potentials
        # We need to know elements and atom counts before building to handle IDs correctly
        self._initialize_component_metadata('sheet', self.config.sheet)
        self._initialize_component_metadata('sub', self.config.sub)
        self._initialize_component_metadata('tip', self.config.tip)

        # 3. Build Components
        logger.info("Building Sheet...")
        sheet_path, sheet_dims, lat_c = components.build_sheet(
            self.config.sheet, self.atomsk, self.work_dir / "build"
        )
        
        logger.info("Building Substrate...")
        # Substrate needs to match sheet dimensions
        sub_path = components.build_substrate(
            self.config.sub, self.atomsk, self.work_dir / "build", sheet_dims
        )
        
        logger.info("Building Tip...")
        tip_path, tip_radius = components.build_tip(
            self.config.tip, self.atomsk, self.work_dir / "build", self.settings
        )

        # 4. Generate Potential Settings Files
        # These files define pair_coeffs and groups for LAMMPS
        self._write_potential_file("system.in.settings", sheet_layers=self.config.sheet.layers, is_slide=False)
        self._write_potential_file("slide.in.settings", sheet_layers=self.config.sheet.layers, is_slide=True)

        # 5. Write Main LAMMPS Scripts
        self.write_inputs(sheet_dims, tip_radius, lat_c)

    def _initialize_component_metadata(self, name: str, component_config: Any) -> None:
        """Reads CIF/Potential data and prepares metadata for a component."""
        
        # Resolve CIF path
        cif_path = Path(component_config.cif_path)
        if not cif_path.exists():
            cif_path = components._get_material_path(component_config.cif_path, 'cif')
            
        # Read Material Data
        data = cifread(cif_path)
        
        # Resolve Potential Path using helper
        pot_path = Path(component_config.pot_path)
        if not pot_path.exists():
             # Explicitly resolve using the helper
             pot_path = components._get_potential_path(str(component_config.pot_path))
        
        # Check again if found
        if not pot_path.exists():
             raise FileNotFoundError(f"Potential file '{component_config.pot_path}' not found in package data.")
             
        # Count atom types based on potential file
        # This determines if 'C' becomes 'C1' and 'C2' in LAMMPS
        counts = count_atomtypes(pot_path, data['elements'])
        
        # Copy potential file to simulation folder
        dest_pot = self.work_dir / "potentials" / pot_path.name
        if pot_path.exists():
            shutil.copy(pot_path, dest_pot)
        
        # Store
        self.meta_data[name] = data
        self.pot_data[name] = {
            'path': dest_pot.name, # Relative path for LAMMPS
            'full_path': dest_pot,
            'counts': counts,
            'natype': sum(counts.values()),
            'config': component_config
        }

    def _write_potential_file(self, filename: str, sheet_layers: List[int], is_slide: bool = False) -> None:
        """Generates the LAMMPS potential settings file."""
        
        # Determine active layer count
        # For system_init, we usually build all layers requested.
        # If layers list is [1, 2], max is 2.
        num_layers = max(sheet_layers) if sheet_layers else 1
        
        filepath = self.work_dir / "potentials" / filename
        
        with open(filepath, 'w') as f:
            self.atom_groups = {} # Reset for this file
            self.group_definitions = {}
            atype = 1

            # 1. Define Elements & Groups
            # Order: Sheet -> Substrate -> Tip (Standard convention in this code)
            # Sheet
            atype = self._define_elemgroup('sheet', atype, layers=num_layers)
            # Substrate
            atype = self._define_elemgroup_3regions('sub', atype)
            # Tip
            atype = self._define_elemgroup_3regions('tip', atype)

            # 2. Set Masses
            self._set_masses('sheet', f, layers=num_layers)
            self._set_masses('sub', f)
            self._set_masses('tip', f)

            # 3. Define LAMMPS Groups
            self._write_lammps_groups(f, num_layers)

            # 4. Configure Pair Styles
            self._write_pair_interactions(f, num_layers, is_slide, atype)

    def _define_elemgroup(self, system: str, start_type: int, layers: int = 1) -> int:
        """Assigns LAMMPS atom type IDs for standard components."""
        counts = self.pot_data[system]['counts']
        elements = self.meta_data[system]['elements'] # Ordered list from CIF
        
        current_type = start_type
        
        # Create a map: system -> layer -> element -> [type_ids]
        if system not in self.atom_groups: self.atom_groups[system] = {}

        # Iterate elements as they appear in CIF (Atomsk order)
        for el in elements:
            count = counts.get(el, 1)
            if system == 'sheet':
                for l in range(layers):
                    if l not in self.atom_groups[system]: self.atom_groups[system][l] = {}
                    if el not in self.atom_groups[system][l]: self.atom_groups[system][l][el] = []
                    
                    for _ in range(count):
                        # Store metadata for this atom type
                        # format: [Group Name, AtomTypeID, ElementName, AtomLabel]
                        self.group_definitions[current_type] = [f"{system}_l{l+1}", str(current_type), el]
                        self.atom_groups[system][l][el].append(current_type)
                        current_type += 1
            else:
                if el not in self.atom_groups[system]: self.atom_groups[system][el] = []
                for _ in range(count):
                    self.group_definitions[current_type] = [f"{system}", str(current_type), el]
                    self.atom_groups[system][el].append(current_type)
                    current_type += 1
                    
        return current_type

    def _define_elemgroup_3regions(self, system: str, start_type: int) -> int:
        """Assigns atom types for components with Fixed/Thermo/Mobile regions."""
        # Used for Tip and Substrate when Langevin is active
        # Each original atom type splits into 3: Mobile, Fixed, Thermo
        
        counts = self.pot_data[system]['counts']
        elements = self.meta_data[system]['elements']
        current_type = start_type
        
        if system not in self.atom_groups: self.atom_groups[system] = {}

        for el in elements:
            count = counts.get(el, 1)
            self.atom_groups[system][el] = []
            
            for i in range(count):
                # 1. Standard/Mobile
                self.group_definitions[current_type] = [f"{system}", str(current_type), el]
                
                # 2. Fixed
                self.group_definitions[current_type+1] = [f"{system}_fix", str(current_type+1), el]
                
                # 3. Thermo
                self.group_definitions[current_type+2] = [f"{system}_thermo", str(current_type+2), el]
                
                self.atom_groups[system][el].extend([current_type, current_type+1, current_type+2])
                current_type += 3
                
        return current_type

    def _set_masses(self, system: str, f, layers: int = 1):
        """Writes mass commands to file."""
        elements = self.meta_data[system]['elements']
        
        for el in set(elements): # Unique elements
            mass = ase_data.atomic_masses[ase_data.atomic_numbers[el]]
            
            if system == 'sheet':
                # Get first type of first layer and last type of last layer for this element
                # Assumes sequential ordering
                first_type = self.atom_groups[system][0][el][0]
                last_type = self.atom_groups[system][layers-1][el][-1]
                f.write(f"mass {first_type}*{last_type} {mass} # {el} ({system})\n")
            else:
                types = self.atom_groups[system][el]
                f.write(f"mass {types[0]}*{types[-1]} {mass} # {el} ({system})\n")

    def _write_lammps_groups(self, f, layers):
        """Defines LAMMPS groups based on type ranges."""
        # Sheet Layers
        for l in range(layers):
            # Collect all types belonging to this layer
            types = []
            for el in self.atom_groups['sheet'][l]:
                types.extend([str(t) for t in self.atom_groups['sheet'][l][el]])
            f.write(f"group layer_{l+1} type {' '.join(types)}\n")

        # Tip/Sub Regions
        for system in ['sub', 'tip']:
            # Gather types for _fix, _thermo, and _all
            fix_types = []
            thermo_types = []
            all_types = []
            
            for t_id, defs in self.group_definitions.items():
                if f"{system}_fix" in defs[0]:
                    fix_types.append(str(t_id))
                if f"{system}_thermo" in defs[0]:
                    thermo_types.append(str(t_id))
                if defs[0].startswith(system): # Matches system, system_fix, system_thermo
                    all_types.append(str(t_id))
            
            f.write(f"group {system}_all type {' '.join(all_types)}\n")
            if fix_types:
                f.write(f"group {system}_fix type {' '.join(fix_types)}\n")
            if thermo_types:
                f.write(f"group {system}_thermo type {' '.join(thermo_types)}\n")

        f.write("group mobile union tip_thermo sub_thermo\n")

    def _write_pair_interactions(self, f, layers, is_slide, max_type):
        """Writes pair_style and pair_coeff commands."""
        
        # 1. Determine Styles
        styles = []
        
        # Helper to check if potential needs explicit LJ
        def needs_lj(pot_type):
            return pot_type in ['sw', 'tersoff', 'rebo', 'edip', 'meam', 'eam', 'bop', 'morse', 'rebomos', 'sw/mod']

        sheet_lj = needs_lj(self.config.sheet.pot_type)
        
        # Add component potentials
        if sheet_lj:
            # If sheet needs LJ, we usually add the potential ONCE per layer or once total depending on type
            # Legacy code added it 'layer' times.
            for _ in range(layers):
                styles.append(self.config.sheet.pot_type)
        else:
            styles.append(self.config.sheet.pot_type)
            
        styles.append(self.config.sub.pot_type)
        styles.append(self.config.tip.pot_type)
        styles.append("lj/cut 11.0") # For interactions
        
        f.write(f"pair_style hybrid {' '.join(styles)}\n")
        
        # 2. Write Component Coefficients
        # Sheet
        sheet_pot_path = self.pot_data['sheet']['path']
        if sheet_lj:
            # Hybrid overlay approach for layered materials
            for l in range(layers):
                # Map types for this layer to the potential file
                # This requires constructing the mapping string "C NULL NULL..."
                # Legacy code: built a list of NULLs and filled in types for specific layer
                line_args = []
                for t in range(1, max_type + 1): # LAMMPS types are 1-indexed
                    if t in self.group_definitions and f"sheet_l{l+1}" in self.group_definitions[t][0]:
                        # Map global type 't' to potential type element
                        el = self.group_definitions[t][2]
                        line_args.append(el) 
                    else:
                        line_args.append("NULL")
                
                f.write(f"pair_coeff * * {self.config.sheet.pot_type} {sheet_pot_path} {' '.join(line_args)}\n")
        else:
             # Non-LJ (AIREBO etc) applies to all sheet atoms at once
             line_args = []
             for t in range(1, max_type + 1):
                if t in self.group_definitions and "sheet" in self.group_definitions[t][0]:
                    el = self.group_definitions[t][2]
                    line_args.append(el)
                else:
                    line_args.append("NULL")
             f.write(f"pair_coeff * * {self.config.sheet.pot_type} {sheet_pot_path} {' '.join(line_args)}\n")

        # Substrate & Tip
        for sys_name in ['sub', 'tip']:
            pot = self.pot_data[sys_name]['config'].pot_type
            path = self.pot_data[sys_name]['path']
            line_args = []
            for t in range(1, max_type + 1):
                if t in self.group_definitions and sys_name in self.group_definitions[t][0]:
                    el = self.group_definitions[t][2]
                    line_args.append(el)
                else:
                    line_args.append("NULL")
            f.write(f"pair_coeff * * {pot} {path} {' '.join(line_args)}\n")

        # 3. Interactions (Lennard-Jones)
        # Inter-layer (if sheet > 1 layer)
        if layers > 1:
            # Simple LJ between layers
            for l1 in range(layers):
                for l2 in range(l1 + 1, layers):
                    # Iterate elements
                    for el1 in self.atom_groups['sheet'][l1]:
                        for el2 in self.atom_groups['sheet'][l2]:
                            eps, sig = lj_params(el1, el2)
                            # Expand type ranges
                            t1_start = self.atom_groups['sheet'][l1][el1][0]
                            t1_end = self.atom_groups['sheet'][l1][el1][-1]
                            t2_start = self.atom_groups['sheet'][l2][el2][0]
                            t2_end = self.atom_groups['sheet'][l2][el2][-1]
                            
                            f.write(f"pair_coeff {t1_start}*{t1_end} {t2_start}*{t2_end} lj/cut {eps} {sig}\n")

        # Tip-Sheet, Sub-Sheet, Tip-Sub
        def write_lj_interaction(sys_a, sys_b, group_a_getter, group_b_getter):
             elements_a = self.meta_data[sys_a]['elements']
             elements_b = self.meta_data[sys_b]['elements']
             
             for el_a in set(elements_a):
                 for el_b in set(elements_b):
                     eps, sig = lj_params(el_a, el_b)
                     
                     # Get type ranges
                     types_a = group_a_getter(sys_a, el_a)
                     types_b = group_b_getter(sys_b, el_b)
                     
                     if not types_a or not types_b: continue
                     
                     f.write(f"pair_coeff {types_a[0]}*{types_a[-1]} {types_b[0]}*{types_b[-1]} lj/cut {eps} {sig}\n")

        # Helpers to get type lists
        def get_sheet_types(sys, el):
            # Flatten all layers
            ts = []
            for l in range(layers):
                if el in self.atom_groups['sheet'][l]:
                    ts.extend(self.atom_groups['sheet'][l][el])
            return sorted(ts)
            
        def get_solid_types(sys, el):
            return sorted(self.atom_groups[sys][el])

        # Write them
        write_lj_interaction('sheet', 'tip', get_sheet_types, get_solid_types)
        write_lj_interaction('sheet', 'sub', get_sheet_types, get_solid_types)
        write_lj_interaction('sub', 'tip', get_solid_types, get_solid_types)

        # Virtual Atom (if driving)
        if is_slide and self.settings.simulation.drive_method == 'virtual_atom':
            virt_type = max_type + 1
            f.write(f"mass {virt_type} 1.0\n")
            f.write(f"pair_coeff * {virt_type} lj/cut 0.0 0.0\n")

    def write_inputs(self, sheet_dims: Dict[str, float], tip_radius: float, lat_c: float) -> None:
        """Writes the LAMMPS input scripts using Jinja2 templates."""
        
        tip_z = self.settings.geometry.tip_base_z
        gap = 3.0
        
        # Calculate total atom types including virtual
        total_types = len(self.group_definitions)
        
        # Context for System Init
        init_context = {
            'xlo': sheet_dims['xlo'], 'xhi': sheet_dims['xhi'],
            'ylo': sheet_dims['ylo'], 'yhi': sheet_dims['yhi'],
            'zhi_box': tip_z + tip_radius + 20,
            'ngroups': total_types,
            
            'sub_file': f"build/sub.lmp",
            'tip_file': f"build/tip.lmp",
            'sheet_file': f"build/{self.config.sheet.mat}_1.lmp", # Layer 1 base
            'potential_file': "potentials/system.in.settings",
            
            'tip_x': (sheet_dims['xhi'] - sheet_dims['xlo']) / 2,
            'tip_y': (sheet_dims['yhi'] - sheet_dims['ylo']) / 2,
            'tip_z': tip_z,
            'sheet_z': self.config.sub.thickness + 0.5 + gap,
            
            'sub_natypes': self.pot_data['sub']['natype'] * 3, # 3 regions
            'tip_natypes': self.pot_data['tip']['natype'] * 3, 
            'offset_2d': (self.pot_data['sub']['natype']*3) + (self.pot_data['tip']['natype']*3),
            
            # Settings
            'neighbor_list': self.settings.simulation.neighbor_list,
            'neigh_modify_command': self.settings.simulation.neigh_modify_command,
            'min_style': self.settings.simulation.min_style,
            'minimization_command': self.settings.simulation.minimization_command,
            'timestep': self.settings.simulation.timestep,
            'thermo': self.settings.simulation.thermo,
            'temp': self.config.general.temp,
            'forces': self.config.general.force,
            'thermostat_type': self.settings.thermostat.type,
            'tip_fix_group': f"tip_all" if self.settings.geometry.rigid_tip else "tip_fix",
            'dump_enabled': self.settings.output.dump['system_init'],
            'dump_freq': self.settings.output.dump_frequency['system_init'],
            'dump_file': f"visuals/system_init.lammpstrj",
            'output_dir': "results"
        }
        
        content = self.render_template("afm/system_init.lmp", init_context)
        with open(self.work_dir / "system.in", "w") as f:
            f.write(content)
            
        # Context for Slide
        slide_context = {
            'neighbor_list': self.settings.simulation.neighbor_list,
            'neigh_modify_command': self.settings.simulation.neigh_modify_command,
            'timestep': self.settings.simulation.timestep,
            'thermo': self.settings.simulation.thermo,
            'temp': self.config.general.temp,
            'forces': self.config.general.force,
            'angles': self.config.general.scan_angle,
            'data_file': "results/load_${find}N.data",
            'potential_file': "potentials/slide.in.settings",
            'extra_atom_types': 1 if self.settings.simulation.drive_method == 'virtual_atom' else 0,
            'dump_enabled': self.settings.output.dump['slide'],
            'dump_freq': self.settings.output.dump_frequency['slide'],
            'dump_file_pattern': "visuals/slide_${find}nN_${a}angle.lammpstrj",
            'thermostat_type': self.settings.thermostat.type,
            'tip_fix_group': f"tip_all" if self.settings.geometry.rigid_tip else "tip_fix",
            'results_freq': self.settings.output.results_frequency,
            'results_file_pattern': "results/friction_${find}nN_${a}angle.txt",
            'layer_group': f"layer_{max(self.config.sheet.layers)}", # Top layer
            'damp_ev': self.config.tip.dspring / 0.016,
            'spring_ev': self.config.tip.cspring / 16.02,
            'tipps': self.config.tip.s / 100,
            'drive_method': self.settings.simulation.drive_method,
            'virtual_offset': 10.0, 
            'virtual_atom_type': total_types + 1,
            'run_steps': self.settings.simulation.slide_run_steps
        }

        content = self.render_template("afm/slide.lmp", slide_context)
        with open(self.work_dir / "slide.in", "w") as f:
            f.write(content)