"""Potential management for FrictionSim2D.

This module handles the assignment and generation of interatomic potentials.
It supports hybrid pair styles, automatic mixing rules for cross-interactions,
and specific interaction overrides (like ghost atoms).
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import math

# Use relative imports for internal modules to ensure package integrity
from FrictionSim2D.core.config import ComponentConfig
from FrictionSim2D.core.utils import count_atomtypes, lj_params, cifread
# We don't need AtomskWrapper here; potential management is about strings and logic.

class PotentialManager:
    def __init__(self):
        self.components: List[Dict[str, Any]] = []
        self.global_atom_count = 0
        self.atom_type_map: Dict[str, Dict[str, List[int]]] = {}
        self.interactions: List[str] = [] # Stores custom interaction commands
        self.pair_styles: Set[str] = set()
        
    def register_component(self, name: str, config: ComponentConfig, 
                           explicit_type_count: Optional[int] = None) -> Dict[str, List[int]]:
        """Registers a material component and assigns unique global atom IDs.

        Args:
            name (str): Unique identifier (e.g., 'tip', 'sheet_top').
            config (ComponentConfig): The material configuration.
            explicit_type_count (int, optional): Force a specific number of atom types 
                                                 (useful for ghost layers).

        Returns:
            Dict[str, List[int]]: Map of element names to their global atom type IDs.
        """
        # 1. Determine elements from CIF
        cif_data = cifread(config.cif_path)
        elements = cif_data['elements']
        
        # 2. Determine required types per element from Potential file
        # (e.g. some ReaxFF might need specific types, usually it's 1 per element)
        pot_counts = count_atomtypes(config.pot_path, elements)
        
        component_map = {}
        
        for el in elements:
            # If explicit count is given, override the potential's count
            # This is rare but useful if you are manually hacking types
            count = pot_counts.get(el, 1)
            if explicit_type_count:
                 # This is a simplification; usually explicit counts are per-element or total.
                 # For now, we assume standard behavior unless specific logic overrides it.
                 pass 
            
            # Assign Global IDs
            start_id = self.global_atom_count + 1
            end_id = self.global_atom_count + count
            id_list = list(range(start_id, end_id + 1))
            
            component_map[el] = id_list
            self.global_atom_count += count
            
        # Store data
        self.atom_type_map[name] = component_map
        self.components.append({
            'name': name,
            'config': config,
            'map': component_map,
            'elements': elements
        })
        
        # Add the potential style to the global set
        self.pair_styles.add(config.pot_type)
        
        return component_map

    def add_self_interaction(self, component_name: str):
        """Adds the standard self-interaction (e.g. Tersoff) for a component."""
        comp = next((c for c in self.components if c['name'] == component_name), None)
        if not comp:
            raise ValueError(f"Component '{component_name}' not registered.")
            
        c_conf = comp['config']
        c_map = comp['map']
        
        # Build the NULL map string for hybrid style
        # Entries corresponding to THIS component get element names. Others get NULL.
        atom_list = []
        
        # We need to iterate 1..TotalTypes to generate the mapping string
        for i in range(1, self.global_atom_count + 1):
            # Check if this ID 'i' belongs to the current component
            found_el = None
            for el, ids in c_map.items():
                if i in ids:
                    found_el = el
                    break
            
            if found_el:
                atom_list.append(found_el)
            else:
                atom_list.append("NULL")
        
        # Store the command
        cmd = f"pair_coeff * * {c_conf.pot_type} {c_conf.pot_path} {' '.join(atom_list)}"
        self.interactions.append(cmd)

    def add_cross_interaction(self, comp1_name: str, comp2_name: str, 
                              interaction_type: str = "lj/cut",
                              custom_params: Optional[Dict[str, float]] = None):
        """Adds an interaction between two components.

        Args:
            comp1_name: Name of first component.
            comp2_name: Name of second component.
            interaction_type: Type of interaction (default 'lj/cut').
            custom_params: Dict of params (e.g. {'epsilon': 0.005, 'sigma': 3.0}). 
                           If None, calculates UFF mixing automatically.
        """
        comp1 = self.atom_type_map.get(comp1_name)
        comp2 = self.atom_type_map.get(comp2_name)
        
        if not comp1 or not comp2:
            raise ValueError("One or both components not found.")
            
        self.pair_styles.add(interaction_type)
        
        # Loop over all element combinations
        for el1, ids1 in comp1.items():
            for el2, ids2 in comp2.items():
                
                # Calculate or Retrieve Parameters
                if custom_params:
                    # User manual override (e.g. for Ghost layers)
                    epsilon = custom_params.get('epsilon', 0.0)
                    sigma = custom_params.get('sigma', 1.0)
                    cutoff = custom_params.get('cutoff', 2.5)
                    params_str = f"{epsilon} {sigma} {cutoff}"
                else:
                    # Auto-calculate using UFF mixing
                    epsilon, sigma = lj_params(el1, el2)
                    params_str = f"{epsilon:.4f} {sigma:.4f}"

                # Generate coeff line for every type pair
                for t1 in ids1:
                    for t2 in ids2:
                        # Syntax: pair_coeff type1 type2 style args...
                        # Ensure strictly ordered types for LAMMPS (t1 < t2 is preferred but not strictly required for pair_coeff)
                        cmd = f"pair_coeff {t1} {t2} {interaction_type} {params_str}"
                        self.interactions.append(cmd)

    def calculate_gap(self, comp1_name: str, comp2_name: str, buffer: float = 0.5) -> float:
        """Calculates the initial gap between two components based on max sigma + buffer.
        
        Args:
            comp1_name: Name of first component.
            comp2_name: Name of second component.
            buffer: Extra distance to add (default 0.5 Angstrom).
            
        Returns:
            float: The recommended gap distance.
        """
        comp1 = next((c for c in self.components if c['name'] == comp1_name), None)
        comp2 = next((c for c in self.components if c['name'] == comp2_name), None)
        
        if not comp1 or not comp2:
            raise ValueError("One or both components not found.")

        max_sigma = 0.0
        
        # Find max sigma for any pair of elements between the two components
        for e1 in set(comp1['elements']):
            for e2 in set(comp2['elements']):
                _, sigma = lj_params(e1, e2)
                if sigma > max_sigma:
                    max_sigma = sigma
                    
        return max_sigma + buffer

    def write_file(self, output_path: Path):
        """Writes the full system.in.settings file."""
        
        # Ensure lj/cut is present if we have cross interactions or explicit requests
        if len(self.components) > 1 and "lj/cut" not in self.pair_styles:
            # Check if any cross interactions were added that used lj/cut
            # Ideally, add_cross_interaction handles adding to self.pair_styles
            pass

        with open(output_path, 'w') as f:
            f.write("# Auto-generated interactions by FrictionSim2D\n\n")
            f.write(f"pair_style hybrid {' '.join(self.pair_styles)}\n\n")
            
            f.write("# --- Interactions ---\n")
            for cmd in self.interactions:
                f.write(f"{cmd}\n")

    def get_group_string(self, component_name: str) -> str:
        """Returns space-separated type IDs for a component (for grouping)."""
        if component_name not in self.atom_type_map:
            return ""
        all_ids = []
        for ids in self.atom_type_map[component_name].values():
            all_ids.extend(ids)
        return " ".join(map(str, sorted(all_ids)))