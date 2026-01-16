"""Potential management for FrictionSim2D.

This module handles the assignment and generation of interatomic potentials.
It supports hybrid pair styles with proper indexing, automatic mixing rules
for cross-interactions, and layer-specific potential handling for multi-layer
2D materials.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import logging

from ase.data import atomic_masses, atomic_numbers

from src.core.config import ComponentConfig, GlobalSettings
from src.core.utils import count_atomtypes, lj_params, cifread

logger = logging.getLogger(__name__)

# Potentials that include interlayer interactions internally (no separate LJ needed)
POTENTIALS_WITH_INTERNAL_LJ = {'airebo', 'comb', 'comb3', 'reaxff'}
# Potentials that require explicit LJ for cross-component/interlayer interactions
POTENTIALS_REQUIRING_LJ = {'sw', 'tersoff', 'rebo', 'edip', 'meam', 'eam', 'bop',
                            'morse', 'rebomos', 'sw/mod', 'extep', 'vashishta'}

class PotentialManager:
    """Manages potential assignments, type mapping, and pair_coeff generation.

    This class handles:
        - Sequential atom type ID assignment across all components
        - Hybrid pair_style construction with proper indexing
        - Self-interactions (many-body potentials like Tersoff, SW)
        - Cross-interactions (LJ between different components)
        - Interlayer interactions for multi-layer 2D materials
        - Langevin thermostat region type expansion (3x types)

    Attributes:
        components: List of registered component data dictionaries.
        global_atom_count: Running count of total atom types assigned.
        atom_type_map: Maps component_name -> {element: [type_ids]}.
        group_def: Maps type_id -> [group_name, type_str, element, pot_label].
        elemgroup: Maps component -> layer (or None) -> element -> [type_ids].
        potential_usage: Tracks how many times each potential type is used.
    """

    def __init__(self, settings: GlobalSettings,
                 use_langevin: Optional[bool] = None):
        """Initialize the PotentialManager.

        Args:
            settings: Global simulation settings containing potential and
                thermostat config.
            use_langevin: Override for Langevin type expansion. If None, uses
                settings. Set to False for temporary calculations like lat_c
                finding.
        """
        self.settings = settings
        # Allow explicit override, otherwise detect from settings
        if use_langevin is not None:
            self.use_langevin = use_langevin
        else:
            self.use_langevin = settings.thermostat.type == 'langevin'
        
        self.components: List[Dict[str, Any]] = []
        self.global_atom_count: int = 0
        self.atom_type_map: Dict[str, Dict[str, List[int]]] = {}
        
        # Detailed type tracking
        # group_def[type_id] = [group_name, type_id_str, element, pot_label]
        self.group_def: Dict[int, List[str]] = {}
        
        # Element grouping by component and layer
        # elemgroup[component][layer_or_element][element] = [type_ids]
        self.elemgroup: Dict[str, Dict[Any, Dict[str, List[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        
        # Track potential usage counts for hybrid indexing
        self.potential_usage: Dict[str, int] = defaultdict(int)
        self.potential_indices: Dict[str, int] = defaultdict(int)
        
        # Interaction commands
        self.self_interaction_commands: List[str] = []
        self.cross_interaction_commands: List[str] = []

    def get_single_component_commands(
        self,
        config: ComponentConfig,
        elements: List[str]
    ) -> List[str]:
        """Generate LAMMPS interatomic potential settings for single-component setup.

        This is useful for standalone simulations like amorphisation where
        only one component is used and no cross-interactions are needed.
        Supports all potential types in POTENTIALS_REQUIRING_LJ and
        POTENTIALS_WITH_INTERNAL_LJ.

        Args:
            config: Component configuration with pot_type and pot_path.
            elements: List of element symbols in the material.

        Returns:
            List of LAMMPS commands (pair_style, pair_coeff, mass).
        """
        commands = []
        pot_type = config.pot_type.lower()
        pot_path = config.pot_path
        
        # Pair style
        commands.append(f"pair_style {pot_type}")
        
        # Pair coeff - format depends on potential type
        element_str = ' '.join(elements)
        commands.append(f"pair_coeff * * {pot_path} {element_str}")
        
        # Mass commands
        for i, elem in enumerate(elements, 1):
            try:
                mass = atomic_masses[atomic_numbers[elem]]
                commands.append(f"mass {i} {mass:.6f}")
            except (KeyError, IndexError):
                logger.warning(f"Could not find mass for element '{elem}', using 1.0")
                commands.append(f"mass {i} 1.0")
        
        return commands

    def register_component(
        self,
        name: str,
        config: ComponentConfig,
        n_layers: int = 1
    ) -> Dict[str, List[int]]:
        """Register a material component and assign unique global atom type IDs.

        For multi-layer 2D materials, each layer gets its own set of atom types.
        For Langevin thermostat (when self.use_langevin is True), each element
        gets 3 types (mobile, fixed, thermo).

        Args:
            name: Unique identifier (e.g., 'tip', 'sheet', 'sub').
            config: The material configuration.
            n_layers: Number of layers (for 2D materials, default 1).

        Returns:
            Dict mapping element names to their global atom type IDs.
        """
        # 1. Read material data
        cif_data = cifread(config.cif_path)
        elements = cif_data['elements']
        
        # 2. Get atom type counts from potential file
        pot_counts = count_atomtypes(config.pot_path, elements)
        
        # 3. Track potential usage for hybrid indexing
        self.potential_usage[config.pot_type] += n_layers if self.is_sheet_lj(config.pot_type) else 1
        
        component_map: Dict[str, List[int]] = {}
        
        # Langevin only applies to tip and substrate, never to sheets
        apply_langevin = self.use_langevin and name in ('tip', 'sub')
        
        # 4. Assign atom types based on configuration
        if n_layers > 1:
            # Multi-layer 2D material: separate types per layer (never Langevin)
            component_map = self._assign_layer_types(
                name, elements, pot_counts, n_layers
            )
        elif apply_langevin:
            # Tip/substrate with Langevin: 3 types per element
            component_map = self._assign_langevin_types(
                name, elements, pot_counts
            )
        else:
            # Standard: one set of types
            component_map = self._assign_standard_types(
                name, elements, pot_counts
            )
        
        # Store component data
        self.atom_type_map[name] = component_map
        self.components.append({
            'name': name,
            'config': config,
            'map': component_map,
            'elements': elements,
            'n_layers': n_layers,
            'use_langevin': self.use_langevin
        })
        
        return component_map

    def _assign_standard_types(
        self,
        name: str,
        elements: List[str],
        pot_counts: Dict[str, int]
    ) -> Dict[str, List[int]]:
        """Assign standard atom types (one set per component).

        Args:
            name: Component name.
            elements: List of element symbols.
            pot_counts: Dict of atom counts per element.

        Returns:
            Dict mapping element to list of type IDs.
        """
        component_map = {}
        type_index = 1
        
        for el in elements:
            count = pot_counts.get(el, 1)
            ids = []
            
            for t in range(count):
                atype = self.global_atom_count + 1
                self.global_atom_count += 1
                ids.append(atype)
                
                # Generate potential label (e.g., 'Mo1' or just 'Mo')
                pot_label = el if count == 1 else f"{el}{t+1}"
                
                self.group_def[atype] = [
                    f"{name}_t{type_index}",  # group_name
                    str(atype),                # type_id_str
                    el,                        # element
                    pot_label                  # pot_label for pair_coeff
                ]
                self.elemgroup[name][None][el].append(atype)
                type_index += 1
                
            component_map[el] = ids
            
        return component_map

    def _assign_layer_types(
        self,
        name: str,
        elements: List[str],
        pot_counts: Dict[str, int],
        n_layers: int
    ) -> Dict[str, List[int]]:
        """Assign atom types for multi-layer 2D materials.

        Uses element-first ordering to match the LAMMPS renumbering loop in
        stack_multilayer_sheet: for each element, assign types for all layers,
        then move to the next element.

        Order: Mo L1 (6 types), Mo L2 (6 types), S L1 (6 types), S L2 (6 types)

        Note: Langevin regions are never applied to sheets - only tip/substrate.

        Args:
            name: Component name.
            elements: List of element symbols.
            pot_counts: Dict of atom counts per element.
            n_layers: Number of layers.

        Returns:
            Dict mapping element to list of type IDs.
        """
        component_map = defaultdict(list)
        type_index = 1
        
        # Element-first ordering: for each element, iterate through all layers
        for el in elements:
            count = pot_counts.get(el, 1)
            for layer in range(n_layers):
                for t in range(count):
                    atype = self.global_atom_count + 1
                    self.global_atom_count += 1
                    component_map[el].append(atype)
                    
                    pot_label = el if count == 1 else f"{el}{t+1}"
                    self.group_def[atype] = [
                        f"{name}_l{layer+1}_t{type_index}",
                        str(atype),
                        el,
                        pot_label
                    ]
                    self.elemgroup[name][layer][el].append(atype)
                    
                    type_index += 1
                    
        return dict(component_map)

    def _assign_langevin_types(
        self,
        name: str,
        elements: List[str],
        pot_counts: Dict[str, int]
    ) -> Dict[str, List[int]]:
        """Assign types for Langevin thermostat (3 regions per element).

        Args:
            name: Component name.
            elements: List of element symbols.
            pot_counts: Dict of atom counts per element.

        Returns:
            Dict mapping element to list of type IDs (tripled).
        """
        component_map = defaultdict(list)
        type_index = 1
        
        for el in elements:
            count = pot_counts.get(el, 1)
            
            for t in range(count):
                pot_label = el if count == 1 else f"{el}{t+1}"
                
                # Three regions: mobile, fixed, thermostat
                for suffix in ['', '_fix', '_thermo']:
                    atype = self.global_atom_count + 1
                    self.global_atom_count += 1
                    component_map[el].append(atype)
                    
                    self.group_def[atype] = [
                        f"{name}{suffix}_t{type_index}",
                        str(atype),
                        el,
                        pot_label
                    ]
                    self.elemgroup[name][None][el].append(atype)
                    
                type_index += 1
                
        return dict(component_map)

    def is_sheet_lj(self, pot_type: str) -> bool:
        """Determine if potential requires explicit LJ for interlayer/cross interactions.

        Args:
            pot_type: The potential type string (e.g., 'sw', 'tersoff', 'airebo').

        Returns:
            True if explicit LJ is needed, False if potential handles it internally.
        """
        pot_lower = pot_type.lower()
        if pot_lower in POTENTIALS_WITH_INTERNAL_LJ:
            return False
        if pot_lower in POTENTIALS_REQUIRING_LJ:
            return True
        # Default: assume LJ is needed for safety
        logger.warning(f"Unknown potential type '{pot_type}', assuming LJ is required.")
        return True

    def add_self_interaction(self, component_name: str,
                            layer: Optional[int] = None):
        """Add the self-interaction (many-body potential) for a component.

        This generates pair_coeff commands with NULL mapping for hybrid style.
        For multi-layer materials, can specify a specific layer or all layers.

        Args:
            component_name: Name of the registered component.
            layer: Specific layer index (0-based) or None for all/single layer.
        """
        comp = self._get_component(component_name)
        c_conf = comp['config']
        n_layers = comp['n_layers']
        
        # Increment potential index for this usage
        self.potential_indices[c_conf.pot_type] += 1
        pot_index = self.potential_indices[c_conf.pot_type]
        
        # Determine if we need indexing in hybrid style
        needs_index = self.potential_usage[c_conf.pot_type] > 1
        index_str = f" {pot_index}" if needs_index else ""
        
        if n_layers > 1 and self.is_sheet_lj(c_conf.pot_type):
            # Multi-layer: each layer gets its own potential entry
            layers_to_process = [layer] if layer is not None else range(n_layers)
            for l in layers_to_process:
                self._add_layer_self_interaction(comp, l, pot_index)
        else:
            # Single component or potential that handles all internally
            atom_list = self._build_null_map(component_name)
            cmd = f"pair_coeff * * {c_conf.pot_type}{index_str} {c_conf.pot_path} {' '.join(atom_list)}"
            self.self_interaction_commands.append(f"{cmd} # {component_name}")

    def _add_layer_self_interaction(self, comp: Dict, layer: int,
                                     base_pot_index: int):
        """Add self-interaction for a specific layer.

        Args:
            comp: Component dictionary.
            layer: Layer index (0-based).
            base_pot_index: Base potential index for hybrid style.
        """
        c_conf = comp['config']
        name = comp['name']
        
        # Each layer gets its own potential index
        pot_index = base_pot_index + layer
        self.potential_indices[c_conf.pot_type] = pot_index
        
        needs_index = self.potential_usage[c_conf.pot_type] > 1
        index_str = f" {pot_index}" if needs_index else ""
        
        # Build NULL map for this specific layer
        atom_list = []
        for i in range(1, self.global_atom_count + 1):
            gdef = self.group_def.get(i)
            if gdef and f"{name}_l{layer+1}" in gdef[0]:
                atom_list.append(gdef[3])  # pot_label
            else:
                atom_list.append("NULL")
        
        cmd = f"pair_coeff * * {c_conf.pot_type}{index_str} {c_conf.pot_path} {' '.join(atom_list)}"
        self.self_interaction_commands.append(f"{cmd} # {name} Layer {layer+1}")

    def _build_null_map(self, component_name: str) -> List[str]:
        """Build the NULL-padded element list for a component's pair_coeff.

        For hybrid potentials, each entry corresponds to a LAMMPS type ID.
        Types belonging to this component get their element/pot_label,
        all others get 'NULL'.

        Args:
            component_name: Name of the component.

        Returns:
            List of element symbols or 'NULL' strings.
        """
        atom_list = []
        for i in range(1, self.global_atom_count + 1):
            gdef = self.group_def.get(i)
            if gdef and component_name in gdef[0]:
                atom_list.append(gdef[3])  # pot_label (e.g., 'Mo1', 'S1', 'Si')
            else:
                atom_list.append("NULL")
        return atom_list

    def add_cross_interaction(
        self,
        comp1_name: str,
        comp2_name: str,
        interaction_type: str = "lj/cut",
        custom_params: Optional[Dict[str, float]] = None
    ):
        """Add LJ cross-interaction between two components.

        Args:
            comp1_name: Name of first component.
            comp2_name: Name of second component.
            interaction_type: Interaction style (default 'lj/cut').
            custom_params: Optional dict with 'epsilon', 'sigma', 'cutoff'
                overrides.
        """
        comp1 = self.atom_type_map.get(comp1_name)
        comp2 = self.atom_type_map.get(comp2_name)
        
        if not comp1 or not comp2:
            raise ValueError(f"Components '{comp1_name}' and/or '{comp2_name}' not found.")
        
        # Generate interactions for all element pairs
        for el1 in comp1.keys():
            for el2 in comp2.keys():
                # Get or calculate LJ parameters
                if custom_params:
                    epsilon = custom_params.get('epsilon', 0.0)
                    sigma = custom_params.get('sigma', 3.0)
                else:
                    epsilon, sigma = lj_params(el1, el2)
                
                # Get type ranges for efficient pair_coeff
                types1 = self._get_element_type_range(comp1_name, el1)
                types2 = self._get_element_type_range(comp2_name, el2)
                
                # Generate command (LAMMPS prefers lower type first)
                if types1[0] > types2[0]:
                    types1, types2 = types2, types1
                    
                cmd = f"pair_coeff {types1} {types2} {interaction_type} {epsilon:.6f} {sigma:.4f}"
                self.cross_interaction_commands.append(f"{cmd} # {el1}({comp1_name})-{el2}({comp2_name})")

    def add_interlayer_interaction(
        self,
        component_name: str,
        layer_pairs: Optional[List[Tuple[int, int]]] = None,
        low_interaction: bool = False
    ):
        """Add LJ interactions between layers of a multi-layer 2D material.

        Args:
            component_name: Name of the sheet component.
            layer_pairs: List of (layer_i, layer_j) tuples, or None for all
                pairs.
            low_interaction: If True, use near-zero epsilon (ghost/weak
                interaction).
        """
        comp = self._get_component(component_name)
        n_layers = comp['n_layers']
        elements = comp['elements']
        
        if n_layers < 2:
            logger.warning(f"Component '{component_name}' has only 1 layer, skipping interlayer interaction.")
            return
            
        # Generate all layer pairs if not specified
        if layer_pairs is None:
            layer_pairs = [(i, j) for i in range(n_layers) for j in range(i+1, n_layers)]
        
        for layer_i, layer_j in layer_pairs:
            for el1 in elements:
                for el2 in elements:
                    epsilon, sigma = lj_params(el1, el2)
                    if low_interaction:
                        epsilon = 1e-100  # Near-zero for ghost interaction
                    
                    # Get types for each layer
                    types_i = self.elemgroup[component_name][layer_i][el1]
                    types_j = self.elemgroup[component_name][layer_j][el2]
                    
                    if not types_i or not types_j:
                        continue
                    
                    t1_str = self._format_type_range(types_i)
                    t2_str = self._format_type_range(types_j)
                    
                    # Ensure ordering
                    if types_i[0] > types_j[0]:
                        t1_str, t2_str = t2_str, t1_str
                    
                    cmd = f"pair_coeff {t1_str} {t2_str} lj/cut {epsilon:.6f} {sigma:.4f}"
                    self.cross_interaction_commands.append(
                        f"{cmd} # {el1}(L{layer_i+1})-{el2}(L{layer_j+1})"
                    )

    def add_interlayer_lj_by_distance(
        self,
        component_name: str,
        max_real_distance: int = 1
    ):
        """Add all interlayer LJ interactions with distance-based ghost handling.

        Layer pairs within max_real_distance get real LJ interactions.
        Layer pairs beyond max_real_distance get ghost (near-zero epsilon)
        interactions to prevent interpenetration without affecting dynamics
        significantly.

        Args:
            component_name: Name of the multi-layer component.
            max_real_distance: Maximum layer separation for real LJ interactions.
                - 1 (default): Only adjacent pairs (L1-L2, L2-L3, L3-L4) are
                  real.
                - 2: Adjacent + next-nearest (L1-L3, L2-L4) are real.
                - n_layers or higher: All pairs are real (no ghost
                  interactions).

        Examples:
            For a 4-layer sheet with max_real_distance=1:
                Real LJ: L1-L2, L2-L3, L3-L4
                Ghost:   L1-L3, L1-L4, L2-L4

            For a 4-layer sheet with max_real_distance=2:
                Real LJ: L1-L2, L2-L3, L3-L4, L1-L3, L2-L4
                Ghost:   L1-L4
        """
        comp = self._get_component(component_name)
        n_layers = comp['n_layers']
        
        if n_layers < 2:
            logger.warning(f"Component '{component_name}' has only 1 layer, skipping interlayer interactions.")
            return
        
        # Separate pairs by distance
        real_pairs = []
        ghost_pairs = []
        
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                distance = j - i
                if distance <= max_real_distance:
                    real_pairs.append((i, j))
                else:
                    ghost_pairs.append((i, j))
        
        # Add real LJ interactions
        if real_pairs:
            self.add_interlayer_interaction(component_name, layer_pairs=real_pairs, low_interaction=False)
        
        # Add ghost interactions (prevent interpenetration)
        if ghost_pairs:
            self.add_interlayer_interaction(component_name, layer_pairs=ghost_pairs, low_interaction=True)

    def _get_element_type_range(self, comp_name: str,
                                element: str) -> str:
        """Get the LAMMPS type range string for an element in a component.

        Args:
            comp_name: Component name.
            element: Element symbol.

        Returns:
            Format like '1*3' for types 1,2,3 or just '5' for single type.
        """
        all_types = []
        eg = self.elemgroup[comp_name]
        
        # Collect types from all layers/regions
        for layer_key in eg:
            if element in eg[layer_key]:
                all_types.extend(eg[layer_key][element])
        
        if not all_types:
            # Fallback to atom_type_map
            all_types = self.atom_type_map.get(comp_name, {}).get(element, [])
            
        return self._format_type_range(all_types)

    def _format_type_range(self, types: List[int]) -> str:
        """Format a list of type IDs into LAMMPS range notation.

        Args:
            types: List of type IDs.

        Returns:
            LAMMPS range notation string (e.g., '1*3' or '5').
        """
        if not types:
            return ""
        types = sorted(types)
        if len(types) == 1:
            return str(types[0])
        return f"{types[0]}*{types[-1]}"

    def _get_component(self, name: str) -> Dict[str, Any]:
        """Retrieve a registered component by name.

        Args:
            name: Component name.

        Returns:
            Component data dictionary.

        Raises:
            ValueError: If component not found.
        """
        comp = next((c for c in self.components if c['name'] == name), None)
        if not comp:
            raise ValueError(f"Component '{name}' not registered.")
        return comp

    def calculate_gap(self, comp1_name: str, comp2_name: str,
                      buffer: float = 0.5) -> float:
        """Calculate the recommended gap between two components.

        Based on the maximum sigma from UFF mixing rules plus a buffer.

        Args:
            comp1_name: Name of first component.
            comp2_name: Name of second component.
            buffer: Additional distance to add (default 0.5 Angstrom).

        Returns:
            Recommended gap distance in Angstroms.
        """
        comp1 = self._get_component(comp1_name)
        comp2 = self._get_component(comp2_name)

        max_sigma = 0.0
        for e1 in set(comp1['elements']):
            for e2 in set(comp2['elements']):
                _, sigma = lj_params(e1, e2)
                max_sigma = max(max_sigma, sigma)
                    
        return max_sigma + buffer

    def get_masses_string(self) -> str:
        """Generate LAMMPS mass commands for all registered types.

        Returns:
            String with mass commands, one per line.
        """
        # Group types by element
        element_types: Dict[str, List[int]] = defaultdict(list)
        for atype in sorted(self.group_def.keys()):
            gdef = self.group_def[atype]
            element = gdef[2]
            element_types[element].append(atype)
        
        lines = []
        for element, types in element_types.items():
            types = sorted(types)
            try:
                mass = atomic_masses[atomic_numbers[element]]
                type_range = self._format_type_range(types)
                lines.append(f"mass {type_range} {mass:.6f} #{element}")
            except (KeyError, IndexError):
                logger.warning(f"Could not find mass for element '{element}', using 1.0")
                type_range = self._format_type_range(types)
                lines.append(f"mass {type_range} 1.0 #{element} (unknown)")
        
        return "\n".join(lines)

    def get_layer_groups_string(self) -> str:
        """Generate LAMMPS group commands for multi-layer components only.

        Only outputs layer groups for components with n_layers > 1.
        Single-layer components define their groups in the main LAMMPS script.

        Returns:
            String with group commands like:
                group layer_1 type 1 2 9 10
                group layer_2 type 3 4 11 12
        """
        lines = []
        for comp in self.components:
            name = comp['name']
            n_layers = comp['n_layers']
            
            # Only output layer groups for multi-layer components
            if n_layers > 1:
                for layer in range(n_layers):
                    types = self.get_layer_group_string(name, layer)
                    if types:
                        lines.append(f"group layer_{layer+1} type {types}")
        
        return "\n".join(lines)

    def write_file(self, output_path: Path):
        """Write the complete potential settings file.

        Args:
            output_path: Path for the output file (e.g., system.in.settings).
        """
        # Build pair_style line - repeat potential name for each usage (no indices)
        # LAMMPS syntax: pair_style hybrid sw sw lj/cut 8.0 (indices only in pair_coeff)
        style_parts = []
        for pot_type, count in self.potential_usage.items():
            for _ in range(count):
                style_parts.append(pot_type)
        
        # Always add lj/cut with cutoff if we have cross-interactions
        lj_cutoff = self.settings.potential.lj_cutoff
        if self.cross_interaction_commands:
            style_parts.append(f"lj/cut {lj_cutoff}")
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Masses (grouped by element)
            f.write(self.get_masses_string())
            f.write("\n")
            
            # Layer groups
            f.write(self.get_layer_groups_string())
            f.write("\n")
            
            # Pair style
            f.write(f"pair_style hybrid {' '.join(style_parts)}\n")
            
            # Self-interactions (many-body potentials)
            if self.self_interaction_commands:
                f.write("# Self-interactions (intra-component)\n")
                for cmd in self.self_interaction_commands:
                    f.write(f"{cmd}\n")
                f.write("\n")
            
            # Cross-interactions (LJ)
            if self.cross_interaction_commands:
                f.write("# Cross-interactions (inter-component LJ)\n")
                for cmd in self.cross_interaction_commands:
                    f.write(f"{cmd}\n")
        
        logger.info(f"Wrote potential settings to {output_path}")

    def get_group_string(self, component_name: str) -> str:
        """Return space-separated type IDs for a component (for LAMMPS grouping).

        Args:
            component_name: Name of the registered component.

        Returns:
            Space-separated string of type IDs (e.g., "1 2 3").
        """
        if component_name not in self.atom_type_map:
            return ""
        all_ids = []
        for ids in self.atom_type_map[component_name].values():
            all_ids.extend(ids)
        return " ".join(map(str, sorted(set(all_ids))))

    def get_layer_group_string(self, component_name: str,
                               layer: int) -> str:
        """Return type IDs for a specific layer of a component.

        Args:
            component_name: Name of the component.
            layer: Layer index (0-based).

        Returns:
            Space-separated string of type IDs for that layer.
        """
        eg = self.elemgroup.get(component_name, {}).get(layer, {})
        all_ids = []
        for ids in eg.values():
            all_ids.extend(ids)
        return " ".join(map(str, sorted(set(all_ids))))

    def get_total_types(self) -> int:
        """Return the total number of atom types registered.

        Returns:
            Total count of atom types.
        """
        return self.global_atom_count