"""Component builders for FrictionSim2D.

This module contains the logic to construct the physical components of the simulation:
the AFM tip, the substrate, and the 2D material sheet. It handles the calculation
of duplication factors based on requested dimensions and unit cell properties.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

from lammps import lammps

from FrictionSim2D.core.config import TipConfig, SubstrateConfig, SheetConfig, GlobalSettings
from FrictionSim2D.core.utils import get_model_dimensions, get_material_path
from FrictionSim2D.interfaces.atomsk import AtomskWrapper
from FrictionSim2D.core.potential_manager import PotentialManager

logger = logging.getLogger(__name__)

def _calculate_duplication(
    cif_path: Path, 
    target_x: float, 
    target_y: float, 
    target_z: Optional[float] = None,
    buffer_x: float = 15.0,
    buffer_y: float = 15.0
) -> Tuple[List[int], Dict[str, float]]:
    """Calculates duplication factors to meet target dimensions."""
    # Strategy: Create a temporary 1x1x1 orthogonal cell to get true dimensions
    atomsk = AtomskWrapper()
    temp_file = cif_path.with_suffix('.temp.lmp')
    
    # Create a 2x2x1 supercell first to ensure valid orthogonality transformations
    atomsk.create_slab(cif_path, temp_file, pre_duplicate=[2, 2, 1])
    atomsk.orthogonalize(temp_file, temp_file)
    
    dims = get_model_dimensions(temp_file)
    len_x = dims['xhi'] - dims['xlo']
    len_y = dims['yhi'] - dims['ylo']
    len_z = dims['zhi'] - dims['zlo']
    
    unit_x = len_x / 2.0
    unit_y = len_y / 2.0
    unit_z = len_z
    
    temp_file.unlink()

    dup_x = max(1, round((target_x + buffer_x) / unit_x))
    dup_y = max(1, round((target_y + buffer_y) / unit_y))
    
    dup_z = 1
    if target_z is not None:
        dup_z = max(1, round(target_z / unit_z))

    return [dup_x, dup_y, dup_z], dims

def run_lammps_commands(commands: List[str]) -> None:
    """Runs a list of LAMMPS commands using the Python interface."""
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        for cmd in commands:
            lmp.command(cmd)
    except Exception as e:
        logger.error(f"LAMMPS execution failed: {e}")
        raise
    finally:
        lmp.close()

def calculate_layer_shifts(mat_name: str, dims: Dict[str, float]) -> Tuple[float, float]:
    """Calculates stacking shifts based on material type."""
    if mat_name.startswith('p-') or mat_name == 'black_phosphorus':
        sx = (dims['xhi'] - dims['xlo']) / 2
        sy = (dims['yhi'] - dims['ylo']) / 2
    else:
        sx = 0
        sy = (dims['yhi'] - dims['ylo']) / 3
    return sx, sy

def find_optimal_lat_c(
    base_layer_path: Path, 
    config: SheetConfig, 
    build_dir: Path,
    box_dims: Dict[str, float]
) -> float:
    """Runs a local LAMMPS minimization to find the optimal interlayer distance (lat_c)."""
    
    # 1. Setup Temporary Simulation
    pot_file = build_dir / "calc_c.in.settings"
    
    # We need to generate potentials for this temporary check
    # We treat the single sheet as a component to get self-interactions
    pm = PotentialManager()
    pm.register_component("sheet", config)
    pm.add_self_interaction("sheet")
    
    # We also need cross interactions (LJ) between layers for the minimization to work
    if "lj" not in config.pot_type:
         pm.add_cross_interaction("sheet", "sheet", interaction_type="lj/cut")
         
    pm.write_file(pot_file)

    # 2. Construct LAMMPS Input
    sx, sy = calculate_layer_shifts(config.mat, box_dims)
    initial_guess = 6.0
    
    cmds = [
        "clear", "units metal", "atom_style atomic", "boundary p p p",
        # Box large enough for 2 layers
        f"region box block {box_dims['xlo']} {box_dims['xhi']} {box_dims['ylo']} {box_dims['yhi']} -10 20 units box",
        "create_box 2 box", 
        
        # Read Layer 1
        f"read_data {base_layer_path} add append shift 0 0 0 group layer_1",
        # Read Layer 2 (shifted up)
        f"read_data {base_layer_path} add append shift {sx} {sy} {initial_guess} group layer_2",
        
        f"include {pot_file}",
        
        # Minimization
        "minimize 1.0e-4 1.0e-6 100 1000",
        
        # Compute COMs
        "compute l1_com layer_1 com",
        "compute l2_com layer_2 com",
        "run 0",
        "variable z1 equal c_l1_com[3]",
        "variable z2 equal c_l2_com[3]",
        "variable lat_c equal v_z2-v_z1"
    ]
    
    # 3. Run and Extract
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        for cmd in cmds:
            lmp.command(cmd)
        lat_c = lmp.extract_variable("lat_c", None, 0)
    except Exception as e:
        logger.error(f"Failed to calculate lat_c: {e}")
        lat_c = 6.0 # Fallback
    finally:
        lmp.close()
        
    return lat_c

def stack_sheets(
    layers_info: List[Dict[str, Any]], 
    box_dims: Dict[str, float], 
    output_path: Path,
    total_natypes: int
) -> Path:
    """Generically stacks multiple layers into a single LAMMPS data file."""
    if not layers_info:
        raise ValueError("No layers provided to stack.")
        
    max_z = max([l['shift'][2] for l in layers_info]) + 20.0
    
    cmds = [
        "clear",
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        f"region box block {box_dims['xlo']} {box_dims['xhi']} {box_dims['ylo']} {box_dims['yhi']} -5 {max_z}",
        f"create_box {total_natypes} box",
    ]
    
    for i, layer in enumerate(layers_info):
        path = layer['path']
        sx, sy, sz = layer.get('shift', (0, 0, 0))
        group_id = layer.get('id', i + 1)
        group_name = f"layer_{group_id}"
        
        # Read data
        cmds.append(f"read_data {path} add append shift {sx} {sy} {sz} group {group_name}")

    cmds.append(f"write_data {output_path}")
    run_lammps_commands(cmds)
    return output_path

def build_tip(
    config: TipConfig, 
    atomsk: AtomskWrapper, 
    build_dir: Path, 
    settings: GlobalSettings
) -> Tuple[Path, float]:
    """Builds the AFM tip structure."""
    cif_path = Path(config.cif_path)
    if not cif_path.exists():
        cif_path = get_material_path(config.cif_path, 'cif') # Fallback if user gave relative name
    
    base_lmp = build_dir / f"{config.mat}_base.lmp"
    final_lmp = build_dir / "tip.lmp"
    radius = config.r

    # Calculate geometry
    box_size = radius * 2.0
    duplication, _ = _calculate_duplication(cif_path, box_size, box_size, box_size)

    # Build Base Block
    atomsk.create_slab(cif_path, base_lmp, duplicate=duplication, orthogonalize=True)

    # Handle Amorphous
    if config.amorph == 'a':
        # TODO: Call melt-quench builder here if required
        pass

    # Carve Sphere (LAMMPS)
    h = radius / settings.geometry.tip_reduction_factor
    commands = [
        "clear", "units metal", "atom_style atomic", "boundary p p p",
        f"read_data {base_lmp}",
        f"region afm_tip sphere 0 0 {radius} {radius} side in units box",
        f"region box block -{radius} {radius} -{radius} {radius} -3 {h} units box",
        "region tip intersect 2 afm_tip box",
        "group tip region tip",
        "group delete_atoms subtract all tip",
        "delete_atoms group delete_atoms",
        f"change_box all x final -{radius} {radius} y final -{radius} {radius} z final -3 {h+1}",
        "reset_atoms id",
        f"write_data {final_lmp}"
    ]
    run_lammps_commands(commands)
    
    return final_lmp, radius

def build_substrate(
    config: SubstrateConfig,
    atomsk: AtomskWrapper,
    build_dir: Path,
    box_dims: dict
) -> Path:
    """Builds the substrate slab."""
    cif_path = Path(config.cif_path)
    if not cif_path.exists():
         cif_path = get_material_path(config.cif_path, 'cif')

    output_path = build_dir / "sub.lmp"
    
    duplication, _ = _calculate_duplication(
        cif_path, 
        box_dims['xhi'] - box_dims['xlo'], 
        box_dims['yhi'] - box_dims['ylo'], 
        config.thickness,
        buffer_x=0, buffer_y=0
    )
    
    atomsk.create_slab(cif_path, output_path, duplicate=duplication, orthogonalize=True)
    atomsk.center_system(output_path, output_path, axis='z')
    
    return output_path

def build_sheet(
    config: SheetConfig,
    atomsk: AtomskWrapper,
    build_dir: Path,
    stack_if_multi: bool = False
) -> Tuple[Path, dict, float]:
    """Builds the 2D material sheet."""
    cif_path = Path(config.cif_path)
    if not cif_path.exists():
         cif_path = get_material_path(config.cif_path, 'cif')

    base_name = f"{config.mat}_1.lmp"
    base_path = build_dir / base_name
    
    # 1. Calculate Duplication for Target Size
    target_x = config.x if isinstance(config.x, (int, float)) else config.x[0]
    target_y = config.y if isinstance(config.y, (int, float)) else config.y[0]
    
    dup, dims = _calculate_duplication(cif_path, target_x, target_y, buffer_x=0, buffer_y=0)
    
    # 2. Create Single Layer
    atomsk.create_slab(cif_path, base_path, duplicate=[dup[0], dup[1], 1], orthogonalize=True)
    
    # 3. Remove Charges
    if config.pot_type in ['tersoff', 'sw', 'rebo', 'airebo']:
        atomsk.remove_properties(base_path, base_path, ["q"])
    
    final_path = base_path
    dims = get_model_dimensions(base_path)
    lat_c = config.lat_c
    
    # 4. Stacking Logic with Minimization
    if stack_if_multi and config.layers and max(config.layers) > 1:
        n_layers = max(config.layers)
        stacked_path = build_dir / f"{config.mat}_{n_layers}.lmp"
        
        # If lat_c is missing, calculate it via minimization
        if lat_c is None:
            logger.info(f"lat_c not provided. Running minimization to find optimal interlayer distance.")
            lat_c = find_optimal_lat_c(base_path, config, build_dir, dims)
            logger.info(f"Found optimal lat_c: {lat_c:.4f}")

        sx, sy = calculate_layer_shifts(config.mat, dims)
        layers_info = []
        
        for l in range(n_layers):
            layers_info.append({
                'path': base_path,
                'shift': (sx * l, sy * l, lat_c * l),
                'id': l + 1
            })
            
        stack_sheets(layers_info, dims, stacked_path, total_natypes=10) # 10 is arbitrary buffer
        final_path = stacked_path

    return final_path, dims, (lat_c if lat_c else 6.0)