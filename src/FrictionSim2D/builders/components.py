"""Component builders for FrictionSim2D.

This module contains functions to build the specific atomic structures
required for simulations: Tips, Substrates, and Sheets.
It orchestrates Atomsk for structure generation and LAMMPS for 
geometric manipulations (carving, stacking).
"""

import os
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from lammps import lammps
from importlib import resources

from FrictionSim2D.core.config import TipConfig, SubstrateConfig, SheetConfig, GlobalSettings
from FrictionSim2D.interfaces.atomsk import AtomskWrapper
from FrictionSim2D.core.utils import get_model_dimensions, cifread

logger = logging.getLogger(__name__)

def _get_material_path(mat_name: str, file_type: str = 'cif') -> Path:
    """Helper to find material files in the package data."""
    # Access the base materials directory
    mat_dir = resources.files('FrictionSim2D.data.materials')
    
    # Potential locations to check
    candidates = [
        mat_dir.joinpath(mat_name),                                # direct match
        mat_dir.joinpath(f"{mat_name}.{file_type}"),               # match with extension
        mat_dir.joinpath('cif', mat_name),                         # inside cif/ subdir
        mat_dir.joinpath('cif', f"{mat_name}.{file_type}"),        # inside cif/ subdir with extension
    ]

    for candidate in candidates:
        if candidate.exists():
            return Path(str(candidate))

    # If not found, return the original path string. 
    # The user might have provided an absolute path.
    return Path(mat_name)

def _get_potential_path(pot_name: str) -> Path:
    """Helper to find potential files in the package data recursively.
    
    This handles cases where potentials are in subfolders like 'sw/', 'tersoff/'.
    """
    # Access the base potentials directory
    pot_dir = resources.files('FrictionSim2D.data.potentials')
    
    # 1. Check direct path (if user provided 'sw/MoS2.sw')
    direct_path = pot_dir.joinpath(pot_name)
    if direct_path.is_file():
        return Path(str(direct_path))
        
    # 2. Check recursively in subdirectories
    target_name = Path(pot_name).name # Extract filename 'MoS2.sw' from 'sw/MoS2.sw' just in case
    
    base_path = Path(str(pot_dir))
    
    # Use rglob for recursive search (cleaner than os.walk)
    try:
        # Look for exact filename match in any subdirectory
        matches = list(base_path.rglob(target_name))
        if matches:
            return matches[0] # Return the first match
    except Exception as e:
        logger.warning(f"Failed recursive search for potential: {e}")

    # If not found in package, assume absolute path
    return Path(pot_name)

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
    """Calculates stacking shifts based on material type.
    
    Args:
        mat_name: Name of the material (e.g., 'MoS2', 'black_phosphorus').
        dims: Dimensions of the simulation box.
        
    Returns:
        Tuple[float, float]: (shift_x, shift_y)
    """
    if mat_name.startswith('p-') or mat_name == 'black_phosphorus':
        sx = (dims['xhi'] - dims['xlo']) / 2
        sy = (dims['yhi'] - dims['ylo']) / 2
    else:
        sx = 0
        sy = (dims['yhi'] - dims['ylo']) / 3
    return sx, sy

def stack_sheets(
    layers_info: List[Dict[str, Any]], 
    box_dims: Dict[str, float], 
    output_path: Path,
    total_natypes: int
) -> Path:
    """Generically stacks multiple layers into a single LAMMPS data file.
    
    This function is modular and supports heterostructures.
    
    Args:
        layers_info: List of dictionaries describing each layer.
            Format: [{'path': Path, 'shift': (x,y,z), 'id': int}, ...]
        box_dims: Dimensions for the simulation box.
        output_path: Path to save the final stacked file.
        total_natypes: Total number of atom types required in the box.
        
    Returns:
        Path: The output path.
    """
    # Calculate box height based on layers (approximate buffer)
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
        
        # Optional: Renumber types for this layer to make them unique?
        # If 'remap_types' is True or provided in layer info.
        if layer.get('remap_types', False):
             pass

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
        cif_path = _get_material_path(config.cif_path, 'cif')
    
    base_lmp = build_dir / f"{config.mat}_base.lmp"
    final_lmp = build_dir / "tip.lmp"
    radius = config.r

    if config.amorph == 'a':
        amor_filename = f"amor_{config.mat}.lmp"
        amor_path = _get_material_path(amor_filename, 'lmp')
        if amor_path.exists():
             shutil.copy(amor_path, base_lmp)
        else:
            logger.warning(f"Amorphous source {amor_filename} not found. Generating crystalline base.")
            atomsk.create_slab(cif_path, base_lmp, pre_duplicate=[2, 2, 1])
    else:
        unit_lmp = build_dir / "unit_cell.lmp"
        atomsk.create_slab(cif_path, unit_lmp, pre_duplicate=[2, 2, 1])
        dim = get_model_dimensions(unit_lmp)
        nx = int(np.ceil((2 * radius + 15) / (dim['xhi'] - dim['xlo'])))
        ny = int(np.ceil((2 * radius + 15) / (dim['yhi'] - dim['ylo'])))
        nz = int(np.ceil((radius + 5) / (dim['zhi'] - dim['zlo'])))
        atomsk.duplicate(unit_lmp, base_lmp, nx, ny, nz, center=True)
        
        dim = get_model_dimensions(base_lmp)
        max_possible_r = min((dim['xhi'] - dim['xlo'])/2, (dim['yhi'] - dim['ylo'])/2)
        if max_possible_r < radius:
            radius = max_possible_r

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
        cif_path = _get_material_path(config.cif_path, 'cif')
        
    base_lmp = build_dir / f"{config.mat}_sub_base.lmp"
    final_lmp = build_dir / "sub.lmp"
    
    atomsk.create_slab(cif_path, base_lmp)
    dim = get_model_dimensions(base_lmp)
    req_x = box_dims['xhi'] - box_dims['xlo']
    req_y = box_dims['yhi'] - box_dims['ylo']
    
    nx = int(round(req_x / (dim['xhi'] - dim['xlo'])))
    ny = int(round(req_y / (dim['yhi'] - dim['ylo'])))
    nz = int(round(config.thickness / (dim['zhi'] - dim['zlo'])))
    nx, ny, nz = max(1, nx), max(1, ny), max(1, nz)
    
    atomsk.duplicate(base_lmp, final_lmp, nx, ny, nz, center=True)
    
    commands = [
        "clear", "units metal", "atom_style atomic",
        f"read_data {final_lmp}",
        f"region box block 0 {req_x} 0 {req_y} -5 {config.thickness} units box",
        "group sub region box",
        "group delete_atoms subtract all sub",
        "delete_atoms group delete_atoms",
        f"change_box all x final 0 {req_x} y final 0 {req_y} z final -5 {config.thickness}",
        "reset_atoms id",
        f"write_data {final_lmp}"
    ]
    run_lammps_commands(commands)
    return final_lmp

def build_sheet(
    config: SheetConfig,
    atomsk: AtomskWrapper,
    build_dir: Path,
    stack_if_multi: bool = False
) -> Tuple[Path, dict, float]:
    """Builds the 2D material sheet."""
    cif_path = Path(config.cif_path)
    if not cif_path.exists():
        cif_path = _get_material_path(config.cif_path, 'cif')
        
    base_name = f"{config.mat}_1.lmp"
    base_path = build_dir / base_name
    
    # 1. Create Base Layer
    atomsk.create_slab(cif_path, base_path, pre_duplicate=[1, 1, 1])
    
    # 2. Duplicate to Size
    target_x = config.x if isinstance(config.x, (int, float)) else config.x[0]
    target_y = config.y if isinstance(config.y, (int, float)) else config.y[0]
    
    dim = get_model_dimensions(base_path)
    nx = int(round(target_x / (dim['xhi'] - dim['xlo'])))
    ny = int(round(target_y / (dim['yhi'] - dim['ylo'])))
    nx, ny = max(1, nx), max(1, ny)
    
    atomsk.duplicate(base_path, base_path, nx, ny, 1, center=True)
    
    dims = get_model_dimensions(base_path)
    lat_c = config.lat_c if config.lat_c is not None else 6.0

    # 3. Stacking Logic
    final_path = base_path
    if stack_if_multi and config.layers and max(config.layers) > 1:
        n_layers = max(config.layers)
        stacked_path = build_dir / f"{config.mat}_{n_layers}.lmp"
        
        layers_info = []
        sx, sy = calculate_layer_shifts(config.mat, dims)
        
        for l in range(n_layers):
            layers_info.append({
                'path': base_path,
                'shift': (sx * l, sy * l, lat_c * l),
                'id': l + 1
            })
            
        stack_sheets(layers_info, dims, stacked_path, total_natypes=10) 
        final_path = stacked_path

    return final_path, dims, lat_c