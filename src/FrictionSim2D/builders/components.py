"""Component builders for FrictionSim2D.

This module contains the logic to construct the physical components of the simulation:
the AFM tip, the substrate, and the 2D material sheet. It handles the calculation
of duplication factors based on requested dimensions and unit cell properties.

Function Organization:
1. Low-level utilities (run_lammps_commands, calculate_layer_shifts)
2. Slab creation (create_orthogonal_slab)
3. Amorphous material helpers (get_amorphous_path, make_amorphous)
4. Sheet stacking helpers (stack_multilayer_sheet)
5. Main builders (build_tip, build_substrate, build_sheet)
"""

import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lammps import lammps

from FrictionSim2D.core.config import GlobalSettings, SheetConfig, SubstrateConfig, TipConfig
from FrictionSim2D.core.potential_manager import PotentialManager
from FrictionSim2D.core.utils import (
    cifread, count_atomtypes, get_material_path, get_model_dimensions,
    renumber_atom_types, check_potential_cif_compatibility
)
from FrictionSim2D.interfaces.atomsk import AtomskWrapper

logger = logging.getLogger(__name__)

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
        mat_name: Material name (e.g., 'graphene', 'black_phosphorus').
        dims: Box dimensions dictionary with xlo, xhi, ylo, yhi keys.
        
    Returns:
        Tuple of (shift_x, shift_y) for layer stacking.
    """
    if mat_name.startswith('p-') or mat_name == 'black_phosphorus':
        sx = (dims['xhi'] - dims['xlo']) / 2
        sy = (dims['yhi'] - dims['ylo']) / 2
    else:
        sx = 0
        sy = (dims['yhi'] - dims['ylo']) / 3
    return sx, sy

def create_orthogonal_slab(
    cif_path: Path,
    output_path: Path,
    target_x: float,
    target_y: float,
    target_z: Optional[float] = None,
    atomsk: Optional[AtomskWrapper] = None
) -> Tuple[Path, Dict[str, float], List[int]]:
    """Creates an orthogonalized slab sized to meet target dimensions.
    
    Atomsk workflow:
    1. CIF → 2x2x1 supercell + orthogonalize → orthogonal "unit cell"
    2. Calculate duplication factors based on orthogonal cell dimensions
    3. Duplicate the orthogonal cell (not the original CIF)
    
    Args:
        cif_path: Path to the input CIF file.
        output_path: Path for the output LAMMPS data file.
        target_x: Target size in X direction (Angstroms).
        target_y: Target size in Y direction (Angstroms).
        target_z: Target size in Z direction (Angstroms), or None for single layer.
        atomsk: AtomskWrapper instance (created if not provided).
        
    Returns:
        Tuple of (output_path, box_dimensions, duplication_factors).
    """
    if atomsk is None:
        atomsk = AtomskWrapper()
    
    cif_path = Path(cif_path).absolute()
    
    # Step 1: Create orthogonalized 2x2x1 supercell - this is our "unit cell" for duplication
    ortho_cell = Path(tempfile.gettempdir()) / f"{cif_path.stem}_{uuid.uuid4().hex[:8]}_ortho.lmp"
    atomsk.create_slab(cif_path, ortho_cell, pre_duplicate=[2, 2, 1])
    
    # Step 2: Get dimensions of the orthogonal cell
    dims = get_model_dimensions(ortho_cell)
    
    # The orthogonal cell is 2x2x1, so divide by 2 to get per-unit dimensions
    unit_x = (dims['xhi'] - dims['xlo']) / 2.0
    unit_y = (dims['yhi'] - dims['ylo']) / 2.0
    unit_z = dims['zhi'] - dims['zlo']
    
    # Step 3: Calculate how many times to duplicate the orthogonal cell
    # Note: we're duplicating the 2x2x1 cell, so we need half as many duplications
    dup_x = max(1, round(target_x / unit_x / 2.0))
    dup_y = max(1, round(target_y / unit_y / 2.0))
    dup_z = 1 if target_z is None else max(1, round(target_z / unit_z))
    
    duplication = [dup_x, dup_y, dup_z]
    
    # Step 4: Duplicate the orthogonal cell (not the original CIF!)
    if dup_x == 1 and dup_y == 1 and dup_z == 1:
        # No further duplication needed, just copy the ortho cell
        shutil.copy(ortho_cell, output_path)
    else:
        # Duplicate the orthogonalized cell
        atomsk.duplicate(ortho_cell, output_path, dup_x, dup_y, dup_z)
    
    # Cleanup temp file
    ortho_cell.unlink()
    
    # Get final dimensions
    final_dims = get_model_dimensions(output_path)
    
    return output_path, final_dims, duplication

def get_amorphous_path(mat_name: str) -> Optional[Path]:
    """Gets the path to a pre-generated amorphous material file if it exists.
    
    Args:
        mat_name: Material name (e.g., 'Si').
        
    Returns:
        Path to the amorphous file if it exists, None otherwise.
    """
    from importlib import resources
    
    mat_dir = resources.files('FrictionSim2D.data.materials')
    amor_file = mat_dir.joinpath(f"amor_{mat_name}.lmp")
    
    if amor_file.is_file():
        return Path(str(amor_file))
    return None

def make_amorphous(
    mat_name: str,
    cif_path: Path,
    target_x: float,
    target_y: float,
    target_z: float,
    pot_path: Path,
    pot_type: str,
    output_dir: Path,
    settings: GlobalSettings
) -> Path:
    """Gets existing amorphous file if dimensions are sufficient, or generates a new one.
    
    This is the main entry point for obtaining amorphous structures. It checks for
    a cached file, validates dimensions, and generates via melt-quench if needed.
    
    The melt-quench procedure uses NPT ensemble (constant pressure) with three phases:
    1. Melt: Equilibrate at melt temperature
    2. Quench: Cool down from melt to target temperature at controlled rate
    3. Relax: Equilibrate at target temperature
    
    Args:
        mat_name: Material name (e.g., 'Si').
        cif_path: Path to crystalline CIF file (for generation if needed).
        target_x: Required X dimension (Angstroms).
        target_y: Required Y dimension (Angstroms).
        target_z: Required Z dimension (Angstroms).
        pot_path: Path to the potential file.
        pot_type: Type of potential (sw, tersoff, etc.).
        output_dir: Directory for temporary files during generation.
        settings: Global settings containing quench parameters.
        
    Returns:
        Path to an amorphous structure file with sufficient dimensions.
    """
    # Check for existing cached file
    amor_path = get_amorphous_path(mat_name)
    
    if amor_path is not None:
        # Check if existing file is large enough
        amor_dims = get_model_dimensions(amor_path)
        amor_x = amor_dims['xhi'] - amor_dims['xlo']
        amor_y = amor_dims['yhi'] - amor_dims['ylo']
        amor_z = amor_dims['zhi'] - amor_dims['zlo']
        
        if amor_x >= target_x and amor_y >= target_y and amor_z >= target_z:
            logger.info(f"Using existing amorphous file: {amor_path}")
            return amor_path
        else:
            logger.info(
                f"Existing amorphous file too small "
                f"({amor_x:.1f}x{amor_y:.1f}x{amor_z:.1f} vs {target_x:.1f}x{target_y:.1f}x{target_z:.1f}), "
                f"regenerating..."
            )
    
    # Generate new amorphous structure via melt-quench
    logger.info(f"Generating amorphous {mat_name} via melt-quench...")
    
    # Melt-quench parameters from settings
    quench = settings.quench
    
    # Use quench_slab_dims from settings, but ensure they're large enough for the simulation
    # For tip: target_x = 2*r (diameter), target_y = 2*r (diameter), target_z = r (radius/height)
    # For substrate: target_x, target_y, target_z = thickness
    slab_x = max(quench.quench_slab_dims[0], target_x)
    slab_y = max(quench.quench_slab_dims[1], target_y)
    slab_z = max(quench.quench_slab_dims[2], target_z)
    
    if (slab_x > quench.quench_slab_dims[0] or 
        slab_y > quench.quench_slab_dims[1] or 
        slab_z > quench.quench_slab_dims[2]):
        logger.info(
            f"Increasing slab dimensions from settings "
            f"({quench.quench_slab_dims[0]}x{quench.quench_slab_dims[1]}x{quench.quench_slab_dims[2]}) "
            f"to required size ({slab_x:.1f}x{slab_y:.1f}x{slab_z:.1f})"
        )
    
    # Create crystalline base block using settings dimensions
    atomsk = AtomskWrapper()
    base_block = output_dir / f"{mat_name}_crystal_block.lmp"
    create_orthogonal_slab(
        cif_path, base_block,
        target_x=slab_x, target_y=slab_y, target_z=slab_z,
        atomsk=atomsk
    )
    
    # Output file for amorphous structure (save in materials folder for reuse)
    from importlib import resources
    mat_dir = resources.files('FrictionSim2D.data.materials')
    # Convert to actual filesystem path
    if hasattr(mat_dir, '__fspath__'):
        materials_path = Path(mat_dir)
    else:
        materials_path = Path(str(mat_dir))
    
    # Ensure directory exists and is writable
    materials_path.mkdir(parents=True, exist_ok=True)
    
    amor_file = materials_path / f"amor_{mat_name}.lmp"
    
    # Get CIF data for elements
    cif_data = cifread(cif_path)
    elements = cif_data['elements']
    
    # Create a temporary ComponentConfig for PotentialManager
    from FrictionSim2D.core.config import ComponentConfig
    temp_config = ComponentConfig(
        mat=mat_name,
        pot_type=pot_type,
        pot_path=str(pot_path),
        cif_path=str(cif_path)
    )
    
    # Get LAMMPS commands from PotentialManager
    pm = PotentialManager(settings, use_langevin=False)
    pot_commands = pm.get_single_component_commands(temp_config, elements)
    
    # Calculate quench rate and number of steps
    # quench_rate is in K/ps, timestep is in ps
    quench_rate = float(quench.quench_rate) * 1e-12 / float(quench.timestep)
    quench_nsteps = max(int((quench.melt_temp - quench.quench_temp) * quench_rate), 1)
    
    # Create LAMMPS input script (run via mpiexec for parallelization)
    lmp_input_path = output_dir / f"lmp_input_amorphous_{mat_name}.in"
    
    with open(lmp_input_path, 'w', encoding='utf-8') as f:
        lines = [
            "# LAMMPS input script for melt-quench amorphization\n\n",
            "clear\n\n",
            "units           metal\n",
            "atom_style      atomic\n",
            "boundary        p p p\n",
            f"neighbor        {settings.simulation.neighbor_list} bin\n",
            f"{settings.simulation.neigh_modify_command}\n\n",
            f"read_data       {base_block}\n\n",
        ]
        
        # Add potential commands
        lines.extend(f"{cmd}\n" for cmd in pot_commands)
        lines.append("\n")
        
        # Minimization, timestep, and thermo
        lines.extend([
            f"min_style       {settings.simulation.min_style}\n",
            f"{settings.simulation.minimization_command}\n\n",
            f"timestep        {quench.timestep}\n",
            "thermo          100\n",
            "thermo_style    custom step temp pe ke etotal press vol\n\n",
            f"velocity        all create {quench.melt_temp} 1234579 rot yes dist gaussian\n",
            "run             0\n\n",

            "# Phase 1: Melt\n",
            f"fix             melt all npt temp {quench.melt_temp} {quench.melt_temp} $(100.0*dt) iso 0.0 0.0 $(1000.0*dt)\n",
            f"run             {quench.melt_steps}\n",
            "unfix           melt\n\n",

            "# Phase 2: Quench\n",
            f"fix             quench all npt temp {quench.melt_temp} {quench.quench_temp} $(100.0*dt) iso 0.0 0.0 $(1000.0*dt)\n",
            f"run             {quench_nsteps}\n",
            "unfix           quench\n\n",

            "# Phase 3: Relax\n",
            f"fix             relax all npt temp {quench.quench_temp} {quench.quench_temp} $(100.0*dt) iso 0.0 0.0 $(1000.0*dt)\n",
            f"run             {quench.equilibrate_steps}\n",
            "unfix           relax\n\n",
            f"write_data      {amor_file}\n",
        ])
        
        f.writelines(lines)
    
    # Run the simulation
    if quench.run_local:
        import subprocess
        logger.info(f"Running melt-quench with {quench.n_procs} processors...")
        try:
            subprocess.run(
                f"mpiexec -np {quench.n_procs} lmp -in {lmp_input_path}",
                shell=True, check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Melt-quench LAMMPS run failed: {e}")
            raise
    else:
        # Save input file for user to run manually
        local_lmp_input = output_dir / f"make_amorphous_{mat_name}.in"
        shutil.copy(lmp_input_path, local_lmp_input)
        logger.info(f"LAMMPS input file for amorphous structure generation saved to: {local_lmp_input}")
        logger.info("Please run the following command to generate the amorphous structure:")
        logger.info(f"mpiexec -np {quench.n_procs} lmp -in {local_lmp_input}")
        logger.info("After the simulation is complete, please rerun the program.")
        raise RuntimeError(
            f"Amorphous structure generation requires manual LAMMPS run. "
            f"See {local_lmp_input} for input script."
        )
    
    # Clean up temporary files
    if base_block.exists():
        base_block.unlink()
    if lmp_input_path.exists():
        lmp_input_path.unlink()
    
    logger.info(f"Created amorphous structure in materials folder: {amor_file}")
    return amor_file

def _create_base_slab(
    config,
    cif_path: Path,
    target_x: float,
    target_y: float,
    target_z: float,
    output_path: Path,
    build_dir: Path,
    settings: GlobalSettings,
    atomsk: AtomskWrapper
) -> None:
    """Creates a base slab (amorphous or crystalline) at output_path.
    
    This is a helper function used by build_tip and build_substrate to avoid
    code duplication for the slab generation logic.
    
    Args:
        config: TipConfig or SubstrateConfig with mat, amorph, pot_path, pot_type.
        cif_path: Resolved path to the CIF file.
        target_x: Target X dimension (Angstroms).
        target_y: Target Y dimension (Angstroms).
        target_z: Target Z dimension (Angstroms).
        output_path: Path for the output slab file.
        build_dir: Directory for temporary files.
        settings: Global simulation settings.
        atomsk: AtomskWrapper instance.
    """
    if config.amorph == 'a':
        pot_path = get_material_path(config.pot_path, config.pot_type)
        amor_path = make_amorphous(
            config.mat, cif_path,
            target_x=target_x, target_y=target_y, target_z=target_z,
            pot_path=pot_path, pot_type=config.pot_type,
            output_dir=build_dir, settings=settings
        )
        shutil.copy(amor_path, output_path)
    else:
        create_orthogonal_slab(
            cif_path, output_path,
            target_x=target_x, target_y=target_y, target_z=target_z,
            atomsk=atomsk
        )


def stack_multilayer_sheet(
    base_layer_path: Path,
    config: SheetConfig,
    output_path: Path,
    box_dims: Dict[str, float],
    n_layers: int,
    types_per_layer: int,
    pot_counts: Dict[str, int],
    lat_c: Optional[float] = None,
    settings: GlobalSettings = None
) -> Tuple[Path, float]:
    """Stacks multiple layers of a 2D material sheet with proper atom type renumbering.
    
    If lat_c is not provided, runs a LAMMPS minimization on a 2-layer system to 
    find the optimal interlayer distance. Then stacks all layers with proper atom 
    type renumbering for each layer.
    
    Uses read_data with z-shift for vertical separation and displace_atoms
    for in-plane stacking shifts (respects periodic boundaries).
    
    Args:
        base_layer_path: Path to the single-layer LAMMPS data file.
        config: Sheet configuration.
        output_path: Path for the stacked output file.
        box_dims: Box dimensions from the base layer.
        n_layers: Number of layers to stack.
        types_per_layer: Number of atom types per single layer (before renumbering).
        pot_counts: Dictionary mapping element to number of types in potential.
        lat_c: Interlayer distance. If None, calculated via minimization.
        settings: Global simulation settings (required if lat_c is None).
        
    Returns:
        Tuple of (output_path, lat_c).
    """
    if n_layers < 2:
        raise ValueError("stack_multilayer_sheet requires at least 2 layers")
    
    sx, sy = calculate_layer_shifts(config.mat, box_dims)
    
    # Calculate lat_c via minimization if not provided
    calculate_lat_c = lat_c is None
    
    if calculate_lat_c:
        logger.info("lat_c not provided. Running minimization to find optimal interlayer distance.")
        initial_guess = settings.geometry.lat_c_default
    else:
        initial_guess = lat_c
    
    # Prepare potential file
    pot_file = Path(tempfile.gettempdir()) / f"sheet_{n_layers}_{uuid.uuid4().hex[:8]}.in.settings"
    
    # For lat_c calculation we need only 2 layers, otherwise use all n_layers
    layers_for_setup = 2 if calculate_lat_c else n_layers
    
    pm = PotentialManager(settings)  # Langevin never applies to sheets
    pm.register_component("sheet", config, n_layers=layers_for_setup)
    pm.add_self_interaction("sheet")
    
    if pm.is_sheet_lj(config.pot_type):
        pm.add_interlayer_interaction("sheet")
    
    pm.write_file(pot_file)

    total_types = pm.get_total_types()
    max_z = initial_guess * (layers_for_setup - 1) + 20.0
    
    # Build LAMMPS commands
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_list([
            "clear", "units metal", "atom_style atomic", "boundary p p p",
            f"region box block {box_dims['xlo']} {box_dims['xhi']} {box_dims['ylo']} {box_dims['yhi']} -10 {max_z}",
            f"create_box {total_types} box",
        ])
        
        # Stack layers: use read_data shift for z, then displace_atoms for x,y
        for l in range(layers_for_setup):
            z_shift = initial_guess * l
            lmp.command(f"read_data {base_layer_path} add append shift 0 0 {z_shift} group layer_{l + 1}")
            if l > 0 and (sx != 0 or sy != 0):
                lmp.command(f"displace_atoms layer_{l + 1} move {sx * l} {sy * l} 0 units box")
        
        # Renumber atom types for each layer
        for t in range(1, types_per_layer + 1):
            lmp.command(f"group 2D_{t} type {t}")
        g = 0
        i = 0
        for count in pot_counts.values():
            for l in range(1, layers_for_setup + 1):
                for t in range(1, count + 1):
                    n = i + t
                    g += 1
                    lmp.commands_list([
                        f"group 2Dtype intersect 2D_{n} layer_{l}",
                        f"set group 2Dtype type {g}",
                        "group 2Dtype delete"
                    ])
            i += count
        
        # Include potential and run 0 (required for displace_atoms to work)
        lmp.commands_list([
            f"include {pot_file}",
            "run 0"
        ])

        # If calculating lat_c, run minimization and extract result
        if calculate_lat_c:
            lmp.commands_list([
                f"min_style {settings.simulation.min_style}",
                settings.simulation.minimization_command,
                f"timestep {settings.simulation.timestep}",
                "compute l1_com layer_1 com",
                "compute l2_com layer_2 com",
                "variable z1 equal c_l1_com[3]",
                "variable z2 equal c_l2_com[3]",
                "variable lat_c equal v_z2-v_z1",
                "run 0"
            ])
            
            lat_c = lmp.extract_variable("lat_c", None, 0)
            logger.info(f"Found optimal lat_c: {lat_c:.4f}")
            
            # Now we need to stack the remaining layers if n_layers > 2
            if n_layers > 2:
                # Close current LAMMPS instance and restart with full layer count
                lmp.close()
                
                # Recursively call with the calculated lat_c (will not calculate again)
                return stack_multilayer_sheet(
                    base_layer_path=base_layer_path,
                    config=config,
                    output_path=output_path,
                    box_dims=box_dims,
                    n_layers=n_layers,
                    types_per_layer=types_per_layer,
                    pot_counts=pot_counts,
                    lat_c=lat_c,
                    settings=settings
                )
        
        # Write the output
        lmp.command(f"write_data {output_path}")
        
        if not lat_c:
            lat_c = initial_guess  # Use initial guess if not calculated

    except Exception as e:
        logger.error(f"Failed to stack layers: {e}")
        if calculate_lat_c:
            lat_c = settings.geometry.lat_c_default
        raise
    finally:
        lmp.close()
        if pot_file.exists():
            pot_file.unlink()
    
    return output_path, lat_c


def build_tip(
    config: TipConfig, 
    atomsk: AtomskWrapper, 
    build_dir: Path, 
    settings: GlobalSettings
) -> Tuple[Path, float]:
    """Builds the AFM tip structure.
    
    For crystalline tips: creates orthogonal slab and carves sphere.
    For amorphous tips: uses pre-generated or melt-quenched amorphous block.
    
    Args:
        config: Tip configuration.
        atomsk: AtomskWrapper instance.
        build_dir: Directory for output files.
        settings: Global simulation settings.
        
    Returns:
        Tuple of (tip_path, actual_radius).
    """
    cif_path = get_material_path(config.cif_path, 'cif')
    
    base_lmp = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_base.lmp"
    final_lmp = build_dir / "tip.lmp"
    radius = config.r
    box_size = radius * 2.0

    # Generate slab: amorphous or crystalline
    _create_base_slab(
        config, cif_path,
        target_x=box_size, target_y=box_size, target_z=box_size,
        output_path=base_lmp, build_dir=build_dir,
        settings=settings, atomsk=atomsk
    )
    
    # Update radius based on generated slab dimensions
    if config.amorph != 'a':
        dim = get_model_dimensions(base_lmp)
        radius = min(dim['xhi'] - dim['xlo'], dim['yhi'] - dim['ylo']) / 2.0

    # Carve Sphere (LAMMPS)
    h = radius / settings.geometry.tip_reduction_factor
    commands = [
        "clear", "units metal", "atom_style atomic", "boundary p p p",
        f"read_data {base_lmp}",
        f"change_box all x final -{radius} {2*radius} y final -{radius} {2*radius}",
        f"displace_atoms  all move -{radius} -{radius} 0 units box",
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
    
    # Clean up intermediate file
    base_lmp.unlink()
    
    return final_lmp, radius

def build_substrate(
    config: SubstrateConfig,
    atomsk: AtomskWrapper,
    build_dir: Path,
    box_dims: dict,
    settings: Optional[GlobalSettings] = None
) -> Path:
    """Builds the substrate slab.
    
    For crystalline substrates: creates orthogonal slab.
    For amorphous substrates: uses pre-generated or melt-quenched amorphous block.
    
    Args:
        config: Substrate configuration.
        atomsk: AtomskWrapper instance.
        build_dir: Directory for output files.
        box_dims: Target box dimensions (from sheet).
        settings: Global simulation settings (required for amorphous generation).
        
    Returns:
        Path to the substrate file.
    """
    cif_path = get_material_path(config.cif_path, 'cif')

    base_lmp = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_base.lmp"
    final_lmp = build_dir / "sub.lmp"
    target_x = box_dims['xhi'] - box_dims['xlo']
    target_y = box_dims['yhi'] - box_dims['ylo']
    target_z = config.thickness
    
    # Generate slab: amorphous or crystalline
    _create_base_slab(
        config, cif_path,
        target_x=target_x, target_y=target_y, target_z=target_z,
        output_path=base_lmp, build_dir=build_dir,
        settings=settings, atomsk=atomsk
    )
    
    # Trim substrate to exact dimensions using LAMMPS
    commands = [
        "clear", "units metal", "atom_style atomic", "boundary p p p",
        f"read_data {base_lmp}",
        f"region box block {box_dims['xlo']} {box_dims['xhi']} {box_dims['ylo']} {box_dims['yhi']} {box_dims['zlo']} {target_z}",
        "group sub region box",
        "group delete_atoms subtract all sub",
        "delete_atoms group delete_atoms",
        f"change_box all x final {box_dims['xlo']} {box_dims['xhi']} y final {box_dims['ylo']} {box_dims['yhi']} z final -5 {target_z}",
        "reset_atoms id",
        f"write_data {final_lmp}"
    ]
    run_lammps_commands(commands)
    
    # Clean up base file
    base_lmp.unlink()
    
    return final_lmp

def build_monolayer(
    config: SheetConfig,
    atomsk: AtomskWrapper,
    build_dir: Path,
    settings: Optional[GlobalSettings] = None
) -> Tuple[Path, dict, float, dict, int]:
    """Builds a single-layer 2D material sheet (monolayer).
    
    This is the base layer that can be stacked for multi-layer systems.
    Building the monolayer once and reusing it for different layer counts
    avoids redundant lattice spacing calculations.
    
    Args:
        config: Sheet configuration.
        atomsk: AtomskWrapper instance.
        build_dir: Directory for output files.
        settings: Global simulation settings.
        
    Returns:
        Tuple of (output_path, box_dimensions, lat_c, pot_counts, total_pot_types).
        - output_path: Path to the monolayer .lmp file
        - box_dimensions: Dict with xlo, xhi, ylo, yhi, zlo, zhi
        - lat_c: Interlayer lattice constant
        - pot_counts: Dict of atom counts per potential type (for stacking)
        - total_pot_types: Total number of atom types in the potential
    """
    cif_path = get_material_path(config.cif_path, 'cif')
    pot_path = get_material_path(config.pot_path, config.pot_type)

    base_name = f"{config.mat}_1.lmp"
    base_path = build_dir / base_name
    
    # Get CIF data and potential type counts for renumbering
    cif_data = cifread(cif_path)
    pot_counts = count_atomtypes(pot_path, cif_data['elements'])
    total_pot_types = sum(pot_counts.values())
    
    # Determine atom type multiplier based on potential compatibility
    if config.pot_type in ['rebo', 'rebomos', 'airebo', 'meam', 'reaxff']:
        multiplier = 1
    else:
        multiplier = check_potential_cif_compatibility(cif_path, pot_path)
    
    # Step 1: Convert CIF to LMP (small unit cell)
    temp_unit_cell = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_unit.lmp"
    atomsk.convert(cif_path, temp_unit_cell)
    
    # Step 2: Renumber atom types if needed
    if any(v != 1 for v in pot_counts.values()) or multiplier != 1:
        renumber_atom_types(temp_unit_cell)
    
    # Step 3: Orthogonalize the unit cell
    ortho_cell = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_ortho.lmp"
    atomsk.orthogonalize(temp_unit_cell, ortho_cell)
    temp_unit_cell.unlink()
    
    # Step 4: Get dimensions and duplicate to match potential types if needed
    dims = get_model_dimensions(ortho_cell)
    
    if multiplier != 1:
        from ase import io as ase_io
        import numpy as np
        atoms = ase_io.read(str(ortho_cell), format="lammps-data")
        natoms = len(atoms)
        if total_pot_types % natoms == 0:
            for i in range(int(np.sqrt(total_pot_types / natoms)) + 1, 0, -1):
                if (total_pot_types / natoms) % i == 0:
                    a = int(i)
                    b = int(total_pot_types / natoms / i)
                    break
            dup_ortho = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_dup.lmp"
            atomsk.duplicate(ortho_cell, dup_ortho, a, b, 1)
            ortho_cell.unlink()
            ortho_cell = dup_ortho
            renumber_atom_types(ortho_cell, pot=list(pot_counts.keys()))
            dims = get_model_dimensions(ortho_cell)
    
    # Step 5: Calculate duplication factors for target dimensions
    unit_x = dims['xhi'] - dims['xlo']
    unit_y = dims['yhi'] - dims['ylo']
    dup_x = max(1, round(config.x / unit_x))
    dup_y = max(1, round(config.y / unit_y))
    
    # Step 6: Duplicate to target dimensions
    if dup_x == 1 and dup_y == 1:
        shutil.copy(ortho_cell, base_path)
    else:
        atomsk.duplicate(ortho_cell, base_path, dup_x, dup_y, 1)
    ortho_cell.unlink()
    
    # Get final dimensions
    dims = get_model_dimensions(base_path)
    
    # Remove Charges (for potentials that don't use them)
    if config.pot_type in ['tersoff', 'sw', 'rebo', 'airebo']:
        atomsk.charge2atom(base_path, base_path, ["q"])
    
    lat_c = config.lat_c
    
    return base_path, dims, lat_c, pot_counts, total_pot_types


def build_sheet(
    config: SheetConfig,
    atomsk: AtomskWrapper,
    build_dir: Path,
    stack_if_multi: bool = False,
    settings: Optional[GlobalSettings] = None,
    n_layers_override: Optional[int] = None
) -> Tuple[Path, dict, float]:
    """Builds the 2D material sheet.
    
    Args:
        config: Sheet configuration.
        atomsk: AtomskWrapper instance.
        build_dir: Directory for output files.
        stack_if_multi: If True, stack multiple layers into one file.
        settings: Global simulation settings (required for multi-layer lat_c calculation).
        n_layers_override: Override the number of layers (ignores config.layers).
        
    Returns:
        Tuple of (output_path, box_dimensions, lat_c).
    """
    # Build the monolayer first
    base_path, dims, lat_c, pot_counts, total_pot_types = build_monolayer(
        config, atomsk, build_dir, settings
    )
    
    final_path = base_path
    
    # Determine number of layers: override takes priority, then config
    n_layers = n_layers_override or (max(config.layers) if config.layers else 1)
    
    # Stacking Logic
    if stack_if_multi and n_layers > 1:
        stacked_path = build_dir / f"{config.mat}_{n_layers}.lmp"
        
        # Use unified stacking function (handles lat_c calculation if needed)
        stacked_path, lat_c = stack_multilayer_sheet(
            base_layer_path=base_path,
            config=config,
            output_path=stacked_path,
            box_dims=dims,
            n_layers=n_layers,
            types_per_layer=total_pot_types,
            pot_counts=pot_counts,
            lat_c=lat_c,
            settings=settings
        )
        
        # Remove the single-layer file since we only need the stacked version
        base_path.unlink()
        
        final_path = stacked_path

    return final_path, dims, lat_c
