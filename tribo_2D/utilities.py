"""A collection of utility functions for file processing and calculations.

This module provides tools for reading, writing, and modifying simulation
files (e.g., CIF, LAMMPS data), parsing configuration files, and calculating
physical parameters like Lennard-Jones coefficients.
"""

import configparser
from fileinput import filename
import json
import os
import re
import shutil

import numpy as np
import yaml

from tribo_2D.Potentials import lj_params as lj


def read_yaml(filepath):
    """Reads a YAML file and returns its contents as a dictionary.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML contents.
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def cifread(cif_path):
    """Reads a CIF file and extracts crystal structure information.

    Args:
        cif_path (str): Path to the CIF file.

    Returns:
        dict: A dictionary containing extracted information such as lattice
        constants ('lat_a', 'lat_b', 'lat_c'), cell angles, chemical formula,
        and a list of elements.
    """
    filename = os.path.basename(cif_path).split('.')[0]

    with open(cif_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    elements = []
    cif_data = {}
    reading_elements = False
    header_skipped = False
    # Map CIF keys to dictionary keys for easier parsing
    keys = {
        '_cell_length_a': 'lat_a',
        '_cell_length_b': 'lat_b',
        '_cell_length_c': 'lat_c',
        '_chemical_formula_structural': 'formula',
        '_cell_angle_alpha': 'ang_a',
        '_cell_angle_beta': 'ang_b',
        '_cell_angle_gamma': 'ang_g'
    }

    for line in lines:
        # Parse lattice parameters and formula
        for key, var in keys.items():
            if line.startswith(key):
                value = line.split(maxsplit=1)[1].strip()
                cif_data[var] = float(
                    value) if 'formula' not in var else value

        # Start reading element symbols after this line
        if line.strip().startswith('_atom_site_type_symbol'):
            reading_elements = True
            continue

        # Skip any other header lines within the atom site loop
        if reading_elements and not header_skipped:
            if line.strip().startswith('_'):
                continue
            header_skipped = True

        # Once headers are skipped, read the element symbols
        if reading_elements and header_skipped:
            parts = line.strip().split()
            if parts:
                element = parts[0]
                elements.append(element)

    # Parse the chemical formula to get element counts
    elem_count = {}
    if cif_data.get('formula'):
        matches = re.findall(r'([A-Z][a-z]*)(\d*)', cif_data['formula'])
        for element, count in matches:
            elem_count[element] = int(count) if count else 1
        nelements = len(elements)

        cif_data.update({
            "elements": elements,
            "elem_comp": elem_count,
            "nelements": nelements,
            "filename": filename
        })

    return cif_data


def count_atomtypes(potential_filepath, elements):
    """Counts the number of different atom types per element in a potential file.

    Args:
        potential_filepath (str): Path to the LAMMPS potential file.
        elements (list): A list of element symbols to look for.

    Returns:
        dict: A dictionary where keys are element names and values are the
        count of unique atom types for that element (e.g., {'C': 2} for C1, C2).
    """
    elem_type = {el: 0 for el in elements}
    pattern = re.compile(r'([A-Za-z]+)(\d*)')

    with open(potential_filepath, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('#') or not stripped_line:
            continue

        parts = stripped_line.split()
        # Check the first few words, which typically contain element types
        for element in parts[:3]:
            match = pattern.match(element)
            if match:
                element_name = match.group(1)
                element_number = int(match.group(2)) if match.group(2) else 1
                if element_name in elem_type:
                    elem_type[element_name] = max(
                        elem_type[element_name], element_number)
    return elem_type


def get_model_dimensions(lmp_path):
    """Reads a LAMMPS data file and extracts the simulation box dimensions.

    Args:
        lmp_path (str): Path to the LAMMPS data file.

    Returns:
        dict: A dictionary containing the box dimensions with keys 'xlo',
        'xhi', 'ylo', 'yhi', 'zlo', 'zhi'.
    """
    xlo, xhi, ylo, yhi, zlo, zhi = [None] * 6
    with open(lmp_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "xlo xhi" in line:
                xlo, xhi = map(float, line.split()[0:2])
            elif "ylo yhi" in line:
                ylo, yhi = map(float, line.split()[0:2])
            elif "zlo zhi" in line:
                zlo, zhi = map(float, line.split()[0:2])
    dim = {
        "xlo": xlo,
        "xhi": xhi,
        "ylo": ylo,
        "yhi": yhi,
        "zlo": zlo,
        "zhi": zhi
    }
    return dim


def lj_params(atom_type_1, atom_type_2):
    """Calculates LJ parameters using Lorentz-Bertholt mixing rules.

    Pulls UFF parameters from the `lj` dictionary and applies
    mixing rules to determine the interaction parameters between two atom types.

    Args:
        atom_type_1 (str): The symbol of the first atom type (e.g., 'C').
        atom_type_2 (str): The symbol of the second atom type (e.g., 'H').

    Returns:
        tuple[float, float]: A tuple containing epsilon (potential well depth)
        and sigma (zero-potential distance).
    """
    e1 = lj.lj_params[atom_type_1][1]
    e2 = lj.lj_params[atom_type_2][1]
    s1 = lj.lj_params[atom_type_1][0]
    s2 = lj.lj_params[atom_type_2][0]
    epsilon = np.sqrt(e1 * e2)
    sigma = (s1 + s2) / 2
    return epsilon, sigma


def _remove_inline_comments(config):
    """Removes inline comments from a ConfigParser object.

    Args:
        config (configparser.ConfigParser): The ConfigParser object to process.

    Returns:
        configparser.ConfigParser: The object with inline comments removed.
    """
    for section in config.sections():
        for item in config.items(section):
            config.set(section, item[0], item[1].split("#")[0].strip())
    return config


def read_config(filepath):
    """Reads a configuration file and returns a dictionary with parsed values.

    Args:
        filepath (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing the parsed configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(filepath)

    config = _remove_inline_comments(config)

    params = {}
    for section in config.sections():
        params[section] = {}
        for key in config[section]:
            value = config.get(section, key)

            # Attempt to cast values to appropriate types
            if value.endswith(']'):
                params[section][key] = json.loads(value)  # Parse lists
            elif value.isdigit():
                params[section][key] = int(value)  # Parse integers
            # Parse floats (standard or scientific notation)
            elif '.' in value and value.replace('.', '', 1).isdigit():
                params[section][key] = float(value)
            elif value.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
                params[section][key] = float(value)
            else:
                params[section][key] = value
    return params


def atomic2molecular(filepath):
    """Converts a LAMMPS data file from atomic to molecular format in-place.

    This function modifies the "Atoms" section of a LAMMPS data file,
    changing the style from 'atomic' to 'molecular' and adding the required
    molecule ID and charge/dipole fields.

    Args:
        filepath (str): Path to the LAMMPS data file to be modified.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    atoms_section = False
    modified_lines = []

    for line in lines:
        line = line.strip()
        # Stop processing if another section (e.g., Velocities) is reached
        if atoms_section and not line.split():
            atoms_section = False

        if line.startswith("Velocities"):
            break

        # Replace "Atoms # atomic" with "Atoms # molecular"
        if line == "Atoms # atomic":
            modified_lines.append("Atoms # molecular")
            atoms_section = True
            continue

        # Modify atom data lines to add molecule ID and charge/dipole fields
        if atoms_section and line:
            parts = line.split()
            if len(parts) >= 4:  # Ensure it's a valid atom line
                atom_id = parts[0]
                atom_type = parts[1]
                x, y, z = parts[2:5]

                # New format: atom-ID molecule-ID atom-type x y z [charge] [dipole]
                new_line = f"{atom_id} 0 {atom_type} {x} {y} {z} 0 0 0"
                modified_lines.append(new_line)
                continue

        # Add unmodified lines to the list
        modified_lines.append(line)

    # Overwrite the original file with the modified content
    with open(filepath, 'w') as f:
        f.write("\n".join(modified_lines) + "\n")


def copy_file(path1, dest):
    """Copies a file from a source path to a destination directory.

    Args:
        path1 (str): Path to the source file.
        dest (str): Path to the destination directory.

    Returns:
        str: Path to the copied file in the destination directory.
    """
    os.makedirs(dest, exist_ok=True)
    filename = os.path.basename(path1)
    path2 = os.path.join(dest, filename)
    shutil.copy2(path1, path2)
    return path2


def renumber_atom_types(filename, pot=None):
    """Renumbers atom types in a LAMMPS data file to a sequential order.

    This function modifies a LAMMPS data file in-place to ensure atom types
    are numbered sequentially from 1. If a potential `pot` is provided, it
    renumbers the types to match the order of elements in that list.

    Args:
        filename (str): Path to the LAMMPS data file to be modified.
        pot (list[str], optional): A list of element symbols in the desired
            order for renumbering. Defaults to None.
    """
    # Open the LAMMPS data file and read all lines
    with open(filename, 'r') as f:
        lines = f.readlines()

    masses_section = False
    atoms_section = False

    atom_types = {}

    # Parse the "Masses" section to extract atom type IDs, masses, and names
    for i, line in enumerate(lines):
        if line.strip() == 'Masses':
            masses_section = True
            continue

        if masses_section:
            if 'Atoms' in line:
                break

            parts = line.split()
            if len(parts) < 2:
                continue

            atom_type_id = int(parts[0])
            mass = float(parts[1])
            if '#' in line:
                atom_type_name = line.split('#')[-1].strip()
                lines[i] = ''
            else:
                atom_type_name = f'Unknown_{atom_type_id}'
            atom_types[atom_type_id] = (atom_type_name, mass)
    # Prepare to update atom types in the "Atoms" section
    modified_lines = set()
    mod_lines = {}
    elem_pot = {}
    elem = {}
    t_add = len(atom_types) if pot is not None else 1
    t=1
    # For each atom type, update its ID and collect info
    for i in range(1, len(atom_types)+1):
        atoms_section = False
        if pot is not None:
            t = i
        for l, line in enumerate(lines):
            stripped_line = line.strip()

            if 'Atoms' in line:
                atoms_section = True
                continue

            if atoms_section and stripped_line and l not in modified_lines:
                parts = stripped_line.split()

                if parts[1] == str(i):
                    parts[1] = parts[0] = str(t)
                    lines[l] = ''
                    mod_lines[t] = '  '.join(parts) + '\n'
                    modified_lines.add(l)
                    elem[t] = atom_types[i]
                    t += t_add


    # Assign new atom type numbers according to the order in 'pot'
    if pot is not None:
        a = 1
        atom_lines = {}
        for i in pot:
            atoms_section = False
            for l in range(1, len(mod_lines)+1):
                stripped_line = mod_lines[l].strip()
                parts = stripped_line.split()
                if elem[int(parts[1])][0] == i:
                    elem_pot[a] = elem[int(parts[1])]
                    parts[1] = str(a)
                    atom_lines[a] = '  '.join(parts) + '\n'
                    a += 1
        elem = elem_pot
        mod_lines = atom_lines

    # Update the number of atom types in the header and rewrite the "Masses" section
    masses_section = False
    for i, line in enumerate(lines):
        if re.match(r'^\s*\d+\s+atom types\s*$', line.strip()):
            lines[i] = f"  {len(elem)}  atom types\n"
            continue

        if line.strip() == 'Masses':
            masses_section = True
            continue

        if masses_section:
            for l in range(1, len(elem)+1):
                lines[i] += f"{l} {elem[l][1]}  #{elem[l][0]}\n"
            break
    # Write the updated lines back to the file
    with open(filename, 'w') as f:
        f.writelines(lines)
    # Append the updated atom lines at the end of the file
    with open(filename, 'a') as f:
        for line in mod_lines.values():
            f.write(line)


def check_potential_cif_compatibility(cif_path, pot_path):
    """Checks if the number of atom types per element is compatible.

    Compares the number of atom types for each element defined in a CIF file
    versus a potential file. It returns a multiplier if the potential defines
    a consistent multiple of atom types compared to the CIF.

    Args:
        cif_path (str): Path to the CIF file.
        pot_path (str): Path to the potential file.

    Returns:
        float: The consistent ratio of atom types in the potential vs. the CIF.

    Raises:
        ValueError: If the ratio of atom types is not consistent across all
            elements.
    """
    cif_data = cifread(cif_path)
    potentials_count = count_atomtypes(pot_path, cif_data['elements'])

    # Calculate the ratio of atom types for each element
    multiples = {
        element: (potentials_count.get(element, 0) / cif_count
                  if cif_count > 0 and potentials_count.get(element, 0) != 1 else 1)
        for element, cif_count in cif_data['elem_comp'].items()
    }

    # Check that the ratio is consistent across all elements
    unique_multiples = set(m for m in multiples.values())
    if len(unique_multiples) > 1:
        raise ValueError(
            'Inconsistent atom type multiplier between potential and CIF file. '
            f'Ratios found: {unique_multiples}'
        )
    multiplier = unique_multiples.pop()

    return multiplier