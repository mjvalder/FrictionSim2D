from pathlib import Path
import os
import subprocess
import warnings

import numpy as np
from ase import io, data
from lammps import lammps

from tribo_2D import utilities


class ModelInit:
    """
    model_init(input_file, model='afm')

    Class for initializing and configuring atomic force microscopy (AFM) and related atomistic simulation models.

    This class reads simulation parameters from a configuration file and sets up the directory structure, material data,
    potential files, and LAMMPS input files required for various simulation models, including AFM tip-sample systems,
    sheet-vs-sheet contacts, and isolated components (tip, substrate, or sheet). It supports both crystalline and amorphous
    materials, handles multi-layer stacking, and automates the preparation of simulation-ready data.

    Attributes:
        model_type (str): Type of model to initialize ('afm', 'sheetvsheet', 'tip', 'substrate', or 'sheet').
        sheet_dir (dict): Directories for each sheet layer.
        data (dict): Material data for each system component.
        potentials (dict): Potential data for each system component.
        ngroups (dict): Number of atom groups per layer.
        elemgroup (dict): Element group definitions.
        group_def (dict): LAMMPS group definitions.
        dir (str or Path): Main working directory for the simulation.
        scan_angle (np.ndarray): Array of scan angles for the simulation.
        dump_load (list): List of load values for dumping simulation data.
        dump_slide (list): List of scan angles for dumping simulation data.
        dim (dict): Simulation box dimensions.
        lat_c (float): Interlayer lattice constant.
        tipx (float): Tip radius or dimension.

    Methods:
        setup_afm_model(): Set up the AFM tip-sample simulation environment.
        setup_sheetvsheet_model(): Set up a sheet-vs-sheet simulation environment.
        setup_sheet_part(): Set up a simulation with only the 2D sheet.
        setup_tip_part(): Set up a simulation with only the AFM tip.
        setup_substrate_part(): Set up a simulation with only the substrate.
        init_sheet(sheetvsheet=False): Initialize the 2D sheet structure and potentials.
        init_substrate(): Initialize the substrate structure and potentials.
        init_tip(): Initialize the tip structure and potentials.
        tip_build(): Generate and equilibrate the AFM tip model.
        sub_build(): Generate and equilibrate the substrate model.
        sheet_build(): Generate and prepare the 2D sheet model.
        make_amorphous(system, tempmelt): Generate an amorphous structure from a crystalline material.
        stacking(layer=2, sheetvsheet=False): Stack multiple layers of 2D materials.

    Raises:
        ValueError: If an unknown model type is specified.
    """

    def __init__(self, input_file, model='afm'):
        """
        Initialize the AFM simulation setup using an .INI configuration file.

        Args:
            input_file (str): Path to the input configuration file.
            model (str, optional): Type of model to initialize. Default is 'afm'.
            Options:
                - 'afm': Generates a tip, substrate, and sheet (AFM tip-sample system).
                - 'sheetvsheet': Generates a four-layer sheet (sheet-vs-sheet contact).
                - 'tip': Initializes only the AFM tip component.
                - 'substrate': Initializes only the substrate component.
                - 'sheet': Initializes only the 2D sheet component.

        This constructor reads the configuration file, initializes class attributes,
        sets up the directory structure, and calls the appropriate setup method
        based on the selected model type.

        Raises:
            ValueError: If an unknown model type is specified.
        """
        # Store input file and model type
        self.input_file = input_file
        self.model_type = model

        # --- Initialize all attributes used in the class ---
        self.sheet_dir = {}         # Used for storing layer directories
        self.data = {}              # Used for storing material data
        self.potentials = {}        # Used for storing potential data
        self.ngroups = {}           # Used for storing group counts
        self.elemgroup = {}         # Used for storing element groups
        self.group_def = {}         # Used for storing LAMMPS group definitions
        self.dir = None             # Used for main directory
        self.scan_angle = None      # Used for scan angles
        self.dump_load = None       # Used for dump load
        self.dump_slide = None      # Used for dump slide
        self.dim = None             # Used for dimensions
        self.lat_c = None
        self.tipx = None

        # --- Read and parse configuration ---
        self.params = utilities.read_config(self.input_file)

        # --- Map model names to setup methods ---
        setup_methods = {
            'afm': self.setup_afm_model,
            'sheetvsheet': self.setup_sheetvsheet_model,
            'tip': self.setup_tip_part,
            'substrate': self.setup_substrate_part,
            'sheet': self.setup_sheet_part,
        }

        try:
            setup_methods[model]()
        except KeyError as exc:
            raise ValueError(f"Unknown model type: {model}") from exc

    def setup_afm_model(self):
        """
        Set up the directory structure, scan parameters, and initialize components for the AFM (Atomic Force Microscopy) model.

        This method performs the following tasks:
            - Constructs a hierarchical working directory path based on simulation parameters.
            - Creates necessary subdirectories for storing visuals, results, build files, and potentials.
            - For each 2D material layer, creates dedicated directories for data and LAMMPS input files.
            - Sets up scan angles and determines intervals for dumping load and slide data.
            - Initializes the atomic structures for the sheet, substrate, and tip.
            - Defines the list of system components involved in the simulation.
            - Calculates and stores the total number of atom types for each layer, considering contributions from the 2D material, substrate, and tip.

        Raises:
            KeyError: If required keys are missing in the `self.params` or `self.data` dictionaries.
            OSError: If directory creation fails due to filesystem issues.

        Side Effects:
            - Modifies instance attributes: `self.dir`, `self.sheet_dir`, `self.scan_angle`, `self.dump_load`, `self.dump_slide`, `self.systems`, and `self.ngroups`.
            - Creates directories on the filesystem.
            - Calls initialization methods for sheet, substrate, and tip structures.
        """
        # --- Set up working directory structure ---
        self.dir = (
            f"afm/{self.params['2D']['mat']}/"
            f"{self.params['2D']['x']}x_{self.params['2D']['y']}y/"
            f"sub_{self.params['sub']['amorph']}{self.params['sub']['mat']}_"
            f"tip_{self.params['tip']['amorph']}{self.params['tip']['mat']}_"
            f"r{self.params['tip']['r']}/K{self.params['general']['temproom']}"
        )

        for subdir in ["visuals", "results", "build", "potentials"]:
            Path(self.dir, subdir).mkdir(parents=True, exist_ok=True)

        for layer in self.params['2D']['layers']:
            self.sheet_dir[layer] = Path(self.dir) / f"l_{layer}"
            for sub in ["data", "lammps"]:
                (self.sheet_dir[layer] /
                 sub).mkdir(parents=True, exist_ok=True)

        # --- Set scan angles and dump intervals ---
        self.scan_angle = np.arange(
            self.params['general']['scan_angle'][0],
            self.params['general']['scan_angle'][1] + 1,
            self.params['general']['scan_angle'][2]
        )

        self.dump_load = [
            self.params['general']['force'][i]
            for i in range(4, len(self.params['general']['force']), 5)
        ]

        self.dump_slide = [
            self.scan_angle[i]
            for i in range(4, len(self.scan_angle), 5)
        ]

        # --- Initialize structures ---
        self.init_sheet()
        self.init_substrate()
        self.init_tip()

        self.systems = ['sub', 'tip', '2D']

        for layer in self.params['2D']['layers']:
            # --- Count total number of atom types based on n of layers ---
            self.ngroups[layer] = (
                self.data['2D']['natype'] * layer +
                self.data['sub']['natype'] * 3 +
                self.data['tip']['natype'] * 3
            )

    def setup_sheetvsheet_model(self):
        """
        Sets up the directory structure and initializes the model for a sheet-vs-sheet simulation.

        This method creates a hierarchical directory structure based on the material and simulation parameters
        specified in `self.params`. The directories created include 'data', 'lammps', 'visuals', 'results',
        'build', and 'potentials', all nested under a path that encodes the material type, dimensions, and
        temperature. After setting up the directories, it calls `self.init_sheet` with the `sheetvsheet` flag
        set to True to initialize the sheet-vs-sheet simulation model.

        Side Effects:
            - Modifies `self.dir` to store the base directory path for the simulation.
            - Creates directories on the filesystem as needed.

        Raises:
            KeyError: If required keys are missing from `self.params`.

        Notes:
            - Assumes `self.params` is a dictionary with the necessary structure.
            - Assumes `Path` is imported from `pathlib`.
            - Assumes `init_sheet` is a method of the class.
        """
        self.dir = (
            f"sheetvsheet/{self.params['2D']['mat']}/"
            f"{self.params['2D']['x']}x_{self.params['2D']['y']}y/"
            f"K{self.params['general']['temproom']}"
        )
        for subdir in ["data", "lammps", "visuals", "results", "build", "potentials"]:
            Path(self.dir, subdir).mkdir(parents=True, exist_ok=True)

        # --- Call init_sheet with sheetvsheet flag ---
        self.init_sheet(sheetvsheet=True)

    def setup_sheet_part(self):
        """
        Sets up the directory structure and initializes the 2D sheet part of the model.

        This method constructs a directory path based on the material and dimensions specified
        in the `params` attribute under the '2D' key. It then creates the necessary subdirectories
        ('build' and 'potentials') within this path, ensuring that all parent directories exist.
        Finally, it calls `init_sheet()` to perform further initialization of the sheet.

        Assumes:
            - `self.params` is a dictionary containing a '2D' key with 'mat', 'x', and 'y' entries.
            - `init_sheet()` is a method of the class that handles additional sheet initialization.
        """
        self.dir = (
            f"sheet/{self.params['2D']['mat']}/"
            f"{self.params['2D']['x']}x_{self.params['2D']['y']}y"
        )
        for subdir in ["build", "potentials"]:
            Path(self.dir, subdir).mkdir(parents=True, exist_ok=True)
        self.init_sheet()

    def setup_tip_part(self):
        """
        Sets up the directory structure and initializes the tip component for the simulation.

        This method constructs a directory path for the tip based on the parameters specified
        in `self.params['tip']`, including whether the tip is amorphous, its material, and its radius.
        It then creates the necessary subdirectories ('build' and 'potentials') within this path,
        ensuring that all parent directories are created if they do not exist.
        Finally, it calls `self.init_tip()` to perform any additional tip initialization.

        Side Effects:
            - Modifies `self.dir` to store the constructed directory path.
            - Creates directories on the filesystem.
            - Calls the `init_tip` method for further initialization.
        """
        self.dir = (
            f"tip/{self.params['tip']['amorph']}{self.params['tip']['mat']}/"
            f"r{self.params['tip']['r']}"
        )
        for subdir in ["build", "potentials"]:
            Path(self.dir, subdir).mkdir(parents=True, exist_ok=True)

        self.init_tip()

    def setup_substrate_part(self):



        """
        Sets up the substrate part of the simulation environment.

        This method performs the following tasks:
            1. Constructs the directory path for the substrate based on the parameters specified in `self.params`.
            2. Creates necessary subdirectories ('build' and 'potentials') within the substrate directory.
            3. Defines the simulation box dimensions (`self.dim`) using the 2D parameters from `self.params`.
            4. Calls `self.init_substrate()` to initialize the substrate structure.

        Assumes that `self.params` is a dictionary containing the keys:
            - 'sub': with subkeys 'amorph' and 'mat'
            - '2D': with subkeys 'x' and 'y'

        Side Effects:
            - Modifies `self.dir` and `self.dim` attributes.
            - Creates directories on the filesystem.
            - Calls the `init_substrate` method.
        """
        self.dir = (
            f"substrate/{self.params['sub']['amorph']}{self.params['sub']['mat']}/"
            f"{self.params['2D']['x']}x_{self.params['2D']['y']}y"
        )

        for subdir in ["build", "potentials"]:
            Path(self.dir, subdir).mkdir(parents=True, exist_ok=True)

        self.dim = {
            'xlo': 0,
            'ylo': 0,
            'zlo': 0,
            'xhi': self.params['2D']['y'],
            'yhi': self.params['2D']['x'],
            'zhi': self.params['sub']['thickness']
        }

        self.init_substrate()

    def init_sheet(self, sheetvsheet=False):
        """
        Initialize the 2D material sheet structure and its associated potentials.

        This method performs the following steps:
            1. Reads material data from a CIF file and potential data from a potential file.
            2. Counts the number of atom types defined in the potential file.
            3. Copies the potential file to the working directory.
            4. Builds the 2D material structure and expands it into multiple layers as specified.
            5. For each additional layer (layer > 1), applies stacking if required.

        Args:
            sheetvsheet (bool, optional): If True, enables sheet-vs-sheet stacking behavior for layers. 
                                          Defaults to False.

        Side Effects:
            - Updates self.potentials and self.data dictionaries with material and potential information.
            - Modifies self.sheet_dir with paths for each layer.
            - Calls self.sheet_build() and self.stacking() for structure construction and stacking.

        Raises:
            KeyError: If required keys are missing in self.params['2D'].
            FileNotFoundError: If the specified CIF or potential files do not exist.

        """
        self.potentials['2D'] = {}
        # --- Read material and potential data ---
        self.data['2D'] = utilities.cifread(self.params['2D']['cif_path'])
        self.potentials['2D']['count'] = utilities.count_atomtypes(
            self.params['2D']['pot_path'])

        # --- Count atom types from the potential file ---
        self.data['2D']['natype'] = sum(
            self.potentials['2D']['count'].values())
        self.potentials['2D']['path'] = utilities.copy_file(
            self.params['2D']['pot_path'], Path(self.dir) / f"potentials")

        # --- Build 2D material and expand layers ---
        self.sheet_build()
        for layer in self.params['2D']['layers']:
            self.sheet_dir[layer] = Path(self.dir) / f"l_{layer}"
            if layer > 1:
                self.stacking(layer, sheetvsheet)

    def init_substrate(self):
        """
        Initializes the substrate data and potential parameters for the simulation.

        This method performs the following steps:
            1. Initializes the substrate potential and data dictionaries.
            2. Reads the substrate structure from a CIF file specified in the parameters.
            3. Counts the atom types in the substrate potential file.
            4. Calculates the total number of atom types in the substrate.
            5. Copies the substrate potential file to the working directory.
            6. Calls the method to build the substrate structure.

        Assumes that the following attributes are defined in the class:
            - self.potentials: Dictionary to store potential information.
            - self.data: Dictionary to store structural data.
            - self.params: Dictionary containing file paths and parameters.
            - self.dir: Working directory path.
            - self.sub_build: Method to build the substrate structure.

        Dependencies:
            - utilities.cifread: Reads CIF files and returns structure data.
            - utilities.count_atomtypes: Counts atom types in a potential file.
            - utilities.copy_file: Copies files to a specified directory.
        """
        self.potentials['sub'] = {}
        self.data['sub'] = utilities.cifread(self.params['sub']['cif_path'])
        self.potentials['sub']['count'] = utilities.count_atomtypes(
            self.params['sub']['pot_path'])

        self.data['sub']['natype'] = sum(
            self.potentials['sub']['count'].values())
        self.potentials['sub']['path'] = utilities.copy_file(
            self.params['sub']['pot_path'], Path(self.dir) / "potentials")
        self.sub_build()

    def init_tip(self):
        """
        Initializes the 'tip' component of the model by performing the following steps:

        1. Initializes the 'tip' entry in the potentials dictionary.
        2. Reads the atomic structure data for the tip from a CIF file specified in the parameters.
        3. Counts the atom types in the tip's potential file and stores the result.
        4. Calculates and stores the total number of atom types for the tip.
        5. Copies the tip's potential file to the model's potentials directory and stores the new path.
        6. Builds the tip structure and assigns it to the 'tipx' attribute.

        Assumes that the following attributes and methods are available:
            - self.potentials: Dictionary to store potential-related data.
            - self.data: Dictionary to store atomic structure data.
            - self.params: Dictionary containing file paths and parameters.
            - self.dir: Directory path for the model.
            - utilities.cifread: Function to read CIF files.
            - utilities.count_atomtypes: Function to count atom types in a potential file.
            - utilities.copy_file: Function to copy files.
            - self.tip_build: Method to build the tip structure.
        """
        self.potentials['tip'] = {}
        self.data['tip'] = utilities.cifread(self.params['tip']['cif_path'])
        self.potentials['tip']['count'] = utilities.count_atomtypes(
            self.params['tip']['pot_path'])

        self.data['tip']['natype'] = sum(
            self.potentials['tip']['count'].values())
        self.potentials['tip']['path'] = utilities.copy_file(
            self.params['tip']['pot_path'], Path(self.dir) / f"potentials")
        self.tipx = self.tip_build()

    def tip_build(self):
        """
        Generate an AFM tip model based on the specified material and properties.

        - Sets up the single-body potential for the tip.
        - Generates a slab for the tip, either amorphous or crystalline.
        - Uses LAMMPS to carve out a spherical tip region and equilibrate the structure.
        - Writes the resulting tip data file.

        Returns:
            float: The radius of the generated tip (x).
        """
        # Prepare file paths for the tip structure
        tip_lmp_filename = f"{self.params['tip']['mat']}.lmp"
        slab_path = os.path.join(os.path.dirname(
            __file__), "materials", tip_lmp_filename)
        x = self.params['tip']['r']  # Tip radius

        # Set up the single-body potential for the tip
        self.__single_body_potential(
            f"{self.dir}/build/tip.in.settings", 'tip')

        # Generate a slab for the tip: amorphous or crystalline
        if self.params['tip']['amorph'] == 'a':
            slab_path = self.make_amorphous('tip', 2500)
        else:
            self.slab_generator(
                slab_path, self.params['tip']['cif_path'], 2*x, 2*x, x)
            dim = utilities.get_model_dimensions(slab_path)
            x = dim['xhi']/2  # Update radius based on generated slab

        h = x / 2.25  # Height of the tip region (empirical factor)

        # Use LAMMPS to carve out the spherical tip and write the data file
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
        lmp.commands_list([
            "boundary p p p\n",
            "units metal\n",
            "atom_style      atomic\n",
            f"region          afm_tip sphere 0 0 {x} {x}  side in units box\n",
            "create_box      1 afm_tip\n",
            f"read_data       {slab_path} add append shift -{x} -{x} 0\n",
            f"region         box block -{x} {x} -{x} {x} -3 {h} units box\n",
            "region tip intersect 2 afm_tip box\n",
            "group           tip region tip\n",
            "group           box subtract all tip\n",
            "delete_atoms    group box\n",
            f"change_box all x final -{x} {x} y final -{x} {x} z final -3 {h+1} \n",
            "reset_atoms     id\n",
            f"write_data      {self.dir}/build/tip.lmp"
        ])

        return x  # Return the tip radius

    def sub_build(self):
        """
        Generate a substrate (slab) based on the specified material and properties.
        Handles both crystalline and amorphous structures, sets up LAMMPS regions,
        groups, and writes the resulting data file.

        This method:
            - Prepares the substrate slab file, either crystalline or amorphous.
            - Sets up the LAMMPS simulation box and region for the substrate.
            - Removes atoms outside the defined region.
            - Writes the processed substrate data file for further simulation.

        Side Effects:
            - Writes LAMMPS data files to disk.
            - Modifies the substrate structure according to simulation box dimensions.
        """

        # Prepare file paths for the substrate slab
        filename = f"{self.params['sub']['mat']}.lmp"
        slab_path = os.path.join(
            os.path.dirname(__file__),
            "materials", f"{filename}"
        )

        # Write the single-body potential settings for the substrate
        self.__single_body_potential(
            f"{self.dir}/build/sub.in.settings", 'sub')

        # Generate a slab for the substrate: amorphous or crystalline
        if self.params['sub']['amorph'] == 'a':
            # If amorphous, generate by melting and quenching
            slab_path = self.make_amorphous('sub', 2500)
        else:
            # If crystalline, generate using the CIF file and desired dimensions
            self.slab_generator(
                slab_path,
                self.params['sub']['cif_path'],
                self.dim['xhi'],
                self.dim['yhi'],
                self.params['sub']['thickness']
            )

        # Initialize and run LAMMPS to process and trim the substrate slab
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
        lmp.commands_list([
            "boundary p p p\n",
            "units metal\n",
            "atom_style      atomic\n",

            # Define the simulation box region for the substrate
            f"region box block {self.dim['xlo']} {self.dim['xhi']} {self.dim['ylo']} {self.dim['yhi']} {self.dim['zlo']} {self.params['sub']['thickness']}\n",
            f"create_box       {self.data['sub']['natype']} box\n\n",
            # Read the slab data and append to the box
            f"read_data {slab_path} add append\n",
            # Group atoms inside the substrate region
            "group sub region box\n",
            # Remove atoms outside the region
            "group box subtract all sub\n",
            "delete_atoms group box\n",
            # Adjust the box dimensions to match the simulation box
            f"change_box all x final {self.dim['xlo']} {self.dim['xhi']} y final {self.dim['ylo']} {self.dim['yhi']} z final {self.dim['zlo']}  {self.params['sub']['thickness']}\n\n",
            "reset_atoms     id\n",
            # Write the processed substrate data file
            f"write_data      {self.dir}/build/sub.lmp"
        ])

    def sheet_build(self):
        """
        Generate a 2D material sheet based on the specified material and properties.

        This method performs the following steps:
            1. Sets the sheet dimensions from the input parameters.
            2. Determines the output filename for the LAMMPS data file.
            3. Checks compatibility between the CIF and potential files, and determines the atom type multiplier.
            4. Generates the initial 2D sheet structure using Atomsk.
            5. Renumbers atom types if necessary to match the potential file.
            6. Orthogonalizes the cell using Atomsk.
            7. If the number of atom types in the potential file does not match the structure, duplicates atoms accordingly.
            8. Duplicates the sheet to match the specified simulation dimensions.
            9. Ensures the LAMMPS file type is correct using an external script.
            10. Updates the simulation box dimensions and centers the structure.
            11. Computes and stores the interlayer lattice constant (lat_c) by stacking two layers.

        Side Effects:
            - Writes and modifies files on disk.
            - Calls external programs (Atomsk, lmp_charge2atom.sh).
            - Updates self.dim and self.lat_c.

        Raises:
            subprocess.CalledProcessError: If any external command fails.
            FileNotFoundError: If required files are missing.
        """

        # --- Set sheet dimensions from input parameters ---
        x = self.params['2D']['x']
        y = self.params['2D']['y']

        # Output filename for the LAMMPS data file (single layer)
        filename = f"{self.dir}/build/{self.params['2D']['mat']}_1.lmp"

        # Check compatibility between CIF and potential files
        multiplier = utilities.check_potential_cif_compatibility(
            self.params['2D']['cif_path'], self.params['2D']['pot_path'])
        typecount = sum(self.potentials['2D']['count'].values())
        Path(filename).unlink(missing_ok=True)

        # Generate the 2D sheet using Atomsk (initial conversion from CIF)
        atomsk_command = f"echo n | atomsk {self.params['2D']['cif_path']} -ow {filename} -v 0"
        subprocess.run(atomsk_command, shell=True, check=True)

        # If atom types in potential file do not match structure, renumber atom types
        if any(v != 1 for v in self.potentials['2D']['count'].values()) or multiplier != 1:
            utilities.renumber_atom_types(filename)

        # Orthogonalize the cell and ensure LAMMPS format
        atomsk_command = f"atomsk {filename} -orthogonal-cell -ow lmp -v 0"
        subprocess.run(atomsk_command, shell=True, check=True)

        # Duplicate atoms to match the number of types in the potential file if necessary
        if multiplier != 1:
            atoms = io.read(filename, format="lammps-data")
            natoms = len(atoms)
            # Find factors a and b such that a*b = typecount/natoms
            if typecount % natoms == 0:
                for i in range(int(np.sqrt(typecount/natoms))+1, 0, -1):
                    if typecount/natoms % i == 0:
                        a = int(i)
                        b = int(typecount / natoms / i)
                        break
            atomsk_command = f"echo n | atomsk {filename} -duplicate {a} {b} 1 -ow lmp -v 0"
            subprocess.run(atomsk_command, shell=True, check=True)
            # Renumber atom types again to match the potential file
            utilities.renumber_atom_types(
                filename, pot=self.potentials['2D']['count'])

        # Duplicate the sheet to match the specified simulation dimensions
        dim = utilities.get_model_dimensions(filename)
        duplicate_a = round(x / dim['xhi'])
        duplicate_b = round(y / dim['yhi'])

        atomsk_command = f"atomsk {filename} -duplicate {duplicate_a} {duplicate_b} 1 -center com -ow lmp -v 0"
        subprocess.run(atomsk_command, shell=True, check=True)

        # Verify that the lmp filetype is correct using an external script
        charge2atom = f"lmp_charge2atom.sh {filename}"
        subprocess.run(charge2atom, shell=True, check=True)

        # Update simulation box dimensions and center the structure
        self.dim = utilities.get_model_dimensions(filename)
        self.center('2D', filename)

        # Compute and store the interlayer lattice constant by stacking two layers
        self.lat_c = self.stacking( )

    def make_amorphous(self, system, tempmelt):
        """
        Generate an amorphous structure from a crystalline material using LAMMPS.
        This is achieved by melting the crystalline structure and then quenching it
        to room temperature.

        Args:
            system (str): The name of the system (e.g., 'tip', 'sub').
            tempmelt (float): Melting temperature in Kelvin.

        Returns:
            str: Path to the generated amorphous structure file.

        Notes:
            - The method generates a slab using the provided CIF file, melts it at `tempmelt`, then quenches it to room temperature.
            - The amorphous structure is only generated if it does not already exist.
            - The method uses LAMMPS for the melt-quench process.
        """
        # Construct filenames for the slab and amorphous output
        filename = f"{self.params[system]['mat']}.lmp"
        slab_path = os.path.join(os.path.dirname(
            __file__), "materials", filename)
        am_filename = os.path.join(os.path.dirname(
            __file__), "materials", f"amor_{filename}")
        print(am_filename)
        # Only generate if the amorphous file does not already exist
        if not os.path.exists(am_filename):
            print("file not found")
            # Generate a large enough crystalline slab as the starting structure
            self.slab_generator(
                slab_path, self.params[system]['cif_path'], 200, 200, 50)

            # Calculate the number of steps for the quench (avoid negative)
            quench = max(
                (tempmelt - self.params['general']['temproom']) * 100, 0)
            pot_file = f"{self.dir}/build/{system}.in.settings"
            self.__single_body_potential(pot_file, system)

            # Initialize LAMMPS and run melt-quench protocol
            lmp = lammps()
            lmp.commands_list([
                "units metal\n",
                "atom_style      atomic\n\n",

                f"read_data {slab_path}\n\n",
                # ------------------Apply potentials-----------------------
                f"include {pot_file}\n\n",
                ##########################################################
                # -------------------Energy Minimization------------------#
                ##########################################################
                "min_style       cg\n",
                "minimize        1.0e-4 1.0e-8 100 1000\n",
                ##########################################################
                # -------------------Equilibrate System-------------------#
                ##########################################################
                "timestep        0.001\n",
                "thermo          100\n",
                "thermo_style    custom step temp pe ke etotal press\n",
                # Specify melting temperature and timestep
                f"velocity        all create {tempmelt} 1234579 rot yes dist gaussian\n",
                "run             0\n",
                # Equilibration at temperature
                f"fix             melt all nvt temp {tempmelt} {tempmelt} $(100.0*dt)\n",
                "run             5000\n",
                "unfix           melt\n",

                ##########################################################
                # ----------------------Quench System---------------------#
                ##########################################################
                f"fix             quench all nvt temp {tempmelt} {self.params['general']['temproom']} $(100.0*dt)\n",
                f"run             {quench}\n",
                "unfix           quench\n",
                f"write_data {am_filename}"
            ])

        # Return the path to the amorphous structure file
        return am_filename

    def stacking(self, layer=2, sheetvsheet=False):
        """
        Stack layers of 2D materials to create a multi-layer structure.

        Args:
            layer (int): The number of layers to stack (default is 2).
            sheetvsheet (bool): If True, creates two sets of two commensurate layers and stacks them accordingly.

        Returns:
            float: The computed interlayer lattice constant (lat_c) if calculated, otherwise the default value.

        Notes:
            - This method generates a multi-layer 2D structure by stacking single-layer data files.
            - It generates the LAMMPS settings file for the stacked sheet.
            - The interlayer distance (lat_c) is either taken from self.lat_c or computed from the center of mass of the layers.
            - For sheetvsheet=True, the stacking and displacement logic is different to create two bilayers.
            - The method returns the computed interlayer distance (lat_c).
        """
        # Generate the LAMMPS settings file for the stacked sheet
        self.__sheet_potential(
            f"{self.dir}/build/sheet_{layer}.in.settings",
            layer,
            sheetvsheet=sheetvsheet
        )

        filename = f"{self.dir}/build/{self.params['2D']['mat']}"
        lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none',  '-nocite'])

        # Set up the simulation box and read the first layer
        lmp.commands_list([
            "units           metal\n",
            "atom_style      atomic\n",
            "neighbor        0.3 bin\n",
            "boundary        p p p\n",
            "neigh_modify    every 1 delay 0 check yes #every 5\n\n",
            f"region box block {self.dim['xlo']} {self.dim['xhi']} {self.dim['ylo']} {self.dim['yhi']} -50 50\n",
            f"create_box       {self.data['2D']['natype']*layer} box\n\n",
            f"read_data       {filename}_1.lmp add append group layer_1 \n",
        ])

        # Determine the interlayer distance (lat_c)
        if self.lat_c is not None:
            lat_c = self.lat_c
        else:
            lat_c = 6  # Default value if not set
            warnings.warn(
                "lat_c is not set; using default value of 6. This may not be appropriate for all materials. Consider setting lat_c explicitly.")

        # Compute x_shift for layer displacement (used for stacking)
        x_shift = (self.dim['xhi'] - self.dim['xlo']) / 10

        # Stack additional layers and apply displacements
        if sheetvsheet:
            for l in range(1, layer):
                lmp.command(
                    f"read_data {filename}_1.lmp add append shift 0 0 {l*lat_c} group layer_{l+1}\n")
            # For sheetvsheet, displace layers 3 and 4 only
            lmp.commands_list([
                f"displace_atoms layer_3 move {x_shift} 0 0 units box\n",
                f"displace_atoms layer_4 move {x_shift} 0 0 units box\n",
            ])
        else:
            for l in range(1, layer):
                lmp.commands_list([
                    f"read_data {filename}_1.lmp add append shift 0 0 {l*lat_c} group layer_{l+1}\n",
                    f"displace_atoms layer_{l+1} move {x_shift*l} 0 0 units box\n",
                ])

        # Define atom type groups for each layer
        for t in range(1, self.data['2D']['natype'] + 1):
            lmp.command(
                f"group 2D_{t} type {t}\n"
            )

        # Renumber atom types for each layer and element type
        g = 0
        i = 0
        for count in self.potentials['2D']['count'].values():
            for l in range(1, layer+1):
                for t in range(1, count+1):
                    n = i + t
                    g += 1
                    lmp.commands_list([
                        f"group 2Dtype intersect 2D_{n} layer_{l}\n",
                        f"set group 2Dtype type {g}\n",
                        "group 2Dtype delete\n"
                    ])
            i += count

        # Energy minimization and equilibration
        lmp.commands_list([
            f"include         {self.dir}/build/sheet_{layer}.in.settings\n\n",
            "min_style       cg\n",
            "minimize        1.0e-4 1.0e-8 1000000 1000000\n\n",
            "timestep        0.001\n",
            "thermo          100\n\n",
            "#----------------- Apply Langevin thermostat -------------\n",
            "velocity        all create 300 492847948\n ",
            "fix             lang all langevin 300 300 $(100.0*dt) 2847563 zero yes\n\n ",
            "fix             nve_all all nve\n\n ",
            "timestep        0.001\n",
            "thermo          100\n\n ",
            "compute l_1 layer_1 com\n",
            "compute l_2 layer_2 com\n",
            "variable comz_1 equal c_l_1[3] \n\n ",
            "variable comz_2 equal c_l_2[3] \n\n ",
            "run 0\n",
            f"write_data  {filename}_{layer}.lmp"
        ])

        # Extract center of mass (COM) for each layer to compute interlayer distance
        com_l1 = lmp.extract_variable('comz_1', None, 0)
        com_l2 = lmp.extract_variable('comz_2', None, 0)
        # Compute lattice constant from the z-coordinates
        lat_c = com_l2 - com_l1

        return lat_c

    def slab_generator(self, filename, cif_path, x, y, z):
        """
        Generates a slab of a material from the provided CIF file and size parameters.

        Args:
            filename (str): Output filename for the generated slab (LAMMPS data file).
            cif_path (str): Path to the CIF file describing the crystal structure.
            x (float): Desired size in the x-direction (in Angstroms).
            y (float): Desired size in the y-direction (in Angstroms).
            z (float): Desired size in the z-direction (in Angstroms).

        Notes:
            - This method uses Atomsk to convert the CIF file to an orthogonal cell and then duplicates it to reach the target dimensions.
            - Temporary files ('a.cif') are created and deleted in the process.
            - The method reads the lattice constants from the CIF file to determine the required duplication factors.
            - Adds a 15 Å buffer to x and y before calculating duplication factors to ensure the slab is large enough.
            - The output file is overwritten if it already exists.
            - Requires Atomsk and the utilities.cifread function to be available.
            - Raises subprocess.CalledProcessError if Atomsk commands fail.
        """
        # Remove existing output and temporary files if they exist
        Path(filename).unlink(missing_ok=True)
        Path('a.cif').unlink(missing_ok=True)

        # Convert CIF to orthogonal cell using Atomsk and write to 'a.cif'
        atomsk_command = f"atomsk {cif_path} -duplicate 2 2 1 -orthogonal-cell a.cif -ow -v 0"
        subprocess.run(atomsk_command, shell=True, check=True)

        # Read lattice constants from the orthogonalized CIF
        cif = utilities.cifread("a.cif")
        # Calculate duplication factors to reach the desired slab size (+15 Å buffer for x and y)
        x2 = round((x+15)/cif["lat_a"])
        y2 = round((y+15)/cif["lat_b"])
        z2 = round(z/cif["lat_c"])

        # Duplicate the cell to reach the target dimensions and write to the output file
        atomsk_command2 = f"atomsk a.cif -duplicate {x2} {y2} {z2} {filename} -ow -v 0"
        subprocess.run(atomsk_command2, shell=True, check=True)

    def center(self, system, filename):
        """
        Centers objects in the simulation box using LAMMPS.

        Args:
            system (str): The name of the system (e.g., 'tip', 'sub', '2D').
            filename (str): Path to the input LAMMPS data file.

        This method:
            - Loads the atomic structure from a LAMMPS data file.
            - Sets atomic masses for each atom type.
            - Applies the appropriate pair_style and pair_coeff commands.
            - Computes the minimum x, y, z coordinates.
            - Calculates displacement variables to center the structure in the box.
            - Displaces all atoms so that the structure is centered.
            - Adjusts the z-dimension of the box to match self.dim.
            - Writes the centered structure back to the same file.

        Notes:
            - Assumes self.potentials[system]['count'] and self.potentials[system]['path'] are set.
            - Assumes self.params[system]['pot_type'] is set.
            - Assumes self.dim is set with 'zlo' and 'zhi' keys.
            - Uses LAMMPS Python interface.
        """
        # Initialize LAMMPS with no log/screen output
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])

        # Read the data file and set up simulation box
        lmp.commands_list([
            "units           metal\n",
            "atom_style      atomic\n",
            "neighbor        0.3 bin\n",
            "boundary        p p p",
            "neigh_modify    every 1 delay 0 check yes #every 5\n\n",
            f"read_data       {filename}\n"
        ])

        elem = []
        i = 0

        # Build list of element names for pair_coeff and set atomic masses
        for element, count in self.potentials[system]['count'].items():
            if not count or count == 1:
                elem.append(element)
                i += 1
            else:
                for t in range(1, count+1):
                    elem.append(element + str(t))
                    i += 1
            mass = data.atomic_masses[data.atomic_numbers[element]]
            lmp.command(f"mass {i} {mass}\n")

        # Set up pair_style and pair_coeff, compute min coordinates, and center structure
        lmp.commands_list([
            f"pair_style  {self.params[system]['pot_type']}\n",
            f"pair_coeff * * {self.potentials[system]['path']} {' '.join(elem)}\n",

            "compute zmin all reduce min z\n",
            "compute xmin all reduce min x\n",
            "compute ymin all reduce min y\n",

            "variable disp_z equal -c_zmin\n",
            "variable disp_x equal -(c_xmin+(xhi-xlo)/2.0)\n",
            "variable disp_y equal -(c_ymin+(yhi-ylo)/2.0)\n",

            "run 0\n\n",

            "displace_atoms all move v_disp_x v_disp_y v_disp_z units box\n\n",

            f"change_box all z final {self.dim['zlo']} {self.dim['zhi']}\n",
            "run 0\n",
            f"write_data  {filename}\n"
        ])
        # The structure in 'filename' is now centered in the simulation box.

    def __sheet_potential(self, filename, layer, sheetvsheet=False):
        """
        Writes the LAMMPS potential settings for a multi-layer 2D sheet system.

        Args:
            filename (str): The file to write LAMMPS settings to.
            layer (int): Number of 2D layers in the system.
            sheetvsheet (bool): If True, modifies interlayer interactions for sheet-vs-sheet models.

        This method:
            - Defines element groups and group definitions for each layer.
            - Writes atomic masses for each element and layer.
            - Defines LAMMPS groups for each layer.
            - Sets up a hybrid pair_style for all layers plus an additional LJ/cut for interlayer interactions.
            - Writes pair_coeff commands for each layer using the correct potential file and atom type mapping.
            - For sheetvsheet=True, sets specific interlayer LJ parameters to very low energy for certain layer pairs.
            - For other layer pairs, sets normal LJ parameters.

        Notes:
            - This method assumes that self.elemgroup, self.group_def, self.potentials, and self.params are properly initialized.
            - The index_pairs logic ensures all unique layer pairs are considered for interlayer LJ interactions.
            - For sheetvsheet, index_pairs_low_e specifies layer pairs with nearly zero interaction.
        """
        with open(filename, 'w') as f:
            # Reset element group and group definition dictionaries
            self.elemgroup['2D'] = {}
            self.group_def = {}
            potentials = {}
            atype = 1

            # Get sequential atom type mapping for 2D system
            arr = self.number_sequential_atoms('2D')

            # Define element groups for all layers
            atype = self.define_elemgroup('2D', arr, layer=layer, atype=atype)
            # Write atomic masses for all elements and layers
            self.set_masses('2D', f, layer=layer)

            # Define LAMMPS groups for each layer based on atom types
            for l in range(layer):
                layer_g = [
                    self.group_def[i][1]
                    for i in range(1, self.data['2D']['natype'] * layer + 1)
                    if "2D_l" + str(l + 1) in self.group_def[i][0]
                ]
                f.write(f"group layer_{l+1} type {' '.join(layer_g)}\n")

            # Set up hybrid pair_style: one for each layer plus an extra LJ/cut for interlayer
            f.write(
                f"pair_style hybrid {(self.params['2D']['pot_type'] + ' ') * layer} lj/cut 11.0\n")

            # Write pair_coeff for each layer: assign correct atom types and potential file
            for l in range(layer):
                potentials[l] = [
                    self.group_def[i][3] if "2D_l" +
                    str(l + 1) in self.group_def[i][0] else "NULL"
                    for i in range(1, self.data['2D']['natype'] * layer + 1)
                ]
                f.write(
                    f"pair_coeff * * {self.params['2D']['pot_type']} {l+1} {self.potentials['2D']['path']} {'  '.join(potentials[l])} # interlayer '2D' Layer {l+1}\n"
                )

            # Generate all unique layer index pairs for interlayer LJ
            index_pairs = [(i, j) for i in range(layer)
                           for j in range(i + 1, layer)]

            if sheetvsheet:
                # For sheet-vs-sheet, set very low LJ energy for specific pairs (decoupling)
                index_pairs_low_e = [(1, 3), (0, 3), (0, 2)]
                index_pairs_sheetvsheet = [
                    pair for pair in index_pairs if pair not in index_pairs_low_e]
                self.set_sheet_LJ_params(f, index_pairs_sheetvsheet)
                self.set_sheet_LJ_params(f, index_pairs_low_e, low=True)
            else:
                # For normal stacking, set LJ for all pairs
                self.set_sheet_LJ_params(f, index_pairs)

    def set_sheet_LJ_params(self, f, index_pairs, low=False):
        """
        Sets Lennard-Jones (LJ) pair coefficients for 2D sheet elements and writes them to a file.

        This method iterates over all combinations of element types in the 2D sheet, computes the LJ parameters
        (epsilon and sigma) for each pair using the `utilities.LJparams` function, and writes the corresponding
        `pair_coeff` commands to the provided file object `f`. The atom type labels are constructed from the
        `elemgroup` dictionary, and formatting is adjusted for single-element types.

        Parameters
        ----------
        f : file-like object
            An open file object to which the LAMMPS pair_coeff commands will be written.
        index_pairs : iterable of tuple of int
            Pairs of indices specifying which element groups to consider for LJ parameter assignment.
        low : bool, optional
            If True, sets the epsilon value to a very small number (1e-100) for the LJ potential,
            effectively turning off the interaction. If False (default), uses the computed epsilon.

        Notes
        -----
        - The function assumes the existence of `self.data['2D']['elem_comp']` and `self.elemgroup['2D']`
          with appropriate structure.
        - The function ensures that the atom type labels are ordered consistently and formatted correctly
          for LAMMPS input.
        - If the first and last elements of an element group are the same, the label is simplified.
        - The `utilities.LJparams` function is expected to return a tuple (epsilon, sigma) for the given element types.
        """
        # Iterate over all element combinations in the 2D sheet
        for t in self.data['2D']['elem_comp']:
            for s in self.data['2D']['elem_comp']:
                # Get LJ parameters (epsilon, sigma) for the element pair
                e, sigma = utilities.LJparams(s, t)
                # Loop over all specified layer index pairs
                for i, j in index_pairs:
                    # Build atom type labels for each layer and element
                    t1 = f"{self.elemgroup['2D'][i][t][0]}*{self.elemgroup['2D'][i][t][-1]}"
                    t2 = f"{self.elemgroup['2D'][j][s][0]}*{self.elemgroup['2D'][j][s][-1]}"
                    # If only one atom type, simplify the label
                    if self.elemgroup['2D'][i][t][0] == self.elemgroup['2D'][i][t][-1]:
                        t1 = f"{self.elemgroup['2D'][i][t][0]}"
                    if self.elemgroup['2D'][j][s][0] == self.elemgroup['2D'][j][s][-1]:
                        t2 = f"{self.elemgroup['2D'][j][s][0]}"
                    # Ensure consistent ordering for LAMMPS pair_coeff
                    if self.elemgroup['2D'][i][t][0] > self.elemgroup['2D'][j][s][0]:
                        t1, t2 = t2, t1
                    # Write the pair_coeff line, using a very small epsilon if low=True
                    if low:
                        f.write(
                            f"pair_coeff {t1} {t2} lj/cut 1e-100 {sigma} \n")
                    else:
                        f.write(f"pair_coeff {t1} {t2} lj/cut {e} {sigma} \n")

    def __single_body_potential(self, filename, system):
        """
        Writes the LAMMPS potential settings for a single-body system (e.g., tip or substrate).

        Args:
            filename (str): The file path to write the LAMMPS settings to.
            system (str): The system for which the settings are being written (e.g., 'tip', 'sub').

        This method:
            - Generates sequential atom type labels for the system.
            - Defines element groups and group definitions for the system.
            - Writes atomic masses for all elements in the system.
            - Writes the pair_style and pair_coeff commands for the system using the appropriate potential file.

        Notes:
            - Assumes self.potentials[system]['path'] and self.params[system]['pot_type'] are set.
            - Uses self.set_masses to write atomic masses.
            - Uses self.group_def to map atom types to element names.
        """
        # Generate sequential atom type labels for the system
        arr = self.number_sequential_atoms(system)
        # Define element groups and group definitions for the system
        _ = self.define_elemgroup(system, arr)

        with open(filename, 'w') as f:
            # Write atomic masses for all elements in the system
            self.set_masses(system, f)

            # Collect the element names for each atom type in order
            potentials = [self.group_def[i][2]
                          for i in range(1, len(arr[system]) + 1)]

            # Write the pair_style and pair_coeff commands for the system
            f.writelines([
                f"pair_style {self.params[system]['pot_type']}\n",
                f"pair_coeff * * {self.potentials[system]['path']} {' '.join(potentials)}\n"
            ])

    def define_elemgroup(self, system, arr, layer=None, atype=1):
        """
        Defines and updates element groups for a given system configuration.

        This method populates `self.group_def` and `self.elemgroup` with group definitions
        based on the provided atomic arrangement and system type. It supports both layered
        ('2D') and non-layered systems.

        Parameters
        ----------
        system : str
            The name of the system (e.g., '2D', '3D') for which element groups are defined.
        arr : dict
            A dictionary mapping system names to lists/arrays of atomic identifiers or properties.
        layer : int, optional
            The number of layers to consider (used only if `system` is '2D'). If None, layering is ignored.
        atype : int, optional
            The starting atom type index. Defaults to 1.

        Returns
        -------
        int
            The next available atom type index after all groups have been defined.

        Notes
        -----
        - Updates `self.group_def` with group definitions in the format:
          {atype: [group_name, atype, element, arr_value]}
        - Updates `self.elemgroup` with groupings by system, (layer), and element.
        - For '2D' systems with layers, groups are defined per layer and element.
        - For other systems, groups are defined per element.
        - Assumes `self.potentials[system]['count']` is a dictionary mapping element names to their counts.
        - Assumes `arr[system]` is indexable and has at least as many entries as the total count of elements.

        Raises
        ------
        KeyError
            If `system` is not found in `self.potentials` or `arr`.
        IndexError
            If `arr[system]` does not have enough entries for the required atoms.

        # TODO: Consider validating input arguments and handling potential errors more gracefully.
        # TODO: The variable naming (e.g., 'i', 'n', 't') could be improved for clarity.
        """

        i = 0

        for element, count in self.potentials[system]['count'].items():
            if layer is not None and system == '2D':
                for l in range(layer):
                    for t in range(1, count+1):
                        n = i + t
                        self.group_def.update(
                            {atype: [f"{system}_l{l+1}_t{n}", str(atype), str(element), arr[system][n-1]]})
                        self.elemgroup.setdefault(system, {}).setdefault(
                            l, {}).setdefault(element, []).append(atype)
                        atype += 1
                i += count
            else:
                for _ in range(1, count+1):
                    self.group_def.update(
                        {atype: [f"{system}_t{i}", str(atype), str(element), arr[system][i]]})
                    self.elemgroup.setdefault(system, {}).setdefault(
                        element, []).append(atype)
                    atype += 1
                    i += 1

        return atype

    def number_sequential_atoms(self, system):
        """
        Generates a sequential numbering of atoms for a given system based on element counts.

        For each element in the specified system, assigns a unique sequential identifier.
        If the element count is 1 or falsy, the element name is assigned directly.
        If the element count is greater than 1, each atom of that element is assigned a name
        with a numeric suffix (e.g., 'C1', 'C2', ...).

        Args:
            system (str): The name or key of the system to process.

        Returns:
            dict: A nested dictionary where the outer key is the system name, and the inner
                  dictionary maps sequential indices to atom names.

        Notes:
            - Assumes self.potentials[system]['count'] is a dictionary mapping element names
              to their respective counts.
            - The function returns a dictionary with the structure: {system: {index: atom_name, ...}}
            - The function does not check if `system` exists in `self.potentials` or if the
              'count' key exists; consider adding error handling for robustness.
        """
        arr = {}
        arr[system] = {}
        i = 0
        # Iterate over each element and its count in the system's potential
        for element, count in self.potentials[system]['count'].items():
            if not count or count == 1:
                # If only one atom of this element, assign the element name directly
                arr[system][i] = element
                i += 1
            else:
                # If multiple atoms, assign element name with numeric suffix
                for t in range(1, count + 1):
                    arr[system][i] = element + str(t)
                    i += 1
        return arr

    def set_masses(self, system, f, layer=None):
        """
        Writes atomic mass definitions to a file for a given system configuration.

        Parameters
        ----------
        system : str
            The name of the system (e.g., '2D' or other system types) whose atomic masses are to be set.
        f : file-like object
            An open file object with write permissions where the mass definitions will be written.
        layer : int, optional
            The specific layer index to consider for 2D systems. If None, all layers are considered.

        Notes
        -----
        - The method iterates over the element composition of the specified system and writes LAMMPS-style mass definitions.
        - For 2D systems, if a layer is specified, only that layer's masses are written; otherwise, all layers are considered.
        - For non-2D systems, all element groups are processed.
        - The method assumes the existence of the following attributes:
            - self.data: Dictionary containing system data, including 'elem_comp'.
            - self.elemgroup: Nested dictionary/list structure mapping systems and elements to group indices.
            - data.atomic_masses: Dictionary mapping atomic numbers to atomic masses.
            - data.atomic_numbers: Dictionary mapping element symbols to atomic numbers.
        - There is a duplicated line in the non-2D else branch that writes the same mass definition twice; this may be unintentional and should be reviewed.
        """
        # Iterate over all elements in the system's composition
        for m in self.data[system]['elem_comp']:
            mass = data.atomic_masses[data.atomic_numbers[m]]
            if system == '2D':
                if layer is not None:
                    # For a single layer, write mass for that layer only
                    if layer == 1 and len(self.elemgroup[system][0][m]) == 1:
                        f.write(
                            f"mass {self.elemgroup[system][0][m][0]} {mass} #{m} {system}\n")
                    else:
                        # For multiple layers, write mass for the range of atom types in all layers
                        f.write(
                            f"mass {self.elemgroup[system][0][m][0]}*{self.elemgroup[system][layer-1][m][-1]} {mass} #{m} {system}\n")
                else:
                    if len(self.elemgroup[system][0][m]) == 1:
                        f.write(
                            f"mass {self.elemgroup[system][0][m][0]} {mass} #{m} {system}\n")
                    else:
                        f.write(
                            f"mass {self.elemgroup[system][0][m][0]}*{self.elemgroup[system][len(self.elemgroup[system])-1][m][-1]} {mass} #{m} {system}\n")
            else:
                if len(self.elemgroup[system][m]) == 1:
                    # For a single atom type, write mass for that atom type
                    f.write(
                        f"mass {self.elemgroup[system][m][0]} {mass} #{m} {system}\n")
                else:
                    # For multiple atom types, write mass for the range of atom types
                    f.write(
                        f"mass {self.elemgroup[system][m][0]}*{self.elemgroup[system][m][-1]} {mass} #{m} {system}\n")
