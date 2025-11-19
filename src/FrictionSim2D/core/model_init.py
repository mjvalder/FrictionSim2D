"""Initializes simulation models for molecular dynamics simulations.

This module provides the `ModelInit` class, which serves as a base for setting
up various atomistic simulations. It handles the reading of configuration
files, creation of directory structures, and initialization of simulation
components like 2D materials, substrates, and tips.
"""

from pathlib import Path
import os
import subprocess
import warnings
import logging
import tempfile

import numpy as np
from ase import io, data
from lammps import lammps

from tribo_2D import utilities

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelInit:
    """Initializes the simulation model and environment.

    This class serves as a base for setting up various atomistic simulations.
    It reads parameters from a configuration file, sets up the necessary
    directory structure, and initializes the components of the simulation

    (e.g., sheet, substrate, tip) based on the specified model type.

    Attributes:
        model_type (str): The type of simulation model to set up.
        params (dict): A dictionary of parameters read from the input file.
        settings (dict): A dictionary of settings read from the YAML file.
        dir (pathlib.Path): The main working directory for the simulation.
        systems (list): A list of components in the simulation.
        data (dict): A dictionary containing material data for each component.
        potentials (dict): A dictionary containing potential data for each component.
        sheet_dir (dict): Directories for each sheet layer.
        ngroups (dict): Number of atom groups per layer.
        elemgroup (dict): Element group definitions.
        group_def (dict): LAMMPS group definitions.
        scan_angle (np.ndarray): Array of scan angles for the simulation.
        dim (dict): Simulation box dimensions.
        lat_c (float): Interlayer lattice constant.
        tipx (float): Tip radius or dimension.
    """

    def __init__(self, input_file, model='afm'):
        """Initializes the ModelInit class.

        Args:
            input_file (str): Path to the primary input configuration file.
            model (str, optional): The type of model to initialize.
                Supported models: 'afm', 'sheetvsheet', 'tip', 'substrate', 'sheet'.
                Defaults to 'afm'.

        Raises:
            ValueError: If an unsupported model type is provided.
        """
        self.input_file = input_file
        self.model_type = model

        # --- Initialize attributes ---
        try:
            self.params = utilities.read_config(self.input_file)
            settings_path = os.path.join(os.path.dirname(__file__), 'settings', 'settings.yaml')
            self.settings = utilities.read_yaml(settings_path)
        except FileNotFoundError as e:
            logging.exception("Configuration file not found: %s", e)
            raise
        except Exception as e:
            logging.exception("Error reading configuration files: %s", e)
            raise

        self.sheet_dir = {}
        self.data = {}
        self.potentials = {}
        self.ngroups = {}
        self.elemgroup = {}
        self.group_def = {}
        self.dir = None
        self.scan_angle = None
        self.dim = None
        self.lat_c = None
        self.tipx = None
        self.shift_x = None
        self.shift_y = None
        self.systems = []

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
        except Exception as e:
            logging.exception("An unexpected error occurred during model setup: %s", e)
            raise

    def _create_directories(self, subdirs):
        """Creates specified subdirectories within the main working directory.

        Args:
            subdirs (list): A list of subdirectory names to create.
        """
        for subdir in subdirs:
            try:
                Path(self.dir, subdir).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logging.exception("Failed to create directory %s: %s", Path(self.dir, subdir), e)
                raise

    def _initialize_component(self, system):
        """Initializes a system component (e.g., sheet, substrate, tip).

        This method reads material and potential data, counts atom types,
        copies potential files, and calls the appropriate build method.

        Args:
            system (str): The name of the system component ('2D', 'sub', or 'tip').
        """
        try:
            self.potentials[system] = {}
            self.data[system] = utilities.cifread(self.params[system]['cif_path'])
            self.potentials[system]['count'] = utilities.count_atomtypes(
                self.params[system]['pot_path'], self.data[system]['elements']
            )
            self.data[system]['natype'] = sum(self.potentials[system]['count'].values())
            self.potentials[system]['path'] = utilities.copy_file(
                self.params[system]['pot_path'], Path(self.dir) / "potentials"
            )                
            build_method_name = "sheet_build" if system == '2D' else f"{system.lower()}_build"
            build_method = getattr(self, build_method_name)
            build_method()

        except FileNotFoundError as e:
            logging.exception("A required file was not found during component initialization for '%s': %s", system, e)
            raise
        except (KeyError, AttributeError) as e:
            logging.exception("Configuration error for component '%s': %s", system, e)
            raise
        except Exception as e:
            logging.exception("An unexpected error occurred while initializing component '%s': %s", system, e)
            raise

    def setup_afm_model(self):
        """Sets up the environment for an AFM tip-sample simulation."""
        self.dir = Path(
            f"afm/{self.params['2D']['mat']}/"
            f"{self.params['2D']['x']}x_{self.params['2D']['y']}y/"
            f"sub_{self.params['sub']['amorph']}{self.params['sub']['mat']}_"
            f"tip_{self.params['tip']['amorph']}{self.params['tip']['mat']}_"
            f"r{self.params['tip']['r']}/K{self.params['general']['temp']}"
        )
        self._create_directories(["visuals", "results", "build", "potentials"])

        for layer in self.params['2D']['layers']:
            self.sheet_dir[layer] = self.dir / f"l_{layer}"
            (self.sheet_dir[layer] / "data").mkdir(parents=True, exist_ok=True)
            (self.sheet_dir[layer] / "lammps").mkdir(parents=True, exist_ok=True)

        scan_angle_params = self.params['general'].get('scan_angle')

        if scan_angle_params is not None:
            if isinstance(scan_angle_params, (list, tuple)):
                if len(scan_angle_params) == 1:
                    self.scan_angle = np.array([scan_angle_params[0]])
                elif len(scan_angle_params) == 4:
                    self.scan_angle = np.arange(
                        scan_angle_params[0],
                        scan_angle_params[1] + 1,
                        scan_angle_params[2]
                    )
                else:
                    raise ValueError(f"scan_angle must have 1 or 4 values, but got {len(scan_angle_params)}")
            else:
                # Handle case where it's a single number, not in a list
                self.scan_angle = np.array([scan_angle_params])
        else:
            self.scan_angle = None  # Default to None if not provided

        self.init_sheet()
        self._initialize_component('sub')
        self._initialize_component('tip')

        self.systems = ['sub', 'tip', '2D']
        multiplier = 3 if self.settings['thermostat']['type'] == 'langevin' else 1

        for layer in self.params['2D']['layers']:
            self.ngroups[layer] = (
                self.data['2D']['natype'] * layer +
                self.data['sub']['natype'] * multiplier +
                self.data['tip']['natype'] * multiplier
            )

    def setup_sheetvsheet_model(self):
        """Sets up the environment for a sheet-vs-sheet simulation."""
        self.dir = Path(
            f"sheetvsheet/{self.params['2D']['mat']}/"
            f"{self.params['2D']['x']}x_{self.params['2D']['y']}y/"
            f"K{self.params['general']['temp']}"
        )
        self._create_directories(["lammps", "visuals", "results", "build", "potentials"])
        scan_angle_params = self.params['general'].get('scan_angle')
        if scan_angle_params is not None:
            if isinstance(scan_angle_params, (list, tuple)):
                if len(scan_angle_params) == 1:
                    self.scan_angle = scan_angle_params[0]
                elif len(scan_angle_params) == 4:
                    self.scan_angle = np.arange(
                        scan_angle_params[0],
                        scan_angle_params[1] + 1,
                        scan_angle_params[2]
                    )
                else:
                    raise ValueError(f"scan_angle must have 1 or 4 values, but got {len(scan_angle_params)}")
            else:
                # Handle case where it's a single number, not in a list
                self.scan_angle = scan_angle_params
        else:
            self.scan_angle = None  # Default to None if not provided


        self.init_sheet(sheetvsheet=True)
        self.ngroups[4] = self.data['2D']['natype'] * 4

    def setup_sheet_part(self):
        """Sets up the environment for a single 2D sheet."""
        self.dir = Path(
            f"sheet/{self.params['2D']['mat']}/"
            f"{self.params['2D']['x']}x_{self.params['2D']['y']}y"
        )
        self._create_directories(["build", "potentials"])
        self.init_sheet()

    def setup_tip_part(self):
        """Sets up the environment for a single AFM tip."""
        self.dir = Path(
            f"tip/{self.params['tip']['amorph']}{self.params['tip']['mat']}/"
            f"r{self.params['tip']['r']}"
        )
        self._create_directories(["build", "potentials"])
        self._initialize_component('tip')

    def setup_substrate_part(self):
        """Sets up the environment for a single substrate."""
        self.dir = Path(
            f"substrate/{self.params['sub']['amorph']}{self.params['sub']['mat']}/"
            f"{self.params['2D']['x']}x_{self.params['2D']['y']}y"
        )
        self._create_directories(["build", "potentials"])

        self.dim = {
            'xlo': 0, 'ylo': 0, 'zlo': 0,
            'xhi': self.params['2D']['y'],
            'yhi': self.params['2D']['x'],
            'zhi': self.params['sub']['thickness']
        }
        self._initialize_component('sub')

    def init_sheet(self, sheetvsheet=False):
        """Initializes the 2D material sheet structure and potentials.

        Args:
            sheetvsheet (bool, optional): If True, enables sheet-vs-sheet
                stacking behavior. Defaults to False.
        """
        self._initialize_component('2D')
        num_layers = [4] if sheetvsheet else self.params['2D']['layers']
        
        for layer in num_layers:
            self.sheet_dir[layer] = self.dir / f"l_{layer}"
            if layer > 1:
                self.stacking(layer, sheetvsheet)

    def tip_build(self):
        """Builds the AFM tip structure for the simulation.

        This method constructs the tip geometry based on the parameters specified
        in `self.params`. It handles both crystalline and amorphous tip
        structures. For crystalline tips, it creates a spherical geometry. For
        amorphous tips, it generates an amorphous structure using a separate
        LAMMPS simulation and then carves a sphere from it.

        Side Effects:
            - Creates and modifies files in the `build` and `potentials` subdirectories.
            - Runs an external LAMMPS process.
            - Modifies `self.tipx` with the final tip dimension.
        """
        # Prepare file paths for the tip structure

        tip_lmp_filename = f"{self.params['tip']['mat']}.lmp"
        slab_path = os.path.join(os.path.dirname(
            __file__), "materials", tip_lmp_filename)
        x = self.params['tip']['r']  # Tip radius

        # Set up the single-body potential for the tip
        self._single_body_potential(
            f"{self.dir}/build/tip.in.settings", 'tip')

        # Generate a slab for the tip: amorphous or crystalline
        if self.params['tip']['amorph'] == 'a':
            slab_path = self.make_amorphous('tip')
        else:
            self.slab_generator(
                slab_path, self.params['tip']['cif_path'], 2*x, 2*x, x)
            dim = utilities.get_model_dimensions(slab_path)
            x = dim['xhi']/2  # Update radius based on generated slab

        # Height of the tip region (empirical factor)
        h = x / self.settings['geometry']['tip_reduction_factor']
        # Use LAMMPS to carve out the spherical tip and write the data file
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
        try:
            lmp.commands_list(self.init())
            lmp.commands_list([
                f"read_data       {slab_path}",
                f"change_box all x final -{x} {2*x} y final -{x} {2*x}",
                f"displace_atoms  all move -{x} -{x} 0 units box",
                f"region          afm_tip sphere 0 0 {x} {x}  side in units box",
                f"region         box block -{x} {x} -{x} {x} -3 {h} units box",
                "region tip intersect 2 afm_tip box",
                "group           tip region tip",
                "group           box subtract all tip",
                "delete_atoms    group box",
                f"change_box all x final -{x} {x} y final -{x} {x} z final -3 {h+1}",
                "reset_atoms     id",
                f"write_data      {self.dir}/build/tip.lmp"
            ])
        except Exception as e:
            logging.exception(f"LAMMPS simulation failed during tip creation: {e}")
            raise
        finally:
            lmp.close()

        self.tipx = x

    def sub_build(self):
        """Builds the substrate structure for the simulation.

        This method generates a substrate slab, which can be either crystalline
        or amorphous. It sets up the LAMMPS simulation box, creates the
        substrate region, and writes the resulting data file.

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
        self._single_body_potential(
            f"{self.dir}/build/sub.in.settings", 'sub')

        # Generate a slab for the substrate: amorphous or crystalline
        if self.params['sub']['amorph'] == 'a':
            slab_path = self.make_amorphous('sub')
        else:
            self.slab_generator(
                slab_path,
                self.params['sub']['cif_path'],
                self.dim['xhi'],
                self.dim['yhi'],
                self.params['sub']['thickness']
            )

        # Initialize and run LAMMPS to process and trim the substrate slab
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
        try:
            lmp.commands_list(self.init())
            lmp.commands_list([
                f"read_data {slab_path}",
                f"region box block {self.dim['xlo']} {self.dim['xhi']} {self.dim['ylo']} {self.dim['yhi']} {self.dim['zlo']} {self.params['sub']['thickness']}",
                "group sub region box",
                "group box subtract all sub",
                "delete_atoms group box",
                f"change_box all x final {self.dim['xlo']} {self.dim['xhi']} y final {self.dim['ylo']} {self.dim['yhi']} z final -5 {self.params['sub']['thickness']}",
                "reset_atoms     id",
                f"write_data      {self.dir}/build/sub.lmp"
            ])
        except Exception as e:
            logging.exception(f"LAMMPS simulation failed during substrate creation: {e}")
            raise
        finally:
            lmp.close()

    def sheet_build(self):
        """Builds the 2D material sheet structure for the simulation.

        This method takes a unit cell of the 2D material, replicates it to the
        desired simulation cell size, and saves the resulting structure as a
        LAMMPS data file. It also calculates the interlayer lattice constant
        (`lat_c`) if it is not already defined.

        Side Effects:
            - Writes and modifies files on disk.
            - Calls external programs (Atomsk, lmp_charge2atom.sh).
            - Updates self.dim, self.lat_c, self.shift_x, and self.shift_y.

        Raises:
            subprocess.CalledProcessError: If any external command fails.
        """
        try:
            x = self.params['2D']['x']
            y = self.params['2D']['y']
            filename = f"{self.dir}/build/{self.params['2D']['mat']}_1.lmp"

            # Determine atom type multiplier based on potential compatibility
            multiplier = 1 if self.params['2D']['pot_type'] in ['rebo', 'rebomos', 'airebo', 'meam', 'reaxff'] \
                else utilities.check_potential_cif_compatibility(
                    self.params['2D']['cif_path'], self.params['2D']['pot_path'])
            typecount = sum(self.potentials['2D']['count'].values())
            Path(filename).unlink(missing_ok=True)

            # Generate, orthogonalize, and format the 2D sheet using Atomsk
            subprocess.run(f"echo n | atomsk {self.params['2D']['cif_path']} -ow {filename} -v 0", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if any(v != 1 for v in self.potentials['2D']['count'].values()) or multiplier != 1:
                utilities.renumber_atom_types(filename)

            subprocess.run(f"atomsk {filename} -orthogonal-cell -ow lmp -v 0", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Duplicate atoms to match potential file if necessary
            dim = utilities.get_model_dimensions(filename)

            # Duplicate atoms to match the number of types in the potential file if necessary
            if multiplier != 1:
                atoms = io.read(filename, format="lammps-data")
                natoms = len(atoms)
                if typecount % natoms == 0:
                    for i in range(int(np.sqrt(typecount/natoms))+1, 0, -1):
                        if typecount/natoms % i == 0:
                            a = int(i)
                            b = int(typecount / natoms / i)
                            break
                    subprocess.run(f"echo n | atomsk {filename} -duplicate {a} {b} 1 -ow lmp -v 0", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    utilities.renumber_atom_types(filename, pot=self.potentials['2D']['count'])

            # Set shift values for stacking based on material type
            dim = utilities.get_model_dimensions(filename)
            material_name = self.params['2D']['mat']
            if material_name.startswith('p-') or material_name == 'black_phosphorus':
                self.shift_x = dim['xhi']/2
                self.shift_y = dim['yhi']/2
            else:
                self.shift_x = 0
                self.shift_y = dim['yhi']/3

            # Duplicate the sheet to match simulation dimensions
            duplicate_a = round(x / dim['xhi'])
            duplicate_b = round(y / dim['yhi'])
            subprocess.run(f"atomsk {filename} -duplicate {duplicate_a} {duplicate_b} 1 -center com -ow lmp -v 0", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Finalize file format and update dimensions
            subprocess.run(f"lmp_charge2atom.sh {filename}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.dim = utilities.get_model_dimensions(filename)
            self.center('2D', filename)

            # Compute interlayer lattice constant
            if self.params['2D']['lat_c'] not in [None, '', ' ']:
                self.lat_c = self.params['2D']['lat_c']
            else:
                self.lat_c = self.stacking()
        except subprocess.CalledProcessError as e:
            logging.exception(f"An external command failed during sheet building: {e}")
            raise
        except (FileNotFoundError, KeyError) as e:
            logging.exception(f"File or configuration key not found during sheet building: {e}")
            raise
        except Exception as e:
            logging.exception(f"An unexpected error occurred during sheet building: {e}")
            raise

    def make_amorphous(self, system):
        """Generates an amorphous structure via a melt-quench procedure.

        This method simulates melting a crystalline material and then rapidly
        cooling it to create an amorphous solid, controlled by parameters in
        `self.settings`. The structure is only generated if it does not
        already exist.

        Args:
            system (str): The system to be made amorphous ('tip' or 'sub').

        Returns:
            str: Path to the generated amorphous structure file.
        """
        filename = f"{self.params[system]['mat']}.lmp"
        slab_path = os.path.join(os.path.dirname(__file__), "materials", filename)
        am_filename = os.path.join(os.path.dirname(__file__), "materials", f"amor_{filename}")

        if not os.path.exists(am_filename):
            try:
                # Generate a crystalline slab as the starting structure
                self.slab_generator(
                    slab_path,
                    self.params[system]['cif_path'],
                    self.settings['quench']['quench_slab_dims'][0],
                    self.settings['quench']['quench_slab_dims'][1],
                    self.settings['quench']['quench_slab_dims'][2]
                )

                # Prepare for melt-quench simulation
                quench_rate = float(self.settings['quench']['quench_rate']) * 1e-12 / float(self.settings['quench']['timestep'])    
                quench_nsteps = max((self.settings['quench']['quench_melt_temp'] - self.params['general']['temp']) * quench_rate, 0)
                pot_file = f"{self.dir}/build/{system}.in.settings"
                self._single_body_potential(pot_file, system)

                # Initialize LAMMPS and run melt-quench-relax protocol
                lmp_input = os.path.join(tempfile.gettempdir(), f"lmp_input_{system}.in")
                with open(lmp_input, 'w', encoding="utf-8") as f:
                    f.writelines(self.init(neigh=True))
                    f.writelines([
                        f"read_data      {slab_path}\n",
                        f"include        {pot_file}\n",
                        f"min_style      {self.settings['simulation']['min_style']}\n",
                        self.settings['simulation']['minimization_command'],
                        f"\ntimestep       {self.settings['quench']['timestep']}\n",
                        f"thermo         100\n",
                        "thermo_style   custom step temp pe ke etotal press vol\n",
                        f"velocity       all create {self.settings['quench']['quench_melt_temp']} 1234579 rot yes dist gaussian\n",
                        "run            0\n",
                        # Melt and equilibrate at constant pressure (NPT)
                        f"fix           melt all npt temp {self.settings['quench']['quench_melt_temp']} {self.settings['quench']['quench_melt_temp']} $(100.0*dt) iso 0.0 0.0 $(1000.0*dt)\n",
                        "run            20000\n",
                        "unfix          melt\n",
                        # Quench at constant pressure (NPT)
                        f"fix            quench all npt temp {self.settings['quench']['quench_melt_temp']} {self.params['general']['temp']} $(100.0*dt) iso 0.0 0.0 $(1000.0*dt)\n",
                        f"run            {int(quench_nsteps)}\n",
                        "unfix          quench\n",
                        # Final relaxation at target temperature
                        f"fix            relax all npt temp {self.params['general']['temp']} {self.params['general']['temp']} $(100.0*dt) iso 0.0 0.0 $(1000.0*dt)\n",
                        "run            10000\n",
                        "unfix          relax\n",
                        f"write_data     {am_filename}"
                    ])
                if self.settings['quench']['run_local']:
                    subprocess.run(f"mpiexec -np {self.settings['quench']['n_procs']} lmp -in {lmp_input}", shell=True, check=True)
                else:
                    local_lmp_input = os.path.join(self.dir, f"make_amorphous_{system}.in")
                    with open(local_lmp_input, 'w', encoding="utf-8") as f_out, open(lmp_input, 'r', encoding="utf-8") as f_in:
                        f_out.write(f_in.read())
                    
                    logging.info(f"LAMMPS input file for amorphous structure generation has been saved to: {local_lmp_input}")
                    logging.info("Please run the following command to generate the amorphous structure:")
                    logging.info(f"mpiexec -np {self.settings['quench']['n_procs']} lmp -in {local_lmp_input}")
                    logging.info("After the simulation is complete, please rerun the program.")
                    exit()
            except Exception as e:
                logging.exception(f"Failed to generate amorphous structure for {system}: {e}")
                raise
        return am_filename

    def stacking(self, layer=2, sheetvsheet=False):
        """Stacks multiple layers of a 2D material and calculates interlayer distance.

        This method generates a multi-layer system by stacking copies of a
        single layer, allowing for custom shifts and vertical separation. It then
        runs a short LAMMPS simulation to relax the structure and compute the
        final interlayer distance (lattice constant `c`).

        Args:
            layer (int, optional): The total number of layers to create. Defaults to 2.
            sheetvsheet (bool, optional): If True, applies specific stacking
                for sheet-vs-sheet simulations. Defaults to False.

        Returns:
            float: The calculated interlayer distance (lat_c).

        Side Effects:
            - Creates a new LAMMPS data file for the multi-layer structure.
            - Modifies `self.shift_x` and `self.shift_y` if they are not already set.
        """

        filename = f"{self.dir}/build/{self.params['2D']['mat']}"
        lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none',  '-nocite'])
        try:
            lmp.commands_list(self.init(neigh=True))
            lmp.commands_list([
                f"region box block {self.dim['xlo']} {self.dim['xhi']} {self.dim['ylo']} {self.dim['yhi']} -5 {self.dim['yhi']+6*layer}",
                f"create_box       {self.data['2D']['natype']*layer} box",
                f"read_data       {filename}_1.lmp add append group layer_1",
            ])

            lat_c = self.lat_c if self.lat_c is not None else 6
            if self.lat_c is None:
                warnings.warn("lat_c is not set; using default value of 6.")

            if self.params['2D']['stack_type'] == 'AA':
                self.shift_x = 0
                self.shift_y = 0

            # Stack additional layers with specified shifts
            if sheetvsheet:
                for l in range(1, layer):
                    lmp.command(f"read_data {filename}_1.lmp add append shift 0 0 {l*lat_c} group layer_{l+1}")
                lmp.commands_list([
                    f"displace_atoms layer_3 move {self.shift_x} {self.shift_y} 0 units box",
                    f"displace_atoms layer_4 move {self.shift_x} {self.shift_y} 0 units box",
                ])
            else:
                for l in range(1, layer):
                    lmp.commands_list([
                        f"read_data {filename}_1.lmp add append shift 0 0 {l*lat_c} group layer_{l+1}",
                        f"displace_atoms layer_{l+1} move {self.shift_x*l} {self.shift_y*l} 0 units box",
                    ])

            # Renumber atom types for each layer
            for t in range(1, self.data['2D']['natype'] + 1):
                lmp.command(f"group 2D_{t} type {t}")
            g = 0
            i = 0
            for count in self.potentials['2D']['count'].values():
                for l in range(1, layer + 1):
                    for t in range(1, count + 1):
                        n = i + t
                        g += 1
                        lmp.commands_list([
                            f"group 2Dtype intersect 2D_{n} layer_{l}",
                            f"set group 2Dtype type {g}",
                            "group 2Dtype delete"
                        ])
                i += count
            self.sheet_potential(
                f"{self.dir}/build/sheet_{layer}.in.settings",
                layer,
                sheetvsheet=sheetvsheet
            )
            lmp.commands_list([
                f"include         {self.dir}/build/sheet_{layer}.in.settings",
                "run 0"
            ])

            # Minimize energy and extract interlayer distance
            if layer == 2:
                lmp.commands_list([
                    f"min_style       {self.settings['simulation']['min_style']}",
                    self.settings['simulation']['minimization_command'],
                    f"timestep        {self.settings['simulation']['timestep']}",
                    "compute l_1 layer_1 com",
                    "compute l_2 layer_2 com",
                    "variable comz_1 equal c_l_1[3]",
                    "variable comz_2 equal c_l_2[3]",
                    "run 0"
                ])
                com_l1 = lmp.extract_variable('comz_1', None, 0)
                com_l2 = lmp.extract_variable('comz_2', None, 0)
                lat_c = com_l2 - com_l1

            lmp.command(f"write_data  {filename}_{layer}.lmp")

        except Exception as e:
            logging.exception(f"LAMMPS simulation for stacking failed: {e}")
            raise
        finally:
            lmp.close()

        return lat_c

    def slab_generator(self, filename, cif_path, x, y, z):
        """Generates a material slab from a CIF file using Atomsk.

        This method converts a CIF file to an orthogonal cell, then duplicates
        it to create a slab of the desired dimensions.

        Args:
            filename (str): Output filename for the generated slab (LAMMPS data file).
            cif_path (str): Path to the CIF file for the crystal structure.
            x (float): Desired size in the x-direction (Angstroms).
            y (float): Desired size in the y-direction (Angstroms).
            z (float): Desired size in the z-direction (Angstroms).

        Side Effects:
            - Creates and deletes temporary files ('a.cif').
            - Overwrites the output file if it already exists.
            - Runs external Atomsk commands.

        Raises:
            subprocess.CalledProcessError: If an Atomsk command fails.
        """
        Path(filename).unlink(missing_ok=True)

        try:

                # Convert CIF to an orthogonal cell
            subprocess.run(f"atomsk {cif_path} -duplicate 2 2 1 -orthogonal-cell {filename} -ow -v 0", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Calculate duplication factors to reach the target size
            dim = utilities.get_model_dimensions(filename)
            x2 = round((x + 15) / dim['xhi'])
            y2 = round((y + 15) / dim['yhi'])
            z2 = round(z / dim['zhi'])

            # Duplicate the cell to create the final slab
            subprocess.run(f"atomsk {filename} -duplicate {x2} {y2} {z2} -center com -ow lmp -v 0", shell=True, check=True)

        except subprocess.CalledProcessError as e:
            logging.exception(f"Atomsk command failed during slab generation: {e}")
            raise
        except (FileNotFoundError, KeyError) as e:
            logging.exception(f"File or configuration error during slab generation: {e}")
            raise
    
    def center(self, system, filename):
        """Centers a structure within the simulation box using LAMMPS.

        Args:
            system (str): The name of the system to center ('2D', 'tip', 'sub').
            filename (str): The path to the LAMMPS data file to process.
        """
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
        try:
            lmp.commands_list(self.init(neigh=True))
            lmp.command(f"read_data       {filename}")

            # Set atomic masses
            elem = []
            i = 0
            for element, count in self.potentials[system]['count'].items():
                if not count or count == 1:
                    elem.append(element)
                    i += 1
                else:
                    for t in range(1, count + 1):
                        elem.append(element + str(t))
                        i += 1
                mass = data.atomic_masses[data.atomic_numbers[element]]
                lmp.command(f"mass {i} {mass}")
            # Define variables to calculate the displacement needed for centering
            lmp.commands_list([
                f"pair_style  {self.params[system]['pot_type']}",
                f"pair_coeff * * {self.potentials[system]['path']} {' '.join(elem)}",
                "compute zmin all reduce min z",
                "compute xmin all reduce min x",
                "compute ymin all reduce min y",
                "variable disp_z equal -c_zmin",
                "variable disp_x equal -(c_xmin+(xhi-xlo)/2.0)",
                "variable disp_y equal -(c_ymin+(yhi-ylo)/2.0)",
                "run 0",
                "displace_atoms all move v_disp_x v_disp_y v_disp_z units box",
                f"change_box all z final {self.dim['zlo']} {self.dim['zhi']}",
                "run 0",
                f"write_data  {filename}"
            ])
        except Exception as e:
            logging.exception(f"LAMMPS simulation for centering structure '{filename}' failed: {e}")
            raise
        finally:
            lmp.close()

    def sheet_potential(self, filename, layer, sheetvsheet=False, virtual = False):
        """Writes the LAMMPS potential settings for a multi-layer 2D sheet.

        This method defines element groups, atomic masses, and pair potentials
        for a stacked 2D material system. It supports hybrid potentials for
        intralayer and interlayer interactions.

        Args:
            filename (str): The file to write the LAMMPS settings to.
            layer (int): The number of 2D layers in the system.
            sheetvsheet (bool): If True, modifies interlayer interactions for
                sheet-vs-sheet models by setting near-zero energy for specific
                layer pairs.
        """
        with open(filename, 'w', encoding="utf-8") as f:
            self.elemgroup['2D'] = {}
            self.group_def = {}
            atype = 1

            # Define element groups and write atomic masses
            arr = self.number_sequential_atoms('2D')
            atype = self.define_elemgroup('2D', arr, layer=layer, atype=atype)
            self.set_masses('2D', f, layer=layer)

            # Define LAMMPS groups for each layer
            for l in range(layer):
                layer_groups = [
                    self.group_def[i][1]
                    for i in range(1, self.data['2D']['natype'] * layer + 1)
                    if f"2D_l{l + 1}" in self.group_def[i][0]
                ]
                f.write(f"group layer_{l+1} type {' '.join(layer_groups)}\n")
            
            potential_counts = {}
            num_atom_types = self.data['2D']['natype'] * layer
            if virtual:
                num_atom_types += 1
                self.group_def[num_atom_types] = ['virtual', str(num_atom_types), 'NULL', 'NULL']                
            
            # Set up pair style and coefficients
            if self.is_sheet_lj():
                f.write(f"pair_style hybrid {(self.params['2D']['pot_type'] + ' ') * layer} lj/cut 11.0\n")
                for l in range(layer):
                    potential_counts[l] = [
                        self.group_def[i][3] if f"2D_l{l + 1}" in self.group_def[i][0] else "NULL"
                        for i in range(1, num_atom_types + 1)
                    ]
                    f.write(f"pair_coeff * * {self.params['2D']['pot_type']} {l+1} {self.potentials['2D']['path']} {'  '.join(potential_counts[l])} # 2D Layer {l+1}\n")

                # Set interlayer LJ parameters
                index_pairs = [(i, j) for i in range(layer) for j in range(i + 1, layer)]
                if sheetvsheet:
                    index_pairs_low_e = [(1, 3), (0, 3), (0, 2)]
                    index_pairs_sheetvsheet = [p for p in index_pairs if p not in index_pairs_low_e]
                    self.set_sheet_LJ_params(f, index_pairs_sheetvsheet)
                    self.set_sheet_LJ_params(f, index_pairs_low_e, low=True)
                else:
                    self.set_sheet_LJ_params(f, index_pairs)
            else:
                f.write(f"pair_style {self.params['2D']['pot_type']}\n")
                potentials = [self.group_def[i][3] for i in range(1, num_atom_types + 1)]
                f.write(f"pair_coeff * * {self.potentials['2D']['path']} {'  '.join(potential_counts)} # 2D\n")
            if virtual:
                f.writelines([f"mass {num_atom_types} 1.0\n",
                                 f"pair_coeff * {num_atom_types} lj/cut 1e-100 1e-100\n"])
                                 
    def set_sheet_LJ_params(self, f, index_pairs, low=False):
        """Sets Lennard-Jones (LJ) pair coefficients for 2D sheet elements.

        This method computes LJ parameters (epsilon and sigma) for each pair of
        element types and writes the corresponding `pair_coeff` commands to the
        provided file.

        Args:
            f (file-like object): An open file to write the LAMMPS commands to.
            index_pairs (iterable): Pairs of layer indices for which to set LJ params.
            low (bool, optional): If True, sets epsilon to a very small number
                to effectively disable the interaction. Defaults to False.
        """
        for t in self.data['2D']['elem_comp']:
            for s in self.data['2D']['elem_comp']:
                e, sigma = utilities.lj_params(s, t)
                for i, j in index_pairs:
                    # Build atom type labels for each layer and element
                    t1 = f"{self.elemgroup['2D'][i][t][0]}*{self.elemgroup['2D'][i][t][-1]}"
                    t2 = f"{self.elemgroup['2D'][j][s][0]}*{self.elemgroup['2D'][j][s][-1]}"
                    if self.elemgroup['2D'][i][t][0] == self.elemgroup['2D'][i][t][-1]:
                        t1 = f"{self.elemgroup['2D'][i][t][0]}"
                    if self.elemgroup['2D'][j][s][0] == self.elemgroup['2D'][j][s][-1]:
                        t2 = f"{self.elemgroup['2D'][j][s][0]}"

                    # Ensure consistent ordering for LAMMPS pair_coeff
                    if self.elemgroup['2D'][i][t][0] > self.elemgroup['2D'][j][s][0]:
                        t1, t2 = t2, t1

                    # Write the pair_coeff line
                    epsilon = 1e-100 if low else e
                    f.write(f"pair_coeff {t1} {t2} lj/cut {epsilon} {sigma}\n")

    def _single_body_potential(self, filename, system):
        """Writes the LAMMPS potential settings for a single-body system.

        This method defines element groups, atomic masses, and the pair
        potential for a single-component system like a tip or substrate.

        Args:
            filename (str): The file path to write the LAMMPS settings to.
            system (str): The system for which to write settings ('tip' or 'sub').
        """
        try:
            arr = self.number_sequential_atoms(system)
            _ = self.define_elemgroup(system, arr)

            with open(filename, 'w') as f:
                self.set_masses(system, f)
                potentials = [self.group_def[i][2] for i in range(1, len(arr[system]) + 1)]
                f.writelines([
                    f"pair_style {self.params[system]['pot_type']}\n",
                    f"pair_coeff * * {self.potentials[system]['path']} {' '.join(potentials)}\n"
                ])
        except (IOError, KeyError) as e:
            logging.exception(f"Failed to write single body potential for {system}: {e}")
            raise

    def define_elemgroup(self, system, arr, layer=None, atype=1):
        """Defines and updates element groups for a given system.

        This method populates `self.group_def` and `self.elemgroup` with group
        definitions based on the provided atomic arrangement, supporting both
        layered ('2D') and non-layered systems.

        Args:
            system (str): The name of the system (e.g., '2D', 'sub', 'tip').
            arr (dict): A dictionary mapping system names to atomic identifiers.
            layer (int, optional): The number of layers for a '2D' system.
            atype (int, optional): The starting atom type index. Defaults to 1.

        Returns:
            int: The next available atom type index.
        """
        i = 0
        for element, count in self.potentials[system]['count'].items():
            if layer is not None and system == '2D':
                for l in range(layer):
                    for t in range(1, count + 1):
                        n = i + t
                        self.group_def[atype] = [f"{system}_l{l+1}_t{n}", str(atype), str(element), arr[system][n-1]]
                        self.elemgroup.setdefault(system, {}).setdefault(l, {}).setdefault(element, []).append(atype)
                        atype += 1
                i += count
            else:
                for _ in range(1, count + 1):
                    self.group_def[atype] = [f"{system}_t{i}", str(atype), str(element), arr[system][i]]
                    self.elemgroup.setdefault(system, {}).setdefault(element, []).append(atype)
                    atype += 1
                    i += 1
        return atype

    def number_sequential_atoms(self, system):
        """Generates a sequential numbering of atoms for a given system.

        For each element, assigns a unique identifier. If an element appears
        more than once, a numeric suffix is added (e.g., 'C1', 'C2').

        Args:
            system (str): The system to number atoms for ('2D', 'tip', 'sub').

        Returns:
            dict: A dictionary mapping atom indices to their new identifiers.
        """
        arr = {system: {}}
        i = 0
        for element, count in self.potentials[system]['count'].items():
            if not count or count == 1:
                arr[system][i] = element
                i += 1
            else:
                for t in range(1, count + 1):
                    arr[system][i] = element + str(t)
                    i += 1
        return arr

    def set_masses(self, system, f, layer=None):
        """Writes atomic mass definitions to a LAMMPS input file.

        Args:
            system (str): The system to set masses for ('2D', 'tip', 'sub').
            f (file-like object): The file to write the mass definitions to.
            layer (int, optional): The number of layers for a 2D system.
        """
        for m in self.data[system]['elem_comp']:
            mass = data.atomic_masses[data.atomic_numbers[m]]
            if system == '2D':
                if layer is not None:
                    if layer == 1 and len(self.elemgroup[system][0][m]) == 1:
                        f.write(f"mass {self.elemgroup[system][0][m][0]} {mass} #{m} {system}\n")
                    else:
                        f.write(f"mass {self.elemgroup[system][0][m][0]}*{self.elemgroup[system][layer-1][m][-1]} {mass} #{m} {system}\n")
                else:
                    if len(self.elemgroup[system][0][m]) == 1:
                        f.write(f"mass {self.elemgroup[system][0][m][0]} {mass} #{m} {system}\n")
                    else:
                        f.write(f"mass {self.elemgroup[system][0][m][0]}*{self.elemgroup[system][len(self.elemgroup[system])-1][m][-1]} {mass} #{m} {system}\n")
            else:
                if len(self.elemgroup[system][m]) == 1:
                    f.write(f"mass {self.elemgroup[system][m][0]} {mass} #{m} {system}\n")
                else:
                    f.write(f"mass {self.elemgroup[system][m][0]}*{self.elemgroup[system][m][-1]} {mass} #{m} {system}\n")

    def is_sheet_lj(self):
        """Checks if the 2D material potential requires a separate LJ term.

        Returns:
            bool or None: True if an explicit LJ term is needed, False if it is
            included, and None if the potential type is not recognized.
        """
        no_lj = ['airebo', 'comb', 'comb3']
        yes_lj = ['sw', 'tersoff', 'rebo', 'edip', 'meam', 'eam', 'bop', 'morse', 'rebomos', 'sw/mod']
        try:
            if self.params['2D']['pot_type'] in yes_lj:
                return True
            if self.params['2D']['pot_type'] in no_lj:
                return False
        except KeyError:
            logging.warning("Potential type for 2D material not specified in config.")
            return None
        return None
    
    def init(self,neigh=False, atom_style='atomic'):
        """Returns a list of LAMMPS commands for system initialization."""
        commands = [
            "# LAMMPS input script\n\n",
            "clear\n\n",
            "units           metal\n",
            f"atom_style      {atom_style}\n",
            "boundary   	p p p\n",
        ]
        if neigh:
            commands.extend([
                f"neighbor        {self.settings['simulation']['neighbor_list']} bin\n",
                f"{self.settings['simulation']['neigh_modify_command']}\n",
            ])
            
        return commands
