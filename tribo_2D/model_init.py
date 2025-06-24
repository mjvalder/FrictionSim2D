from pathlib import Path
import os
import subprocess

import numpy as np
from ase import io, data
from lammps import lammps

from tribo_2D import utilities


class model_init:
    def __init__(self, input_file, model='afm'):
        """
        Initialize the AFM simulation setup using an .INI configuration file.

        Args:
            input_file (str): Path to the input configuration file.
            model (str): Type of model to initialize, default is 'afm', which generates a tip, substrate and sheet. 
                Other options include 'sheetvsheet', which generates a four layer sheet.
                Alternatively, 'tip', 'substrate' and 'sheet' can be used to initialize only the respective components.
        """
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

        # Map model names to setup methods
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
                (self.sheet_dir[layer] / sub).mkdir(parents=True, exist_ok=True)

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
        self.dir = (
            f"sheet/{self.params['2D']['mat']}/"
            f"{self.params['2D']['x']}x_{self.params['2D']['y']}y"
        )
        for subdir in ["build", "potentials"]:
            Path(self.dir, subdir).mkdir(parents=True, exist_ok=True)        
        self.init_sheet()

    def setup_tip_part(self):
        self.dir = (
            f"tip/{self.params['tip']['amorph']}{self.params['tip']['mat']}/"
            f"r{self.params['tip']['r']}"
        )
        for subdir in ["build", "potentials"]:
            Path(self.dir, subdir).mkdir(parents=True, exist_ok=True)

        self.init_tip()

    def setup_substrate_part(self):
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
            'zhi': 10
        }
        self.init_substrate()

    def init_sheet(self, sheetvsheet=False):
        self.potentials['2D'] = {}
        # --- Read material and potential data ---
        self.data['2D'] = utilities.cifread(self.params['2D']['cif_path'])
        self.potentials['2D']['count'] = utilities.count_atomtypes(self.params['2D']['pot_path'])

        # --- Count atom types from the potential file ---
        self.data['2D']['natype'] = sum(self.potentials['2D']['count'].values())
        self.potentials['2D']['path'] = utilities.copy_file(self.params['2D']['pot_path'], Path(self.dir) / f"potentials")

        # --- Build 2D material and expand layers ---
        self.sheet_build()
        for layer in self.params['2D']['layers']:
            self.sheet_dir[layer] = Path(self.dir) / f"l_{layer}"
            if layer > 1:
                self.stacking(layer, sheetvsheet)

    def init_substrate(self):
        self.potentials['sub'] = {}
        self.data['sub'] = utilities.cifread(self.params['sub']['cif_path'])
        self.potentials['sub']['count'] = utilities.count_atomtypes(self.params['sub']['pot_path'])

        self.data['sub']['natype'] = sum(self.potentials['sub']['count'].values())
        self.potentials['sub']['path'] = utilities.copy_file(self.params['sub']['pot_path'], Path(self.dir) / f"potentials")
        self.sub_build()

    def init_tip(self):
        self.potentials['tip'] = {}
        self.data['tip'] = utilities.cifread(self.params['tip']['cif_path'])
        self.potentials['tip']['count'] = utilities.count_atomtypes(self.params['tip']['pot_path'])

        self.data['tip']['natype'] = sum(self.potentials['tip']['count'].values())
        self.potentials['tip']['path'] =utilities.copy_file(self.params['tip']['pot_path'], Path(self.dir) / f"potentials")
        self.tipx = self.tip_build()

    def tip_build(self):
        """
        Generate an AFM tip model based on the specified material and properties.
        Handles both crystalline and amorphous tips, sets up LAMMPS regions,
        groups, and writes the resulting data file.

        Inputs:
            self(dict): A dictionary containing simulation parameters
        """
        # Prepare file paths
        filename = f"{self.params['tip']['mat']}.lmp"
        slab_path = os.path.join(os.path.dirname(__file__),
                        "materials", f"{filename}")
        x = self.params['tip']['r']
        self.__single_body_potential(
            f"{self.dir}/build/tip.in.settings", 'tip')

        # Generate a slab for the tip of either amorphous or crystalline material
        if self.params['tip']['amorph'] == 'a':
            slab_path = self.make_amorphous('tip', 2500)

        else:
            self.slab_generator(slab_path,self.params['tip']['cif_path'], 2*x, 2*x, x)
            dim = utilities.get_model_dimensions(slab_path)
            x = dim['xhi']/2

        h = x / 2.25  # Define region height

        # Initialize and run LAMMPS generation and equilibration of the AFM tip
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

        lmp.close

        return x

    def sub_build(self):
        """
        Generate a substrate (slab) based on the specified material and properties.
        Handles both crystalline and amorphous structures, sets up LAMMPS regions,
        groups, and writes the resulting data file.

        Inputs:
            var (dict): A dictionary containing simulation parameters
        """

        # Prepare file paths
        filename = f"{self.params['sub']['mat']}.lmp"
        slab_path = os.path.join(
            os.path.dirname(__file__),
            "materials", f"{filename}"
        )
        self.__single_body_potential(
            f"{self.dir}/build/sub.in.settings", 'sub')

        # Generate a slab for the substrate of either amorphous or crystalline material
        if self.params['sub']['amorph'] == 'a':
            slab_path = self.make_amorphous('sub', 2500)

        else:
            self.slab_generator(slab_path, self.params['sub']['cif_path'], self.params['2D']['x'], self.params['2D']['y'], 10)

        # Initialize and run LAMMPS generation and equilibration of the AFM tip
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
        lmp.commands_list([
            "boundary p p p\n",
            "units metal\n",
            "atom_style      atomic\n",

            f"region box block {self.dim['xlo']} {self.dim['xhi']} {self.dim['ylo']} {self.dim['yhi']} {self.dim['zlo']} 12\n",
            f"create_box       {self.data['sub']['natype']} box\n\n",
            f"read_data {slab_path} add append\n",
            "group sub region box\n",
            "group box subtract all sub\n",
            "delete_atoms group box\n",
            f"change_box all x final {self.dim['xlo']} {self.dim['xhi']} y final {self.dim['ylo']} {self.dim['yhi']} z final {self.dim['zlo']} {self.dim['zhi']}\n\n",
            "reset_atoms     id\n",
            f"write_data      {self.dir}/build/sub.lmp"
        ])

        lmp.close

    def sheet_build(self):
        """
        Generate a 2D material sheet based on the specified material and properties.
        Handles the creation of the sheet, potential assignment, and LAMMPS data file generation.
        Inputs:
            var (dict): A dictionary containing simulation parameters
        """

        # --- Set sheet dimensions ---
        x = self.params['2D']['x']
        y = self.params['2D']['y']

        filename = f"{self.dir}/build/{self.params['2D']['mat']}_1.lmp"

        multiplier = utilities.check_potential_cif_compatibility(self.params['2D']['cif_path'], self.params['2D']['pot_path'])
        print('this is the multiplier', multiplier)
        typecount = sum(self.potentials['2D']['count'].values())
        Path(filename).unlink(missing_ok=True)

        # Generate the 2D sheet using Atomsk
        atomsk_command = f"echo n | atomsk {self.params['2D']['cif_path']} -ow {filename} -v 0"
        subprocess.run(atomsk_command, shell=True, check=True)
        if any(v != 1 for v in self.potentials['2D']['count'].values()) or multiplier != 1:
            utilities.renumber_atom_types(self.potentials['2D']['count'], filename)
            print('renumbered atom types')
        atomsk_command = f"atomsk {filename} -orthogonal-cell -ow lmp -v 0"
        subprocess.run(atomsk_command, shell=True, check=True)

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
            atomsk_command = f"echo n | atomsk {filename} -duplicate {a} {b} 1 -ow lmp -v 0"
            subprocess.run(atomsk_command, shell=True, check=True)
            utilities.renumber_atom_types(self.potentials['2D']['count'], filename)

        # Duplicate the sheet to match the specified dimensions
        dim = utilities.get_model_dimensions(filename)
        duplicate_a = round(x / dim['xhi'])
        duplicate_b = round(y / dim['yhi'])

        atomsk_command = f"atomsk {filename} -duplicate {duplicate_a} {duplicate_b} 1 -center com -ow lmp -v 0"
        subprocess.run(atomsk_command, shell=True, check=True)

        #  Verify that the lmp filetype is correct
        charge2atom = f"lmp_charge2atom.sh {filename}"
        subprocess.run(charge2atom, shell=True, check=True)

        self.dim = utilities.get_model_dimensions(filename)
        self.center('2D', filename)
        self.lat_c = self.stacking(2)

    def make_amorphous(self, system, tempmelt):
        """
        Generate an amorphous structure from a crystalline material using LAMMPS.
        This is acheived by melting the crystalline structure and then quenching it
        to room temperature.
        Inputs:
            system (str): The name of the system (e.g., 'tip', 'sub')
            filename (str): Path to the input LAMMPS data file for the crystalline structure
            am_filename (str): Path to the output LAMMPS data file for the amorphous structure
            tempmelt (float): Melting temperature in Kelvin
            var (dict): A dictionary containing simulation parameters
        """
        filename = f"{self.params[system]['mat']}.lmp"
        slab_path = os.path.join(os.path.dirname(__file__),
                    "materials", f"{filename}")
        am_filename = os.path.join(os.path.dirname(
                        __file__), "materials", f"amor_{filename}")
        filename = os.path.join(os.path.dirname(
                        __file__), "materials", filename)

        if not os.path.exists(am_filename):
            self.slab_generator(slab_path, self.params[system]['cif_path'], 200, 200, 50)

            quench = (tempmelt-self.params['general']['temproom'])*100
            pot_file = f"{self.dir}/build/{system}.in.settings"
            self.__single_body_potential(pot_file, system)
            lmp = lammps()
            lmp.commands_list([
                "boundary p p p\n",
                "units metal\n",
                "atom_style      atomic\n\n",

                f"read_data {filename}\n\n",
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
                f"fix             quench all nvt temp {tempmelt} {self.params['general']['temproom']} $(100.0*dt)\n ",
                f"run             {quench}\n ",
                "unfix           quench \n",
                f"write_data {am_filename}"
            ])

            lmp.close

        # os.remove(filename)

        return am_filename

    def stacking(self, layer=2, sheetvsheet=False):
        """
        Stack layers of 2D materials to create a multi-layer structure.

        Inputs:
            var (dict): A dictionary containing simulation parameters
            layer (int): The number of layers to stack (default is 2)
            sheetvsheet (bool): If True, creates two sets of two
                commensurate layers and stacks them accordingly.
        """
        # generate file for potentials
        self.__sheet_potential(
            f"{self.dir}/build/sheet_{layer}.in.settings", layer, sheetvsheet=sheetvsheet)

        filename = f"{self.dir}/build/{self.params['2D']['mat']}"
        lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none',  '-nocite'])

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

        x_shift = self.dim['xhi'] / 4 if self.params['2D']['stack_type'] == 'AB' else 0
        # Determine the interlayer distance (lat_c)
        if self.lat_c is not None:
            lat_c = self.lat_c
        else:
            lat_c = 6  # Default value if not set

        if sheetvsheet:
            for l in range(1, layer):
                lmp.command(
                    f"read_data {filename}_1.lmp add append shift 0 0 {l*lat_c} group layer_{l+1}\n")
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

        for t in range(self.data['2D']['natype']):
            t += 1
            lmp.command(
                f"group 2D_{t} type {t}\n"
            )

        g = 0
        i = 0
        c = 0

        for count in self.potentials['2D']['count'].values():
            i += c
            for l in range(1, layer+1):
                for t in range(1, count+1):
                    n = i + t
                    g += 1
                    lmp.commands_list([
                        f"group 2Dtype intersect 2D_{n} layer_{l}\n",
                        f"set group 2Dtype type {g}\n",
                        "group 2Dtype delete\n"
                    ])
                    c = count

        lmp.commands_list([
            f"include         {self.dir}/build/sheet_{layer}.in.settings\n\n",
            "#----------------- Minimize the system -------------------\n\n",
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

        # Extract center of mass (COM) for each layer
        com_l1 = lmp.extract_variable('comz_1', None, 0)
        com_l2 = lmp.extract_variable('comz_2', None, 0)
        # Compute lattice constant from the z-coordinates
        lat_c = com_l2 - com_l1

        return lat_c

    def slab_generator(self, filename, cif_path, x, y, z):
        """
        Generates a slab of a material from the provided CIF file and size parameters.
        Inputs:
            system (str): The name of the system (e.g., 'tip', 'sub')
            var (dict): A dictionary containing simulation parameters
            x (float): The size in the x-direction
            y (float): The size in the y-direction
            z (float): The size in the z-direction
        """

        Path(filename).unlink(missing_ok=True)
        Path('a.cif').unlink(missing_ok=True)
        atomsk_command = f"atomsk {cif_path} -duplicate 2 2 1 -orthogonal-cell a.cif -ow -v 0"
        subprocess.run(atomsk_command, shell=True, check=True)
        cif = utilities.cifread("a.cif")
        x2 = round((x+15)/cif["lat_a"])
        y2 = round((y+15)/cif["lat_b"])
        z2 = round(z/cif["lat_c"])
        atomsk_command2 = f"atomsk a.cif -duplicate {x2} {y2} {z2} {filename} -ow -v 0"
        subprocess.run(atomsk_command2, shell=True, check=True)
    
    def center(self, system, filename):
        """
        Centers objects in the simulation box using LAMMPS.
        Inputs:
            system (str): The name of the system (e.g., 'tip', 'sub', '2D')
            filename (str): Path to the input LAMMPS data file
            var (dict): A dictionary containing simulation parameters
        """
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])

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

        lmp.commands_list([
            f"pair_style  {self.params[system]['pot_type']}\n",
            f"pair_coeff * * {self.params[system]['pot_path']} {' '.join(elem)}\n",

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
        lmp.close

    def __sheet_potential(self,filename,layer,sheetvsheet=False):
        """
        Writes the settings for applying potentials in LAMMPS to a specified file for simulations with only 2D sheets.
        Args:
        filename (str): The name of the file to write to.
            layer (int): The number of layers in the 2D system.
            sheetvsheet (bool): If True, the simulation requires stage layers and so the potentials interactions between layers needs to be modified
            """    
        with open(filename, 'w') as f:
            self.elemgroup['2D'] = {}
            self.group_def = {}
            potentials= {}
            atype = 1

            arr = self.number_sequential_atoms('2D')

            atype = self.define_elemgroup('2D', arr, layer=layer, atype=atype)
            self.set_masses('2D', f, layer=layer)

            for l in range(layer):
                layer_g = [self.group_def[i][1] for i in range(1,self.data['2D']['natype']*layer+1) if "2D_l"+str(l+1) in self.group_def[i][0]]
                f.write(f"group layer_{l+1} type {' '.join(layer_g)}\n")

            f.write(f"pair_style hybrid {(self.params['2D']['pot_type']+' ') * layer} lj/cut 11.0\n")

            for l in range(layer):
                potentials[l] = [
                    self.group_def[i][3] if "2D_l"+str(l+1) in self.group_def[i][0] else "NULL"
                    for i in range(1,self.data['2D']['natype']*layer+1)
                ]         

                f.write(f"pair_coeff * * {self.params['2D']['pot_type']} {l+1} {self.potentials['2D']['path']} {'  '.join(potentials[l])} # interlayer '2D' Layer {l+1}\n")

            index_pairs = [(i, j) for i in range(layer) for j in range(i+1, layer)]

            if sheetvsheet:
                index_pairs_low_e = [(1, 3), (0, 3), (0, 2)]
                index_pairs_sheetvsheet = [pair for pair in index_pairs if pair not in index_pairs_low_e]
                self.set_sheet_LJ_params(f, index_pairs_sheetvsheet)
                self.set_sheet_LJ_params(f, index_pairs_low_e, low=True)

            else:
                self.set_sheet_LJ_params(f, index_pairs)

    def set_sheet_LJ_params(self, f, index_pairs, low=False):

        for t in self.data['2D']['elem_comp']:
            for s in self.data['2D']['elem_comp']:
                e,sigma = utilities.LJparams(s,t)
                for i,j in index_pairs:
                    t1 = f"{self.elemgroup['2D'][i][t][0]}*{self.elemgroup['2D'][i][t][-1]}"
                    t2 = f"{self.elemgroup['2D'][j][s][0]}*{self.elemgroup['2D'][j][s][-1]}"
                    if self.elemgroup['2D'][i][t][0] == self.elemgroup['2D'][i][t][-1]:
                        t1 = f"{self.elemgroup['2D'][i][t][0]}"
                    if self.elemgroup['2D'][j][s][0] == self.elemgroup['2D'][j][s][-1]:
                        t2 = f"{self.elemgroup['2D'][j][s][0]}"
                    if self.elemgroup['2D'][i][t][0]>self.elemgroup['2D'][j][s][0]:
                        t1, t2 = t2, t1
                    if low:
                        f.write(f"pair_coeff {t1} {t2} lj/cut 1e-100 {sigma} \n")
                    else:
                        f.write(f"pair_coeff {t1} {t2} lj/cut {e} {sigma} \n")

    def __single_body_potential(self,filename,system):
        """
        Writes the settings for applying potentials in LAMMPS to a single body system.
        Args:
        filename (str): The name of the file to write to.
        system (str): The system for which the settings are being written.
        var (dict): A dictionary containing the potential and data information for the simulation.
        """

        potentials= {}
        arr = self.number_sequential_atoms(system)
        _ = self.define_elemgroup(system,arr)
        
        with open(filename, 'w') as f:   
            self.set_masses(system, f)

            potentials = [self.group_def[i][2] for i in range(1,len(arr[system])+1)]

            f.writelines([
                f"pair_style {self.params[system]['pot_type']}\n",
                f"pair_coeff * * {self.potentials[system]['path']} {' '.join((potentials))}\n"])

    def define_elemgroup(self, system, arr, layer=None, atype=1): 

        i = 0

        for element, count in self.potentials[system]['count'].items():
            if layer is not None and system == '2D':
                for l in range(layer):
                    for t in range(1, count+1):
                        n = i + t
                        self.group_def.update({atype: [f"{system}_l{l+1}_t{n}", str(atype), str(element), arr[system][n-1]]})
                        self.elemgroup.setdefault(system, {}).setdefault(l, {}).setdefault(element, []).append(atype)
                        atype += 1
                i += count    
            else:
                for _ in range(1, count+1):
                    self.group_def.update({atype: [f"{system}_t{i}", str(atype), str(element), arr[system][i]]})
                    self.elemgroup.setdefault(system, {}).setdefault(element, []).append(atype)
                    atype += 1
                    i += 1

        return atype

    def number_sequential_atoms(self, system):
        arr = {}
        arr[system] = {} 
        i=0    
        for element, count in self.potentials[system]['count'].items():
            if not count or count == 1:
                arr[system][i] = element
                i+=1
            else:
                for t in range(1,count+1):
                    arr[system][i] = element + str(t)
                    i+=1
        return arr
    
    def set_masses(self, system, f, layer = None):
        for m in self.data[system]['elem_comp']: 
            mass=data.atomic_masses[data.atomic_numbers[m]] 
            if system == '2D':
                if layer is not None:
                    if layer == 1 and len(self.elemgroup[system][0][m])==1:
                        f.write(f"mass {self.elemgroup[system][0][m][0]} {mass} #{m} {system}\n")
                    else:
                        f.write(f"mass {self.elemgroup[system][0][m][0]}*{self.elemgroup[system][layer-1][m][-1]} {mass} #{m} {system}\n")
            else:
                if len(self.elemgroup[system][m])==1:
                    f.write(f"mass {self.elemgroup[system][m][0]} {mass} #{m} {system}\n")
                else:
                    f.write(f"mass {self.elemgroup[system][m][0]}*{self.elemgroup[system][m][-1]} {mass} #{m} {system}\n")