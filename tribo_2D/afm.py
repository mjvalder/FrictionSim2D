"""AFM simulation module for the Prandtl-Tomlinson model.

This module generates LAMMPS simulation cells and input files for AFM
simulations. It handles configuration parsing, atomic structure building
(2D material, substrate, and tip), potential assignment, and directory setup.
"""

import os
import re
import logging
import tempfile

from lammps import lammps

from tribo_2D import model_init, utilities
from tribo_2D.settings import file as settings_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AFMSimulation(model_init.ModelInit):
    """Generates simulation cells and input files for AFM simulations.

    This class builds atomic structures (2D material, substrate, tip) and
    prepares the necessary directory structure and LAMMPS scripts for simulation
    setup, execution, and post-processing.

    Attributes:
        input_file (str): Path to the input configuration file.
        langevin_multiplier (int): A multiplier for atom types when using a
            Langevin thermostat (1 for regular, 3 for Langevin).
    """

    def __init__(self, input_file):
        """Initializes and runs the AFM simulation workflow.

        Args:
            input_file (str): Path to the input configuration file.
        """
        self.input_file = input_file
        try:
            self._run_simulations()
        except Exception:
            logging.exception("A critical error occurred during the AFM simulation setup.")
            raise

    def _run_simulations(self):
        """Iterates through materials and sizes, running a simulation for each.

        Reads the main configuration to get a list of materials and generates
        a simulation run for each one. If no materials are specified, it runs
        a single simulation with the base configuration.
        """
        try:
            config = utilities.read_config(self.input_file)
            materials = self._get_materials(config)

            with open(self.input_file, "r", encoding="utf-8") as config_file:
                base_config_str = config_file.read()
        except FileNotFoundError:
            logging.error(f"Input file not found: {self.input_file}")
            return
        except Exception as e:
            logging.error(f"Failed to read or parse configuration: {e}")
            return

        if materials:
            for mat in materials:
                try:
                    print(f"Setting up simulation for material {mat}...")
                    mat_config_str = base_config_str.replace("{mat}", mat)
                    self._run_simulations_for_sizes(config, mat_config_str)
                except Exception as e:
                    logging.error(f"Simulation setup failed for material {mat}: {e}")
                    continue 
        else:
            self._run_simulations_for_sizes(config, base_config_str)

    def _get_materials(self, config):
        """Reads the list of materials from the configuration.

        Args:
            config (dict): The parsed configuration dictionary.

        Returns:
            list: A list of material names, or an empty list if none are found.
        """
        materials_list = config['2D'].get('materials_list')
        if not materials_list:
            return []
        if isinstance(materials_list, list):
            return materials_list
        try:
            with open(materials_list, "r", encoding="utf-8") as materials_file:
                return [line.strip() for line in materials_file]
        except FileNotFoundError:
            logging.warning(f"Materials list file not found: {materials_list}")
            return []

    def _run_simulations_for_sizes(self, config, config_str):
        """Reads simulation sizes from config and runs a simulation for each.

        If specific 'x' and 'y' sizes are defined in the config, this method
        iterates through each pair, creating a temporary config file and
        running the system setup for each. If no sizes are specified, it
        runs a single simulation using the provided config string.

        Args:
            config (dict): The parsed configuration dictionary.
            config_str (str): The string content of the configuration file.
        """
        x_sizes = config['2D'].get('x')
        y_sizes = config['2D'].get('y')

        if not isinstance(x_sizes, list):
            # If no sizes are specified, run a single simulation.
            try:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".ini", encoding="utf-8") as temp_config_file:
                    temp_config_file.write(config_str)
                    temp_config_name = temp_config_file.name
                self.system_setup(temp_config_name)
            except Exception:
                logging.exception(f"Single simulation run failed")
                raise
            finally:
                if os.path.exists(temp_config_name):
                    os.remove(temp_config_name)
            return

        if len(x_sizes) != len(y_sizes):
            raise ValueError("The number of x and y sizes must be the same.")
        
        sizes = list(zip(x_sizes, y_sizes))

        for x_val, y_val in sizes:
            temp_config_name = None
            try:
                size_config_str = re.sub(r'^(x\s*=\s*.*)$', f'x = {x_val}', config_str, flags=re.MULTILINE)
                size_config_str = re.sub(r'^(y\s*=\s*.*)$', f'y = {y_val}', size_config_str, flags=re.MULTILINE)
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".ini", encoding="utf-8") as temp_config_file:
                    temp_config_file.write(size_config_str)
                    temp_config_name = temp_config_file.name
                self.system_setup(temp_config_name)
            except Exception:
                logging.exception(f"Simulation for size {x_val}x{y_val} failed")
                raise  # Re-raise to stop the process
            finally:
                if temp_config_name and os.path.exists(temp_config_name):
                    os.remove(temp_config_name)

    def system_setup(self, config):
        """Initializes the system and generates LAMMPS scripts.

        This method calls the parent initializer, sets up thermostat regions
        if a Langevin thermostat is used, and then generates the LAMMPS
        scripts for system initialization and sliding.

        Args:
            config (str): Path to the temporary configuration file.
        """
        try:
            super().__init__(config, model='afm')
            for system in self.systems:
                if system not in self.params:
                    raise ValueError(f"System '{system}' is defined in 'systems' but not in the parameters.")
            
            if self.settings['thermostat']['type'] == 'langevin':
                self.langevin_multiplier = 3
                for key in ('tip', 'sub'):
                    self.set_three_regions(key)
            else:
                self.langevin_multiplier = 1

            self.generate_system_init_script()
            self.generate_slide_script()
        except Exception as e:
            logging.error(f"System setup failed: {e}")
            raise
        

    def _write_langevin_thermostat_commands(self, f_out):
        """Writes the LAMMPS commands for a Langevin thermostat to a file.

        Args:
            f_out (file): The file object to write the commands to.
        """
        if not self.settings['geometry']['rigid_tip']:
            f_out.writelines([
                "compute         temp_tip tip_thermo temp/partial 0 1 0\n",
                f"fix             lang_tip tip_thermo langevin {self.params['general']['temp']} {self.params['general']['temp']} $(100.0*dt) 699483 zero yes\n",
                "fix_modify      lang_tip temp temp_tip\n\n"
            ])
        f_out.writelines([
            "compute         temp_sub sub_thermo temp/partial 0 1 0\n",
            f"fix             lang_sub sub_thermo langevin {self.params['general']['temp']} {self.params['general']['temp']} $(100.0*dt) 2847563 zero yes\n",
            "fix_modify      lang_sub temp temp_sub\n\n",
            "fix             nve_all all nve\n\n",
        ])

    def generate_system_init_script(self):
        """Generates a LAMMPS script for system initialization and indentation.

        The script handles simulation box creation, reading data files,
        setting up potentials, and running an initial equilibration. It then
        indents the AFM tip into the 2D material to a specified load.
        """
        for layer in self.params['2D']['layers']:
        
            self.elemgroup = {}
            self.group_def = {}
            tip_x = self.dim['xhi'] / 2
            tip_y = self.dim['yhi'] / 2
            tip_z = self.settings['geometry']['tip_base_z'] + self.lat_c * (layer - 1) / 2
            filename = f"{self.sheet_dir[layer]}/lammps/system.lmp"

            # Find gap between the 2D material and the substrate.
            gap = self.afm_potentials_setup(layer)
            height_2d = self.params['sub']['thickness'] + 0.5 + gap
            
            with open(filename, 'w', encoding="utf-8") as f_out:
                f_out.writelines(self.init(neigh=True))
                f_out.writelines([
                    "comm_style tiled\n",
                    f"region box block {self.dim['xlo']} {self.dim['xhi']} {self.dim['ylo']} {self.dim['yhi']} -5 {tip_z + self.params['tip']['r']}\n",
                    f"create_box      {self.ngroups[layer]} box\n\n",

                    "# Read data files.\n",
                    f"read_data       {self.dir}/build/sub.lmp add append group sub\n",
                    f"read_data       {self.dir}/build/tip.lmp add append shift {tip_x} {tip_y} {tip_z}  group tip offset {self.data['sub']['natype']*self.langevin_multiplier} 0 0 0 0\n",
                    f"read_data       {self.dir}/build/{self.params['2D']['mat']}_{layer}.lmp add append shift 0.0 0.0 {height_2d} group 2D offset {self.data['tip']['natype']*self.langevin_multiplier+self.data['sub']['natype']*self.langevin_multiplier} 0 0 0 0\n\n"

                    "# Apply potentials.\n",
                    f"include        {self.sheet_dir[layer]}/lammps/system.in.settings\n\n",
                    "balance 1.0 rcb\n"
                ])

                if self.settings['output']['dump']['system_init']:
                    f_out.writelines([
                        "# Create visualization files.\n",
                        f"dump            sys all atom {self.settings['output']['dump_frequency']['system_init']} ./{self.dir}/visuals/system_{layer}.lammpstrj\n\n",
                    ])
                
                tip_fix_group = "tip_all" if self.settings['geometry']['rigid_tip'] else "tip_fix"
                
                f_out.writelines([
                    "# Minimize the system.\n",
                    f"min_style      {self.settings['simulation']['min_style']}\n",
                    f"{self.settings['simulation']['minimization_command']}\n",
                    f"timestep       {self.settings['simulation']['timestep']}\n",
                    f"thermo         {self.settings['simulation']['thermo']}\n",
                    "# Apply thermostat.\n",
                    f"group           fixset union sub_fix {tip_fix_group}\n",
                    "group           system subtract all fixset\n\n",
                    f"velocity        system create {self.params['general']['temp']} 492847948\n\n",
                ])
                if self.settings['thermostat']['type'] == 'langevin':
                    self._write_langevin_thermostat_commands(f_out)

                f_out.writelines([
                    "fix             sub_fix sub_fix setforce 0.0 0.0 0.0 \n",
                    "velocity        sub_fix set 0.0 0.0 0.0\n\n",
                    f"fix             tip_f {tip_fix_group} rigid/nve single force * off off off torque * off off off\n\n",
                    "run             10000\n\n",
                    "unfix           tip_f \n\n",
                    "# Tip Indentation.\n",
                    "displace_atoms  tip_all move 0.0 0.0 -20.0 units box\n\n",
                    "# Apply constraints.\n",
                    f"fix             tip_f {tip_fix_group} rigid/nve single force * off off on torque * off off off\n\n",
                    "variable        f equal 0.0\n",

                    f"variable find index {' '.join(str(x) for x in self.params['general']['force'])}\n",
                    "label force_loop\n",
                    "balance 1.0 rcb\n",
                    "# Set up initial parameters.\n",
                    "variable        num_floads equal 100\n",
                    "variable        r equal 0.0\n",
                    "variable        fincr equal (${find}-${f})/${num_floads}\n",
                    "thermo_modify   lost ignore flush yes\n\n",
                    "# Apply pressure to the tip.\n",
                    "variable i loop ${num_floads}\n",
                    "label loop_load\n\n",
                    "variable f equal ${f}+${fincr} \n\n",
                    "# Set force variable.\n",
                    f"variable n equal -v_f/(count({tip_fix_group})*1.602176565)\n",
                    f"fix forcetip {tip_fix_group} aveforce 0.0 0.0 $n\n",
                    "run 100 \n\n",
                    "unfix forcetip\n\n",
                    "next i\n",
                    "jump SELF loop_load\n\n",
                    "# Equilibration.\n",
                    f"fix forcetip {tip_fix_group} aveforce 0.0 0.0 $n\n",
                    "variable        dispz equal xcm(tip_all,z)\n\n",
                    "run 100 pre yes post no\n\n",
                    "# Loop to check for displacement stabilization.\n",
                    "label check_r\n\n",
                    "variable disp_l equal ${dispz}\n",
                    "variable disp_h equal ${dispz}\n\n",
                    "variable disploop loop 50\n",
                    "label disp\n\n",
                    "run 100 pre no post no\n\n",
                    "if '${dispz}>${disp_h}' then 'variable disp_h equal ${dispz}'\n",
                    "if '${dispz}<${disp_l}' then 'variable disp_l equal ${disp_l}'\n\n",
                    "next disploop\n",
                    "jump SELF disp\n\n",
                    "variable r equal ${disp_h}-${disp_l}\n\n",
                    "# Check if displacement has stabilized.\n",
                    "if '${r} < 0.2' then 'jump SELF loop_end' else 'jump SELF check_r'\n\n",
                    "label loop_end\n\n",
                    f"write_data {self.sheet_dir[layer]}/data/load_$(v_find)N.data\n",
                    "next find\n",
                    "jump SELF force_loop"
                ])

    def generate_slide_script(self):
        """Generates a LAMMPS script for the AFM sliding simulation.

        This script reads the previously indented system, applies a constant
        normal load, and then pulls the tip laterally at a constant velocity
        to simulate sliding and measure friction.
        """
        spring_ev = self.params['tip']['cspring'] / 16.02176565  # eV/A^2
        damp_ev = self.params['tip']['dspring'] / 0.01602176565  # eV/(A^2/ps)
        tipps = self.params['tip']['s'] / 100  # Angstrom/ps

        tip_fix_group = "tip_all" if self.settings['geometry']['rigid_tip'] else "tip_fix"

        for layer in self.params['2D']['layers']:
            filename = f"{self.sheet_dir[layer]}/lammps/slide_{self.params['tip']['s']}ms.lmp"
            with open(filename, 'w', encoding="utf-8") as f_out:
                f_out.writelines([
                    f"variable find index {' '.join(str(x) for x in self.params['general']['force'])}\n",
                    "label force_loop\n",
                    f"variable a index 0 {' '.join(str(x) for x in self.scan_angle)} 0\n",
                    "label angle_loop\n",
                ])
                
                f_out.writelines(self.init(neigh=True))

                f_out.writelines([
                    f"timestep       {self.settings['simulation']['timestep']}\n",
                    f"thermo         {self.settings['simulation']['thermo']}\n",
                    "comm_style       tiled\n",
                    f"read_data       {self.sheet_dir[layer]}/data/load_$(v_find)N.data\n\n",
                    f"include         {self.sheet_dir[layer]}/lammps/system.in.settings\n\n",
                    "balance 1.0 rcb\n",
                ])

                if self.settings['output']['dump']['slide']:
                    f_out.writelines([
                        "# Create visualization files.\n",
                        f"dump            sys all atom {self.settings['output']['dump_frequency']['slide']} ./{self.dir}/visuals/system_{layer}.lammpstrj\n\n",
                        "dump_modify sys append yes\n",
                    ])

                f_out.writelines([
                    "# Apply constraints.\n",
                    "fix             sub_fix sub_fix setforce 0.0 0.0 0.0 \n",
                    f"fix             tip_f {tip_fix_group} rigid/nve single force * on on on torque * off off off\n\n",
                    "# Apply thermostat.\n",
                ])

                if self.settings['thermostat']['type'] == 'langevin':
                    self._write_langevin_thermostat_commands(f_out)

                f_out.writelines([

                    "# Apply constant normal force to the tip.\n",
                    "variable        Ftotal          equal -v_find/1.602176565\n",
                    f"variable        n           equal v_Ftotal/count({tip_fix_group})\n",
                    f"fix             forcetip {tip_fix_group} aveforce 0.0 0.0 $n\n\n",

                    "# Define computes for output.\n",
                    f"compute COM_top layer_{layer} com\n",
                    "variable comx equal c_COM_top[1] \n",
                    "variable comy equal c_COM_top[2] \n",
                    "variable comz equal c_COM_top[3] \n\n",

                    "compute COM_tip tip_fix com\n",
                    "variable comx_tip equal c_COM_tip[1] \n",
                    "variable comy_tip equal c_COM_tip[2] \n",
                    "variable comz_tip equal c_COM_tip[3] \n\n",
                    "# Calculate friction forces.\n",
                    "variable        fz_tip   equal  f_forcetip[3]*1.602176565\n\n",
                    "variable        fx_spr   equal  f_spr[1]*1.602176565\n\n",
                    "variable        fy_spr   equal f_spr[2]*1.602176565\n\n",
                    f"fix             fc_ave all ave/time 1 1000 {self.settings['output']['results_frequency']} v_fz_tip v_fx_spr v_fy_spr v_comx v_comy v_comz v_comx_tip v_comy_tip v_comz_tip file ./{self.dir}/results/fc_ave_slide_$(v_find)nN_$(v_a)angle_{self.params['tip']['s']}ms_l{layer}\n\n",

                    "# Apply spring loading for sliding.\n",
                    f"fix             damp tip_fix viscous {damp_ev}\n\n",

                    "variable spring_x equal cos(v_a*PI/180)\n",
                    "variable spring_y equal sin(v_a*PI/180)\n\n",
                    "# Add lateral harmonic spring to pull the tip.\n",
                    f"fix             spr tip_fix smd cvel {spring_ev} {tipps} tether $(v_spring_x) $(v_spring_y) NULL 0.0\n\n",
                    "run 200000\n\n",

                    f"if '$(v_a) == {self.params['general']['scan_angle'][1]}' then &\n",
                    "'next a' & \n",
                    "'jump SELF find_incr'\n\n",

                    f"if '$(v_find) == {self.params['general']['scan_angle'][3]}' then &\n",
                    "'next a' & \n",
                    "'clear' & \n",
                    "'jump SELF angle_loop'\n\n",

                    "label find_incr\n\n",
                    "next find\n",
                    "clear\n",
                    "jump SELF force_loop"
                ])

    def set_three_regions(self, system):
        """Divides a system into fixed, thermostat, and mobile regions.

        This method is used for applying a Langevin thermostat, where only a
        portion of the tip and substrate are thermalized. It rewrites the
        LAMMPS data file for the specified system, assigning new atom types
        to distinguish between the fixed, thermostat-controlled, and fully
        mobile parts of the body.

        Args:
            system (str): The system to partition ('tip' or 'sub').
        """
        if system == 'tip':
            h = self.tipx / self.settings['geometry']['tip_reduction_factor']
            boundaries = self.settings['thermostat']['langevin_boundaries']['tip']
            f_zlo, f_zhi = [h - val for val in boundaries['fix']]
            t_zlo, t_zhi = [h - val for val in boundaries['thermo']]
        elif system == 'sub':
            thickness = self.params['sub']['thickness']
            boundaries = self.settings['thermostat']['langevin_boundaries']['sub']
            f_zlo = boundaries['fix'][0] * thickness
            f_zhi = boundaries['fix'][1] * thickness
            t_zlo = boundaries['thermo'][0] * thickness
            t_zhi = boundaries['thermo'][1] * thickness
            h = thickness
            
        dim = utilities.get_model_dimensions(f'{self.dir}/build/{system}.lmp')
        potential_file = f'{self.dir}/build/{system}_3layers.in.settings'
        self.__single_body_3layer(potential_file, system)
        
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
        lmp.commands_list([
            "boundary p p p",
            "units metal",
            "atom_style      atomic",
            f"region box block {dim['xlo']} {dim['xhi']} {dim['ylo']} {dim['yhi']} -5 {h}",
            f"create_box      {self.data[system]['natype']*3} box",
            f"read_data       {self.dir}/build/{system}.lmp add append",
            f"include         {potential_file}",
            f"# Identify the fixed atoms of {system}.",
            f"region          {system}_fix block INF INF INF INF {f_zlo} {f_zhi} units box",
            f"group           {system}_fix region {system}_fix",
            f"# Identify thermostat region of {system}.",
            f"region          {system}_thermo block INF INF INF INF {t_zlo} {t_zhi} units box",
            f"group           {system}_thermo region {system}_thermo",
        ])

        # Assign new atom types for fixed and thermostat regions.
        for t in range(self.data[system]['natype']):
            t += 1
            lmp.command(f"group {system}_{t} type {t}")

        i = 1
        for t in range(self.data[system]['natype']):
            t += 1
            lmp.commands_list([
                f"set group {system}_{t} type {i}",
                f"group {system}_fix_{t} intersect {system}_fix {system}_{t}",
                f"set group {system}_fix_{t} type {i+1}",
                f"group {system}_fix_{t} delete",
                f"group {system}_thermo_{t} intersect {system}_thermo {system}_{t}",
                f"set group {system}_thermo_{t} type {i+2}",
                f"group {system}_thermo_{t} delete",
                f"group {system}_{t} delete"
            ])
            i += 3

        lmp.command(f"write_data {self.dir}/build/{system}.lmp")
        lmp.close

    def afm_potentials_setup(self, layer):
        """Writes the potential settings file for the AFM simulation.

        This method configures hybrid potentials and defines pair coefficients
        for all interactions, including intra-system potentials (e.g., EAM for
        a metal tip), inter-system Lennard-Jones interactions (e.g., tip-sheet),
        and inter-layer interactions for multi-layer 2D materials.

        Args:
            layer (int): The number of layers in the 2D material.

        Returns:
            float: The maximum sigma value from Lennard-Jones interactions
                   between the 2D material and the substrate, used for
                   calculating the initial gap.
        """
        lj_sheet = self.is_sheet_lj()
        filename = f"{self.dir}/l_{layer}/lammps/system.in.settings"
        with open(filename, 'w', encoding="utf-8") as f_out:
            atype = 1

            # Define element groups for all systems.
            for system in self.systems:
                arr = super().number_sequential_atoms(system)
                if system == '2D':
                    atype = super().define_elemgroup(system, arr, layer=layer, atype=atype)
                else:
                    atype = self.__define_elemgroup_3regions(
                        system, arr, atype=atype)

            # Set masses for all atoms.
            for system in self.systems:
                super().set_masses(system, f_out, layer=layer)

            # Define groups for different parts of the model.
            for system in self.systems:
                all_types = [self.group_def[i][1] for i in range(
                    1, self.ngroups[layer]+1) if system in self.group_def[i][0]]
                f_out.write(f"group {system}_all type {' '.join(all_types)}\n")
                if system == '2D':
                    for l in range(layer):
                        layer_g = [self.group_def[i][1] for i in range(
                            1, self.ngroups[layer]+1) if f"2D_l{l+1}" in self.group_def[i][0]]
                        f_out.write(
                            f"group layer_{l+1} type {' '.join(layer_g)}\n")
                else:
                    for n in ["_fix", "_thermo"]:
                        sub_group = [self.group_def[i][1] for i in range(
                            1, self.ngroups[layer]+1) if system+n in self.group_def[i][0]]
                        f_out.write(
                            f"group {system}{n} type {' '.join(sub_group)}\n")

            # Determine potential types for pair_style hybrid.
            potential_counts = {}
            for system in ['sub', 'tip']:
                pot_type = self.params[system]['pot_type']
                potential_counts[pot_type] = potential_counts.get(pot_type, 0) + 1

            pot_2d = self.params['2D']['pot_type']
            if lj_sheet:
                potential_counts[pot_2d] = potential_counts.get(pot_2d, 0) + layer
            else:
                potential_counts[pot_2d] = potential_counts.get(pot_2d, 0) + 1

            # Write pair_style hybrid command.
            f_out.write("group mobile union tip_thermo sub_thermo\n")
            hybrid_style_parts = [pot for pot, count in potential_counts.items() for _ in range(count)]
            hybrid_style_parts.append("lj/cut 8.0")
            f_out.write(f"pair_style hybrid {' '.join(hybrid_style_parts)}\n")

            # Write pair_coeff commands for intra-system potentials.
            potential_indices = {pot: 0 for pot in potential_counts}
            for system in self.systems:
                pot_type = self.params[system]['pot_type']
                
                index_str = ""
                if potential_counts[pot_type] > 1:
                    potential_indices[pot_type] += 1
                    index_str = f" {potential_indices[pot_type]}"

                if system == '2D':
                    if lj_sheet:
                        for l in range(layer):
                            layer_index_str = ""
                            if potential_counts[pot_type] > 1:
                                layer_index_str = f" {potential_indices[pot_type] + l}"
                            
                            potentials = [
                                self.group_def[i][3] if f"2D_l{l+1}" in self.group_def[i][0] else "NULL"
                                for i in range(1, self.ngroups[layer] + 1)
                            ]
                            f_out.write(
                                f"pair_coeff * * {pot_type}{layer_index_str} {self.potentials['2D']['path']} {' '.join(potentials)} # 2D Layer {l+1}\n")
                        if potential_counts[pot_type] > 1:
                            potential_indices[pot_type] += layer -1
                    else:
                        potentials = [
                            self.group_def[i][3] if "2D" in self.group_def[i][0] else "NULL"
                            for i in range(1, self.ngroups[layer] + 1)
                        ]
                        f_out.write(
                            f"pair_coeff * * {pot_type}{index_str} {self.potentials['2D']['path']} {' '.join(potentials)} # 2D\n")
                else:
                    potentials = [
                        self.group_def[i][2] if system in self.group_def[i][0] else "NULL"
                        for i in range(1, self.ngroups[layer] + 1)
                    ]
                    f_out.write(
                        f"pair_coeff * * {pot_type}{index_str} {self.potentials[system]['path']} {' '.join(potentials)} # {system.capitalize()}\n")

            # Write pair_coeff commands for inter-system LJ interactions.
            max_sigma = 0
            if lj_sheet:
                for t in self.data['2D']['elem_comp']:
                    for key in ('sub', 'tip'):
                        for s in self.data[key]['elem_comp']:
                            e, sigma = utilities.lj_params(t, s)
                            if key == 'sub' and sigma > max_sigma:
                                max_sigma = sigma
    
                            key_s_types = f"{self.elemgroup[key][s][0]}*{self.elemgroup[key][s][-1]}" if self.settings['thermostat']['type'] == 'langevin' else f"{self.elemgroup[key][s][0]}"
                            sheet_t_types = f"{self.elemgroup['2D'][0][t][0]}*{self.elemgroup['2D'][layer-1][t][-1]}" if len(self.elemgroup['2D'][layer-1][t]) > 1 or layer > 1 else f"{self.elemgroup['2D'][0][t][0]}"
    
                            f_out.write(
                                f"pair_coeff {key_s_types} {sheet_t_types} lj/cut {e} {sigma}\n")
    
            if layer > 1:
                index_pairs = [(i, j) for i in range(layer) for j in range(i+1, layer)]
                super().set_sheet_LJ_params(f_out, index_pairs)
    
            for s in self.data['sub']['elem_comp']:
                for t in self.data['tip']['elem_comp']:
                    e, sigma = utilities.lj_params(s, t)
    
                    sub_types = f"{self.elemgroup['sub'][s][0]}*{self.elemgroup['sub'][s][-1]}" if self.settings['thermostat']['type'] == 'langevin' else f"{self.elemgroup['sub'][s][0]}"
                    tip_types = f"{self.elemgroup['tip'][t][0]}*{self.elemgroup['tip'][t][-1]}" if self.settings['thermostat']['type'] == 'langevin' else f"{self.elemgroup['tip'][t][0]}"
    
                    f_out.write(
                        f"pair_coeff {sub_types} {tip_types} lj/cut {e} {sigma} \n")
    
        return max_sigma

    def __single_body_3layer(self, filename, system):
        """Writes a potential file for a single body with 3 regions.

        Args:
            filename (str): The name of the file to write to.
            system (str): The system for which the settings are being written.
        """
        with open(filename, 'w', encoding="utf-8") as f_out:
            arr = super().number_sequential_atoms(system)
            _ = self.__define_elemgroup_3regions(system, arr)

            super().set_masses(system, f_out)

            potentials = [self.group_def[i][2]
                          for i in range(1, self.data[system]['natype']*3+1)]

            f_out.writelines([
                f"pair_style {self.params[system]['pot_type']}\n",
                f"pair_coeff * * {self.potentials[system]['path']} {' '.join((potentials))}\n"])

    def __define_elemgroup_3regions(self, system, arr, atype=1):
        """Defines element groups for a system with fixed, thermo, and mobile regions.

        Args:
            system (str): The system being defined (e.g., 'tip', 'sub').
            arr (dict): A dictionary of atom counts for each element.
            atype (int): The starting atom type index.

        Returns:
            int: The next available atom type index.
        """
        i = 1
        for element, count in self.potentials[system]['count'].items():
            for _ in range(1, count+1):
                self.group_def.update({
                    atype:   [f"{system}_t{i}",        str(atype),   str(element), arr[system][i-1]],
                    atype+1: [f"{system}_fix_t{i}",    str(atype+1), str(element), arr[system][i-1]],
                    atype+2: [f"{system}_thermo_t{i}", str(atype+2), str(element), arr[system][i-1]]
                })
                self.elemgroup.setdefault(system, {}).setdefault(
                    element, []).extend([atype, atype+1, atype+2])
                i += 1
                atype += 3

        return atype
