import os
import re
import logging
import tempfile

from tribo_2D import model_init, utilities
from tribo_2D.settings import file as settings_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SheetvsheetSimulation(model_init.ModelInit):
    """Generates simulation cells and input files for sheet-vs-sheet simulations.

    This class builds atomic structures for two opposing 2D material sheets
    and prepares the necessary directory structure and LAMMPS scripts for
    simulation setup, execution, and post-processing.

    Attributes:
        input_file (str): Path to the input configuration file.
    """

    def __init__(self, input_file):
        """Initializes and runs the sheet-vs-sheet simulation workflow.

        Args:
            input_file (str): Path to the input configuration file.
        """
        self.input_file = input_file
        try:
            self._run_simulations()
        except Exception:
            logging.exception("A critical error occurred during the sheet-vs-sheet simulation setup.")
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

            with open(self.input_file, "r", encoding="utf-8") as f:
                base_config_str = f.read()
        except FileNotFoundError:
            logging.error(f"Input file not found: {self.input_file}")
            return
        except Exception as e:
            logging.error(f"Failed to read or parse configuration: {e}")
            return

        if materials:
            for mat in materials:
                try:
                    mat_config_str = base_config_str.replace("{mat}", mat)
                    self._run_simulations_for_sizes(config, mat_config_str)
                except Exception as e:
                    logging.error(f"Simulation failed for material {mat}: {e}")
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
            with open(materials_list, "r", encoding="utf-8") as f:
                return [line.strip() for line in f]
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
            temp_config_name = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".ini", encoding="utf-8") as temp_config_file:
                    temp_config_file.write(config_str)
                    temp_config_name = temp_config_file.name
                self.system_setup(temp_config_name)
            except Exception:
                logging.exception(f"Single simulation run failed")
                raise
            finally:
                if temp_config_name and os.path.exists(temp_config_name):
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
                raise  # Continue with the next size
            finally:
                if temp_config_name and os.path.exists(temp_config_name):
                    os.remove(temp_config_name)

    def system_setup(self, config):
        """Initializes the system and generates LAMMPS scripts.

        Args:
            config (str): Path to the temporary configuration file.
        """
        try:
            super().__init__(config, model='sheetvsheet')
            self.generate_lammps_script()
        except Exception as e:
            logging.error(f"System setup for sheet-vs-sheet failed: {e}")
            raise
        
    def generate_lammps_script(self):
        """Generates the main LAMMPS script for the sliding simulation.

        This script handles the complete simulation workflow, including:
        - Reading the pre-built atomic structures.
        - Applying inter-layer bond potentials.
        - Setting up thermostats and rigid body constraints.
        - Applying normal load and initiating sliding.
        - Defining output computes for friction and other metrics.
        """
        settings_filename = f"{self.dir}/lammps/system.in.settings"
        super().__sheet_potential(settings_filename, 4, True)

        filename = f"{self.dir}/lammps/slide.lmp"

        with open(filename, 'w', encoding="utf-8") as f_out:
            f_out.writelines([
                f"variable find index {' '.join(str(x) for x in self.params['general']['force'])}\n",
                "label force_loop\n",

                f"variable a index 0 {' '.join(str(x) for x in self.scan_angle)} 0\n",
                "label angle_loop\n",
            ])
                
            f_out.writelines(self.init(neigh=True))
            f_out.writelines([
                "comm_style       tiled\n",
                "balance 1.0 rcb\n",
                "#------------------Create Geometry------------------------\n",
                "#----------------- Define the simulation box -------------\n",
                f"region          box block {self.dim['xlo']} {self.dim['xhi']} {self.dim['ylo']} {self.dim['yhi']} -40.0 40.0 units box\n",
                f"create_box      {self.ngroups[4]} box bond/types 1 extra/bond/per/atom 100\n\n",
                f"read_data       {self.dir}/build/{self.params['2D']['mat']}_4.lmp extra/bond/per/atom 100 add append group bot\n\n",

                "#----------------- Create visualisation files ------------\n\n"
                f"include {settings_filename}\n",
                "# Create bonds\n",
                "bond_style harmonic\n",
                f"bond_coeff 1 {self.params['general']['cspring']} {self.lat_c} \n",
                f"create_bonds many layer_1 layer_2 1 {self.lat_c - 0.15} {self.lat_c + 0.15}\n",
                f"create_bonds many layer_3 layer_4 1 {self.lat_c - 0.15} {self.lat_c + 0.15}\n\n",
            ])

            if self.settings['output']['dump']['slide']:
                f_out.writelines([
                    f"dump            sys all atom {self.settings['output']['dump_frequency']['slide']} ./{self.dir}/visuals/$(v_find)nN_$(v_a)angle_{self.params['general']['scan_speed']}ms.lammpstrj\n\n",
                    "dump_modify sys append yes\n"
                ])

            f_out.writelines([
                "##########################################################\n",
                "#------------------- Apply Constraints ------------------#\n",
                "##########################################################\n\n",

                "#----------------- Apply Langevin thermostat -------------\n",
                "group center union layer_2 layer_3\n",
                f"velocity        center create {self.params['general']['temp']} 492847948\n",
                f"fix             lang center langevin {self.params['general']['temp']} {self.params['general']['temp']} $(100.0*dt) 2847563 zero yes\n\n",

                "fix             nve_all all nve\n\n",

                f"timestep        {self.settings['simulation']['timestep']}\n",
                f"thermo          {self.settings['simulation']['thermo']}\n\n",

                "compute COM_top layer_4 com\n",
                "variable comx_top equal c_COM_top[1] \n",
                "variable comy_top equal c_COM_top[2] \n",
                "variable comz_top equal c_COM_top[3] \n\n",

                "compute COM_ctop layer_3 com\n",
                "variable comx_ctop equal c_COM_ctop[1] \n",
                "variable comy_ctop equal c_COM_ctop[2] \n",
                "variable comz_ctop equal c_COM_ctop[3] \n\n",

                "compute COM_cbot layer_2 com\n",
                "variable comx_cbot equal c_COM_cbot[1] \n",
                "variable comy_cbot equal c_COM_cbot[2] \n",
                "variable comz_cbot equal c_COM_cbot[3] \n\n",

                "fix             fstage_top layer_4 rigid single force * on on off torque * off off off\n",
                "fix             fsbot layer_1 setforce 0.0 0.0 0.0 \n",
                "velocity        layer_1 set 0.0 0.0 0.0 units box\n\n",

                "run 1000\n\n",
                
                f"if '$(v_a) != 0' then &\n",
                "'variable omega equal v_a/10000' &\n",
                "'fix rot layer_4 move rotate ${comx_top} ${comy_top} ${comz_top} 0 0 1 ${omega}' &\n\n",
                "'run             10000' &\n\n",
                "'unfix rot'\n\n",

                "unfix fstage_top\n",

                "fix             fstage_top layer_4 rigid single force * off off on torque * off off off\n\n",

                "variable        Ftotal          equal -v_find/1.602176565\n",
                f"variable Fatom equal v_Ftotal/(count(layer_4)\n",
                "fix force layer_4 aveforce 0.0 0.0 ${Fatom}\n\n",

                "run             10000\n\n",

                "variable        fx   equal  f_force[1]*1.602176565\n",
                "variable        fy   equal  f_force[2]*1.602176565\n",
                "variable        fz   equal  f_force[3]*1.602176565\n\n",

                "#----------------- Output values -------------------------\n",
                f"fix             fc_ave all ave/time 1 1000 {self.settings['output']['results_frequency']} v_fx v_fy v_fz v_comx_ctop v_comy_ctop v_comz_ctop v_comx_cbot v_comy_cbot v_comz_cbot file ./{self.dir}/results/fc_ave_slide_$(v_find)nN_$(v_a)angle_{self.params['tip']['s']}ms\n\n",

                f"velocity        layer_4 set 0.0 {self.params['general']['scan_speed']} 0.0 0.0 \n",
                f"run             {self.settings['simulation']['slide_run_steps']}\n\n",

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

