"""
AFM simulation module for generating simulation cells and input files
for the Prandtl-Tomlinson model in LAMMPS.

This module handles configuration parsing, structure building
(2D material, substrate, tip), potential assignment, and directory
setup for AFM simulations.
"""

from lammps import lammps

from tribo_2D import settings, model_init, utilities


class AFMSimulation(model_init.ModelInit):
    """
    A class to generate a simulation cell for the Prandtl-Tomlinson
    model in LAMMPS simulations.

    This class reads configuration data, materials, and potential files,
    builds the necessary atomic structures (e.g., 2D material, substrate, tip),
    and prepares the directory structure and scripts needed for
    simulation setup, execution, and post-processing.

    """

    def __init__(self, input_file):
        """
        Initialize the model with the parent class model_init

        Args:
            input_file (str): Path to the input configuration file.
        """

        super().__init__(input_file, model='afm')

        for system in self.systems:
            if system not in self.params:
                raise ValueError(
                    f"Missing required parameter: {system} in input file.")

        for key in ('tip', 'sub'):
            self.set_three_regions(key)

    def generate_system_init_script(self):
        """
        Generate LAMMPS input files for the AFM system setup.
        This entails creating the simulation box, reading data files,
        and applying potentials for each layer of the 2D material.

        During the simulations, the system is equilibrated.
        The tip, positioned in the center of the box,
        is indented into the 2D material at self.ious loads.
        Output files are generated as well as visualisation files
        in preperation for sliding simulations.

        """
        for layer in self.params['2D']['layers']:
            self.elemgroup = {}
            self.group_def = {}
            tip_x = self.dim['xhi'] / 2
            tip_y = self.dim['yhi'] / 2
            tip_z = 55 + self.lat_c * (layer - 1) / 2
            filename = f"{self.sheet_dir[layer]}/lammps/system.lmp"

            # --- Find gap between the 2D material and the substrate ---
            gap = self.afm_potentials_setup(layer)
            height_2d = self.params['sub']['thickness'] + 0.5 + gap

            # --- Create the system input file ---
            lammps_lines = [
                "comm_style tiled\n",
                f"region box block {self.dim['xlo']} {self.dim['xhi']} {self.dim['ylo']} {self.dim['yhi']} -5 100\n",
                f"create_box      {self.ngroups[layer]} box\n\n",

                "#----------------- Read data files -----------------------\n\n",
                f"read_data       {self.dir}/build/sub.lmp add append group sub\n",
                f"read_data       {self.dir}/build/tip.lmp add append shift {tip_x} {tip_y} {tip_z}  group tip offset {self.data['sub']['natype']*3} 0 0 0 0\n",
                f"read_data       {self.dir}/build/{self.params['2D']['mat']}_{layer}.lmp add append shift 0.0 0.0 {height_2d} group 2D offset {self.data['tip']['natype']*3+self.data['sub']['natype']*3} 0 0 0 0\n\n"

                "# Apply potentials\n\n",
                f"include        {self.sheet_dir[layer]}/lammps/system.in.settings\n\n",
                "balance 1.0 rcb\n",
                "#----------------- Create visualisation files ------------\n\n",
                f"dump            sys all atom 10000 ./{self.dir}/visuals/system_{layer}.lammpstrj\n\n",

                "#----------------- Minimize the system -------------------\n\n",
                "min_style       cg\n",
                "minimize        1.0e-4 1.0e-8 100 1000\n\n",
                "timestep        0.001\n",
                "thermo          100\n\n",

                "# ----------------- Apply Nose-Hoover thermostat ----------\n\n",
                "group           fixset union sub_fix tip_all\n",
                "group           system subtract all fixset\n\n",
                f"velocity        system create {self.params['general']['temproom']} 492847948\n\n",
                "compute         temp_tip tip_thermo temp/partial 0 1 0\n",
                f"fix             lang_tip tip_thermo langevin {self.params['general']['temproom']} {self.params['general']['temproom']} $(100.0*dt) 699483 zero yes\n",
                "fix_modify      lang_tip temp temp_tip\n\n",
                "compute         temp_sub sub_thermo temp/partial 0 1 0\n",
                f"fix             lang_sub sub_thermo langevin {self.params['general']['temproom']} {self.params['general']['temproom']} $(100.0*dt) 2847563 zero yes\n",
                "fix_modify      lang_sub temp temp_sub\n\n",
                "fix             nve_all all nve\n\n",
                "fix             sub_fix sub_fix setforce 0.0 0.0 0.0 \n",
                "velocity        sub_fix set 0.0 0.0 0.0\n\n",
                "fix             tip_f tip_all rigid/nve single force * off off off torque * off off off\n\n",
                "run             10000\n\n",
                "unfix           tip_f \n\n",
                "##########################################################\n",
                "#--------------------Tip Indentation---------------------#\n",
                "##########################################################\n",
                "#----------------- Displace tip closer -------------------\n\n",
                "displace_atoms  tip_all move 0.0 0.0 -20.0 units box\n\n",
                "#----------------- Apply constraints ---------------------\n\n",

                "fix             tip_f tip_all rigid/nve single force * off off on torque * off off off\n\n",
                "variable        f equal 0.0\n",

                f"variable find index {' '.join(str(x) for x in self.params['general']['force'])}\n",
                "label force_loop\n",
                "balance 1.0 rcb\n",
                "#----------------- Set up initial parameters -------------\n\n",
                "variable        num_floads equal 100\n",
                "variable        r equal 0.0\n",
                "variable        fincr equal (${find}-${f})/${num_floads}\n",
                "thermo_modify   lost ignore flush yes\n\n",
                "#----------------- Apply pressure to the tip -------------\n\n",
                "variable i loop ${num_floads}\n",
                "label loop_load\n\n",
                "variable f equal ${f}+${fincr} \n\n",
                "# Set force variable\n\n",
                "variable Fatom equal -v_f/(count(tip_fix)*1.602176565)\n",
                "fix forcetip tip_fix aveforce 0.0 0.0 ${Fatom}\n",
                "run 100 \n\n",
                "unfix forcetip\n\n",
                "next i\n",
                "jump SELF loop_load\n\n",
                "##########################################################\n",
                "#---------------------Equilibration----------------------#\n",
                "##########################################################\n\n",
                "fix forcetip tip_fix aveforce 0.0 0.0 ${Fatom}\n",
                "variable        dispz equal xcm(tip_fix,z)\n\n",
                "run 100 pre yes post no\n\n",
                "# Prepare to loop for displacement checks\n\n",
                "label check_r\n\n",
                "variable disp_l equal ${dispz}\n",
                "variable disp_h equal ${dispz}\n\n",
                "variable disploop loop 50\n",
                "label disp\n\n",
                "run 100 pre no post no\n\n",
                "if '${dispz}>${disp_h}' then 'variable disp_h equal ${dispz}'\n",
                "if '${dispz}<${disp_l}' then 'variable disp_l equal ${dispz}'\n\n",
                "next disploop\n",
                "jump SELF disp\n\n",
                "variable r equal ${disp_h}-${disp_l}\n\n",
                "# Check if r is less than 0.1\n\n",
                "if '${r} < 0.2' then 'jump SELF loop_end' else 'jump SELF check_r'\n\n",
                "# End of the loop\n\n",
                "label loop_end\n\n",
                f"write_data {self.sheet_dir[layer]}/data/load_$(v_find)N.data\n",
                "next find\n",
                "jump SELF force_loop"
            ]

            with open(filename, 'w', encoding="utf-8") as f:
                settings.file.init(f)
                f.writelines(lammps_lines)

    def generate_slide_script(self):
        """
        Generate LAMMPS input files for the AFM sliding setup.
        This reads the output file generated during the indentation
        and applies lateral motion to the tip while maintaining the 
        normal load to generate friction. 

        Output data is collected in a .txt file. 

        """
        # 1.602176565 nN = 1 eV/Angstrom
        # 1 Angstrom = 10^(-10) m
        # 1 ps = 10^ (-12) s

        # Convert spring constant to eV/A^2
        spring_ev = self.params['tip']['cspring'] / 16.02176565
        # Spring Damper to eV/(A^2/ps)
        damp_ev = self.params['tip']['dspring'] / 0.01602176565
        # Tip speed to Angstrom/ps
        tipps = self.params['tip']['s']/100

        for layer in self.params['2D']['layers']:

            filename = f"{self.sheet_dir[layer]}/lammps/slide_{self.params['tip']['s']}ms.lmp"
            with open(filename, 'w', encoding="utf-8") as f:
                f.writelines([
                    f"variable find index {' '.join(str(x) for x in self.params['general']['force'])}\n",
                    "label force_loop\n",

                    f"variable a index 0 {' '.join(str(x) for x in self.scan_angle)} 0\n",
                    "label angle_loop\n",
                ])
                settings.file.init(f)

                f.writelines([
                    "comm_style       tiled\n",
                    f"read_data       {self.sheet_dir[layer]}/data/load_$(v_find)N.data # Read system data\n\n",
                    f"include         {self.sheet_dir[layer]}/lammps/system.in.settings\n\n",
                    "balance 1.0 rcb\n",

                    "#----------------- Create visualisation files ------------\n\n",
                    f"dump            sys all atom 10000 ./{self.dir}/visuals/slide_{self.params['tip']['s']}ms_l{layer}.lammpstrj\n\n"
                    "dump_modify sys append yes\n",

                    "##########################################################\n",
                    "#--------------------Tip Indentation---------------------#\n",
                    "##########################################################\n",
                    "#----------------- Apply constraints ---------------------\n\n",

                    "fix             sub_fix sub_fix setforce 0.0 0.0 0.0 \n",
                    "fix             tip_f tip_all rigid/nve single force * on on on torque * off off off\n\n",

                    "#----------------- Apply Langevin thermostat -------------\n\n",
                    "compute         temp_tip tip_thermo temp/partial 0 1 0\n",
                    f"fix             lang_tip tip_thermo langevin {self.params['general']['temproom']} {self.params['general']['temproom']} $(100.0*dt) 699483 zero yes\n",
                    "fix_modify      lang_tip temp temp_tip\n\n",
                    "compute         temp_base sub_thermo temp/partial 0 1 0\n",
                    f"fix             lang_bot sub_thermo langevin {self.params['general']['temproom']} {self.params['general']['temproom']} $(100.0*dt) 2847563 zero yes\n",
                    "fix_modify      lang_bot temp temp_base\n\n",
                    "fix             nve_all all nve\n",

                    "timestep        0.001\n",
                    "thermo          100\n\n",

                    "#----------------- Apply pressure to the tip -------------\n\n",
                    "variable        Ftotal          equal -v_find/1.602176565\n",
                    "variable        Fatom           equal v_Ftotal/count(tip_fix)\n",
                    "fix             forcetip tip_fix aveforce 0.0 0.0 ${Fatom}\n\n",

                    "##########################################################\n",
                    "#------------------------Compute-------------------------#\n",
                    "##########################################################\n\n",

                    f"compute COM_top layer_{layer} com\n",
                    "variable comx equal c_COM_top[1] \n",
                    "variable comy equal c_COM_top[2] \n",
                    "variable comz equal c_COM_top[3] \n\n",

                    "compute COM_tip tip_fix com\n",
                    "variable comx_tip equal c_COM_tip[1] \n",
                    "variable comy_tip equal c_COM_tip[2] \n",
                    "variable comz_tip equal c_COM_tip[3] \n\n",
                    "#----------------- Calculate total friction --------------\n\n",
                    "variable        fz_tip   equal  f_forcetip[3]*1.602176565\n\n",
                    "variable        fx_spr   equal  f_spr[1]*1.602176565\n\n",
                    "variable        fy_spr   equal f_spr[2]*1.602176565\n\n",
                    f"fix             fc_ave all ave/time 1 1000 1000 v_fz_tip v_fx_spr v_fy_spr v_comx v_comy v_comz v_comx_tip v_comy_tip v_comz_tip file ./{self.dir}/results/fc_ave_slide_$(v_find)nN_$(v_a)angle_{self.params['tip']['s']}ms_l{layer}\n\n",

                    "##########################################################\n",
                    "#---------------------Spring Loading---------------------#\n",
                    "##########################################################\n\n",
                    "#----------------- Add damping force ---------------------\n\n",
                    f"fix             damp tip_fix viscous {damp_ev}\n\n",

                    "variable spring_x equal cos(v_a*PI/180)\n",
                    "variable spring_y equal sin(v_a*PI/180)\n\n",
                    "#------------------Add lateral harmonic spring------------\n\n",
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
        if system == 'tip':
            h = self.tipx / 2.25
            f_zlo, f_zhi = h - 3, 'INF'
            t_zlo, t_zhi = h - 5, h - 3
        elif system == 'sub':
            f_zlo, f_zhi = 'INF', 0.2*self.params['sub']['thickness']
            t_zlo, t_zhi = f_zhi, 0.5*self.params['sub']['thickness']
        dim = utilities.get_model_dimensions(f'{self.dir}/build/{system}.lmp')
        potential_file = f'{self.dir}/build/{system}_3layers.in.settings'
        self.__single_body_3layer(potential_file, system)
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
        lmp.commands_list([
            "boundary p p p\n",
            "units metal\n",
            "atom_style      atomic\n",
            f"region box block {dim['xlo']} {dim['xhi']} {dim['ylo']} {dim['yhi']} -5 100\n",
            f"create_box      {self.data[system]['natype']*3} box\n\n",
            f"read_data       {self.dir}/build/{system}.lmp add append\n",
            f"include         {potential_file}\n\n",
            f"\n#Identify the top atoms of {system}\n\n",
            f"region          {system}_fix block INF INF INF INF {f_zlo} {f_zhi} units box\n",
            f"group           {system}_fix region {system}_fix\n\n",
            f"#Identify thermostat region of {system}\n\n",
            f"region          {system}_thermo block INF INF INF INF {t_zlo} {t_zhi} units box\n",
            f"group           {system}_thermo region {system}_thermo\n\n",
        ])

        # Assign atom types and groups for tip, fix, and thermostat regions
        for t in range(self.data[system]['natype']):
            t += 1
            lmp.command(f"group {system}_{t} type {t}\n")

        i = 1
        for t in range(self.data[system]['natype']):
            t += 1
            lmp.commands_list([
                "# Set atom types\n",
                f"set group {system}_{t} type {i}\n",

                f"group {system}_fix_{t} intersect {system}_fix {system}_{t}\n",
                f"set group {system}_fix_{t} type {i+1}\n",
                f"group {system}_fix_{t} delete\n\n",

                f"group {system}_thermo_{t} intersect {system}_thermo {system}_{t}\n",
                f"set group {system}_thermo_{t} type {i+2}\n",
                f"group {system}_thermo_{t} delete\n\n",

                f"group {system}_{t} delete\n\n"
            ])
            i += 3

        lmp.commands_list([f"write_data      {self.dir}/build/{system}.lmp"])

        lmp.close

    def afm_potentials_setup(self, layer):
        """Writes the settings for applying potentials in LAMMPS to a specified file for Prandtl-Tomlinson simulations.
        Args:
        layer (int): The number of layers in the 2D system. 
        self. (dict): A dictionary containing the potential and data information for the simulation.
        """
        filename = f"{self.dir}/l_{layer}/lammps/system.in.settings"
        with open(filename, 'w') as f:
            atype = 1

            for system in self.systems:
                arr = super().number_sequential_atoms(system)
                if system == '2D':
                    atype = super().define_elemgroup(system, arr, layer=layer, atype=atype)
                else:
                    atype = self.__define_elemgroup_3regions(
                        system, arr, atype=atype)

            for system in self.systems:
                super().set_masses(system, f, layer=layer)

            for system in self.systems:
                all = [self.group_def[i][1] for i in range(
                    1, self.ngroups[layer]+1) if system in self.group_def[i][0]]
                f.write(f"group {system}_all type {' '.join(all)}\n")
                if system == '2D':
                    for l in range(layer):
                        layer_g = [self.group_def[i][1] for i in range(
                            1, self.ngroups[layer]+1) if "2D_l"+str(l+1) in self.group_def[i][0]]
                        f.write(
                            f"group layer_{l+1} type {' '.join(layer_g)}\n")
                else:
                    for n in ["_fix", "_thermo"]:
                        sub_group = [self.group_def[i][1] for i in range(
                            1, self.ngroups[layer]+1) if system+n in self.group_def[i][0]]
                        f.write(
                            f"group {system}{n} type {' '.join(sub_group)}\n")

            f.writelines(["group mobile union tip_thermo sub_thermo\n",
                          f"pair_style hybrid {self.params['sub']['pot_type']} {self.params['tip']['pot_type']} {(self.params['2D']['pot_type'] + ' ') * layer} lj/cut 8.0\n"])

            potentials = {}
            t = 0
            for system in self.systems:
                t += 1
                if system == '2D':
                    for l in range(layer):
                        potentials[l] = [
                            self.group_def[i][3] if "2D_l" +
                            str(l+1) in self.group_def[i][0] else "NULL"
                            for i in range(1, self.ngroups[layer]+1)
                        ]

                        f.write(
                            f"pair_coeff * * {self.params['2D']['pot_type']} {t+l} {self.potentials['2D']['path']} {'  '.join(potentials[l])} # interlayer '2D' Layer {l+1}\n")
                else:
                    potentials[system] = [self.group_def[i][2] if system in self.group_def[i][0] else "NULL"
                                          for i in range(1, self.ngroups[layer]+1)]
                    f.write(
                        f"pair_coeff * * {self.params[system]['pot_type']} {t} {self.potentials[system]['path']} {'  '.join(potentials[system])} # interlayer {system.capitalize()}\n")

            max_sigma = 0
            for t in self.data['2D']['elem_comp']:
                for key in ('sub', 'tip'):
                    for s in self.data[key]['elem_comp']:
                        e, sigma = utilities.LJparams(t, s)
                        if key == 'sub' and sigma > max_sigma:
                            max_sigma = sigma
                        if len(self.elemgroup['2D'][layer-1][t]) == 1 and layer == 1:
                            f.write(
                                f"pair_coeff {self.elemgroup[key][s][0]}*{self.elemgroup[key][s][-1]} {self.elemgroup['2D'][0][t][0]} lj/cut {e} {sigma}\n")
                        else:
                            f.write(
                                f"pair_coeff {self.elemgroup[key][s][0]}*{self.elemgroup[key][s][-1]} {self.elemgroup['2D'][0][t][0]}*{self.elemgroup['2D'][layer-1][t][-1]} lj/cut {e} {sigma}\n")
            if layer > 1:
                index_pairs = [(i, j) for i in range(layer)
                               for j in range(i+1, layer)]
                super().set_sheet_LJ_params(f, index_pairs)

            for s in self.data['sub']['elem_comp']:
                for t in self.data['tip']['elem_comp']:
                    e, sigma = utilities.LJparams(s, t)
                    f.write(
                        f"pair_coeff {self.elemgroup['sub'][t][0]}*{self.elemgroup['sub'][t][-1]} {self.elemgroup['tip'][t][0]}*{self.elemgroup['tip'][t][-1]}  lj/cut {e} {sigma} \n")

        return max_sigma

    def __single_body_3layer(self, filename, system):
        """
        Writes the settings for applying potentials in LAMMPS to a system with 3 sections, fixed, thermal, and rest of body.
        Args:
        filename (str): The name of the file to write to.
        system (str): The system for which the settings are being written.
        self. (dict): A dictionary containing the potential and data information for the simulation.
        """
        potentials = {}
        with open(filename, 'w') as f:

            arr = super().number_sequential_atoms(system)
            _ = self.__define_elemgroup_3regions(system, arr)

            super().set_masses(system, f)

            potentials = [self.group_def[i][2]
                          for i in range(1, self.data[system]['natype']*3+1)]

            f.writelines([
                f"pair_style {self.params[system]['pot_type']}\n",
                f"pair_coeff * * {self.potentials[system]['path']} {' '.join((potentials))}\n"])

    def __define_elemgroup_3regions(self, system, arr, atype=1):
        """
        Defines the element groups for a system with multiple layers.
        Args:
        system (str): The system for which the groups are being defined.
        arr (dict): A dictionary containing the atom counts for each system.
        l (int): The layer number.
        atype (int): The starting atom type index.
        Returns:
        atype (int): The next available atom type index.
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
