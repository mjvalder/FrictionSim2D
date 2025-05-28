from .tools import *
from .build import *
from .pot import *
from .settings.file import *
from .Potentials import *
import numpy as np
from pathlib import Path


class afm:
    def __init__(self,input):
        
        # Generate data and variables

        var = read_config(input) # Read input file settings
        
        
        self.group = ['2D' ,'sub', 'tip']

        var['data'] = {mat: cifread(var[mat]['cif_path']) for mat in self.group}  # Read materials  
        var['pot'] = {mat: count_elemtypes(var[mat]['pot_path']) for mat in self.group}  # Count potentials  
        
        for mat in self.group:
            if mat == '2D':
                var['data'][mat].update({
                    'natype':sum(var['pot'][mat].values())
                })
            else:
                var['data'][mat].update({
                    'natype':sum(var['pot'][mat].values())*3
                })
        # Build one layer of 2D material to generate important variables
        var['dir'] = f"{var['2D']['mat']}/afm/{var['2D']['x']}x_{var['2D']['y']}y/sub_{var['sub']['amorph']}{var['sub']['mat']}_tip_{var['tip']['amorph']}{var['tip']['mat']}_r{var['tip']['r']}/K{var['general']['temproom']}"

        # Create file locations

        self.scripts = var['dir'] +"/scripts"

        dirs = ["visuals", "results", "system_build", "potentials"]

        for d in dirs:
            Path(var['dir'], d).mkdir(parents=True, exist_ok=True)
            
        var['pot']['path'] = {mat: copy_file(var[mat]['pot_path'],f"{var['dir']}/potentials/") for mat in self.group}
        print(var['pot']['path'])
        self.var = sheet(var)

        #Expand to multiple layers if required

        self.var['ngroups'] = {}
        self.directory = {}
        for l in self.var['2D']['layers']:
            self.directory[l] = Path(self.var['dir']) / f"l_{l}"

            for sub in ["data", "lammps"]:
                (self.directory[l] / sub).mkdir(parents=True, exist_ok=True)
            if l > 1:
                _ = stacking(self.var,l)
            self.var['ngroups'][l] = self.var['data']['2D']['natype']*l + self.var['data']['sub']['natype'] + self.var['data']['tip']['natype']
            


        self.scan_angle = np.arange(self.var['general']['scan_angle'][0],self.var['general']['scan_angle'][1]+1,self.var['general']['scan_angle'][2])

        # Limit the number of visual files generated
        self.dump_load = [self.var['general']['force'][i] for i in range(4, len(self.var['general']['force']), 5)]
        self.dump_slide = [self.scan_angle[i] for i in range(4, len(self.scan_angle), 5)]
        
        # Generate substrate and tip

        tip_build(self.var)
        sub_build(self.var)

    def system(self):
        for layer in self.var['2D']['layers']:
            h = settings_afm(self.var,layer)
            [tip_x,tip_y,tip_z] = [self.var['dim']['xhi']/2, self.var['dim']['yhi']/2, 55+self.var['2D']['lat_c']*(layer-1)+2*h]# Tip placement
            

            filename = f"{self.directory[layer]}/lammps/system.lmp"
            
            h_2D        = 12.5 + h
            with open(filename, 'w') as f:
                init(f)
                f.writelines([
                    f"region box block {self.var['dim']['xlo']} {self.var['dim']['xhi']} {self.var['dim']['ylo']} {self.var['dim']['yhi']} -5 100\n",
                    f"create_box      {self.var['ngroups'][layer]} box\n\n",
                    "#----------------- Read data files -----------------------\n\n",
                    f"read_data       {self.var['dir']}/system_build/sub.lmp add append group sub\n",
                    f"read_data       {self.var['dir']}/system_build/tip.lmp add append shift {tip_x} {tip_y} {tip_z}  group tip offset {self.var['data']['sub']['natype']} 0 0 0 0\n",
                    f"read_data       {self.var['dir']}/system_build/{self.var['2D']['mat']}_{layer}.lmp add append shift 0.0 0.0 {h_2D} group 2D offset {self.var['data']['tip']['natype']+self.var['data']['sub']['natype']} 0 0 0 0\n\n"
                ])
                #generate potentials

                f.writelines([
                "# Apply potentials\n\n",
                f"include        {self.directory[layer]}/lammps/system.in.settings\n\n",
                "#----------------- Create visualisation files ------------\n\n",
                f"dump            sys all atom 10000 ./{self.var['dir']}/visuals/system_{layer}.lammpstrj\n\n",
                "#----------------- Minimize the system -------------------\n\n",
                "min_style       cg\n",
                "minimize        1.0e-4 1.0e-8 100 1000\n\n",
                "timestep        0.001\n",
                "thermo          100\n\n",
                #----------------- Apply Nose-Hoover thermostat ----------
                "group           fixset union sub_fix tip_all\n",
                "group           system subtract all fixset\n\n",
                f"velocity        system create {self.var['general']['temproom']} 492847948\n\n",
                "compute         temp_tip tip_thermo temp/partial 0 1 0\n",
                f"fix             lang_tip tip_thermo langevin {self.var['general']['temproom']} {self.var['general']['temproom']} $(100.0*dt) 699483 zero yes\n",
                "fix_modify      lang_tip temp temp_tip\n\n",
                "compute         temp_sub sub_thermo temp/partial 0 1 0\n",
                f"fix             lang_sub sub_thermo langevin {self.var['general']['temproom']} {self.var['general']['temproom']} $(100.0*dt) 2847563 zero yes\n",
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
                "displace_atoms  tip_all move 0.0 0.0 -40.0 units box\n\n",
                "#----------------- Apply constraints ---------------------\n\n",

                "fix             tip_f tip_all rigid/nve single force * off off on torque * off off off\n\n",
                "variable        f equal 0.0\n",

                f"variable find index {' '.join(str(x) for x in self.var['general']['force'])}\n",
                "label force_loop\n",

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
                f"write_data {self.directory[layer]}/data/load_$(v_find)N.data\n",
                "next find\n",
                "jump SELF force_loop"
                ])

    def slide(self):
        # 1.602176565x10^(-9) N = 1 eV/Angstrom and 1 Angstrom = 10^(-10) m
        springeV  = self.var['tip']['cspring']/16.02176565 # Spring constant in eV/A^2
        # 1.602176565 nN = 1 eV/Angstrom and 1 Angstrom = 10^(-10) m and 1 ps = 10^ (-12) s
        DspringeV  = self.var['tip']['dspring']/0.01602176565 # Spring damper in ev/(A^2/ps)
        tipps = self.var['tip']['s']/100 # in A/ps

        for layer in self.var['2D']['layers']:

            filename = f"{self.directory[layer]}/lammps/slide_{self.var['tip']['s'] }ms.lmp"
            with open(filename, 'w') as f:
                # init(f)
                f.writelines([

                f"variable find index {' '.join(str(x) for x in self.var['general']['force'])}\n",
                "label force_loop\n",
                
                f"variable a index 0 {' '.join(str(x) for x in self.scan_angle)} 0\n",
                "label angle_loop\n",
                ])
                init(f)


                f.writelines([
                f"read_data       {self.directory[layer]}/data/load_$(v_find)N.data # Read system data\n\n",
                f"include         {self.directory[layer]}/lammps/system.in.settings\n\n",

                "#----------------- Create visualisation files ------------\n\n",
                f"dump            sys all atom 10000 ./{self.var['dir']}/visuals/slide_{self.var['tip']['s']}ms_l{layer}.lammpstrj\n\n"
                "dump_modify sys append yes\n",
                "##########################################################\n",
                "#--------------------Tip Indentation---------------------#\n",
                "##########################################################\n",
                "#----------------- Apply constraints ---------------------\n\n",

                "fix             sub_fix sub_fix setforce 0.0 0.0 0.0 \n",
                "fix             tip_f tip_all rigid/nve single force * on on on torque * off off off\n\n",

                "#----------------- Apply Langevin thermostat -------------\n\n",
                "compute         temp_tip tip_thermo temp/partial 0 1 0\n",
                f"fix             lang_tip tip_thermo langevin {self.var['general']['temproom']} {self.var['general']['temproom']} $(100.0*dt) 699483 zero yes\n",
                "fix_modify      lang_tip temp temp_tip\n\n",
                "compute         temp_base sub_thermo temp/partial 0 1 0\n",
                f"fix             lang_bot sub_thermo langevin {self.var['general']['temproom']} {self.var['general']['temproom']} $(100.0*dt) 2847563 zero yes\n",
                "fix_modify      lang_bot temp temp_base\n\n",
                "fix             nve_all all nve\n",

                "timestep        0.001\n",
                "thermo          100\n\n",

                "#----------------- Apply pressure to the tip -------------\n\n",
                f"variable        Ftotal          equal -v_find/1.602176565\n",
                "variable        Fatom           equal v_Ftotal/count(tip_fix)\n",
                "fix             forcetip tip_fix aveforce 0.0 0.0 ${Fatom}\n\n",

                "##########################################################\n",
                "#------------------------Compute-------------------------#\n",
                "##########################################################\n\n",
			    
                f"compute COM_top layer_{layer} com\n",
                "variable comx equal c_COM_top[1] \n",
                "variable comy equal c_COM_top[2] \n",
                "variable comz equal c_COM_top[3] \n\n",

                f"compute COM_tip tip_fix com\n",
                "variable comx_tip equal c_COM_tip[1] \n",
                "variable comy_tip equal c_COM_tip[2] \n",
                "variable comz_tip equal c_COM_tip[3] \n\n",
                "#----------------- Calculate total friction --------------\n\n",
                "variable        fz_tip   equal  f_forcetip[3]*1.602176565\n\n",
                "variable        fx_spr   equal  f_spr[1]*1.602176565\n\n",
                "variable        fy_spr   equal f_spr[2]*1.602176565\n\n",
                f"fix             fc_ave all ave/time 1 1000 1000 v_fz_tip v_fx_spr v_fy_spr v_comx v_comy v_comz v_comx_tip v_comy_tip v_comz_tip file ./{self.var['dir']}/results/fc_ave_slide_$(v_find)nN_$(v_a)angle_{self.var['tip']['s']}ms_l{layer}\n\n",
                
                "##########################################################\n",
                "#---------------------Spring Loading---------------------#\n",
                "##########################################################\n\n",
                "#----------------- Add damping force ---------------------\n\n",
                f"fix             damp tip_fix viscous {DspringeV}\n\n",

                "variable spring_x equal cos(v_a*PI/180)\n",
                "variable spring_y equal sin(v_a*PI/180)\n\n",
                "#------------------Add lateral harmonic spring------------\n\n",
                f"fix             spr tip_fix smd cvel {springeV} {tipps} tether $(v_spring_x) $(v_spring_y) NULL 0.0\n\n",
                "run 200000\n\n",

                f"if '$(v_a) == {self.var['general']['scan_angle'][3]}' then &\n",
                "'next a' & \n",
                "'jump SELF find_incr'\n\n",

                f"if '$(v_find) == {self.var['general']['scan_angle'][3]}' then &\n",
                "'next a' & \n",
                "'clear' & \n",
                "'jump SELF angle_loop'\n\n",

                "label find_incr\n\n",
                "next find\n",
                "clear\n",
                "jump SELF force_loop"
                ])
    
    def pbs(self):
 
        pbs_type = ['system','slide']
        for layer in self.var['2D']['layers']:
            for type in pbs_type:
                filename = f"{self.scripts}/{self.var['2D']['mat']}_{type}.pbs"
                PBS = '"${PBS_ARRAY_INDEX}p"'
                with open(f"{self.scripts}/list_{type}_{layer}", 'r' ) as f:
                    n = len(f.readlines())
                with open(filename,'w') as f:
                    f.writelines([
                        "#!/bin/bash\n",
                        "#PBS -l select=1:ncpus=32:mem=62gb:mpiprocs=32:cpu_type=rome\n",
                        "#PBS -l walltime=08:00:00\n",
                        f"#PBS -J 1-{n}\n",
                        f"#PBS -o /rds/general/user/mv923/home/logs_{type}/\n",
                        f"#PBS -e /rds/general/user/mv923/home/logs_{type}/\n\n",

                        "module purge\n",
                        "module load tools/dev\n",
                        "module load LAMMPS/23Jun2022-foss-2021b-kokkos\n",
                        "#module load OpenMPI/4.1.4-GCC-11.3.0\n\n",

                        "#Go to the temp directory (ephemeral) and create a new folder for this run\n",
                        "cd $EPHEMERAL\n\n",


                        "# $PBS_O_WORKDIR is the directory where the pbs script was sent from. Copy everything from the work directory to the temporary directory to prepare for the run\n\n",

                        f"mpiexec lmp -l none -in $(sed -n {PBS} scripts/afm2/scripts/list_{type}_{layer})\n\n",
                    ])

        filename = f"{self.scripts}/{self.var['2D']['mat']}_transfer.pbs"
        with open(filename,'w') as f: 
            f.writelines([
                "#!/bin/bash\n",
                "#PBS -l select=1:ncpus=1:mem=62gb:cpu_type=rome\n",
                "#PBS -l walltime=00:30:00\n\n",
                f"#PBS -o /rds/general/user/mv923/home/scripts/\n",
                f"#PBS -e /rds/general/user/mv923/home/scripts/\n\n",

                "cd $HOME\n",
                f"mkdir -p logs_system/\n\n",
                f"mkdir -p logs_slide/\n\n",

                "cd $EPHEMERAL\n",
                f"mkdir -p {self.var['dir']}/\n\n",

                f"cp -r $PBS_O_WORKDIR/{self.var['dir']}/* {self.var['dir']}\n",
                "cp -r $PBS_O_WORKDIR/tribo_2D/Potentials/ .\n"
            ])

        filename = f"{self.scripts}/{self.var['2D']['mat']}_transfer2.pbs"
        with open(filename,'w') as f: 
            f.writelines([
                "#!/bin/bash\n",
                "#PBS -l select=1:ncpus=1:mem=62gb:cpu_type=rome\n",
                "#PBS -l walltime=00:30:00\n\n",
                f"#PBS -o /rds/general/user/mv923/home/scripts/\n",
                f"#PBS -e /rds/general/user/mv923/home/scripts/\n\n",

                "cd $EPHEMERAL\n",
                "#After the end of the run copy everything back to the parent directory\n",
                f"cp -r ./scripts/afm/* $PBS_O_WORKDIR/scripts/afm\n\n",
                f"rm -r ./scripts/afm\n\n"
            ])
        
        filename = f"{self.scripts}/{self.var['2D']['mat']}_instructions.txt"
        with open(filename,'w') as f: 
            f.writelines([
                f"# The first step is transferring the whole {self.var['2D']['mat']} folder to the RDS Home Directory\n",
                "# This can be done by adding the RDS Path to your file system as seen in\n",
                "# https://icl-rcs-user-guide.readthedocs.io/en/latest/rds/paths/ \n\n",

                "# Next, we need to transfer the files to the Ephemeral directory, run the following command:\n",
                f"qsub {self.scripts}/{self.var['2D']['mat']}_transfer.pbs\n\n",

                "# Once this is done, you can run the system intialisation as follows:\n",
                f"qsub -W depend=afterok:XXXX.pbs {self.scripts}/{self.var['2D']['mat']}_system.pbs\n\n",

                "# Where XXXX.pbs is the job number given to you after submitting transfer.pbs\n\n",
                
                "# Then we can run the loading as follows:\n",
                f"qsub -W depend=afterany:XXXX[].pbs {self.scripts}/{self.var['2D']['mat']}_load.pbs\n\n",
                
                "# Where XXXX[].pbs is the job number given to you after submitting system.pbs\n\n",
                
                "# Finally, we can run the sliding as follows:\n",
                f"qsub -W depend=afterany:XXXX[].pbs {self.scripts}/{self.var['2D']['mat']}_slide.pbs\n\n",

                "# Where XXXX[].pbs is the job number given to you after submitting load.pbs\n\n",
                
                "# Transfer your results back to the home directory with:\n",
                f"qsub -W depend=afterany:XXXX[].pbs {self.scripts}/{self.var['2D']['mat']}_transfer2.pbs\n\n",

                "# Where XXXX[].pbs is the job number given to you after submitting slide.pbs\n\n",

                "# Make sure to transfer your results and visuals back to your personal computer\n"
            ])
