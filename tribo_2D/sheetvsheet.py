from ase import data
from lammps import lammps

from tribo_2D import settings, model_init, utilities

class Sheetvsheet_simulation(model_init.model_init):
    def __init__(self, input_file):
        
        super().__init__(input_file, model='sheetvsheet')

    def sheet_system(self):
                    
        settings_filename = f"{self.var['dir']}/lammps/system.in.settings"
        build.sheet_potential(self, settings_filename,4) 
        print(self.lat_c)
        for force in self.var['general']['force']:
            if force == self.var['general']['scan_angle'][3]:
                dump = False
                scan_angle = np.arange(self.var['general']['scan_angle'][0],self.var['general']['scan_angle'][1]+1,self.var['general']['scan_angle'][2])
            else:
                dump = False
                scan_angle = [0]
                if force in self.dump_load:
                    dump = True
                
            for a in scan_angle:
                if a in self.dump_slide:
                    dump = True
                
                filename = f"{self.var['dir']}/lammps/{force}nN_{a}angle_{self.var['general']['scan_s']}ms.lmp"

                with open(f"{self.var['dir']}/scripts/sheetvsheet", 'a') as f:
                    f.write(f"{filename}\n")

                with open(filename, 'w') as f:
                    init_mol(f)
                    f.writelines([
                        "#------------------Create Geometry------------------------\n",
                        "#----------------- Define the simulation box -------------\n",
                        f"region          box block {self.var['dim']['xlo']} {self.var['dim']['xhi']} {self.var['dim']['ylo']} {self.var['dim']['yhi']} -40.0 40.0 units box\n",
                        f"create_box      {self.var['ngroups']} box bond/types 1 extra/bond/per/atom 100\n\n",

                        f"read_data       {self.file} extra/bond/per/atom 100 add append group bot\n\n",

                    "#----------------- Create visualisation files ------------\n\n"
                    ])

                    if dump == True:
                        f.write(f"dump            sys all atom 1000 ./{self.var['dir']}/visuals/load_{force}N_{a}angle_{self.var['general']['scan_s']}ms.lammpstrj\n\n",)
                    
                    
                    f.writelines([
                        f"include {settings_filename}\n",
                        "# Create bonds\n",
                        "bond_style harmonic\n",
                        f"bond_coeff 1 {self.var['general']['cspring']} {self.lat_c} \n",
                        f"create_bonds many layer_1 layer_2 1 {self.lat_c-0.15} {self.lat_c+0.15}\n",
                        f"create_bonds many layer_3 layer_4 1 {self.lat_c-0.15} {self.lat_c+0.15}\n\n",

                        "##########################################################\n",
                        "#------------------- Apply Constraints ------------------#\n",
                        "##########################################################\n\n",


                        "#----------------- Apply Langevin thermostat -------------\n",
                        "group center union layer_2 layer_3\n",
                        f"velocity        center create {self.var['general']['temproom']} 492847948\n",
                        f"fix             lang center langevin {self.var['general']['temproom']} {self.var['general']['temproom']} $(100.0*dt) 2847563 zero yes\n\n",

                        "fix             nve_all all nve\n\n",

                        "timestep        0.001\n",
                        "thermo          100\n\n",

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
                        ])
                    
                    if a != 0:
                        f.writelines([
                            f"variable omega equal {a}/10000\n",
                            "fix rot layer_4 move rotate ${comx_top} ${comy_top} ${comz_top} 0 0 1 ${omega}\n\n",

                            "run             10000\n\n",
                            "unfix rot\n\n",
                        ])


                    f.writelines([
                        "unfix fstage_top\n",
                        

                        "fix             fstage_top layer_4 rigid single force * off off on torque * off off off\n\n",
                        
                        f"variable Fatom equal -{force}/(count(layer_4)*1.602176565)\n",
                        "fix force layer_4 aveforce 0.0 0.0 ${Fatom}\n\n",

                        "run             10000\n\n",

                        "variable        fx   equal  f_force[1]*1.602176565\n",
                        "variable        fy   equal  f_force[2]*1.602176565\n",
                        "variable        fz   equal  f_force[3]*1.602176565\n\n",



                        "#----------------- Output values -------------------------\n",
                        f"fix             fc_ave all ave/time 1 1000 1000 v_fx v_fy v_fz v_comx_ctop v_comy_ctop v_comz_ctop v_comx_cbot v_comy_cbot v_comz_cbot file {self.var['dir']}/data/{force}nN_{a}angle_{self.var['general']['scan_s']}ms\n\n",

                        f"velocity        layer_4 set 0.0{self.var['general']['scan_s']} 0.0 0.0 \n",
                        "run             100000\n\n",

                        "##########################################################\n",
                        "#-----------------------Write Data-----------------------#\n",
                        "##########################################################\n\n",

                        "#----------------- Save final configuration in data file -\n",
                        f"write_data     {self.var['dir']}/data/{force}nN_{a}angle_{self.var['general']['scan_s']}ms.data\n"
                    ])
    def sheet_pbs(self):

        filename = f"{self.var['dir']}/scripts/sheetvsheet.pbs"
        PBS = '"${PBS_ARRAY_INDEX}p"'
        PBS_log = "{PBS_ARRAY_INDEX}"
        with open(f"{self.scripts}/sheetvsheet", 'r' ) as f:
            n = len(f.readlines())
        with open(filename,'w') as f: 
            f.writelines([
                "#!/bin/bash\n",
                "#PBS -l select=1:ncpus=32:mem=62gb:mpiprocs=32:cpu_type=rome\n",
                "#PBS -l walltime=08:00:00\n",
                f"#PBS -J 1-{n}\n",
                f"#PBS -o /rds/general/user/mv923/home/logs_{self.var['2D']['mat']}/\n",
                f"#PBS -e /rds/general/user/mv923/home/logs_{self.var['2D']['mat']}/\n\n",

                "module purge\n",
                "module load tools/dev\n",
                "module load LAMMPS/23Jun2022-foss-2021b-kokkos\n",
                "#module load OpenMPI/4.1.4-GCC-11.3.0\n\n",

                "#Go to the temp directory (ephemeral) and create a new folder for this run\n",
                "cd $EPHEMERAL\n\n",


                "# $PBS_O_WORKDIR is the directory where the pbs script was sent from. Copy everything from the work directory to the temporary directory to prepare for the run\n\n",

                f"mpiexec lmp -l none -in $(sed -n {PBS} {self.var['dir']}/scripts/sheetvsheet)\n\n",
            ])

        filename = f"{self.scripts}/{self.var['2D']['mat']}_transfer.pbs"
        with open(filename,'w') as f: 
            f.writelines([ 
                "#!/bin/bash\n",
                "#PBS -l select=1:ncpus=1:mem=62gb:cpu_type=rome\n",
                "#PBS -l walltime=00:30:00\n\n",
                f"#PBS -o /rds/general/user/mv923/home/scripts/{self.var['2D']['mat']}/\n",
                f"#PBS -e /rds/general/user/mv923/home/scripts/{self.var['2D']['mat']}/\n\n",

                "cd $HOME\n",
                f"mkdir -p logs_{self.var['2D']['mat']}/\n\n",
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
                f"#PBS -o /rds/general/user/mv923/home/logs_{self.var['2D']['mat']}/\n",
                f"#PBS -e /rds/general/user/mv923/home/logs_{self.var['2D']['mat']}/\n\n",

                "cd $EPHEMERAL\n",
                "#After the end of the run copy everything back to the parent directory\n",
                f"cp -r ./{self.var['dir']}/* $PBS_O_WORKDIR/{self.var['dir']}\n\n",
                f"rm -r ./scripts/{self.var['2D']['mat']}\n\n"
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
                f"qsub -W depend=afterok:XXXX.pbs {self.scripts}/sheetvsheet.pbs\n\n",

                "# Where XXXX.pbs is the job number given to you after submitting transfer.pbs\n\n",
                
                "# Transfer your results back to the home directory with:\n",
                f"qsub -W depend=afterany:XXXX[].pbs {self.scripts}/{self.var['2D']['mat']}_transfer2.pbs\n\n",

                "# Where XXXX[].pbs is the job number given to you after submitting slide.pbs\n\n",

                "# Make sure to transfer your results and visuals back to your personal computer\n"
            ])
