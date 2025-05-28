from .tools import *
from .settings import *
from .pot import *
from ase import io,data
import subprocess
import os
from lammps import lammps
import numpy as np
import re
from pathlib import Path

def tip_build(var):
        
    filename = f"{var['tip']['mat']}.lmp"
    settings = f"{var['dir']}/system_build/tip.in.settings"
    settings_sb(var,settings,'tip')

    x = 2*var['tip']['r']
    z = var['tip']['r']
    side = round(2*var['tip']['r']/2.5)
    
    if var['tip']['amorph'] == 'a':
        am_filename = os.path.join(os.path.dirname(__file__), "materials", f"amor_{filename}")
        filename = os.path.join(os.path.dirname(__file__), "materials", f"{var['tip']['mat']}.lmp")

        if os.path.exists(am_filename):
            pass
        else:
            slab_generator('tip',var, 200, 200, 50)
            amorph('tip',filename, am_filename, 2500,var)
            os.remove(filename)
            
        
        filename = am_filename
        dim_x = z

    else:
        slab_generator('tip',var,x,x,z)
        filename = os.path.join(os.path.dirname(__file__), "materials", f"{var['tip']['mat']}.lmp")
        dim = get_model_dimensions(filename)
        dim_x = dim['xhi']/2
        
    h = dim_x/2.25
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
    lmp.commands_list([
        "boundary p p p\n",
        "units metal\n",
        "atom_style      atomic\n",

        f"region          afm_tip sphere 0 0 {dim_x} {dim_x}  side in units box\n",
        f"create_box      3 afm_tip\n",
        f"read_data       {filename} add append shift -{dim_x} -{dim_x} 0\n",
        f"include {settings}\n",
        f"region         box block -{dim_x} {dim_x} -{dim_x} {dim_x} -3 {h} units box\n",
        # f"region         box block -{side} {side} -{side} {side} -3 {dim_x/2.5} units box\n",
        "region tip intersect 2 afm_tip box\n",
        "group           tip region tip\n",
        "group           box subtract all tip\n",
        "delete_atoms    group box\n",
        f"change_box all x final -{dim_x} {dim_x} y final -{dim_x} {dim_x} z final -3 {h+1} \n",
        #------------------Save the final configuration to a data file
        "reset_atoms     id\n",
        "\n#Identify the top atoms of AFM tip\n\n",
        f"region          tip_fix block INF INF INF INF {h-3} INF units box\n",
        "group           tip_fix region tip_fix\n\n",
        "#Identify thermostat region of AFM tip\n\n",
        f"region          tip_thermo block INF INF INF INF {h-5} {h-3} units box\n",
        "group           tip_thermo region tip_thermo\n\n",
    ])
    
    for t in range(var['data']['tip']['natype']):
        t+=1
        lmp.command(f"group tip_{t} type {t}\n")

    i=1
    for t in range(var['data']['tip']['natype']):
        t+=1
        lmp.commands_list([
            f"set group tip_{t} type {i}\n",
            f"group tip_fix_{t} intersect tip_fix tip_{t}\n",
            f"set group tip_fix_{t} type {i+1}\n",
            f"group tip_fix_{t} delete\n\n",
            
            f"group tip_thermo_{t} intersect tip_thermo tip_{t}\n",
            f"set group tip_thermo_{t} type {i+2}\n",
            f"group tip_thermo_{t} delete\n\n"

            f"group tip_{t} delete\n\n"
        ])
        i+=3

    lmp.commands_list([f"write_data      {var['dir']}/system_build/tip.lmp"])
    
    lmp.close

def sub_build(var):
    
    filename = f"{var['sub']['mat']}.lmp"
    settings = f"{var['dir']}/system_build/sub.in.settings"
    settings_sb(var,settings,'sub')

    if var['sub']['amorph'] == 'a':
        
        am_filename = os.path.join(os.path.dirname(__file__), "materials", f"amor_{filename}")
        filename = os.path.join(os.path.dirname(__file__), "materials", f"{var['sub']['mat']}.lmp")

        if os.path.exists(am_filename):
            pass
        else:
            slab_generator('sub',var, 200, 200, 20)
            amorph('sub',filename, am_filename,2500,var)
            
            os.remove(filename)
            
        
        filename = am_filename
    
    else:
        slab_generator('sub',var,var['2D']['x']*1.2,var['2D']['y']*1.2,20)
        filename = os.path.join(os.path.dirname(__file__), "materials", f"{var['sub']['mat']}.lmp")
    
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
    lmp.commands_list([
        "boundary p p p\n",
        "units metal\n",
        "atom_style      atomic\n",
        f"region box block {var['dim']['xlo']} {var['dim']['xhi']} {var['dim']['ylo']} {var['dim']['yhi']} 0 12\n",
        f"create_box       {var['data']['sub']['natype']} box\n\n",
        f"read_data {filename} add append\n",
        f"include {settings}\n",
        "group sub region box\n",
        "group box subtract all sub\n",
        "delete_atoms group box\n",
        f"change_box all x final 0 {var['dim']['xhi']} y final 0 {var['dim']['yhi']} z final 0 12\n\n",

        "#Identify bottom atoms of Amorphous Silicon substrate\n\n",
        "region          sub_fix block INF INF INF INF INF 5 units box\n",
        "group           sub_fix region sub_fix\n\n",
        "#Identify thermostat region of Amorphous Silicon substrate\n\n",
        "region          sub_thermo block INF INF INF INF 5 8 units box\n",
        "group           sub_thermo region sub_thermo\n\n",

        "# Define sub groups and atom types\n\n"
        ])
    
    for t in range(var['data']['sub']['natype']):
        t+=1
        lmp.command(f"group sub_{t} type {t}\n")

    i=1
    for t in range(var['data']['sub']['natype']):
        t+=1
        lmp.commands_list([
            f"set group sub_{t} type {i}\n",

            f"group sub_fix_{t} intersect sub_fix sub_{t}\n",
            f"set group sub_fix_{t} type {i+1}\n",
            f"group sub_fix_{t} delete\n\n",
            
            f"group sub_thermo_{t} intersect sub_thermo sub_{t}\n",
            f"set group sub_thermo_{t} type {i+2}\n",
            f"group sub_thermo_{t} delete\n\n"

            f"group sub_{t} delete\n\n"
        ])

        i+=3

    lmp.command(f"write_data      {var['dir']}/system_build/sub.lmp")
    
    lmp.close

def amorph(system,filename,am_filename, tempmelt,var):
    
    quench = (tempmelt-var['general']['temproom'])*100
    # lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
    settings = f"{var['dir']}/system_build/{system}.in.settings"
    settings_ob(var,settings,system)
    lmp = lammps()
    lmp.commands_list([
    "boundary p p p\n",
    "units metal\n",
    "atom_style      atomic\n\n",

    f"read_data {filename}\n\n", 
    #------------------Apply potentials-----------------------
    f"include {settings}\n\n",
    ##########################################################
    #-------------------Energy Minimization------------------#
    ##########################################################
    "min_style       cg\n",
    "minimize        1.0e-4 1.0e-8 100 1000\n",
    ##########################################################
    #-------------------Equilibrate System-------------------#
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
    #----------------------Quench System---------------------#
    ##########################################################
    f"fix             quench all nvt temp {tempmelt} {var['general']['temproom']} $(100.0*dt)\n ",
    f"run             {quench}\n ",
    "unfix           quench \n",
    f"write_data {am_filename}"
    ])
    
    lmp.close

def sheet(var):

    x=var['2D']['x']
    y=var['2D']['y']

    filename = f"{var['dir']}/system_build/{var['2D']['mat']}_1.lmp"

    multiples = {}

    for element, cif_count in var['data']['2D']['elem_comp'].items():
        potential_count = var['pot']['2D'].get(element, 0)  
        if cif_count > 0 and potential_count != 1:  
            multiples[element] = potential_count / cif_count
        else:
            multiples[element] = None  
    
    first_multiple = next(iter(multiples.values()))  

    for multiple in multiples.values():
        if multiple != first_multiple:  
            if multiple is None and first_multiple == 1 or multiple == 1 and first_multiple is None:
                first_multiple=1
            else:
                raise ValueError('multiples must be the same')
    
    typecount = 0
    for atomcount in var['pot']['2D'].values():
        typecount += atomcount

    Path(filename).unlink(missing_ok=True)

    if first_multiple is None:
        atomsk_command = f"echo n | atomsk {var['2D']['cif_path']} -ow {filename} -v 0"
        subprocess.run(atomsk_command, shell=True, check=True)
        atomsk_command = f"atomsk {filename} -orthogonal-cell -ow lmp -v 0"
        subprocess.run(atomsk_command, shell=True, check=True)

    elif isinstance(first_multiple, float) and not first_multiple.is_integer():
        raise ValueError('potential file or cif file is not formatted properly')
    
    else:
        if first_multiple ==1:
            atomsk_command = f"echo n | atomsk {var['2D']['cif_path']} -ow {filename} -v 0"
            subprocess.run(atomsk_command, shell=True, check=True)
            num_atoms_lmp(filename)
            atomsk_command = f"atomsk {filename} -orthogonal-cell -ow lmp -v 0"
            subprocess.run(atomsk_command, shell=True, check=True)
            
        else:
            atomsk_command = f"echo n | atomsk {var['2D']['cif_path']} -ow {filename} -v 0"
            subprocess.run(atomsk_command, shell=True, check=True)
            num_atoms_lmp(filename)

            atomsk_command = f"echo n | atomsk {filename} -orthogonal-cell -ow lmp -v 0"
            subprocess.run(atomsk_command, shell=True, check=True)
            atoms = io.read(filename, format="lammps-data")
            # Get number of atoms
            natoms = len(atoms)
            if typecount % natoms == 0:
                for i in range(int(np.sqrt(typecount/natoms))+1, 0, -1):
                    if typecount/natoms % i == 0:
                        a = int(i)
                        b = int(typecount/natoms / i)
                        break        
            atomsk_command = f"echo n | atomsk {filename} -duplicate {a} {b} 1 -ow lmp -v 0"
            subprocess.run(atomsk_command, shell=True, check=True)
            process_h(var['pot']['2D'],filename)

    dim = get_model_dimensions(filename)
    duplicate_a = round(x / dim['xhi'])
    duplicate_b = round(y / dim['yhi'])

    atomsk_command = f"atomsk {filename} -duplicate {duplicate_a} {duplicate_b} 1 -ow lmp -v 0"
    subprocess.run(atomsk_command, shell=True, check=True)

    charge2atom = f"lmp_charge2atom.sh {filename}"
    subprocess.run(charge2atom, shell=True, check=True)
        
    var['dim'] = get_model_dimensions(filename)

    center('2D',filename,var)

    var['2D']['lat_c'] = stacking(var,2)

    return var

def stacking(var,layer=2,sheetvsheet=False):
    
    settings_sheet(var,f"{var['dir']}/system_build/sheet_{layer}.in.settings",layer) # generate file for potentials

    filename = f"{var['dir']}/system_build/{var['2D']['mat']}"
    lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none',  '-nocite'])

    # lmp.file('lammps/in.init')
    lmp.commands_list([
    "units           metal\n",
    "atom_style      atomic\n",
    "neighbor        0.3 bin\n",
    "boundary        p p p",
    "neigh_modify    every 1 delay 0 check yes #every 5\n\n",
    f"region box block {var['dim']['xlo']} {var['dim']['xhi']} {var['dim']['ylo']} {var['dim']['yhi']} -50 50\n",
    f"create_box       {var['data']['2D']['natype']*layer} box\n\n",
    f"read_data       {filename}_1.lmp add append group layer_1",
    ])
    
    if var['2D']['stack_type'] == 'AB':
        x_shift = var['dim']['xhi']/4
    else:
        x_shift = 0
    if 'lat_c' not in var['2D']:
        lat_c = 6
    else:
        lat_c = var['2D']['lat_c']

    if sheetvsheet:
        for l in range(1,4):
            lmp.command(f"read_data {filename}_1.lmp add append shift 0 0 {l*lat_c} group layer_{l+1}\n") 
        lmp.commands_list([
            f"displace_atoms layer_3 move {x_shift} 0 0 units box\n",
            f"displace_atoms layer_4 move {x_shift} 0 0 units box\n",
            ])
    else:
        for l in range(1,layer):
            lmp.commands_list([
                f"read_data {filename}_1.lmp add append shift 0 0 {l*lat_c} group layer_{l+1}\n",
                f"displace_atoms layer_{l+1} move {x_shift*l} 0 0 units box\n",
                ]) 

    for t in range(var['data']['2D']['natype']):
        t+=1
        lmp.command(
        f"group 2D_{t} type {t}\n"
        )    

    g = 0
    i = 0
    c = 0

    for element,count in var['pot']['2D'].items():
        i += c
        for l in range(1,layer+1):
            for t in range(1,count+1):
                n = i + t
                g+=1
                lmp.commands_list([
                f"group 2Dtype intersect 2D_{n} layer_{l}\n",
                f"set group 2Dtype type {g}\n",
                f"group 2Dtype delete\n"
                ])
                c = count


    lmp.commands_list([
    f"include         {var['dir']}/system_build/sheet_{layer}.in.settings\n\n",
    "#----------------- Minimize the system -------------------\n\n"
    "min_style       cg\n",
    "minimize        1.0e-4 1.0e-8 1000000 1000000\n\n",
    "timestep        0.001\n",
    "thermo          100\n\n",
        "#----------------- Apply Langevin thermostat -------------\n",

    f"velocity        all create 300 492847948\n ",
    f"fix             lang all langevin 300 300 $(100.0*dt) 2847563 zero yes\n\n ",

    "fix             nve_all all nve\n\n ",

    "timestep        0.001\n",
    "thermo          100\n\n ",

    "compute l_1 layer_1 com\n",

    "compute l_2 layer_2 com\n",
    "variable comz_1 equal c_l_1[3] \n\n "
    "variable comz_2 equal c_l_2[3] \n\n "
    "run 0\n",

    f"write_data  {filename}_{layer}.lmp"

    ])

    
    # Extract center of mass (COM) for each layer
    com_l1 = lmp.extract_variable('comz_1', None,0)  # Returns an array [x, y, z]
    com_l2 = lmp.extract_variable('comz_2', None,0)  # Returns an array [x, y, z]
    # Compute lattice constant from the z-coordinates
    lat_c = com_l2 - com_l1  # Subtract z-components

    return lat_c

def center(system,filename,var):
    
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])

    lmp.commands_list([
    "units           metal\n",
    "atom_style      atomic\n",
    "neighbor        0.3 bin\n",
    "boundary        p p p",
    "neigh_modify    every 1 delay 0 check yes #every 5\n\n",
    f"region box block {var['dim']['xlo']} {var['dim']['xhi']} {var['dim']['ylo']} {var['dim']['yhi']} -50 50\n",
    f"create_box      {var['data'][system]['natype']} box\n\n",
    f"read_data       {filename} add append\n"
    ])
    
    elem = []
    i=0
    
    for element, count in var['pot'][system].items():
        if not count or count == 1:
            elem.append(element)
            i+=1
        else:
            for t in range(1,count+1):
                elem.append(element + str(t))
                i+=1
        mass=data.atomic_masses[data.atomic_numbers[element]]
        lmp.command(f"mass {i} {mass}\n")
    
    lmp.commands_list([
    f"pair_style  {var[system]['pot_type']}\n",
    f"pair_coeff * * {var[system]['pot_path']} {' '.join(elem)}\n",

    "compute zmin all reduce min z\n",
    "compute xmin all reduce min x\n",
    "compute ymin all reduce min y\n",
    
    "variable disp_z equal -c_zmin\n",
    "variable disp_x equal -(c_xmin+(xhi-xlo)/2.0)\n",
    "variable disp_y equal -(c_ymin+(yhi-ylo)/2.0)\n",
    
    "run 0\n\n",

    "displace_atoms all move v_disp_x v_disp_y v_disp_z units box\n\n",

    f"change_box all z final {var['dim']['zlo']} {var['dim']['zhi']}\n",
    "run 0\n",
    f"write_data  {filename}\n"
    ])
    lmp.close

def slab_generator(system,var,x,y,z):
    filename = os.path.join(os.path.dirname(__file__), "materials", f"{var[system]['mat']}.lmp")
    Path(filename).unlink(missing_ok=True)
    Path('a.cif').unlink(missing_ok=True)
    atomsk_command = f"atomsk {var[system]['cif_path']} -duplicate 2 2 1 -orthogonal-cell a.cif -ow -v 0"
    subprocess.run(atomsk_command, shell=True, check=True)
    cif = cifread("a.cif")
    x2 = round((x+15)/cif["lat_a"])
    y2 = round((y+15)/cif["lat_b"])
    z2 = round(z/cif["lat_c"])
    atomsk_command2 = f"atomsk a.cif -duplicate {x2} {y2} {z2} {filename} -ow -v 0"
    subprocess.run(atomsk_command2, shell=True, check=True)