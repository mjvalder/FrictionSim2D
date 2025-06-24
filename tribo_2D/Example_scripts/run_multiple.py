import os

from tribo_2D import afm, sheetvsheet, utilities

# Import the list of materials you want to run from a text file
with open("run/material_list.txt", "r") as f:
    materials = [line.strip() for line in f]

# Generate your configuration files based on the template
with open(f"run/afm_config_temp.ini", "r") as file:
    template_afm = file.read()

#  Loop through each material and generate the scripts
for m in materials:
    updated_afm = template_afm.replace("{mat}", m)

    with open(f"run/afm_config.ini", "w") as file:
        file.write(updated_afm)

    # updated_sheet = template_sheet.replace("{mat}", m)
    # with open(f"run/sheet_config.ini", "w") as file:
    #     file.write(updated_sheet)

    run=afm('run/afm_config.ini')
    run.system()
    run.slide()
    # run.pbs()

    # run=sheetvsheet('sheet_config.ini')
    # run.sheet_system()
    # run.sheet_pbs()

for file in os.listdir():
    if file.endswith(".cif") or file.endswith(".lmp") or file.endswith(".json"):
        os.remove(file)