import os
from tribo_2D import afm, model_init

# Import the list of materials you want to run from a text file
# with open("run/material_list.txt", "r", encoding="utf-8") as f:
#     materials = [line.strip() for line in f]
materials = ["h-MoS2"]
# Generate your configuration files based on the template
with open("run/afm_config_temp.ini", "r", encoding="utf-8") as file:
    template_afm = file.read()

#  Loop through each material and generate the scripts
for m in materials:
    updated_afm = template_afm.replace("{mat}", m)
    with open("run/afm_config.ini", "w", encoding="utf-8") as file:
        file.write(updated_afm)

    # updated_sheet = template_sheet.replace("{mat}", m)
    # with open(f"run/sheet_config.ini", "w") as file:
    #     file.write(updated_sheet)

    # run = model_init.model_init('run/afm_config.ini','afm')
    run = afm.AFM_simulation('run/afm_config.ini')
    run.generate_system_init_script()
    run.generate_slide_script()

for file in os.listdir():
    if file.endswith("cif") or file.endswith("lmp") or file.endswith("json"):
        os.remove(file)
