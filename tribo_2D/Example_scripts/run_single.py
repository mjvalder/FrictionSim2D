from tribo_2D.afm import *
from tribo_2D.utilities import *
from tribo_2D.sheetvsheet import *


run=afm('afm_config.ini')
run.system()
run.load()
run.slide()
run.pbs()

run=sheetvsheet('sheet_config.ini')
run.sheet_system()
run.sheet_pbs()

for file in os.listdir():
    if file.endswith(".cif") or file.endswith(".lmp") or file.endswith(".json"):
        os.remove(file)