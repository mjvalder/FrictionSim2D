"""AiiDA workflows for the tribo_2D LAMMPS simulations."""
from aiida.engine import WorkChain, ToContext, while_
from aiida.plugins import CalculationFactory
from aiida.orm import Code, SinglefileData, FolderData, Dict, Str, List
import configparser
import io

LammpsTribo2DCalculation = CalculationFactory('lammps.tribo_2d')


class LammpsTribo2DWorkChain(WorkChain):
    """WorkChain to run a batch of LAMMPS simulations."""

    @classmethod
    def define(cls, spec):
        """Define the workflow."""
        super().define(spec)
        spec.input('code', valid_type=Code)
        spec.input('simulation_script', valid_type=SinglefileData)
        spec.input('config', valid_type=SinglefileData, help="The template .ini config file with '{mat}' placeholder.")
        spec.input('settings', valid_type=SinglefileData)
        spec.input('potentials', valid_type=FolderData)
        spec.input('cifs', valid_type=FolderData)
        spec.input('supporting_scripts', valid_type=FolderData)
        spec.input('post_process_script', valid_type=SinglefileData)
        spec.input('materials_list', valid_type=List, help="A list of material names to iterate over.")


        spec.outline(
            cls.setup,
            while_(cls.has_next_material)(
                cls.run_simulation,
            ),
            cls.finalize,
        )
        spec.output('results', valid_type=Dict, help="A dictionary of results, with material names as keys.")

    def setup(self):
        """Set up the calculation."""
        self.ctx.materials = self.inputs.materials_list.get_list()
        self.ctx.current_index = 0
        self.ctx.results = {}

    def has_next_material(self):
        """Return whether there is another material to run."""
        return self.ctx.current_index < len(self.ctx.materials)

    def run_simulation(self):
        """Run the LAMMPS simulation for the current material."""
        material = self.ctx.materials[self.ctx.current_index]
        self.report(f'Submitting calculation for material: {material}')

        # Create a material-specific config file on the fly
        config_str = self.inputs.config.get_content()
        config_str_updated = config_str.replace("{mat}", material)

        # For simplicity, we can also set the materials_list path to be empty or a dummy
        # as we are now controlling the material directly.
        config = configparser.ConfigParser()
        config.read_string(config_str_updated)
        config['2D']['materials_list'] = '' # Disable list reading in the script
        
        with io.StringIO() as s:
            config.write(s)
            s.seek(0)
            material_config = SinglefileData(s)


        inputs = {
            'code': self.inputs.code,
            'simulation_script': self.inputs.simulation_script,
            'config': material_config,
            'settings': self.inputs.settings,
            'potentials': self.inputs.potentials,
            'cifs': self.inputs.cifs,
            'supporting_scripts': self.inputs.supporting_scripts,
            'post_process_script': self.inputs.post_process_script,
            'metadata': {
                'label': f'tribo_2d_{material}', # Unique label for this job
                'options': {
                    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 32},
                    'parser_name': 'lammps.tribo_2d',
                    'withmpi': True,
                    'output_filename': 'stdout.txt',
                    'error_filename': 'stderr.txt',
                }
            }
        }

        running = self.submit(LammpsTribo2DCalculation, **inputs)
        self.ctx.current_index += 1
        return ToContext(jobs=append_(running))


    def finalize(self):
        """Finalize the calculation and gather results."""
        for i, job in enumerate(self.ctx.jobs):
             material = self.ctx.materials[i]
             if job.is_finished_ok:
                 self.ctx.results[material] = job.outputs.results.get_dict()
             else:
                 self.ctx.results[material] = {'error': 'Calculation failed.'}

        self.out('results', Dict(dict=self.ctx.results))