"""AiiDA calculation plugin for the tribo_2D LAMMPS simulations."""
import os

from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.orm import SinglefileData, FolderData, Dict


class LammpsTribo2DCalculation(CalcJob):
    """AiiDA calculation plugin for LAMMPS."""

    @classmethod
    def define(cls, spec):
        """Define the inputs and outputs of the calculation."""
        super().define(spec)

        # Input parameters
        spec.input('metadata.options.resources', valid_type=dict, default={'num_machines': 1, 'num_mpiprocs_per_machine': 32})
        spec.input('metadata.options.parser_name', valid_type=str, default='lammps.tribo_2d')
        spec.input('metadata.options.withmpi', valid_type=bool, default=True)

        spec.input('simulation_script', valid_type=SinglefileData, help='The main Python script to execute (e.g., afm.py).')
        spec.input('config', valid_type=SinglefileData, help='The .ini configuration file.')
        spec.input('settings', valid_type=SinglefileData, help='The settings.yaml file.')
        spec.input('potentials', valid_type=FolderData, help='The folder containing the potential files.')
        spec.input('cifs', valid_type=FolderData, help='The folder containing the CIF files.')
        spec.input('supporting_scripts', valid_type=FolderData, help='Folder with model_init.py, utilities.py, etc.')
        spec.input('post_process_script', valid_type=SinglefileData, help='The read_data.py post-processing script.')


        # Output parameters
        spec.output('results', valid_type=Dict, help='The parsed results of the simulation.')
        spec.default_output_node = 'results'

    def prepare_for_submission(self, folder):
        """Prepare the calculation for submission."""
        # --- Create the necessary directory structure ---
        run_dir = folder.get_subfolder('run', create=True)
        cif_dir = run_dir.get_subfolder('cif', create=True)
        potentials_dir = folder.get_subfolder('potentials', create=True)

        # --- Copy input files into the correct directories ---
        main_script_name = self.inputs.simulation_script.filename
        folder.put_object_from_filelike(self.inputs.simulation_script.open(), main_script_name)
        folder.put_object_from_filelike(self.inputs.config.open(), 'config.ini')
        folder.put_object_from_filelike(self.inputs.settings.open(), 'settings.yaml')
        folder.put_object_from_filelike(self.inputs.post_process_script.open(), 'read_data.py')

        for filename in self.inputs.potentials.list_object_names():
            with self.inputs.potentials.open(filename, 'rb') as f:
                potentials_dir.put_object_from_filelike(f, filename)

        for filename in self.inputs.cifs.list_object_names():
            with self.inputs.cifs.open(filename, 'rb') as f:
                cif_dir.put_object_from_filelike(f, filename)

        for filename in self.inputs.supporting_scripts.list_object_names():
            with self.inputs.supporting_scripts.open(filename, 'rb') as f:
                folder.put_object_from_filelike(f, filename)


        # --- Create the submission script ---
        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        # Command to run the main simulation script
        codeinfo.cmdline_params = ['python', main_script_name]
        # Command to run the post-processing script
        # Assumes read_data.py takes the output directory as an argument
        codeinfo.post_exec_commands = ['python', 'read_data.py', './']


        codeinfo.stdout_name = self.metadata.options.output_filename
        codeinfo.stderr_name = self.metadata.options.error_filename


        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        # Retrieve the final JSON file and any other important output
        calcinfo.retrieve_list = [self.metadata.options.output_filename, self.metadata.options.error_filename, 'results.json']


        return calcinfo