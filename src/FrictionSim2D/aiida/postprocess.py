"""
AiiDA calculation plugin for post-processing Tribo_2D simulation results.
"""
from aiida.engine import CalcJob
from aiida.orm import SinglefileData, FolderData, RemoteData
from aiida.common import CalcInfo, CodeInfo

class PostProcessCalculation(CalcJob):
    """
    AiiDA calculation plugin for post-processing Tribo_2D simulation results.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('metadata.options.resources', valid_type=dict, default={'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        spec.input('metadata.options.parser_name', valid_type=str, default='tribo_2d.parsers.postprocess')
        spec.input('metadata.options.output_filename', valid_type=str, default='postprocess.log')

        spec.input('simulation_results_folder', valid_type=RemoteData, help='The remote folder containing the simulation results.')
        spec.input('postprocess_script', valid_type=SinglefileData, help='The post-processing script (read_data.py).')

        spec.output('json_files', valid_type=FolderData, help='Folder containing the output JSON files.')
        spec.default_output_node = 'json_files'

    def prepare_for_submission(self, folder):
        """
        Create the input files for the calculation.
        """
        # The postprocess_script will be in the working directory
        script_name = self.inputs.postprocess_script.filename

        # Create a run script
        run_script_path = folder.get_abs_path('run_postprocess.sh')
        with open(run_script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            # Copy the results from the remote simulation folder into a 'results' directory
            f.write(f"cp -r {self.inputs.simulation_results_folder.get_remote_path()}/* .\n")
            # Run the post-processing script
            f.write(f"python {script_name} export --resultsdir .\n")

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = ['run_postprocess.sh']
        codeinfo.stdout_name = self.metadata.options.output_filename
        codeinfo.withmpi = False

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        # We need to copy the post-processing script to the remote machine
        calcinfo.local_copy_list = [
            (self.inputs.postprocess_script.uuid, self.inputs.postprocess_script.filename, self.inputs.postprocess_script.filename)
        ]
        calcinfo.remote_copy_list = []
        # Retrieve the 'outputs' folder which contains the JSON files
        calcinfo.retrieve_list = [self.metadata.options.output_filename, 'outputs/']

        return calcinfo
