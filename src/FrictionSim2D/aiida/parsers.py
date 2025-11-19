"""
Parsers for the Tribo_2D AiiDA plugin.
"""
import json
from aiida.parsers.parser import Parser
from aiida.orm import Dict, FolderData

class LammpsParser(Parser):
    """
    Parser for the results of a LammpsCalculation.
    This is a placeholder and does not parse any output yet.
    """
    def parse(self, **kwargs):
        """
        Parse the output files.
        """
        # For now, we just pass the retrieved folder
        pass


class PostProcessParser(Parser):
    """
    Parser for the results of a PostProcessCalculation.
    """

    def parse(self, **kwargs):
        """
        Parse the output files.
        """
        output_folder = self.retrieved
        output_files = output_folder.list_object_names()

        if 'outputs' not in output_files:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        json_folder = FolderData()
        json_folder.put_object_from_tree(output_folder.get_abs_path('outputs'))
        self.out('json_files', json_folder)
