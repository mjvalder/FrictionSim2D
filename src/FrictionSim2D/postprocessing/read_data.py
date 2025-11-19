import os
import re
import json
import numpy as np
import pandas as pd
import argparse

class DataReader:
    """
    Reads and processes friction simulation data.

    This class is designed to walk through a directory of simulation results,
    parse filenames and file paths to extract metadata, read the time-series
    data from each valid file, and then store it in a structured way.

    It separates the full time-series data (for plotting) from the mean
    data (for ranking).
    """
    def __init__(self, results_dir='results_110725_test'):
        """
        Initializes the DataReader.

        Args:
            results_dir (str): The path to the directory containing simulation results.
        """
        self.settings = self.get_inputs(results_dir)
        self.output_dir = os.path.join(self.settings['resultsdir'], 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)

        # Dictionaries to hold the parsed data
        self.full_data_nested = {}
        
        # Read the data and populate the dictionaries and metadata
        self.time_series, self.incomplete_files, self.incomplete_materials, self.metadata, self.ntimestep = self.read_data()

    def _get_output_path(self, filename):
        """Constructs the full path for an output file in the designated output directory."""
        return os.path.join(self.output_dir, filename)

    def get_inputs(self, results_dir):
        """Defines the settings for data processing."""
        settings = {
            'resultsdir': results_dir,
            'fields': ['time', 'nf', 'lfx', 'lfy', 'comx', 'comy', 'comz', 'tipx', 'tipy', 'tipz'],
        }
        return settings

    def read_data(self):
        """
        Walks through the results directory, reads all simulation data,
        and populates the instance's data structures using a two-pass approach
        to dynamically determine the correct number of timesteps for a complete file.
        """
        results_dir = self.settings['resultsdir']
        file_pattern = re.compile(r'fc_ave_slide_(\d+\.?\d*)nN_(\d+)angle_(\d+)ms_l(\d+)')
        path_pattern = re.compile(r'(\d+x_\d+y)/sub_(\w+)_tip_(\w+)_(r\d+)')

        # First pass: Find the maximum number of timesteps to define a "complete" file
        ntimestep = 0
        print("Starting first pass to determine ntimestep...")
        for root, _, files in os.walk(results_dir):
            if not path_pattern.search(root):
                continue
            for filename in files:
                if file_pattern.match(filename):
                    filepath = os.path.join(root, filename)
                    try:
                        # Read only the row count for efficiency
                        df = pd.read_csv(filepath, sep=r'\s+', header=None, usecols=[0], skiprows=2)
                        if len(df) > ntimestep:
                            ntimestep = len(df)
                    except (pd.errors.EmptyDataError, IndexError):
                        continue # Ignore empty or malformed files in the first pass

        if ntimestep == 0:
            print("Warning: No valid data files found. Could not determine ntimestep.")
            return None, {}, {}, {}, 0
        
        print(f"Determined ntimestep for a complete file to be: {ntimestep}")

        # Second pass: Process only the complete files
        time_series = None
        incomplete_files = {}
        incomplete_materials = {}
        metadata = {
            'materials': set(), 'substrates': set(), 'tip_materials': set(),
            'tip_radii': set(), 'layers': set(), 'speeds': set(),
            'forces_and_angles': {}
        }

        for root, _, files in os.walk(results_dir):
            path_match = path_pattern.search(root)
            if not path_match:
                continue

            size, substrate_material, tip_material, tip_radius = path_match.groups()
            
            try:
                search_path = os.path.join(results_dir, 'afm')
                start_dir = search_path if os.path.commonpath([root, search_path]) == search_path else results_dir
                material_path_end_index = root.find(size)
                material_path_full = root[:material_path_end_index]
                material = os.path.relpath(material_path_full, start_dir).strip(os.sep)
                if not material or material == '.':
                    continue
            except (IndexError, ValueError):
                continue

            safe_material = material.replace('-', '_').replace(os.sep, '__')
            size_key = size.replace('x_', 'x')

            for filename in files:
                file_match = file_pattern.match(filename)
                if not file_match:
                    continue
                
                filepath = os.path.join(root, filename)
                try:
                    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=self.settings['fields'], skiprows=2)
                    
                    if len(df) != ntimestep:
                        incomplete_files.setdefault(size_key, []).append(filepath)
                        incomplete_materials.setdefault(size_key, set()).add(material)
                        continue

                    if time_series is None:
                        time_series = df['time'].to_list()

                    force_str, angle_str, speed_str, layer_str = file_match.groups()
                    force = float(force_str)
                    angle, speed, layer = map(int, [angle_str, speed_str, layer_str])

                    metadata['materials'].add(safe_material)
                    metadata['substrates'].add(substrate_material)
                    metadata['tip_materials'].add(tip_material)
                    metadata['tip_radii'].add(tip_radius)
                    
                    if force not in metadata['forces_and_angles']:
                        metadata['forces_and_angles'][force] = set()
                    metadata['forces_and_angles'][force].add(angle)
                    metadata['speeds'].add(speed)
                    metadata['layers'].add(layer)

                    df_processed = df.drop(columns=['time'])

                    full_path = self.full_data_nested.setdefault(safe_material, {}).setdefault(size_key, {}).setdefault(substrate_material, {})\
                        .setdefault(tip_material, {}).setdefault(tip_radius, {}).setdefault(f'l{layer}', {})\
                        .setdefault(f's{speed}', {}).setdefault(f'f{force}', {})
                    full_path[f'a{angle}'] = df_processed
                        
                except (pd.errors.EmptyDataError, IndexError, ValueError):
                    incomplete_files.setdefault(size_key, []).append(filepath)
                    incomplete_materials.setdefault(size_key, set()).add(material)
        
        final_metadata = {}
        for k, v in metadata.items():
            if k == 'forces_and_angles':
                final_metadata[k] = {force: sorted(list(angles)) for force, angles in v.items()}
            else:
                final_metadata[k] = sorted(list(v))

        material_types = { 'b_type': [], 'h_type': [], 't_type': [], 'p_type': [], 'other': [] }
        for material_name in final_metadata.get('materials', []):
            try:
                prefix = material_name.split('_', 1)[0]
                type_key = f"{prefix}_type"
                if type_key in material_types:
                    material_types[type_key].append(material_name)
                else:
                    material_types['other'].append(material_name)
            except IndexError:
                material_types['other'].append(material_name)
        final_metadata['material_types'] = material_types

        return time_series, incomplete_files, incomplete_materials, final_metadata, ntimestep

    def export_full_data_to_json(self):
        """Exports the full time-series data to JSON files, one for each 'size'."""
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, pd.DataFrame):
                    # Convert DataFrame to a dictionary for JSON serialization
                    return {'columns': obj.columns.tolist(), 'data': obj.values.tolist()}
                return super(NpEncoder, self).default(obj)

        # Reorganize data by size for size-specific output files
        data_by_size = {}
        for material, mat_data in self.full_data_nested.items():
            for size, size_data in mat_data.items():
                data_by_size.setdefault(size, {})[material] = size_data

        for size_key, size_data in data_by_size.items():
            full_data_output_path = self._get_output_path(f'output_full_{size_key}.json')
            
            # Create a metadata copy for this specific output file
            full_output_metadata = self.metadata.copy()
            full_output_metadata['time_series'] = self.time_series
            
            # Filter tip_radii to only include those present in this size's data
            relevant_tip_radii = set()
            for mat_data in size_data.values():
                for sub_data in mat_data.values():
                    for tip_mat_data in sub_data.values():
                        relevant_tip_radii.update(tip_mat_data.keys())
            
            full_output_metadata['tip_radii'] = sorted(list(relevant_tip_radii))

            # Add the specific size for this file to the metadata
            full_output_metadata['size'] = size_key

            output_with_metadata = {
                'metadata': full_output_metadata,
                'results': size_data
            }
            
            with open(full_data_output_path, 'w') as f:
                json.dump(output_with_metadata, f, cls=NpEncoder)
            print(f"Full time-series data for size {size_key} exported to {full_data_output_path}")

    def export_issue_reports(self):
        """Exports reports on incomplete files and materials to text files."""
        if self.incomplete_files:
            for size, files in self.incomplete_files.items():
                filepath = self._get_output_path(f'incomplete_files_{size}.txt')
                with open(filepath, 'w') as f:
                    f.write('\n'.join(sorted(files)))
                print(f"List of incomplete files for size {size} saved to {filepath}")

        if self.incomplete_materials:
            for size, materials in self.incomplete_materials.items():
                filepath = self._get_output_path(f"incomplete_materials_{size}.txt")
                with open(filepath, 'w') as f:
                    f.write('\n'.join(sorted(list(materials))))
                print(f"List of incomplete materials for size {size} saved to {filepath}")

def main():
    """Main function to parse arguments and run the data processing."""
    parser = argparse.ArgumentParser(description="Read and process friction simulation data, exporting full data and rankings to JSON.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Common argument for results directory
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--resultsdir', type=str, default='results_110725_test', help="Directory containing the simulation results.")

    # Export command: Exports full time-series data
    subparsers.add_parser('export', parents=[parent_parser], help='Export full time-series data to JSON files.')

    args = parser.parse_args()
    
    # Instantiate the main class
    dr = DataReader(results_dir=args.resultsdir)
    
    # Always generate reports on incomplete data
    dr.export_issue_reports()

    # Execute the chosen command
    if args.command == 'export':
        dr.export_full_data_to_json()

if __name__ == '__main__':
    main()
