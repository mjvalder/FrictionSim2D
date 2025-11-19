import json
import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import glob

matplotlib.use('Agg') # Use a non-interactive backend to prevent Qt errors
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, data_dirs, labels, output_dir):
        self.data_dirs = data_dirs
        self.labels = labels
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.time_step_fs = 1.0  # Time step in femtoseconds, adjustable here.
        self.figure_size = (10, 6)
        self.plot_style = 'line'

        self.full_data_files = {label: {} for label in self.labels}
        self.full_data_cache = {label: {} for label in self.labels}
        self.metadata = {}
        self.summary_df_cache = None
        self.material_type_map = {}

        self._discover_data_files()
        self._load_all_metadata()
        self._create_material_type_map()

    def _create_material_type_map(self):
        """Creates a map from material_id to material_type from the combined metadata."""
        material_types_dict = self.metadata.get('material_types')

        if isinstance(material_types_dict, dict):
            self.material_type_map = {
                material_id.strip(): type_name.strip()
                for type_name, material_list in material_types_dict.items()
                for material_id in material_list
            }
        else:
            print("Warning: 'material_types' not found in combined metadata. Plotting by type may fail.")

    def _deep_merge_dict(self, d1, d2):
        """
        Recursively merges dictionary d2 into d1. Overwrites values, updates lists, and merges dicts.
        """
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                self._deep_merge_dict(d1[k], v)
            elif k in d1 and isinstance(d1[k], list) and isinstance(v, list):
                d1[k].extend(v) # Simple extend, can be refined if needed
            else:
                d1[k] = v

    def _discover_data_files(self):
        """Finds all output_full_*.json files in each data directory."""
        for label, data_dir in zip(self.labels, self.data_dirs):
            # Data is expected to be in an 'outputs' subdirectory based on the read_data.py script
            search_dir = os.path.join(data_dir, 'outputs')

            if not os.path.isdir(search_dir):
                print(f"Warning: 'outputs' directory not found in {data_dir}. Searching in the base directory instead.")
                search_dir = data_dir
            
            if not os.path.isdir(search_dir):
                print(f"Error: Data directory not found for label '{label}': {data_dir}")
                continue

            for filename in os.listdir(search_dir):
                match = re.match(r'output_full_(.+)\.json', filename)
                if match:
                    file_key = match.group(1) # Use the unique part of the filename as the key
                    self.full_data_files[label][file_key] = os.path.join(search_dir, filename)
            
            if not self.full_data_files[label]:
                print(f"Warning: No 'output_full_*.json' files found for label '{label}' in the searched directory: {search_dir}")

    def _load_all_metadata(self):
        """Loads and merges metadata from ALL available data files."""
        # print("Loading and merging metadata from all data files...")
        for label in self.labels:
            if not self.full_data_files[label]:
                continue
            for file_key in self.full_data_files[label]:
                _, metadata = self._load_full_data(label, file_key)
                if metadata:
                    self._deep_merge_dict(self.metadata, metadata)
        # print("Metadata loading and merging complete.")

    def _load_full_data(self, label, file_key):
        """Loads a single data file and returns (results, metadata)."""
        file_path = self.full_data_files.get(label, {}).get(file_key)
        if not file_path:
            print(f"Warning: No data file found for label '{label}' and file_key '{file_key}'")
            return None, None
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Expecting 'metadata' and 'results' keys in the file
            metadata = data.get('metadata', {})
            results = data.get('results', {})
            return results, metadata
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading data from {file_path}: {e}")
            return None, None

    def _extract_all_runs(self, label, file_key):
        """
        Recursively traverses the results structure for a single file
        and yields a dictionary for each individual simulation run found.
        Each dictionary contains the parsed parameters and the processed DataFrame.
        """
        results, _ = self._load_full_data(label, file_key)
        if not results:
            return

        def process_level(data_dict, params_so_far):
            # Base case: we found the data for a run
            if 'columns' in data_dict and 'data' in data_dict:
                df = pd.DataFrame(data_dict['data'], columns=data_dict['columns'])
                df = self._add_derived_columns(df)
                run_data = params_so_far.copy()
                run_data['df'] = df
                yield run_data
                return
            # Recursive step: parse parameters and go deeper
            for key, value in data_dict.items():
                if isinstance(value, dict):
                    new_params = params_so_far.copy()
                    if 'id' not in new_params:
                        new_params['id'] = key.strip()
                    else:
                        match_prefix = re.match(r'([a-zA-Z]+)(\d+\.?\d*)', key)
                        if match_prefix:
                            prefix, val_str = match_prefix.groups()
                            val = float(val_str)
                            if prefix == 'f': new_params['force'] = val
                            elif prefix == 'a': new_params['angle'] = val
                            elif prefix == 'r': new_params['tip_radius'] = val
                            elif prefix == 'l': new_params['layer'] = val
                            elif prefix == 's': new_params['speed'] = val
                        match_suffix_force = re.match(r'(\d+\.?\d*)nN', key)
                        if match_suffix_force:
                            new_params['force'] = float(match_suffix_force.group(1))
                        match_suffix_angle = re.match(r'(\d+\.?\d*)deg', key)
                        if match_suffix_angle:
                            new_params['angle'] = float(match_suffix_angle.group(1))
                    yield from process_level(value, new_params)
        yield from process_level(results, {})

    def _add_derived_columns(self, df):
        """
        Adds derived quantities (lf, cof, tip_sep) to a raw data DataFrame.
        """
        # Ensure required columns exist
        required_cols = ['lfx', 'lfy', 'nf', 'tipz', 'comz']
        if not all(col in df.columns for col in required_cols):
            print("Warning: Missing one or more required columns for derived calculations. Skipping.")
            return df

        df['lf'] = np.sqrt(df['lfx']**2 + df['lfy']**2)
        # Avoid division by zero for COF
        df['cof'] = (df['lf'] / df['nf']).replace([np.inf, -np.inf], np.nan)
        df['tip_sep'] = df['tipz'] - df['comz']
        
        # Add tipspeed calculation, assuming time step is in fs
        if 'tipx' in df.columns and 'tipy' in df.columns:
            time_interval_ps = self.time_step_fs / 1000.0  # Convert fs to ps
            if time_interval_ps > 0:
                df['tipspeed'] = np.sqrt(df['tipx'].diff().pow(2) + df['tipy'].diff().pow(2)).fillna(0) / time_interval_ps
            else:
                df['tipspeed'] = 0 # Avoid division by zero if time step is 0

        return df

    def _get_summary_data_df(self):
        """
        Returns the summary DataFrame, calculating it if it's not already cached.
        """
        if self.summary_df_cache is None:
            self._calculate_summary_statistics()
            self.summary_df_cache = self.summary_df
        return self.summary_df_cache

    def _calculate_summary_statistics(self):
        """
        Calculates summary statistics by processing all runs from all files.
        This function is now much simpler, leveraging the _extract_all_runs helper.
        """
        print("Calculating summary statistics for all data files...")
        all_records = []
        for label in self.full_data_files.keys():
            for file_key in self.full_data_files[label].keys():
                # Use the generator to get all processed runs from the file
                for run_data in self._extract_all_runs(label, file_key):
                    df = run_data.pop('df') # Get the DataFrame and remove it from the dict
                    summary_stats = df.mean().to_dict()
                    
                    record = {
                        'dataset_label': label,
                        'file_key': file_key,
                        **run_data, # Parameters from the run
                        **summary_stats # Averaged stats
                    }
                    all_records.append(record)

        self.summary_df = pd.DataFrame(all_records)
        
        # Add material type mapping
        if not self.summary_df.empty and 'id' in self.summary_df.columns:
            self.summary_df['material_type'] = self.summary_df['id'].map(self.material_type_map)
            # Add a general 'size' column for filtering if it doesn't exist
            if 'size' not in self.summary_df.columns:
                 self.summary_df['size'] = self.summary_df['file_key'].str.extract(r'(\d+x\d+y?)')[0]
        else:
            print("Warning: 'id' column not found in summary data. Cannot map material types.")

        print("Summary DataFrame created:")
        print(self.summary_df.head())
        if not self.summary_df.empty:
            print("Columns available in summary_df:", self.summary_df.columns.tolist())

    def _setup_plot(self, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, which='both', linestyle=':')
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc='best')

    def _get_filtered_summary_data(self, filters):
        """
        Retrieves summarized data and applies a set of filters.
        """
        df = self._get_summary_data_df()
        if df.empty:
            return pd.DataFrame()

        active_filters = {k: v for k, v in filters.items() if v is not None}
        for key, value in active_filters.items():
            if key in df.columns:
                if isinstance(value, list):
                    df = df[df[key].isin(value)]
                else:
                    df = df[df[key] == value]
        return df

    def _generate_summary_plot(self, plot_config):
        title = plot_config.get('title', 'Untitled')
        dataset_labels_to_plot = plot_config.get('datasets')
        
        summary_df = self._get_summary_data_df()
        if summary_df.empty:
            print(f"Warning: Summary data is empty. Skipping plot '{title}'.")
            return

        print(f"\n--- Debugging Plot: {title} ---")
        print(f"Initial summary_df shape: {summary_df.shape}")

        filtered_df = summary_df.copy()

        if dataset_labels_to_plot:
            filtered_df = filtered_df[filtered_df['dataset_label'].isin(dataset_labels_to_plot)]
            print(f"After dataset filter ({dataset_labels_to_plot}): {filtered_df.shape}")

        plot_by = plot_config.get('plot_by', 'id')

        # Apply general filters first
        filters = {
            'angle': plot_config.get('angle'), 'force': plot_config.get('force'),
            'size': plot_config.get('filter_size'), 'layer': plot_config.get('filter_layer'),
            'speed': plot_config.get('filter_speed'), 'tip_radius': plot_config.get('filter_tip_radius'),
        }
        
        # If layer filter is not specified, default to 1 for summary plots
        if filters['layer'] is None:
            print("Info: No layer filter specified for summary plot; defaulting to layer 1.")
            filters['layer'] = 1

        # If plotting vs force (load) and angle is not specified, default to 0
        x_col = plot_config.get('x_axis')
        if x_col == 'force' and filters['angle'] is None:
            print("Info: Plotting vs. force without angle specified for summary plot; defaulting to angle = 0.0.")
            filters['angle'] = 0.0
        
        active_filters = {k: v for k, v in filters.items() if v is not None}
        for key, value in active_filters.items():
            if key in filtered_df.columns:
                original_shape = filtered_df.shape
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]
                print(f"After filter '{key}' == '{value}': {original_shape} -> {filtered_df.shape}")

        # Now apply material/type filters to the already-reduced dataframe
        if plot_config.get('filter_materials'):
            filter_values = [v.strip() for v in plot_config.get('filter_materials')]
            
            if plot_by == 'id':
                original_shape = filtered_df.shape
                # Use .str.contains() to find materials within the ID string
                # This allows for partial matches, e.g., finding 'h_MoS2' in 'mos2_jing2__afm__h_MoS2'
                # We join the filter_values with '|' to create a regex OR condition
                pattern = '|'.join(filter_values)
                filtered_df = filtered_df[filtered_df['id'].str.contains(pattern, regex=True)]
                print(f"After id filter ({len(filter_values)} ids using contains): {original_shape} -> {filtered_df.shape}")
            
            elif plot_by == 'material_type':
                ids_to_plot = [mid for mid, mtype in self.material_type_map.items() if mtype in filter_values]
                original_shape = filtered_df.shape
                filtered_df = filtered_df[filtered_df['id'].isin(ids_to_plot)]
                print(f"After material_type filter ({filter_values}): {original_shape} -> {filtered_df.shape}")

        if filtered_df.empty:
            print(f"Warning: No data left after filtering for plot '{title}'. Skipping.")
            return

        x_col = plot_config['x_axis']
        y_col = plot_config['y_axis']

        if y_col not in filtered_df.columns:
            print(f"Error: y-axis column '{y_col}' not found in the summary data for plot '{title}'. Skipping.")
            return

        # --- More precise, group-based outlier removal by magnitude ---
        initial_rows = len(filtered_df)
        if not filtered_df.empty:
            # Define a function to filter outliers within a single group
            def remove_magnitude_outliers(group):
                if len(group) < 3:  # Don't filter very small groups
                    return group
                
                median_y = group[y_col].median()
                
                # Avoid filtering if the median is close to zero, as the ratio is meaningless
                if abs(median_y) < 1e-6:
                    return group
                
                # Define an outlier as any point 10x greater in magnitude than the median.
                # This removes points that are "orders of magnitude" different.
                magnitude_threshold = 10.0
                # We use absolute values to handle both positive and negative outliers correctly.
                is_outlier = np.abs(group[y_col]) > magnitude_threshold * np.abs(median_y)
                
                # Keep the non-outliers
                return group[~is_outlier]

            # Apply the outlier removal for each group on the x-axis.
            # This ensures we only remove outliers relative to their own x-group.
            # The `include_groups=False` argument is to silence a FutureWarning and prevent the ValueError.
            cleaned_df = filtered_df.groupby(x_col).apply(remove_magnitude_outliers, include_groups=False).reset_index()
            
            removed_count = initial_rows - len(cleaned_df)
            if removed_count > 0:
                print(f"Info: Removed {removed_count} individual outlier points based on magnitude for plot '{title}'.")
        else:
            cleaned_df = filtered_df
        # --- End of outlier removal ---

        if cleaned_df.empty:
            print(f"Warning: No data left after outlier removal for plot '{title}'. Skipping.")
            return

        plt.figure(figsize=self.figure_size)

        if plot_by == 'material_type':
            # Group by material type and calculate mean and std for each group
            # CRITICAL: This grouping now happens on the CLEANED dataframe
            grouped = cleaned_df.groupby('material_type')
            for material_type, group in grouped:
                # For 'per material type' plots, we average the results.
                # The mean and std are now robust as they are calculated on data with outliers removed.
                plot_data = group.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
                plot_data = plot_data.sort_values(by=x_col)

                # Plot the mean line
                plt.plot(plot_data[x_col], plot_data['mean'], marker='o', linestyle='-', label=material_type)
                
                # Add a shaded region for the standard deviation (error bars)
                # This will no longer be distorted by outliers.
                plt.fill_between(plot_data[x_col], 
                                 plot_data['mean'] - plot_data['std'], 
                                 plot_data['mean'] + plot_data['std'], 
                                 alpha=0.2)
                
                print(f"  - Plotted average for {material_type} ({len(group)} runs)")

        elif plot_by == 'id':
            # Plot each individual run (material id) from the cleaned dataframe
            grouped = cleaned_df.groupby('id')
            for material_id, group in grouped:
                if self.plot_style == 'line':
                    # For line plots, sort by x-axis value to prevent zig-zag lines
                    group = group.sort_values(by=x_col)
                    plt.plot(group[x_col], group[y_col], marker='o', linestyle='-', label=material_id)
                elif self.plot_style == 'scatter':
                    plt.scatter(group[x_col], group[y_col], label=material_id)
        
        # --- Final Y-axis scaling to ensure a clear view ---
        if not cleaned_df.empty:
            # For COF vs Force plots, base the zoom on data where force > 10
            if y_col == 'cof' and x_col == 'force':
                zoom_df = cleaned_df[cleaned_df[x_col] > 10]
                print(f"Info: Basing y-axis zoom for '{title}' on data where force > 10 nN.")
                # If there's no data above 10, fall back to the full dataset to avoid errors
                if zoom_df.empty:
                    print("Warning: No data with force > 10 nN. Using full data range for zoom.")
                    zoom_df = cleaned_df
            else:
                # For all other plots, use the full cleaned dataset
                zoom_df = cleaned_df

            min_y = zoom_df[y_col].min()
            max_y = zoom_df[y_col].max()
            padding = (max_y - min_y) * 0.1  # 10% padding on each side

            if padding < 1e-9: # Handle flat data case
                padding = abs(max_y * 0.1) if abs(max_y) > 1e-9 else 0.1

            plt.ylim(min_y - padding, max_y + padding)
            print(f"Final y-axis view set to range: [{min_y - padding:.3f}, {max_y + padding:.3f}]")

        plt.xlabel(plot_config.get('x_label', x_col))
        plt.ylabel(plot_config.get('y_label', y_col))
        plt.title(title)
        if any(plt.gca().get_legend_handles_labels()):
             plt.legend()
        plt.grid(True)
        
        filename = plot_config.get('filename')
        if filename:
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, format=filename.split('.')[-1])
            print(f"Generated summary plot: {output_path}")
        else:
            print(f"Warning: No filename specified for plot '{title}'. Plot not saved.")

        plt.close()

    def _generate_timeseries_plot(self, plot_config):
        """
        Generates a plot of a variable over time for specific materials.
        This function is now much simpler, leveraging the _extract_all_runs helper.
        """
        dataset_labels_to_plot = plot_config.get('datasets', [self.labels[0]] if self.labels else [])
        if not dataset_labels_to_plot:
            print("Error: No dataset specified or available for timeseries plot.")
            return

        label_to_plot = dataset_labels_to_plot[0]
        if len(dataset_labels_to_plot) > 1:
            print(f"Warning: Timeseries plots are generated for a single dataset. Using '{label_to_plot}'.")

        file_key = plot_config.get('filter_size') # Still using filter_size for file key selection
        if not file_key:
            print("Error: 'filter_size' (used as the file key) is required for timeseries plots.")
            return

        # --- FIX: Load local metadata for the specific file to get the correct time_series ---
        _, local_metadata = self._load_full_data(label_to_plot, file_key)
        if not local_metadata:
            print(f"Error: Could not load local metadata for file_key '{file_key}'.")
            return
        time_series = local_metadata.get('time_series')
        if not time_series:
            print(f"Error: 'time_series' not found in the local metadata for file_key '{file_key}'.")
            return
        # --- END FIX ---

        # Get all processed runs from the specified file
        all_runs = list(self._extract_all_runs(label_to_plot, file_key))
        if not all_runs:
            print(f"No data runs found for label '{label_to_plot}' and file_key '{file_key}'.")
            return

        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # --- Filtering Logic ---
        filters = {
            'id': plot_config.get('filter_materials'),
            'force': plot_config.get('force'),
            'angle': plot_config.get('angle'),
            'layer': plot_config.get('filter_layer'),
            'speed': plot_config.get('filter_speed'),
            'tip_radius': plot_config.get('filter_tip_radius'),
        }
        
        # If layer filter is not specified, default to 1 for timeseries plots
        if filters['layer'] is None:
            print("Info: No layer filter specified for timeseries plot; defaulting to layer 1.")
            filters['layer'] = 1

        # If angle is not specified, assume angle is 0 for timeseries plots
        if filters['angle'] is None:
            print("Info: Angle not specified; assuming angle = 0.0 for timeseries plot.")
            filters['angle'] = 0.0
        
        filtered_runs = []
        for run in all_runs:
            match = True
            for key, value in filters.items():
                if value is not None:
                    run_value = run.get(key)
                    if run_value is None:
                        match = False
                        break
                    
                    # Special handling for 'id' to allow partial matching
                    if key == 'id' and isinstance(value, list):
                        # Check if any of the filter values are present in the run's ID string
                        if not any(v in run_value for v in value):
                            match = False
                            break
                    elif isinstance(value, list):
                        if run_value not in value:
                            match = False
                            break
                    else:
                        if run_value != value:
                            match = False
                            break
            if match:
                filtered_runs.append(run)

        if not filtered_runs:
            print("No runs matched the specified filters for the timeseries plot.")
            return
        # --- End Filtering ---
        
        y_axis_col = plot_config.get('y_axis')
        if not y_axis_col:
            print(f"Error: Invalid or missing y-axis for timeseries plot."); return

        for run in filtered_runs:
            df = run['df']
            if y_axis_col not in df.columns:
                print(f"Warning: y-axis '{y_axis_col}' not found in data for run {run.get('id')}. Skipping.")
                continue

            label_parts = [run.get('id', 'N/A')]
            if 'force' in run: label_parts.append(f"F={run['force']}nN")
            if 'angle' in run: label_parts.append(f"A={run['angle']}deg")
            
            ax.plot(time_series, df[y_axis_col], label=", ".join(map(str, label_parts)))

        self._setup_plot(ax, plot_config.get('title', 'Timeseries Plot'), 
                         plot_config.get('x_label', 'Time (ps)'), 
                         plot_config.get('y_label', y_axis_col))
        
        filename = plot_config.get('filename')
        if filename:
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, format=filename.split('.')[-1])
            print(f"Generated timeseries plot: {output_path}")
        else:
            print(f"Warning: No filename specified for plot '{plot_config.get('title', 'Untitled')}'. Plot not saved.")
        
        plt.close()

    def _generate_correlation_plots(self, plot_config):
        """Generates correlation heatmaps based on friction ranking files."""
        print("\n--- Generating Rank Correlation Plots ---")
        
        # Find all ranking files
        ranking_files = glob.glob(os.path.join(self.output_dir, 'friction_ranking_*.json'))
        if not ranking_files:
            print("Error: No 'friction_ranking_*.json' files found. Please generate rankings first.")
            return

        all_ranks = []
        for f_path in ranking_files:
            size_match = re.search(r'friction_ranking_(.+)\.json', os.path.basename(f_path))
            if not size_match:
                continue
            size = size_match.group(1)
            
            with open(f_path, 'r') as f:
                data = json.load(f)
                for force_key, material_list in data.items():
                    force = float(force_key[1:])
                    for rank, material_data in enumerate(material_list, 1):
                        all_ranks.append({
                            'size': size,
                            'force': force,
                            'material': material_data['material'],
                            'rank': rank
                        })
        
        if not all_ranks:
            print("Error: Could not parse any ranking data from files.")
            return

        rank_df = pd.DataFrame(all_ranks)

        correlate_by = plot_config.get('correlate_by')

        if correlate_by == 'size':
            self._plot_correlation_by_size(rank_df, plot_config)
        elif correlate_by == 'force':
            self._plot_correlation_by_force(rank_df, plot_config)
        elif correlate_by == 'pairwise':
            self._generate_force_vs_force_correlation_heatmap(rank_df, plot_config)
        else:
            print(f"Error: Unknown correlation type '{correlate_by}'. Use 'size', 'force', or 'pairwise'.")

    def _generate_force_vs_force_correlation_heatmap(self, rank_df, plot_config):
        """
        Generates a heatmap correlating ranks between two sizes across all forces.
        X-axis: Forces for size 1. Y-axis: Forces for size 2.
        """
        sizes_to_compare = plot_config.get('sizes_to_compare')
        if not sizes_to_compare or len(sizes_to_compare) != 2:
            print("Error: 'pairwise' correlation requires 'sizes_to_compare' to be a list of two sizes.")
            return

        size1, size2 = sizes_to_compare
        print(f"\n--- Generating Force vs. Force Correlation Heatmap: {size1} vs {size2} ---")

        # Separate the data for each size
        df1 = rank_df[rank_df['size'] == size1]
        df2 = rank_df[rank_df['size'] == size2]

        forces1 = sorted(df1['force'].unique())
        forces2 = sorted(df2['force'].unique())

        if not forces1 or not forces2:
            print(f"Error: No force data available for one or both sizes: {size1}, {size2}")
            return

        # Initialize an empty correlation matrix
        corr_matrix = pd.DataFrame(index=forces2, columns=forces1, dtype=float)

        # Iterate over each pair of forces to calculate the correlation
        for f1 in forces1:
            for f2 in forces2:
                # Get the rank series for each scenario, handling potential duplicates
                ranks1 = df1[df1['force'] == f1].groupby('material')['rank'].mean()
                ranks2 = df2[df2['force'] == f2].groupby('material')['rank'].mean()

                # Combine, keeping only materials present in both
                combined_ranks = pd.DataFrame({'rank1': ranks1, 'rank2': ranks2}).dropna()

                if len(combined_ranks) > 1:
                    # Calculate Spearman correlation for this pair of forces
                    correlation = combined_ranks['rank1'].corr(combined_ranks['rank2'], method='spearman')
                    corr_matrix.loc[f2, f1] = correlation
                else:
                    corr_matrix.loc[f2, f1] = np.nan

        # Plotting the heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='crest', 
            cbar_kws={'label': "Spearman's Rank Correlation"},
            linewidths=.5
        )
        
        plt.title(f'Force vs. Force Rank Correlation ({size1} vs {size2})')
        plt.xlabel(f'Force (nN) for {size1}')
        plt.ylabel(f'Force (nN) for {size2}')
        plt.tight_layout()

        filename = plot_config.get('filename', f'force_vs_force_corr_{size1}_vs_{size2}.png')
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        print(f"Saved force vs. force correlation heatmap to {output_path}")
        plt.close()

    def _plot_correlation_by_size(self, rank_df, plot_config):
        """Plots the correlation of material ranks across different sizes for a fixed force."""
        force_to_compare = plot_config.get('correlation_force', 30)
        print(f"Generating rank correlation across sizes for force = {force_to_compare}nN")

        # Filter for the specific force
        df_filtered = rank_df[rank_df['force'] == force_to_compare]
        
        if df_filtered.empty:
            print(f"Error: No data found for force {force_to_compare}nN to correlate by size.")
            return

        # Pivot the table to have materials as rows, sizes as columns, and ranks as values
        # Use pivot_table to handle potential duplicates by averaging them.
        pivot_df = df_filtered.pivot_table(index='material', columns='size', values='rank', aggfunc='mean')
        
        # Drop materials that don't have a rank for all sizes (optional, but good for clean correlation)
        pivot_df.dropna(inplace=True)

        if len(pivot_df) < 2 or len(pivot_df.columns) < 2:
            print("Error: Not enough overlapping data (materials or sizes) to create a correlation matrix.")
            return

        # Calculate Spearman's rank correlation
        corr_matrix = pivot_df.corr(method='spearman')

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='crest', fmt='.2f')
        plt.title(f'Material Friction Rank Correlation Across Sizes\n(Force = {force_to_compare}nN)')
        plt.tight_layout()

        filename = plot_config.get('filename', f'rank_correlation_by_size_f{force_to_compare}.png')
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        print(f"Saved size correlation plot to {output_path}")
        plt.close()

    def _plot_correlation_by_force(self, rank_df, plot_config):
        """Plots the correlation of material ranks across different forces for each size."""
        print("Generating rank correlation across forces for each size.")

        for size, group in rank_df.groupby('size'):
            # Pivot the table to have materials as rows, forces as columns, and ranks as values
            pivot_df = group.pivot_table(index='material', columns='force', values='rank', aggfunc='mean')
            pivot_df.dropna(inplace=True)

            if len(pivot_df) < 2 or len(pivot_df.columns) < 2:
                print(f"Warning: Not enough overlapping data for size '{size}' to create a correlation matrix. Skipping.")
                continue

            # Calculate Spearman's rank correlation
            corr_matrix = pivot_df.corr(method='spearman')

            # Plot the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='crest', fmt='.2f')
            plt.title(f'Material Friction Rank Correlation Across Forces\n(Size = {size})')
            plt.xlabel("Force (nN)")
            plt.ylabel("Force (nN)")
            plt.tight_layout()

            filename = plot_config.get('filename_prefix', 'rank_correlation_by_force') + f'_{size}.png'
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path)
            print(f"Saved force correlation plot for size '{size}' to {output_path}")
            plt.close()

    def rank_friction(self):
        """Ranks materials by mean friction for each size and force, and exports to JSON."""
        print("\n--- Generating Friction Rankings ---")
        summary_df = self._get_summary_data_df()
        if summary_df.empty:
            print("No summary data available to generate friction rankings.")
            return

        # Filter for baseline runs: angle == 0 and meaningful friction
        rank_df = summary_df[(summary_df['angle'] == 0) & (summary_df['lf'] > 0)].copy()

        if rank_df.empty:
            print("No valid friction data (angle=0, lf>0) found to generate rankings.")
            return

        # Ensure required columns exist
        required_cols = ['size', 'force', 'id', 'lf']
        if not all(col in rank_df.columns for col in required_cols):
            print(f"Error: Missing one or more required columns for ranking. Needed: {required_cols}")
            return

        # Group by size first to create separate files
        for size, group in rank_df.groupby('size'):
            friction_ranking = {}
            # Then group by force to create rankings for each load
            for force, force_group in group.groupby('force'):
                # Sort by mean friction ('lf') and select relevant columns
                sorted_materials = force_group.sort_values('lf')[['id', 'lf']]
                
                # Rename columns for the final JSON output
                sorted_materials = sorted_materials.rename(columns={'id': 'material', 'lf': 'mean_friction'})
                
                # Convert to the desired list of records format
                friction_ranking[f'f{force}'] = sorted_materials.to_dict('records')

            if friction_ranking:
                filename = f'friction_ranking_{size}.json'
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump(friction_ranking, f, indent=4)
                print(f"Friction ranking for size '{size}' saved to {filepath}")

    def generate_plot(self, plot_config):
        """
        Generates a plot based on a configuration dictionary.
        Dispatches to the appropriate plot generation method.
        """
        plot_type = plot_config.get('plot_type', 'summary') # Default to summary
        if plot_type == 'summary':
            self._generate_summary_plot(plot_config)
        elif plot_type == 'timeseries':
            self._generate_timeseries_plot(plot_config)
        elif plot_type == 'rank_friction':
            self.rank_friction()
        elif plot_type == 'correlation':
            self._generate_correlation_plots(plot_config)
        else:
            print(f"Error: Unknown plot type '{plot_type}'")

def main():
    parser = argparse.ArgumentParser(description="Generate plots from simulation data.")
    parser.add_argument('plot_config', help="Path to the JSON file with plot configurations.")
    parser.add_argument('--output_dir', default='plots', help="Directory to save the plots.")
    args = parser.parse_args()

    try:
        with open(args.plot_config, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading or parsing plot config file {args.plot_config}: {e}")
        return

    data_dirs = config.get('data_dirs', [])
    labels = config.get('labels', [])
    plots = config.get('plots', [])

    if not data_dirs or not labels or not plots:
        print("Error: 'data_dirs', 'labels', and 'plots' must be defined in the config file.")
        return
    
    if len(data_dirs) != len(labels):
        print("Error: The number of 'data_dirs' must match the number of 'labels'.")
        return

    plotter = Plotter(data_dirs, labels, args.output_dir)
    
    for plot_config in plots:
        plotter.generate_plot(plot_config)

if __name__ == '__main__':
    main()
