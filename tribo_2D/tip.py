from tribo_2D import model_init

class Tip_simulation(model_init.model_init):
    """
    A class to generate a simulation cell for the Prandtl-Tomlinson
    model in LAMMPS simulations.

    This class reads configuration data, materials, and potential files,
    builds the necessary atomic structures (e.g., 2D material, substrate, tip),
    and prepares the directory structure and scripts needed for
    simulation setup, execution, and post-processing.

    """

    def __init__(self, input_file):
        """
        Initialize the model with the parent class model_init

        Args:
            input_file (str): Path to the input configuration file.
        """

        super().__init__(input_file, model='tip')