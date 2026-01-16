import pytest
from pathlib import Path
from src.core.simulation_base import SimulationBase
from src.core.config import AFMSimulationConfig

# Mock class to allow instantiation of abstract SimulationBase
class ConcreteSimulation(SimulationBase):
    def build(self): pass
    def write_inputs(self): pass

def test_directory_creation(tmp_path, afm_config):
    # afm_config comes from your conftest.py fixture
    builder = ConcreteSimulation(afm_config, output_dir=tmp_path)
    
    builder.setup_directories(['visuals', 'results'])
    
    assert (tmp_path / 'visuals').exists()
    assert (tmp_path / 'results').exists()

def test_template_rendering(tmp_path, afm_config):
    builder = ConcreteSimulation(afm_config, output_dir=tmp_path)
    
    # Render a real template or a mock one if you inject a mock env
    # Testing a real template:
    context = {'atom_style': 'molecular'}
    result = builder.render_template('common/init.lmp', context)
    
    assert "atom_style      molecular" in result