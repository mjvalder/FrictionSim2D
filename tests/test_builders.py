import pytest
from pathlib import Path
from unittest.mock import patch

from src.builders.afm import AFMSimulation
from src.builders import components
from src.core.config import AFMSimulationConfig, load_settings

# Reusing valid config data
VALID_DATA = {
    'general': {'temp': 300.0, 'force': 10.0},
    'tip': {'mat': 'Si', 'pot_type': 'tersoff', 'pot_path': 'Si.tersoff', 'cif_path': 'Si.cif', 'r': 10.0, 'cspring': 10, 's': 10},
    'sub': {'mat': 'Si', 'pot_type': 'tersoff', 'pot_path': 'Si.tersoff', 'cif_path': 'Si.cif', 'thickness': 10.0},
    'sheet': {'mat': 'MoS2', 'pot_type': 'sw', 'pot_path': 'MoS2.sw', 'cif_path': 'MoS2.cif', 'x': 50.0, 'y': 50.0, 'layers': [1]},
    'settings': load_default_settings()
}

@pytest.fixture
def afm_config():
    # We need to create a full config object
    # Note: We pass the 'settings' object extracted above
    return AFMSimulationConfig(**VALID_DATA)

@patch("FrictionSim2D.builders.components.run_lammps_commands") # Mock LAMMPS
def test_component_build_tip(mock_lmp, afm_config, mock_atomsk, temp_dir):
    """Test tip builder logic."""
    settings = afm_config.settings
    
    # We mock resources lookup to avoid file errors
    with patch("FrictionSim2D.builders.components.resources.files") as mock_res:
        mock_res.return_value.joinpath.return_value = Path("mock_path.cif")
        
        path, radius = components.build_tip(
            afm_config.tip, mock_atomsk, temp_dir, settings
        )
        
        assert radius == 10.0
        assert path == temp_dir / "tip.lmp"
        # Check that LAMMPS was called for carving
        mock_lmp.assert_called()

def test_afm_builder_structure(afm_config, mock_atomsk, temp_dir):
    """Test the full AFM builder orchestration."""
    
    # Mock components functions to avoid complex logic
    with patch("FrictionSim2D.builders.components.build_tip", return_value=(Path("tip.lmp"), 10.0)), \
         patch("FrictionSim2D.builders.components.build_substrate", return_value=Path("sub.lmp")), \
         patch("FrictionSim2D.builders.components.build_sheet", return_value=(Path("sheet.lmp"), {'xlo':0,'xhi':10,'ylo':0,'yhi':10,'zlo':0,'zhi':10}, 3.0)), \
         patch("FrictionSim2D.builders.afm.AFMSimulation._initialize_component_metadata"):
         
        builder = AFMSimulation(afm_config, output_dir=temp_dir)
        builder.atomsk = mock_atomsk # Inject mock
        
        # Run build
        builder.build()
        
        # Verify directories created
        assert (temp_dir / "visuals").exists()
        assert (temp_dir / "results").exists()
        
        # Verify input scripts generated
        assert (temp_dir / "system.in").exists()
        assert (temp_dir / "slide.in").exists()