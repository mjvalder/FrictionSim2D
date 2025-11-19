import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock
import sys

# Add source directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from FrictionSim2D.interfaces.atomsk import AtomskWrapper

@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for file output."""
    return tmp_path

@pytest.fixture
def mock_atomsk(monkeypatch):
    """Mocks the AtomskWrapper to prevent actual binary execution."""
    mock = MagicMock(spec=AtomskWrapper)
    # Mock the 'executable' attribute as it's accessed directly
    mock.executable = "/mock/path/to/atomsk" 
    
    # Mock run methods to just touch the output file so builders succeed
    def side_effect_run(args, verbose=False):
        # The last arg is usually the output file in our wrapper methods
        output = Path(args[-1])
        if not output.parent.exists():
             output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()

    mock.run.side_effect = side_effect_run
    
    # Also mock specific methods if they don't call run directly in the test scope
    # But our wrapper calls self.run, so mocking run is usually enough.
    # However, create_slab logic:
    def side_effect_create_slab(cif, out, pre_duplicate=None):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).touch()
        
    mock.create_slab.side_effect = side_effect_create_slab
    mock.duplicate.side_effect = lambda inp, out, x, y, z, center=False: Path(out).touch()
    
    return mock