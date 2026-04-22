import pytest
import subprocess
from unittest.mock import patch
from src.interfaces.atomsk import AtomskWrapper, AtomskError

def test_atomsk_binary_not_found():
    """Ensure error is raised if atomsk is missing."""
    with patch("shutil.which", return_value=None), \
         patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="Atomsk binary not found"):
            AtomskWrapper()

def test_atomsk_run_success():
    """Test generic run command construction."""
    with patch("shutil.which", return_value="/usr/bin/atomsk"):
        wrapper = AtomskWrapper()
        
        with patch("subprocess.run") as mock_run:
            wrapper.run(["input.xsf", "output.cfg"])
            
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "/usr/bin/atomsk"
            assert "-ow" in args
            assert "-v" in args
            assert "0" in args
            assert "input.xsf" in args
            assert "output.cfg" in args

def test_create_slab():
    """Test create_slab command logic."""
    with patch("shutil.which", return_value="/bin/atomsk"):
        wrapper = AtomskWrapper()
        with patch("subprocess.run") as mock_run:
            wrapper.create_slab("cell.cif", "out.lmp")
            args = mock_run.call_args[0][0]
            # Expected: atomsk cell.cif -duplicate 2 2 1 -orthogonal-cell out.lmp
            assert "-orthogonal-cell" in args
            assert "-duplicate" in args


def test_charge2atom_uses_exec_args():
    """Ensure charge2atom avoids shell=True and passes argv list."""
    with patch("shutil.which", return_value="/bin/atomsk"):
        wrapper = AtomskWrapper()
        with patch("subprocess.run") as mock_run:
            wrapper.charge2atom("input data.lmp")
            mock_run.assert_called_once()

            called_args = mock_run.call_args.kwargs
            assert called_args["check"] is True
            assert called_args["stdout"] == subprocess.DEVNULL
            assert called_args["stderr"] == subprocess.DEVNULL

            cmd = mock_run.call_args.args[0]
            assert isinstance(cmd, list)
            assert cmd[0] == "lmp_charge2atom.sh"
            assert "input data.lmp" in cmd[1]