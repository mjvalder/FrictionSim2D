import pytest
from unittest.mock import patch
from src.interfaces.atomsk import AtomskWrapper

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


import os
import subprocess
from pathlib import Path

from src.interfaces.atomsk import AtomskError


def _wrapper(monkeypatch, calls):
    """A wrapper whose subprocess.run is recorded, never executed."""
    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    return AtomskWrapper(executable_path='/opt/fake/atomsk')


def test_executable_resolves_from_atomsk_path_env(monkeypatch):
    monkeypatch.setenv('ATOMSK_PATH', '/opt/env/atomsk')
    assert AtomskWrapper().executable == '/opt/env/atomsk'


def test_run_verbose_streams_output_instead_of_capturing(monkeypatch):
    calls = []
    w = _wrapper(monkeypatch, calls)
    w.run(['a.cif', 'b.lmp'], verbose=True)
    _, kwargs = calls[0]
    assert kwargs['stdout'] is None and kwargs['stderr'] is None


def test_run_quiet_captures_output(monkeypatch):
    calls = []
    w = _wrapper(monkeypatch, calls)
    w.run(['a.cif', 'b.lmp'])
    _, kwargs = calls[0]
    assert kwargs['stdout'] is subprocess.PIPE and kwargs['stderr'] is subprocess.PIPE


def test_run_failure_with_stderr_reports_it(monkeypatch):
    def failing_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(3, cmd, stderr='bad lattice')
    monkeypatch.setattr(subprocess, 'run', failing_run)
    w = AtomskWrapper(executable_path='/opt/fake/atomsk')
    with pytest.raises(AtomskError, match=r'(?s)exit code 3.*bad lattice'):
        w.run(['a.cif', 'b.lmp'])


def test_run_failure_without_stderr_hints_verbose(monkeypatch):
    def failing_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(2, cmd, stderr='')
    monkeypatch.setattr(subprocess, 'run', failing_run)
    w = AtomskWrapper(executable_path='/opt/fake/atomsk')
    with pytest.raises(AtomskError, match='verbose=True'):
        w.run(['a.cif', 'b.lmp'])


def test_remove_existing_output_deletes_a_stale_output_file(tmp_path, monkeypatch):
    calls = []
    w = _wrapper(monkeypatch, calls)
    src = tmp_path / 'in.cif'; src.write_text('x')
    out = tmp_path / 'out.lmp'; out.write_text('stale')
    w._remove_existing_output([str(src), '-duplicate', '2', '2', '1', str(out)])
    assert not out.exists()


def test_remove_existing_output_keeps_the_file_when_output_is_the_input(tmp_path, monkeypatch):
    calls = []
    w = _wrapper(monkeypatch, calls)
    f = tmp_path / 'same.lmp'; f.write_text('keep me')
    w._remove_existing_output([str(f), '-center', 'com', str(f)])
    assert f.read_text() == 'keep me'


def test_remove_existing_output_ignores_empty_args(monkeypatch):
    calls = []
    w = _wrapper(monkeypatch, calls)
    w._remove_existing_output([])   # must not raise


def test_convert_appends_options_after_paths(monkeypatch):
    calls = []
    w = _wrapper(monkeypatch, calls)
    w.convert('a.cif', 'b.xsf', options=['-wrap'])
    assert calls[0][0][-3:] == ['a.cif', 'b.xsf', '-wrap']


def test_orthogonalize_emits_orthogonal_cell_flag(monkeypatch):
    calls = []
    w = _wrapper(monkeypatch, calls)
    w.orthogonalize('a.cif', 'b.lmp')
    assert calls[0][0][-3:] == ['a.cif', '-orthogonal-cell', 'b.lmp']


def test_duplicate_with_center_adds_center_com(monkeypatch):
    calls = []
    w = _wrapper(monkeypatch, calls)
    w.duplicate('a.cif', 'b.lmp', 2, 3, 1, center=True)
    assert calls[0][0][-7:] == ['-duplicate', '2', '3', '1', '-center', 'com', 'b.lmp']


def test_center_in_place_goes_through_a_temp_file(tmp_path, monkeypatch):
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        Path(cmd[-1]).write_text('centered')          # atomsk writes the temp output
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    w = AtomskWrapper(executable_path='/opt/fake/atomsk')
    f = tmp_path / 'sys.lmp'; f.write_text('raw')
    w.center(f, f)
    assert f.read_text() == 'centered'
    assert not (tmp_path / '.tmp_sys.lmp').exists()
    assert calls[0][-1].endswith('.tmp_sys.lmp')


def test_center_to_a_different_file_runs_directly(monkeypatch):
    calls = []
    w = _wrapper(monkeypatch, calls)
    w.center('a.lmp', 'b.lmp')
    assert calls[0][0][-4:] == ['a.lmp', '-center', 'com', 'b.lmp']
