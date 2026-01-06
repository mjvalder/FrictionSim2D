"""Interface wrapper for the Atomsk executable.

This module handles the interaction with the external 'atomsk' binary.
It is responsible for locating the executable and running commands to 
generate, manipulate, and convert atomic structures.
"""

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Union

class AtomskError(RuntimeError):
    """Raised when Atomsk execution fails."""
    pass

class AtomskWrapper:
    """Wraps the Atomsk binary for Pythonic usage."""
    
    # Class-level lock to serialize Atomsk calls and prevent concurrent execution
    # This prevents Fortran memory allocation errors when Atomsk is called rapidly
    _lock = threading.Lock()

    def __init__(self, executable_path: Optional[Union[str, Path]] = None):
        """Initialize the wrapper.

        Args:
            executable_path: Explicit path to the atomsk binary. If None,
                             attempts to find it in env vars or system PATH.
        
        Raises:
            RuntimeError: If atomsk cannot be found.
        """
        self.executable = (
            executable_path 
            or os.environ.get("ATOMSK_PATH") 
            or shutil.which("atomsk")
        )

        if not self.executable:
            raise RuntimeError(
                "Atomsk binary not found! Please install Atomsk and ensure it is "
                "in your PATH, or set the ATOMSK_PATH environment variable.\n"
                "Installation guide: https://atomsk.univ-lille.fr/installation.php"
            )
        
        self.executable = str(self.executable)

    def run(self, args: List[str], verbose: bool = False) -> None:
        """Executes a raw command with Atomsk.

        Args:
            args: List of command line arguments (e.g. ['file.cif', '-duplicate', ...])
            verbose: If True, prints Atomsk's stdout/stderr.
        """
        # Build command with -ow (overwrite) and -v 0 (silent) flags
        cmd = [self.executable, '-ow', '-v', '0'] + [str(a) for a in args]
        
        # Atomsk hangs waiting for user input if output file already exists.
        # Find and delete output file before running to prevent this.
        # The output file is typically the last non-flag argument.
        self._remove_existing_output(args)
        
        # Default to quiet execution unless verbose requested
        stdout_setting = None if verbose else subprocess.PIPE
        stderr_setting = None if verbose else subprocess.PIPE

        # Serialize Atomsk calls to prevent Fortran memory allocation errors
        # Atomsk has internal state issues when called concurrently
        with self._lock:
            try:
                # Pipe "n" to stdin to auto-answer prompts (e.g., triclinic skew warnings)
                subprocess.run(
                    cmd, 
                    input="n\n",
                    check=True, 
                    stdout=stdout_setting, 
                    stderr=stderr_setting,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                # Try to capture stderr if we silenced it, to show the user why it failed
                error_msg = f"Atomsk command failed with exit code {e.returncode}"
                if e.stderr:
                    error_msg += f"\nError Output: {e.stderr}"
                elif not verbose:
                    error_msg += ". Run with verbose=True to see details."
                raise AtomskError(error_msg) from e
    
    def _remove_existing_output(self, args: List[str]) -> None:
        """Remove existing output file to prevent atomsk from prompting for input.
        
        Atomsk will ask the user which file to use if both input and output exist.
        To prevent this hang, we delete any existing output file beforehand.
        However, we must NOT delete if input == output (in-place modification).
        
        Args:
            args: The list of arguments being passed to atomsk.
        """
        if not args:
            return
        
        # Find input file (first non-flag argument)
        input_file = None
        for arg in args:
            arg_str = str(arg)
            if not arg_str.startswith('-') and ('.' in arg_str or '/' in arg_str):
                input_file = Path(arg_str).resolve() if Path(arg_str).exists() else Path(arg_str)
                break
            
        # The output file is typically the last argument that looks like a file path
        # (not starting with '-')
        for arg in reversed(args):
            arg_str = str(arg)
            # Skip flags
            if arg_str.startswith('-'):
                continue
            # Check if it looks like a file path (has an extension or path separator)
            if '.' in arg_str or '/' in arg_str:
                output_path = Path(arg_str)
                # Don't delete if input == output (in-place modification)
                if input_file is not None:
                    input_resolved = input_file.resolve() if input_file.exists() else input_file
                    output_resolved = output_path.resolve() if output_path.exists() else output_path
                    if input_resolved == output_resolved:
                        return  # In-place modification, don't delete
                        
                if output_path.exists() and output_path.is_file():
                    output_path.unlink()
                break  # Only delete the first (last in original order) file-like argument

    def convert(self, input_file: Union[str, Path], output_file: Union[str, Path], options: List[str] = None) -> None:
        """Converts a file from one format to another, optionally applying flags.
        
        Args:
            input_file: Path to source file.
            output_file: Path to destination file (format inferred from extension).
            options: Additional flags (e.g. ['-v', '0']).
        """
        args = [str(input_file), str(output_file)]
        if options:
            args.extend(options)
        self.run(args)

    def orthogonalize(self, input_file: Union[str, Path], output_file: Union[str, Path]) -> None:
        """Converts a structure to an orthogonal cell.
        
        Corresponds to: atomsk input -orthogonal-cell output
        """
        self.run([str(input_file), "-orthogonal-cell", str(output_file)])

    def duplicate(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        nx: int,
        ny: int,
        nz: int,
        center: bool = False
    ) -> None:
        """Duplicates the system in x, y, z directions.
        
        Args:
            nx, ny, nz: Multiplication factors.
            center: If True, adds '-center com'.
        """
        args = [str(input_file), "-duplicate", str(nx), str(ny), str(nz)]
        if center:
            args.extend(["-center", "com"])
        args.append(str(output_file))
        self.run(args)

    def create_slab(
        self,
        cif_path: Union[str, Path],
        output_path: Union[str, Path],
        pre_duplicate: List[int] = [2, 2, 1]
    ) -> None:
        """Helper to create an orthogonalized slab from a CIF.
        
        This replicates the first step of the old slab_generator:
        1. Duplicates (usually 2x2x1) to ensure enough atoms for orthogonalization.
        2. Orthogonalizes.
        3. Writes to output.
        """
        # Atomsk command: atomsk {cif} -duplicate 2 2 1 -orthogonal-cell {out}
        args = [
            str(cif_path),
            "-duplicate", *map(str, pre_duplicate),
            "-orthogonal-cell",
            str(output_path)
        ]
        self.run(args)

    def center(self, input_file: Union[str, Path], output_file: Union[str, Path], axis: str = "z") -> None:
        """Centers the system (usually to Center of Mass).
        
        Args:
            input_file: Path to input.
            output_file: Path to output.
            axis: (Unused by Atomsk's basic -center com, but kept for API consistency)
        """
        import shutil
        
        input_path = Path(input_file).resolve()
        output_path = Path(output_file).resolve()
        
        # Handle in-place modification by using a temp file
        if input_path == output_path:
            temp_output = input_path.parent / f".tmp_{input_path.name}"
            self.run([str(input_path), "-center", "com", str(temp_output)])
            shutil.move(str(temp_output), str(input_path))
        else:
            self.run([str(input_file), "-center", "com", str(output_file)])

    def charge2atom(self, input_file: Union[str, Path], output_file: Union[str, Path], properties: List[str] = None) -> None:
        """Removes charge column from LAMMPS data file using lmp_charge2atom.sh.
        
        Useful for stripping charges for potentials that don't support them.
        
        Args:
            input_file: Path to input file.
            output_file: Path to output file (currently must be same as input for in-place modification).
            properties: Unused, kept for API compatibility.
        """
        input_path = Path(input_file).resolve()
        subprocess.run(f"lmp_charge2atom.sh {input_path}", shell=True, check=True, 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
