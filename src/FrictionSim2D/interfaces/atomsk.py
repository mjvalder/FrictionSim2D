"""Interface wrapper for the Atomsk executable.

This module handles the interaction with the external 'atomsk' binary.
It is responsible for locating the executable and running commands to 
generate, manipulate, and convert atomic structures.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

class AtomskError(RuntimeError):
    """Raised when Atomsk execution fails."""
    pass

class AtomskWrapper:
    """Wraps the Atomsk binary for Pythonic usage."""

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
        cmd = [self.executable] + [str(a) for a in args]
        
        # Default to quiet execution unless verbose requested
        stdout_setting = None if verbose else subprocess.PIPE
        stderr_setting = None if verbose else subprocess.PIPE

        try:
            subprocess.run(
                cmd, 
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

    def merge(self, input_files: List[Union[str, Path]], output_file: Union[str, Path]) -> None:
        """Merges multiple structures into one.
        
        Corresponds to: atomsk --merge file1 file2 ... output
        """
        if len(input_files) < 2:
            raise ValueError("Merge requires at least two input files.")
            
        args = ["--merge", str(len(input_files))]
        args.extend([str(f) for f in input_files])
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
        self.run([str(input_file), "-center", "com", str(output_file)])

    def remove_properties(self, input_file: Union[str, Path], output_file: Union[str, Path], properties: List[str]) -> None:
        """Removes specific auxiliary properties (columns) from the file.
        
        Useful for stripping charges ('q') for potentials that don't support them.
        
        Args:
            input_file: Path to input.
            output_file: Path to output.
            properties: List of properties to remove (e.g. ['q', 'vx']).
        """
        props_str = ",".join(properties)
        self.run([str(input_file), "-properties", "remove", props_str, str(output_file)])