"""FrictionProvenanceData - Stores provenance files for reproducibility.

This node stores the files needed to reproduce a simulation:
- config.ini and settings.yaml
- CIF files (crystal structures)
- Potential files (.sw, .tersoff, etc.)
- Materials list files
"""

import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from aiida.orm import Data


class FrictionProvenanceData(Data):
    """AiiDA Data node storing provenance files for reproducibility.
    
    This node is populated from a provenance/ folder created by FrictionSim2D
    during simulation setup. The folder structure is:
    
        provenance/
            config.ini
            settings.yaml (optional)
            materials_list.txt (optional)
            cif/
                material.cif
                ...
            potentials/
                material.sw
                ...
    """
    
    def __init__(self, **kwargs):
        """Initialize the provenance data node."""
        super().__init__(**kwargs)
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def simulation_type(self) -> str:
        """Type of simulation ('afm' or 'sheetonsheet')."""
        return self.base.attributes.get('simulation_type', 'afm')
    
    @simulation_type.setter
    def simulation_type(self, value: str):
        self.base.attributes.set('simulation_type', value)
    
    @property
    def config_filename(self) -> str:
        """Original filename of config file."""
        return self.base.attributes.get('config_filename', 'config.ini')
    
    @property
    def materials_list(self) -> List[str]:
        """List of materials if a materials_list file was used."""
        return self.base.attributes.get('materials_list', [])
    
    @property
    def cif_files(self) -> Dict[str, str]:
        """Dictionary mapping CIF filenames to their checksums."""
        return self.base.attributes.get('cif_files', {})
    
    @property
    def potential_files(self) -> Dict[str, str]:
        """Dictionary mapping potential filenames to their checksums."""
        return self.base.attributes.get('potential_files', {})
    
    @property
    def file_count(self) -> int:
        """Total number of stored files."""
        return len(self.cif_files) + len(self.potential_files)
    
    # =========================================================================
    # FILE METHODS
    # =========================================================================
    
    def _compute_checksum(self, content: bytes) -> str:
        """Compute SHA-256 checksum of file content."""
        return hashlib.sha256(content).hexdigest()
    
    def add_file(self, filepath: Path, category: str) -> str:
        """Add a file to the provenance store.
        
        Args:
            filepath: Path to the file
            category: 'cif', 'potential', or 'config'
            
        Returns:
            The checksum of the stored file
        """
        filepath = Path(filepath)
        content = filepath.read_bytes()
        checksum = self._compute_checksum(content)
        
        filename = filepath.name
        repo_path = f'{category}/{filename}'
        
        self.base.repository.put_object_from_filelike(
            filepath.open('rb'), repo_path
        )
        
        # Update registry
        if category == 'cif':
            files = dict(self.cif_files)
            files[filename] = checksum
            self.base.attributes.set('cif_files', files)
        elif category == 'potential':
            files = dict(self.potential_files)
            files[filename] = checksum
            self.base.attributes.set('potential_files', files)
        
        return checksum
    
    def get_file_content(self, filename: str, category: str) -> bytes:
        """Get the content of a stored file."""
        repo_path = f'{category}/{filename}'
        with self.base.repository.open(repo_path, 'rb') as f:
            return f.read()
    
    def export_to_directory(self, output_dir: Path) -> Dict[str, Path]:
        """Export all stored files to recreate the provenance folder."""
        output_dir = Path(output_dir)
        exported = {}
        
        # Export CIF files
        cif_dir = output_dir / 'cif'
        cif_dir.mkdir(parents=True, exist_ok=True)
        for filename in self.cif_files:
            content = self.get_file_content(filename, 'cif')
            out_path = cif_dir / filename
            out_path.write_bytes(content)
            exported[filename] = out_path
        
        # Export potential files
        pot_dir = output_dir / 'potentials'
        pot_dir.mkdir(parents=True, exist_ok=True)
        for filename in self.potential_files:
            content = self.get_file_content(filename, 'potential')
            out_path = pot_dir / filename
            out_path.write_bytes(content)
            exported[filename] = out_path
        
        return exported
    
    # =========================================================================
    # FACTORY METHOD
    # =========================================================================
    
    @classmethod
    def from_provenance_folder(cls, 
                                provenance_dir: Path,
                                simulation_type: str = 'afm') -> 'FrictionProvenanceData':
        """Create provenance node from a provenance folder.
        
        Args:
            provenance_dir: Path to the provenance folder created by FrictionSim2D
            simulation_type: 'afm' or 'sheetonsheet'
            
        Returns:
            A new FrictionProvenanceData instance
        """
        provenance_dir = Path(provenance_dir)
        node = cls()
        node.simulation_type = simulation_type
        
        # Find and store config file
        config_files = list(provenance_dir.glob('*.ini'))
        if config_files:
            config_path = config_files[0]
            node.base.attributes.set('config_filename', config_path.name)
            node.add_file(config_path, 'config')
        
        # Store settings.yaml if present
        settings_path = provenance_dir / 'settings.yaml'
        if settings_path.exists():
            node.add_file(settings_path, 'config')
        
        # Read materials list if present
        for list_file in provenance_dir.glob('*.txt'):
            if 'material' in list_file.name.lower():
                materials = [
                    line.strip()
                    for line in list_file.read_text().splitlines()
                    if line.strip() and not line.startswith('#')
                ]
                node.base.attributes.set('materials_list', materials)
                node.add_file(list_file, 'config')
                break
        
        # Add CIF files
        cif_dir = provenance_dir / 'cif'
        if cif_dir.exists():
            for cif_file in cif_dir.glob('*.cif'):
                node.add_file(cif_file, 'cif')
        
        # Add potential files
        pot_dir = provenance_dir / 'potentials'
        if pot_dir.exists():
            for pot_file in pot_dir.iterdir():
                if pot_file.is_file():
                    node.add_file(pot_file, 'potential')
        
        return node
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metadata as a dictionary."""
        return {
            'uuid': str(self.uuid),
            'simulation_type': self.simulation_type,
            'config_filename': self.config_filename,
            'materials_list': self.materials_list,
            'cif_files': self.cif_files,
            'potential_files': self.potential_files,
            'file_count': self.file_count,
        }
    
    def __repr__(self) -> str:
        n_materials = len(self.materials_list) if self.materials_list else 0
        return (
            f"<FrictionProvenanceData: {self.simulation_type} "
            f"({n_materials} materials, {self.file_count} files)>"
        )
