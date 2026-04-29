"""AiiDA Data node for simulation provenance and reproducibility.

Stores all files required to reproduce a friction simulation:
CIF structures, interatomic potentials, config files, and a
manifest mapping files to their simulation components (tip, substrate, etc.).

The node is populated from the ``provenance/`` folder created by
:class:`~src.core.simulation_base.SimulationBase` during simulation setup.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from aiida.orm import Data


class FrictionProvenanceData(Data):
    """AiiDA Data node storing provenance files for reproducibility.

    Populated from the ``provenance/`` folder structure::

        provenance/
            config.json
            settings.yaml  (optional)
            materials_list.txt  (optional)
            manifest.json
            cif/
                material.cif
            potentials/
                material.sw

    Attributes:
        simulation_type: ``'afm'`` or ``'sheetonsheet'``.
        config_filename: Original config file name.
        materials_list: Materials from a materials list file.
        cif_files: Mapping of CIF filenames to SHA-256 checksums.
        potential_files: Mapping of potential filenames to SHA-256 checksums.
        file_manifest: Component-to-file mapping from ``manifest.json``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -- Properties -----------------------------------------------------------

    @property
    def simulation_type(self) -> str:
        """Type of simulation (``'afm'`` or ``'sheetonsheet'``)."""
        return self.base.attributes.get('simulation_type', 'afm')

    @simulation_type.setter
    def simulation_type(self, value: str):
        self.base.attributes.set('simulation_type', value)

    @property
    def config_filename(self) -> str:
        """Original config file name."""
        return self.base.attributes.get('config_filename', 'config.ini')

    @property
    def materials_list(self) -> List[str]:
        """Materials from a materials list file (if used)."""
        return self.base.attributes.get('materials_list', [])

    @property
    def cif_files(self) -> Dict[str, str]:
        """Mapping of CIF filenames to SHA-256 checksums."""
        return self.base.attributes.get('cif_files', {})

    @property
    def potential_files(self) -> Dict[str, str]:
        """Mapping of potential filenames to SHA-256 checksums."""
        return self.base.attributes.get('potential_files', {})

    @property
    def file_manifest(self) -> Dict[str, Any]:
        """Component-to-file mapping from the provenance manifest.

        Example return value::

            {
                'tip': {'potential': {'filename': 'Au.sw', ...}},
                'substrate': {'cif': {'filename': 'MoS2.cif', ...}},
            }
        """
        return self.base.attributes.get('file_manifest', {})

    @property
    def file_count(self) -> int:
        """Total number of stored files (CIF + potential)."""
        return len(self.cif_files) + len(self.potential_files)

    # -- File helpers ---------------------------------------------------------

    @staticmethod
    def _compute_checksum(content: bytes) -> str:
        """Compute SHA-256 checksum of raw bytes."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
        """Parse ``manifest.json`` into a component-to-file mapping.

        Args:
            manifest_path: Path to ``manifest.json``.

        Returns:
            Dict mapping component names to file metadata.
        """
        if not manifest_path.exists():
            return {}

        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        component_map: Dict[str, Any] = {}

        for entry in manifest.get('files', []):
            component = entry.get('component') or entry.get('components')
            if not component:
                continue

            # Normalise to list (manifest may store str or list)
            components = component if isinstance(component, list) else [component]
            for comp in components:
                if comp not in component_map:
                    component_map[comp] = {}
                category = entry.get('category', 'other')
                component_map[comp][category] = {
                    'filename': entry.get('filename'),
                    'original_path': entry.get('original_path'),
                    'checksum': entry.get('checksum'),
                    'added_at': entry.get('added_at'),
                }

        return component_map

    def add_file(self, filepath: Path, category: str) -> str:
        """Add a file to the provenance store.

        Args:
            filepath: Path to the file to store.
            category: One of ``'cif'``, ``'potential'``, or ``'config'``.

        Returns:
            SHA-256 checksum of the stored file.
        """
        filepath = Path(filepath)
        content = filepath.read_bytes()
        checksum = self._compute_checksum(content)

        repo_path = f'{category}/{filepath.name}'
        with filepath.open('rb') as fobj:
            self.base.repository.put_object_from_filelike(fobj, repo_path)

        if category == 'cif':
            files = dict(self.cif_files)
            files[filepath.name] = checksum
            self.base.attributes.set('cif_files', files)
        elif category == 'potential':
            files = dict(self.potential_files)
            files[filepath.name] = checksum
            self.base.attributes.set('potential_files', files)

        return checksum

    def get_file_content(self, filename: str, category: str) -> bytes:
        """Retrieve stored file content.

        Args:
            filename: Name of the file.
            category: Category the file was stored under.

        Returns:
            Raw bytes of the file.
        """
        repo_path = f'{category}/{filename}'
        with self.base.repository.open(repo_path, 'rb') as fobj:
            content = fobj.read()
        if isinstance(content, str):
            return content.encode()
        return content  # type: ignore[return-value]

    def export_to_directory(self, output_dir: Path) -> Dict[str, Path]:
        """Export all stored files to recreate the provenance folder.

        Args:
            output_dir: Target directory for exported files.

        Returns:
            Mapping of filenames to their written paths.
        """
        output_dir = Path(output_dir)
        exported: Dict[str, Path] = {}

        for category, registry in [('cif', self.cif_files),
                                   ('potential', self.potential_files)]:
            sub = 'potentials' if category == 'potential' else category
            cat_dir = output_dir / sub
            cat_dir.mkdir(parents=True, exist_ok=True)
            for filename in registry:
                content = self.get_file_content(filename, category)
                out_path = cat_dir / filename
                out_path.write_bytes(content)
                exported[filename] = out_path

        return exported

    # -- Factory methods ------------------------------------------------------

    @classmethod
    def from_provenance_folder(
        cls,
        provenance_dir: Path,
        simulation_type: str = 'afm',
    ) -> 'FrictionProvenanceData':
        """Create a provenance node from a ``provenance/`` folder.

        Reads ``manifest.json`` for component-to-file mappings, stores all
        CIF, potential, and config files, and records checksums.

        Args:
            provenance_dir: Path to the provenance folder.
            simulation_type: ``'afm'`` or ``'sheetonsheet'``.

        Returns:
            A new (unstored) ``FrictionProvenanceData`` instance.
        """
        provenance_dir = Path(provenance_dir)
        node = cls()
        node.simulation_type = simulation_type

        manifest_path = provenance_dir / 'manifest.json'
        component_map = cls._load_manifest(manifest_path)
        node.base.attributes.set('file_manifest', component_map)

        for pattern in ('*.ini', '*.json'):
            for config_file in provenance_dir.glob(pattern):
                if config_file.name == 'manifest.json':
                    continue
                node.base.attributes.set('config_filename', config_file.name)
                node.add_file(config_file, 'config')

        settings_path = provenance_dir / 'settings.yaml'
        if settings_path.exists():
            node.add_file(settings_path, 'config')

        for list_file in provenance_dir.glob('*.txt'):
            if 'material' in list_file.name.lower():
                materials = [
                    line.strip()
                    for line in list_file.read_text(encoding='utf-8').splitlines()
                    if line.strip() and not line.startswith('#')
                ]
                node.base.attributes.set('materials_list', materials)
                node.add_file(list_file, 'config')
                break

        cif_dir = provenance_dir / 'cif'
        if cif_dir.exists():
            for cif_file in cif_dir.glob('*.cif'):
                node.add_file(cif_file, 'cif')

        pot_dir = provenance_dir / 'potentials'
        if pot_dir.exists():
            for pot_file in pot_dir.iterdir():
                if pot_file.is_file():
                    node.add_file(pot_file, 'potential')

        return node

    # -- Serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export metadata as a plain dictionary."""
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
