"""AiiDA integration module for FrictionSim2D.

Provides functions to register simulations with AiiDA after file generation
and to import completed results. Works in tandem with the HPC manifest
system (:mod:`src.hpc.manifest`) for end-to-end job tracking.

Typical workflow on HPC
-----------------------
1. Generate simulation files::

        FrictionSim2D run afm config.ini --aiida

2. AiiDA registers each simulation (this module).
3. Jobs are submitted via AiiDA CalcJob or manually.
4. After completion, import results::

        FrictionSim2D aiida import ./results
"""

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import AIIDA_AVAILABLE

if TYPE_CHECKING:
    from .data import (
        FrictionProvenanceData,
        FrictionSimulationData,
        FrictionResultsData,
    )

logger = logging.getLogger(__name__)

_REGISTRATION_EXCEPTIONS = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
    json.JSONDecodeError,
)


def _sanitize_for_aiida(obj: Any) -> Any:
    """Recursively replace float nan/inf with None so AiiDA can store the value."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_aiida(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_aiida(v) for v in obj]
    return obj


def _require_aiida():
    """Raise if AiiDA is not available."""
    if not AIIDA_AVAILABLE:
        raise ImportError(
            "AiiDA is not installed. Install with: pip install 'FrictionSim2D[aiida]'"
        )


def _ensure_aiida_profile(profile_name: Optional[str] = None) -> None:
    """Load an AiiDA profile if one is not already active.

    Args:
        profile_name: Profile to load. ``None`` loads the default profile.
    """
    try:
        from aiida.manage import get_manager  # pylint: disable=import-outside-toplevel
        manager = get_manager()
        # If a profile storage is already loaded, nothing to do.
        if manager.get_profile() is not None:
            return
    except Exception:  # pylint: disable=broad-except
        pass

    try:
        from aiida.manage.configuration import load_profile  # pylint: disable=import-outside-toplevel
        load_profile(profile_name)
        logger.debug("Loaded AiiDA profile: %s", profile_name or "(default)")
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            f"Could not load AiiDA profile {profile_name!r}. "
            "Run 'verdi presto' or 'verdi profile list' to check available profiles."
        ) from exc


# =============================================================================
# Registration (after file generation)
# =============================================================================

def register_simulation_batch(
    simulation_dirs: List[Path],
    config_path: Path,
    manifest_path: Optional[Path] = None,
    aiida_profile: Optional[str] = None,
) -> List[str]:
    """Register a batch of simulations with AiiDA.

    Scans each simulation directory for a ``provenance/`` folder and creates
    the corresponding AiiDA nodes. Optionally updates the HPC
    :class:`~src.hpc.manifest.JobManifest` with node UUIDs.

    Args:
        simulation_dirs: Simulation output directories.
        config_path: Path to the original config file.
        manifest_path: Optional path to the HPC manifest JSON. If provided,
            job entries are updated with AiiDA node UUIDs.
        aiida_profile: AiiDA profile name to load. Defaults to the configured
            default profile.

    Returns:
        List of created simulation node UUIDs.
    """
    _require_aiida()
    _ensure_aiida_profile(aiida_profile)

    manifest = _load_manifest(manifest_path) if manifest_path else None
    created_uuids: List[str] = []

    for sim_dir in simulation_dirs:
        try:
            uuid = register_single_simulation(sim_dir, config_path)
            if uuid:
                created_uuids.append(uuid)
                if manifest:
                    _update_manifest_entry(manifest, sim_dir, uuid)
        except _REGISTRATION_EXCEPTIONS:
            logger.warning("Failed to register %s", sim_dir, exc_info=True)

    if manifest and manifest_path:
        manifest.save(manifest_path)
        logger.info("Updated manifest at %s", manifest_path)

    return created_uuids


def register_single_simulation(
    sim_dir: Path,
    config_path: Path,
) -> Optional[str]:
    """Register a single simulation with AiiDA.

    Creates a ``FrictionProvenanceData`` node from the ``provenance/`` folder
    and a ``FrictionSimulationData`` node populated from ``config.json``.

    Args:
        sim_dir: Simulation output directory.
        config_path: Path to the original config file.

    Returns:
        UUID of the created simulation node, or ``None`` on failure.
    """
    _require_aiida()
    sim_dir = Path(sim_dir)

    prov_dir = sim_dir / 'provenance'
    if not prov_dir.exists():
        logger.warning("No provenance folder in %s — skipping", sim_dir)
        return None

    # --- Provenance node ---
    prov_node = _create_provenance_node(prov_dir, config_path)

    # --- Config data ---
    config_file = prov_dir / 'config.json'
    if not config_file.exists():
        logger.warning("No config.json in %s/provenance", sim_dir)
        return None

    config_data = json.loads(config_file.read_text(encoding='utf-8'))

    # --- Simulation node ---
    from .data import FrictionSimulationData  # pylint: disable=import-outside-toplevel
    sim_node = FrictionSimulationData()
    sim_type = 'afm' if 'tip' in config_data else 'sheetonsheet'
    sim_node.set_from_config(config_data, simulation_type=sim_type)

    # Relative path for portability
    try:
        sim_node.simulation_path = str(sim_dir.relative_to(sim_dir.parent.parent))
    except ValueError:
        sim_node.simulation_path = str(sim_dir)

    sim_node.status = 'prepared'

    if prov_node:
        sim_node.provenance_uuid = str(prov_node.uuid)

    sim_node.store()

    # Update the on-disk manifest with node UUIDs
    _update_provenance_manifest(prov_dir, sim_node, prov_node)

    logger.info("Registered: %s (UUID: %s)", sim_dir.name, sim_node.uuid)
    return str(sim_node.uuid)


def _create_provenance_node(
    prov_dir: Path,
    config_path: Path,
) -> Optional['FrictionProvenanceData']:
    """Create and store a provenance node from the provenance directory.

    Args:
        prov_dir: Path to the ``provenance/`` folder.
        config_path: Original config file path.

    Returns:
        Stored provenance node, or ``None`` on failure.
    """
    try:
        from .data import FrictionProvenanceData  # pylint: disable=import-outside-toplevel
        prov_node = FrictionProvenanceData.from_provenance_folder(prov_dir)
        prov_node.base.attributes.set('config_filename', config_path.name)
        prov_node.store()
        logger.info("Created provenance node: %s", prov_node.uuid)
        return prov_node
    except _REGISTRATION_EXCEPTIONS:
        logger.error("Failed to create provenance node from %s", prov_dir, exc_info=True)
        return None


def _update_provenance_manifest(
    prov_dir: Path,
    sim_node: 'FrictionSimulationData',
    prov_node: Optional['FrictionProvenanceData'],
) -> None:
    """Write AiiDA node UUIDs back into the on-disk manifest.json."""
    manifest_path = prov_dir / 'manifest.json'
    if not manifest_path.exists():
        return

    manifest_data = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest_data['simulation_node_uuid'] = str(sim_node.uuid)
    if prov_node:
        manifest_data['provenance_node_uuid'] = str(prov_node.uuid)
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding='utf-8')


# =============================================================================
# Result import
# =============================================================================

def import_results_to_aiida(
    results_dir: Path,
    aiida_profile: Optional[str] = None,
) -> List[str]:
    """Import completed simulation results into AiiDA.

    Reads results using :class:`~src.postprocessing.read_data.DataReader`
    (which calculates COF and lateral force), then creates
    ``FrictionResultsData`` nodes with automatic summary statistics.

    Args:
        results_dir: Directory containing simulation results.
        aiida_profile: AiiDA profile name to load. Defaults to the configured
            default profile.

    Returns:
        List of created result node UUIDs.
    """
    _require_aiida()
    _ensure_aiida_profile(aiida_profile)
    from ..postprocessing.read_data import DataReader  # pylint: disable=import-outside-toplevel

    reader = DataReader(results_dir=str(results_dir))
    created_uuids: List[str] = []

    for material, size_data in reader.full_data_nested.items():
        for size_key, substrate_data in size_data.items():
            _import_substrate_tree(
                reader, material, size_key, substrate_data, created_uuids
            )

    logger.info("Imported %d result nodes from %s", len(created_uuids), results_dir)
    return created_uuids


def _import_substrate_tree(
    reader, material: str, size_key: str,
    substrate_data: Dict, created_uuids: List[str],
) -> None:
    """Walk the nested DataReader tree and create result nodes.

    Separated from :func:`import_results_to_aiida` to reduce nesting depth.
    """
    for _substrate, tip_data in substrate_data.items():
        for _tip_mat, radius_data in tip_data.items():
            for _radius, layer_data in radius_data.items():
                for layer_key, speed_data in layer_data.items():
                    layers = int(layer_key.replace('l', ''))
                    for speed_key, force_data in speed_data.items():
                        speed = int(speed_key.replace('s', ''))
                        _import_force_angle_data(
                            reader, material, size_key, layers, speed,
                            force_data, created_uuids,
                        )


def _import_force_angle_data(
    reader, material: str, size_key: str, layers: int, speed: int,
    force_data: Dict, created_uuids: List[str],
) -> None:
    """Create result nodes for each force/angle combination."""
    for load_key, angle_data in force_data.items():
        is_pressure = load_key.startswith('p')
        load_val = float(load_key[1:])

        for angle_key, df in angle_data.items():
            angle = int(angle_key.replace('a', ''))
            try:
                node = _create_result_node(
                    reader, material, size_key, layers, speed,
                    load_val, is_pressure, angle, df,
                )
                node.store()
                created_uuids.append(str(node.uuid))
                logger.info(
                    "Imported: %s L%d %s%.1f A%d — COF: %.4f",
                    material, layers,
                    'P' if is_pressure else 'F', load_val, angle,
                    node.mean_cof,
                )
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to import %s/l%d/%s/a%d", material, layers, load_key, angle,
                    exc_info=True,
                )


def _create_result_node(
    reader, material: str, size_key: str, layers: int, speed: int,
    load_val: float, is_pressure: bool, angle: int, df,
) -> 'FrictionResultsData':
    """Construct a single FrictionResultsData node from a DataFrame."""
    from .data import FrictionResultsData  # pylint: disable=import-outside-toplevel
    node = FrictionResultsData()

    node.material = material.replace('_', '-')
    node.layers = layers
    node.speed = speed
    node.angle = angle
    node.size = size_key

    if is_pressure:
        node.base.attributes.set('pressure', load_val)
    else:
        node.force = load_val

    time_series = {col: df[col].tolist() for col in df.columns}
    if reader.time_series:
        time_series['time'] = reader.time_series

    node.time_series = _sanitize_for_aiida(time_series)  # auto-calculates summary stats
    node.is_complete = True

    return node


# =============================================================================
# Manifest helpers
# =============================================================================

def _load_manifest(manifest_path: Path):
    """Load HPC ``JobManifest`` from disk.

    Returns:
        A :class:`~src.hpc.manifest.JobManifest` instance.
    """
    from ..hpc.manifest import JobManifest  # pylint: disable=import-outside-toplevel
    return JobManifest.load(manifest_path)


def _update_manifest_entry(manifest, sim_dir: Path, uuid: str) -> None:
    """Update the manifest job entry for *sim_dir* with the AiiDA UUID."""
    sim_path_str = str(sim_dir)
    sim_name = sim_dir.name
    for job in manifest.jobs:
        if (
            sim_name in job.simulation_path
            or job.simulation_path in sim_path_str
        ):
            job.simulation_node_uuid = uuid


# =============================================================================
# Archive export / import helpers
# =============================================================================

def export_archive(output_path: Path, materials: Optional[List[str]] = None) -> Path:
    """Export FrictionSim2D AiiDA nodes to a portable archive.

    Wraps ``verdi archive create`` to export all FrictionSim2D nodes
    (or a filtered subset) into a single ``.aiida`` archive file suitable
    for transferring between HPC and local AiiDA instances.

    Args:
        output_path: Destination path for the archive file.
        materials: Optional list of materials to filter by. If ``None``,
            all FrictionSim2D nodes are exported.

    Returns:
        Path to the created archive file.
    """
    _require_aiida()
    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel  # pyright: ignore[reportMissingImports]
    from aiida.tools.archive import create_archive  # pylint: disable=import-outside-toplevel  # pyright: ignore[reportMissingImports]

    output_path = Path(output_path)

    from .data import (  # pylint: disable=import-outside-toplevel
        FrictionSimulationData,
        FrictionResultsData,
        FrictionProvenanceData,
    )

    # Collect all FrictionSim2D node PKs
    node_pks = set()
    for node_cls in (FrictionSimulationData, FrictionResultsData, FrictionProvenanceData):
        qb = QueryBuilder()
        qb.append(node_cls, project=['id'])
        if materials and node_cls is FrictionSimulationData:
            qb.add_filter(node_cls, {'attributes.material': {'in': materials}})
        for (pk,) in qb.all():
            node_pks.add(pk)

    if not node_pks:
        logger.warning("No FrictionSim2D nodes found to export")
        return output_path

    create_archive(list(node_pks), filename=output_path)
    logger.info("Exported %d nodes to %s", len(node_pks), output_path)
    return output_path


def import_archive(archive_path: Path) -> int:
    """Import an AiiDA archive into the current profile.

    Wraps ``verdi archive import`` for transferring data from HPC
    to a local mirror database.

    Args:
        archive_path: Path to the ``.aiida`` archive file.

    Returns:
        Number of new nodes imported.
    """
    _require_aiida()
    from aiida.tools.archive import import_archive as aiida_import  # pylint: disable=import-outside-toplevel  # pyright: ignore[reportMissingImports]

    archive_path = Path(archive_path)
    result = aiida_import(archive_path)

    n_imported = getattr(result, 'new_nodes', 0) if result else 0
    logger.info("Imported %d nodes from %s", n_imported, archive_path)
    return n_imported
