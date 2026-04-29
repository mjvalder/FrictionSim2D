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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from . import AIIDA_AVAILABLE
from ..core.path_utils import format_dimension_token

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
# Simulation-set import (new-style, full provenance hierarchy)
# =============================================================================

def import_simulation_set(
    simulation_folder: Path,
    label: str,
    description: str = '',
    aiida_profile: Optional[str] = None,
) -> str:
    """Import a complete simulation set into AiiDA with a full node hierarchy.

    Creates:

    * 1 ``FrictionSimulationSetData`` per run (shared config + label)
    * 1 ``FrictionProvenanceData`` per material (CIF + potential files)
    * 1 ``FrictionSimulationData`` per material (material-level metadata)
    * N ``FrictionResultsData`` per material (one per load/angle combination)

    Every child node carries a ``set_uuid`` attribute pointing back to the
    set node, and every ``FrictionResultsData`` node carries a
    ``simulation_uuid`` pointing to its parent ``FrictionSimulationData``.

    Args:
        simulation_folder: Path to the ``simulation_XXXXXXXX`` directory
            produced by a single ``FrictionSim2D run`` invocation.
        label: Required human-readable label for the simulation set.
        description: Optional free-text description.
        aiida_profile: AiiDA profile name to load.

    Returns:
        UUID of the created ``FrictionSimulationSetData`` node.

    Raises:
        ValueError: If ``label`` is empty or ``simulation_folder`` does not
            exist.
    """
    _require_aiida()
    _ensure_aiida_profile(aiida_profile)

    simulation_folder = Path(simulation_folder)
    if not simulation_folder.is_dir():
        raise ValueError(f"simulation_folder does not exist: {simulation_folder}")
    if not label or not label.strip():
        raise ValueError("label is required and cannot be empty")

    sim_type = _detect_simulation_type(simulation_folder)
    logger.info("Detected simulation type: %s in %s", sim_type, simulation_folder.name)

    # Scan material directories before creating any nodes
    material_prov_map = _find_material_provenance_dirs(simulation_folder, sim_type)
    materials_list = sorted(material_prov_map.keys())
    logger.info(
        "Found %d materials (%s…)",
        len(materials_list),
        ', '.join(materials_list[:4]),
    )

    shared_config = _load_shared_config_from_set(simulation_folder)

    # --- Store the simulation set node first (immutable after store) ---
    from .data import FrictionSimulationSetData  # pylint: disable=import-outside-toplevel
    set_node = FrictionSimulationSetData()
    set_node.label = label.strip()
    set_node.description = description
    set_node.simulation_type = sim_type
    set_node.run_folder = simulation_folder.name
    set_node.batch_path = str(simulation_folder)
    if shared_config:
        set_node.set_from_config(shared_config)
    set_node.materials_list = materials_list
    set_node.store()
    logger.info("Stored simulation set node %s (label='%s')", set_node.uuid, label)

    # --- Read results with DataReader (if any result files exist) ---
    from ..postprocessing.read_data import DataReader  # pylint: disable=import-outside-toplevel
    reader = DataReader(results_dir=str(simulation_folder))
    results_nested = reader.full_data_nested

    # DataReader uses safe material names (dashes → underscores).
    safe_to_original: Dict[str, str] = {
        m.replace('-', '_').replace('/', '__'): m for m in materials_list
    }

    # Process every material found in the directory structure. For materials
    # that have result files (in results_nested) we also create ResultsData
    # nodes; for those that don't we still create Provenance + Simulation nodes
    # (status='prepared') so provenance is fully captured.
    for original_material in materials_list:
        prov_dir = material_prov_map.get(original_material)
        safe_material = original_material.replace('-', '_').replace('/', '__')

        prov_node = _create_set_provenance_node(
            material=original_material,
            prov_dir=prov_dir,
            sim_type=sim_type,
            set_uuid=str(set_node.uuid),
        )

        config_data = _load_material_config(prov_dir) if prov_dir else (shared_config or {})
        has_results = safe_material in results_nested
        status = 'imported' if has_results else 'prepared'
        sim_node = _create_set_simulation_node(
            material=original_material,
            config_data=config_data,
            sim_type=sim_type,
            set_uuid=str(set_node.uuid),
            prov_uuid=str(prov_node.uuid) if prov_node else None,
            status=status,
        )

        result_count = 0
        if has_results:
            result_count = _import_set_results_tree(
                reader=reader,
                safe_material=safe_material,
                original_material=original_material,
                size_data=results_nested[safe_material],
                sim_type=sim_type,
                sim_uuid=str(sim_node.uuid),
                set_uuid=str(set_node.uuid),
            )

        logger.info(
            "  %s → prov=%s sim=%s results=%d",
            original_material,
            prov_node.uuid[:8] if prov_node else 'none',
            sim_node.uuid[:8],
            result_count,
        )

    # --- Create an AiiDA Group so all nodes in this set are visible together
    # in database tools (DBeaver, verdi group list) without relying on UUID
    # attribute cross-references alone.
    try:
        from aiida.orm import Group, QueryBuilder, load_group  # pylint: disable=import-outside-toplevel
        from aiida.common.exceptions import NotExistent  # pylint: disable=import-outside-toplevel
        group_label = f"friction2d/{label.strip()}"
        # Get or create the group (handles re-import without unique constraint errors)
        try:
            group = load_group(label=group_label)
            # Clear existing members so we get a fresh membership list
            group.clear()
        except NotExistent:
            group = Group(label=group_label,
                          description=description or f"{sim_type} simulation set")
            group.store()
        # Collect all child nodes that reference this set
        qb = QueryBuilder()
        from aiida.orm import Data as _Data  # pylint: disable=import-outside-toplevel
        qb.append(_Data, filters={'attributes.set_uuid': str(set_node.uuid)},
                  project=['*'])
        member_nodes = [r[0] for r in qb.all()]
        if member_nodes:
            group.add_nodes(member_nodes)
        group.add_nodes([set_node])
        logger.info(
            "Created/updated group '%s' with %d member nodes",
            group_label, len(member_nodes) + 1,
        )
    except Exception:  # pylint: disable=broad-except
        logger.warning("Failed to create Group for set %s", set_node.uuid, exc_info=True)

    logger.info(
        "Import complete: set=%s, %d materials, label='%s'",
        set_node.uuid,
        len(materials_list),
        label,
    )
    return str(set_node.uuid)


def _detect_simulation_type(simulation_folder: Path) -> str:
    """Return ``'afm'`` or ``'sheetonsheet'`` based on subfolder presence."""
    if (simulation_folder / 'sheetonsheet').is_dir():
        return 'sheetonsheet'
    if (simulation_folder / 'afm').is_dir():
        return 'afm'
    # Fallback: look one level deeper for run output folders
    for child in simulation_folder.iterdir():
        if child.is_dir() and child.name.startswith('simulation_'):
            if (child / 'sheetonsheet').is_dir():
                return 'sheetonsheet'
            if (child / 'afm').is_dir():
                return 'afm'
    raise ValueError(
        f"Could not detect simulation type in {simulation_folder}. "
        "Expected 'sheetonsheet/' or 'afm/' subdirectory."
    )


def _find_material_provenance_dirs(
    simulation_folder: Path,
    sim_type: str,
) -> Dict[str, Optional[Path]]:
    """Return mapping of material name → path to its provenance directory.

    For sheet-on-sheet the expected layout is::

        simulation_XXXXXXXX/sheetonsheet/{material}/{size}/{temp}/provenance/

    For AFM::

        simulation_XXXXXXXX/afm/{material}/{size}/{sub_type}/{temp}/provenance/

    If a material directory exists but has no ``provenance/`` folder (e.g.
    incomplete runs) the value is ``None``.
    """
    type_root = simulation_folder / sim_type
    if not type_root.is_dir():
        return {}

    result: Dict[str, Optional[Path]] = {}

    # Enumerate top-level material directories
    for mat_dir in sorted(type_root.iterdir()):
        if not mat_dir.is_dir():
            continue
        material = mat_dir.name

        # Find the provenance directory – search up to 4 levels deep
        prov_dir: Optional[Path] = None
        for prov_candidate in mat_dir.rglob('provenance'):
            if prov_candidate.is_dir():
                prov_dir = prov_candidate
                break  # take the first one found

        result[material] = prov_dir

    return result


def _load_shared_config_from_set(simulation_folder: Path) -> Optional[Dict[str, Any]]:
    """Load and return the first ``config.json`` found in any provenance dir."""
    for prov_config in simulation_folder.rglob('provenance/config.json'):
        try:
            return json.loads(prov_config.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _load_material_config(prov_dir: Optional[Path]) -> Dict[str, Any]:
    """Load ``config.json`` from a provenance directory."""
    if prov_dir is None:
        return {}
    config_file = prov_dir / 'config.json'
    if not config_file.exists():
        return {}
    try:
        return json.loads(config_file.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _create_set_provenance_node(
    material: str,
    prov_dir: Optional[Path],
    sim_type: str,
    set_uuid: str,
) -> Optional['FrictionProvenanceData']:
    """Create, store and return a ``FrictionProvenanceData`` for one material."""
    from .data import FrictionProvenanceData  # pylint: disable=import-outside-toplevel

    if prov_dir is None or not prov_dir.is_dir():
        logger.warning("No provenance directory for %s — skipping provenance node", material)
        return None

    try:
        prov_node = FrictionProvenanceData.from_provenance_folder(
            prov_dir, simulation_type=sim_type
        )
        prov_node.material = material
        prov_node.set_uuid = set_uuid
        prov_node.label = f"provenance:{material}"
        prov_node.description = f"{sim_type} provenance for {material}"
        prov_node.store()
        return prov_node
    except _REGISTRATION_EXCEPTIONS:
        logger.error(
            "Failed to create provenance node for %s from %s",
            material, prov_dir, exc_info=True,
        )
        return None


def _create_set_simulation_node(
    material: str,
    config_data: Dict[str, Any],
    sim_type: str,
    set_uuid: str,
    prov_uuid: Optional[str],
    status: str = 'imported',
) -> 'FrictionSimulationData':
    """Create, store and return a ``FrictionSimulationData`` for one material."""
    from .data import FrictionSimulationData  # pylint: disable=import-outside-toplevel

    sim_node = FrictionSimulationData()
    sim_node.simulation_type = sim_type
    sim_node.set_uuid = set_uuid
    if prov_uuid:
        sim_node.provenance_uuid = prov_uuid

    if config_data:
        sim_node.set_from_config(config_data, simulation_type=sim_type)

    # Ensure the material name from the directory overrides config (config may
    # contain a different material name if the same config was reused)
    sim_node.material = material
    sim_node.status = status
    sim_node.label = f"{material} [{sim_type}]"
    sim_node.description = f"{sim_type} simulation for {material}, status={status}"
    sim_node.store()
    return sim_node


def _import_set_results_tree(
    reader,
    safe_material: str,
    original_material: str,
    size_data: Dict,
    sim_type: str,
    sim_uuid: str,
    set_uuid: str,
) -> int:
    """Create ``FrictionResultsData`` nodes for one material's result tree.

    Returns the number of result nodes created.
    """
    count = 0
    for size_key, substrate_data in size_data.items():
        for _substrate, tip_data in substrate_data.items():
            for _tip_mat, radius_data in tip_data.items():
                for _radius, layer_data in radius_data.items():
                    for layer_key, speed_data in layer_data.items():
                        layers = int(layer_key.replace('l', ''))
                        for speed_key, force_data in speed_data.items():
                            speed = int(speed_key.replace('s', ''))
                            for load_key, angle_data in force_data.items():
                                is_pressure = load_key.startswith('p')
                                load_val = float(load_key[1:])
                                for angle_key, df in angle_data.items():
                                    angle = int(angle_key.replace('a', ''))
                                    try:
                                        node = _create_result_node(
                                            reader,
                                            safe_material,
                                            size_key,
                                            layers,
                                            speed,
                                            load_val,
                                            is_pressure,
                                            angle,
                                            df,
                                        )
                                        node.simulation_type = sim_type
                                        node.simulation_uuid = sim_uuid
                                        node.set_uuid = set_uuid
                                        # Restore original material name with dashes
                                        node.material = original_material
                                        load_str = (f"{load_val:g}GPa"
                                                    if is_pressure
                                                    else f"{load_val:g}nN")
                                        node.label = (f"{original_material} "
                                                      f"{load_str} a{angle}")
                                        node.store()
                                        count += 1
                                    except Exception:  # pylint: disable=broad-except
                                        logger.warning(
                                            "Failed to import result for %s/l%d/%s/a%d",
                                            original_material, layers, load_key, angle,
                                            exc_info=True,
                                        )
    return count


# =============================================================================
# Database maintenance
# =============================================================================

def clear_all_nodes(aiida_profile: Optional[str] = None) -> int:
    """Delete all FrictionSim2D nodes from the active AiiDA profile.

    This removes every node whose type starts with
    ``data.friction2d.`` from the database.  Use with caution — this
    action is irreversible.

    Args:
        aiida_profile: AiiDA profile to load.

    Returns:
        Number of nodes deleted.
    """
    _require_aiida()
    _ensure_aiida_profile(aiida_profile)

    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel

    # Collect PKs of all FrictionSim2D nodes
    qb = QueryBuilder()
    from aiida.orm import Data  # pylint: disable=import-outside-toplevel
    qb.append(Data, project=['id', 'node_type'])

    pks_to_delete: List[int] = []
    for (pk, node_type) in qb.iterall():
        if node_type.startswith('data.friction2d.'):
            pks_to_delete.append(pk)

    if not pks_to_delete:
        logger.info("No FrictionSim2D nodes found.")
    else:
        from aiida.tools import delete_nodes  # pylint: disable=import-outside-toplevel
        _, deletion_ok = delete_nodes(pks_to_delete, dry_run=False)
        deleted_count = len(pks_to_delete) if deletion_ok else 0
        logger.info("Deleted %d FrictionSim2D nodes.", deleted_count)

    # Also delete friction2d/* groups
    from aiida.orm import Group, QueryBuilder as _QB  # pylint: disable=import-outside-toplevel
    gqb = _QB()
    gqb.append(Group, filters={'label': {'like': 'friction2d/%'}}, project=['id', 'uuid'])
    group_pks = [r[0] for r in gqb.all()]
    for pk in group_pks:
        Group.collection.delete(pk=pk)
    if group_pks:
        logger.info('Deleted %d friction2d/* groups.', len(group_pks))
    # Expunge all SQLAlchemy session state so subsequent operations in the
    # same process do not hit stale references to the deleted objects.
    # expire_all() alone is insufficient — expunge_all() removes objects from
    # the identity map so SQLAlchemy doesn't try to reload deleted rows.
    try:
        from aiida.manage import get_manager  # pylint: disable=import-outside-toplevel
        get_manager().get_profile_storage().get_session().expunge_all()
    except Exception:  # pylint: disable=broad-except
        pass
    return len(pks_to_delete)


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
# Simulation rebuild from provenance
# =============================================================================

def rebuild_simulation_set(
    label_or_uuid: str,
    output_root: Optional[Union[str, Path]] = None,
    simulation_root_name: Optional[str] = None,
    aiida_profile: Optional[str] = None,
    generate_hpc: bool = False,
) -> 'tuple[List[Path], Path]':
    """Rebuild all simulation input files for a set from its AiiDA provenance.

    Queries the database for every ``FrictionProvenanceData`` node that belongs
    to the requested simulation set, exports the stored CIF and potential files
    to a temporary location, patches the per-material ``config.json`` so that
    all file paths resolve correctly, and then runs the appropriate builder
    (:class:`~src.builders.afm.AFMSimulation` or
    :class:`~src.builders.sheetonsheet.SheetOnSheetSimulation`) to regenerate
    the full directory tree of LAMMPS input files.

    The reconstructed tree mirrors the layout produced by
    :func:`src.core.run.run_simulations` so that the result can be used directly
    with the HPC submission workflow.

    Args:
        label_or_uuid: Human-readable set label (e.g. ``'251125-sheetonsheet'``)
            or full UUID of a :class:`~src.aiida.data.FrictionSimulationSetData`
            node.
        output_root: Root directory under which the simulation tree is written.
            Defaults to the current working directory.
        simulation_root_name: Name of the ``simulation_XXXXXXXX`` folder that
            will be created inside *output_root*.  Defaults to
            ``simulation_YYYYMMDD_HHMMSS`` (current timestamp).
        aiida_profile: AiiDA profile to load.  ``None`` reuses the currently
            active profile.
        generate_hpc: If ``True``, generate HPC job-submission scripts after
            all builders have run (equivalent to passing ``--hpc`` to the CLI).

    Returns:
        ``(created_dirs, simulation_root)`` — list of individual simulation
        directories that were successfully built, and the common root directory.

    Raises:
        RuntimeError: If AiiDA is not available or the profile cannot be loaded.
        LookupError: If no simulation set matching *label_or_uuid* is found.
    """
    _require_aiida()
    _ensure_aiida_profile(aiida_profile)

    import tempfile  # pylint: disable=import-outside-toplevel
    from datetime import datetime  # pylint: disable=import-outside-toplevel

    from aiida.orm import QueryBuilder  # pylint: disable=import-outside-toplevel  # pyright: ignore[reportMissingImports]

    from .data import (  # pylint: disable=import-outside-toplevel
        FrictionSimulationSetData,
        FrictionProvenanceData,
    )

    # ------------------------------------------------------------------
    # 1. Resolve the simulation set node
    # ------------------------------------------------------------------
    set_node = _resolve_set_node(label_or_uuid, QueryBuilder, FrictionSimulationSetData)
    set_uuid = str(set_node.uuid)
    sim_type = set_node.simulation_type  # 'afm' or 'sheetonsheet'

    logger.info("Rebuilding set %r (type=%s, uuid=%s)", set_node.label, sim_type, set_uuid)

    # ------------------------------------------------------------------
    # 2. Set up output directories
    # ------------------------------------------------------------------
    output_root = Path(output_root) if output_root is not None else Path.cwd()
    root_name = simulation_root_name or datetime.now().strftime("simulation_%Y%m%d_%H%M%S")
    simulation_root = output_root / root_name
    simulation_root.mkdir(parents=True, exist_ok=True)
    (simulation_root / 'logs').mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Load shared settings (HPC defaults, thermostat, etc.)
    # ------------------------------------------------------------------
    from ..core.config import (  # pylint: disable=import-outside-toplevel
        AFMSimulationConfig,
        SheetOnSheetSimulationConfig,
        load_settings,
    )
    from ..builders.afm import AFMSimulation  # pylint: disable=import-outside-toplevel
    from ..builders.sheetonsheet import SheetOnSheetSimulation  # pylint: disable=import-outside-toplevel

    defaults = load_settings()
    if sim_type == 'sheetonsheet':
        if defaults.hpc.lammps_scripts == ['system.in', 'slide.in']:
            defaults.hpc.lammps_scripts = ['slide.in']

    # ------------------------------------------------------------------
    # 4. Query all provenance nodes for this set
    # ------------------------------------------------------------------
    qb = QueryBuilder()
    qb.append(FrictionProvenanceData,
              filters={'attributes.set_uuid': set_uuid},
              project=['*'])
    prov_nodes = qb.all(flat=True)

    if not prov_nodes:
        raise LookupError(
            f"No FrictionProvenanceData nodes found for set {label_or_uuid!r}. "
            "Has the set been imported with import_simulation_set()?"
        )

    logger.info("Found %d provenance nodes for set %r", len(prov_nodes), set_node.label)

    # ------------------------------------------------------------------
    # 5. Rebuild each material
    # ------------------------------------------------------------------
    created_dirs: List[Path] = []

    with tempfile.TemporaryDirectory(prefix='friction2d_rebuild_') as tmp_root:
        tmp_root_path = Path(tmp_root)

        for prov_node in prov_nodes:
            material = prov_node.material
            try:
                output_dir, success = _rebuild_single_material(
                    prov_node=prov_node,
                    material=material,
                    sim_type=sim_type,
                    simulation_root=simulation_root,
                    tmp_root=tmp_root_path,
                    defaults=defaults,
                    AFMSimulationConfig=AFMSimulationConfig,
                    SheetOnSheetSimulationConfig=SheetOnSheetSimulationConfig,
                    AFMSimulation=AFMSimulation,
                    SheetOnSheetSimulation=SheetOnSheetSimulation,
                )
                if success:
                    created_dirs.append(output_dir)
            except Exception:  # pylint: disable=broad-except
                logger.error(
                    "Failed to rebuild material %r in set %r",
                    material, label_or_uuid, exc_info=True,
                )

    logger.info(
        "Rebuilt %d/%d simulations in %s",
        len(created_dirs), len(prov_nodes), simulation_root,
    )

    if generate_hpc and created_dirs:
        try:
            from ..hpc.scripts import generate_hpc_scripts_for_root  # pylint: disable=import-outside-toplevel
            generate_hpc_scripts_for_root(simulation_root, defaults)
        except ImportError:
            logger.warning(
                "generate_hpc_scripts_for_root not available in hpc.scripts — skipping HPC script generation"
            )

    return created_dirs, simulation_root


def _resolve_set_node(label_or_uuid: str, QueryBuilder, FrictionSimulationSetData):
    """Return the ``FrictionSimulationSetData`` matching *label_or_uuid*.

    Tries an exact label match first, then falls back to a UUID prefix match.
    """
    # Try exact label
    qb = QueryBuilder()
    qb.append(FrictionSimulationSetData,
              filters={'attributes.label': label_or_uuid},
              project=['*'])
    results = qb.all(flat=True)
    if results:
        return results[0]

    # Try UUID (or UUID prefix)
    qb2 = QueryBuilder()
    qb2.append(FrictionSimulationSetData, project=['*'])
    for node in qb2.all(flat=True):
        if str(node.uuid).startswith(label_or_uuid):
            return node

    raise LookupError(
        f"No FrictionSimulationSetData found for label or UUID {label_or_uuid!r}. "
        "Use verdi node list or Friction2DDB().list_sets() to see available sets."
    )


def _rebuild_single_material(
    prov_node,
    material: str,
    sim_type: str,
    simulation_root: Path,
    tmp_root: Path,
    defaults,
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    AFMSimulation,
    SheetOnSheetSimulation,
) -> 'tuple[Path, bool]':
    """Rebuild simulation files for one material from its provenance node.

    Returns ``(output_dir, success)``.
    """
    # ---- Extract config.json from the provenance node --------------------
    try:
        config_bytes = prov_node.get_file_content('config.json', 'config')
    except (KeyError, FileNotFoundError):
        logger.warning("No config.json in provenance node for %r — skipping", material)
        return Path(), False

    config_dict = json.loads(config_bytes)

    # ---- Export CIF and potential files to a per-material temp dir -------
    mat_tmp = tmp_root / material.replace('/', '_').replace(' ', '_')
    exported = prov_node.export_to_directory(mat_tmp)

    # Patch file paths in the config so the builder can find the exported files.
    # Both cif_path and pot_path may be absolute paths from the original machine.
    _patch_config_paths(config_dict, exported, sim_type)

    # ---- Inject shared GlobalSettings ------------------------------------
    config_dict['settings'] = defaults.model_dump()

    # ---- Determine output directory from config --------------------------
    sheet_cfg = config_dict.get('2D') or config_dict.get('sheet', {})
    mat = sheet_cfg.get('mat', material)
    x = sheet_cfg.get('x', 100)
    y = sheet_cfg.get('y', 100)
    temp = config_dict.get('general', {}).get('temp', 300)
    size_token = f"{format_dimension_token(x)}x_{format_dimension_token(y)}y"

    if sim_type == 'afm':
        tip_cfg = config_dict.get('tip', {})
        sub_cfg = config_dict.get('sub', {})
        tip_mat = tip_cfg.get('mat', 'Si')
        tip_amorph = tip_cfg.get('amorph', 'c')
        tip_r = tip_cfg.get('r', 25)
        sub_mat = sub_cfg.get('mat', 'Si')
        sub_amorph = sub_cfg.get('amorph', 'a')
        sub_str = f"{sub_amorph}{sub_mat}" if sub_amorph == 'a' else sub_mat
        tip_str = f"{tip_amorph}{tip_mat}" if tip_amorph == 'a' else tip_mat
        output_dir = (
            simulation_root / 'afm' / mat / size_token /
            f"sub_{sub_str}_tip_{tip_str}_r{int(tip_r)}" / f"K{int(temp)}"
        )
    else:
        output_dir = (
            simulation_root / 'sheetonsheet' / mat / size_token / f"K{int(temp)}"
        )

    # ---- Write patched config.json to provenance sub-directory -----------
    prov_out = output_dir / 'provenance'
    prov_out.mkdir(parents=True, exist_ok=True)
    config_json_path = prov_out / 'config.json'
    config_json_path.write_text(json.dumps(config_dict, indent=2), encoding='utf-8')

    # ---- Instantiate config and builder, then build ----------------------
    # config.json is serialised with field names (model_dump_json default), but
    # the Pydantic models declare the sheet section with alias '2D'.  Rename the
    # key so the constructor receives the alias it expects.
    if 'sheet' in config_dict and '2D' not in config_dict:
        config_dict['2D'] = config_dict.pop('sheet')

    try:
        if sim_type == 'afm':
            config_obj = AFMSimulationConfig(**config_dict)
            builder = AFMSimulation(config_obj, output_dir,
                                    config_path=str(config_json_path))
        else:
            config_obj = SheetOnSheetSimulationConfig(**config_dict)
            builder = SheetOnSheetSimulation(config_obj, output_dir,
                                             config_path=str(config_json_path))

        builder.set_base_output_dir(simulation_root)
        builder.build()
    except Exception:  # pylint: disable=broad-except
        logger.error("Builder failed for material %r at %s", material, output_dir,
                     exc_info=True)
        return output_dir, False

    logger.info("Rebuilt %s → %s", material, output_dir.relative_to(simulation_root))
    return output_dir, True


def _patch_config_paths(
    config_dict: Dict[str, Any],
    exported: Dict[str, Path],
    sim_type: str,
) -> None:
    """Update ``cif_path`` and ``pot_path`` entries in *config_dict* in-place.

    Replaces the absolute paths recorded at build time (which may no longer
    exist) with the paths of the files exported from the AiiDA repository.

    Only fields that are already present in the config are updated; no new
    keys are inserted.
    """
    # Separate CIF files (exported into cif/) from potential files (potentials/).
    # This prevents stem-based matching from confusing e.g. h-MoS2.cif and
    # h-MoS2.sw which share the same stem.
    cif_by_stem: Dict[str, Path] = {}
    pot_by_stem: Dict[str, Path] = {}
    pot_by_ext: Dict[str, Path] = {}

    for path in exported.values():
        parts = path.parts
        if 'cif' in parts:
            cif_by_stem[path.stem] = path
        else:
            pot_by_stem[path.stem] = path
            pot_by_ext[path.suffix.lstrip('.')] = path

    def _patch_component(component: Dict[str, Any]) -> None:
        """Patch a single component sub-dict (sheet, tip, sub, …)."""
        if 'cif_path' in component:
            orig = Path(component['cif_path'])
            candidate = cif_by_stem.get(orig.stem)
            if candidate and candidate.exists():
                component['cif_path'] = str(candidate)
        if 'pot_path' in component:
            orig = Path(component['pot_path'])
            candidate = pot_by_stem.get(orig.stem)
            if candidate is None:
                candidate = pot_by_ext.get(orig.suffix.lstrip('.'))
            if candidate and candidate.exists():
                component['pot_path'] = str(candidate)

    _patch_component(config_dict.get('2D') or config_dict.get('sheet', {}))
    if sim_type == 'afm':
        _patch_component(config_dict.get('tip', {}))
        _patch_component(config_dict.get('sub', {}))


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
