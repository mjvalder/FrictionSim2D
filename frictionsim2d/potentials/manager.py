"""Potential management for FrictionSim2D: atom-type registry and LAMMPS pair settings.

Builds the ``pair_style``/``pair_coeff``/``mass`` file for single- and multi-component
cells (hybrid indexing, LJ mixing, interlayer styles). Which potentials carry an internal LJ
is listed in ``utils.POTENTIALS_WITH_INTERNAL_LJ`` / ``POTENTIALS_REQUIRING_LJ``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Iterator, Sequence
from collections import defaultdict
import hashlib
import logging
import re
import os
import shutil

from ..core.config import ComponentConfig, GlobalSettings
from ..core.mlip import (MLIP_MODEL_EXTENSIONS, d3_coeff_line, d3_style_token,
                         get_mlip, is_mlip, model_elements)
from ..core.utils.structures import cifread, count_atomtypes
from ..core.utils.tokens import normalize_element_symbol
from .utils import (
    POTENTIALS_REQUIRING_LJ,
    POTENTIALS_WITH_INTERNAL_LJ,
    REAXFF_TYPES,
    internal_pair_cutoff,
    lj_params,
    normalize_potential_type,
)
from .interlayer import (
    ILP_CUTOFF,
    INTERLAYER_STYLES,
    chalcogen_split_elements,
    KC_FULL_CUTOFF,
    TAPER_FLAG,
    default_interlayer_file,
    gaussian_g,
    ilp_substyle,
    mixed_g,
    resolve_interlayer_file,
    splits_chalcogens,
    write_gaussian_kc_file,
)

logger = logging.getLogger(__name__)


#: Cutoff for the build-time ``pair_style zero`` surrogate (see
#: :meth:`PotentialManager.write_surrogate_file`). It only has to be a legal
#: cutoff — the style computes nothing, so the value cannot affect any geometry.
ZERO_SURROGATE_CUTOFF = 10.0

def is_mlip_model_file(path: Union[str, Path]) -> bool:
    """True if ``path`` names a serialized MLIP model file (by extension)."""
    return str(path).lower().endswith(MLIP_MODEL_EXTENSIONS)


def stage_model_file(src: Path, dest: Path,
                     link_from: Optional[Path] = None) -> None:
    """Place ``src`` at ``dest``, hard-linking MLIP model files.

    Used for both destinations a model reaches: the campaign-root
    ``potentials/`` store the deck references, and each run's
    ``provenance/potentials/`` record. Both need a real, openable file -- LAMMPS
    reads the first, provenance must carry the second -- but neither needs its
    own *bytes*. A hard link is a real file; identity is pinned by the sha256 the
    provenance manifest already records. So one campaign holds ONE inode of a
    model instead of one per run plus one for the store. For an 18 MB-multi-GB
    foundation model over a sweep that is the dominant cost on disk.

    Classical potentials (KB-scale text) keep the plain copy: linking them would
    change long-standing behaviour for no benefit, and a small editable text file
    is one a user may reasonably expect to be independent per run.

    Args:
        src: the file to place. Its bytes are the identity of the entry
            whatever we end up linking from.
        dest: where it goes.
        link_from: preferred link source when it already exists -- normally
            another copy already inside this campaign, which keeps the tree's
            aliasing internal. Falls back to ``src``.

    Falls back to a plain copy whenever linking cannot work: a different
    filesystem (``EXDEV``), a filesystem without hard links, or a permission
    refusal. The result is then byte-identical to the pre-existing behaviour, so
    this can only save space, never fail a build.

    .. warning::

       Hard links survive a move but **not every transfer**. ``rsync`` collapses
       them only with ``-H``, and ``tar`` must be told as well; without those
       flags the links re-expand into independent copies at the far end. The
       bytes still arrive -- this is a transfer-size question, not a provenance
       one.

       Linked entries also alias their source for *in-place* writes. Replacing a
       file makes a new inode and breaks the link, which is the normal case; an
       open-and-overwrite would propagate. Nothing edits a serialized model in
       place, but do not treat these as write-protected.
    """
    if is_mlip_model_file(src):
        for candidate in (link_from, src):
            if candidate is None or not Path(candidate).is_file():
                continue
            try:
                os.link(candidate, dest)
                return
            except OSError as exc:  # cross-device, no-hardlink FS, permissions
                logger.debug("Hard link from %s failed (%s)", candidate, exc)
    shutil.copy2(src, dest)


def _element_mass(element: str) -> Optional[float]:
    """Atomic mass of ``element`` in amu, or ``None`` (with a warning) if ASE has none."""
    from ase.data import atomic_masses, atomic_numbers  # pylint: disable=import-outside-toplevel
    # (deferred: ase)
    try:
        return float(atomic_masses[atomic_numbers[element]])
    except (KeyError, IndexError):
        logger.warning("Could not find mass for element '%s', using 1.0", element)
        return None


@dataclass
class AtomType:
    """Metadata for a single LAMMPS atom type.
    
    Attributes:
        type_id: Global LAMMPS type ID (1-indexed).
        component: Component name (e.g., 'tip', 'sheet', 'sub').
        element: Chemical element symbol (e.g., 'Mo', 'S').
        pot_label: Label for pair_coeff (e.g., 'Mo1', 'S', 'Si').
        layer: Layer index (0-indexed) for multi-layer materials, None otherwise.
        region: Langevin region suffix ('fix', 'thermo'), None for mobile/standard.
    """
    type_id: int
    component: str
    element: str
    pot_label: str
    layer: Optional[int] = None
    region: Optional[str] = None


class TypeRegistry:
    """Centralized registry for atom types with query methods for LAMMPS pair_coeff generation."""

    def __init__(self):
        self._types: Dict[int, AtomType] = {}
        self._next_id: int = 1

    def register(self, component: str, element: str, pot_label: str,
                    layer: Optional[int] = None, region: Optional[str] = None) -> int:
        """Register a new atom type and return its ID."""
        type_id = self._next_id
        self._next_id += 1
        self._types[type_id] = AtomType(
            type_id=type_id,
            component=component,
            element=element,
            pot_label=pot_label,
            layer=layer,
            region=region
        )
        return type_id

    def __len__(self) -> int:
        return len(self._types)

    def __iter__(self) -> Iterator[AtomType]:
        """Iterate over all types in type_id order."""
        for tid in sorted(self._types.keys()):
            yield self._types[tid]

    # === Query methods ===

    def ids_by_component(self, component: str) -> List[int]:
        """All type IDs for a component."""
        return sorted([t.type_id for t in self._types.values() if t.component == component])

    def ids_by_component_element(self, component: str, element: str) -> List[int]:
        """Type IDs for a specific element in a component."""
        return sorted([t.type_id for t in self._types.values()
                if t.component == component and t.element == element])

    def ids_by_component_layer(self, component: str, layer: int) -> List[int]:
        """Type IDs for a specific layer of a component."""
        return sorted([t.type_id for t in self._types.values()
                if t.component == component and t.layer == layer])

    def ids_by_component_layer_element(self, component: str, layer: int, element: str) -> List[int]:
        """Type IDs for element in specific layer."""
        return sorted([t.type_id for t in self._types.values()
                if t.component == component and t.layer == layer and t.element == element])

    def ids_by_component_region(self, component: str, region: Optional[str]) -> List[int]:
        """Type IDs for a specific region of a component."""
        return sorted([t.type_id for t in self._types.values()
                if t.component == component and t.region == region])

    def elements_in_component(self, component: str) -> List[str]:
        """Unique elements in a component (preserving order of first occurrence)."""
        seen = set()
        result = []
        for t in self:
            if t.component == component and t.element not in seen:
                seen.add(t.element)
                result.append(t.element)
        return result

    def _padded_map(self, components: Sequence[str], attr: str,
                    layer: Optional[int]) -> List[str]:
        """One ``attr`` per type id, ``NULL`` where the type is not in ``components``/``layer``."""
        wanted = set(components)
        result = []
        for tid in range(1, self._next_id):
            atype = self._types.get(tid)
            if atype and atype.component in wanted:
                if layer is None or atype.layer == layer:
                    result.append(getattr(atype, attr))
                    continue
            result.append("NULL")
        return result

    def build_null_map(self, component: str, layer: Optional[int] = None) -> List[str]:
        """Build NULL-padded pot_label list for pair_coeff.
        
        Returns a list where each position corresponds to a type ID.
        Types matching the component (and optionally layer) get their pot_label,
        others get 'NULL'.
        """
        return self._padded_map([component], 'pot_label', layer)

    def build_element_map(self, components: Sequence[str],
                          layer: Optional[int] = None) -> List[str]:
        """NULL-padded *element* list for pair_coeff over several components.

        Like :meth:`build_null_map`, but each matching type contributes its
        chemical element (``Mo``), never its potential label (``Mo1``):
        interlayer parameter files are keyed by element. Every type of the
        listed components matches -- all layers and regions -- unless
        ``layer`` narrows it to one layer.
        """
        return self._padded_map(components, 'element', layer)

    def build_full_map(self) -> List[str]:
        """Build a pot_label list covering EVERY atom type in the cell.

        This is the map a full-cell MLIP requires. Unlike
        :meth:`build_null_map`, which partitions the cell by component and pads
        the rest with ``NULL``, this names a real element for every type --
        because both MLIP frameworks reject NULL entries
        and DeePMD additionally refuses a type-range restriction
        (``pair_deepmd.cpp:816``), so an MLIP's ``pair_coeff`` must be ``* *``
        with one entry per type.

        Returns:
            One pot_label per type, in type-id order. Never contains ``NULL``.
        """
        return [atype.pot_label for atype in self]

    def get_element_map(self, component: str) -> Dict[str, List[int]]:
        """Get element -> [type_ids] mapping for a component."""
        result: Dict[str, List[int]] = defaultdict(list)
        for t in self._types.values():
            if t.component == component:
                result[t.element].append(t.type_id)
        return {k: sorted(v) for k, v in result.items()}

    def format_type_range(self, types: List[int]) -> str:
        """Format a list of type IDs into LAMMPS range notation.

        Args:
            types: List of type IDs.

        Returns:
            LAMMPS range notation string (e.g., '1*3' or '5').
        """
        if not types:
            return ""
        types = sorted(types)
        if len(types) == 1:
            return str(types[0])
        return f"{types[0]}*{types[-1]}"

    @staticmethod
    def contiguous_runs(types: List[int]) -> List[List[int]]:
        """Split a list of type IDs into maximal consecutive runs.

        ``[1, 3]`` -> ``[[1], [3]]``; ``[1, 2, 3, 5]`` -> ``[[1, 2, 3], [5]]``.
        Used to emit LAMMPS ``mass`` lines whose ``lo*hi`` spans never engulf a
        type absent from the list (which the plain :meth:`format_type_range` span
        would), so one element's mass cannot overwrite another's when the same
        element occupies non-contiguous types across hetero components.
        """
        if not types:
            return []
        ordered = sorted(types)
        runs: List[List[int]] = [[ordered[0]]]
        for t in ordered[1:]:
            if t == runs[-1][-1] + 1:
                runs[-1].append(t)
            else:
                runs.append([t])
        return runs

    def get_element_type_range(self, component: str, element: str) -> str:
        """Get the LAMMPS type range string for an element in a component.

        Args:
            component: Component name.
            element: Element symbol.

        Returns:
            Format like '1*3' for types 1,2,3 or just '5' for single type.
        """
        all_types = self.ids_by_component_element(component, element)
        return self.format_type_range(all_types)

    def get_group_string(self, component: str) -> str:
        """Return space-separated type IDs for a component (for LAMMPS grouping).

        Args:
            component: Name of the component.

        Returns:
            Space-separated string of type IDs (e.g., "1 2 3").
        """
        all_ids = self.ids_by_component(component)
        return " ".join(map(str, all_ids))

    def get_layer_group_string(self, component: str, layer: int) -> str:
        """Return type IDs for a specific layer of a component.

        Args:
            component: Name of the component.
            layer: Layer index (0-based).

        Returns:
            Space-separated string of type IDs for that layer.
        """
        all_ids = self.ids_by_component_layer(component, layer)
        return " ".join(map(str, all_ids))

class PotentialManager:
    """Manages potential assignments, type mapping, and pair_coeff generation.

    This class handles:
        - Sequential atom type ID assignment across all components
        - Hybrid pair_style construction with proper indexing
        - Self-interactions (many-body potentials like Tersoff, SW)
        - Cross-interactions (LJ between different components)
        - Interlayer interactions for multi-layer 2D materials
        - Langevin thermostat region type expansion (3x types)

    Attributes:
        components: List of registered component data dictionaries.
        types: TypeRegistry containing all atom type metadata.
        potential_usage: Tracks how many times each potential type is used.
    """

    def __init__(self, settings: GlobalSettings,
                use_langevin: Optional[bool] = None,
                potentials_dir: Optional[Union[str, Path]] = None,
                potentials_prefix: Optional[str] = None,
                shared_potentials_dir: Optional[Union[str, Path]] = None,
                shared_potentials_prefix: Optional[str] = None):
        """Initialize the PotentialManager.

        Args:
            settings: Global simulation settings containing potential and
                thermostat config.
            use_langevin: Override for Langevin type expansion. If None, uses
                settings. Set to False for temporary calculations like lat_c
                finding.
            potentials_dir: Absolute path to where potential files should be copied.
            potentials_prefix: String prefix for pair_coeff paths (e.g.,
                'afm/.../potentials'). If None, falls back to the copied file path.
            shared_potentials_dir: Absolute path to a simulation-root potentials
                directory used for MLIP **model files only**. A foundation model
                is hundreds of MB, so copying it into every run directory of a
                sweep costs gigabytes over rsync; it is staged once here instead.
                Classical potentials keep their per-run staging either way.
            shared_potentials_prefix: pair_style/pair_coeff prefix for the shared
                directory, relative to a run's lammps dir (e.g. '../potentials').
        """
        self.settings = settings

        self.potentials_dir: Optional[Path] = (
            Path(potentials_dir).resolve() if potentials_dir else None
        )
        self.potentials_prefix = potentials_prefix
        self.shared_potentials_dir: Optional[Path] = (
            Path(shared_potentials_dir).resolve() if shared_potentials_dir else None
        )
        self.shared_potentials_prefix = shared_potentials_prefix

        if use_langevin is not None:
            self.use_langevin = use_langevin
        else:
            self.use_langevin = settings.thermostat.type == 'langevin'

        self.components: Dict[str, Dict[str, Any]] = {}
        self.types = TypeRegistry()

        self.potential_usage: Dict[str, int] = defaultdict(int)
        self.potential_indices: Dict[str, int] = defaultdict(int)

        self.self_interaction_commands: List[str] = []
        self.cross_interaction_commands: List[str] = []
        self.lj_overrides: Dict[Tuple[str, str], Tuple[float, float]] = {}

        self.virtual_atom_type: Optional[int] = None

        # Interlayer style. 'lj' keeps every output unchanged.
        self.interlayer_style: str = 'lj'
        self._interlayer_file_name: Optional[str] = None
        self._interlayer_base_dir: Optional[Path] = None
        self._g_by_component: Dict[str, float] = {}
        self._interlayer_components: List[str] = []
        self._interlayer_source: Optional[Path] = None
        self._interlayer_substyle: Optional[str] = None
        # Many-body (ilp/kc) pair_coeff lines, ready to write.
        self._interlayer_commands: List[str] = []
        # LJ-G: pair table and (sign, element map, comment) per KC/z instance;
        # the .KC files and their pair_coeff lines are written by write_file.
        self._gaussian_pairs: Dict[Tuple[str, str], float] = {}
        self._gaussian_instances: List[Tuple[int, List[str], str]] = []
        self._generated_files: Dict[str, str] = {}
        # MLIP state. `_mlip_model_paths` maps pot_type -> staged model path so
        # write_file can build a style token from a usage table keyed only by
        # pot_type.
        self._mlip_model_paths: Dict[str, str] = {}
        # pot_type -> component names that asked for a self-interaction, in call
        # order. An MLIP's pair_coeff is materialised ONCE at write time from
        # this (see _mlip_self_interaction_lines), not per component.
        self._mlip_components: Dict[str, List[str]] = {}
        # pot_type -> the model's real filesystem path, as configured. The
        # deck path in _mlip_model_paths may be a relative staging prefix,
        # which is not openable from here; the coverage check needs this one.
        self._mlip_source_paths: Dict[str, str] = {}
        self.d3_enabled: bool = False

    @property
    def lj_cutoff(self) -> float:
        """Expose the configured LJ cutoff."""
        return self.settings.potential.lj_cutoff

    @staticmethod
    def _pair_key(el1: str, el2: str) -> Tuple[str, str]:
        """Build an order-independent key for an element pair."""
        a = normalize_element_symbol(el1)
        b = normalize_element_symbol(el2)
        sorted_pair = sorted((a, b))
        return (sorted_pair[0], sorted_pair[1])

    @staticmethod
    def _parse_override_pair(pair_key: str) -> Tuple[str, str]:
        """Parse a pair key like 'Mo-S', 'Mo_S', 'Mo S', or 'Mo,S'."""
        tokens = [tok for tok in re.split(r'[^A-Za-z]+', pair_key) if tok]
        if len(tokens) != 2:
            raise ValueError(
                f"Invalid LJ override pair '{pair_key}'. "
                "Expected two element symbols (e.g., 'Mo-S')."
            )
        return normalize_element_symbol(tokens[0]), normalize_element_symbol(tokens[1])

    @staticmethod
    def _parse_override_values(pair_key: str, values: Any) -> Tuple[float, float]:
        """Parse epsilon/sigma override values from list/tuple or dict."""
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            epsilon = float(values[0])
            sigma = float(values[1])
            return epsilon, sigma

        if isinstance(values, dict):
            if 'epsilon' not in values or 'sigma' not in values:
                raise ValueError(
                    f"Invalid LJ override values for '{pair_key}'. "
                    "Dict entries must include 'epsilon' and 'sigma'."
                )
            return float(values['epsilon']), float(values['sigma'])

        raise ValueError(
            f"Invalid LJ override values for '{pair_key}'. "
            "Use [epsilon, sigma] or {epsilon: ..., sigma: ...}."
        )

    def set_lj_overrides(self, overrides: Optional[Dict[str, Any]]) -> None:
        """Set user-provided LJ overrides keyed by element pair.

        Example:
            {
            "Mo-Mo": [1.0624, 3.878597],
            "Mo-S": [0.4124, 3.75114],
            "S-S": [0.198443, 3.62368]
            }
        """
        self.lj_overrides = {}
        if not overrides:
            return

        for pair_key, values in overrides.items():
            if pair_key == 'file':
                # A config built without going through expand_config_sweeps
                # (or handed a raw [lj_override] section directly) still has
                # the shared-file reference in it; expansion is what resolves
                # and drops it for the normal run path, so a literal 'file'
                # key reaching here is skipped rather than crashing on
                # _parse_override_pair.
                continue
            el1, el2 = self._parse_override_pair(str(pair_key))
            epsilon, sigma = self._parse_override_values(str(pair_key), values)
            self.lj_overrides[self._pair_key(el1, el2)] = (epsilon, sigma)

    def _get_lj_params(self, el1: str, el2: str) -> Tuple[float, float]:
        """Return LJ parameters, preferring user overrides when present."""
        override = self.lj_overrides.get(self._pair_key(el1, el2))
        if override is not None:
            return override
        return lj_params(normalize_element_symbol(el1), normalize_element_symbol(el2))

    @staticmethod
    def _format_lj_value(value: float) -> str:
        """Format LJ numeric values with high precision and no forced rounding."""
        return f"{float(value):.15g}"

    def get_single_component_commands(
        self,
        config: ComponentConfig,
        elements: List[str]
    ) -> List[str]:
        """Generate LAMMPS interatomic potential settings for single-component setup.

        This is useful for standalone simulations like amorphisation where
        only one component is used and no cross-interactions are needed.

        Args:
            config: Component configuration with pot_type and pot_path.
            elements: List of element symbols in the material.

        Returns:
            List of LAMMPS commands (pair_style, pair_coeff, mass).
        """
        commands = []
        pot_type = config.pot_type.lower()
        if is_mlip(pot_type):
            # The line below is wrong for BOTH frameworks -- deepmd carries the
            # model in the pair_style line, mace in the pair_coeff line -- and
            # the only callers are the generation-time melt-quench builds, which
            # cannot evaluate an MLIP anyway. Refuse, never emit.
            raise ValueError(
                f"'{pot_type}' cannot be used for a single-component "
                f"generation-time build (this is the amorph = 'a' melt-quench "
                f"path). That run needs a potential the GENERATING machine can "
                f"evaluate, and its result depends on forces, so the "
                f"`pair_style zero` surrogate is not an option either. Use "
                f"amorph = 'c', or prepare the amorphous structure separately "
                f"with a classical potential. See documentation/mlip.md §1."
            )
        pot_path = self._get_potential_path(config.pot_path)

        commands.append(f"pair_style {pot_type}")

        element_str = ' '.join(elements)
        commands.append(f"pair_coeff * * {pot_path} {element_str}")

        for i, elem in enumerate(elements, 1):
            mass = _element_mass(elem)
            commands.append(f"mass {i} {mass:.6f}" if mass is not None else f"mass {i} 1.0")

        return commands

    def register_component(
        self,
        name: str,
        config: ComponentConfig,
        n_layers: int = 1,
        regions: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, List[int]]:
        """Register a material component and assign unique global atom type IDs.

        For multi-layer 2D materials, each layer gets its own set of atom types.
        For Langevin thermostat (when self.use_langevin is True), each element
        gets 3 types (mobile, fixed, thermo).

        Args:
            name: Unique identifier (e.g., 'tip', 'sheet', 'sub').
            config: The material configuration.
            n_layers: Number of layers (for 2D materials, default 1).
            regions: Optional list of region identifiers to expand per type.

        Returns:
            Dict mapping element names to their global atom type IDs.
        """
        cif_data = cifread(config.cif_path)
        elements = cif_data['elements']

        pot_type = normalize_potential_type(config.pot_type)
        if is_mlip(pot_type):
            if self.interlayer_style != 'lj':
                raise ValueError(
                    f"interlayer_style '{self.interlayer_style}' cannot be used with "
                    f"MLIP component '{name}' ({pot_type}): an MLIP cell is full-cell "
                    "only. Use interlayer_style = lj and dispersion = d3.")
            # One type per element; never open the (binary) model as text.
            pot_counts = {el: 1 for el in elements}
        else:
            pot_counts = count_atomtypes(config.pot_path, elements, pot_type=pot_type)
        # ilp/tmd: one type per chalcogen sub-layer (see chalcogen_split_elements).
        # The data file gets the same split from build_monolayer.
        split = (chalcogen_split_elements(self.interlayer_style, elements, pot_counts)
                 if splits_chalcogens(name) and not is_mlip(pot_type) else set())
        if split:
            pot_counts = {el: (2 if el in split else n) for el, n in pot_counts.items()}

        if is_mlip(pot_type):
            # An MLIP is scoped to the CELL, not the component:
            # ONE instance covers every type, so registering a second MLIP
            # component must not add a second sub-style token. Set rather than
            # increment. Rule 1 of validate_mlip_cell already guarantees every
            # MLIP component of a framework names the same model, so there is
            # nothing a second instance could express.
            self.potential_usage[pot_type] = 1

            # write_file needs the staged model path to build the style token,
            # but potential_usage is keyed by pot_type alone. Record it now so
            # two components sharing one framework resolve the same string.
            staged = self._get_potential_path(config.pot_path)
            self._mlip_source_paths.setdefault(pot_type, str(config.pot_path))
            existing = self._mlip_model_paths.setdefault(pot_type, staged)
            if existing != staged:
                # Every instance of one framework shares a single style token,
                # so a second, different model cannot be expressed. Config
                # validation should reject this; warn loudly if it slips past.
                logger.warning(
                    "Component '%s' registers a second %s model (%s); the first "
                    "(%s) is what every instance will use.",
                    name, pot_type, staged, existing,
                )
        else:
            self.potential_usage[pot_type] += (
                n_layers if self.is_sheet_lj(pot_type) else 1
            )

        apply_langevin = self.use_langevin and name in ('tip', 'sub')

        if n_layers > 1:
            self._assign_types(
                name, elements, pot_counts, n_layers, use_layer_ids=True, regions=regions,
                shared_label=split,
            )
        elif apply_langevin:
            self._assign_types(name, elements, pot_counts, regions=[None, 'fix', 'thermo'],
                               shared_label=split)
        else:
            self._assign_types(name, elements, pot_counts, regions=regions,
                               shared_label=split)

        component_map = self.types.get_element_map(name)

        self.components[name] = {
            'name': name,
            'config': config,
            'map': component_map,
            'elements': elements,
            'n_layers': n_layers,
            'use_langevin': self.use_langevin,
            'split_chalcogens': split,
        }

        return component_map

    def register_virtual_atom(self) -> int:
        """Register a virtual atom type for SMD spring attachment.
        
        The virtual atom is used in the 'virtual_atom' drive method where
        a massless particle is moved and the real atoms are tethered to it.
        It has minimal LJ interactions with all other types.
        
        Returns:
            The type ID assigned to the virtual atom.
        """
        if self.virtual_atom_type is not None:
            logger.warning("Virtual atom already registered as type %d", self.virtual_atom_type)
            return self.virtual_atom_type

        virtual_type = self.types.register(
            component='virtual',
            element='Virtual',
            pot_label='Virtual',
            layer=None,
            region=None
        )
        self.virtual_atom_type = virtual_type
        logger.info("Registered virtual atom as type %d", virtual_type)
        return virtual_type

    def _assign_types(
        self,
        name: str,
        elements: List[str],
        pot_counts: Dict[str, int],
        n_layers: int = 1,
        use_layer_ids: bool = False,
        regions: Optional[List[Optional[str]]] = None,
        shared_label: Optional[set] = None,
    ) -> None:
        """Assign atom types for components (with optional layer and region support).

        Uses element-first ordering to match the LAMMPS renumbering loop in
        stack_multilayer_sheet: for each element, assign types for all layers,
        then move to the next element.

        Args:
            name: Component name.
            elements: List of element symbols.
            pot_counts: Dict of atom counts per element.
            n_layers: Number of layers to create types for (default 1).
            use_layer_ids: If True, assign layer IDs to types; if False, layer=None.
            regions: List of region identifiers for Langevin thermostat
                (e.g., [None, 'fix', 'thermo']). If None, defaults to [None]
                (standard single type per element).
            shared_label: Elements whose ``count`` types all keep the bare
                element label (the ilp/tmd chalcogen split: two types the
                many-body potential knows as one ``S``).
        """
        if regions is None:
            regions = [None]
        shared_label = shared_label or set()

        for el in elements:
            count = pot_counts.get(el, 1)
            for layer_idx in range(n_layers):
                for t in range(count):
                    pot_label = el if count == 1 or el in shared_label else f"{el}{t+1}"
                    layer = layer_idx if use_layer_ids else None
                    for region in regions:
                        self.types.register(
                            component=name,
                            element=el,
                            pot_label=pot_label,
                            layer=layer,
                            region=region
                        )

    def _get_potential_path(self, original_path: str) -> str:
        """Return a LAMMPS-friendly potential path and stage file locally if requested.

        If `potentials_dir` is configured, copies the potential file there (once)
        and returns either `<potentials_prefix>/<filename>` if provided, or the
        absolute staged path. Otherwise, returns the original path unchanged.

        MLIP **model files** (recognised by extension) go to
        `shared_potentials_dir` instead when one is configured: one copy at the
        simulation root serves every run of a sweep. Classical potential files
        are unaffected and keep their per-run copy.
        """
        src_path = Path(original_path)

        if self.shared_potentials_dir and is_mlip_model_file(src_path):
            dest_dir = self.shared_potentials_dir
            prefix = self.shared_potentials_prefix
        elif self.potentials_dir:
            dest_dir = self.potentials_dir
            prefix = self.potentials_prefix
        else:
            return original_path

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name

        if not dest_path.exists():
            stage_model_file(src_path, dest_path)

        if prefix:
            return f"{prefix}/{src_path.name}"

        return str(dest_path)

    def is_sheet_lj(self, pot_type: str) -> bool:
        """Determine if potential requires explicit LJ for interlayer/cross interactions.

        Args:
            pot_type: The potential type string (e.g., 'sw', 'tersoff', 'airebo').

        Returns:
            True if explicit LJ is needed, False if potential handles it internally.
        """
        pot_lower = pot_type.lower()
        if is_mlip(pot_lower):
            # An MLIP behaves like an internal-LJ potential, and has no other
            # option: the per-layer alternative needed one instance per layer
            # with NULL-padded type maps, which BOTH frameworks reject.
            # One instance spans every layer of the component and
            # supplies the interlayer physics itself, so no explicit lj/cut is
            # added. Overlaying LJ on top is rejected on physics too:
            # the MLIP cutoff already spans the interlayer gap, so a UFF-mixed
            # LJ parameterised to BE the interlayer potential double-counts.
            return False
        if pot_lower in POTENTIALS_WITH_INTERNAL_LJ:
            return False
        if pot_lower in POTENTIALS_REQUIRING_LJ:
            return True

        logger.warning("Unknown potential type '%s', assuming LJ is required.", pot_type)
        return True

    # === MLIP helpers ======================================================

    def enable_d3(self) -> None:
        """Turn on the ``dispersion/d3`` overlay for this settings file.

        D3 is a full-cell-only capability (stock ``pair_style dispersion/d3``
        needs a real element for every atom type, so it cannot be restricted to
        the MLIP pairs of a partial cell) -- which costs nothing now that
        full-cell is the only supported topology. Enabling it switches the
        header to ``pair_style hybrid/overlay``. Builders call this; the
        cell-level capability check lives in ``mlip.validate_mlip_cell``.
        """
        self.d3_enabled = True

    def has_mlip(self) -> bool:
        """True if any registered component uses an MLIP framework."""
        return bool(self._mlip_pot_types())

    def mlip_framework(self) -> Optional[str]:
        """The single MLIP framework in this cell, or ``None`` if classical.

        A cell is uniform by construction (every component must use the
        SAME MLIP), so "the frameworks in use" is at most one name. Builders
        that assemble their own render context use this to supply the
        ``mlip_sheet`` / ``mlip_deck_directives`` template keys without having
        to guess which component speaks for the cell.

        The >1 case is refused at emission by
        :meth:`_mlip_self_interaction_lines`; this accessor stays quiet and
        returns the first so a context lookup can never be the thing that
        raises.
        """
        frameworks = self._mlip_pot_types()
        return frameworks[0] if frameworks else None

    def deck_directives(self) -> List[str]:
        """Deck lines this cell's MLIP style requires before ``read_data``.

        Empty for a classical cell and for deepmd (which declares none), so a
        non-MLIP deck renders byte-identically to before.
        """
        framework = self.mlip_framework()
        return list(get_mlip(framework).deck_directives) if framework else []

    def write_surrogate_file(self, output_path: Path) -> Path:
        """Write the build-time ``pair_style zero`` stand-in for this cell.

        FrictionSim2D runs LAMMPS *at generation time* to stack layers and to
        assemble hetero/flake cells. Those runs only place atoms and renumber
        types — but they ``include`` the real settings file, which for an MLIP
        would demand an MLIP-enabled LAMMPS on the generation machine and in CI.
        ``pair_style zero`` is stock LAMMPS, is defined for every
        element and exerts exactly zero force, so it satisfies the parser
        without being able to influence the geometry.

        **Only for LAMMPS runs that do not integrate or minimize.** A run whose
        result depends on the forces (an interface-spacing minimization, say)
        must use the configured ``lat_c`` instead — under this surrogate it
        would "minimize" a flat energy landscape and return nonsense.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"{self.get_masses_string()}\n"
            f"pair_style zero {ZERO_SURROGATE_CUTOFF}\n"
            f"pair_coeff * *\n",
            encoding='utf-8',
        )
        return output_path

    def _mlip_atom_list(self, pot_type: str) -> List[str]:
        """Build the cell-wide MLIP type map, with the §1.4 virtual placeholder.

        The map spans EVERY atom type in the cell (:meth:`TypeRegistry.build_full_map`)
        because an MLIP is cell-scoped, not component-scoped: both
        frameworks reject NULL entries and DeePMD refuses a
        type-range restriction (``pair_deepmd.cpp:816``), so the only shape
        either accepts is ``pair_coeff * *`` naming a real element per type.

        A virtual drive atom cannot stay ``NULL`` either -- ``pair_mace``
        segfaults on it, ``pair_deepmd`` errors out ("Pair hybrid sub-style
        deepmd is not used"), and ``dispersion/d3`` rejects it -- so it takes
        the framework's placeholder element; ``neigh_modify exclude`` (see
        :meth:`_virtual_exclusion_lines`) then empties its neighbour list, so it
        contributes only a constant isolated-atom energy and zero force.

        The map can no longer contain NULL by construction. What it *can* do is
        claim types belonging to a classical component, which is why
        :meth:`_mlip_self_interaction_lines` refuses a non-uniform cell before
        this map is ever emitted.
        """
        atom_list = self.types.build_full_map()
        if self.virtual_atom_type is None:
            return atom_list

        v_idx = self.virtual_atom_type - 1
        if v_idx >= len(atom_list):
            return atom_list

        atom_list = list(atom_list)
        atom_list[v_idx] = get_mlip(pot_type).virtual_atom_element
        return atom_list

    def _mlip_self_interaction_lines(self, hybrid: bool = True) -> List[str]:
        """Materialise the single ``pair_coeff`` line an MLIP cell needs.

        Deferred to write time deliberately: the full type map has to span
        components registered *after* the first :meth:`add_self_interaction`
        call, which is exactly what ``hetero_afm`` and ``flake_slide`` do.
        Emitting eagerly would silently truncate the map.

        Raises:
            ValueError: if the cell is not uniformly one MLIP. That is the
                "loud fail" for an SW tip beside an MLIP sheet, or MACE
                beside DeePMD. Neither can be expressed: one ``pair_coeff * *``
                claims every type, so a classical component would have its types
                silently absorbed by the MLIP, and a second framework's ``* *``
                would simply override the first.
        """
        frameworks = self._mlip_pot_types()
        if not frameworks or not self._mlip_components:
            return []

        if len(frameworks) > 1:
            raise ValueError(
                f"Two MLIP frameworks in one cell ({', '.join(sorted(frameworks))}). "
                f"Each needs its own 'pair_coeff * *' spanning every atom type, "
                f"so the second silently overrides the first. Every component in "
                f"a cell must use the SAME MLIP. See documentation/mlip.md §1."
            )

        pot_type = frameworks[0]
        foreign = [
            (name, normalize_potential_type(comp['config'].pot_type))
            for name, comp in self.components.items()
            if normalize_potential_type(comp['config'].pot_type) != pot_type
        ]
        if foreign:
            listed = ', '.join(f"{name} ({pot})" for name, pot in foreign)
            raise ValueError(
                f"Cannot mix '{pot_type}' with a classical potential in one cell: "
                f"{listed}. An MLIP's type map covers EVERY atom type "
                f"('pair_coeff * *' is the only form either framework accepts), "
                f"so those components' types would be silently "
                f"absorbed by {pot_type} rather than handled by their own "
                f"potential. Make the whole cell {pot_type}, or the whole cell "
                f"classical. See documentation/mlip.md §1."
            )

        atom_list = self._mlip_atom_list(pot_type)
        self._check_model_covers(pot_type, atom_list)
        pot_path = self._mlip_model_paths.get(pot_type, '')
        # potential_usage is 1 for an MLIP by construction, so a
        # full-cell MLIP is never a numbered hybrid sub-style.
        index_str = f" {self.potential_indices[pot_type]}" \
            if self.potential_usage[pot_type] > 1 else ""
        cmd = get_mlip(pot_type).coeff_line(index_str, pot_path, atom_list, hybrid=hybrid,
                                            precision=self._mlip_precision(pot_type))
        return [f"{cmd} # {', '.join(self._mlip_components[pot_type])}"]

    def _mlip_precision(self, pot_type: str) -> str:
        """``float64`` | ``float32`` of the cell's ``pot_type`` components.

        Read from the components' configs (``SheetConfig.precision``); a cell
        whose sections disagree is refused at config time, so the first wins.
        """
        for comp in self.components.values():
            cfg = comp['config']
            if normalize_potential_type(cfg.pot_type) == pot_type:
                return getattr(cfg, 'precision', 'float64')
        return 'float64'

    def _check_model_covers(self, pot_type: str, atom_list: List[str]) -> None:
        """Refuse a type map naming an element the model does not have.

        This matters more than a typical validation because the
        failure is **silent**: measured, an element the model lacks
        does not error -- LAMMPS maps the types to the model's first N elements
        and returns a plausible, wrong energy.

        Only enforced where the answer is knowable without loading the model
        (MACE's symmetrix json). :func:`mlip.model_elements` returns ``None``
        otherwise -- for DeePMD, and for any file it cannot parse -- and this
        is then a no-op, so it can only add a diagnosis, never block a build
        that would have worked.
        """
        covered = model_elements(pot_type, self._mlip_source_paths.get(pot_type, ''))
        if covered is None:
            return
        missing = sorted(set(atom_list) - covered)
        if not missing:
            return
        raise ValueError(
            f"The {pot_type} model does not cover every element in this cell: "
            f"missing {', '.join(missing)} (model has "
            f"{', '.join(sorted(covered))}). Naming an element the model lacks "
            f"does NOT error at run time -- LAMMPS maps those types to the "
            f"model's first N elements and returns a plausible, WRONG energy. "
            f"Re-extract the model with every element the cell "
            f"needs, e.g. `symmetrix_extract_mace -m <model> --chemical-symbols "
            f"{' '.join(sorted(set(atom_list)))}`. See documentation/mlip.md §1."
        )

    def _all_self_interaction_lines(self, hybrid: bool = True) -> List[str]:
        """Every self-interaction line, MLIP ones materialised at write time.

        Classical lines are always stored in their hybrid form; outside
        ``pair_style hybrid`` the style name and index are dropped here.
        """
        classical = (self.self_interaction_commands if hybrid else
                     [self._strip_hybrid_prefix(cmd) for cmd in self.self_interaction_commands])
        return self._mlip_self_interaction_lines(hybrid) + classical

    def _is_mlip_component(self, name: str) -> bool:
        """True if ``name`` is a registered component using an MLIP."""
        comp = self.components.get(name)
        if not comp:
            return False
        return is_mlip(normalize_potential_type(comp['config'].pot_type))

    def _mlip_pot_types(self) -> List[str]:
        """MLIP frameworks in use, in registration order."""
        return [pot for pot in self.potential_usage if is_mlip(pot)]

    def _d3_element_list(self) -> List[str]:
        """One real element symbol per atom type, for the D3 coeff line."""
        mlips = self._mlip_pot_types()
        placeholder = get_mlip(mlips[0]).virtual_atom_element if mlips else 'H'
        return [
            placeholder if atype.element == 'Virtual' else atype.element
            for atype in self.types
        ]

    def _virtual_exclusion_lines(self) -> List[str]:
        """``neigh_modify`` lines hiding the virtual drive atom from MLIP/D3.

        DELIBERATELY one line per type rather than a single wildcard line:
        ``neigh_modify exclude type M N`` takes two explicit numeric type IDs
        and has **no** wildcard form, so ``exclude type <v> *`` is a parse error
        that would kill every MLIP run. The Milestone-0 deck ``m0_4_virtual_exclude.in``
        enumerated the same pairs (including the v-v self pair) for this reason.
        """
        v = self.virtual_atom_type
        if v is None:
            return []
        return [f"neigh_modify exclude type {v} {t}"
                for t in range(1, len(self.types) + 1)]

    def add_self_interaction(self, component_name: str,
                            layer: Optional[int] = None):
        """Add self-interaction (many-body potential) for a component.
        
        For multi-layer materials, can specify a specific layer or all layers.

        Args:
            component_name: Name of the registered component.
            layer: Specific layer index (0-based) or None for all/single layer.
        """
        comp = self.components.get(component_name)
        if not comp:
            raise ValueError(f"Component '{component_name}' not registered.")
        c_conf = comp['config']
        n_layers = comp['n_layers']

        pot_type = normalize_potential_type(c_conf.pot_type)

        if is_mlip(pot_type):
            # An MLIP is cell-scoped: record the request and emit ONE line for
            # the whole cell at write time (_mlip_self_interaction_lines). The
            # map must span components registered after this call, so it cannot
            # be built now. No potential_indices bump either -- there is only
            # ever one instance, hence no hybrid sub-style number.
            self._mlip_components.setdefault(pot_type, []).append(component_name)
            return

        self.potential_indices[pot_type] += 1
        pot_index = self.potential_indices[pot_type]

        if n_layers > 1 and self.is_sheet_lj(pot_type):
            layers_to_process = [layer] if layer is not None else range(n_layers)
            for l in layers_to_process:
                self._add_layer_self_interaction(comp, l, pot_index)
        else:
            cmd = self._self_pair_coeff(component_name, c_conf, pot_type, pot_index, None)
            self.self_interaction_commands.append(f"{cmd} # {component_name}")

    def _self_pair_coeff(self, name: str, c_conf: ComponentConfig, pot_type: str,
                         pot_index: int, layer: Optional[int]) -> str:
        """``pair_coeff * *`` line of one classical self-interaction (no comment).

        The hybrid sub-style index is written only when ``pot_type`` is used by
        more than one component; ``layer`` restricts the NULL map to that layer.
        """
        index_str = f" {pot_index}" if self.potential_usage[pot_type] > 1 else ""
        atom_list = self.types.build_null_map(name, layer=layer)
        pot_path = self._get_potential_path(c_conf.pot_path)
        return f"pair_coeff * * {pot_type}{index_str} {pot_path} {' '.join(atom_list)}"

    def _add_layer_self_interaction(self, comp: Dict, layer: int,
                                    base_pot_index: int):
        """Add self-interaction for a specific layer.

        Classical potentials only: this is the per-layer, NULL-partitioned
        topology, which no MLIP can express, and
        :meth:`is_sheet_lj` is ``False`` for every MLIP so an MLIP component
        never reaches here.

        Args:
            comp: Component dictionary.
            layer: Layer index (0-based).
            base_pot_index: Base potential index for hybrid style.
        """
        c_conf = comp['config']
        name = comp['name']
        pot_type = normalize_potential_type(c_conf.pot_type)

        pot_index = base_pot_index + layer
        self.potential_indices[pot_type] = pot_index

        cmd = self._self_pair_coeff(name, c_conf, pot_type, pot_index, layer)
        self.self_interaction_commands.append(f"{cmd} # {name} Layer {layer+1}")

    def add_cross_interaction(
        self,
        comp1_name: str,
        comp2_name: str,
    ):
        """Add LJ cross-interaction between two components.

        Args:
            comp1_name: Name of first component.
            comp2_name: Name of second component.
        """
        mlip1 = self._is_mlip_component(comp1_name)
        mlip2 = self._is_mlip_component(comp2_name)
        if mlip1 and mlip2:
            # The MLIP already describes this pair -- its cutoff spans the
            # inter-component gap -- so a UFF-mixed LJ parameterised to BE that
            # interaction double-counts it (the same argument as between layers,
            # applied between components).
            logger.debug(
                "Skipping %s-%s cross LJ: both components are covered by one "
                "MLIP, which supplies that interaction itself.",
                comp1_name, comp2_name,
            )
            return
        if mlip1 != mlip2:
            mlip_name, other = (comp1_name, comp2_name) if mlip1 \
                else (comp2_name, comp1_name)
            mlip_pot = normalize_potential_type(
                self.components[mlip_name]['config'].pot_type)
            other_pot = normalize_potential_type(
                self.components[other]['config'].pot_type)
            raise ValueError(
                f"Cannot add a cross-interaction between {mlip_pot} component "
                f"'{mlip_name}' and {other_pot} component '{other}': an MLIP's "
                f"type map already claims every atom type, so this LJ term "
                f"would double-count on types the MLIP owns. Such a cell is "
                f"refused outright; it should not have reached "
                f"here. Make the whole cell one MLIP, or the whole cell "
                f"classical."
            )

        elements1 = self.types.elements_in_component(comp1_name)
        elements2 = self.types.elements_in_component(comp2_name)

        if not elements1 or not elements2:
            raise ValueError(f"Components '{comp1_name}' and/or '{comp2_name}' not found.")

        for el1 in elements1:
            for el2 in elements2:
                epsilon, sigma = self._get_lj_params(el1, el2)

                types1 = self.types.get_element_type_range(comp1_name, el1)
                types2 = self.types.get_element_type_range(comp2_name, el2)

                if (min(self.types.ids_by_component_element(comp1_name, el1))
                        > min(self.types.ids_by_component_element(comp2_name, el2))):
                    types1, types2 = types2, types1

                eps_str = self._format_lj_value(epsilon)
                sig_str = self._format_lj_value(sigma)
                cmd = f"pair_coeff {types1} {types2} lj/cut {eps_str} {sig_str}"
                self.cross_interaction_commands.append(
                    f"{cmd} # {el1}({comp1_name})-{el2}({comp2_name})"
                )

    def add_interlayer_interaction(
        self,
        component_name: str,
        layer_pairs: Optional[List[Tuple[int, int]]] = None,
        low_interaction: bool = False
    ):
        """Add LJ interactions between layers of a multi-layer 2D material.

        Args:
            component_name: Name of the sheet component.
            layer_pairs: List of (layer_i, layer_j) tuples, or None for all
                pairs.
            low_interaction: If True, use near-zero epsilon (ghost/weak
                interaction).
        """
        comp = self.components.get(component_name)
        if not comp:
            raise ValueError(f"Component '{component_name}' not registered.")
        if self._is_mlip_component(component_name):
            # One MLIP instance spans every layer and supplies the interlayer
            # physics itself; overlaying LJ double-counts.
            logger.debug(
                "Skipping interlayer LJ for '%s': its MLIP covers every layer.",
                component_name,
            )
            return
        n_layers = comp['n_layers']
        elements = comp['elements']

        if n_layers < 2:
            logger.warning(
                "Component '%s' has only 1 layer, skipping interlayer interaction.", component_name
            )
            return

        if layer_pairs is None:
            layer_pairs = [(i, j) for i in range(n_layers) for j in range(i+1, n_layers)]

        for layer_i, layer_j in layer_pairs:
            for el1 in elements:
                for el2 in elements:
                    epsilon, sigma = self._get_lj_params(el1, el2)
                    if low_interaction:
                        epsilon = 1e-100  # Near-zero for ghost interaction

                    types_i = self.types.ids_by_component_layer_element(
                        component_name, layer_i, el1
                    )
                    types_j = self.types.ids_by_component_layer_element(
                        component_name, layer_j, el2
                    )

                    if not types_i or not types_j:
                        continue

                    t1_str = self.types.format_type_range(types_i)
                    t2_str = self.types.format_type_range(types_j)

                    if types_i[0] > types_j[0]:
                        t1_str, t2_str = t2_str, t1_str

                    eps_str = self._format_lj_value(epsilon)
                    sig_str = self._format_lj_value(sigma)
                    cmd = f"pair_coeff {t1_str} {t2_str} lj/cut {eps_str} {sig_str}"
                    self.cross_interaction_commands.append(
                        f"{cmd} # {el1}(L{layer_i+1})-{el2}(L{layer_j+1})"
                    )

    # === Interlayer styles ===

    def set_interlayer(self, style: str, file: Optional[str] = None,
                       g_by_component: Optional[Dict[str, float]] = None,
                       base_dir: Optional[Union[str, Path]] = None) -> None:
        """Choose the interaction between 2D layers for :meth:`add_interlayer`.

        Args:
            style: One of ``lj``, ``lj_g``, ``ilp``, ``kc``.
            file: ILP/KC parameter file (name or path); None picks the bundled
                default for the sheet elements.
            g_by_component: LJ-G Gaussian strength (meV) per component name.
                A missing component falls back to its config's
                ``interlayer_g`` or the built-in table.
            base_dir: Directory the parameter file is resolved against
                (normally the config's directory); None = working directory.

        Raises:
            ValueError: On an unknown style.
        """
        if style not in INTERLAYER_STYLES:
            raise ValueError(
                f"Unknown interlayer_style '{style}'; expected one of "
                f"{list(INTERLAYER_STYLES)}.")
        self._refuse_mlip_interlayer(style, list(self.components))
        for name, comp in self.components.items():
            if self._is_mlip_component(name):
                continue
            if splits_chalcogens(name) and comp.get('split_chalcogens', set()) != \
                    chalcogen_split_elements(style, comp['elements'], count_atomtypes(
                        comp['config'].pot_path, comp['elements'],
                        pot_type=normalize_potential_type(comp['config'].pot_type))):
                raise ValueError(
                    f"set_interlayer('{style}') must be called before register_component"
                    f"('{name}'): under ilp/tmd its chalcogen sub-layers need their "
                    "own atom types, which are assigned at registration.")
        self.interlayer_style = style
        self._interlayer_file_name = file
        self._interlayer_base_dir = Path(base_dir) if base_dir is not None else None
        self._g_by_component = dict(g_by_component or {})

    @property
    def needs_molecule_ids(self) -> bool:
        """True when the style excludes intra-layer pairs by molecule ID."""
        return self.interlayer_style in ('ilp', 'kc')

    @property
    def interlayer_substyle(self) -> Optional[str]:
        """LAMMPS sub-style carrying the interlayer term; None for ``lj``.

        For ``ilp`` this depends on the sheet elements: those of the
        components passed to :meth:`add_interlayer`, or -- before that call --
        of every registered component whose name starts with ``sheet``.
        """
        style = self.interlayer_style
        if style == 'lj':
            return None
        if style == 'lj_g':
            return 'kolmogorov/crespi/z'
        if style == 'kc':
            return 'kolmogorov/crespi/full'
        if self._interlayer_substyle is not None:
            return self._interlayer_substyle
        names = self._interlayer_components or [
            n for n in self.components if n.startswith('sheet')]
        return ilp_substyle(self._elements_of(names))

    def _elements_of(self, components: Sequence[str]) -> set:
        """Union of the registered elements of ``components``."""
        found = set()
        for name in components:
            found.update(self.types.elements_in_component(name))
        return found

    def add_interlayer(
        self,
        components: Sequence[str],
        low_interaction_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        """Add the interaction between the layers of ``components``.

        The single entry point for every builder:

        - ``lj``: :meth:`add_interlayer_interaction` per component -- the same
          lines, in the same order, as calling it directly. With
          ``low_interaction_pairs`` those layer pairs get the ghost epsilon
          after the real ones.
        - ``lj_g``: the same LJ, plus the Gaussian as three or more
          ``kolmogorov/crespi/z`` instances: +g/2 over every listed sheet
          type and -g/2 within each layer.
        - ``ilp``/``kc``: no sheet-sheet ``lj/cut``; one many-body
          ``pair_coeff * *`` over all listed components.

        Cross-material LJ between hetero components is not touched: callers
        emit it with :meth:`add_cross_interaction` as before.

        Raises:
            ValueError: On an unregistered component, ``low_interaction_pairs``
                with a non-``lj`` style, a missing Gaussian g, an element with
                two different g, or an ILP/KC file that cannot be defaulted.
        """
        components = list(components)
        for name in components:
            if name not in self.components:
                raise ValueError(f"Component '{name}' not registered.")
        style = self.interlayer_style
        self._refuse_mlip_interlayer(style, components)
        if low_interaction_pairs and style != 'lj':
            raise ValueError(
                f"interlayer_style '{style}' cannot use ghost (low-interaction) "
                "layer pairs; those belong to the atom_bonds/com_spring "
                "constraint modes, which need interlayer_style = lj.")

        if style in ('lj', 'lj_g'):
            for name in components:
                # lj_g passes single-layer components (hetero materials,
                # flakes) for the Gaussian alone: no LJ of their own to add.
                if style == 'lj' or self.components[name]['n_layers'] > 1:
                    self._add_lj_interlayer(name, low_interaction_pairs)

        n_layers_total = sum(self.components[n]['n_layers'] for n in components)
        if style == 'lj' or n_layers_total < 2:
            return
        self._interlayer_components = components

        if style == 'lj_g':
            self._add_gaussian(components)
        else:
            self._add_many_body(style, components)

    def _refuse_mlip_interlayer(self, style: str, components: Sequence[str]) -> None:
        """Raise if a non-``lj`` interlayer style would touch an MLIP component.

        A full-cell MLIP already supplies the interlayer physics through one
        ``pair_coeff * *`` over every type; there is no pair slot left for
        ILP/KC/LJ-G, and ``dispersion = d3`` is the only vdW lever
        (documentation/mlip.md). Config validation refuses this first; this
        is the manager-level backstop for programmatic callers.
        """
        if style == 'lj':
            return
        mlip = [n for n in components
                if n in self.components and self._is_mlip_component(n)]
        if mlip:
            raise ValueError(
                f"interlayer_style '{style}' cannot be used with MLIP component(s) "
                f"{mlip}: an MLIP cell is full-cell only. Use interlayer_style = lj "
                "and dispersion = d3 for the vdW term.")

    def _add_lj_interlayer(self, name: str,
                           low_pairs: Optional[Sequence[Tuple[int, int]]]) -> None:
        """Interlayer LJ for one component (plain or with ghost pairs)."""
        if not low_pairs:
            self.add_interlayer_interaction(name)
            return
        n_layers = self.components[name]['n_layers']
        ghost = [tuple(p) for p in low_pairs]
        real = [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)
                if (i, j) not in ghost]
        if real:
            self.add_interlayer_interaction(name, layer_pairs=real)
        self.add_interlayer_interaction(name, layer_pairs=ghost,
                                        low_interaction=True)

    def _component_g(self, name: str) -> float:
        """Gaussian g (meV) of one component: explicit, config override, table."""
        if name in self._g_by_component:
            return float(self._g_by_component[name])
        config = self.components[name]['config']
        g = gaussian_g(config.mat, getattr(config, 'interlayer_g', None))
        if g is None:
            raise ValueError(
                f"interlayer_style 'lj_g' needs a Gaussian strength g (meV) for "
                f"component '{name}' ({config.mat}): pass g_by_component or set "
                "interlayer_g.")
        return g

    def _add_gaussian(self, components: List[str]) -> None:
        """Record the LJ-G pair table and the KC/z instances."""
        g_of: Dict[str, float] = {name: self._component_g(name) for name in components}
        owner: Dict[str, str] = {}
        el_g: Dict[str, float] = {}
        for name in components:
            for el in self.types.elements_in_component(name):
                if el in el_g and el_g[el] != g_of[name]:
                    first = owner[el]
                    raise ValueError(
                        f"lj_g: element '{el}' is in components '{first}' "
                        f"({self.components[first]['config'].mat}, g = {el_g[el]} meV) "
                        f"and '{name}' ({self.components[name]['config'].mat}, "
                        f"g = {g_of[name]} meV) with different Gaussian strengths; "
                        "a KC/z file holds one value per element pair.")
                owner.setdefault(el, name)
                el_g[el] = g_of[name]

        pairs: Dict[Tuple[str, str], float] = {}
        for el1, g1 in el_g.items():
            for el2, g2 in el_g.items():
                pairs[(el1, el2)] = g1 if owner[el1] == owner[el2] else mixed_g(g1, g2)
        self._gaussian_pairs = pairs

        label = ' '.join(components)
        instances = [(1, self.types.build_element_map(components),
                      f"{label}: all layers +g/2")]
        for name in components:
            n_layers = self.components[name]['n_layers']
            layers = range(n_layers) if n_layers > 1 else [None]
            for layer in layers:
                where = f"{name} layer {layer + 1}" if layer is not None else name
                instances.append((-1, self.types.build_element_map([name], layer),
                                  f"{where}: same-layer -g/2"))
        self._gaussian_instances = instances

    def _add_many_body(self, style: str, components: List[str]) -> None:
        """One ``pair_coeff * *`` for ILP or KC-full over ``components``."""
        elements = self._elements_of(components)
        name = self._interlayer_file_name or default_interlayer_file(style, elements)
        if name is None:
            raise ValueError(
                f"interlayer_style '{style}' has no bundled parameter file for "
                f"elements {sorted(elements)}: set general interlayer_file.")
        source = resolve_interlayer_file(name, style, self._interlayer_base_dir)
        self._interlayer_source = source
        substyle = ilp_substyle(elements) if style == 'ilp' else 'kolmogorov/crespi/full'
        self._interlayer_substyle = substyle
        path = self._get_potential_path(str(source))
        emap = ' '.join(self.types.build_element_map(components))
        self._interlayer_commands = [
            f"pair_coeff * * {substyle} {path} {emap} # {' '.join(components)}"]

    def _interlayer_active(self) -> bool:
        """True when add_interlayer emitted a non-LJ interlayer term."""
        return bool(self._interlayer_commands or self._gaussian_instances)

    def _interlayer_style_parts(self) -> List[str]:
        """pair_style parts appended after the normal ones under overlay."""
        if self._gaussian_instances:
            return [f"kolmogorov/crespi/z {self.lj_cutoff}"] * len(self._gaussian_instances)
        if self._interlayer_commands:
            if self.interlayer_style == 'ilp':
                return [f"{self._interlayer_substyle} {ILP_CUTOFF} {TAPER_FLAG}"]
            # KC-full's taper flag must match its parameter file: the
            # `*_taper.KC` files are fitted with the taper, the others without.
            source = self._interlayer_source
            tap = 1 if source is not None and source.name.endswith('_taper.KC') else 0
            return [f"{self._interlayer_substyle} {KC_FULL_CUTOFF} {tap}"]
        return []

    def max_pair_cutoff(self) -> float:
        """Largest pair cutoff the written settings apply (Angstrom).

        What the periodic-z vacuum must exceed: the ``lj/cut`` cutoff when an
        LJ term is written, the ILP/KC cutoff (or KC/z's, for ``lj_g``), and
        the long-range part of any potential with its own LJ (ReaxFF taper
        and QEq, REBOMoS LJ), the ``dispersion/d3`` overlay's ``d3_cutoff``,
        and ``lj_cutoff`` for an MLIP (its model cutoff is not read from the
        model file, and every shipped model's is shorter). Short-ranged
        many-body potentials (sw, tersoff, ...) only ever appear beside an LJ
        term with a longer cutoff.
        """
        lj_cutoff = float(self.settings.potential.lj_cutoff)
        cutoffs = [0.0]
        if self.cross_interaction_commands or self.virtual_atom_type is not None:
            cutoffs.append(lj_cutoff)
        for part in self._interlayer_style_parts():
            cutoffs.append(float(part.split()[1]))
        for comp in self.components.values():
            config = comp['config']
            cutoffs.append(internal_pair_cutoff(
                config.pot_type, config.pot_path, fallback=lj_cutoff))
        if self.d3_enabled:
            cutoffs.append(float(self.settings.potential.d3_cutoff))
        return max(cutoffs)

    def _write_gaussian_commands(self, settings_dir: Path) -> List[str]:
        """Write gaussian_plus/minus.KC and return the KC/z pair_coeff lines.

        The files go into ``potentials_dir`` (referenced through
        ``potentials_prefix`` like a copied potential) or, without one, next
        to the settings file (referenced by absolute path).
        """
        target = self.potentials_dir or settings_dir
        refs: Dict[int, str] = {}
        for sign, fname in ((1, 'gaussian_plus.KC'), (-1, 'gaussian_minus.KC')):
            path = write_gaussian_kc_file(target / fname, self._gaussian_pairs, sign)
            self._generated_files[fname] = hashlib.sha256(path.read_bytes()).hexdigest()
            if self.potentials_dir and self.potentials_prefix:
                refs[sign] = f"{self.potentials_prefix}/{fname}"
            else:
                refs[sign] = str(path)
        return [
            f"pair_coeff * * kolmogorov/crespi/z {i} {refs[sign]} {' '.join(emap)} # {note}"
            for i, (sign, emap, note) in enumerate(self._gaussian_instances, 1)
        ]

    def interlayer_provenance(self) -> Dict[str, Any]:
        """Interlayer choice for provenance; ``comparison`` is the caller's."""
        sha = None
        if self._interlayer_source is not None and self._interlayer_source.exists():
            sha = hashlib.sha256(self._interlayer_source.read_bytes()).hexdigest()
        g_mev: Dict[str, Dict[str, Any]] = {}
        if self.interlayer_style == 'lj_g':
            for name in self._interlayer_components:
                config = self.components[name]['config']
                explicit_override = getattr(config, 'interlayer_g', None) is not None
                g_mev[name] = {
                    'value': self._component_g(name),
                    'source': 'override' if explicit_override else 'table',
                }
        return {
            'style': self.interlayer_style,
            'file': self._interlayer_source.name if self._interlayer_source else None,
            'sha256': sha,
            'g_meV': g_mev,
            'generated_files': dict(self._generated_files),
            'comparison': None,
        }

    def calculate_gap(self, comp1_name: str, comp2_name: str,
                        buffer: float = 0.5) -> float:
        """Calculate the recommended gap between two components.

        Based on the maximum sigma from UFF mixing rules plus a buffer.

        Args:
            comp1_name: Name of first component.
            comp2_name: Name of second component.
            buffer: Additional distance to add (default 0.5 Angstrom).

        Returns:
            Recommended gap distance in Angstroms.
        """
        comp1 = self.components.get(comp1_name)
        if not comp1:
            raise ValueError(f"Component '{comp1_name}' not registered.")
        comp2 = self.components.get(comp2_name)
        if not comp2:
            raise ValueError(f"Component '{comp2_name}' not registered.")

        max_sigma = 0.0
        for e1 in set(comp1['elements']):
            for e2 in set(comp2['elements']):
                _, sigma = self._get_lj_params(e1, e2)
                max_sigma = max(max_sigma, sigma)

        return max_sigma + buffer

    def get_masses_string(self) -> str:
        """Generate LAMMPS mass commands for all registered types.

        Returns:
            String with mass commands, one per line.
        """
        element_types: Dict[str, List[int]] = defaultdict(list)
        for atype in self.types:
            element_types[atype.element].append(atype.type_id)

        lines = []
        for element, types in element_types.items():
            # Emit one `mass <range> <m>` per MAXIMAL CONTIGUOUS RUN of this
            # element's type IDs. The LAMMPS `mass` command takes only a single
            # ID or a `lo*hi` span (never an ID list), so collapsing a
            # NON-contiguous list (e.g. Mo=[1,3] when the same element appears in
            # two hetero components) into one `1*3` span would silently engulf the
            # intervening type (2) -- and a later element's span (S=[2,4] -> `2*4`)
            # would then overwrite the engulfed type under LAMMPS last-wins,
            # corrupting its mass. Per-run emission keeps every span disjoint, so
            # no element can clobber another's type. A normally-contiguous
            # single-material block stays a single `lo*hi` line (output
            # unchanged); only cross-component non-contiguous cases split.
            for run in self.types.contiguous_runs(types):
                type_range = self.types.format_type_range(run)
                if element == 'Virtual':
                    lines.append(f"mass {type_range} 1.0 #Virtual atom")
                    continue
                mass = _element_mass(element)
                if mass is None:
                    lines.append(f"mass {type_range} 1.0 #{element} (unknown)")
                else:
                    lines.append(f"mass {type_range} {mass:.6f} #{element}")

        return "\n".join(lines)

    def get_component_groups_string(self) -> str:
        """Generate LAMMPS group commands for all components.
        
        For Langevin components (tip, sub), generates:
            - {component}_all: All types for component
            - {component}_fix: Fix region types
            - {component}_thermo: Thermo region types
        
        For sheet components, generates:
            - 2D_all: All sheet types
            - layer_N: Types for each layer (if multi-layer)
        
        For mobile group, generates union of thermo regions if Langevin is used.
        
        Returns:
            String with group commands, one per line.
        """
        lines = []

        def add_group(group: str, ids) -> None:
            if ids:
                lines.append(f"group {group} type {' '.join(map(str, ids))}")

        has_langevin = False
        langevin_thermo_groups = []
        for comp in self.components.values():
            name = comp['name']
            n_layers = comp['n_layers']
            use_langevin = comp.get('use_langevin', False)

            all_types = self.types.ids_by_component(name)

            if name == 'sheet':
                add_group("2D_all", all_types)

                core_types = [
                    atype.type_id for atype in self.types
                    if atype.component == name and atype.region is None
                ]
                edge_types = [
                    atype.type_id for atype in self.types
                    if atype.component == name and atype.region == 'edge'
                ]

                add_group("2D_core", core_types)
                add_group("2D_edge", edge_types)

                # Layer groups for EVERY layer count, monolayers included:
                # the registry's comx_L1... channels reference layer_1
                # unconditionally. A monolayer whose types carry
                # no per-layer tag falls back to the whole sheet.
                for layer in range(n_layers):
                    layer_types = self.types.get_layer_group_string(name, layer)
                    if not layer_types and n_layers == 1 and all_types:
                        layer_types = ' '.join(map(str, all_types))
                    if layer_types:
                        lines.append(f"group layer_{layer+1} type {layer_types}")

            elif use_langevin and name in ('tip', 'sub'):
                # Langevin components: split into all, fix, thermo
                has_langevin = True

                normal_types = []
                fix_types = []
                thermo_types = []

                for atype in self.types:
                    if atype.component == name:
                        if atype.region is None:
                            normal_types.append(atype.type_id)
                        elif atype.region == 'fix':
                            fix_types.append(atype.type_id)
                        elif atype.region == 'thermo':
                            thermo_types.append(atype.type_id)

                add_group(f"{name}_all", sorted(all_types))
                add_group(f"{name}_fix", sorted(fix_types))
                add_group(f"{name}_thermo", sorted(thermo_types))
                if thermo_types:
                    langevin_thermo_groups.append(f"{name}_thermo")

            else:
                add_group(f"{name}_all", all_types)

        if has_langevin and langevin_thermo_groups:
            lines.append(f"group mobile union {' '.join(langevin_thermo_groups)}")

        return "\n".join(lines)

    def _is_single_potential(self) -> bool:
        """Check if only one potential style is in use (no hybrid needed)."""
        total = sum(self.potential_usage.values())
        return total <= 1 and not self.cross_interaction_commands

    @staticmethod
    def _strip_hybrid_prefix(cmd: str) -> str:
        """Strip the pot_type and index from a hybrid-style pair_coeff command.

        Converts 'pair_coeff * * sw 1 /path Mo S # comment'
        to 'pair_coeff * * /path Mo S # comment' for non-hybrid mode.
        """
        parts = cmd.split('#', 1)
        main = parts[0].strip()
        comment = f" # {parts[1].strip()}" if len(parts) > 1 else ""

        tokens = main.split()
        # pair_coeff * * pot_type [index] path elements...
        # Find the path token (starts with / or contains .)
        for i in range(3, len(tokens)):
            if '/' in tokens[i] or '.' in tokens[i]:
                cleaned = ' '.join(tokens[:3] + tokens[i:])
                return f"{cleaned}{comment}"

        return cmd

    def _pair_style_parts(self, has_lj: bool) -> List[str]:
        """The ``pair_style`` sub-style tokens, in emission order.

        ReaxFF always carries ``NULL`` plus its keywords. MLIPs take their
        argument string from the registry (deepmd carries the model path here;
        symmetrix/mace carries neither -- its model path goes in the pair_coeff
        line and it takes no style keywords).
        """
        style_parts = []
        for pot_type, count in self.potential_usage.items():
            for _ in range(count):
                if pot_type.lower() in REAXFF_TYPES:
                    safezone = self.settings.potential.reaxff_safezone
                    mincap = self.settings.potential.reaxff_mincap
                    style_parts.append(
                        f"{pot_type} NULL safezone {safezone} mincap {mincap}"
                    )
                elif is_mlip(pot_type):
                    style_parts.append(
                        get_mlip(pot_type).style_token(
                            self._mlip_model_paths.get(pot_type, ''),
                            self._mlip_precision(pot_type),
                        )
                    )
                else:
                    style_parts.append(pot_type)

        # D3 sits next to the potential it overlays, before the LJ sub-style.
        if self.d3_enabled:
            style_parts.append(d3_style_token(self.settings))

        if has_lj:
            style_parts.append(f"lj/cut {self.settings.potential.lj_cutoff}")
        style_parts.extend(self._interlayer_style_parts())
        return style_parts

    def write_file(self, output_path: Path):
        """Write the complete potential settings file.

        Args:
            output_path: Path for the output file (e.g., system.in.settings).
        """
        # An MLIP's `pair_coeff * *` already spans EVERY type including the
        # virtual drive atom: no NULL type map is possible for an MLIP, so
        # _mlip_atom_list fills the virtual slot with the framework's
        # virtual_atom_element, and _virtual_exclusion_lines keeps any virtual
        # pair from ever being built. The placeholder lj/cut is therefore not
        # just redundant there but harmful -- under plain `hybrid` a
        # `pair_coeff * <v> lj/cut` REASSIGNS the virtual type away from the
        # MLIP, leaving its type map covering types it no longer owns, which
        # segfaults as soon as create_atoms makes a real virtual atom.
        # Classical potentials still need it: their `* *` lines NULL out types
        # the potential does not know, leaving the virtual type unassigned.
        virtual_needs_lj = (self.virtual_atom_type is not None
                            and not self._mlip_pot_types())
        has_lj = bool(self.cross_interaction_commands) or virtual_needs_lj
        # Both a many-body/Gaussian interlayer term and a D3 dispersion term
        # are second pair styles over pairs another style already owns, so
        # either one needs a hybrid/overlay header -- even for a lone
        # full-cell MLIP (D3) or a lone classical sheet potential (interlayer).
        overlay = self._interlayer_active() or self.d3_enabled
        use_hybrid = ((not self._is_single_potential())
                      or virtual_needs_lj
                      or overlay)
        # Materialise MLIP lines BEFORE opening the file: a non-uniform cell
        # raises here, and it must not leave a half-written deck behind.
        self_interaction_lines = self._all_self_interaction_lines(use_hybrid)
        style_parts = self._pair_style_parts(has_lj)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        interlayer_cmds = list(self._interlayer_commands)
        if self._gaussian_instances:
            interlayer_cmds = self._write_gaussian_commands(output_path.parent.resolve())

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.get_masses_string())
            f.write("\n")
            f.write(self.get_component_groups_string())
            f.write("\n")

            if self.virtual_atom_type is not None and (self._mlip_pot_types()
                                                       or self.d3_enabled):
                f.write("# Virtual drive atom is invisible to the MLIP/D3 graph\n")
                for line in self._virtual_exclusion_lines():
                    f.write(f"{line}\n")
                f.write("\n")

            if use_hybrid:
                hybrid_style = 'hybrid/overlay' if overlay else 'hybrid'
                f.write(f"pair_style {hybrid_style} {' '.join(style_parts)}\n")
            elif style_parts:
                f.write(f"pair_style {style_parts[0]}\n")

            if self_interaction_lines:
                f.write("# Self-interactions (intra-component)\n")
                for cmd in self_interaction_lines:
                    f.write(f"{cmd}\n")
                f.write("\n")

            if self.d3_enabled:
                f.write("# D3 dispersion overlay (full-cell only)\n")
                f.write(f"{d3_coeff_line(self._d3_element_list())}\n")
                f.write("\n")

            if self.cross_interaction_commands:
                f.write("# Cross-interactions (inter-component LJ)\n")
                for cmd in self.cross_interaction_commands:
                    f.write(f"{cmd}\n")
                f.write("\n")

            if interlayer_cmds:
                f.write(f"# Interlayer ({self.interlayer_style})\n")
                for cmd in interlayer_cmds:
                    f.write(f"{cmd}\n")
                f.write("\n")

            if virtual_needs_lj:
                f.write("# Virtual atom (minimal LJ)\n")
                f.write(f"pair_coeff * {self.virtual_atom_type} lj/cut 1e-100 1e-100\n")

        logger.info("Wrote potential settings to %s", output_path)
