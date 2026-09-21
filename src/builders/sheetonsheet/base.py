"""Sheet-on-Sheet Simulation Builder.

This module orchestrates the setup of a friction simulation between two
2D material sheets. The model is an N-layer stack (minimum 3):
    - Layer 1: Fixed (bottom)
    - Layers 2 to N-1: Mobile with Langevin thermostat (friction interface)
    - Layer N: Rigid, driven by virtual atom (top)

Constraint modes control interlayer bonding and ghost interactions:
    - atom_bonds: Harmonic bonds between top 2 layers only,
        ghost LJ for non-adjacent layers.
    - com_spring: COM spring between top 2 layers, ghost LJ for non-adjacent
        layers, no constraints on bottom layers.
    - none: No bonds or springs, real LJ for all layer pairs.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Union

from ..simulation_base import SimulationBase
from ...core.config import SheetOnSheetSimulationConfig
from ...potentials.manager import PotentialManager
from ...potentials.utils import POTENTIALS_WITH_INTERNAL_LJ
from ..utils.lammps_data import atomic2molecular
from ...core.utils.tokens import format_numeric_token
from ...potentials.utils import normalize_potential_type
from ...data.models import EV_A_TO_NN, EV_A3_TO_GPA, NM_TO_EV_A2
from ..construction import components

logger = logging.getLogger(__name__)

#: Default minimum layer count for a dynamic sheet-on-sheet stack (fixed bottom,
#: thermostatted middle, driven top). Static PES bilayer scans override this.
MIN_LAYERS = 3


#: Recorded as ``cutoff_source`` when the run has no contact cutoff to record.
#: A static PES scan computes no ``coord/atom`` coordination at all, but
#: ``ProvenanceRecord.contact_cutoff`` is a required float, so it gets 0.0 and
#: this source says the number means nothing rather than inventing a plausible
#: one.
CUTOFF_NOT_APPLICABLE = 'not_applicable_static_scan'


def sheet_thermostat_scheme(settings) -> str:
    """What actually thermostats a dynamic sheet run, for the provenance record.

    The sheet templates thermostat the ``center`` group -- layers 2..N-1, the
    mobile middle of the stack -- and leave layer 1 fixed and layer N rigid.
    Either ``fix lang center langevin`` plus ``fix nve_center center nve``, or
    ``fix nvt_center center nvt`` (``sheetonsheet/slide.lmp``, and the same
    pair in the hetero and flake templates). AFM's equivalent string names its
    own arrangement (`langevin_y_tip_sub_nve_sheet`), so these must not be
    confused when both families' records are read together.
    """
    if settings.thermostat.type == 'langevin':
        return 'langevin_center_nve'
    return f'{settings.thermostat.type}_center'


def sheet_layer_type_map(builder) -> Dict[str, List[int]]:
    """``{'layer_1': [7, 8], ...}`` from a builder's atom-type layer groups.

    Empty for a family whose layers are not atom-type groups. The hetero and
    flake builders partition by z-band region (``compute_layer_zbands``), so
    there are no per-layer type ids to record and ``{}`` is the honest answer --
    the same thing hetero AFM writes, and why
    :class:`~src.data.provenance.ProvenanceRecord` allows the map to be empty.
    """
    groups = getattr(builder, 'groups', None)
    if not isinstance(groups, dict):
        return {}
    out: Dict[str, List[int]] = {}
    for name, types in groups.items():
        if not (isinstance(name, str) and name.startswith('layer_')):
            continue
        if not isinstance(types, str):
            continue
        ids = [int(tok) for tok in types.split() if tok.isdigit()]
        if ids:
            out[name] = ids
    return out


def write_sheet_provenance(builder, *, contact_cutoff: float,
                           cutoff_source: str, thermostat_scheme: str,
                           output_root: Optional[Path] = None) -> Path:
    """Write ``provenance.json`` for one sheet-family run.

    The sheet family's counterpart to
    :meth:`~src.builders.afm.base.AFMSimulation._write_provenance`, and a
    module-level function for the same reason :func:`sheet_obs_ctx` is one:
    :class:`SheetOnSheetSimulation`, ``HeteroSheetOnSheetSimulation`` and
    ``FlakeSlideSimulation`` are *siblings* under
    :class:`~src.builders.simulation_base.SimulationBase`, not a class and its
    subclasses, so there is no shared method to put this on.

    Until this existed, **no sheet-on-sheet, hetero, flake or PES-sheet run
    wrote a provenance record at all** while every AFM run did:
    ``ProvenanceRecord.seed_sheet_center`` ("sheet runs only") had no writer,
    ``SheetOnSheetSimulation._contact_cutoff`` set a ``_contact_cutoff_source``
    nothing read, and ``hpc enrich-provenance`` reported ``skipped`` for every
    sheet run it swept.

    Four fields differ in meaning from an AFM record, and the differences are
    the reason this is not simply AFM's method moved up:

    * **Load is a pressure.** A sheet run is driven at constant normal
      *pressure* (GPa), not a tip load (nN), so ``pressure_requested`` carries
      it and ``load_requested`` stays ``None``. Both are optional in the schema.
      A *list* of pressures records ``None``, not the first element: the list is
      an in-script loop, so one run covers every value and no single number is
      "the" requested pressure. AFM records a force list the same way.
    * **The seeds are different channels.** Only ``sheet_center`` is drawn -- it
      thermostats the mobile middle layers -- so ``seed_thermal_tip``,
      ``seed_thermal_sub`` and ``seed_velocity`` are ``None``. There is no tip
      and no separate velocity channel.
    * **There is no substrate realization.** ``substrate_realization_id``
      identifies the cached amorphous slab every AFM run shares; a sheet stack
      has none.
    * **``materials`` is still a placeholder here.** Schema 2.0 replaced the
      single ``pot_path``/``pot_type``/``pot_file_sha256`` triple with a
      ``materials`` list that can hold one entry per component of a hetero
      stack, but this function still writes it empty (``family``/``model`` are
      literals too) -- filling it in with the real per-material potentials is
      the next task's job, not this one's.

    Args:
        builder: the builder instance; its ``config``, ``_seeds``,
            ``output_dir`` and git identity are read.
        contact_cutoff: the cutoff the emitted script actually uses -- passed
            in rather than recomputed, so the record cannot disagree with the
            script. Hetero and flake pass ``sheet_obs_ctx``'s documented
            fallback because that is what their ``obs_ctx`` used.
        cutoff_source: how that number was arrived at, e.g.
            ``'uff_sigma_k2.0'``, ``'fallback_default'`` or
            :data:`CUTOFF_NOT_APPLICABLE`.
        thermostat_scheme: what actually thermostats the run, e.g.
            ``'langevin_center_nve'``.
        output_root: directory to write into; defaults to the builder's own
            run directory.

    Returns:
        The path written.
    """
    # Deferred for the same reason AFM defers it: `src.data.provenance` costs
    # ~15-23 ms (pydantic model construction) and only an actual build should
    # pay it -- `--help` and every non-build subcommand instantiate nothing.
    # pylint: disable-next=import-outside-toplevel
    from ...data.provenance import ProvenanceRecord, write_provenance
    import hashlib  # pylint: disable=import-outside-toplevel

    config = builder.config
    git_commit, git_dirty = builder._git_identity()  # pylint: disable=protected-access

    pressure = getattr(config.general, 'pressure', None)
    pressure_requested = (float(pressure)
                          if isinstance(pressure, (int, float)) else None)

    record = ProvenanceRecord(
        family='sheetonsheet',
        model='sheetonsheet',
        materials=[],
        seed_thermal_tip=None,
        seed_thermal_sub=None,
        seed_velocity=None,
        seed_sheet_center=builder._seeds.get('sheet_center'),  # pylint: disable=protected-access
        substrate_realization_id=None,
        damping_ratio=None,
        damp_ev=None,
        git_commit=git_commit,
        git_dirty=git_dirty,
        config_hash=hashlib.sha256(
            config.model_dump_json().encode()).hexdigest(),
        layer_type_map=sheet_layer_type_map(builder),
        contact_cutoff=contact_cutoff,
        cutoff_source=cutoff_source,
        thermostat_scheme=thermostat_scheme,
        run_role='label',
        load_requested=None,
        pressure_requested=pressure_requested,
    )
    return write_provenance(output_root or builder.output_dir, record)


def sheet_obs_ctx(settings, *, sim_type: str, n_layers: int,
                  results_file_pattern: str,
                  contact_cutoff: Optional[float] = None,
                  extra_filter=None) -> Dict[str, object]:
    """Build the observable context shared by every sheet-family builder.

    The three ``_build_obs_ctx`` methods in this package -- this module's,
    :meth:`~src.builders.sheetonsheet.hetero.HeteroSheetOnSheetSimulation._build_obs_ctx`
    and
    :meth:`~src.builders.sheetonsheet.flake_slide.FlakeSlideSimulation._build_obs_ctx`
    -- called :func:`~src.builders.utils.obs_ctx.build_obs_ctx` with the same
    seven arguments and differed only in ``sim_type``, ``contact_cutoff`` and
    ``extra_filter``, which pylint reported as three overlapping
    ``duplicate-code`` blocks. Those three ARE the per-family decisions, so
    they stay at the call sites where a reader can see them; everything the
    families agree on lives here.

    ``interface_layers`` is ``(n_layers // 2, n_layers // 2 + 1)`` -- the
    identical rule ``sheetonsheet/slide.lmp``, ``hetero/slide.lmp`` and
    ``flake/slide.lmp`` use for their own ``{% set il_bot %}``/
    ``{% set il_top %}``, so the sliding interface has ONE definition shared
    between Python and Jinja rather than two that can drift (see
    MIDDLE_INTERFACE in tests/builders/test_template_contract.py).

    Args:
        settings: the run's :class:`~src.core.config.GlobalSettings`; the
            thermostat type and the whole output block are read from it.
        sim_type: the observable registry's family key -- ``'sheetonsheet'``,
            ``'hetero'`` or ``'flake'``. Selects which channels exist at all.
        n_layers: layer count, used for the layer-resolved channels and the
            interface rule above.
        results_file_pattern: the window tier's output path pattern; the raw
            and peratom patterns are derived from it by ``build_obs_ctx``.
        contact_cutoff: ``coord/atom``'s cutoff for ``n_contact_il``. ``None``
            (the default) means the documented fallback -- correct for both
            hetero and flake, whose ``extra_filter``/registry entries strip
            ``n_contact_il`` before it can consume the value, and neither of
            which can inherit :meth:`SheetOnSheetSimulation._contact_cutoff`
            anyway (they extend ``SimulationBase``, and that method reads
            ``self.config.sheet``, which they do not have). The homogeneous
            builder passes its real derivation.
        extra_filter: an optional predicate dropping channels the family's
            template cannot support. ``None`` keeps every registered channel.
    """
    # Both deferred, for two different reasons, stated once here rather than
    # repeated at each call site:
    #
    # `obs_ctx`: import weight. `src.cli` imports `src.run.simulate`, which
    # imports this module at module scope, so every CLI invocation --
    # `--help` included -- pays this module's imports. `build_obs_ctx` pulls
    # in `src.core.observables` (~11 ms cumulative after `import src.cli`,
    # measured with `python -X importtime`), which nothing else on that path
    # loads, so only an actual build should pay it.
    #
    # `afm.base`: keeping the sheet family from naming the AFM builder module
    # at import time for the one constant it borrows. NOT a cycle (nothing
    # under `src/builders/afm/` imports `sheetonsheet`) and NOT import weight
    # (~0.5 ms once `src.builders.sheetonsheet` is loaded, measured the same
    # way) -- neither is claimed. It is also read only on the default path,
    # so the homogeneous builder never triggers it.
    # pylint: disable=import-outside-toplevel
    from ..utils.obs_ctx import build_obs_ctx
    if contact_cutoff is None:
        from ..afm.base import _CONTACT_CUTOFF_FALLBACK
        contact_cutoff = _CONTACT_CUTOFF_FALLBACK
    # pylint: enable=import-outside-toplevel

    out = settings.output
    return build_obs_ctx(
        sim_type=sim_type,
        n_layers=n_layers,
        results_file_pattern=results_file_pattern,
        use_langevin=(settings.thermostat.type == 'langevin'),
        enabled_tiers=dict(out.channels),
        contact_cutoff=contact_cutoff,
        results_freq=out.results_frequency,
        optional_channels=out.optional_channels,
        interface_layers=(n_layers // 2, n_layers // 2 + 1),
        extra_filter=extra_filter,
    )


class SheetOnSheetSimulation(SimulationBase):
    """Builder for homogeneous sheet-on-sheet friction simulations.

    Builds one N-layer stack of a SINGLE 2D material (N >= ``MIN_LAYERS``)
    and slides its top layer over the rest:

        - Layer 1: fixed bottom layer
        - Layers 2 to N-1: mobile, Langevin-thermostatted
        - Layer N: rigid top layer, driven (virtual atom by default)

    :meth:`build` creates the run-directory tree, seeds ``provenance/``,
    writes the stacked structure to ``build/`` via
    :func:`src.builders.construction.components.build_sheet`, writes the pair
    coefficients to ``lammps/system.in.settings`` through
    :class:`~src.potentials.manager.PotentialManager`, renders
    ``sheetonsheet/slide.lmp`` into ``lammps/slide.in`` (or into one script
    per outer-loop value, see :meth:`write_inputs`), and finally emits the
    HPC job scripts.

    Differences from :class:`~src.builders.afm.AFMSimulation`, which shares
    :class:`~src.builders.simulation_base.SimulationBase`:

    * There is no tip and no substrate. Both sliding bodies are layers of the
      same material, so the contact cutoff is derived from the INTERLAYER LJ
      sigma rather than a tip-sheet one (:meth:`_contact_cutoff`), and there
      is no ``self.config.tip`` to read.
    * Layers are selected by ATOM TYPE (``layer_N`` groups written by the
      potential manager), not by z-band region -- that is why the
      heterostructure variant needs its own templates.
    * Loading is a normal PRESSURE on the top layer, so the run phase splits
      per pressure / scan speed / scan angle (``SPLIT_OUTER_LOOPS``) where
      AFM splits per normal load.
    * Interlayer bonding is configurable (``constraint_mode``:
      ``atom_bonds`` / ``com_spring`` / ``none``), which has no AFM analogue.
    """

    #: Minimum number of layers this builder accepts. Subclasses (e.g. the
    #: static PES bilayer scan) may lower it.
    MIN_LAYERS: int = MIN_LAYERS

    #: The sheet-on-sheet run phase splits per pressure, per scan speed and per
    #: scan angle (see write_inputs); there is no normal load to split by.
    SPLIT_OUTER_LOOPS = frozenset({'pressure', 'scan_speed', 'scan_angle'})

    def __init__(self, config: SheetOnSheetSimulationConfig, output_dir: str,
                 config_path: Optional[str] = None):
        super().__init__(config, output_dir, config_path=config_path)
        self.config: SheetOnSheetSimulationConfig = config
        self.structure_paths: Dict[str, Path] = {}
        self.z_positions: Dict[str, float] = {}
        self.groups: Dict[str, str] = {}
        self.pm: Optional[PotentialManager] = None
        self.lat_c: Optional[float] = None
        self.sheet_dims: Optional[Dict] = None
        self.sheet_unit_cell: Dict[str, float] = {}

    @property
    def n_layers(self) -> int:
        """Number of layers derived from sheet configuration.

        Sheet-on-sheet currently supports a single explicit layer count
        (not a layer sweep list).
        """
        layers = self.config.sheet.layers
        if len(layers) != 1:
            raise ValueError(
                "Sheet-on-sheet currently requires exactly one value in 2D.layers "
                "(e.g., layers=[3])."
            )
        return int(layers[0])

    @staticmethod
    def _to_list(value: Optional[Union[float, List[float]]]) -> List[float]:
        """Normalize scalar/list sweep values to a list of floats."""
        if value is None:
            return []
        if isinstance(value, list):
            return [float(v) for v in value]
        return [float(value)]


    def build(self) -> None:
        """Constructs the N-layer sheet stack."""
        n_layers = self.n_layers
        if n_layers < self.MIN_LAYERS:
            raise ValueError(
                f"Sheet-on-sheet requires at least {self.MIN_LAYERS} layers, "
                f"got {n_layers}"
            )

        logger.info("Starting Sheet-vs-Sheet Build (%d-layer model)...", n_layers)
        self._create_directories()

        build_dir = self.output_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        self._init_provenance()

        constraint_mode = self.config.settings.simulation.constraint_mode
        pot_type_lower = normalize_potential_type(self.config.sheet.pot_type)

        if pot_type_lower in POTENTIALS_WITH_INTERNAL_LJ and constraint_mode != 'none':
            raise ValueError(
                f"Potential '{pot_type_lower}' has internal interlayer interactions "
                f"and does not support external LJ. "
                f"Set constraint_mode to 'none' in settings.yaml."
            )

        use_pair_bonding = constraint_mode == 'atom_bonds'

        logger.info("Building %d-layer sheet stack...", n_layers)
        stacking_type = getattr(self.config.sheet, 'stack_type', 'AB')
        sheet_path, dims, lat_c = components.build_sheet(
            self.config.sheet, build_dir,
            stack_if_multi=True, settings=self.config.settings,
            n_layers_override=n_layers, use_pair_bonding=use_pair_bonding,
            stacking_type=stacking_type,
            edge_mode=self.config.settings.geometry.finite_sheet_sheetonsheet_edge_mode,
            edge_width=self.config.settings.geometry.finite_sheet_edge_width,
            unit_cell_out=self.sheet_unit_cell,
        )
        self.structure_paths['sheet'] = sheet_path

        self.pm = self._generate_potentials()
        assert lat_c is not None
        for i in range(n_layers):
            self.z_positions[f'layer_{i + 1}'] = i * lat_c

        self.lat_c = lat_c
        self.sheet_dims = dims

        self.write_inputs()
        self._generate_hpc_scripts()
        logger.info("Build complete.")

    def _material_id(self) -> str:
        """Material token for HPC job names: this builder's single material.

        The sheet-family counterpart of
        :meth:`~src.builders.afm.AFMSimulation._material_id`, added so the PES
        scan's job name can be composed from one hook rather than hardcoding
        ``config.sheet.mat``: the hetero builder overrides it with the
        ``mat1+mat2`` pair token (two pairs sharing material 1 would otherwise
        collide on one job name).
        """
        return self.config.sheet.mat

    def _get_hpc_job_name(self) -> str:
        """Get sheet-on-sheet specific job name.

        Through :meth:`_material_id` rather than ``config.sheet.mat`` directly:
        two spellings of the same token one line apart is how they drift, and
        the hook is what the hetero builders override to key the name on the
        full pair. Inert here (this builder's ``_material_id`` IS
        ``config.sheet.mat``) — it is the consistency that matters.
        """
        return f"sheet_{self._material_id()}"

    def _init_provenance(self) -> None:
        """Initialize provenance folder and collect input files."""
        prov_dir = self.output_dir / 'provenance'
        prov_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Provenance folder initialized at: %s", prov_dir)

        self._add_component_files_to_provenance('sheet', self.config.sheet)

        logger.info("Initialized provenance folder: %s", prov_dir)

    def _generate_potentials(self) -> PotentialManager:
        """Configure potential file for the sheet-on-sheet simulation.

        Interlayer LJ handling depends on constraint_mode and potential type:
            - Potentials with internal LJ (ReaxFF, AIREBO, etc.): No explicit LJ.
            - atom_bonds / com_spring: Ghost LJ for non-adjacent layers.
            - none: Real LJ for all layer pairs.

        Returns:
            Configured PotentialManager instance.
        """
        n_layers = self.n_layers
        constraint_mode = self.config.settings.simulation.constraint_mode

        pm = PotentialManager(
            self.config.settings,
            potentials_dir=self.output_dir / "provenance" / "potentials",
            potentials_prefix=str(
                self.relative_run_dir / "provenance" / "potentials"
            )
        )
        pm.set_lj_overrides(self.config.lj_override)

        edge_mode = (
            self.config.settings.geometry.finite_sheet_sheetonsheet_edge_mode
        )
        sheet_edge_regions = [None, 'edge'] if edge_mode != 'none' else None
        pm.register_component(
            'sheet', self.config.sheet, n_layers=n_layers,
            regions=sheet_edge_regions,
        )

        self._register_drive_atom(pm)

        pm.add_self_interaction('sheet')

        has_internal_lj = not pm.is_sheet_lj(self.config.sheet.pot_type)
        if has_internal_lj:
            pass  # ReaxFF, AIREBO, etc. handle interlayer interactions internally
        elif constraint_mode in ('atom_bonds', 'com_spring'):
            pm.add_ghost_lj('sheet', max_real_distance=1)
        else:
            pm.add_interlayer_interaction('sheet')

        settings_path = self.output_dir / "lammps" / "system.in.settings"
        pm.write_file(settings_path)

        for layer in range(n_layers):
            layer_num = layer + 1
            self.groups[f'layer_{layer_num}'] = (
                pm.types.get_layer_group_string('sheet', layer)
            )

        self.groups['center'] = ' '.join(
            [self.groups[f'layer_{i}'] for i in range(2, n_layers)]
        )
        self.groups['all_types'] = pm.types.get_group_string('sheet')

        return pm

    def _register_drive_atom(self, pm: PotentialManager) -> None:
        """Register the virtual driving atom when the drive method needs one.

        A dynamic sliding run drives the top layer with a virtual atom. The
        static PES scan has no driving, so :class:`PESSheetSimulation` overrides
        this to a no-op (keeping the atom-type count aligned with the packed
        bilayer, which contains real atoms only).
        """
        if self.config.settings.simulation.drive_method == 'virtual_atom':
            pm.register_virtual_atom()

    def _build_layer_groups(self) -> Dict[str, str]:
        """Build a dict mapping 'layer_N_types' to group type strings."""
        layer_groups = {}
        for i in range(1, self.n_layers + 1):
            layer_groups[f'layer_{i}_types'] = self.groups[f'layer_{i}']
        return layer_groups

    def _build_obs_ctx(self, n_layers: int, results_file_pattern: str,
                       has_internal_lj: bool) -> Dict[str, object]:
        """Observable context for the sheet family.

        `extra_filter` drops the interlayer channels for a many-body
        potential: the template's fallback branch defines xfrict as v_sx and
        never defines v_e_int_il, so the registry cannot decide this on its
        own -- only the builder knows the potential type.

        `interface_layers` is `(n_layers // 2, n_layers // 2 + 1)` -- the
        identical rule `sheetonsheet/slide.lmp` uses for its own
        `{% set il_bot %}`/`{% set il_top %}`, so the sliding interface has
        one definition shared between Python and Jinja rather than two that
        can drift (see MIDDLE_INTERFACE in tests/builders/test_template_contract.py).
        """
        interlayer = {'E_int_interlayer', 'n_contact_il'}
        return sheet_obs_ctx(
            self.config.settings,
            sim_type='sheetonsheet',
            n_layers=n_layers,
            results_file_pattern=results_file_pattern,
            contact_cutoff=self._contact_cutoff(),
            extra_filter=(
                None if not has_internal_lj
                else lambda ob: ob.name not in interlayer),
        )

    def _contact_cutoff(self) -> float:
        """Per-material contact cutoff from the interlayer LJ sigma.

        K_CONTACT * max(sigma_ij) over the sheet's own element pairs -- the
        same rule as AFM's tip-sheet cutoff (`AFMSimulation._contact_cutoff`,
        parent spec §5 F3), but reading the INTERLAYER sigma (the sheet
        against itself, mirroring `PotentialManager.add_interlayer_interaction`'s
        element product) rather than a tip-sheet one, since sheet builders
        have no `self.config.tip`. A fixed absolute cutoff would conflate
        "more atoms in contact" with "larger atoms". Capped below the pair
        cutoff -- LAMMPS errors if coord/atom's cutoff exceeds it. Falls back
        to a documented default when the sheet elements cannot be read;
        provenance records which route was taken.

        Where `is_sheet_lj()` is false there is no interlayer lj/cut sigma to
        read, so `n_contact_il` is unavailable for exactly the materials that
        also lose `E_int_interlayer` -- `_build_obs_ctx`'s `extra_filter`
        already drops both.
        """
        # Deferred as one block so the sheet family does not name the AFM
        # builder module at import time: the two constants below are the only
        # thing it borrows from `afm.base`, and they are read once per build.
        # Neither of the two usual reasons applies, and neither is claimed
        # here -- nothing under `src/builders/afm/` imports `sheetonsheet`, so
        # this is not breaking a cycle, and importing `afm.base` once
        # `src.builders.sheetonsheet` is loaded costs ~0.5 ms (measured with
        # `python -X importtime`), so it is not import weight either. The
        # other three ride along in the same block; `potentials.utils` is
        # already a module-scope import of this file.
        # pylint: disable=import-outside-toplevel
        from itertools import product
        from ...core.utils.structures import cifread
        from ...potentials.utils import lj_params
        from ..afm.base import K_CONTACT, _CONTACT_CUTOFF_FALLBACK
        # pylint: enable=import-outside-toplevel

        lj_cut = getattr(self.config.settings.potential, 'lj_cutoff', 11.0)
        try:
            sheet_els = cifread(self.config.sheet.cif_path)['elements']
            sigma_max = max(lj_params(a, b)[1]
                            for a, b in product(sheet_els, sheet_els))
            # Not declared in __init__, mirroring AFMSimulation._contact_cutoff
            # (see the note there): assigning only on the route actually taken
            # is what lets a reader tell "never computed" from a real value
            # via getattr(self, '_contact_cutoff_source', 'unknown'), which an
            # __init__-assigned None/'' would collapse. write_inputs reads it
            # exactly that way when it writes the provenance record.
            # pylint: disable-next=attribute-defined-outside-init
            self._contact_cutoff_source = f'uff_sigma_k{K_CONTACT}'
            cut = K_CONTACT * sigma_max
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Contact cutoff fallback: could not derive UFF sigma for "
                "sheet '%s'; using %.1f A.",
                self.config.sheet.mat, _CONTACT_CUTOFF_FALLBACK)
            # See the getattr-distinguishes-absence note above. This is the
            # same string the hetero and flake builders pass literally: they
            # never reach this method, but their obs_ctx takes sheet_obs_ctx's
            # documented fallback, which is the same number for the same reason.
            # pylint: disable-next=attribute-defined-outside-init
            self._contact_cutoff_source = 'fallback_default'
            cut = _CONTACT_CUTOFF_FALLBACK
        return min(cut, 0.95 * lj_cut)

    def write_inputs(self) -> None:
        """Generate LAMMPS scripts."""
        logger.info("Writing LAMMPS inputs...")

        assert self.pm is not None
        assert self.sheet_dims is not None
        assert self.lat_c is not None

        n_layers = self.n_layers
        total_types = len(self.pm.types)
        constraint_mode = self.config.settings.simulation.constraint_mode
        pot_type = normalize_potential_type(self.config.sheet.pot_type)

        sim = self.config.settings.simulation
        out = self.config.settings.output

        rel_run_dir_str = str(self.relative_run_dir)

        if constraint_mode == 'atom_bonds':
            n_bond_types = 1
        else:
            n_bond_types = 0

        # Auto-detect atom_style based on potential and constraint mode
        if pot_type in ('reaxff', 'reax/c'):
            atom_style = 'charge'
        elif constraint_mode == 'atom_bonds':
            atom_style = 'molecular'
        else:
            atom_style = 'atomic'

        if atom_style == 'molecular':
            atomic2molecular(self.structure_paths['sheet'])

        has_internal_lj = not self.pm.is_sheet_lj(pot_type)
        results_file_pattern = (
            f"{rel_run_dir_str}/results/"
            f"friction_p${{pressure}}_a${{a}}_s${{speed}}"
        )

        base_context = {
            'seed_sheet_center': self._run_seed('sheet_center'),
            'data_dir': f"{rel_run_dir_str}/data",   # final-state retention
            'temp': self.config.general.temp,
            'scan_angle_config': self.config.general.scan_angle,
            'scan_angle_force': self.config.general.scan_angle_force,
            'xlo': self.sheet_dims.get('xlo', 0.0),
            'xhi': self.sheet_dims.get('xhi', 100.0),
            'ylo': self.sheet_dims.get('ylo', 0.0),
            'yhi': self.sheet_dims.get('yhi', 100.0),
            'zhi': self.sheet_dims.get('zhi', 15.0),
            'data_file': (
                f"{rel_run_dir_str}/build/"
                f"{self.structure_paths['sheet'].name}"
            ),
            'potential_file': (
                f"{rel_run_dir_str}/lammps/system.in.settings"
            ),
            'num_atom_types': total_types,
            'ngroups': total_types,
            'n_layers': n_layers,
            'constraint_mode': constraint_mode,
            'n_bond_types': n_bond_types,
            **self._build_layer_groups(),
            'center_types': self.groups['center'],
            'lat_c': self.lat_c,
            'sheet_dims': self.sheet_dims,
            'bond_spring_ev': (
                (self.config.general.bond_spring or 80.0) / NM_TO_EV_A2
            ),
            'bond_min': self.lat_c - 0.15,
            'bond_max': self.lat_c + 0.15,
            'driving_spring_ev': (
                (self.config.general.driving_spring or 50.0) / NM_TO_EV_A2
            ),
            'timestep': sim.timestep,
            'thermo': sim.thermo,
            'neighbor_list': sim.neighbor_list,
            'neigh_modify_command': sim.neigh_modify_command,
            'run_steps': sim.slide_run_steps,
            'min_style': sim.min_style,
            'minimization_command': sim.minimization_command,
            'results_freq': out.results_frequency,
            'dump_freq': out.dump_frequency.get('slide', 1000),
            'dump_enabled': out.dump.get('slide', False),
            'results_file_pattern': results_file_pattern,
            'obs_ctx': self._build_obs_ctx(
                n_layers, results_file_pattern, has_internal_lj),
            'results_file_raw': (
                f"{rel_run_dir_str}/results/"
                f"raw_p${{pressure}}_a${{a}}_s${{speed}}"
            ),
            'results_file_peratom': (
                f"{rel_run_dir_str}/results/"
                f"peratom_p${{pressure}}_a${{a}}_s${{speed}}"
            ),
            'hold_steps': sim.sheet_hold_steps,
            'dump_file_pattern': (
                f"{rel_run_dir_str}/visuals/"
                f"slide_p${{pressure}}_a${{a}}_s${{speed}}.lammpstrj"
            ),
            'drive_method': sim.drive_method,
            'thermostat_type': self.config.settings.thermostat.type,
            'atom_style': atom_style,
            'pot_type': pot_type,
            'has_internal_lj': has_internal_lj,
            'ev_a_to_nn': EV_A_TO_NN,
            'ev_a3_to_gpa': EV_A3_TO_GPA,
        }

        # Before the branching below, not after: write_inputs has four exits
        # (per-angle split, single script, per-pressure split, per-speed split)
        # and the record belongs to the build, not to any one of them.
        write_sheet_provenance(
            self,
            contact_cutoff=base_context['obs_ctx']['contact_cutoff'],
            # Set by _contact_cutoff, which _build_obs_ctx called just above.
            # 'unknown' would mean it never ran, which cannot happen here.
            cutoff_source=getattr(self, '_contact_cutoff_source', 'unknown'),
            thermostat_scheme=sheet_thermostat_scheme(self.config.settings),
        )

        outer_loop = getattr(self.config.general, 'outer_loop', None)
        # A builder splits only by what it declares in SPLIT_OUTER_LOOPS;
        # anything else falls through to the single-script path below. The
        # job-count prediction reads the same declaration, so what it predicts
        # is what is written here.
        if outer_loop not in self.SPLIT_OUTER_LOOPS:
            outer_loop = None

        # outer_loop='scan_angle' emits one slide_a<angle>.in per angle,
        # mirroring AFMSimulation's split (src/builders/afm.py). A
        # single-element (or empty/scalar) angle list falls through to the
        # single-script path below, same as outer_loop unset, so the
        # combined-HPC path that references the literal name slide.in still
        # applies.
        angles = self._to_list(self.config.general.scan_angle)
        if outer_loop == 'scan_angle' and len(angles) > 1:
            # src.run.expand.outer_loop_conflict rejects this first (before the
            # build); the raise below is the backstop for direct builds.
            if self.config.general.scan_angle_force is not None:
                raise ValueError(
                    "outer_loop='scan_angle' cannot be combined with "
                    "scan_angle_force: splitting into one script per angle "
                    "applies each angle to every pressure value, silently "
                    "dropping the pressure-selective filter. Remove "
                    "scan_angle_force, or drop outer_loop to keep the "
                    "in-script angle loop (which honors it)."
                )
            for angle in angles:
                context = dict(base_context)
                context['scan_angle_config'] = float(angle)
                context['pressures'] = self.config.general.pressure
                context['scan_speed_config'] = self.config.general.scan_speed
                script_name = f"slide_a{format_numeric_token(float(angle))}.in"
                script = self.render_template("sheetonsheet/slide.lmp", context)
                self.write_file(f"lammps/{script_name}", script)
            logger.info("Split slide phase into %d per-angle scripts.", len(angles))
            logger.info("Inputs written to %s/lammps/", self.output_dir)
            return

        if outer_loop not in ('pressure', 'scan_speed'):
            context = dict(base_context)
            context['pressures'] = self.config.general.pressure
            context['scan_speed_config'] = self.config.general.scan_speed
            script = self.render_template("sheetonsheet/slide.lmp", context)
            self.write_file("lammps/slide.in", script)
            logger.info("Using single-script mode (slide.in).")
            logger.info("Inputs written to %s/lammps/", self.output_dir)
            return

        speeds = self._to_list(self.config.general.scan_speed)
        pressures = self._to_list(self.config.general.pressure)

        if outer_loop == 'pressure':
            outer_values = pressures or [0.0]
            inner_speeds: Union[float, List[float]] = (
                speeds if len(speeds) > 1
                else (speeds[0] if speeds else 0.0)
            )
            for pressure in outer_values:
                context = dict(base_context)
                context['pressures'] = pressure
                context['scan_speed_config'] = inner_speeds
                script_name = (
                    f"slide_p{format_numeric_token(float(pressure))}gpa.in"
                )
                script = self.render_template(
                    "sheetonsheet/slide.lmp", context
                )
                self.write_file(f"lammps/{script_name}", script)
        else:
            outer_values = speeds or [0.0]
            inner_pressures: Union[float, List[float]] = (
                pressures if len(pressures) > 1
                else (pressures[0] if pressures else 0.0)
            )
            for speed in outer_values:
                context = dict(base_context)
                context['pressures'] = inner_pressures
                context['scan_speed_config'] = speed
                script_name = (
                    f"slide_{format_numeric_token(float(speed))}ms.in"
                )
                script = self.render_template(
                    "sheetonsheet/slide.lmp", context
                )
                self.write_file(f"lammps/{script_name}", script)

        logger.info("Inputs written to %s/lammps/", self.output_dir)
