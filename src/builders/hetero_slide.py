"""D1 hetero sheet-on-sheet slide assembly (consumes hetero.py's structure builder)."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from jinja2 import Environment

from ..core.config import SheetOnSheetSimulationConfig
from .base import SimulationBase
from ..data.models import EV_A_TO_NN, EV_A3_TO_GPA, NM_TO_EV_A2
from ..interfaces.jinja import PackageLoader
from ..interfaces.ase import read_lammps_data
from .hetero import (
    _detect_atom_style,
    build_hetero_structure,
    register_hetero_potentials,
)

logger = logging.getLogger(__name__)

PAD = 1.0  # Å pad above/below each layer's atom z-range

# Same Jinja environment construction as SimulationBase.__init__
# (src/builders/base.py:80-84): package-resource loader, trimmed blocks.
_ENV = Environment(
    loader=PackageLoader('src.templates'),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_hetero_slide(context: Dict) -> str:
    """Render ``hetero/slide.lmp`` with the given context.

    Thin module-level helper mirroring how builders render templates via
    ``SimulationBase.render_template``, without needing a full builder
    instance.
    """
    return _ENV.get_template("hetero/slide.lmp").render(context)


def compute_layer_zbands(data_path, layers) -> List[Dict]:
    """Compute per-layer z-bands from each layer's KNOWN placement.

    Each layer in ``layers`` (a :class:`~src.builders.hetero.HeteroStackLayer`)
    records ``.source`` (the per-material supercell data file it was read
    from) and ``.z`` (the z-shift applied via
    ``read_data ... shift 0 0 z``, see templates/hetero/stack_hetero.lmp).
    So the placed atoms of a layer occupy exactly
    ``[native_zmin + z, native_zmax + z]``, where ``native_z`` comes from
    reading ``.source``. This is exact and independent of gap heuristics, so
    (unlike clustering on the (n-1) largest z-gaps) it cannot be fooled by a
    puckered layer whose internal sub-plane gap exceeds an inter-layer gap
    (e.g. black phosphorus, GeS).

    Multiple layers of the SAME material share one ``.source`` file but have
    DIFFERENT ``.z``, so their bands are still distinct.

    Returns one ``{'idx', 'zlo', 'zhi'}`` per layer (idx 1..n bottom->top).

    Raises:
        ValueError: If ``data_path``'s actual atom z-coordinates disagree with
            the bands derived from known placement (i.e. some atom falls in
            zero or more than one band). This is a self-check that makes a
            silent mis-grouping impossible -- it fails loud instead.
    """
    native_zrange: Dict[str, tuple] = {}
    placed = []
    for layer in layers:
        key = str(layer.source)
        if key not in native_zrange:
            zz = read_lammps_data(key).positions[:, 2]
            native_zrange[key] = (float(zz.min()), float(zz.max()))
        zmin, zmax = native_zrange[key]
        zlo, zhi = zmin + layer.z - PAD, zmax + layer.z + PAD
        center = 0.5 * (zmin + zmax) + layer.z
        placed.append((center, zlo, zhi))

    placed.sort(key=lambda t: t[0])
    bands = [{"idx": i + 1, "zlo": zlo, "zhi": zhi} for i, (_, zlo, zhi) in enumerate(placed)]

    # Self-check (fail loud): every atom in the actual assembled data file
    # must fall in exactly one band. Catches any placement/write discrepancy
    # instead of silently mis-grouping.
    z = read_lammps_data(data_path).positions[:, 2]
    for zi in z:
        hits = [b for b in bands if b["zlo"] <= zi <= b["zhi"]]
        if len(hits) != 1:
            raise ValueError(
                f"compute_layer_zbands: atom at z={zi:.4f} falls in {len(hits)} "
                f"bands (expected exactly 1); bands={bands}. The assembled "
                f"{data_path} disagrees with the known layer placement "
                f"(source native z-range + z-shift) -- check read_data shift "
                f"semantics or PAD={PAD}."
            )
    return bands


class HeteroSheetOnSheetSimulation(SimulationBase):
    """Builder for a periodic heterostructure sheet-on-sheet slide.

    Turns a two-material :class:`SheetOnSheetSimulationConfig` into a runnable
    periodic hetero slide by combining the Phase-B structure builder
    (:func:`~src.builders.hetero.build_hetero_structure`, which assembles
    ``hetero.data`` + ``lammps/system.in.settings`` and stages the potentials)
    with the D1 slide template (``templates/hetero/slide.lmp``).

    The assembled N-layer stack (N = sum of each material's layer count, >= 4)
    is a fixed bottom / thermostatted middle / driven top slide, exactly like the
    homogeneous :class:`~src.builders.sheetonsheet.SheetOnSheetSimulation`, but
    the layer groups are defined by z-slab region (hetero layers share atom
    types, so type-based groups are impossible) and the interlayer friction is
    read from the bottom-layer reaction force (many-body ``sw`` hybrid, no
    ``compute group/group``).

    Path convention: the SAME as every other builder in the package -- every
    path in the render context, and the ``provenance/potentials/...`` prefix in
    the written ``system.in.settings``, is prefixed with
    :attr:`~src.builders.base.SimulationBase.relative_run_dir`, i.e. it
    is relative to the SIMULATION ROOT. That is the directory the generated HPC
    job scripts run LAMMPS from (``cd $PBS_O_WORKDIR``, or the scratch mirror
    of it), so a hetero run's job scripts open the same files a local run does.
    The stack is still assembled directly into ``output_dir``; only how the
    scripts *name* those files is root-relative. When the builder is used
    standalone (no ``set_base_output_dir``), ``relative_run_dir`` is the
    ``output_dir`` argument itself, so an absolute ``output_dir`` yields
    absolute paths that resolve from any working directory.
    """

    #: A dynamic hetero slide needs a fixed bottom, a thermostatted middle and a
    #: driven top across TWO materials -> at least 2 + 2 layers.
    MIN_LAYERS: int = 4

    #: This builder writes a single lammps/slide.in and never reads outer_loop
    #: (pressure/scan_speed/scan_angle all stay in-script), so a config that
    #: sets outer_loop produces one job here, not one per value. Explicit (the
    #: base default is the same) because a twist or a second material silently
    #: reroutes a sheet-on-sheet config here.
    SPLIT_OUTER_LOOPS = frozenset()

    def __init__(self, config: SheetOnSheetSimulationConfig, output_dir: str,
                 config_path: Optional[str] = None):
        super().__init__(config, output_dir, config_path=config_path)
        self.config: SheetOnSheetSimulationConfig = config
        # Assembled-stack state, populated by build() and consumed by
        # write_run_scripts() (the run-phase seam a subclass overrides to write
        # something other than the dynamic slide -- see
        # :class:`~src.builders.pes_scan.HeteroPESSheetSimulation`).
        self._stack = None
        self._layer_bands: List[Dict] = []
        self._hetero_box: Dict[str, float] = {}
        #: (lx, ly) of the assembled stack's in-plane box. The hetero
        #: sheet-on-sheet stack is NOT replicated in-plane, so this box IS the
        #: matched interface's coincidence cell -- the cell both materials'
        #: lattices were strained onto, and the only in-plane periodicity the
        #: interface actually has. Exposed under the same name
        #: ``HeteroAFMSimulation._build_hetero_sheet`` uses (where the box IS
        #: replicated, so the two are distinct there) for the static PES scan,
        #: whose scan period must be this cell rather than either
        #: constituent's own unit cell.
        self._hetero_coincidence_cell: Optional[tuple] = None
        #: xy tilt of the assembled box; a triclinic cell cannot be scanned by
        #: an axis-aligned (dx, dy) grid, so the PES variant refuses it.
        self._hetero_cell_xy_tilt: float = 0.0
        self._hetero_atom_style: str = "atomic"
        self._hetero_lat_c: Optional[float] = None
        self._hetero_num_atom_types: int = 0
        self._hetero_virtual_type_reserved: bool = False

    def _material_id(self) -> str:
        """Every configured material, numeric ``[2D-N]`` order, joined with ``+``.

        The hetero override of
        :meth:`~src.builders.sheetonsheet.SheetOnSheetSimulation._material_id`
        (same convention as :func:`src.run.naming.material_token` and
        :meth:`src.builders.hetero_afm.HeteroAFMSimulation._material_id`), so
        every job name built from it -- this class's ``friction_<pair>`` and
        the PES variant's ``pes_sheet_<pair>`` -- keys on the full pair rather
        than material 1 alone.
        """
        return '+'.join(sheet.mat for sheet in self.config.sheets)

    def _get_hpc_job_name(self) -> str:
        """Job name keyed by every configured material, not just sheet 1.

        This class doesn't subclass :class:`~src.builders.sheetonsheet.SheetOnSheetSimulation`
        (it subclasses :class:`SimulationBase` directly), so without this
        override it fell back to :meth:`SimulationBase._get_hpc_job_name`'s
        generic ``friction_{sheet 1's material}`` default — the same
        sheet-1-alone collision :class:`~src.builders.hetero_afm.HeteroAFMSimulation`
        had: two hetero pairs sharing material 1 (or a twist-only single
        material, whose token is just that one material) produced identical
        HPC job identities despite building into distinct output directories
        (already pair-token-safe via Task 1's ``output_dir_for``). Same
        numeric-``[2D-N]``-order ``+``-joined convention as
        :func:`src.run.naming.material_token` /
        :func:`~src.builders.hetero_afm._pair_material_token`; kept as a
        private inline join (in :meth:`_material_id`) rather than importing
        from ``hetero_afm`` since the two builder families are otherwise
        independent.
        """
        return f"friction_{self._material_id()}"

    @property
    def n_layers(self) -> int:
        """Total layer count = sum of each material's single layer count.

        Each material's ``layers`` is a single-element list (e.g. ``[2]``); a
        2+2 stack is 4 layers. Mirrors
        :attr:`SheetOnSheetSimulation.n_layers`' single-count requirement.
        """
        for sheet in self.config.sheets:
            if len(sheet.layers) != 1:
                raise ValueError(
                    "Hetero sheet-on-sheet currently requires exactly one value "
                    "in each material's 2D.layers (e.g., layers=[2])."
                )
        return sum(int(sheet.layers[0]) for sheet in self.config.sheets)

    def _init_provenance(self) -> None:
        """Create the provenance folder and record each material's input files.

        :class:`SimulationBase` has no ``_init_provenance`` (only
        :class:`SheetOnSheetSimulation` does, and that one is single-material),
        so the hetero builder provides its own: one provenance component per
        stacked material. ``build_hetero_structure`` separately stages the real
        potential files under ``provenance/potentials/`` for the run itself.
        """
        prov_dir = self.output_dir / "provenance"
        prov_dir.mkdir(parents=True, exist_ok=True)
        for i, sheet in enumerate(self.config.sheets):
            self._add_component_files_to_provenance(f"sheet_{i + 1}", sheet)

    def build(self) -> None:
        """Assemble the hetero stack and write the run phase (``lammps/slide.in``).

        The run phase itself is delegated to :meth:`write_run_scripts` (the
        seam a subclass overrides to emit something other than the dynamic
        slide); everything here is stack assembly shared by both.
        """
        n = self.n_layers
        if n < self.MIN_LAYERS:
            raise ValueError(
                f"Hetero sheet-on-sheet requires at least {self.MIN_LAYERS} "
                f"layers, got {n}"
            )

        logger.info("Building %d-layer heterostructure sheet-on-sheet slide...", n)
        self._create_directories()
        self._init_provenance()

        # Build the stack INTO the run dir (output_dir/hetero.data,
        # output_dir/lammps/system.in.settings, output_dir/provenance/), but
        # have the settings NAME the staged potentials sim-root-relatively --
        # that is the directory the HPC job scripts run LAMMPS from.
        stack = build_hetero_structure(
            self.config, self.config.settings, workdir=self.output_dir,
            potentials_prefix=self._potentials_prefix(),
        )

        bands = compute_layer_zbands(stack.data_path, stack.layers)

        # Box from the assembled hetero.data (template adds the z-vacuum itself
        # via change_box, so zhi is just the topmost atom z).
        at = read_lammps_data(stack.data_path)
        cell = at.cell
        xlo, xhi = 0.0, float(cell[0, 0])
        ylo, yhi = 0.0, float(cell[1, 1])
        zhi = float(at.positions[:, 2].max())

        settings = self.config.settings

        atom_style = _detect_atom_style(stack.data_path)  # 'atomic' for sw
        lat_c = stack.interface_spacing or settings.geometry.lat_c_default

        # ------------------------------------------------------------------ #
        # virtual_atom drive: reserve a DEDICATED non-interacting virtual type.
        #
        # The reused drive block does ``create_atoms {{num_atom_types}}`` +
        # ``group virtual type {{num_atom_types}}`` + ``fix move``. If
        # ``num_atom_types`` were the highest REAL type (``stack.total_types``),
        # that group would capture the ENTIRE top-type sublattice -- spread
        # across the driven top AND the thermostatted center layer -- and
        # rigidly drag those REAL atoms (wrong friction, silent). So, exactly
        # like the homogeneous path
        # (``SheetOnSheetSimulation._register_drive_atom`` ->
        # ``PotentialManager.register_virtual_atom``), add one extra
        # non-interacting type and make the box reserve room for it.
        reserve_virtual_type = self._reserve_drive_virtual_type()
        num_atom_types = stack.total_types
        if reserve_virtual_type:
            # Re-register the (geometry-independent, deterministic) hetero
            # potential spine, this time reserving a DEDICATED non-interacting
            # virtual type. register_hetero_potentials adds it BEFORE the
            # many-body self-interactions -- the same order the homogeneous path
            # uses -- so every ``pair_coeff * * sw ...`` element list is sized to
            # the full box type count (real types + trailing NULL for the virtual
            # type) the first time, and write_file emits the virtual type's
            # ``mass N 1.0`` + zero-LJ ``pair_coeff * N lj/cut 1e-100``. This
            # re-registration OVERWRITES lammps/system.in.settings (which
            # build_hetero_structure wrote with real types only); hetero.data
            # (real types unchanged) stays consistent with the new settings.
            hp = register_hetero_potentials(
                self.config.sheets,
                self.config.settings,
                self.output_dir,
                reserve_virtual_type=True,
                potentials_prefix=self._potentials_prefix(),
            )
            # Fail loud on any real-type-layout divergence: never ship an
            # inconsistent settings/data pairing. type_ids covers the REAL
            # materials only, so their total must equal the assembled stack's
            # real-type count; pm.types additionally holds the virtual type.
            real_type_count = sum(len(ids) for ids in hp.type_ids)
            if real_type_count != stack.total_types:
                raise RuntimeError(
                    "Re-registered hetero potentials do not match the assembled "
                    f"stack: {real_type_count} real types vs "
                    f"stack.total_types={stack.total_types}. Refusing to write an "
                    "inconsistent settings/hetero.data pairing for the "
                    "virtual_atom drive."
                )
            if len(hp.pm.types) != stack.total_types + 1:
                raise RuntimeError(
                    "Re-registered hetero potentials must hold exactly one extra "
                    f"(virtual) type: len(pm.types)={len(hp.pm.types)} vs "
                    f"stack.total_types + 1={stack.total_types + 1}."
                )
            expected_settings = self.output_dir / "lammps" / "system.in.settings"
            if Path(hp.settings_path).resolve() != expected_settings.resolve():
                raise RuntimeError(
                    f"Re-registered settings path {hp.settings_path} != the "
                    f"settings the slide includes ({expected_settings})."
                )
            # register_hetero_potentials already wrote the settings (with the
            # virtual type) to hp.settings_path -- no second write needed.
            num_atom_types = len(hp.pm.types)  # == stack.total_types + 1

        # Hand the assembled-stack state to the run-phase seam. Everything
        # below this point that a subclass might want to replace lives in
        # write_run_scripts(); everything above (the stack assembly, the layer
        # bands, the type/virtual-atom spine) is shared.
        self._stack = stack
        self._layer_bands = bands
        self._hetero_box = {"xlo": xlo, "xhi": xhi, "ylo": ylo, "yhi": yhi, "zhi": zhi}
        self._hetero_coincidence_cell = (xhi - xlo, yhi - ylo)
        self._hetero_cell_xy_tilt = float(cell[1, 0])
        self._hetero_atom_style = atom_style
        self._hetero_lat_c = lat_c
        self._hetero_num_atom_types = num_atom_types
        self._hetero_virtual_type_reserved = reserve_virtual_type

        self.write_run_scripts()

        # D3: record the achieved twist angle + moiré period into the run
        # manifest (achieved != target for discrete commensurate angles; always
        # log the achieved value). Inert for the non-twist hetero slide
        # (twist_angle_achieved is None), so the aligned path is unaffected.
        if stack.twist_angle_achieved is not None:
            self._record_twist_manifest(stack, self.config.general.twist_angle)

        # The homogeneous HPC scaffolding, used unmodified: this builder renders
        # sim-root-relative paths (see the class docstring), which is the
        # convention that scaffolding — and the job scripts it writes — assume.
        # Called bare, like SheetOnSheetSimulation.build and AFMSimulation.build:
        # a failure here (e.g. settings.hpc.modules unset) is a real, reportable
        # misconfiguration, and swallowing it produced a tree with no job scripts
        # and nothing but a log line to say why.
        self._generate_hpc_scripts()

    def _reserve_drive_virtual_type(self) -> bool:
        """Whether the run phase needs a dedicated non-interacting drive atom.

        Seam (not inlined in :meth:`build`) because it is a property of the RUN
        PHASE, not of the stack: a static PES scan has no drive at all, so
        :class:`~src.builders.pes_scan.HeteroPESSheetSimulation` returns False
        and the extra virtual type is never registered — mirroring
        :meth:`PESSheetSimulation._register_drive_atom`'s no-op on the
        homogeneous side, which likewise keeps the type count aligned with the
        real atoms in the data file.
        """
        return self.config.settings.simulation.drive_method == "virtual_atom"

    def write_run_scripts(self) -> None:
        """Render and write the run phase (``lammps/slide.in``): the dynamic slide.

        The run-script seam of :meth:`build`. Everything it needs was placed on
        ``self`` by ``build`` (``_stack``, ``_layer_bands``, ``_hetero_box``,
        ``_hetero_atom_style``, ``_hetero_lat_c``, ``_hetero_num_atom_types``,
        ``_hetero_virtual_type_reserved``), so a subclass can swap the whole
        run phase — e.g. a static PES grid scan instead of a slide — without
        reimplementing the stack assembly above it.
        """
        stack = self._stack
        bands = self._layer_bands
        box = self._hetero_box
        n = self.n_layers
        settings = self.config.settings
        sim = settings.simulation
        out = settings.output
        general = self.config.general

        # Same key set as SheetOnSheetSimulation.write_inputs' base_context (only
        # the keys the template actually references), with hetero-specific values
        # and the same SIM-ROOT-RELATIVE paths that builder uses.
        context = {
            "temp": general.temp,
            # One seed per channel, drawn in the builder like every other
            # family -- see SimulationBase._run_seed for why, and note a
            # template-side `| random` re-draws per occurrence.
            "seed_sheet_center": self._run_seed('sheet_center'),
            "xlo": box["xlo"],
            "xhi": box["xhi"],
            "ylo": box["ylo"],
            "yhi": box["yhi"],
            "zhi": box["zhi"],
            "data_file": self._run_dir_path("hetero.data"),
            "potential_file": self._run_dir_path("lammps", "system.in.settings"),
            "num_atom_types": self._hetero_num_atom_types,
            "reserve_virtual_type": self._hetero_virtual_type_reserved,
            "ngroups": stack.total_types,
            "n_layers": n,
            "constraint_mode": "none",
            "n_bond_types": 0,
            "atom_style": self._hetero_atom_style,
            "pot_type": "sw",
            "has_internal_lj": True,  # hybrid sw -> reaction-force friction proxy
            # Default an unset load to 0.0 (zero-load slide). A raw None renders
            # ``variable pressure equal None`` -> LAMMPS "Invalid thermo keyword
            # 'None' in variable formula", breaking every default-config run.
            # Mirrors the homogeneous loop-path fallback ``pressures or [0.0]``
            # (src/builders/sheetonsheet.py) which never reaches the template
            # with a bare None.
            "pressures": general.pressure if general.pressure is not None else 0.0,
            "scan_speed_config": general.scan_speed,
            "scan_angle_config": general.scan_angle,
            "scan_angle_force": general.scan_angle_force,
            "drive_method": sim.drive_method,
            "thermostat_type": settings.thermostat.type,
            "timestep": sim.timestep,
            "thermo": sim.thermo,
            "neighbor_list": sim.neighbor_list,
            "neigh_modify_command": sim.neigh_modify_command,
            "run_steps": sim.slide_run_steps,
            "min_style": sim.min_style,
            "minimization_command": sim.minimization_command,
            "results_freq": out.results_frequency,
            "dump_freq": out.dump_frequency.get("slide", 1000),
            "dump_enabled": out.dump.get("slide", False),
            "results_file_pattern": self._run_dir_path(
                "results", "friction_p${pressure}_a${a}_s${speed}"
            ),
            "dump_file_pattern": self._run_dir_path(
                "visuals", "slide_p${pressure}_a${a}_s${speed}.lammpstrj"
            ),
            "driving_spring_ev": (general.driving_spring or 50.0) / NM_TO_EV_A2,
            "bond_spring_ev": (general.bond_spring or 80.0) / NM_TO_EV_A2,
            "lat_c": self._hetero_lat_c,
            "ev_a_to_nn": EV_A_TO_NN,
            "ev_a3_to_gpa": EV_A3_TO_GPA,
            "layers": bands,
        }

        script = self.render_template("hetero/slide.lmp", context)
        self.write_file("lammps/slide.in", script)
        logger.info("Wrote hetero slide script to %s/lammps/slide.in", self.output_dir)

    def _record_twist_manifest(self, stack, twist_angle_target) -> None:
        """Write the achieved twist angle + moiré period into the run manifest.

        Augments the provenance manifest (``provenance/manifest.json``, created
        by :meth:`SimulationBase.write_file` during provenance init) with a
        ``twist`` block recording the user-selected target, the ACHIEVED
        commensurate angle (the two differ -- commensurate angles are discrete),
        and the moiré period. Downstream (postprocessing/AiiDA) can read the
        realised twist straight from provenance.
        """
        manifest_path = self.output_dir / "provenance" / "manifest.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"version": "1.0", "files": []}
        data["twist"] = {
            "twist_angle_target_deg": twist_angle_target,
            "twist_angle_achieved_deg": stack.twist_angle_achieved,
            "moire_period_ang": stack.moire_period,
        }
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(
            "Twisted moiré: target %.3f deg -> ACHIEVED %.3f deg, moiré period "
            "%.3f A (recorded in %s)",
            twist_angle_target, stack.twist_angle_achieved, stack.moire_period,
            manifest_path,
        )
