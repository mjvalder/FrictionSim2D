"""D2 hetero AFM tip-slide assembly (consumes hetero.py's structure builder).

``HeteroAFMSimulation`` subclasses :class:`~src.builders.afm.AFMSimulation`,
overriding only the sheet-structure build (hetero stack -> in-plane replicate)
and potential registration (``sheet_1``/``sheet_2`` split + ``2D`` union groups).
The AFM template bodies (``afm/system_init.lmp``, ``afm/slide.lmp``) are reused
unchanged (D2 prototype-validated).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from ase.io import read

from src.builders import components
from src.builders.afm import AFMSimulation
from src.builders.hetero import build_hetero_structure, register_sheet_block
from src.builders.indent import IndentSimulation
from src.core.potential_manager import PotentialManager
from src.core.utils import get_model_dimensions

logger = logging.getLogger(__name__)


def compute_replication_factors(lx: float, ly: float, tip_r: float) -> tuple[int, int]:
    """Return ``(Nx, Ny)`` so the periodic footprint is >= 2x tip diameter each way.

    The replicated in-plane footprint must be >= ``4 * tip_r`` (two tip
    diameters) in each direction, else the tip interacts with its own periodic
    image (see the afm-tip-box-sizing rule). Never returns a factor below 1.
    """
    nx = max(1, math.ceil(4.0 * tip_r / lx))
    ny = max(1, math.ceil(4.0 * tip_r / ly))
    return nx, ny


def replicate_hetero_in_plane(data_path, out_path, nx: int, ny: int, settings) -> Path:
    """Replicate the coincidence cell ``nx*ny`` in-plane; write ``nocoeff``.

    Runs a tiny headless LAMMPS script (``read_data`` -> ``replicate nx ny 1`` ->
    ``write_data out nocoeff``) so the output carries no ``Pair Coeffs`` section
    (integration contract §3.6). Even for ``nx == ny == 1`` the data is
    round-tripped so the written file is uniformly ``nocoeff``. Reused by D5 for
    finite-patch sizing.
    """
    from src.builders.hetero import _run_lammps

    data_path, out_path = Path(data_path), Path(out_path)
    workdir = out_path.parent
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "_replicate.lmp"
    script.write_text(
        "units metal\n"
        "atom_style atomic\n"
        "boundary p p p\n"
        f"read_data {data_path.name}\n"
        f"replicate {nx} {ny} 1\n"
        f"write_data {out_path.name} nocoeff\n"
    )
    _run_lammps(workdir, script)
    return out_path


def _read_data_masses(data_path) -> list[float]:
    """Return the ``Masses`` section values of a LAMMPS data file, in type order."""
    masses: dict[int, float] = {}
    lines = Path(data_path).read_text().splitlines()
    in_masses = False
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if not in_masses:
            if stripped == "Masses":
                in_masses = True
            continue
        if not stripped:
            if masses:            # blank line after the section ends it
                break
            continue              # blank line(s) right after the header
        parts = stripped.split()
        try:
            masses[int(parts[0])] = float(parts[1])
        except (ValueError, IndexError):
            break                 # reached the next section header
    return [masses[tid] for tid in sorted(masses)]


def _read_settings_masses(settings_path) -> list[float]:
    """Real per-type masses from a settings file's ``mass <i|i*j> <value>`` commands.

    ``apply_sheet_edge_types`` (B3) retypes the assembled patch by DOUBLING the
    atom-type count; LAMMPS gives every fresh type the default mass ``1.0``, so the
    edge-typed ``hetero.data`` (and its replicated ``hetero_rep.data``) carries a
    uniform placeholder ``Masses`` block. The real per-type masses are the
    ``mass`` commands the potential spine writes into ``system.in.settings`` (e.g.
    ``mass 1*2 95.95 #Mo``). This parses those (expanding ``i*j`` ranges) so the
    layout guard can compare element identity even for the edge-typed file.
    """
    masses: dict[int, float] = {}
    for line in Path(settings_path).read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped.startswith("mass"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        try:
            value = float(parts[2])
        except ValueError:
            continue
        rng = parts[1]
        if "*" in rng:
            lo_s, hi_s = rng.split("*", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                continue
        else:
            try:
                lo = hi = int(rng)
            except ValueError:
                continue
        for tid in range(lo, hi + 1):
            masses[tid] = value
    return [masses[tid] for tid in sorted(masses)]


def assert_sheet_block_matches_data(pm: PotentialManager, data_path, offset: int = 0) -> None:
    """Fail loud if the unified PM's sheet block diverges from the data file.

    ``hetero_rep.data`` contains ONLY the sheet stack, so its ``Masses`` (in type
    order) are exactly the ``sheet_1`` then ``sheet_2`` block. The unified AFM PM
    places that same block at ``[offset+1 …]``. Because both sides are numbered
    through :func:`register_sheet_block`, the sequences must match element-for-
    element.

    The check is per-type, not count-only: it walks the ordered
    ``(pot_label, element, mass, region)`` block the PM registered and compares the
    mass sequence (mass encodes element identity) against the data file's
    ``Masses`` section position-by-position. That trips any equal-count reordering
    — the future-works §2 corruption mode where the ``regions=[None, 'edge']``
    split doubles the type count and a count-only assert would pass a permuted
    layout — not merely a size mismatch. With ``regions`` the block interleaves
    core/edge types per element, so a swap of two same-mass same-element types
    (core<->edge) leaves the mass sequence intact; the ``offset+1`` start check and
    the single-sourced numbering through :func:`register_sheet_block` are what pin
    that ordering.
    """
    id_to_type = {t.type_id: t for t in pm.types}
    sheet_ids = (pm.types.ids_by_component('sheet_1')
                 + pm.types.ids_by_component('sheet_2'))
    pm_masses = [atomic_masses[atomic_numbers[id_to_type[tid].element]]
                 for tid in sheet_ids]
    data_masses = _read_data_masses(data_path)

    # The edge-typed (finite) data file carries uniform placeholder masses (1.0):
    # apply_sheet_edge_types doubles the type count and LAMMPS defaults each fresh
    # type's mass to 1.0, with the real masses set by the companion settings file's
    # `mass` commands. Recover the real per-type masses from that settings file so
    # the element-wise comparison stays meaningful for the region-split path.
    if data_masses and all(abs(m - 1.0) < 1e-6 for m in data_masses):
        companion = Path(data_path).parent / "lammps" / "system.in.settings"
        if companion.is_file():
            real = _read_settings_masses(companion)
            if real:
                data_masses = real

    if len(pm_masses) != len(data_masses):
        raise ValueError(
            f"Hetero sheet block layout mismatch: unified PM has {len(pm_masses)} "
            f"sheet types but {Path(data_path).name} has {len(data_masses)}."
        )
    first_sheet_id = sheet_ids[0] if sheet_ids else None
    if first_sheet_id is not None and first_sheet_id != offset + 1:
        raise ValueError(
            f"Hetero sheet block layout mismatch: block starts at type "
            f"{first_sheet_id}, expected {offset + 1} (sub+tip offset)."
        )
    for i, (pm_m, data_m) in enumerate(zip(pm_masses, data_masses)):
        if abs(pm_m - data_m) > 1e-3:
            raise ValueError(
                f"Hetero sheet block layout mismatch / wrong order vs "
                f"{Path(data_path).name} at sheet position {i} (data type {i + 1}, "
                f"PM type {sheet_ids[i]}): PM mass {pm_m:.4f} != data mass "
                f"{data_m:.4f}. The unified-PM numbering and the assembled data "
                "file disagree."
            )


#: D2 call sites reference the private spelling; keep it pointing at the public
#: guard so both names are one implementation (no divergence).
_assert_sheet_block_matches_data = assert_sheet_block_matches_data


def _pair_material_token(config) -> str:
    """The numeric-``[2D-N]``-order ``+``-joined material pair token.

    E.g. ``h-MoS2+h-WS2``. Shared by :class:`HeteroAFMSimulation` and
    :class:`HeteroIndentSimulation`'s ``_material_id`` overrides — the latter
    can't just inherit the former's: :class:`~src.builders.indent.IndentSimulation`
    sits before :class:`HeteroAFMSimulation` in :class:`HeteroIndentSimulation`'s
    MRO (it needs its own single-material ``_material_id`` for the
    homogeneous indent path), so :class:`HeteroIndentSimulation` needs its own
    override too — but both should compute the pair token the same way.
    Mirrors :func:`src.core.run.material_token`'s convention on the raw-dict
    side (that function isn't reusable directly here: the builder only has
    the validated config object, not the raw config mapping).
    """
    return '+'.join(sheet.mat for sheet in config.sheets)


class HeteroAFMSimulation(AFMSimulation):
    """Periodic hetero (2-material) AFM tip-slide on an amorphous substrate.

    Inherits the tip-approach / indent-load-loop / slide / z-placement / langevin
    machinery from :class:`AFMSimulation`, overriding only the sheet-structure
    build (hetero stack -> in-plane replicate -> ``nocoeff`` data) and potential
    registration (``sheet`` -> ``sheet_1`` + ``sheet_2`` + cross-interaction +
    ``2D``/``2D_all`` union groups). The AFM templates are reused unchanged.

    ``_material_id`` is overridden (not just ``_get_hpc_job_name``) so the pair
    token also reaches :class:`FiniteHeteroAFMSimulation` for free through
    inheritance, and any future material-keyed metadata added to the AFM base
    picks it up automatically rather than needing its own override.
    """

    #: Run-phase template — unchanged from AFMSimulation; explicit for clarity.
    SLIDE_TEMPLATE = "afm/slide.lmp"

    def _material_id(self) -> str:
        """Pair token (``h-MoS2+h-WS2``): every ``self.config.sheets`` entry,
        in the numeric ``[2D-N]`` order the config fold already preserves,
        joined with ``+`` — same convention as
        :func:`src.core.run.material_token` and
        :meth:`HeteroIndentSimulation._material_id` below. Without this,
        ``_get_hpc_job_name`` (inherited from :class:`AFMSimulation`) named
        every hetero AFM job after sheet 1 alone, so two pairs sharing
        material 1 (e.g. ``h-MoS2+h-WS2`` vs ``h-MoS2+h-WSe2``) produced the
        identical HPC job name ``afm_h-MoS2`` despite building into distinct
        output directories (the directory itself was already pair-token-safe
        via Task 1's ``output_dir_for``/``material_token`` — only the job
        *name* still collided). Also includes
        :meth:`~src.builders.afm.AFMSimulation._flake_id_suffix` (inherited
        unmodified) so a hetero AFM's ``[flake]`` also keys the job name.
        """
        return _pair_material_token(self.config) + self._flake_id_suffix()

    #: Per-material sheet region split; None here, D5 (finite) sets [None, 'edge'].
    _sheet_regions = None

    def _build_hetero_sheet(self, build_dir):
        """Build the hetero stack, guard orthogonality, replicate in-plane.

        Sets ``self._hetero_rep_path`` (the replicated ``nocoeff`` data file),
        ``self._hetero_rep_box`` = ``(lx, ly, z_extent)`` of the replicated cell,
        and ``self._hetero_replication`` = ``(nx, ny)``. Raises ``ValueError`` on
        a triclinic coincidence cell (xy tilt != 0) — an orthogonal amorphous box
        cannot tile it; twisted/triclinic pairs are D3.
        """
        build_dir = Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        stack = build_hetero_structure(self.config, self.config.settings, build_dir)
        self._hetero_stack = stack

        atoms = read(str(stack.data_path), format="lammps-data")
        cell = atoms.get_cell()
        xy_tilt = float(cell[1, 0])
        if abs(xy_tilt) > 1e-6:
            raise ValueError(
                f"Hetero coincidence cell is triclinic (xy tilt={xy_tilt:.4f} A); "
                "an orthogonal amorphous substrate cannot tile it. Twisted/triclinic "
                "pairs are D3. Use a pair whose coincidence cell canonicalises orthogonal."
            )

        lx_cell, ly_cell = float(cell[0, 0]), float(cell[1, 1])
        nx, ny = compute_replication_factors(lx_cell, ly_cell, self.config.tip.r)

        rep = build_dir / "hetero_rep.data"
        replicate_hetero_in_plane(stack.data_path, rep, nx, ny, self.config.settings)

        dims = get_model_dimensions(rep)
        lx = float(dims["xhi"]) - float(dims["xlo"])
        ly = float(dims["yhi"]) - float(dims["ylo"])
        z_extent = float(dims["zhi"]) - float(dims["zlo"])

        self._hetero_rep_path = rep
        self._hetero_rep_box = (lx, ly, z_extent)
        self._hetero_replication = (nx, ny)
        # The replicated hetero stack is the "sheet": expose its box as sheet_dims
        # (xlo/xhi/... keys, matching the homogeneous path) so the inherited
        # substrate build sizes/places the amorphous slab to this footprint, and
        # build_render_context reads the in-plane extents from it.
        self.sheet_dims = {k: float(dims[k])
                           for k in ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")}
        # A single assembled read_data drives one build() iteration (n_layers=1),
        # so lat_c is only ever multiplied by (n_layers-1)=0; set it non-None to
        # skip the base's 6.0 A fallback.
        self.lat_c = z_extent
        logger.info(
            "Built hetero AFM sheet: coincidence cell %.2f x %.2f A, replicated "
            "%dx%d -> %.2f x %.2f A footprint.", lx_cell, ly_cell, nx, ny, lx, ly,
        )
        return rep

    def _build_components(self, build_dir):
        """D2 tip + substrate build, with an opt-in crystalline-matched substrate (D2b).

        Mirrors :meth:`AFMSimulation._build_components` (tip, optional flake, and
        langevin region handling are unchanged), overriding only the substrate: when
        ``sub.amorph == 'c' and sub.match`` the substrate is built by straining its
        surface cell onto the fixed hetero box (``self._hetero_rep_box``) via
        :func:`~src.builders.components.build_matched_crystalline_substrate`, so it
        tiles the periodic cell with no seam. Otherwise the amorphous/unmatched D2
        path (:func:`~src.builders.components.build_substrate`) is used unchanged.

        ``_build_components`` is fully overridden (not extended) because the base
        builds the substrate inline; Task 5's scope is this file only, so the tip /
        flake / langevin blocks are re-expressed here rather than by adding a hook to
        ``afm.py``. Returns the base's ``(tip_path, tip_radius, sub_path)`` tuple.
        """
        tip_path, tip_radius = components.build_tip(
            self.config.tip, build_dir, self.config.settings
        )
        logger.info("Built tip: %s", tip_path.name)

        if self.config.flake is not None:
            self.flake_path, self.flake_dims = components.build_flake_component(
                self.config.flake, build_dir
            )
            logger.info("Built flake: %s", self.flake_path.name)
            self._maybe_twist_flake()

        sub_path = None
        if self.config.sub is not None:
            if self.config.sub.amorph == 'c' and self.config.sub.match:
                lx, ly, _ = self._hetero_rep_box
                hetero_box = np.array([[lx, 0.0], [0.0, ly]])
                sub_path = components.build_matched_crystalline_substrate(
                    hetero_box, self.config.sub, self.config.sub.thickness, build_dir,
                )
                logger.info("Built crystalline-matched substrate: %s", sub_path.name)
            else:
                sub_path = components.build_substrate(
                    self.config.sub,
                    build_dir,
                    self.sheet_dims,
                    settings=self.config.settings,
                    target_x=self.config.sub.x if self.config.general.finite_sheet else None,
                    target_y=self.config.sub.y if self.config.general.finite_sheet else None,
                )
                logger.info("Built substrate: %s", sub_path.name)

        if self.config.settings.thermostat.type == 'langevin':
            tip_height = tip_radius / self.config.settings.geometry.tip_reduction_factor
            components.apply_langevin_regions(
                component_name='tip',
                component_path=tip_path,
                config=self.config.tip,
                settings=self.config.settings,
                component_height=tip_height,
            )
            logger.info("Applied Langevin regions to tip")

            if self.config.sub is not None:
                components.apply_langevin_regions(
                    component_name='sub',
                    component_path=sub_path,
                    config=self.config.sub,
                    settings=self.config.settings,
                    component_height=self.config.sub.thickness,
                    substrate_thickness=self.config.sub.thickness,
                )
                logger.info("Applied Langevin regions to substrate")

        return tip_path, tip_radius, sub_path

    def _build_sheet_for_layers(self, build_dir, n_layers):
        """Override the AFM sheet source: the replicated hetero stack is the sheet.

        One assembled ``read_data`` drives a single build() iteration
        (``config.sheet.layers == [1]``); ``_build_hetero_sheet`` sets
        ``sheet_dims``/``lat_c`` from the replicated box.
        """
        self._build_hetero_sheet(Path(build_dir))
        self.sheet_paths[n_layers] = self._hetero_rep_path

    def _calculate_z_positions(self, n_layers, tip_radius):
        """Z-placement for the hetero stack (no single 'sheet' component exists).

        The inherited base method calls ``calculate_gap('sub','sheet')`` /
        ``('sheet','tip')``, but the hetero PM has ``sheet_1``/``sheet_2`` instead,
        and the assembled stack is thick (its z-extent is baked into
        ``hetero_rep.data``). Use the conservative max gap across BOTH materials
        and place the tip above the TOP of the stack; the indent script then
        live-measures ``2D_all`` and lowers the tip to ``contact_gap``, so a
        slightly-high start is safe.
        """
        pm = self.pm[n_layers]
        sheet_names = [n for n in ('sheet_1', 'sheet_2') if n in pm.components]

        self.z_positions[n_layers] = {}
        gap_sub_sheet = max(pm.calculate_gap('sub', s, buffer=0.5) for s in sheet_names)
        sub_thickness = self.config.sub.thickness
        self.z_positions[n_layers]['sub'] = 0.0
        sheet_base_z = sub_thickness + gap_sub_sheet
        self.z_positions[n_layers]['sheet'] = sheet_base_z

        z_extent = self._hetero_rep_box[2]        # full assembled-stack thickness
        gap_sheet_tip = max(pm.calculate_gap(s, 'tip', buffer=0.5) for s in sheet_names)
        start_clearance = self.config.settings.geometry.tip_start_clearance
        tip_z = sheet_base_z + z_extent + gap_sheet_tip + tip_radius + start_clearance

        self.z_positions[n_layers]['tip'] = tip_z
        self.z_positions[n_layers]['tip_contact_gap'] = gap_sheet_tip
        logger.info(
            "Hetero z-placement: sub-sheet gap %.2fA, stack z-extent %.2fA, "
            "sheet-tip gap %.2fA -> tip start z %.2fA.",
            gap_sub_sheet, z_extent, gap_sheet_tip, tip_z,
        )

    def _generate_potentials(self, n_sheet_layers):
        """Register sub/tip/sheet_1/sheet_2 in ONE PM and emit the 2D union groups.

        Mirrors :meth:`AFMSimulation._generate_potentials` for sub/tip (no explicit
        regions — the PM auto-splits the langevin [None,'fix','thermo'] block), then
        splits the homogeneous ``sheet`` into ``sheet_1``/``sheet_2`` via the shared
        :func:`register_sheet_block` (so the unified-PM layout cannot diverge from
        ``hetero_rep.data``). Adds the sheet_1<->sheet_2 cross-interaction, appends
        ``group 2D_all union sheet_1_all sheet_2_all`` into ``system.in.settings``
        (both AFM scripts include it), and sets the ``sheet_types`` context var to
        the sheet_1 ∪ sheet_2 type-id union that ``slide.lmp`` groups on.
        """
        pm = PotentialManager(
            self.config.settings,
            potentials_dir=self.output_dir / "provenance" / "potentials",
            potentials_prefix=str(self.relative_run_dir / "provenance" / "potentials"),
        )
        pm.set_lj_overrides(self.config.lj_override)

        # sub/tip exactly as the homogeneous path (no regions -> PM langevin split).
        if self.config.sub is not None:
            pm.register_component('sub', self.config.sub)
        pm.register_component('tip', self.config.tip)

        # sheet_1/sheet_2 through the SAME spine that numbered hetero_rep.data.
        sheet_names = register_sheet_block(pm, self.config.sheets,
                                           regions=self._sheet_regions)

        if self.config.settings.simulation.drive_method == 'virtual_atom':
            pm.register_virtual_atom()

        if self.config.sub is not None:
            pm.add_self_interaction('sub')
        pm.add_self_interaction('tip')
        for name in sheet_names:
            pm.add_self_interaction(name)

        if self.config.sub is not None:
            pm.add_cross_interaction('sub', 'tip')
        for name in sheet_names:
            if self.config.sub is not None:
                pm.add_cross_interaction('sub', name)
            pm.add_cross_interaction('tip', name)
        # every distinct material pair at the interface
        for i in range(len(sheet_names)):
            for j in range(i + 1, len(sheet_names)):
                pm.add_cross_interaction(sheet_names[i], sheet_names[j])

        settings_path = self.output_dir_layer[n_sheet_layers] / "lammps" / "system.in.settings"
        pm.write_file(settings_path)
        # 2D_all union of the per-material *_all groups get_component_groups_string
        # already emits — appended so BOTH system_init.lmp and slide.lmp see it
        # (the templates are untouched). `group 2D` itself comes from read_data.
        union_members = ' '.join(f"{name}_all" for name in sheet_names)
        with open(settings_path, 'a', encoding='utf-8') as fh:
            fh.write(f"\ngroup 2D_all union {union_members}\n")

        # Base bookkeeping build_render_context reads; sheet_types = sheet_1 ∪ sheet_2.
        sheet_ids = []
        for name in sheet_names:
            sheet_ids.extend(pm.types.ids_by_component(name))
        self.groups[n_sheet_layers] = {}
        if self.config.sub is not None:
            self.groups[n_sheet_layers]['sub_types'] = pm.types.get_group_string('sub')
        self.groups[n_sheet_layers]['tip_types'] = pm.types.get_group_string('tip')
        self.groups[n_sheet_layers]['sheet_types'] = ' '.join(map(str, sheet_ids))

        # Belt-and-suspenders: element-for-element layout guard vs the data file.
        offset = len(pm.types.ids_by_component('sub')) + len(pm.types.ids_by_component('tip'))
        _assert_sheet_block_matches_data(pm, self._hetero_rep_path, offset)

        return pm


class FiniteHeteroAFMSimulation(HeteroAFMSimulation):
    """Finite (edge-typed) hetero AFM tip-slide on an amorphous substrate (D5).

    The finite counterpart of :class:`HeteroAFMSimulation`: the hetero stack is
    built as an isolated **patch** whose perimeter is edge-typed (B3's finite
    ``build_hetero_structure`` path, auto-selected because ``general.finite_sheet``
    is ``True``), replicated so the patch is >= 2x the tip diameter (reusing the
    D2 replication helpers, inherited via :meth:`_build_hetero_sheet`), and centred
    on an amorphous substrate sized to ``[sub] x/y`` (the inherited finite handling
    from :class:`~src.builders.afm.AFMSimulation` — ``_build_components`` sizes the
    slab to the ``[sub]`` footprint, ``_compute_sheet_offset`` centres the patch,
    ``_validate_final_substrate_dims`` guards the margin).

    Because both the assembled ``hetero_rep.data`` and the unified AFM PM route
    their sheet block through the SAME :func:`register_sheet_block`, setting
    ``self._sheet_regions = [None, 'edge']`` splits every material's types into an
    interleaved core/edge pair on BOTH sides consistently (the future-works §2
    guard, :func:`assert_sheet_block_matches_data`, verifies this element-wise).
    The ``2D_edge`` union group and the finite-edge context flags (Task 2) make the
    UNCHANGED ``afm/system_init.lmp`` / ``afm/slide.lmp`` edge-tether block fire on
    the hetero ``2D_edge`` group.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Split each material's sheet types into core (None) + edge -> the same
        # region split B3's finite build_hetero_structure applies to the data file,
        # kept single-sourced through register_sheet_block so they cannot diverge.
        self._sheet_regions = [None, 'edge']
        self._finite = True

    def _generate_potentials(self, n_sheet_layers):
        """D2 sub/tip/sheet_1/sheet_2 registration, exposing the single PM.

        Delegates to :meth:`HeteroAFMSimulation._generate_potentials` (which,
        because ``self._sheet_regions == [None, 'edge']``, splits the sheet block
        into core/edge types matching the edge-typed ``hetero_rep.data`` and runs
        the element-wise layout guard), and records ``self._pm`` so the finite
        layout-guard test can assert the unified PM's sheet block against the data
        file directly.
        """
        pm = super()._generate_potentials(n_sheet_layers)
        self._pm = pm
        self._emit_edge_groups(pm, n_sheet_layers)
        return pm

    def _emit_edge_groups(self, pm, n_sheet_layers):
        """Append the per-material + combined ``2D_edge`` groups to the run settings.

        :meth:`PotentialManager.get_component_groups_string` special-cases only the
        literal component name ``sheet`` for the ``2D_edge`` split, so the hetero
        ``sheet_1`` / ``sheet_2`` components fall to the generic ``{name}_all``
        branch with NO edge groups; D2 additionally appends only ``2D_all``. The
        finite path therefore appends the per-material ``sheet_i_edge`` type groups
        and their ``2D_edge`` union HERE (mirroring B3's structure-side emission),
        so the UNCHANGED ``afm/system_init.lmp`` / ``afm/slide.lmp`` finite-edge
        block ``group sheet_edge intersect 2D 2D_edge`` resolves at runtime. The
        finite context flags (``finite_sheet_enabled`` / ``finite_sheet_edge_mode``
        / width / spring-k) that gate that block are already emitted by
        :meth:`AFMSimulation.build_render_context` straight from
        ``config.general.finite_sheet`` + ``geometry`` — inherited unchanged.
        """
        sheet_names = [n for n in ('sheet_1', 'sheet_2') if n in pm.components]
        settings_path = (self.output_dir_layer[n_sheet_layers]
                         / "lammps" / "system.in.settings")
        lines = ["", "# Finite-sheet edge groups (per-material + combined 2D_edge)"]
        for name in sheet_names:
            edge_ids = pm.types.ids_by_component_region(name, 'edge')
            lines.append(f"group {name}_edge type {' '.join(str(t) for t in edge_ids)}")
        lines.append(
            f"group 2D_edge union {' '.join(f'{name}_edge' for name in sheet_names)}"
        )
        with open(settings_path, 'a', encoding='utf-8') as fh:
            fh.write("\n".join(lines) + "\n")


class HeteroIndentSimulation(IndentSimulation, HeteroAFMSimulation):
    """Indent (hold + retract) on a two-material periodic heterostructure sheet.

    Pure MRO composition: :class:`IndentSimulation` supplies the run-script
    identity (``SLIDE_TEMPLATE``, ``SPLIT_OUTER_LOOPS``, the hold/retract
    render-context additions, ``build()``'s ``_write_indent_meta`` tail),
    :class:`HeteroAFMSimulation` supplies the component build (hetero stack ->
    in-plane replicate, ``sheet_1``/``sheet_2`` potential split). No method
    bodies are duplicated here; only the material-token trio below is added.

    **Step-1 findings (verified by printing, not inferred):**

    - MRO is ``HeteroIndentSimulation -> IndentSimulation ->
      HeteroAFMSimulation -> AFMSimulation -> SimulationBase -> ABC ->
      object``. Both class attributes ``IndentSimulation`` declares resolve
      from it first: ``SLIDE_TEMPLATE == "afm/indent.lmp"`` (not
      ``HeteroAFMSimulation``'s ``"afm/slide.lmp"``) and
      ``SPLIT_OUTER_LOOPS == frozenset({'force'})`` (not AFM's
      ``{'force', 'scan_angle'}``) — correct here too: the hold+retract
      template (``afm/indent.lmp``) declares the angle variable but never
      drives laterally with it, hetero or not, so splitting per scan_angle
      would still just emit physically-identical scripts. No explicit
      re-declaration was needed; :func:`test_hetero_indent_mro_and_split_outer_loops`
      pins both values so a future MRO/base-class change cannot silently
      flip them.
    - ``IndentSimulation.build()`` calls ``super().build()`` then
      ``self._write_indent_meta()``. ``super()`` from an instance of this
      class resolves to ``HeteroAFMSimulation`` next — which does not
      override ``build`` — so it falls through to ``AFMSimulation.build()``.
      That method's internal calls (``self._build_sheet_for_layers``,
      ``self._build_components``, ``self._generate_potentials``,
      ``self._calculate_z_positions``) dispatch dynamically on the instance,
      landing on ``HeteroAFMSimulation``'s overrides, exactly as for plain
      ``HeteroAFMSimulation``.
    - ``build_render_context`` chain: ``IndentSimulation.build_render_context``
      calls ``super().build_render_context(n_layers)``, which — since
      ``HeteroAFMSimulation`` has no override — resolves to
      ``AFMSimulation.build_render_context``. That base method reads
      ``self.pm[n]``, ``self.box_dims``, ``self.sheet_offset``, ``self.lat_c``,
      ``self.z_positions[n]`` (keys ``sub``/``sheet``/``tip``/
      ``tip_contact_gap``), ``self.groups[n]`` (keys ``sub_types``/
      ``tip_types``/``sheet_types``, plus flake keys when a flake is
      configured), ``self.sheet_paths[n]``, ``self.tip_path``,
      ``self.sub_path``, ``self.sheet_dims``, and
      ``self.config.sheet.pot_type`` (for the reaxff/atom_style check — safe:
      ``config.sheet`` is always the first sheet, mirrored by
      ``_fold_sheet_sections`` even in the hetero case). Every one of these is
      already populated by ``HeteroAFMSimulation``'s overridden
      ``_build_hetero_sheet`` / ``_build_components`` / ``_generate_potentials``
      / ``_calculate_z_positions`` (the same attributes, same shapes, that
      :class:`HeteroAFMSimulation`'s own docstring documents) — so no
      additional override is needed for the build/render path itself.
    - The one contract MRO does *not* satisfy for free: ``IndentSimulation``
      keys the job name (``_get_hpc_job_name``), the ``derive_seed`` input
      (``_velocity_seed``), and ``indent_meta.json``'s ``"material"`` field
      off a single ``self._material_id()`` hook (added alongside this class),
      which defaults to ``self.config.sheet.mat`` — sheet 1 only. Overridden
      below to the numeric-order ``+``-joined pair token, so two hetero pairs
      sharing material 1 (e.g. ``h-MoS2+h-WS2`` vs ``h-MoS2+h-BN``) get
      distinct job names, seeds, and manifest keys instead of colliding.
    """

    def _material_id(self) -> str:
        """Pair token (``h-MoS2+h-WS2``) via the shared :func:`_pair_material_token`
        helper (see its docstring for why this can't just inherit
        :meth:`HeteroAFMSimulation._material_id` despite computing the same
        thing). Also includes
        :meth:`~src.builders.afm.AFMSimulation._flake_id_suffix` (inherited
        unmodified) so a hetero indent's ``[flake]`` also keys the job name,
        derived seed, and ``indent_meta.json``.
        """
        return _pair_material_token(self.config) + self._flake_id_suffix()
