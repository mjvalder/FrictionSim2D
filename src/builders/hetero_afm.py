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

from ase.data import atomic_masses, atomic_numbers
from ase.io import read

from src.builders.afm import AFMSimulation
from src.builders.hetero import build_hetero_structure, register_sheet_block
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


def _assert_sheet_block_matches_data(pm: PotentialManager, data_path, offset: int) -> None:
    """Fail loud if the unified PM's sheet block diverges from the data file.

    ``hetero_rep.data`` contains ONLY the sheet stack, so its ``Masses`` (in type
    order) are exactly the ``sheet_1`` then ``sheet_2`` block. The unified AFM PM
    places that same block at ``[offset+1 …]``. Because both sides are numbered
    through :func:`register_sheet_block`, the sequences must match element-for-
    element; comparing the per-type mass sequence (mass encodes element identity)
    trips any equal-count reordering — the future-works §2 corruption mode — not
    merely a count mismatch.
    """
    id_to_type = {t.type_id: t for t in pm.types}
    sheet_ids = (pm.types.ids_by_component('sheet_1')
                 + pm.types.ids_by_component('sheet_2'))
    pm_masses = [atomic_masses[atomic_numbers[id_to_type[tid].element]]
                 for tid in sheet_ids]
    data_masses = _read_data_masses(data_path)

    if len(pm_masses) != len(data_masses):
        raise ValueError(
            f"Hetero sheet block size mismatch: unified PM has {len(pm_masses)} "
            f"sheet types but {Path(data_path).name} has {len(data_masses)}."
        )
    first_sheet_id = sheet_ids[0] if sheet_ids else None
    if first_sheet_id is not None and first_sheet_id != offset + 1:
        raise ValueError(
            f"Hetero sheet block starts at type {first_sheet_id}, expected "
            f"{offset + 1} (sub+tip offset)."
        )
    for i, (pm_m, data_m) in enumerate(zip(pm_masses, data_masses)):
        if abs(pm_m - data_m) > 1e-3:
            raise ValueError(
                f"Hetero sheet block diverges from {Path(data_path).name} at "
                f"sheet position {i} (data type {i + 1}, PM type {sheet_ids[i]}): "
                f"PM mass {pm_m:.4f} != data mass {data_m:.4f}. The unified-PM "
                "numbering and the assembled data file disagree."
            )


class HeteroAFMSimulation(AFMSimulation):
    """Periodic hetero (2-material) AFM tip-slide on an amorphous substrate.

    Inherits the tip-approach / indent-load-loop / slide / z-placement / langevin
    machinery from :class:`AFMSimulation`, overriding only the sheet-structure
    build (hetero stack -> in-plane replicate -> ``nocoeff`` data) and potential
    registration (``sheet`` -> ``sheet_1`` + ``sheet_2`` + cross-interaction +
    ``2D``/``2D_all`` union groups). The AFM templates are reused unchanged.
    """

    #: Run-phase template — unchanged from AFMSimulation; explicit for clarity.
    SLIDE_TEMPLATE = "afm/slide.lmp"

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
