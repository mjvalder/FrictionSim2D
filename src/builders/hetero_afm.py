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

from ase.io import read

from src.builders.afm import AFMSimulation
from src.builders.hetero import build_hetero_structure
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
        logger.info(
            "Built hetero AFM sheet: coincidence cell %.2f x %.2f A, replicated "
            "%dx%d -> %.2f x %.2f A footprint.", lx_cell, ly_cell, nx, ny, lx, ly,
        )
        return rep
