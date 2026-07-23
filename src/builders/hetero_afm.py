"""D2 hetero AFM tip-slide assembly (consumes hetero.py's structure builder).

``HeteroAFMSimulation`` subclasses :class:`~src.builders.afm.AFMSimulation`,
overriding only the sheet-structure build (hetero stack -> in-plane replicate)
and potential registration (``sheet_1``/``sheet_2`` split + ``2D`` union groups).
The AFM template bodies (``afm/system_init.lmp``, ``afm/slide.lmp``) are reused
unchanged (D2 prototype-validated).
"""
from __future__ import annotations

import math
from pathlib import Path


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
