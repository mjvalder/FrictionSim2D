#!/usr/bin/env python3
"""Bilayer MoS2 binding-curve benchmark — the MLIP go/no-go physics gate.

Spec: ``docs/superpowers/specs/2026-08-06-mlip-support-design.md`` §5 (this is
"the go/no-go physics gate before production"), feeding §3.2 (VRAM sizing) and
§3.3 (the ``lat_c`` / dome-plow caveat).

What it does
------------
Scans the interlayer separation of a 2H-MoS2 bilayer from 5.5 to 9.0 A in 0.1 A
steps as **rigid single points** (``run 0``, no minimize, no dynamics) and
records the potential energy at each point. Rigidity is the whole point: a
relaxation would move the layers off the separation being scanned and the
"binding curve" would collapse to a single number.

From that curve it reports

* the equilibrium interlayer separation (parabola through the bracketing
  triple, so the answer is not quantised to the 0.1 A grid) — this is the
  number a config's ``lat_c`` should be set to (§3.3: a too-small initial gap
  under an attractive foundation model is the "dome plow" failure mode);
* the binding energy per unit area, referenced to the widest separation;
* the in-plane lattice constant, from a 3-point quadratic fit of scaled-cell
  single points at the equilibrium separation;
* measured VRAM (``nvidia-smi --query-gpu=memory.used``, sampled while LAMMPS
  runs) against the atom count — the table §3.2 asks for.

Run it once per condition and diff the JSONs: SW+LJ reference vs MLIP-base vs
MLIP+D3, per framework.

    # DeePMD, no dispersion
    python scripts/mlip_bilayer_benchmark.py --framework deepmd \\
        --model $HOME/models/dpa3_mos2_frozen.pth \\
        --lmp $PREFIX/lammps-deepmd/bin/lmp --out deepmd_base.json

    # MACE + D3 (Kokkos flags come from the registry's LauncherSpec).
    # symmetrix consumes a .json extracted from the torch checkpoint:
    #   symmetrix_extract_mace -m mace-mpa-0-medium.model --chemical-symbols Mo S
    python scripts/mlip_bilayer_benchmark.py --framework mace \\
        --model $HOME/models/mace-mpa-0-medium.json \\
        --lmp $PREFIX/lammps-mace/build/lmp --d3 --out mace_d3.json

    # SW + LJ classical reference
    python scripts/mlip_bilayer_benchmark.py --framework sw \\
        --pot potentials/MoS.sw --lmp lmp --out sw_reference.json

Structure
---------
Everything that can be wrong without a GPU is a **pure function**
(:func:`render_deck`, :func:`parse_potential_energy`, :func:`quadratic_minimum`,
:func:`assemble_results`, :func:`build_bilayer`) and is covered by
``tests/test_benchmark_script.py``. The two impure operations — running LAMMPS
and asking ``nvidia-smi`` — are isolated in :func:`run_lammps` and
:func:`sample_gpu_memory`.

Geometry note
-------------
The MoS2 recipe is the same one ``scripts/milestone0/make_geometries.py`` uses
(hexagonal primitive -> **orthogonal** 2x1 supercell, so LAMMPS never sees a
triclinic skew at its "too large" threshold; explicit per-layer atom types
1=Mo/2=S in layer 1, 3=Mo/4=S in layer 2; ``Masses`` written by ASE). It is
re-implemented here rather than imported because the benchmark must vary both
the interlayer separation and the in-plane lattice constant, which that script
holds as module constants — but a test asserts the two agree atom-for-atom at
the shared default (``a = 3.18 A``, separation ``6.15 A``).

Only ASE + numpy are needed; ``frictionsim2d.core.mlip`` is imported read-only so the
pair_style/pair_coeff/D3 strings come from the same registry the builders use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The registry is the single source of truth for every MLIP string we emit
# (and for the launcher flags / env exports). Never re-derive them here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frictionsim2d.core.mlip import (  # noqa: E402  (path shim must precede the import)
    MLIP_REGISTRY,
    d3_coeff_line,
    d3_style_token,
    get_mlip,
    is_mlip,
)

# --- geometry constants, mirroring scripts/milestone0/make_geometries.py -----
LATTICE_A = 3.18          # in-plane lattice constant of 2H-MoS2 (A)
THICKNESS = 3.19          # S-S vertical separation within one layer (A)
INTERLAYER = 6.15         # default layer-to-layer spacing (bulk 2H c/2, A)
NX, NY = 2, 1             # repeats of the orthogonal cell -> 12 atoms/layer
VACUUM = 12.0             # vacuum padding on each side along z (A)
STACK_SHIFT = (2.0 / 3.0, 1.0 / 3.0)   # 2H (AA') stacking shift, fractional

DEFAULT_ELEMENTS: Tuple[str, ...] = ("Mo", "S", "Mo", "S")
FRAMEWORKS = ("sw", "deepmd", "mace")

# 1 eV/A^2 = 1.602176634e-19 J / 1e-20 m^2
EV_PER_A2_TO_J_PER_M2 = 16.02176634

GPU_POLL_SECONDS = 0.2


class LogParseError(RuntimeError):
    """Raised when a LAMMPS log carries no usable thermo output."""


@dataclass(frozen=True)
class D3Settings:
    """The four D3 knobs, shaped like ``settings.potential``.

    :func:`frictionsim2d.core.mlip.d3_style_token` takes a settings object and reads
    ``settings.potential.d3_*``; exposing ``potential`` as self lets this
    standalone script reuse that function verbatim, so the argument order
    stays defined in exactly one place (it is the one Milestone 0 verifies).
    """

    d3_damping: str = "bj"
    d3_functional: str = "pbe"
    d3_cutoff: float = 16.0
    d3_cn_cutoff: float = 8.0

    @property
    def potential(self) -> "D3Settings":
        return self


# ---------------------------------------------------------------------------
# (a) deck text generation — pure
# ---------------------------------------------------------------------------

def render_deck(
    framework: str,
    data_file: str,
    *,
    model_path: Optional[str] = None,
    sw_path: Optional[str] = None,
    elements: Sequence[str] = DEFAULT_ELEMENTS,
    d3: Optional[D3Settings] = None,
    lj_epsilon: float = 0.0224,
    lj_sigma: float = 3.13,
    lj_cutoff: float = 11.0,
) -> str:
    """Return the complete LAMMPS deck for one rigid single point.

    Args:
        framework: ``'sw'``, ``'deepmd'`` or ``'mace'``.
        data_file: name of the data file the deck reads (relative to the run
            directory, or absolute).
        model_path: MLIP model file (required for deepmd/mace).
        sw_path: Stillinger-Weber potential file (required for ``sw``).
        elements: one element symbol per atom type, layer-major
            (``Mo S Mo S`` for a bilayer).
        d3: enable a ``dispersion/d3`` overlay. MLIP only — D3 needs a real
            element for every type, which is exactly the full-cell condition.
        lj_epsilon, lj_sigma, lj_cutoff: interlayer LJ for the SW reference.

    Raises:
        ValueError: unknown framework, missing potential/model, or D3 asked
            for on the classical reference.
    """
    fw = (framework or "").lower()
    if fw not in FRAMEWORKS:
        raise ValueError(
            f"unknown framework {framework!r}; expected one of {', '.join(FRAMEWORKS)}"
        )
    if d3 is not None and not is_mlip(fw):
        raise ValueError(
            "dispersion/d3 requires full-cell MLIP coverage (it needs a real "
            "element for every atom type); it cannot overlay the sw reference"
        )

    lines: List[str] = [
        "# FrictionSim2D bilayer MoS2 binding benchmark -- rigid single point",
        f"# framework: {fw}",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
    ]
    if is_mlip(fw):
        # Style-mandated deck lines (symmetrix/mace: global atom map + newton
        # on). They must precede read_data, hence their placement here.
        lines.extend(d.strip() for d in get_mlip(fw).deck_directives)
    lines.append(f"read_data {data_file}")

    if is_mlip(fw):
        if not model_path:
            raise ValueError(f"framework {fw!r} requires a model file (--model)")
        spec = get_mlip(fw)
        style = spec.style_token(model_path)
        if d3 is None:
            lines.append(f"pair_style {style}")
            lines.append(spec.coeff_line("", model_path, list(elements), hybrid=False))
        else:
            lines.append(f"pair_style hybrid/overlay {style} {d3_style_token(d3)}")
            lines.append(spec.coeff_line("", model_path, list(elements)))
            lines.append(d3_coeff_line(list(elements)))
    else:
        if not sw_path:
            raise ValueError(
                "framework 'sw' requires an SW potential file (--pot)"
            )
        lines.extend(
            _sw_reference_pair_lines(
                sw_path, elements, lj_epsilon, lj_sigma, lj_cutoff
            )
        )

    lines.extend([
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
        "thermo_style custom step pe",
        "thermo 1",
        "run 0",
    ])
    return "\n".join(lines) + "\n"


def _sw_reference_pair_lines(
    sw_path: str,
    elements: Sequence[str],
    lj_epsilon: float,
    lj_sigma: float,
    lj_cutoff: float,
) -> List[str]:
    """Classical reference: one SW instance per layer + interlayer lj/cut.

    SW is a covalent intralayer potential and carries no dispersion, so the
    interlayer binding it would otherwise report is zero. The LJ term is what
    the production sheet-on-sheet decks use for the interlayer, so the
    reference curve here is the one an MLIP has to beat.
    """
    n_layers = len(elements) // 2
    lines = [
        "pair_style hybrid " + " ".join(["sw"] * n_layers)
        + f" lj/cut {lj_cutoff}"
    ]
    for layer in range(n_layers):
        atom_map = [
            elements[i] if i // 2 == layer else "NULL"
            for i in range(len(elements))
        ]
        lines.append(
            f"pair_coeff * * sw {layer + 1} {sw_path} {' '.join(atom_map)}"
        )
    for lower in range(n_layers):
        for upper in range(lower + 1, n_layers):
            lines.append(
                f"pair_coeff {2 * lower + 1}*{2 * lower + 2} "
                f"{2 * upper + 1}*{2 * upper + 2} "
                f"lj/cut {lj_epsilon} {lj_sigma}"
            )
    return lines


# ---------------------------------------------------------------------------
# (b) log energy parsing — pure
# ---------------------------------------------------------------------------

def parse_potential_energy(log_text: str, column: str = "PotEng") -> float:
    """Return the last ``PotEng`` value in a LAMMPS log.

    LAMMPS thermo output is a whitespace table under a header row naming the
    columns. A deck may print several tables (setup + run); the last numeric
    row of the last table is the answer.

    Raises:
        LogParseError: if no thermo header names ``column`` (typically the run
            died before setup — an unknown pair style, a missing plugin), or a
            header appeared but no numeric row followed it (the run died during
            setup, e.g. a neighbor-bin or model-element mismatch).
    """
    header: Optional[List[str]] = None
    col_index: Optional[int] = None
    value: Optional[float] = None
    saw_header = False

    for line in log_text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if column in tokens:
            header = tokens
            col_index = tokens.index(column)
            saw_header = True
            continue
        if header is None or col_index is None:
            continue
        if len(tokens) != len(header):
            continue
        try:
            row = [float(tok) for tok in tokens]
        except ValueError:
            continue
        value = row[col_index]

    if not saw_header:
        raise LogParseError(
            f"no thermo header naming {column!r} in the LAMMPS log — the run "
            "almost certainly failed before setup (unknown pair style, missing "
            "LAMMPS_PLUGIN_PATH, or a bad model file). Inspect the log."
        )
    if value is None:
        raise LogParseError(
            f"thermo header found but no numeric {column} row followed it — "
            "the run died during setup. Inspect the log."
        )
    return value


# ---------------------------------------------------------------------------
# (c) quadratic minimum fit — pure
# ---------------------------------------------------------------------------

def quadratic_minimum(
    xs: Sequence[float], ys: Sequence[float]
) -> Tuple[float, float]:
    """Fit ``y = a x^2 + b x + c`` and return the vertex ``(x_min, y_min)``.

    Used twice: to refine the equilibrium separation off the 0.1 A grid, and
    to turn the three scaled-cell single points into an in-plane lattice
    constant.

    Raises:
        ValueError: fewer than three points, or a fit that curves downward
            (``a <= 0``) and therefore has a maximum, not a minimum.
    """
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if len(xs) < 3 or len(ys) < 3:
        raise ValueError("a quadratic fit needs at least three points")
    if len(xs) != len(ys):
        raise ValueError("xs and ys must be the same length")

    a, b, c = np.polyfit(np.asarray(xs), np.asarray(ys), 2)
    if a <= 0:
        raise ValueError(
            "quadratic fit curves downward (a <= 0): these points bracket a "
            "maximum, not a minimum"
        )
    x_min = float(-b / (2.0 * a))
    y_min = float(a * x_min ** 2 + b * x_min + c)
    return x_min, y_min


def equilibrium_from_curve(
    curve: Dict[float, float]
) -> Tuple[float, float]:
    """Refine the binding minimum from its bracketing triple.

    Falls back to the lowest sampled point when the minimum sits on an
    endpoint (a monotonic curve — no binding well inside the scanned window,
    itself a reportable result) or when the local fit is not convex.
    """
    if not curve:
        raise ValueError("empty binding curve")
    xs = sorted(curve)
    ys = [curve[x] for x in xs]
    i = int(np.argmin(ys))
    if 0 < i < len(xs) - 1:
        try:
            x_min, y_min = quadratic_minimum(xs[i - 1:i + 2], ys[i - 1:i + 2])
        except ValueError:
            return xs[i], ys[i]
        if xs[i - 1] <= x_min <= xs[i + 1]:
            return x_min, y_min
    return xs[i], ys[i]


def separation_grid(
    start: float = 5.5, stop: float = 9.0, step: float = 0.1
) -> List[float]:
    """The scanned separations, rounded so they make clean JSON keys."""
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 2) for i in range(n)]


def fit_in_plane_constant(
    in_plane_curve: Optional[Dict[float, float]]
) -> Tuple[Optional[float], Optional[str]]:
    """In-plane lattice constant from the 3-point scaled-cell scan.

    Returns ``(value, note)``. The value is ``None`` — with a note saying why —
    whenever the fit is not trustworthy: fewer than three points, a downward
    parabola, or a vertex extrapolated *outside* the sampled strain window.
    That last case is the one that bites: three monotonically falling points
    still admit a positive-curvature fit whose vertex is far outside the scan,
    and reporting that number as a lattice constant would be fiction. Widen
    ``--scale-step`` (or re-centre ``--lattice-a``) and re-run.
    """
    if not in_plane_curve:
        return None, None
    xs = sorted(in_plane_curve)
    ys = [in_plane_curve[x] for x in xs]
    try:
        x_min, _ = quadratic_minimum(xs, ys)
    except ValueError as exc:
        return None, f"no in-plane minimum: {exc}"
    if not xs[0] <= x_min <= xs[-1]:
        return None, (
            f"fitted vertex {x_min:.4f} A lies outside the sampled window "
            f"[{xs[0]:.4f}, {xs[-1]:.4f}] A — the scan does not bracket the "
            "minimum; widen --scale-step or re-centre --lattice-a"
        )
    return x_min, None


def binding_energy_per_area(
    e_min: float, e_reference: float, area: float,
    n_atoms: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """Binding energy relative to the widest sampled separation.

    Positive means bound. ``area`` is the in-plane cell cross-section (one
    interface), so the result is per interface, not per cell surface.
    """
    if area <= 0:
        raise ValueError("area must be positive")
    delta = float(e_reference) - float(e_min)
    return {
        "eV_per_A2": delta / area,
        "J_per_m2": delta / area * EV_PER_A2_TO_J_PER_M2,
        "total_eV": delta,
        "meV_per_atom": (1000.0 * delta / n_atoms) if n_atoms else None,
    }


# ---------------------------------------------------------------------------
# (d) JSON assembly — pure
# ---------------------------------------------------------------------------

def assemble_results(
    *,
    framework: str,
    model: Optional[str],
    lmp: Optional[str],
    d3: bool,
    n_atoms: int,
    area: float,
    curve: Dict[float, float],
    in_plane_curve: Optional[Dict[float, float]] = None,
    gpu_memory: Optional[Dict[float, Optional[int]]] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build the result document written to ``--out``.

    Keys of the two curves are stringified floats so the JSON round-trips
    losslessly (JSON object keys are strings either way; doing it here keeps
    the formatting stable and diffable between runs).
    """
    eq_sep, eq_energy = equilibrium_from_curve(curve)
    ref_sep = max(curve)
    ref_energy = curve[ref_sep]

    lat_c_in_plane, lat_c_note = fit_in_plane_constant(in_plane_curve)

    used = [v for v in (gpu_memory or {}).values() if v is not None]
    results: Dict[str, object] = {
        "script": Path(__file__).name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framework": framework,
        "model": model,
        "lmp": lmp,
        "dispersion": "d3" if d3 else "none",
        "n_atoms": n_atoms,
        "in_plane_area_A2": area,
        "binding_curve_eV": {f"{s:.2f}": curve[s] for s in sorted(curve)},
        "equilibrium_separation_A": eq_sep,
        "equilibrium_energy_eV": eq_energy,
        "reference_separation_A": ref_sep,
        "reference_energy_eV": ref_energy,
        "binding_energy_per_area": binding_energy_per_area(
            eq_energy, ref_energy, area, n_atoms
        ),
        # The equilibrium separation IS the config's `lat_c` (interlayer
        # distance); the in-plane constant is the `a` the sheet is built at.
        # Spec 3.3: set both from here or risk the dome-plow failure mode.
        "lat_c_recommendation_A": eq_sep,
        "lat_c_in_plane_A": lat_c_in_plane,
        "lat_c_in_plane_note": lat_c_note,
        "in_plane_scan_eV": (
            {f"{a:.4f}": in_plane_curve[a] for a in sorted(in_plane_curve)}
            if in_plane_curve else None
        ),
        "vram": {
            "available": bool(used),
            "peak_MiB": max(used) if used else None,
            "n_atoms": n_atoms,
            "per_point_MiB": (
                {f"{s:.2f}": gpu_memory[s] for s in sorted(gpu_memory)}
                if gpu_memory else None
            ),
            "note": (
                "whole-device nvidia-smi memory.used sampled while LAMMPS ran; "
                "includes anything else resident on the GPU"
            ),
        },
    }
    if extra:
        results.update(extra)
    return results


# ---------------------------------------------------------------------------
# geometry — pure (needs ASE)
# ---------------------------------------------------------------------------

def build_bilayer(
    separation: float = INTERLAYER,
    a: float = LATTICE_A,
    *,
    n_layers: int = 2,
    thickness: float = THICKNESS,
    vacuum: float = VACUUM,
):
    """Build an n-layer 2H-MoS2 slab with per-layer atom types.

    Mirrors ``scripts/milestone0/make_geometries.py``: hexagonal primitive ->
    orthogonal 2x1 supercell (avoids LAMMPS' triclinic-skew limit), 2H (2/3,
    1/3) stacking shift, symmetric vacuum, in-plane wrap only.

    Args:
        separation: interlayer distance in A (the scanned variable).
        a: in-plane lattice constant in A (the scaled-cell variable).
    """
    from ase import Atoms
    from ase.build import make_supercell, mx2

    prim = mx2(formula="MoS2", kind="2H", a=a, thickness=thickness,
               size=(1, 1, 1), vacuum=None)
    shift = (STACK_SHIFT[0] * np.array(prim.cell[0])
             + STACK_SHIFT[1] * np.array(prim.cell[1]))
    prim.cell[2] = [0.0, 0.0, 1.0]   # make_supercell needs a non-singular cell
    unit = make_supercell(prim, [[1, 0, 0], [1, 2, 0], [0, 0, 1]]).repeat(
        (NX, NY, 1)
    )

    stack = Atoms(cell=np.array(unit.cell), pbc=True)
    for i in range(n_layers):
        layer = unit.copy()
        layer.translate(i * shift + np.array([0.0, 0.0, i * separation]))
        stack += layer

    slab_height = (n_layers - 1) * separation + thickness
    stack.cell[2] = [0.0, 0.0, slab_height + 2.0 * vacuum]
    stack.pbc = (True, True, True)
    stack.center(axis=2)
    stack.wrap(pbc=(True, True, False))
    positions = np.round(stack.positions, 10)
    stack.positions = np.where(np.abs(positions) < 1e-10, 0.0, positions)

    per_layer = len(stack) // n_layers
    symbols = stack.get_chemical_symbols()
    stack.arrays["type"] = np.array(
        [2 * (i // per_layer) + (1 if symbols[i] == "Mo" else 2)
         for i in range(len(stack))],
        dtype=int,
    )
    return stack


def in_plane_area(atoms) -> float:
    """Cross-sectional area of the (orthogonal) cell, A^2 — one interface."""
    cell = np.array(atoms.cell)
    return float(abs(np.cross(cell[0], cell[1])[2]))


def stage_potential(path: str, workdir: Path) -> str:
    """Link (or copy) a model/potential into ``workdir``; return its basename.

    LAMMPS splits deck lines on whitespace, so a ``pair_coeff`` naming a file
    under a path with a space in it — "ML Analysis/..." — fails with
    "Incorrect args for pair coefficients". Staging the file next to the deck
    and referring to it by basename sidesteps that, keeps the run directory
    self-contained, and (via symlink) does not duplicate multi-GB models.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"potential/model file not found: {path}")
    target = workdir / source.name
    if target.resolve() == source:
        return source.name
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)
    return source.name


def write_data_file(atoms, path: Path) -> Path:
    """Write an ``atom_style atomic`` data file with an explicit Masses block."""
    from ase.io import write
    write(str(path), atoms, format="lammps-data", atom_style="atomic",
          masses=True)
    return path


# ---------------------------------------------------------------------------
# thin impure wrappers: LAMMPS subprocess + nvidia-smi
# ---------------------------------------------------------------------------

def sample_gpu_memory() -> Optional[int]:
    """Return the busiest GPU's used memory in MiB, or None without nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    values = [int(tok.strip()) for tok in out.split() if tok.strip().isdigit()]
    return max(values) if values else None


# symmetrix/mace corrupts the heap in teardown and glibc aborts *after* the
# single-point energy has already been written to the log. The abort point is
# NOT deterministic -- usually post-run, occasionally early enough to lose the
# result -- so we accept a non-zero exit only when the energy actually parsed,
# and retry otherwise. Measured 2026-08-24 on LAMMPS 10Sep2025 + symmetrix,
# both the CPU and the Kokkos/GPU path.
HEAP_ABORT_SIGNATURES = (
    "corrupted size vs. prev_size",
    "free(): invalid next size",
    "Fatal glibc error",
    "malloc(): ",
)


def _log_has_energy(log_text: str) -> bool:
    """True if a thermo energy survived in the log despite a crash."""
    try:
        parse_potential_energy(log_text)
        return True
    except Exception:
        return False


def run_lammps(
    lmp: str,
    deck: Path,
    workdir: Path,
    flags: Sequence[str] = (),
    env: Optional[Dict[str, str]] = None,
    poll_gpu: bool = True,
) -> Tuple[str, Optional[int]]:
    """Run one LAMMPS deck; return ``(log_text, peak_gpu_MiB)``.

    VRAM is polled *during* the run — sampling after the process exits would
    only ever see the memory already freed.
    """
    cmd = [lmp, *flags, "-in", str(deck)]
    proc = subprocess.Popen(
        cmd, cwd=str(workdir), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
        env={**os.environ, **(env or {})},
    )
    peak: Optional[int] = None
    if poll_gpu:
        while proc.poll() is None:
            sample = sample_gpu_memory()
            if sample is not None:
                peak = sample if peak is None else max(peak, sample)
            time.sleep(GPU_POLL_SECONDS)
    log = proc.communicate()[0] or ""
    if proc.returncode != 0:
        heap_abort = any(sig in log for sig in HEAP_ABORT_SIGNATURES)
        if heap_abort and _log_has_energy(log):
            # Physics finished; only the destructor blew up. Keep the result
            # but make the compromise visible rather than silent.
            print(
                f"WARNING: {deck.name} exited {proc.returncode} via a known "
                f"symmetrix heap abort AFTER producing its energy; result "
                f"kept. See HEAP_ABORT_SIGNATURES.",
                file=sys.stderr,
            )
            return log, peak
        raise RuntimeError(
            f"LAMMPS exited {proc.returncode} for {deck.name}\n"
            f"command: {' '.join(cmd)}\n{log[-4000:]}"
        )
    return log, peak


def resolve_launcher(framework: str) -> Tuple[List[str], List[str]]:
    """Return ``(lmp_flags, env_warnings)`` from the registry's LauncherSpec.

    The flags (MACE's ``-k on g 1 -sf kk``) and the DeePMD
    ``LAMMPS_PLUGIN_PATH`` export live in ``frictionsim2d/core/mlip.py``; this script
    never re-derives them. Env lines are *shell* lines, so they are reported
    rather than executed — export them in the calling shell (see
    ``documentation/milestone0.md``).
    """
    if not is_mlip(framework):
        return [], []
    spec = get_mlip(framework)
    flags = spec.launcher.lmp_flags.split()
    warnings: List[str] = []
    if framework == "deepmd" and not os.environ.get("LAMMPS_PLUGIN_PATH"):
        warnings.append(
            "LAMMPS_PLUGIN_PATH is not set; pair_style deepmd will be an "
            "unknown style. Export it first:\n  "
            + "\n  ".join(spec.launcher.env_lines)
        )
    return flags, warnings


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _single_point(
    args, workdir: Path, atoms, tag: str, d3: Optional[D3Settings],
    flags: Sequence[str],
) -> Tuple[float, Optional[int]]:
    """Write geometry + deck for one configuration and evaluate it."""
    data = write_data_file(atoms, workdir / f"bilayer_{tag}.data")
    deck_text = render_deck(
        framework=args.framework,
        data_file=data.name,
        model_path=stage_potential(args.model, workdir) if args.model else None,
        sw_path=stage_potential(args.pot, workdir) if args.pot else None,
        d3=d3,
        lj_epsilon=args.lj_epsilon,
        lj_sigma=args.lj_sigma,
        lj_cutoff=args.lj_cutoff,
    )
    deck = workdir / f"in.{tag}"
    deck.write_text(deck_text, encoding="utf-8")
    if args.dry_run:
        print(f"--- {deck} ---\n{deck_text}", end="")
        return float("nan"), None
    log, peak = run_lammps(args.lmp, deck, workdir, flags=flags)
    (workdir / f"log.{tag}").write_text(log, encoding="utf-8")
    return parse_potential_energy(log), peak


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--framework", required=True, choices=FRAMEWORKS)
    p.add_argument("--model",
                   help="MLIP model file; accepted extensions come from the "
                        "registry so they cannot drift: "
                        + "; ".join(
                            f"{name} {'/'.join(spec.file_extensions)}"
                            for name, spec in sorted(MLIP_REGISTRY.items())))
    p.add_argument("--pot", help="SW potential file (--framework sw)")
    p.add_argument("--lmp", default="lmp", help="LAMMPS binary to invoke")
    p.add_argument("--lmp-flags",
                   help="override the registry LauncherSpec flags (pass '' for "
                        "none). Needed to force symmetrix/MACE onto its CPU "
                        "path: the Kokkos/CUDA build returns wrong energies "
                        "(-5.20 vs -7.14 eV/atom) and aborts ~80%% of runs.")
    p.add_argument("--d3", action="store_true",
                   help="overlay dispersion/d3 (full-cell MLIP only)")
    p.add_argument("--out", help="output JSON (default: "
                                 "mlip_bilayer_benchmark_<framework>[_d3].json)")
    p.add_argument("--workdir", help="where decks/logs land "
                                     "(default: a bench_<framework>/ dir)")
    p.add_argument("--min-sep", type=float, default=5.5)
    p.add_argument("--max-sep", type=float, default=9.0)
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--lattice-a", type=float, default=LATTICE_A,
                   help="in-plane lattice constant the scan is run at (A)")
    p.add_argument("--scale-step", type=float, default=0.02,
                   help="fractional +/- in-plane strain for the 3-point "
                        "lattice-constant fit")
    p.add_argument("--no-in-plane", action="store_true",
                   help="skip the scaled-cell lattice-constant fit")
    p.add_argument("--lj-epsilon", type=float, default=0.0224)
    p.add_argument("--lj-sigma", type=float, default=3.13)
    p.add_argument("--lj-cutoff", type=float, default=11.0)
    p.add_argument("--d3-damping", default="bj")
    p.add_argument("--d3-functional", default="pbe")
    p.add_argument("--d3-cutoff", type=float, default=16.0)
    p.add_argument("--d3-cn-cutoff", type=float, default=8.0)
    p.add_argument("--dry-run", action="store_true",
                   help="write geometries and decks, print them, run nothing")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if is_mlip(args.framework) and not args.model:
        build_parser().error(f"--framework {args.framework} requires --model")
    if args.framework == "sw" and not args.pot:
        build_parser().error("--framework sw requires --pot")
    if args.d3 and not is_mlip(args.framework):
        build_parser().error("--d3 requires an MLIP framework (full-cell only)")

    d3 = D3Settings(
        d3_damping=args.d3_damping, d3_functional=args.d3_functional,
        d3_cutoff=args.d3_cutoff, d3_cn_cutoff=args.d3_cn_cutoff,
    ) if args.d3 else None

    flags, warnings = resolve_launcher(args.framework)
    if args.lmp_flags is not None:
        flags = args.lmp_flags.split()
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    workdir = Path(args.workdir or f"bench_{args.framework}"
                                   f"{'_d3' if args.d3 else ''}").resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    seps = separation_grid(args.min_sep, args.max_sep, args.step)
    curve: Dict[float, float] = {}
    gpu: Dict[float, Optional[int]] = {}
    n_atoms = 0
    area = 0.0

    print(f"[{args.framework}] {len(seps)} rigid single points, "
          f"{seps[0]}-{seps[-1]} A, workdir {workdir}")
    for sep in seps:
        atoms = build_bilayer(separation=sep, a=args.lattice_a)
        n_atoms, area = len(atoms), in_plane_area(atoms)
        energy, peak = _single_point(args, workdir, atoms, f"sep{sep:.2f}",
                                     d3, flags)
        curve[sep] = energy
        gpu[sep] = peak
        if not args.dry_run:
            print(f"  d = {sep:5.2f} A   PotEng = {energy:14.6f} eV"
                  + (f"   VRAM {peak} MiB" if peak is not None else ""))

    if args.dry_run:
        print("dry run: no LAMMPS invoked, no JSON written")
        return 0

    eq_sep, _ = equilibrium_from_curve(curve)
    in_plane: Optional[Dict[float, float]] = None
    if not args.no_in_plane:
        in_plane = {}
        for scale in (1.0 - args.scale_step, 1.0, 1.0 + args.scale_step):
            a_val = args.lattice_a * scale
            atoms = build_bilayer(separation=eq_sep, a=a_val)
            energy, peak = _single_point(args, workdir, atoms,
                                         f"a{a_val:.4f}", d3, flags)
            in_plane[a_val] = energy
            print(f"  a = {a_val:6.4f} A   PotEng = {energy:14.6f} eV")

    results = assemble_results(
        framework=args.framework, model=args.model, lmp=args.lmp, d3=args.d3,
        n_atoms=n_atoms, area=area, curve=curve, in_plane_curve=in_plane,
        gpu_memory=gpu,
        extra={"lammps_flags": " ".join(flags), "workdir": str(workdir),
               "lattice_a_scanned_A": args.lattice_a},
    )
    out = Path(args.out or f"mlip_bilayer_benchmark_{args.framework}"
                           f"{'_d3' if args.d3 else ''}.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\nequilibrium separation : {results['equilibrium_separation_A']:.4f} A"
          "   (use as lat_c)")
    if results["lat_c_in_plane_A"] is not None:
        print(f"in-plane lattice const : {results['lat_c_in_plane_A']:.4f} A")
    elif results["lat_c_in_plane_note"]:
        print(f"in-plane lattice const : unresolved — {results['lat_c_in_plane_note']}")
    binding = results["binding_energy_per_area"]
    print(f"binding energy         : {binding['eV_per_A2']:.6f} eV/A^2 "
          f"({binding['J_per_m2']:.4f} J/m^2)")
    print(f"peak VRAM              : {results['vram']['peak_MiB']} MiB "
          f"for {n_atoms} atoms")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
