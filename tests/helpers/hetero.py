"""Hetero (D1/D2) config dicts and LAMMPS-run helpers shared by the hetero test files.

MAT/POT are cwd-relative (``examples/...``) — a pre-existing fragility recorded as a
Phase 2 finding; kept verbatim here so this commit stays a pure move.
"""
import re
import subprocess
from pathlib import Path

MAT, POT = "frictionsim2d/data/materials", "frictionsim2d/data/potentials/sw"

# --- bodies moved verbatim from test_hetero_afm.py and test_hetero_slide.py ---


def _afm_hetero_raw():
    return {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "tip": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "r": 10.0},
        "sub": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "amorph": "a", "thickness": 12.0},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif", "pot_path": f"{POT}/sw_lammps/t-WS2.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
    }


def _afm_flake_raw():
    """A homogeneous AFM config WITH a [Flake] on a [2D] MoS2 sheet + [sub] + [tip].

    Mirrors ``examples/afm_flake_config.ini`` (a finite MoS2 hexagon between the
    Si tip and the supported MoS2 sheet on an amorphous-Si substrate), sized with
    the D2 capstone's runtime-bounding knobs (tip.r=5, temp=1, force=10) so the
    real-LAMMPS approach+indent settles quickly.
    """
    return {
        "general": {"temp": 1.0, "force": 10.0, "scan_speed": 2.0},
        "tip": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "r": 5.0},
        "sub": {"mat": "Si", "cif_path": f"{MAT}/Si.cif", "pot_path": f"{POT}/Si.sw",
                "pot_type": "sw", "amorph": "a", "thickness": 12.0},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
               "pot_type": "sw", "x": 40.0, "y": 40.0, "layers": [1]},
        "flake": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                  "pot_type": "sw", "shape": "hexagon", "edge_length": 10.0,
                  "corner_radius": 5.0, "corner_spring_k": 9.0},
    }


#: Every path a LAMMPS script opens for reading, as written into the script.
_READ_PATH_RE = re.compile(r'^(?:read_data|include)\s+(\S+)', re.M)


def _referenced_input_paths(script_text):
    return _READ_PATH_RE.findall(script_text)


def _staged_potential_paths(settings_text):
    """The potential files ``pair_coeff`` lines point at, as written.

    A ``pair_coeff`` line mixes path tokens with pair-style names that also
    contain a slash (``lj/cut``); only the former carry a file extension.
    """
    return [
        token
        for line in settings_text.splitlines()
        if line.strip().startswith('pair_coeff')
        for token in line.split()
        if '/' in token and Path(token).suffix
    ]


def _run_lammps_slide(run_dir, slide_in_name, timeout=180):
    """Run ``lmp_serial`` on the rendered hetero slide and return
    ``(completed_process, combined_output)``.

    ``cwd`` MUST be ``run_dir``: every path baked into ``slide.in`` is
    RUN-DIR-RELATIVE (``hetero.data``, ``lammps/system.in.settings``, the
    settings' ``provenance/potentials/...`` prefix, ``results/...``), so they
    only resolve when LAMMPS is launched from the run dir. ``combined_output``
    is stdout + stderr + the LAMMPS ``log.lammps`` (if written), so callers can
    grep it for errors / "Lost atoms".
    """
    run_dir = Path(run_dir)
    proc = subprocess.run(
        ["lmp_serial", "-in", str(run_dir / "lammps" / slide_in_name)],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    log = run_dir / "log.lammps"
    if log.exists():
        combined += "\n" + log.read_text()
    return proc, combined


def _read_friction_records(friction_path):
    """Parse a ``fix ave/time`` results file into ``(column_names, rows)``.

    The second comment line names the columns
    (``# TimeStep v_xfrict ... v_comx_top v_comy_top ...``); data rows are the
    non-comment lines. Column order is taken from that header (which mirrors the
    rendered ``fix fc_win ... ave/time`` output list -- Task 15 renamed the fix
    from ``fc_ave`` to ``fc_win``), so the parse survives any
    reordering of the emitted variables. ``column_names`` includes the leading
    ``TimeStep`` and is index-aligned with every data row.
    """
    cols = None
    rows = []
    for line in Path(friction_path).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            toks = s.lstrip("#").split()
            if toks and toks[0] == "TimeStep":
                cols = toks
            continue
        rows.append([float(x) for x in s.split()])
    return cols, rows
