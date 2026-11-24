"""Indent-simulation delivery: penetration (hold) and adhesion (retract) in one pass.

Each AFM indent run (:mod:`frictionsim2d.builders.afm.indent`) writes, per
``(material, layer, load)``, a hold file ``results/friction_f<load>_a0_s<speed>_layer<N>.txt``
(read by :class:`~frictionsim2d.postprocessing.read_data.DataReader`) and a retract curve
``results/indent_f<load>N_l<layer>.csv`` (``step, z_tip, fz_tip, fx_tip, fy_tip, pe``).

:func:`run` reduces a run root to three outputs:

    - ``<out_dir>/output_indent_f<load>_s<seed>.json``: the penetration, one file per
      (load, seed batch), in the campaign schema.
    - ``<data_dir>/indent_features.csv``: one row per ``(material, layer, load)`` of
      adhesion descriptors (:func:`retract_features`), merge key ``material``.
    - ``<out_dir>/indent_manifest.csv``: one row per condition with the velocity
      ``seed``, ``hold_steps``, ``z_step``, ``n_steps`` and a best-effort ``wall_time``.

The tidied retract curves go to ``<out_dir>/indent/<material>_l<layer>[_f<load>].csv``.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.utils.tokens import material_id
from ..core.observables import EV_A_TO_NN
from ..run.tree import discover_runs

logger = logging.getLogger(__name__)

#: Top-level run directory name for indent runs (matches the model name).
INDENT_TOPDIR = "indent"

#: Retract CSV filename, e.g. ``indent_f10N_l2.csv``.
_CSV_RE = re.compile(r"indent_f(?P<load>[0-9.]+)N_l(?P<layer>\d+)\.csv$")

_WALL_TIME_RE = re.compile(r"Total wall time:\s*([0-9:]+)")
#: Log-like filenames worth scanning for a wall time (avoid huge trajectories).
_LOG_GLOBS = ("log.lammps", "*.log", "logs/*", "*.out", "*.o*")


def _load_token(load_val: float) -> str:
    """Filename token for a load: ``10.0`` -> ``10``, ``10.5`` -> ``10.5``."""
    return str(int(load_val)) if float(load_val).is_integer() else str(load_val)


def _parse_wall_seconds(value: str) -> Optional[int]:
    """Parse a LAMMPS ``H:MM:SS`` (or ``S``) wall-time string to seconds."""
    try:
        nums = [int(p) for p in value.split(":")]
    except ValueError:
        return None
    seconds = 0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


def _run_wall_seconds(run_dir: Path) -> Optional[int]:
    """Best-effort: last ``Total wall time`` found in a log under ``run_dir``."""
    seen: set = set()
    for pattern in _LOG_GLOBS:
        for path in run_dir.rglob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            matches = _WALL_TIME_RE.findall(text)
            if matches:
                return _parse_wall_seconds(matches[-1])
    return None


def load_indent_meta(run_root: Path) -> Dict[str, Any]:
    """Collect the per-run ``indent_meta.json`` records written by the builder.

    Returns a dict keyed by the underscored material id (matching both the
    DataReader tree and the feature-table merge id):
        - ``base_seed[material]`` — campaign base seed (names the per-load JSON),
        - ``seed[(material, layer)]`` — the fresh derived velocity seed,
        - ``hold_steps[material]`` — hold length in steps,
        - ``z_step[material]`` / ``n_steps[material]`` — retract discretization,
        - ``run_dir[material]`` — the simulation directory (for wall-time lookup).
    """
    base_seed: Dict[str, int] = {}
    seed: Dict[Tuple[str, int], int] = {}
    hold_steps: Dict[str, int] = {}
    z_step: Dict[str, float] = {}
    n_steps: Dict[str, int] = {}
    run_dir: Dict[str, Path] = {}

    for locator in discover_runs(Path(run_root)):
        if locator.tokens.family != 'indent':
            continue
        meta_path = locator.provenance_dir / 'indent_meta.json'
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable indent meta %s: %s", meta_path, exc)
            continue
        mat = material_id(str(meta.get("material", "")))
        if not mat:
            continue
        base_seed[mat] = int(meta.get("base_seed", 0))
        hold_steps[mat] = int(meta.get("hold_steps", 0))
        z_step[mat] = float(meta.get("z_step", 0.0))
        n_steps[mat] = int(meta.get("n_steps", 0))
        # Was meta_path.parent.parent; locator.run_dir is the same directory for
        # a sheet-like run and the correct L{n} level for an indent one --
        # _run_wall_seconds rglobs the logs under it either way.
        run_dir[mat] = locator.run_dir
        for layer_str, layer_seed in (meta.get("layers") or {}).items():
            try:
                seed[(mat, int(layer_str))] = int(layer_seed)
            except (TypeError, ValueError):
                continue
    return {"base_seed": base_seed, "seed": seed, "hold_steps": hold_steps,
            "z_step": z_step, "n_steps": n_steps, "run_dir": run_dir}


# =============================================================================
# Penetration (hold) — per-load campaign-schema JSON
# =============================================================================

def split_by_load(runs: List['RunFrames'],
                  base_seed: Dict[str, int]
                  ) -> Dict[Tuple[str, int], List['RunFrames']]:
    """Regroup runs into ``(load_key, seed) -> [RunFrames]``.

    With a shared ``[indent] seed`` every material carries the same base seed,
    so this yields one file per load; ad-hoc unseeded runs fall into
    per-material seed batches (still correct, just more files).
    """
    per_file: Dict[Tuple[str, int], List['RunFrames']] = {}
    for frames in runs:
        seed = base_seed.get(frames.material, 0)
        per_file.setdefault((f'f{frames.load}', seed), []).append(frames)
    return per_file


def iter_hold_conditions(runs: List['RunFrames']):
    """Yield ``(material, layer, load)`` for every hold run."""
    for frames in runs:
        yield (frames.material, int(frames.tokens.layers or 1),
               float(frames.load))


def _deliver_penetration(run_root: Path, out_dir: Path,
                         meta: Dict[str, Any]) -> Tuple[Dict[str, Path], set]:
    """Write the per-load penetration JSON; return (written, hold-condition set).

    The document is the v2 export (`read_data.DataReader.export_payload`)
    restricted to one load, plus the two indent keys. The old nested
    ``results`` block was `DataReader.full_data_nested`, which this rework
    deletes -- one export schema, not two.
    """
    from .read_data import DataReader, _NpEncoder  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    from .run_status import DELIVERED_STATUSES  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    reader = DataReader(results_dir=str(run_root))
    # Only delivered statuses: a halted or failed_integrity hold still has
    # frames, but they describe a blow-up or a broken sheet, not an indent.
    indent_runs = [r for r in reader.runs if r.family == 'indent'
                   and r.status in DELIVERED_STATUSES]
    if not indent_runs:
        logger.info("No indent hold results found under %s", run_root)
        return {}, set()

    written: Dict[str, Path] = {}
    for (load_key, seed), runs in sorted(
            split_by_load(indent_runs, meta["base_seed"]).items()):
        load_val = float(load_key[1:])
        payload = reader.export_payload(runs[0].size, runs)
        payload["metadata"]["indent_load_nN"] = load_val
        payload["metadata"]["indent_base_seed"] = seed
        name = f"output_indent_f{_load_token(load_val)}_s{seed}.json"
        path = out_dir / name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, cls=_NpEncoder)
        written[name] = path
        logger.info("Wrote %d run(s) at load %g nN -> %s",
                    len(runs), load_val, path)

    return written, set(iter_hold_conditions(indent_runs))


# =============================================================================
# Adhesion (retract) — features + tidied curves
# =============================================================================

def retract_features(df: pd.DataFrame) -> Dict[str, float]:
    """Reduce one retract force–distance curve to adhesion descriptors.

    ``df`` carries ``z_tip`` (Å) and ``fz_tip`` (nN); rows are re-sorted by
    ascending ``z_tip`` (the retract direction) so the descriptors are robust to
    the row order. ``fz_tip`` is + repulsive / - adhesive.
    """
    z = df["z_tip"].to_numpy(dtype=float)
    fz = df["fz_tip"].to_numpy(dtype=float)
    order = np.argsort(z)
    z, fz = z[order], fz[order]

    out: Dict[str, float] = {"n_points": int(z.size)}
    out["z_span"] = float(z[-1] - z[0]) if z.size > 1 else 0.0

    f_min = float(np.min(fz))
    i_min = int(np.argmin(fz))
    out["f_adh_min"] = f_min                        # signed (nN); the tensile trough
    out["pulloff_force"] = float(max(0.0, -f_min))  # peak adhesive force (nN)
    out["snapoff_z"] = float(z[i_min])

    # Work of adhesion: area under the attractive (fz < 0) branch, ∫ (-fz) dz.
    tensile = np.clip(-fz, 0.0, None)
    w_nn_a = float((np.diff(z) * (tensile[1:] + tensile[:-1]) / 2.0).sum()) if z.size > 1 else 0.0
    out["work_of_adhesion_nnA"] = w_nn_a
    out["work_of_adhesion"] = w_nn_a / EV_A_TO_NN     # eV (1 nN·Å = 1/EV_A_TO_NN eV)

    # Contact stiffness: slope of the *leading repulsive wall* only — from the
    # loaded contact (fz > 0) up to and including the first zero crossing. Fitting
    # every fz > 0 point would fold in the near-zero tail past detachment and wash
    # the steep wall out; the wall is the near-contact segment before fz goes tensile.
    if fz[0] > 0.0:
        below = np.nonzero(fz <= 0.0)[0]
        end = max(int(below[0]) + 1 if below.size else int(fz.size), 2)
        z_wall, f_wall = z[:end], fz[:end]
        if float(np.ptp(z_wall)) > 0.0:
            out["contact_stiffness"] = -float(np.polyfit(z_wall, f_wall, 1)[0])  # nN/Å > 0
        else:
            out["contact_stiffness"] = float("nan")
    else:
        # No repulsive contact at the start (curve did not begin loaded).
        out["contact_stiffness"] = float("nan")

    # Did the curve reach detachment (fz returns toward zero past the trough)?
    reached_min_before_end = i_min < z.size - 1
    tail_small = out["pulloff_force"] == 0.0 or abs(fz[-1]) <= 0.1 * out["pulloff_force"]
    out["detached"] = float(bool(reached_min_before_end and tail_small))
    return out


def discover_curves(run_root: Path) -> List[Tuple[str, int, float, Path]]:
    """Find ``(material, layer, load, csv_path)`` for every retract curve.

    Walks with `discover_runs`: a run is one `provenance.json`, and the
    material comes from the parsed tokens. The old rglob took the directory
    one level below `indent/` as the material, so a stray CSV under a
    run-shaped path was picked up and filed under whatever sat in that slot.
    """
    found: List[Tuple[str, int, float, Path]] = []
    for locator in discover_runs(Path(run_root)):
        if locator.tokens.family != 'indent':
            continue
        material = (material_id(locator.tokens.materials[0])
                    if locator.tokens.materials else '')
        if not material:
            continue
        for csv_path in sorted(locator.results_dir.glob("indent_f*N_l*.csv")):
            m = _CSV_RE.search(csv_path.name)
            if m is None:
                continue
            found.append((material, int(m.group("layer")),
                          float(m.group("load")), csv_path))
    return sorted(found)


def _deliver_adhesion(run_root: Path, out_dir: Path, data_dir: Path,
                      curves: List[Tuple[str, int, float, Path]]) -> Dict[str, Path]:
    """Write indent_features.csv + tidied retract curves from the retract data."""
    (out_dir / INDENT_TOPDIR).mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    # Whether more than one load per (material, layer) is present decides whether the
    # tidied-curve filename needs a load token to stay unique.
    per_ml: Dict[Tuple[str, int], int] = {}
    for material, layer, _load, _p in curves:
        per_ml[(material, layer)] = per_ml.get((material, layer), 0) + 1

    feature_rows: List[Dict[str, Any]] = []
    for material, layer, load_val, csv_path in curves:
        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Skipping unreadable retract curve %s: %s", csv_path, exc)
            continue
        if len(df) < 3 or not {"z_tip", "fz_tip"} <= set(df.columns):
            logger.warning("Skipping degenerate retract curve %s (%d rows)", csv_path, len(df))
            continue

        feats = retract_features(df)
        if not feats.get("detached"):
            logger.warning(
                "Retract curve %s did not reach detachment (min at last steps) — "
                "consider a larger [indent] n_steps/z_step.", csv_path,
            )
        feature_rows.append({"material": material, "layer": layer, "load": load_val, **feats})

        # Tidied per-condition curve (recipe hand-back).
        stem = f"{material}_l{layer}"
        if per_ml[(material, layer)] > 1:
            stem += f"_f{_load_token(load_val)}"
        (df.to_csv(out_dir / INDENT_TOPDIR / f"{stem}.csv", index=False))

    if not feature_rows:
        logger.info("No usable indent retract curves under %s", run_root)
        return {}

    feat_df = pd.DataFrame(feature_rows).sort_values(
        ["material", "layer", "load"]).reset_index(drop=True)
    feat_path = data_dir / "indent_features.csv"
    feat_df.to_csv(feat_path, index=False)
    written["indent_features.csv"] = feat_path
    logger.info("Wrote %d indent feature row(s) -> %s", len(feat_df), feat_path)
    return written


# =============================================================================
# Combined manifest + entry point
# =============================================================================

def _write_manifest(out_dir: Path, conditions: List[Tuple[str, int, float]],
                    meta: Dict[str, Any]) -> Path:
    """One row per condition with the seed + hold/retract discretization."""
    wall_cache: Dict[str, Optional[int]] = {}
    rows: List[Dict[str, Any]] = []
    for material, layer, load_val in sorted(set(conditions)):
        if material not in wall_cache:
            rdir = meta["run_dir"].get(material)
            wall_cache[material] = _run_wall_seconds(rdir) if rdir else None
        rows.append({
            "material": material,
            "layer": layer,
            "load": load_val,
            "seed": meta["seed"].get((material, layer), ""),
            "hold_steps": meta["hold_steps"].get(material, ""),
            "z_step": meta["z_step"].get(material, ""),
            "n_steps": meta["n_steps"].get(material, ""),
            "wall_time": wall_cache[material] if wall_cache[material] is not None else "",
        })

    manifest_path = out_dir / "indent_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["material", "layer", "load", "seed",
                            "hold_steps", "z_step", "n_steps", "wall_time"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote indent manifest (%d rows) -> %s", len(rows), manifest_path)
    return manifest_path


def run(run_root: Path, out_dir: Path, data_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Reduce an indent run to penetration JSON + adhesion features + a manifest.

    Args:
        run_root: The ``simulation_YYYYMMDD_HHMMSS`` root (or its ``indent/``
            subdirectory) containing the hold + retract outputs.
        out_dir: Destination for ``output_indent_f*_s*.json``,
            ``indent/<material>_l<layer>.csv`` and ``indent_manifest.csv``.
        data_dir: Destination for ``indent_features.csv`` (default: ``out_dir``).

    Returns:
        Mapping of output name -> written path (empty if no indent data was found).
    """
    run_root = Path(run_root)
    out_dir = Path(out_dir)
    data_dir = Path(data_dir) if data_dir is not None else out_dir
    meta = load_indent_meta(run_root)
    curves = discover_curves(run_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    pen_written, hold_conditions = _deliver_penetration(run_root, out_dir, meta)
    written.update(pen_written)

    written.update(_deliver_adhesion(run_root, out_dir, data_dir, curves))

    conditions = list(hold_conditions) + [(m, l, ld) for m, l, ld, _p in curves]
    if not conditions:
        logger.info("No indent results found under %s", run_root)
        return written
    written["indent_manifest.csv"] = _write_manifest(out_dir, conditions, meta)

    return written
