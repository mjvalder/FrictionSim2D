"""Typed façade over :func:`frictionsim2d.run.simulate.run_simulations`, plus preview lookup.

``run_simulations`` is already the shared entry point the CLI uses, so this is
not a second implementation -- it is the same call with a result you can act on:
which members failed and why, rather than a success count.

:func:`discover_preview_target` lives here rather than in the GUI because it
encodes where a builder puts things, which is simulation knowledge.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .simulate import run_simulations

logger = logging.getLogger(__name__)

__all__ = [
    'GenerationFailure', 'GenerationResult', 'generate_package',
    'PreviewTarget', 'discover_preview_target', 'find_assembly_script',
]

#: Matches the box definition an assembly script opens with.
_REGION_BOX = re.compile(r'^\s*region\s+\S+\s+block\s', re.MULTILINE)

#: Directory level inserted by AFM-family builders, one per sheet layer count.
_LAYER_DIR = re.compile(r'^L(\d+)$')


@dataclass
class GenerationFailure:
    """One config that failed to build."""

    output_dir: Path
    config: Dict[str, Any]
    error: Exception

    def render(self) -> str:
        """One line naming the failed directory and the error."""
        return f"{self.output_dir}: {type(self.error).__name__}: {self.error}"


@dataclass
class GenerationResult:
    """Outcome of generating a simulation package."""

    created: List[Path] = field(default_factory=list)
    simulation_root: Optional[Path] = None
    configs: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[GenerationFailure] = field(default_factory=list)
    settings: Any = None

    @property
    def ok(self) -> bool:
        """True when something was built and nothing failed."""
        return bool(self.created) and not self.failures

    @property
    def requested(self) -> int:
        """Number of expanded configs the run set out to build."""
        return len(self.configs)

    def render(self) -> str:
        """Multi-line summary: count, output root, one line per failure."""
        lines = [f"{len(self.created)}/{self.requested} simulations generated"]
        if self.simulation_root:
            lines.append(f"output: {self.simulation_root}")
        lines.extend(f"failed -- {f.render()}" for f in self.failures)
        return "\n".join(lines)


def generate_package(config_file: Union[str, Path],
                     model: str,
                     *,
                     output_root: Union[str, Path],
                     settings_file: Optional[Union[str, Path]] = None,
                     generate_hpc: bool = False,
                     simulation_root_name: Optional[str] = None,
                     ensure_hpc_settings=None) -> GenerationResult:
    """Generate a simulation package and report per-member outcomes.

    Args:
        config_file: Path to the run config.
        model: ``afm`` | ``indent`` | ``pes_tip`` | ``sheetonsheet`` | ``pes_sheet``
            | ``quench`` (a ``[quench]`` config: writes a quench root, runs nothing).
        output_root: Directory to create the simulation root inside.
        settings_file: Optional settings.yaml override.
        generate_hpc: Also emit HPC job scripts.
        simulation_root_name: Explicit root directory name. Without it the name
            is a timestamp to the second, so two runs in the same second merge.
        ensure_hpc_settings: Optional callable applied to the HPC settings.

    Returns:
        A :class:`GenerationResult`.
    """
    failures: List[GenerationFailure] = []

    def record(output_dir: Path, config: Dict[str, Any], error: Exception) -> None:
        failures.append(GenerationFailure(output_dir, config, error))

    created, simulation_root, configs, settings = run_simulations(
        config_file=str(config_file),
        model=model,
        settings_file=settings_file,
        output_root=Path(output_root),
        ensure_hpc_settings=ensure_hpc_settings,
        generate_hpc=generate_hpc,
        simulation_root_name=simulation_root_name,
        on_failure=record,
    )

    return GenerationResult(
        created=list(created),
        simulation_root=simulation_root,
        configs=list(configs),
        failures=failures,
        settings=settings,
    )


@dataclass
class PreviewTarget:
    """Where the pieces of one built simulation live."""

    simulation_root: Path
    run_dir: Path
    init_script: Path
    settings_file: Optional[Path]
    layer: Optional[int] = None


def find_assembly_script(lammps_dir: Path) -> Optional[Path]:
    """The script in ``lammps_dir`` that assembles the system.

    Identified by content -- it is the one containing a ``region ... block``
    line -- rather than by filename. Filenames vary by model (``system.in`` for
    the AFM family, ``slide.in`` for a PES tip scan) and the sliding scripts
    that sit beside them read a post-indentation state file whose name is an
    unexpanded LAMMPS variable, so picking by name is both fragile and wrong.

    Falls back to the single pre-assembled script when no candidate has a box.
    """
    scripts = sorted(lammps_dir.glob('*.in'))
    if not scripts:
        return None

    for script in scripts:
        if _REGION_BOX.search(script.read_text(encoding='utf-8')):
            return script

    # Pre-assembled form: prefer the plainest name over refinement variants.
    preferred = [s for s in scripts if 'refine' not in s.name]
    return (preferred or scripts)[0]


def discover_preview_target(simulation_root: Path,
                            run_dir: Path,
                            layer: Optional[int] = None) -> Optional[PreviewTarget]:
    """Locate the assembly script and identity file for one built simulation.

    Args:
        simulation_root: The simulation root; every ``read_data`` path inside a
            generated script is relative to this, not to the script's directory.
        run_dir: A directory created by the build.
        layer: Which ``L{n}`` level to preview. AFM-family builds have one per
            sheet layer count; sheet-like builds have none. Defaults to the
            lowest available.

    Returns:
        A :class:`PreviewTarget`, or None when nothing was built.
    """
    levels = _layer_levels(run_dir)

    if levels:
        chosen = levels.get(layer) if layer is not None else levels[min(levels)]
        if chosen is None:
            available = ', '.join(f"L{n}" for n in sorted(levels))
            raise ValueError(f"No L{layer} in {run_dir}; available: {available}")
        level_dir, resolved_layer = chosen, (layer if layer is not None else min(levels))
    else:
        level_dir, resolved_layer = run_dir, None

    lammps_dir = level_dir / 'lammps'
    if not lammps_dir.is_dir():
        return None

    script = find_assembly_script(lammps_dir)
    if script is None:
        return None

    settings_file = lammps_dir / 'system.in.settings'
    return PreviewTarget(
        simulation_root=simulation_root,
        run_dir=level_dir,
        init_script=script,
        settings_file=settings_file if settings_file.exists() else None,
        layer=resolved_layer,
    )


def _layer_levels(run_dir: Path) -> Dict[int, Path]:
    """``{layer count: directory}`` for the ``L{n}`` levels under ``run_dir``."""
    levels: Dict[int, Path] = {}
    for child in run_dir.iterdir() if run_dir.is_dir() else ():
        match = _LAYER_DIR.match(child.name)
        if match and child.is_dir():
            levels[int(match.group(1))] = child
    return levels
