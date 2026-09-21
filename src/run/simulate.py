"""The runner: route a config to its builder and execute the builds.

Split out of src/core/run.py. This is the only module in the package that
instantiates builders, which is why src/run/__init__.py stays free of eager
re-exports.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..builders.afm import AFMSimulation
from ..builders.afm.freestanding import FreestandingPESTipSimulation
from ..builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation
from ..builders.afm.indent import IndentSimulation
from ..builders.afm.pes_tip import PESTipSimulation
from ..builders.sheetonsheet.pes_sheet import PESSheetSimulation
from ..builders.sheetonsheet import SheetOnSheetSimulation
from ..core.config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
    parse_config,
)
from .expand import (
    _validate_runtime_sweep_ordering,
    expand_config_sweeps,
    outer_loop_conflict,
)
from .hpc_dispatch import generate_hpc_scripts_for_root
from .naming import AFM_LIKE_MODELS, SHEET_LIKE_MODELS, output_dir_for

logger = logging.getLogger(__name__)


class FreestandingConfigError(ValueError):
    """A substrate-free config was used with a model lacking a freestanding builder.

    Raised by :func:`_select_afm_builder_cls` and deliberately NOT caught by
    ``run_simulations``'s per-config broad-except (which exists to let a
    material/parameter sweep continue past isolated build failures) — this is
    a hard misconfiguration, not a recoverable per-config build error, so it
    must propagate and abort the whole run.
    """


class NoHeteroBuilderError(ValueError):
    """No builder exists for a hetero (>=2-material) model/config combination.

    Sibling to :class:`FreestandingConfigError` — same category ("this
    config-level setting has no builder that can serve it", not a
    per-sweep-member build failure) and the same hard-abort treatment:
    deliberately NOT caught by ``run_simulations``'s per-config broad-except.
    Raised by :func:`_select_afm_builder_cls` for ``finite_sheet`` + hetero
    ``indent`` (only plain ``afm`` has a finite hetero variant). ``finite_sheet``
    is set once in ``[general]``, so every member of a sweep would fail
    identically — catching it per-config would silently skip every member and
    report a misleading "created 0 simulation(s)" instead of aborting with a
    message naming the problem.
    """


# pylint: disable=import-outside-toplevel
# The hetero/finite/PES builder classes below (src.builders.afm.hetero,
# src.builders.afm.pes_tip, src.builders.sheetonsheet.pes_sheet,
# src.builders.sheetonsheet.flake_slide) are imported lazily in both selector
# functions because they are needed only by the hetero/twist/flake/PES
# branches -- most configs route to the plain builders already imported
# at module level above, so a homogeneous single-material run never pays
# for them.
def _select_afm_builder_cls(model, config_obj):
    """Pick the builder class, routing substrate-free configs to freestanding.

    A substrate-free config (no ``[sub]`` section) is only supported today for
    the freestanding tip PES scan. Freestanding dynamic slide/indent is a
    future (Phase 2) feature, so any other model without a substrate is a
    hard error rather than a silently broken build.
    """
    afm_builders = {'pes_tip': PESTipSimulation, 'indent': IndentSimulation}
    default_cls = afm_builders.get(model, AFMSimulation)
    sheets = getattr(config_obj, 'sheets', []) or []
    is_hetero = len(sheets) >= 2
    if getattr(config_obj, 'sub', None) is None:
        if is_hetero:
            raise FreestandingConfigError(
                "A substrate-free config (no [sub] section) with 2 or more [2D-N] "
                "materials is not supported: heterostructure freestanding is "
                "unsupported. Add a [sub] section for a two-material config, or "
                "drop to a single [2D] material for the freestanding tip PES."
            )
        freestanding = {'pes_tip': FreestandingPESTipSimulation}
        if model in freestanding:
            return freestanding[model]
        raise FreestandingConfigError(
            f"A substrate-free config (no [sub] section) is only supported for "
            f"'pes-tip' (the freestanding tip PES). Model '{model}' requires a [sub] "
            f"section; freestanding dynamic slide/indent is a future (Phase 2) feature."
        )
    # Hetero AFM when >=2 materials are configured (amorphous sub). ONE hetero
    # decision point: a finite ([general] finite_sheet) stack routes to the D5
    # edge-typed FiniteHeteroAFMSimulation, else the D2 periodic HeteroAFMSimulation.
    # 'indent' reroutes to the pure-MRO HeteroIndentSimulation (no finite variant
    # exists for it — only plain 'afm' has one). 'pes_tip' reroutes to the
    # pure-MRO HeteroPESTipSimulation (scans the matched interface's coincidence
    # cell rather than either material's own unit cell).
    if is_hetero:
        if model == 'indent':
            from src.builders.afm.hetero import HeteroIndentSimulation
            if getattr(config_obj.general, 'finite_sheet', False):
                raise NoHeteroBuilderError(
                    "finite_sheet hetero indent is not supported: only the "
                    "periodic (non-finite) hetero AFM builder has an indent "
                    "variant. Only plain 'afm' has a finite hetero variant "
                    "(FiniteHeteroAFMSimulation). Drop [general] finite_sheet "
                    "for a hetero indent run."
                )
            return HeteroIndentSimulation
        if model == 'afm':
            from src.builders.afm.hetero import HeteroAFMSimulation, FiniteHeteroAFMSimulation
            if getattr(config_obj.general, 'finite_sheet', False):
                return FiniteHeteroAFMSimulation
            return HeteroAFMSimulation
        if model == 'pes_tip':
            from src.builders.afm.pes_tip import HeteroPESTipSimulation
            return HeteroPESTipSimulation
    return default_cls


def _select_sheet_builder_cls(model, config_obj):
    """Route sheet-on-sheet to the hetero builder for >=2 materials or a twist.

    A single ``[2D]`` material with ``general.twist_angle`` set is a genuine
    same-material moiré, so it must also reach the hetero builder (which
    synthesises the two sheets). A single material with no twist stays on the
    homogeneous builder (unchanged).

    The same rule applies to the static ``pes_sheet`` scan: a two-material (or
    twisted) bilayer is scanned over the matched interface's coincidence cell
    by :class:`~src.builders.sheetonsheet.pes_sheet.HeteroPESSheetSimulation`, since the
    homogeneous scan builder can neither assemble the stack nor group its
    layers (hetero layers share atom types).
    """
    sheets = getattr(config_obj, 'sheets', None) or []
    twist = getattr(getattr(config_obj, 'general', None), 'twist_angle', None)
    is_hetero = len(sheets) >= 2 or twist is not None

    if model == 'pes_sheet':
        if is_hetero:
            from src.builders.sheetonsheet.pes_sheet import HeteroPESSheetSimulation
            return HeteroPESSheetSimulation
        return PESSheetSimulation
    # A [Flake] section switches sheet-on-sheet to the D4 flake-driven slide,
    # regardless of the substrate being homogeneous or hetero (precedence).
    if model == 'sheetonsheet' and getattr(config_obj, 'flake', None) is not None:
        from src.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
        return FlakeSlideSimulation
    if is_hetero:
        return HeteroSheetOnSheetSimulation
    return SheetOnSheetSimulation
# pylint: enable=import-outside-toplevel


def run_simulations(
    config_file: str,
    model: str,
    settings_file: Optional[Union[str, Path]] = None,
    output_root: Optional[Union[str, Path]] = None,
    ensure_hpc_settings=None,
    use_aiida: bool = False,
    generate_hpc: bool = False,
    simulation_root_name: Optional[str] = None,
    on_failure: Optional[Callable[[Path, Dict[str, Any], Exception], None]] = None,
    allow_generate_amorphous: bool = True,
) -> Tuple[List[Path], Path, List[Dict[str, Any]], Any]:
    """Run all simulations defined in a config file.

    Args:
        on_failure: Called as ``(output_dir, run_dict, exception)`` for each
            config whose build fails. Failures are otherwise only logged, and
            the caller sees just a success count with no way to say which
            member failed.
        allow_generate_amorphous: When False, a missing amorphous structure
            raises instead of triggering a melt-quench. See
            :func:`src.builders.construction.components.make_amorphous`.

    Returns a tuple of (created_simulations, simulation_root, configs_to_run, settings).
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_dict = parse_config(config_path)
    defaults = load_settings(settings_file=settings_file)

    if use_aiida:
        defaults.aiida.enabled = True
        defaults.aiida.create_provenance = True

    # Carried on the settings rather than threaded through every builder: the
    # builders already pass `settings` down to make_amorphous.
    if not allow_generate_amorphous:
        defaults.quench.allow_generate = False

    if model in SHEET_LIKE_MODELS:
        if defaults.hpc.lammps_scripts == ['system.in', 'slide.in']:
            defaults.hpc.lammps_scripts = ['slide.in']
    elif model in AFM_LIKE_MODELS:
        if defaults.hpc.lammps_scripts == ['slide.in']:
            defaults.hpc.lammps_scripts = ['system.in', 'slide.in']

    # pes_tip uses the AFM stack (geometry/output layout) but is self-contained:
    # a single static scan script, no separate system.in indentation phase.
    if model == 'pes_tip':
        defaults.hpc.lammps_scripts = ['slide.in']

    if ensure_hpc_settings is not None and generate_hpc:
        ensure_hpc_settings(defaults.hpc)

    configs_to_run = expand_config_sweeps(base_dict)

    for run_dict in configs_to_run:
        _validate_runtime_sweep_ordering(run_dict.get('general', {}))
        # Before the simulation root exists: a conflict the builders would
        # only raise on from write_inputs, i.e. after a full geometry build
        # onto a tree that then stays half written.
        conflict = outer_loop_conflict(run_dict, model)
        if conflict is not None:
            raise ValueError(conflict)

    root_name = simulation_root_name or datetime.now().strftime("simulation_%Y%m%d_%H%M%S")
    base_root = Path(output_root) if output_root is not None else Path.cwd()
    simulation_root = base_root / root_name
    simulation_root.mkdir(parents=True, exist_ok=True)
    (simulation_root / 'logs').mkdir(parents=True, exist_ok=True)

    created_simulations: List[Path] = []

    for run_dict in configs_to_run:
        run_dict['settings'] = defaults.model_dump()

        output_dir = simulation_root / output_dir_for(run_dict, model)

        prov_dir = output_dir / 'provenance'
        prov_dir.mkdir(parents=True, exist_ok=True)
        # The rebuild path needs the settings the run was built with; nothing
        # wrote them before, so `aiida rebuild` silently used the defaults.
        import yaml  # pylint: disable=import-outside-toplevel
        (prov_dir / 'settings.yaml').write_text(
            yaml.safe_dump(defaults.model_dump(), sort_keys=False), encoding='utf-8')

        try:
            if model in AFM_LIKE_MODELS:
                config_obj = AFMSimulationConfig(**run_dict)
                config_json_path = prov_dir / 'config.json'
                config_json_path.write_text(config_obj.model_dump_json(indent=2), encoding='utf-8')
                builder_cls = _select_afm_builder_cls(model, config_obj)
                builder = builder_cls(config_obj, output_dir, config_path=str(config_json_path))
                builder.set_model(model)
            else:
                config_obj = SheetOnSheetSimulationConfig(**run_dict)
                config_json_path = prov_dir / 'config.json'
                config_json_path.write_text(config_obj.model_dump_json(indent=2), encoding='utf-8')
                builder_cls = _select_sheet_builder_cls(model, config_obj)
                builder = builder_cls(config_obj, output_dir, config_path=str(config_json_path))
                builder.set_model(model)

            builder.set_base_output_dir(simulation_root)
            builder.build()
            created_simulations.append(output_dir)

        except (FreestandingConfigError, NoHeteroBuilderError):
            # Hard misconfiguration (substrate-free config on a model with no
            # freestanding builder, or a hetero combination -- e.g. finite_sheet
            # hetero indent -- with no builder at all) — abort the whole run
            # instead of silently skipping this config and reporting a false
            # "N/M successful". Both are config-level settings that would fail
            # identically for every member of a sweep, not an isolated
            # per-config build error.
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Build failed for %s: %s", output_dir, exc)
            # Logging alone leaves callers with only "N/M successful" and no way
            # to say which member failed or why. The callback lets them report it.
            if on_failure is not None:
                on_failure(output_dir, run_dict, exc)
            continue

    if generate_hpc and created_simulations:
        generate_hpc_scripts_for_root(simulation_root, defaults)

    return created_simulations, simulation_root, configs_to_run, defaults
