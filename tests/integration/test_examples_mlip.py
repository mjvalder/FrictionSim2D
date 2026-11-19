"""Every shipped ``examples/mlip/*.ini`` must actually validate.

These files are the documented entry point for MLIP runs
(``documentation/mlip.md``), so a config that no longer parses is a broken
deliverable, not a stale sample. The loop below drives the *real* loading path
that :func:`src.run.simulate.run_simulations` uses --

    parse_config -> expand_config_sweeps -> {AFM,SheetOnSheet}SimulationConfig

-- and then applies :func:`src.core.mlip.validate_mlip_cell`, which is where the
capability matrix's cell-level rules live (no MLIP beside a classical
component, no multi-component full cell, one model per framework per cell).
Together those are exactly the two gates an example has to clear before a
builder ever touches it.

Only single-component sheet examples survive the v4 matrix
(``docs/superpowers/specs/2026-08-11-mlip-post-m0-design.md`` §3/§6): the AFM,
hetero and per-layer-LJ examples described topologies no LAMMPS build accepts
and were deleted rather than annotated, because a shipped example is a
recommendation.

The MLIP model files the examples name (``../models/*.pth``) are multi-GB
foundation models and are deliberately NOT in the repo. They are stubbed into
``tmp_path`` and the config's ``pot_path`` repointed at the stub, so the
framework/extension check in ``ComponentConfig.validate_mlip_model_extension``
is exercised against a file that genuinely exists rather than a dangling
relative path.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
    parse_config,
)
from src.core.mlip import is_mlip, validate_mlip_cell
from src.run.expand import expand_config_sweeps
from src.run.naming import (
    AFM_LIKE_MODELS,
    mlip_flavor_from_pot_token,
    output_dir_for,
)
from src.run.simulate import _select_afm_builder_cls, _select_sheet_builder_cls

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_MLIP = REPO_ROOT / 'examples' / 'mlip'

#: Which ``FrictionSim2D run <model>`` subcommand each example is written for.
#: Kept explicit rather than inferred from the filename: the routing decides
#: which pydantic model validates the file, so a wrong guess would silently
#: test the wrong thing.
EXAMPLE_MODELS = {
    'sheet_mlip_full.ini': 'sheetonsheet',
    'pes_sheet_mlip.ini': 'pes_sheet',
    'afm_mlip_full.ini': 'afm',
}


def _example_paths():
    return sorted(EXAMPLES_MLIP.glob('*.ini'))


#: Examples kept as they are although they fail validation, by user decision
#: (2026-11-19): afm_mlip_full.ini is the cell the MLIP AFM gate was run on
#: (documentation/mlip_afm_gate.md), and main's tip-footprint check (sheet >= 2x
#: the tip diameter, AFMSimulationConfig.validate_tip_sheet_footprint) refuses
#: its 30x30 sheet under an r=15 tip. strict=True: the day the cell is fixed,
#: these xfails turn into failures and the marker must be removed.
KNOWN_INVALID = {
    'afm_mlip_full.ini': 'sheet footprint 30x30 < 2x tip diameter (r=15); kept for the AFM gate',
}


def _validated_example_params():
    """The examples, with KNOWN_INVALID ones as strict xfails (tests that validate)."""
    return [pytest.param(p, id=p.name, marks=pytest.mark.xfail(
                reason=KNOWN_INVALID[p.name], strict=True,
                raises=ValidationError))
            if p.name in KNOWN_INVALID else pytest.param(p, id=p.name)
            for p in _example_paths()]


def _stub_models(run_dict, tmp_path):
    """Create an empty stand-in for every MLIP model the config names.

    Returns the number of sheet sections repointed, so a test can assert the
    example really is an MLIP example and not silently a classical one.
    """
    staged = 0
    for key, section in run_dict.items():
        # 'tip'/'sub' matter now that AFM-like examples exist (v6): their model
        # paths need stubbing too, or an AFM example cannot validate.
        if not isinstance(section, dict) or str(key) not in ('tip', 'sub') \
                and not str(key).startswith('2D'):
            continue
        if not is_mlip(section.get('pot_type') or ''):
            continue
        stub = tmp_path / Path(str(section['pot_path'])).name
        stub.touch()
        section['pot_path'] = str(stub)
        staged += 1
    return staged


def _sheets_of(config_obj):
    """Every sheet config on a validated model, primary first."""
    sheets = list(getattr(config_obj, 'sheets', None) or [])
    if not sheets:
        sheets = [config_obj.sheet]
    return sheets


def test_examples_mlip_directory_is_populated():
    """Guard the glob: an empty directory would make every test below vacuous."""
    assert sorted(p.name for p in _example_paths()) == sorted(EXAMPLE_MODELS)


@pytest.mark.parametrize('path', _validated_example_params())
def test_example_config_is_valid(path, tmp_path):
    """Each example parses, expands and validates, cell rules included."""
    model = EXAMPLE_MODELS[path.name]
    base = parse_config(path)
    configs = expand_config_sweeps(base)
    assert configs, f"{path.name} expanded to no runs"

    settings = load_settings()
    for run_dict in configs:
        assert _stub_models(run_dict, tmp_path) >= 1, (
            f"{path.name} names no MLIP sheet -- it does not belong in "
            f"examples/mlip/"
        )
        run_dict['settings'] = settings.model_dump()

        if model in AFM_LIKE_MODELS:
            config_obj = AFMSimulationConfig(**run_dict)
            components = [('tip', config_obj.tip), ('sub', config_obj.sub),
                          ('flake', config_obj.flake)]
            sheets = _sheets_of(config_obj)
        else:
            config_obj = SheetOnSheetSimulationConfig(**run_dict)
            sheets = _sheets_of(config_obj)
            components = [(f'sheet_{i + 1}', sheet)
                          for i, sheet in enumerate(sheets[1:], start=1)]

        # The capability-matrix rules that need the whole cell (spec §1.1/§3.4):
        # this is the check the builders run, and the reason a MACE example may
        # not carry an SW tip.
        validate_mlip_cell(
            sheets[0],
            components=components,
            constraint_mode=config_obj.settings.simulation.constraint_mode,
        )


@pytest.mark.parametrize('path', _validated_example_params())
def test_example_declares_explicit_lat_c(path, tmp_path):
    """``lat_c`` is mandatory for MLIP sheets and must be an interlayer distance.

    ``SheetConfig`` already refuses a missing ``lat_c``; the extra assertion
    here is physical. ``lat_c`` in this package is the LAYER-TO-LAYER distance
    (``geometry.lat_c_default`` is 6.0 A), not an in-plane lattice constant, so
    an example carrying MoS2's in-plane a (~3.18 A) would start the stack with
    the layers interpenetrating -- the "dome plow" failure documented in
    documentation/mlip.md.
    """
    run_dict = expand_config_sweeps(parse_config(path))[0]
    _stub_models(run_dict, tmp_path)
    run_dict['settings'] = load_settings().model_dump()

    model = EXAMPLE_MODELS[path.name]
    cls = (AFMSimulationConfig if model in AFM_LIKE_MODELS
           else SheetOnSheetSimulationConfig)
    for sheet in _sheets_of(cls(**run_dict)):
        assert sheet.lat_c is not None
        assert sheet.lat_c > 4.0, (
            f"{path.name}: lat_c = {sheet.lat_c} A is an in-plane lattice "
            f"constant, not an interlayer distance"
        )


@pytest.mark.parametrize('path', _validated_example_params())
def test_example_satisfies_its_builder_layer_floor(path, tmp_path):
    """The example must clear the ``MIN_LAYERS`` of the builder it routes to.

    The floors differ per family and are easy to get wrong when copying a
    config: ``SheetOnSheetSimulation`` needs >= 3, the *hetero* sheet-on-sheet
    builder that two ``[2D-N]`` sections route to needs >= 4, and the PES sheet
    scan needs exactly 2. Every one of those raises only from ``build()``,
    i.e. after a full geometry build -- far too late for an example file.
    """
    model = EXAMPLE_MODELS[path.name]
    run_dict = expand_config_sweeps(parse_config(path))[0]
    _stub_models(run_dict, tmp_path)
    run_dict['settings'] = load_settings().model_dump()

    if model in AFM_LIKE_MODELS:
        config_obj = AFMSimulationConfig(**run_dict)
        builder_cls = _select_afm_builder_cls(model, config_obj)
    else:
        config_obj = SheetOnSheetSimulationConfig(**run_dict)
        builder_cls = _select_sheet_builder_cls(model, config_obj)

    n_layers = sum(sheet.layers[0] for sheet in _sheets_of(config_obj))
    floor = getattr(builder_cls, 'MIN_LAYERS', None)
    if floor is not None:
        assert n_layers >= floor, (
            f"{path.name} routes to {builder_cls.__name__} (MIN_LAYERS "
            f"{floor}) but declares {n_layers} layers"
        )
    if builder_cls.__name__ == 'PESSheetSimulation':
        assert n_layers == 2, "the static PES scan needs exactly 2 layers"


@pytest.mark.parametrize('path', _example_paths(), ids=lambda p: p.name)
def test_example_gets_its_own_flavour_directory(path):
    """Each example lands in a distinct MLIP flavour segment (merge decision 1).

    The flavour rides in the ``pot_<slug>_<stem>`` segment of the run path
    (``pot_deepmd-d3_<stem>``); it is what stops a sweep over potentials or
    dispersion settings from overwriting itself, so an example whose path
    decoded to no flavour (a classical config that slipped into this
    directory) would silently collide with the plain run.
    """
    run_dict = expand_config_sweeps(parse_config(path))[0]
    model = EXAMPLE_MODELS[path.name]
    rel = output_dir_for(run_dict, model)
    flavors = [f for f in map(mlip_flavor_from_pot_token, rel.parts) if f]
    assert len(flavors) == 1, f"{path.name}: no MLIP pot_ segment in {rel}"
    framework, disp = flavors[0]
    assert framework in ('mace', 'deepmd')
    assert disp in ('none', 'd3')
    # The K<temp> leaf carries no flavour suffix any more.
    assert not any(part.startswith('K') and '_' in part for part in rel.parts)


@pytest.mark.parametrize('path', _example_paths(), ids=lambda p: p.name)
def test_example_does_not_carry_a_removed_key(path):
    """A shipped example must not teach a key the config layer now rejects."""
    text = path.read_text(encoding='utf-8')
    assert 'mlip_interlayer' not in text


@pytest.mark.parametrize('path', _validated_example_params())
def test_example_is_uniformly_one_mlip(path, tmp_path):
    """v6 §4: a cell may be multi-component, but never mixed.

    This replaces the v4 §6 single-component check. What matters now is
    uniformity: every component on the SAME MLIP, because one `pair_coeff * *`
    claims every atom type. A classical component beside an MLIP, or a second
    framework, is refused -- so no example may ship one.
    """
    run_dict = expand_config_sweeps(parse_config(path))[0]
    _stub_models(run_dict, tmp_path)
    run_dict['settings'] = load_settings().model_dump()

    model = EXAMPLE_MODELS[path.name]
    if model in AFM_LIKE_MODELS:
        config_obj = AFMSimulationConfig(**run_dict)
        components = [('tip', config_obj.tip), ('sub', config_obj.sub)]
    else:
        config_obj = SheetOnSheetSimulationConfig(**run_dict)
        components = []

    sheets = _sheets_of(config_obj)
    pot_types = {s.pot_type.lower() for s in sheets}
    pot_types |= {c.pot_type.lower() for _, c in components if c is not None}
    assert len(pot_types) == 1, f"{path.name} mixes potentials: {pot_types}"
    assert is_mlip(pot_types.pop())

    # And the cell-level rules agree.
    validate_mlip_cell(sheets[0], components=components)
