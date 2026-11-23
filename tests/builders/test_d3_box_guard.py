"""A D3 overlay whose cutoff does not fit the built periodic box fails at build."""
from types import SimpleNamespace

import pytest

from frictionsim2d.builders.simulation_base import SimulationBase
from frictionsim2d.core.config import SheetOnSheetSimulationConfig, load_settings
from tests.builders.test_family_mlip import (
    _all_mlip_afm_like, _build, _deepmd_sheet, _model, needs_lmp)

D3_MATCH = r'd3_cutoff = .* needs a periodic box'


class _Sim(SimulationBase):
    def build(self):
        return None

    def write_inputs(self):
        return None


def _stub(dispersion, cutoff=16.0, skin=2.0):
    settings = SimpleNamespace(
        potential=SimpleNamespace(d3_cutoff=cutoff),
        simulation=SimpleNamespace(neighbor_list=skin))
    sheet = SimpleNamespace(dispersion=dispersion)
    return _Sim(SimpleNamespace(sheet=sheet, settings=settings), output_dir='.')


def test_check_d3_fits_rejects_a_box_below_cutoff_plus_skin():
    box = {'xlo': 0.0, 'xhi': 12.0, 'ylo': 0.0, 'yhi': 40.0}
    with pytest.raises(ValueError, match=D3_MATCH):
        _stub('d3')._check_d3_fits(box)


def test_check_d3_fits_accepts_a_box_that_fits():
    _stub('d3')._check_d3_fits({'xlo': 0.0, 'xhi': 30.0, 'ylo': 0.0, 'yhi': 30.0})


def test_check_d3_fits_is_inert_without_d3():
    _stub('none')._check_d3_fits({'xlo': 0.0, 'xhi': 1.0, 'ylo': 0.0, 'yhi': 1.0})


def test_check_d3_fits_reads_the_first_sheet_of_a_hetero_config():
    sim = _stub('none')
    sim.config.sheets = [SimpleNamespace(dispersion='d3'), SimpleNamespace(dispersion='none')]
    with pytest.raises(ValueError, match=D3_MATCH):
        sim._check_d3_fits({'xlo': 0.0, 'xhi': 5.0, 'ylo': 0.0, 'yhi': 5.0})


def test_afm_with_d3_in_a_small_box_fails_at_build(tmp_path):
    from frictionsim2d.builders.afm.indent import IndentSimulation
    config = _all_mlip_afm_like(tmp_path, {'indent': {}})
    sheet = config.sheet
    sheet.x = sheet.y = 12.0
    with pytest.raises(ValueError, match=D3_MATCH):
        _build(IndentSimulation, config, tmp_path)


def test_hetero_afm_checks_the_replicated_box(tmp_path, monkeypatch):
    """The hetero AFM sheet is the replicated stack; its box is what D3 must fit."""
    from frictionsim2d.builders.afm.hetero import HeteroAFMSimulation
    sim = HeteroAFMSimulation.__new__(HeteroAFMSimulation)
    sim.config = SimpleNamespace(
        general=SimpleNamespace(interlayer_style='lj'),
        sheets=[SimpleNamespace(dispersion='d3')],
        settings=SimpleNamespace(potential=SimpleNamespace(d3_cutoff=16.0),
                                 simulation=SimpleNamespace(neighbor_list=2.0)))
    sim.sheet_paths = {}

    def _fake_build(_build_dir):
        sim._hetero_rep_path = tmp_path / 'hetero_rep.data'
        sim.sheet_dims = {'xlo': 0.0, 'xhi': 12.0, 'ylo': 0.0, 'yhi': 12.0}

    monkeypatch.setattr(sim, '_build_hetero_sheet', _fake_build)
    with pytest.raises(ValueError, match=D3_MATCH):
        sim._build_sheet_for_layers(tmp_path, 1)


@needs_lmp
def test_hetero_sheet_with_d3_in_a_small_box_fails_at_build(tmp_path):
    from frictionsim2d.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation
    model = _model(tmp_path)
    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D-1': _deepmd_sheet(tmp_path, mat='h-MoS2', layers=(2,), x=10.0, y=10.0, model=model),
        '2D-2': _deepmd_sheet(tmp_path, mat='h-WS2', layers=(2,), x=10.0, y=10.0, model=model),
        'settings': load_settings().model_dump(),
    })
    with pytest.raises(ValueError, match=D3_MATCH):
        _build(HeteroSheetOnSheetSimulation, config, tmp_path)


@needs_lmp
def test_flake_slide_with_d3_in_a_small_box_fails_at_build(tmp_path):
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
    from tests.builders.test_family_mlip import MAT
    model = _model(tmp_path)
    settings = load_settings().model_dump()
    settings['potential']['d3_cutoff'] = 40.0   # the flake's own clearance rule needs a wide cell
    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D': _deepmd_sheet(tmp_path, mat='h-MoS2', layers=(2,), x=24.0, y=24.0, model=model),
        'Flake': {'shape': 'hexagon', 'edge_length': 3.0, 'corner_radius': 2.0,
                  'corner_spring_k': 10.0, 'mat': 'h-MoS2', 'pot_type': 'deepmd',
                  'pot_path': model, 'cif_path': str(MAT / 'h-MoS2.cif')},
        'settings': settings,
    })
    with pytest.raises(ValueError, match=D3_MATCH):
        _build(FlakeSlideSimulation, config, tmp_path)
