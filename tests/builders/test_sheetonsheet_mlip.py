"""MLIP routing through the sheet-on-sheet builder.

Sheet-on-sheet is the family the collapsed v4 matrix still supports: a single
sheet component, so ONE MLIP instance with a full type map covers every atom —
the only topology either framework accepts (v4 spec §3).

The per-layer "MLIP intralayer + LJ interlayer" variant (v3 setup 1) is struck:
it needed NULL-padded type maps, and ``pair_deepmd`` rejects them outright
("Pair hybrid sub-style deepmd is not used"), as does ``pair_mace`` (ACEsuit
#1314). Overlaying LJ on a full-cell MLIP instead is rejected on *physics*
(v4 §2): the model's cutoff already spans the interlayer gap, so a UFF-mixed LJ
parameterised to be the interlayer potential double-counts. D3 is the supported
vdW lever.

The first test in this file is the no-regression anchor: an all-SW config's
``system.in.settings`` must stay byte-identical to what the builder produced
before MLIP routing existed.
"""

from pathlib import Path

import pytest

from frictionsim2d.builders.sheetonsheet import SheetOnSheetSimulation
from frictionsim2d.core.config import SheetOnSheetSimulationConfig, load_settings
from frictionsim2d.core.mlip import is_mlip

REPO_ROOT = Path(__file__).resolve().parents[2]
MOS2_CIF = REPO_ROOT / 'frictionsim2d' / 'data' / 'materials' / 'h-MoS2.cif'
MOS2_SW = REPO_ROOT / 'frictionsim2d' / 'data' / 'potentials' / 'sw' / 'MoS2_wen.sw'


def _model(tmp_path: Path, name: str) -> Path:
    """A fake serialized MLIP model (binary, deliberately not UTF-8)."""
    path = tmp_path / name
    path.write_bytes(b"\x80\x02not-utf8\xff")
    return path


def make_config(tmp_path: Path, **sheet_overrides) -> SheetOnSheetSimulationConfig:
    """Minimal 3-layer sheet-on-sheet config; sheet keys overridable.

    An MLIP override gets ``dispersion='d3'`` by default: this is a 3-layer
    stack, and a multilayer MLIP sheet without D3 is rejected at config time
    because the bare model has no interlayer binding (see
    ``test_config.TestMultilayerMLIPRequiresD3``). Tests that specifically
    exercise the base path pass ``dispersion`` explicitly.
    """
    sheet = {
        'mat': 'h-MoS2',
        'pot_type': 'sw',
        'pot_path': str(MOS2_SW),
        'cif_path': str(MOS2_CIF),
        'x': 20.0,
        'y': 20.0,
        'layers': [3],
    }
    sheet.update(sheet_overrides)
    if is_mlip(sheet.get('pot_type', '')) and 'dispersion' not in sheet:
        sheet['dispersion'] = 'd3'
    return SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D': sheet,
        'lj_override': {},
        'settings': load_settings().model_dump(),
    })


def settings_text(config, tmp_path: Path) -> str:
    """Run _generate_potentials and return the written settings file.

    Absolute staging paths are collapsed to stable placeholders so the golden
    does not embed a tmpdir.
    """
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / 'out'))
    builder._generate_potentials()
    text = (tmp_path / 'out' / 'lammps' / 'system.in.settings').read_text(encoding='utf-8')
    return text.replace(str(tmp_path / 'out' / 'provenance' / 'potentials'), '<POT>')


# ---------------------------------------------------------------------------
# Regression anchor: classical output is untouched
# ---------------------------------------------------------------------------

_SW_GOLDEN = """\
mass 1*3 95.950000 #Mo
mass 4*6 32.060000 #S
mass 7 1.0 #Virtual atom
group 2D_all type 1 2 3 4 5 6
group 2D_core type 1 2 3 4 5 6
group layer_1 type 1 4
group layer_2 type 2 5
group layer_3 type 3 6
pair_style hybrid sw sw sw lj/cut 11.0
# Self-interactions (intra-component)
pair_coeff * * sw 1 <POT>/MoS2_wen.sw Mo NULL NULL S NULL NULL NULL # sheet Layer 1
pair_coeff * * sw 2 <POT>/MoS2_wen.sw NULL Mo NULL NULL S NULL NULL # sheet Layer 2
pair_coeff * * sw 3 <POT>/MoS2_wen.sw NULL NULL Mo NULL NULL S NULL # sheet Layer 3

# Cross-interactions (inter-component LJ)
pair_coeff 1 2 lj/cut 0.0024283504 3.052 # Mo(L1)-Mo(L2)
pair_coeff 1 5 lj/cut 0.00537146340837286 3.5435 # Mo(L1)-S(L2)
pair_coeff 2 4 lj/cut 0.00537146340837286 3.5435 # S(L1)-Mo(L2)
pair_coeff 4 5 lj/cut 0.0118815716 4.035 # S(L1)-S(L2)
pair_coeff 1 3 lj/cut 0.0024283504 3.052 # Mo(L1)-Mo(L3)
pair_coeff 1 6 lj/cut 0.00537146340837286 3.5435 # Mo(L1)-S(L3)
pair_coeff 3 4 lj/cut 0.00537146340837286 3.5435 # S(L1)-Mo(L3)
pair_coeff 4 6 lj/cut 0.0118815716 4.035 # S(L1)-S(L3)
pair_coeff 2 3 lj/cut 0.0024283504 3.052 # Mo(L2)-Mo(L3)
pair_coeff 2 6 lj/cut 0.00537146340837286 3.5435 # Mo(L2)-S(L3)
pair_coeff 3 5 lj/cut 0.00537146340837286 3.5435 # S(L2)-Mo(L3)
pair_coeff 5 6 lj/cut 0.0118815716 4.035 # S(L2)-S(L3)

# Virtual atom (minimal LJ)
pair_coeff * 7 lj/cut 1e-100 1e-100
"""


def test_all_sw_settings_file_is_byte_identical(tmp_path):
    """Captured from the builder BEFORE MLIP routing was wired in."""
    assert settings_text(make_config(tmp_path), tmp_path) == _SW_GOLDEN


# ---------------------------------------------------------------------------
# Full-cell MLIP: one instance spans the stack, with and without D3
# ---------------------------------------------------------------------------

def test_deepmd_spans_the_stack_without_interlayer_lj(tmp_path):
    model = _model(tmp_path, 'dpa.pth')
    config = make_config(tmp_path, pot_type='deepmd', pot_path=str(model),
                         lat_c=6.15)
    text = settings_text(config, tmp_path)

    style_line = next(l for l in text.splitlines() if l.startswith('pair_style'))
    assert style_line.count('deepmd') == 1

    # ONE un-indexed map covering every sheet type. The virtual drive atom is
    # mapped to the placeholder element rather than NULL.
    assert 'pair_coeff * * deepmd Mo Mo Mo S S S Mo # sheet' in text
    assert 'deepmd 1' not in text
    # E2/E3: a NULL entry, or a type-range coeff line, is a hard LAMMPS error.
    assert 'NULL' not in text
    assert 'pair_coeff 1*' not in text
    # E7: the element list is never omitted -- doing so silently mis-maps types
    # to the model's first N elements.
    assert 'pair_coeff * * deepmd\n' not in text

    # No interlayer LJ: the model itself supplies the interlayer physics.
    assert '# Mo(L1)-Mo(L2)' not in text
    assert 'Cross-interactions' not in text

    # The drive atom is hidden from the MLIP graph, one explicit pair per type
    # (neigh_modify exclude type has no wildcard form).
    for t_id in range(1, 8):
        assert f'neigh_modify exclude type 7 {t_id}' in text


def test_deepmd_never_emits_the_model_deviation_form(tmp_path):
    """'pair_style deepmd m1 m2' is uncertainty mode, not two potentials."""
    model = _model(tmp_path, 'dpa.pth')
    config = make_config(tmp_path, pot_type='deepmd', pot_path=str(model),
                         lat_c=6.15)
    text = settings_text(config, tmp_path)
    assert '<POT>/dpa.pth <POT>/dpa.pth' not in text
    assert 'potentials/dpa.pth potentials/dpa.pth' not in text


def test_deepmd_plus_d3_uses_hybrid_overlay(tmp_path):
    model = _model(tmp_path, 'dpa.pth')
    config = make_config(tmp_path, pot_type='deepmd', pot_path=str(model),
                         dispersion='d3', lat_c=6.15)
    text = settings_text(config, tmp_path)

    assert 'pair_style hybrid/overlay deepmd' in text
    assert 'dispersion/d3 bj pbe 16.0 8.0' in text
    # D3 needs a real element for EVERY type, virtual placeholder included.
    assert 'pair_coeff * * dispersion/d3 Mo Mo Mo S S S Mo' in text


def test_mace_plus_d3(tmp_path):
    model = _model(tmp_path, 'mace-mp.json')
    config = make_config(tmp_path, pot_type='mace', pot_path=str(model),
                         dispersion='d3', lat_c=6.15)
    text = settings_text(config, tmp_path)

    assert ('pair_style hybrid/overlay symmetrix/mace '
            'dispersion/d3 bj pbe 16.0 8.0') in text
    # MACE carries its model path in the coeff line, not the style line.
    assert ('pair_coeff * * symmetrix/mace potentials/mace-mp.json '
            'Mo Mo Mo S S S Mo') in text


def test_model_is_staged_once_at_the_simulation_root(tmp_path):
    """MLIP models go to <campaign root>/potentials, not into each run dir.

    Mirrors a real campaign: ``run_simulations`` calls ``set_base_output_dir``
    on every builder, which is what makes ``simulation_root`` the shared root
    of the sweep rather than the run directory.
    """
    model = _model(tmp_path, 'dpa.pth')
    root = tmp_path / 'sim_root'
    config = make_config(tmp_path, pot_type='deepmd', pot_path=str(model),
                         lat_c=6.15)
    builder = SheetOnSheetSimulation(
        config, output_dir=str(root / 'sheetonsheet' / 'h-MoS2' / 'K300'))
    builder.set_base_output_dir(root)
    builder._generate_potentials()
    text = (builder.output_dir / 'lammps' / 'system.in.settings').read_text(
        encoding='utf-8')

    assert (root / 'potentials' / 'dpa.pth').is_file()
    assert not (builder.output_dir / 'provenance' / 'potentials' / 'dpa.pth').exists()
    # This test is about the STAGED PATH, not the pair_style keyword: a 3-layer
    # MLIP stack now carries dispersion='d3' (make_config's default), so the
    # header is hybrid/overlay. Assert only the staged reference.
    assert 'deepmd potentials/dpa.pth' in text


def test_an_mlip_sheet_marks_the_template_flag_internal_lj(tmp_path):
    """slide.lmp's has_internal_lj is unconditional for an MLIP now."""
    model = _model(tmp_path, 'dpa.pth')
    config = make_config(tmp_path, pot_type='deepmd', pot_path=str(model),
                         lat_c=6.15)
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / 'out'))
    pm = builder._generate_potentials()
    assert pm.is_sheet_lj('deepmd') is False
    assert pm.is_sheet_lj('mace') is False


# ---------------------------------------------------------------------------
# Cell-level validation (v4 spec §3/§6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mode', ['atom_bonds', 'com_spring'])
def test_constraint_modes_are_rejected_with_an_mlip_sheet(tmp_path, mode):
    model = _model(tmp_path, 'dpa.pth')
    config = make_config(tmp_path, pot_type='deepmd', pot_path=str(model),
                         lat_c=6.15)
    config.settings.simulation.constraint_mode = mode
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / 'out'))
    with pytest.raises(ValueError, match='constraint_mode'):
        builder._generate_potentials()
