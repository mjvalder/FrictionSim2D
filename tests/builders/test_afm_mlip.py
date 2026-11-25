"""MLIP rules for the AFM builder — every AFM cell is multi-component.

v3's setups 3 and 4 (MLIP sheet + SW tip/substrate; MLIP sheet spanning the
stack inside an otherwise-SW cell) are **struck**. Both are partitioned cells,
and a partitioned cell needs a NULL-padded ``pair_coeff`` type map, which

* ``pair_deepmd`` rejects -- ``ERROR: Pair hybrid sub-style deepmd is not
  used``, for one instance as well as three (v4 spec §1, E2), and restricting
  by type range instead fails at ``pair_deepmd.cpp:816`` (E3);
* ``pair_mace`` segfaults on (ACEsuit #1314).

An all-MLIP AFM cell is unreachable too, but for a second reason (v4 §6): the
builders register one MLIP instance *per component*, so even a uniformly-deepmd
tip/sheet/substrate cell is emitted as three NULL-partitioned instances. A
genuine multi-component full cell needs ONE instance whose type map spans every
component, and no builder has a path to that.

So this file pins the *rejections*, plus the regression that a classical AFM
cell is untouched. The golden emission tests that used to live here asserted
text no LAMMPS build accepts and have been deleted, not rewritten.
"""

from pathlib import Path

import pytest

from frictionsim2d.builders.afm import AFMSimulation
from frictionsim2d.core.config import AFMSimulationConfig, load_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALS = REPO_ROOT / 'frictionsim2d' / 'data' / 'materials'
POTENTIALS = REPO_ROOT / 'frictionsim2d' / 'data' / 'potentials' / 'sw'


def _model(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"\x80\x02not-utf8\xff")
    return str(path)


def make_config(tmp_path: Path, sheet_overrides=None, tip_overrides=None,
                sub_overrides=None) -> AFMSimulationConfig:
    """A minimal 3-layer AFM config: SW Si tip + amorphous SW Si substrate."""
    si = {
        'mat': 'Si',
        'pot_type': 'sw',
        'pot_path': str(POTENTIALS / 'Si.sw'),
        'cif_path': str(MATERIALS / 'Si.cif'),
    }
    tip = {**si, 'r': 5.0, 'dspring': 0.1}
    tip.update(tip_overrides or {})
    sub = {**si, 'thickness': 10.0, 'amorph': 'a'}
    sub.update(sub_overrides or {})
    sheet = {
        'mat': 'h-MoS2',
        'pot_type': 'sw',
        'pot_path': str(POTENTIALS / 'MoS2_wen.sw'),
        'cif_path': str(MATERIALS / 'h-MoS2.cif'),
        'x': 20.0, 'y': 20.0, 'layers': [3],
    }
    sheet.update(sheet_overrides or {})
    return AFMSimulationConfig(**{
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': tip,
        'sub': sub,
        '2D': sheet,
        'lj_override': {},
        'settings': load_settings().model_dump(),
    })


def settings_text(config, tmp_path: Path, n_layers: int = 3) -> str:
    builder = AFMSimulation(config, output_dir=str(tmp_path / 'out'))
    builder.output_dir_layer[n_layers] = builder.output_dir / f'L{n_layers}'
    builder._generate_potentials(n_layers)
    path = builder.output_dir / f'L{n_layers}' / 'lammps' / 'system.in.settings'
    text = path.read_text(encoding='utf-8')
    return text.replace(str(tmp_path / 'out' / 'provenance' / 'potentials'), '<POT>')


def _deepmd_sheet(tmp_path, **overrides):
    # dispersion='d3': AFM sheets here are multilayer, and a multilayer MLIP
    # stack without D3 is rejected at config time (the bare model has no
    # interlayer binding -- test_config.TestMultilayerMLIPRequiresD3). These
    # tests are about CELL topology, not dispersion.
    return {'pot_type': 'deepmd', 'pot_path': _model(tmp_path, 'dpa.pth'),
            'lat_c': 6.112, 'dispersion': 'd3', **overrides}


# ---------------------------------------------------------------------------
# Regression anchor: a classical AFM cell is untouched
# ---------------------------------------------------------------------------

def test_all_sw_afm_cell_still_builds_its_settings_file(tmp_path):
    """No MLIP anywhere -> the pre-MLIP hybrid SW/LJ deck, unchanged."""
    text = settings_text(make_config(tmp_path), tmp_path)

    assert 'pair_style hybrid sw sw sw sw sw lj/cut 11.0' in text
    assert ('pair_coeff * * sw 1 <POT>/Si.sw Si Si Si NULL NULL NULL NULL NULL '
            'NULL NULL NULL NULL NULL # sub') in text
    assert ('pair_coeff * * sw 2 <POT>/Si.sw NULL NULL NULL Si Si Si NULL NULL '
            'NULL NULL NULL NULL NULL # tip') in text
    # Per-layer SW instances and interlayer LJ: classical behaviour is intact.
    assert '# sheet Layer 1' in text and '# sheet Layer 3' in text
    assert '# Mo(L1)-Mo(L2)' in text
    assert 'deepmd' not in text and 'mace' not in text


# ---------------------------------------------------------------------------
# Rejections (v4 spec §3: no partial cell; §6: no multi-component full cell)
# ---------------------------------------------------------------------------

def _deepmd_sheet_cell(tmp_path):
    """v3 setups 3 and 4, both struck by E2/E3."""
    return make_config(tmp_path, _deepmd_sheet(tmp_path))


def _mace_sheet_cell(tmp_path):
    return make_config(tmp_path, {
        'pot_type': 'mace',
        'pot_path': _model(tmp_path, 'mace.json'),
        'lat_c': 6.112, 'dispersion': 'd3',
    })


def _mlip_tip_alone_cell(tmp_path):
    """The mirror image: classical sheet, MLIP tip."""
    return make_config(
        tmp_path,
        tip_overrides={'pot_type': 'deepmd', 'pot_path': _model(tmp_path, 'dpa.pth')},
    )


def _d3_beside_sw_cell(tmp_path):
    return make_config(tmp_path, _deepmd_sheet(tmp_path, dispersion='d3'))


@pytest.mark.parametrize('cell', [_deepmd_sheet_cell, _mace_sheet_cell,
                                  _mlip_tip_alone_cell, _d3_beside_sw_cell],
                         ids=['deepmd-sheet', 'mace-sheet', 'tip-alone', 'd3-beside-sw'])
def test_partial_cell_mlip_is_rejected(tmp_path, cell):
    with pytest.raises(ValueError, match='Partial-cell MLIP is not available'):
        settings_text(cell(tmp_path), tmp_path)


def test_the_rejection_names_the_doc_and_the_lammps_errors(tmp_path):
    """A user hitting this must be able to see it is measured, not a policy."""
    config = make_config(tmp_path, _deepmd_sheet(tmp_path))
    with pytest.raises(ValueError) as exc:
        settings_text(config, tmp_path)
    message = str(exc.value)
    assert 'documentation/mlip.md' in message
    assert 'Pair hybrid sub-style deepmd is not used' in message
    assert 'pair_deepmd.cpp:816' in message
    assert '#1314' in message


def test_two_different_models_of_one_framework_are_rejected(tmp_path):
    """One pair_style token carries one model path per framework."""
    config = make_config(
        tmp_path,
        _deepmd_sheet(tmp_path),
        tip_overrides={'pot_type': 'deepmd',
                       'pot_path': _model(tmp_path, 'other.pth')},
        sub_overrides={'pot_type': 'deepmd',
                       'pot_path': _model(tmp_path, 'other.pth')},
    )
    with pytest.raises(ValueError, match='Two different deepmd model files'):
        settings_text(config, tmp_path)


def test_a_uniformly_deepmd_afm_cell_emits_one_full_map(tmp_path):
    """v6 §3/§4: tip + sheet + substrate on one MLIP is now buildable.

    This is the capability v4 §6 recorded as missing. The cell has 13 types --
    sub Si x3 and tip Si x3 (Langevin splits each into mobile/fix/thermo),
    a 3-layer MoS2 sheet as Mo(L1..L3) then S(L1..L3) (``_assign_types`` is
    element-first), and the virtual drive atom last. ONE ``pair_coeff * *``
    names a real element for every one of them: no NULL, no per-component
    instance, and no inter-component lj/cut, because the model supplies those
    pairs itself (v4 §2 applied between components).
    """
    config = make_config(
        tmp_path,
        _deepmd_sheet(tmp_path),
        tip_overrides={'pot_type': 'deepmd',
                       'pot_path': _model(tmp_path, 'dpa.pth')},
        # The helper's default substrate is AMORPHOUS, which v6 §5 refuses
        # under an MLIP (melt-quench is generation-time MD and cannot evaluate
        # the model). A non-finite-sheet AFM config requires either an
        # amorphous substrate OR a lattice-matched crystalline one, so
        # `amorph='c', match=True` is the supported MLIP route -- and the
        # reason rule 3c does not make AFM+MLIP unreachable.
        sub_overrides={'pot_type': 'deepmd',
                       'pot_path': _model(tmp_path, 'dpa.pth'),
                       'amorph': 'c', 'match': True},
    )
    config.tip.pot_path = config.sheet.pot_path
    config.sub.pot_path = config.sheet.pot_path

    text = settings_text(config, tmp_path)
    lines = text.splitlines()

    assert ('pair_coeff * * deepmd Si Si Si Si Si Si Mo Mo Mo S S S Mo '
            '# sub, tip, sheet') in lines
    # Exactly one deepmd instance, never a numbered hybrid sub-style.
    assert len([l for l in lines if l.startswith('pair_coeff * * deepmd')]) == 1
    assert 'deepmd 1' not in text
    assert 'NULL' not in text
    assert 'lj/cut' not in text
    # The virtual drive atom takes the placeholder element and is excluded from
    # the graph rather than left NULL (v4 E6).
    assert 'neigh_modify exclude type 13 13' in lines
    # D3 overlays the same 13 types.
    assert ('pair_coeff * * dispersion/d3 '
            'Si Si Si Si Si Si Mo Mo Mo S S S Mo') in lines


def test_amorphous_substrate_is_refused_under_an_mlip(tmp_path):
    """v6 §5, and a real limitation worth pinning.

    The AFM examples ship an AMORPHOUS substrate (`examples/afm_config.ini`,
    `afm_flake_config.ini`, `indent_campaign.ini`); only the tip is
    crystalline. So the standard AFM configuration cannot host an MLIP until
    the build-time/production potential split exists -- amorphisation is
    melt-quench MD run at GENERATION time and cannot evaluate the model. The
    error must say so rather than emit a deck.
    """
    config = make_config(
        tmp_path,
        _deepmd_sheet(tmp_path),
        tip_overrides={'pot_type': 'deepmd',
                       'pot_path': _model(tmp_path, 'dpa.pth')},
        sub_overrides={'pot_type': 'deepmd',
                       'pot_path': _model(tmp_path, 'dpa.pth')},  # amorph='a'
    )
    config.tip.pot_path = config.sheet.pot_path
    config.sub.pot_path = config.sheet.pot_path

    with pytest.raises(ValueError) as exc:
        settings_text(config, tmp_path)
    message = str(exc.value)
    assert "amorph = 'a'" in message
    assert 'sub' in message
    assert 'documentation/mlip.md' in message
