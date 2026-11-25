"""MLIP handling in the multilayer stacking step (spec §3.3).

``stack_multilayer_sheet`` runs LAMMPS **in-process** (the python ``lammps``
module) for every multilayer build and ``include``s a potential file derived
from the sheet's ``pot_type``. Left alone, merely *generating* MLIP inputs
would therefore require an MLIP-enabled python-LAMMPS on the generation
machine and in CI — which is not a thing we can assume.

For MLIP sheets the stacking step instead writes a ``pair_style zero``
surrogate: stock LAMMPS, defined for every element, needs no potential file
and exerts exactly zero force, so it cannot influence the stacked geometry.
"""

from pathlib import Path

import pytest

from frictionsim2d.builders.construction import components
from frictionsim2d.core.config import SheetConfig, load_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
MOS2_CIF = REPO_ROOT / 'frictionsim2d' / 'data' / 'materials' / 'h-MoS2.cif'
MOS2_SW = REPO_ROOT / 'frictionsim2d' / 'data' / 'potentials' / 'sw' / 'MoS2_wen.sw'


def _sheet(tmp_path, **overrides) -> SheetConfig:
    kwargs = {
        'mat': 'h-MoS2',
        'pot_type': 'sw',
        'pot_path': str(MOS2_SW),
        'cif_path': str(MOS2_CIF),
        'x': 20.0, 'y': 20.0, 'layers': [3],
    }
    kwargs.update(overrides)
    return SheetConfig(**kwargs)


def _deepmd_sheet(tmp_path, **overrides) -> SheetConfig:
    """A 3-layer deepmd sheet.

    ``lat_c`` is the INTERLAYER distance, so 6.112 A (the measured
    DPA-3.1/Alex2D + D3 equilibrium) -- not the in-plane 3.16 A, which is the
    single easiest thing to get wrong here. ``dispersion='d3'`` because a
    multilayer MLIP stack without it is rejected at config time: the bare model
    has no interlayer binding at all (see
    ``test_config.TestMultilayerMLIPRequiresD3``).
    """
    model = tmp_path / 'dpa.pth'
    model.write_bytes(b"\x80\x02not-utf8\xff")
    overrides.setdefault('dispersion', 'd3')
    overrides.setdefault('lat_c', 6.112)
    return _sheet(tmp_path, pot_type='deepmd', pot_path=str(model), **overrides)


def test_mlip_stacking_writes_a_pair_style_zero_surrogate(tmp_path):
    settings = load_settings()
    pot_file = tmp_path / 'stack.in.settings'
    components.build_stacking_potential_file(
        _deepmd_sheet(tmp_path), settings, layers_for_setup=3,
        pot_file=pot_file, edge_mode='none')

    text = pot_file.read_text(encoding='utf-8')
    assert 'pair_style zero 10.0' in text
    assert 'pair_coeff * *' in text
    # Masses (which LAMMPS requires for every type) survive.
    assert 'mass 1*3 95.950000 #Mo' in text
    assert 'mass 4*6 32.060000 #S' in text
    # The MLIP itself is never loaded during stacking.
    assert 'deepmd' not in text
    assert 'dpa.pth' not in text


def test_sw_stacking_still_writes_the_real_potential(tmp_path):
    """Regression: a classical sheet's stacking file is unchanged."""
    settings = load_settings()
    pot_file = tmp_path / 'stack.in.settings'
    components.build_stacking_potential_file(
        _sheet(tmp_path), settings, layers_for_setup=3,
        pot_file=pot_file, edge_mode='none')

    text = pot_file.read_text(encoding='utf-8')
    assert 'pair_style hybrid sw sw sw lj/cut 11.0' in text
    assert str(MOS2_SW) in text
    assert 'pair_style zero' not in text
    # Explicit interlayer LJ is part of the classical stacking file.
    assert '# Mo(L1)-Mo(L2)' in text


def test_mlip_stacking_refuses_to_minimize_for_lat_c(tmp_path):
    """MLIP sheets always carry lat_c, so the minimization branch is dead.

    If it is ever reached, the run would silently need an MLIP-enabled
    python-LAMMPS (or, worse, minimize under the zero surrogate and return a
    meaningless lat_c). Fail loudly instead.
    """
    settings = load_settings()
    config = _deepmd_sheet(tmp_path)
    base = tmp_path / 'layer.lmp'
    base.write_text('# dummy\n', encoding='utf-8')

    with pytest.raises(ValueError, match='lat_c'):
        components.stack_multilayer_sheet(
            base_layer_path=base,
            config=config,
            output_path=tmp_path / 'stacked.lmp',
            box_dims={'xlo': 0.0, 'xhi': 20.0, 'ylo': 0.0, 'yhi': 20.0,
                      'zlo': 0.0, 'zhi': 10.0},
            n_layers=3,
            types_per_layer=2,
            pot_counts={'Mo': 1, 'S': 1},
            supercell_dims={'x': 20.0, 'y': 20.0},
            lat_c=None,
            settings=settings,
        )


def test_mlip_stacking_data_file_carries_no_pair_coeffs(tmp_path):
    """The stacked structure must not embed a ``Pair Coeffs`` section.

    The MLIP stacking surrogate is ``pair_style zero``, which -- unlike the
    many-body classical styles -- HAS per-type coefficients, so LAMMPS's
    ``write_data`` happily serialises a ``Pair Coeffs # zero`` block into the
    structure file. The production script then does ``read_data`` BEFORE it
    ``include``s the settings file that defines ``pair_style``, and LAMMPS
    aborts with::

        ERROR: Must define pair_style before Pair Coeffs (read_data.cpp:782)

    Found by generating a real sheet-on-sheet MLIP run and executing it -- the
    component-level tests all passed, because the surrogate file and the
    settings file are each individually correct. ``write_data ... nocoeff``
    is the fix.
    """
    template = (Path(__file__).resolve().parents[2]
                / 'frictionsim2d' / 'templates' / 'common' / 'stack_layers.lmp')
    text = template.read_text(encoding='utf-8')
    write_lines = [ln for ln in text.splitlines() if ln.strip().startswith('write_data')]
    assert write_lines, 'stack_layers.lmp should write the stacked structure'
    for line in write_lines:
        assert 'nocoeff' in line, (
            f"{line!r} must pass 'nocoeff', or the pair_style zero surrogate "
            f"leaks a Pair Coeffs section into the stacked data file"
        )
