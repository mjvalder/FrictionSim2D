"""MLIP support across the builder families, under the v4 capability matrix.

The deliverable used to be "an MLIP ``pot_type`` works in *every* family". The
executed Milestone-0 results narrow that (see
``docs/superpowers/specs/2026-08-11-mlip-post-m0-design.md``): a partitioned
cell needs a NULL-padded ``pair_coeff`` type map, and **both** frameworks reject
one (§1, E2/E3 for DeePMD; ACEsuit #1314 for MACE). The builders register one
MLIP instance per *component*, so any multi-component cell is a NULL-map
topology however uniform its ``pot_type`` looks (§6).

What is left, and what this file tests:

* **single-component sheet families** (sheetonsheet, pes-sheet) build;
* **multi-component families** (afm, pes-tip, indent, hetero, flake) are
  refused up front, with a message pointing at the spec.

The building tests run a **full** ``build()``. That is the real proof that this
machine, which has no MLIP-enabled python-LAMMPS, can still *generate* MLIP
inputs: the multilayer stacking step falls back to the ``pair_style zero``
surrogate.
"""

import shutil
from pathlib import Path

import pytest

from frictionsim2d.core.config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAT = REPO_ROOT / 'examples' / 'materials'
POT = REPO_ROOT / 'examples' / 'potentials' / 'sw'

#: The flake/hetero builders assemble their stacks by shelling out to LAMMPS.
LMP = shutil.which('lmp_serial') or shutil.which('lmp')
needs_lmp = pytest.mark.skipif(
    LMP is None, reason='lmp_serial not on PATH (builder assembles via LAMMPS)')


def _model(tmp_path: Path, name: str = 'dpa.pth') -> str:
    path = tmp_path / name
    path.write_bytes(b"\x80\x02not-utf8\xff")
    return str(path)


def _deepmd_sheet(tmp_path, mat='h-MoS2', layers=(2,), x=20.0, y=20.0,
                  model=None):
    return {
        'mat': mat, 'pot_type': 'deepmd',
        'pot_path': model or _model(tmp_path),
        'cif_path': str(MAT / f'{mat}.cif'),
        'x': x, 'y': y, 'layers': list(layers),
        'lat_c': 6.112,
        # Multilayer MLIP stacks require D3 -- the bare model has no interlayer
        # binding (test_config.TestMultilayerMLIPRequiresD3). These tests are
        # about builder-family routing, not dispersion.
        'dispersion': 'd3',
    }


def _si(kind):
    si = {'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(POT / 'Si.sw'),
          'cif_path': str(MAT / 'Si.cif')}
    if kind == 'tip':
        return {**si, 'r': 5.0, 'dspring': 0.1}
    return {**si, 'thickness': 10.0, 'amorph': 'a'}


def _si_mlip(kind, model):
    """An all-MLIP tip/substrate on the SAME model as the sheet (v6 §4).

    The substrate is crystalline and lattice-matched: `amorph='a'` is refused
    under an MLIP (v6 §5 -- melt-quench is generation-time MD), and a
    non-finite-sheet AFM cell requires one form or the other.
    """
    si = {'mat': 'Si', 'pot_type': 'deepmd', 'pot_path': model,
          'cif_path': str(MAT / 'Si.cif')}
    if kind == 'tip':
        return {**si, 'r': 5.0, 'dspring': 0.1}
    return {**si, 'thickness': 10.0, 'amorph': 'c', 'match': True}


def _build(builder_cls, config, tmp_path):
    """Run a real build under a campaign root and return the settings text."""
    root = tmp_path / 'root'
    builder = builder_cls(config, output_dir=str(root / 'run'))
    builder.set_base_output_dir(root)
    builder.build()
    written = sorted((root / 'run').rglob('system.in.settings'))
    assert written, 'build produced no system.in.settings'
    return '\n'.join(p.read_text(encoding='utf-8') for p in written)


# ---------------------------------------------------------------------------
# Single-component sheet families: the supported topology
# ---------------------------------------------------------------------------

def test_pes_sheet_full_cell_mace(tmp_path):
    from frictionsim2d.builders.sheetonsheet.pes_sheet import PESSheetSimulation

    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 0.0},
        '2D': {**_deepmd_sheet(tmp_path, layers=(2,), x=30.0, y=30.0),
               'pot_type': 'mace',
               'pot_path': _model(tmp_path, 'mace-mp.json')},
        'pes': {'grid_n': 6},
        'settings': load_settings().model_dump(),
    })
    text = _build(PESSheetSimulation, config, tmp_path)

    assert 'symmetrix/mace' in text
    assert 'pair_coeff * * symmetrix/mace potentials/mace-mp.json' in text
    # A static scan registers no drive atom, so nothing needs the placeholder.
    assert 'Virtual' not in text
    assert 'NULL' not in text


def test_pes_sheet_mace_deck_has_directives_and_no_tiled(tmp_path):
    """The PES scan decks get the MLIP deck directives and never comm_style tiled.

    ``PESGridMixin.write_scan_scripts`` builds its own render context, so the
    MLIP keys have to be threaded through it explicitly (a missing key renders
    an MLIP deck without ``atom_modify map``/``newton on`` and with ``tiled``).
    """
    from frictionsim2d.builders.sheetonsheet.pes_sheet import PESSheetSimulation

    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 0.0},
        '2D': {**_deepmd_sheet(tmp_path, layers=(2,), x=30.0, y=30.0),
               'pot_type': 'mace',
               'pot_path': _model(tmp_path, 'mace-mp.json')},
        'pes': {'grid_n': 6},
        'settings': load_settings().model_dump(),
    })
    builder = PESSheetSimulation(config, output_dir=str(tmp_path / 'root' / 'run'))
    builder.set_base_output_dir(tmp_path / 'root')
    builder.build()

    decks = list((tmp_path / 'root').rglob('slide*.in'))
    assert decks
    for deck in decks:
        text = deck.read_text(encoding='utf-8')
        assert 'atom_modify' in text and 'newton' in text
        assert 'comm_style      tiled' not in text and 'rcb' not in text


def test_pes_sheet_full_cell_deepmd(tmp_path):
    """The DeePMD half of the same row -- one instance, full element list."""
    from frictionsim2d.builders.sheetonsheet.pes_sheet import PESSheetSimulation

    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 0.0},
        '2D': _deepmd_sheet(tmp_path, layers=(2,), x=30.0, y=30.0),
        'pes': {'grid_n': 6},
        'settings': load_settings().model_dump(),
    })
    text = _build(PESSheetSimulation, config, tmp_path)

    # A multilayer MLIP stack carries dispersion='d3', so the header is
    # hybrid/overlay; assert the deepmd sub-style and its staged model path
    # rather than a bare `pair_style deepmd`.
    assert 'deepmd potentials/dpa.pth' in text
    assert 'dispersion/d3' in text
    # E3 forbids a type-range coeff line; E7 forbids omitting the element list.
    assert 'pair_coeff * * deepmd Mo Mo S S' in text or 'pair_coeff * * Mo Mo S S' in text
    assert 'NULL' not in text
    assert 'deepmd 1' not in text


def test_sheetonsheet_full_cell_deepmd(tmp_path):
    from frictionsim2d.builders.sheetonsheet import SheetOnSheetSimulation

    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D': _deepmd_sheet(tmp_path, layers=(3,), x=20.0, y=20.0),
        'settings': load_settings().model_dump(),
    })
    text = _build(SheetOnSheetSimulation, config, tmp_path)

    assert 'deepmd potentials/dpa.pth' in text
    assert 'NULL' not in text
    assert 'deepmd 1' not in text


# ---------------------------------------------------------------------------
# Multi-component families: refused (v4 §3 / §6)
# ---------------------------------------------------------------------------

def test_pes_tip_deepmd_sheet_is_refused(tmp_path):
    """A tip PES scan is tip + sheet + substrate -- a partitioned cell."""
    from frictionsim2d.builders.afm.pes_tip import PESTipSimulation

    config = AFMSimulationConfig(**{
        'general': {'temp': 300.0, 'force': 5.0, 'scan_speed': 2.0},
        'tip': _si('tip'), 'sub': _si('sub'),
        '2D': _deepmd_sheet(tmp_path, layers=(2,)),
        'pes': {'grid_n': 6},
        'settings': load_settings().model_dump(),
    })
    with pytest.raises(ValueError, match='Partial-cell MLIP is not available'):
        _build(PESTipSimulation, config, tmp_path)


def test_indent_deepmd_sheet_is_refused(tmp_path):
    from frictionsim2d.builders.afm.indent import IndentSimulation

    config = AFMSimulationConfig(**{
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': _si('tip'), 'sub': _si('sub'),
        '2D': _deepmd_sheet(tmp_path, layers=(2,)),
        'indent': {},
        'settings': load_settings().model_dump(),
    })
    with pytest.raises(ValueError, match='Partial-cell MLIP is not available'):
        _build(IndentSimulation, config, tmp_path)


@needs_lmp
def test_hetero_sheetonsheet_deepmd_builds_as_one_full_cell(tmp_path):
    """The headline hetero case, unblocked by v6 §3.

    MoS2-on-WS2 has no reliable cross-material SW parametrization, and one
    foundation model covers both. It needs a single instance whose type map
    spans both materials -- which is exactly what the builders now emit.
    """
    from frictionsim2d.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation

    model = _model(tmp_path)
    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D-1': _deepmd_sheet(tmp_path, mat='h-MoS2', layers=(2,),
                              x=24.0, y=24.0, model=model),
        '2D-2': _deepmd_sheet(tmp_path, mat='h-WS2', layers=(2,),
                              x=24.0, y=24.0, model=model),
        'settings': load_settings().model_dump(),
    })
    text = _build(HeteroSheetOnSheetSimulation, config, tmp_path)
    assert 'pair_style hybrid/overlay deepmd' in text
    assert any(l.startswith('pair_coeff * * dispersion/d3') for l in text.splitlines())
    coeffs = [l for l in text.splitlines()
              if l.startswith('pair_coeff') and 'dispersion/d3' not in l]
    assert len(coeffs) == 1, coeffs
    # One real element per type, spanning BOTH materials.
    assert 'Mo' in coeffs[0] and 'W' in coeffs[0] and 'S' in coeffs[0]
    assert 'NULL' not in text
    assert 'lj/cut' not in text
    assert 'deepmd 1' not in text


@needs_lmp
def test_hetero_rejects_mace_beside_a_classical_material(tmp_path):
    """An MLIP may not share a hetero cell with SW."""
    from frictionsim2d.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation

    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D-1': {**_deepmd_sheet(tmp_path, mat='h-MoS2', layers=(2,),
                                 x=24.0, y=24.0),
                 'pot_type': 'mace',
                 'pot_path': _model(tmp_path, 'mace-mp.json')},
        '2D-2': {'mat': 'h-WS2', 'pot_type': 'sw',
                 'pot_path': str(POT / 'sw_lammps' / 't-WS2.sw'),
                 'cif_path': str(MAT / 'h-WS2.cif'),
                 'x': 24.0, 'y': 24.0, 'layers': [2]},
        'settings': load_settings().model_dump(),
    })
    with pytest.raises(ValueError, match='Partial-cell MLIP is not available'):
        _build(HeteroSheetOnSheetSimulation, config, tmp_path)


@needs_lmp
def test_flake_slide_deepmd_builds_as_one_full_cell(tmp_path):
    """Sheet + flake is two components, and one instance now spans both."""
    from frictionsim2d.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    model = _model(tmp_path)
    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D': _deepmd_sheet(tmp_path, mat='h-MoS2', layers=(2,),
                            x=24.0, y=24.0, model=model),
        'Flake': {'shape': 'hexagon', 'edge_length': 5.0,
                  'corner_radius': 3.5, 'corner_spring_k': 10.0,
                  'mat': 'h-MoS2', 'pot_type': 'deepmd', 'pot_path': model,
                  'cif_path': str(MAT / 'h-MoS2.cif')},
        'settings': load_settings().model_dump(),
    })
    text = _build(FlakeSlideSimulation, config, tmp_path)
    coeffs = [l for l in text.splitlines()
              if l.startswith('pair_coeff') and 'dispersion/d3' not in l]
    assert len(coeffs) == 1, coeffs
    # The 2-layer sheet registers n_layers=2 (element-first: Mo L1, Mo L2,
    # S L1, S L2 -- matching the stacked sheet file's per-layer types), then
    # flake Mo,Mo(corner),S,S(corner), then the virtual atom taking the
    # placeholder element.
    assert coeffs[0] == 'pair_coeff * * deepmd Mo Mo S S Mo Mo S S Mo # sheet, flake'
    assert 'pair_style hybrid/overlay deepmd' in text
    assert 'dispersion/d3' in text
    assert 'NULL' not in text
    assert 'lj/cut' not in text
    assert 'deepmd 1' not in text


# ---------------------------------------------------------------------------
# hetero Virtual-type gating (virtual-atom placeholder dependency)
# ---------------------------------------------------------------------------

def test_hetero_virtual_type_is_gated_on_the_drive_method():
    """``hetero.py`` registers the Virtual type only when asked to.

    The plan called the registration unconditional and a latent bug; it is not.
    Both sites take ``reserve_virtual_type``, which callers derive from
    ``drive_method`` (``hetero_slide.py``/``hetero_afm.py``). This pins the
    contract directly on the two registration functions so it cannot regress
    into the unconditional form the placeholder mechanism assumes.
    """
    import inspect

    from frictionsim2d.builders.construction import hetero_stack as hetero

    for func in (hetero.register_hetero_potentials,
                 hetero.register_hetero_flake_potentials):
        signature = inspect.signature(func)
        assert signature.parameters['reserve_virtual_type'].default is False
        source = inspect.getsource(func)
        assert 'if reserve_virtual_type:\n        pm.register_virtual_atom()' in source


# ---------------------------------------------------------------------------
# Multi-component families on ONE MLIP: buildable from v6
# ---------------------------------------------------------------------------

def _all_mlip_afm_like(tmp_path, extra):
    model = _model(tmp_path)
    return AFMSimulationConfig(**{
        'general': {'temp': 300.0, 'force': 5.0, 'scan_speed': 2.0},
        'tip': _si_mlip('tip', model), 'sub': _si_mlip('sub', model),
        '2D': _deepmd_sheet(tmp_path, layers=(2,), model=model),
        'settings': load_settings().model_dump(),
        **extra,
    })


def _assert_one_full_map(text):
    """One MLIP instance, a real element per type, no NULL and no cross LJ."""
    coeffs = [l for l in text.splitlines()
              if l.startswith('pair_coeff') and 'dispersion/d3' not in l]
    assert len(coeffs) == 1, coeffs
    assert 'Si' in coeffs[0] and 'Mo' in coeffs[0] and 'S' in coeffs[0]
    assert 'NULL' not in text
    assert 'lj/cut' not in text
    assert 'deepmd 1' not in text
    return coeffs[0]


def test_pes_tip_all_deepmd_builds_as_one_full_cell(tmp_path):
    """`pes-tip` inherits the AFM build path, so v6 unblocks it for free."""
    from frictionsim2d.builders.afm.pes_tip import PESTipSimulation

    config = _all_mlip_afm_like(tmp_path, {'pes': {'grid_n': 6}})
    _assert_one_full_map(_build(PESTipSimulation, config, tmp_path))


def test_indent_all_deepmd_builds_as_one_full_cell(tmp_path):
    """Same build path again -- indent is AFM_LIKE too."""
    from frictionsim2d.builders.afm.indent import IndentSimulation

    config = _all_mlip_afm_like(tmp_path, {'indent': {}})
    _assert_one_full_map(_build(IndentSimulation, config, tmp_path))


def test_amorphous_substrate_is_refused_in_the_afm_like_family(tmp_path):
    """v6 §5 applies to every AFM-like builder, not just `run afm`."""
    from frictionsim2d.builders.afm.indent import IndentSimulation

    model = _model(tmp_path)
    config = AFMSimulationConfig(**{
        'general': {'temp': 300.0, 'force': 5.0, 'scan_speed': 2.0},
        'tip': _si_mlip('tip', model),
        'sub': {**_si_mlip('sub', model), 'amorph': 'a', 'match': False},
        '2D': _deepmd_sheet(tmp_path, layers=(2,), model=model),
        'indent': {},
        'settings': load_settings().model_dump(),
    })
    with pytest.raises(ValueError) as exc:
        _build(IndentSimulation, config, tmp_path)
    assert "amorph = 'a'" in str(exc.value)
