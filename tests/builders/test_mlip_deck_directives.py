"""Every MLIP-capable builder must emit the style's mandatory deck directives.

`MLIPSpec.deck_directives` carries lines the pair style REQUIRES before
`read_data` -- symmetrix/mace needs `atom_modify map yes` (hard error at
`pair_symmetrix_mace.cpp:172` without it) and `newton on` (its Kokkos variant
demands `newton pair on`). Five templates render them:

    afm/system_init.lmp, afm/indent.lmp, afm/pes_scan.lmp,
    hetero/slide.lmp, flake/slide.lmp

...each behind `{% for d in mlip_deck_directives | default([]) %}`. That
`default([])` is what makes this worth a test: when a builder forgets to supply
the key, Jinja does NOT raise -- it renders nothing, and the omission surfaces
only when LAMMPS aborts on a cluster.

Before v6 this was latent: AFM/hetero/flake cells could not host an MLIP at all,
so the loop was unreachable. v6 made those cells legal without wiring the
context, so every generated AFM/indent/pes-tip MACE deck was unrunnable.
Measured 2026-09-09 on a generated 467-atom cell: it died at
`pair_symmetrix_mace.cpp:172`, and adding exactly these two lines by hand made
it evaluate (`PotEng = -2758.7689 eV`).
"""

from pathlib import Path

import pytest

from src.core.mlip import get_mlip

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = REPO_ROOT / 'src' / 'templates'

#: Generation-time templates: LAMMPS runs FrictionSim2D performs *while
#: building* (stacking layers, assembling hetero/flake cells, amorphising).
#: They never load the real MLIP -- `PotentialManager.write_surrogate_file`
#: swaps in `pair_style zero` so generation needs no MLIP-enabled LAMMPS and no
#: GPU (v4 §3.3). No MLIP is loaded, so no MLIP directive is required. That the
#: family tests build hetero/flake cells from a deliberately unreadable
#: checkpoint and still pass is the standing proof this holds.
SURROGATE_TEMPLATES = {
    'common/apply_langevin_regions.lmp',
    'common/apply_sheet_edge_types.lmp',
    'common/build_substrate.lmp',
    'common/build_tip.lmp',
    'common/make_amorphous.lmp',
    'common/stack_layers.lmp',
    'hetero/stack_hetero.lmp',
}

#: Derived, NOT hand-listed: every production template that builds a box or
#: reads a data file needs the directives, because that is exactly the point at
#: which symmetrix/mace demands them. An earlier hand-maintained list silently
#: omitted `afm/slide.lmp` -- the AFM *sliding* deck, the production model --
#: and the omission was invisible until a cluster run aborted. Deriving the set
#: means a new template joins these tests by existing.
DIRECTIVE_TEMPLATES = sorted(
    str(p.relative_to(TEMPLATES))
    for p in TEMPLATES.rglob('*.lmp')
    if str(p.relative_to(TEMPLATES)) not in SURROGATE_TEMPLATES
    and any(l.strip().startswith(('read_data', 'create_box'))
            for l in p.read_text(encoding='utf-8').splitlines())
)


def test_the_directive_template_set_is_not_empty():
    """Guard the guard: a glob matching nothing would pass every test below."""
    assert len(DIRECTIVE_TEMPLATES) >= 9, DIRECTIVE_TEMPLATES
    assert 'afm/slide.lmp' in DIRECTIVE_TEMPLATES


def test_surrogate_templates_all_exist():
    """A renamed template must not silently fall out of the exemption list."""
    missing = [r for r in SURROGATE_TEMPLATES if not (TEMPLATES / r).exists()]
    assert not missing, f"stale SURROGATE_TEMPLATES entries: {missing}"


@pytest.mark.parametrize('rel', DIRECTIVE_TEMPLATES)
def test_template_renders_the_directives_before_read_data(rel):
    """The loop must sit above the box/read_data, or the directives are useless.

    `atom_modify` and `newton` are rejected by LAMMPS once a box exists, which
    is precisely why the registry owns them rather than the caller appending
    them wherever convenient.
    """
    lines = (TEMPLATES / rel).read_text(encoding='utf-8').splitlines()

    def first_line(pred):
        for i, line in enumerate(lines):
            # Comments mention these words in prose (indent.lmp cites
            # "postprocessing/read_data.py" 13 lines above the real command),
            # so match COMMANDS only.
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('{#'):
                continue
            if pred(stripped):
                return i
        return None

    loop = first_line(lambda s: 'mlip_deck_directives' in s)
    assert loop is not None, f"{rel} lost its mlip_deck_directives loop"

    for anchor in ('read_data', 'create_box'):
        pos = first_line(lambda s, a=anchor: s.startswith(a))
        if pos is not None:
            assert loop < pos, (
                f"{rel}: the directives loop is at line {loop + 1} but "
                f"{anchor!r} is at line {pos + 1} -- LAMMPS rejects "
                f"atom_modify/newton once a box exists, so it must come first"
            )


def test_mace_declares_the_two_directives():
    """Pin the contract the templates depend on."""
    directives = get_mlip('mace').deck_directives
    joined = ' '.join(directives)
    assert 'atom_modify' in joined and 'map yes' in joined
    assert 'newton' in joined and 'on' in joined
    # deepmd needs neither; an empty tuple must stay empty.
    assert get_mlip('deepmd').deck_directives == ()


MACE_MODEL_JSON = (
    '{"num_channels": 128, "r_cut": 6.0, '
    '"atomic_numbers": [14, 16, 42], "num_elements": 3}'
)


def _afm_mlip_config(tmp_path):
    """A minimal all-MACE AFM cell: Si tip + lattice-matched Si sub + MoS2."""
    from src.core.config import write_config

    model = tmp_path / 'mace-test-Mo-S-Si.json'
    model.write_text(MACE_MODEL_JSON, encoding='utf-8')
    mats = REPO_ROOT / 'examples' / 'materials'
    mlip = {'pot_type': 'mace', 'pot_path': str(model)}
    raw = {
        '2D': {'mat': 'h-MoS2', 'cif_path': str(mats / 'h-MoS2.cif'),
               'x': 25.0, 'y': 25.0, 'layers': [1], 'stack_type': 'AB',
               'lat_c': 6.717, 'dispersion': 'd3', **mlip},
        # amorph 'c' + match: an MLIP refuses an amorphous tip/substrate (the
        # melt-quench build needs forces the generating machine cannot supply),
        # and a non-finite-sheet AFM refuses a non-matched crystalline one.
        'tip': {'mat': 'Si', 'cif_path': str(mats / 'Si.cif'),
                'r': 6.0, 'dspring': 0.1, **mlip},
        'sub': {'mat': 'Si', 'cif_path': str(mats / 'Si.cif'),
                'thickness': 8.0, 'amorph': 'c', 'match': True, **mlip},
        'general': {'temp': 0.0, 'force': 10.0, 'scan_speed': 2.0},
    }
    return write_config(raw, tmp_path / 'afm_mlip.ini')


@pytest.mark.parametrize('model_name', ['afm', 'pes_tip', 'indent'])
def test_generated_afm_family_deck_carries_the_directives(model_name, tmp_path,
                                                          monkeypatch):
    """The regression, at the level a user meets it: the emitted deck.

    All three models share `AFMSimulation.build_render_context`, so one missing
    key broke every one of them -- and the template's `| default([])` meant the
    build still reported success.
    """
    from src.run.simulate import run_simulations

    monkeypatch.chdir(REPO_ROOT)
    created, _, _, _ = run_simulations(
        config_file=str(_afm_mlip_config(tmp_path)), model=model_name,
        settings_file=None, output_root=tmp_path / 'out', generate_hpc=False,
    )
    decks = [p for p in Path(created[0]).rglob('*.in')
             if p.name in ('system.in', 'slide.in')]
    assert decks, f"{model_name} generated no deck"

    init = next((d for d in decks if d.name == 'system.in'), decks[0])
    text = init.read_text(encoding='utf-8')
    assert 'atom_modify' in text and 'map yes' in text, (
        f"{model_name}: generated deck lacks `atom_modify map yes`; "
        f"symmetrix/mace aborts at pair_symmetrix_mace.cpp:172"
    )
    assert 'newton' in text, f"{model_name}: generated deck lacks `newton on`"


def test_generated_classical_deck_is_unchanged(tmp_path, monkeypatch):
    """A classical AFM cell must not gain the directives.

    Built here rather than from `examples/afm_config.ini`, which is a *template*
    (`{mat}` placeholders, a `materials_list` path relative to a working dir,
    and a 4-element scan_angle sweep spec) and is not directly runnable.
    """
    from src.core.config import write_config
    from src.run.simulate import run_simulations

    monkeypatch.chdir(REPO_ROOT)
    mats = REPO_ROOT / 'examples' / 'materials'
    pots = REPO_ROOT / 'examples' / 'potentials' / 'sw'
    si = {'mat': 'Si', 'cif_path': str(mats / 'Si.cif'),
          'pot_path': str(pots / 'Si.sw'), 'pot_type': 'sw'}
    raw = {
        '2D': {'mat': 'h-MoS2', 'cif_path': str(mats / 'h-MoS2.cif'),
               'pot_path': str(pots / 'MoS2_wen.sw'), 'pot_type': 'sw',
               'x': 25.0, 'y': 25.0, 'layers': [1], 'stack_type': 'AB'},
        'tip': {**si, 'r': 6.0, 'dspring': 0.1},
        'sub': {**si, 'thickness': 8.0, 'amorph': 'a'},
        'general': {'temp': 0.0, 'force': 10.0, 'scan_speed': 2.0},
    }
    config_path = write_config(raw, tmp_path / 'afm_sw.ini')
    created, _, _, _ = run_simulations(
        config_file=str(config_path), model='afm', settings_file=None,
        output_root=tmp_path / 'out', generate_hpc=False,
    )
    text = next(iter(Path(created[0]).rglob('system.in'))).read_text(
        encoding='utf-8')

    assert 'atom_modify' not in text
    assert 'symmetrix' not in text and 'deepmd' not in text


# ---------------------------------------------------------------------------
# The same failure mode, second key: `mlip_sheet`
# ---------------------------------------------------------------------------
#
# `mlip_sheet` gates the MPI decomposition commands `comm_style tiled` and
# `balance 1.0 rcb`. Under `pair_deepmd` these are not a tuning choice: tiled
# makes the pair style derive a bogus ghost count and try to allocate hundreds
# of GB (measured 2026-08-11, F2 in documentation/milestone0.md). Suppressing
# them costs a classical-free MLIP run nothing, because both frameworks run one
# MPI rank per GPU and tiled/RCB only pay off across many ranks.
#
# It is guarded as `{% if not mlip_sheet | default(false) %}`, so -- exactly as
# with `mlip_deck_directives` -- a builder that never supplies the key silently
# emits the commands. Before v6 only sheet-on-sheet could host an MLIP and only
# `sheetonsheet.py` supplied it; v6 made the AFM/hetero/flake families legal
# without extending the wiring.


def _jinja_guard_violations(rel):
    """Command lines not enclosed by an `mlip_sheet` condition.

    Walks the template tracking `{% if %}`/`{% elif %}`/`{% endif %}` nesting so
    a second or third occurrence lower in the file cannot hide behind a guard
    added around the first.
    """
    text = (TEMPLATES / rel).read_text(encoding='utf-8')
    bad, stack = [], []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        s = raw.strip()
        if s.startswith('{% if'):
            stack.append(s)
        elif s.startswith('{% elif'):
            if stack:
                stack[-1] = s
        elif s.startswith('{% endif'):
            if stack:
                stack.pop()
        if s.startswith('#') or s.startswith('{#'):
            continue
        is_cmd = (s.startswith('comm_style') and 'tiled' in s) or \
                 (s.startswith('balance') and 'rcb' in s)
        if is_cmd and not any('mlip_sheet' in c for c in stack):
            bad.append((lineno, s))
    return bad


#: Every template that issues a tiled/rcb decomposition command.
DECOMPOSITION_TEMPLATES = sorted(
    str(p.relative_to(TEMPLATES))
    for p in TEMPLATES.rglob('*.lmp')
    if any(l.strip().startswith(('comm_style', 'balance'))
           and ('tiled' in l or 'rcb' in l)
           for l in p.read_text(encoding='utf-8').splitlines())
)


def test_the_decomposition_template_set_is_not_empty():
    """Guard the guard: a glob that matches nothing would pass every test."""
    assert len(DECOMPOSITION_TEMPLATES) >= 8, DECOMPOSITION_TEMPLATES


@pytest.mark.parametrize('rel', DECOMPOSITION_TEMPLATES)
def test_decomposition_commands_are_gated_on_mlip_sheet(rel):
    violations = _jinja_guard_violations(rel)
    assert not violations, (
        f"{rel}: unguarded decomposition command(s) "
        f"{violations} -- under pair_deepmd `comm_style tiled` triggers a "
        f"hundreds-of-GB ghost allocation; wrap in "
        f"`{{% if not mlip_sheet | default(false) %}}`"
    )


@pytest.mark.parametrize('model_name', ['afm', 'pes_tip', 'indent'])
def test_generated_afm_family_deck_omits_tiled(model_name, tmp_path,
                                               monkeypatch):
    """The AFM family shares `system_init.lmp`, so one gap broke all three."""
    from src.run.simulate import run_simulations

    monkeypatch.chdir(REPO_ROOT)
    created, _, _, _ = run_simulations(
        config_file=str(_afm_mlip_config(tmp_path)), model=model_name,
        settings_file=None, output_root=tmp_path / 'out', generate_hpc=False,
    )
    for deck in Path(created[0]).rglob('*.in'):
        text = deck.read_text(encoding='utf-8')
        assert 'comm_style      tiled' not in text and \
               'comm_style tiled' not in text, (
            f"{model_name}/{deck.name}: emits `comm_style tiled` under an MLIP")
        assert 'balance         1.0 rcb' not in text, (
            f"{model_name}/{deck.name}: emits `balance 1.0 rcb` under an MLIP")


def test_generated_classical_deck_keeps_tiled(tmp_path, monkeypatch):
    """The suppression must be MLIP-only -- classical decks want the balance."""
    from src.core.config import write_config
    from src.run.simulate import run_simulations

    monkeypatch.chdir(REPO_ROOT)
    mats = REPO_ROOT / 'examples' / 'materials'
    pots = REPO_ROOT / 'examples' / 'potentials' / 'sw'
    si = {'mat': 'Si', 'cif_path': str(mats / 'Si.cif'),
          'pot_path': str(pots / 'Si.sw'), 'pot_type': 'sw'}
    raw = {
        '2D': {'mat': 'h-MoS2', 'cif_path': str(mats / 'h-MoS2.cif'),
               'pot_path': str(pots / 'MoS2_wen.sw'), 'pot_type': 'sw',
               'x': 25.0, 'y': 25.0, 'layers': [1], 'stack_type': 'AB'},
        'tip': {**si, 'r': 6.0, 'dspring': 0.1},
        'sub': {**si, 'thickness': 8.0, 'amorph': 'a'},
        'general': {'temp': 0.0, 'force': 10.0, 'scan_speed': 2.0},
    }
    created, _, _, _ = run_simulations(
        config_file=str(write_config(raw, tmp_path / 'afm_sw.ini')),
        model='afm', settings_file=None, output_root=tmp_path / 'out',
        generate_hpc=False,
    )
    text = next(iter(Path(created[0]).rglob('system.in'))).read_text(
        encoding='utf-8')
    assert 'comm_style' in text and 'tiled' in text


# ---------------------------------------------------------------------------
# hetero and flake: standalone builders, so they inherit neither fix
# ---------------------------------------------------------------------------
#
# `HeteroSheetOnSheetSimulation` and `FlakeSlideSimulation` subclass
# `SimulationBase` directly, not `SheetOnSheetSimulation`/`AFMSimulation`, so
# they assemble their own render context and pick up nothing from the wiring
# above. Their templates carry both guards; only the keys were missing.
# (`HeteroAFMSimulation` DOES subclass `AFMSimulation` and is covered by it.)

import shutil

from src.core.config import SheetOnSheetSimulationConfig
from src.core.config import load_settings

MAT = REPO_ROOT / 'examples' / 'materials'

needs_lmp = pytest.mark.skipif(
    shutil.which('lmp_serial') is None and shutil.which('lmp') is None,
    reason='lmp_serial not on PATH (builder assembles its stack via LAMMPS)')


def _opaque_model(tmp_path, name='dpa.pth'):
    """A stand-in checkpoint: never deserialised at generation time (v4 §3.3)."""
    path = tmp_path / name
    path.write_bytes(b"\x80\x02not-utf8\xff")
    return str(path)


def _deepmd_sheet(tmp_path, mat, model, x=16.0, y=16.0):
    return {
        'mat': mat, 'pot_type': 'deepmd', 'pot_path': model,
        'cif_path': str(MAT / f'{mat}.cif'),
        'x': x, 'y': y, 'layers': [2], 'lat_c': 6.112,
        # A multilayer MLIP stack requires D3; the bare model has no
        # interlayer binding.
        'dispersion': 'd3',
    }


def _build_deck(builder_cls, config, tmp_path):
    """Build for real and return the concatenated slide deck text."""
    root = tmp_path / 'root'
    builder = builder_cls(config, output_dir=str(root / 'run'))
    builder.set_base_output_dir(root)
    builder.build()
    decks = sorted(p for p in (root / 'run').rglob('*.in')
                   if p.name.startswith('slide'))
    assert decks, 'build produced no slide deck'
    return '\n'.join(p.read_text(encoding='utf-8') for p in decks)


def _assert_mlip_deck_context_applied(text, who):
    assert 'comm_style' not in text or 'tiled' not in text, (
        f"{who}: emits `comm_style tiled` under an MLIP -- pair_deepmd then "
        f"derives a bogus ghost count and tries to allocate hundreds of GB")
    assert 'balance         1.0 rcb' not in text, (
        f"{who}: emits `balance 1.0 rcb` under an MLIP")


@needs_lmp
def test_hetero_slide_deck_carries_the_mlip_context(tmp_path):
    from src.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation

    model = _opaque_model(tmp_path)
    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D-1': _deepmd_sheet(tmp_path, 'h-MoS2', model),
        '2D-2': _deepmd_sheet(tmp_path, 'h-WS2', model),
        'settings': load_settings().model_dump(),
    })
    text = _build_deck(HeteroSheetOnSheetSimulation, config, tmp_path)
    _assert_mlip_deck_context_applied(text, 'hetero')


@needs_lmp
def test_flake_slide_deck_carries_the_mlip_context(tmp_path):
    from src.builders.sheetonsheet.flake_slide import FlakeSlideSimulation

    model = _opaque_model(tmp_path)
    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        '2D': _deepmd_sheet(tmp_path, 'h-MoS2', model, x=24.0, y=24.0),
        'Flake': {'shape': 'hexagon', 'edge_length': 5.0,
                  'corner_radius': 3.5, 'corner_spring_k': 10.0,
                  'mat': 'h-MoS2', 'pot_type': 'deepmd', 'pot_path': model,
                  'cif_path': str(MAT / 'h-MoS2.cif')},
        'settings': load_settings().model_dump(),
    })
    text = _build_deck(FlakeSlideSimulation, config, tmp_path)
    _assert_mlip_deck_context_applied(text, 'flake')


@needs_lmp
def test_hetero_mace_slide_deck_carries_the_directives(tmp_path):
    """MACE is where the missing directives are fatal, not merely wasteful."""
    from src.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation

    model = tmp_path / 'mace-Mo-W-S.json'
    model.write_text(
        '{"num_channels": 128, "r_cut": 6.0, '
        '"atomic_numbers": [16, 42, 74], "num_elements": 3}', encoding='utf-8')
    sheets = {k: {**_deepmd_sheet(tmp_path, m, str(model)),
                  'pot_type': 'mace', 'pot_path': str(model)}
              for k, m in (('2D-1', 'h-MoS2'), ('2D-2', 'h-WS2'))}
    config = SheetOnSheetSimulationConfig(**{
        'general': {'temp': 300.0, 'pressure': 1.0},
        **sheets, 'settings': load_settings().model_dump(),
    })
    text = _build_deck(HeteroSheetOnSheetSimulation, config, tmp_path)
    assert 'atom_modify' in text and 'map yes' in text, (
        'hetero MACE deck lacks `atom_modify map yes`; symmetrix/mace aborts '
        'at pair_symmetrix_mace.cpp:172')
    assert 'newton' in text
