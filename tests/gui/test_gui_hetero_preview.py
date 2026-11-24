"""Previewing a heterostructure.

Two questions, and the second one only exists because of what the first one
answered.

**Does it work at all?** Yes -- and that had to be established by building one,
not by reading the code. Every one of the five modes routes a two-material
config to a hetero builder, and each of those runs a real LAMMPS pass to
assemble the stack. So the first class here builds one for real (no ovito stub,
no fake data file: an ovito stub is exactly what would let a broken preview
pass) and walks it through the same steps ``frictionsim2d.gui.preview_worker`` and
``PreviewController._deliver`` do.

All five modes are built, not a representative two. An earlier version of this
file built ``afm`` and ``sheetonsheet`` on the premise that the other three
emit the same assembly script; they do not. ``pes_tip`` writes
``afm/pes_scan.lmp`` to ``lammps/slide.in``, which sorts *before* ``system.in``
and so is what ``find_assembly_script`` picks -- a different template from
``afm``'s, without its optional ``flake`` placement. ``hetero/pes_scan.lmp`` is
likewise maintained as a deliberate duplicate of ``hetero/slide.lmp``. Four
distinct templates, two shapes, two of them duplicates kept in sync by hand:
exactly the situation where "it is the same script" ages badly. Each mode's
previewed script is named in ``EXPECTED`` so a re-route fails loudly instead of
silently previewing something else.

**Is what it then says about the result true?** Not quite, and that is the fix
this file carries. The two shapes are:

* the AFM family (``afm``, ``indent``, ``pes_tip``) -- an explicit box and one
  ``read_data`` per component: ``sub``, ``tip``, ``2D``;
* the sheet family (``sheetonsheet``, ``pes_sheet``) -- the whole stack
  assembled into a single ``hetero.data`` first, so the script reads exactly
  one file, under the LAMMPS group name ``all_atoms``.

That single component spans the coincidence cell by construction -- the cell
*is* the interface's only in-plane periodicity -- and the viewport's
periodic-image check treated every component not literally named ``sub`` or
``2D`` as one that should not be touching the wall. So a perfectly good hetero
sheet preview reported "all_atoms reaches the periodic boundary ... enlarge the
sheet footprint": false, and not even actionable, since a coincidence cell is
not something a user can enlarge.
"""

from __future__ import annotations

import os
import shutil
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('ovito')
pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from frictionsim2d.core.config import write_config  # noqa: E402
from frictionsim2d.gui.preview.assemble import (  # noqa: E402
    AssembledSystem, ComponentSlice, assemble,
)
from frictionsim2d.gui.preview.init_script import parse_assembly  # noqa: E402
from frictionsim2d.gui.preview.settings_file import parse_identity  # noqa: E402
from frictionsim2d.gui.preview_worker import run as run_preview_build  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALS = REPO_ROOT / 'examples' / 'materials'
POTENTIALS = REPO_ROOT / 'examples' / 'potentials' / 'sw'

#: `frictionsim2d.builders.hetero.run_lammps` shells out to this by name. Without it
#: the whole build half of this file is skipped -- see `_require_lammps`.
LMP_SERIAL = shutil.which('lmp_serial')

#: The AFM family's substrate is amorphous, and a preview is forbidden from
#: running a melt-quench (`allow_generate_amorphous=False`), so it can only be
#: built from this cached structure. Missing, the AFM-family modes are skipped
#: rather than failing on something that is not what they test.
#: Amorphous cache lives in package data, not the examples library.
AMORPHOUS_SI = (Path(str(resources.files('frictionsim2d.data'))) / 'materials' / 'amor_Si.lmp')

AFM_FAMILY = ('afm', 'indent', 'pes_tip')

#: model -> (previewed script's filename, component names in order).
#:
#: The filename is pinned deliberately: `find_assembly_script` picks by
#: *content* across `sorted(lammps/*.in)`, so a builder that starts writing an
#: extra script, or renames one, silently changes which template is previewed.
EXPECTED = {
    'sheetonsheet': ('slide.in', ['all_atoms']),
    'pes_sheet': ('slide.in', ['all_atoms']),
    'afm': ('system.in', ['sub', 'tip', '2D']),
    'indent': ('system.in', ['sub', 'tip', '2D']),
    'pes_tip': ('slide.in', ['sub', 'tip', '2D']),
}


def _require_lammps(model: str) -> None:
    """Skip loudly, naming what goes unverified without the prerequisite."""
    if LMP_SERIAL is None:
        pytest.skip(
            f"lmp_serial not on PATH: cannot assemble a heterostructure, so "
            f"'{model}' previews are UNVERIFIED here -- the central claim of "
            f"this file (a two-material config previews correctly in every "
            f"mode) is not being checked in this run.")
    if model in AFM_FAMILY and not AMORPHOUS_SI.exists():
        pytest.skip(
            f"{AMORPHOUS_SI} is missing and a preview may not run a "
            f"melt-quench, so the AFM-family hetero preview ('{model}') is "
            f"UNVERIFIED here.")


def _hetero_config(model: str) -> dict:
    """A two-material config for `model`, deliberately at its smallest.

    `h-MoS2` + `h-WS2` because they are commensurate well inside
    `[lattice_match] strain_tol` (2%): a pair that is not is refused by the
    builder, which is a different path from the one under test here.
    """
    config = {
        '2D-1': {
            'mat': 'h-MoS2',
            'cif_path': str(MATERIALS / 'h-MoS2.cif'),
            'pot_path': str(POTENTIALS / 'sw_lammps' / 'h-MoS2.sw'),
            'pot_type': 'sw', 'x': 12.0, 'y': 12.0, 'layers': [2],
        },
        '2D-2': {
            'mat': 'h-WS2',
            'cif_path': str(MATERIALS / 'h-WS2.cif'),
            'pot_path': str(POTENTIALS / 'sw_lammps' / 'h-WS2.sw'),
            'pot_type': 'sw', 'x': 12.0, 'y': 12.0, 'layers': [2],
        },
        # `pressure` is required for every `sheetonsheet` slide (plain or
        # hetero; `frictionsim2d.builders.sheetonsheet.base.SheetOnSheetSimulation.
        # REQUIRES_PRESSURE`) -- `run_simulations` refuses a config with none
        # before building anything (`frictionsim2d.run.expand.missing_pressure`). It is
        # set unconditionally here rather than only for `model == 'sheetonsheet'`
        # because it is harmless noise for the other four modes: the AFM family
        # loads via `[general] force`, not `pressure`, and `PESGridMixin`
        # (pes_sheet, pes_tip) sets `REQUIRES_PRESSURE = False` and never reads
        # the key at all.
        'general': {'temp': 300, 'scan_speed': 1, 'hetero_stacking': 'grouped',
                    'pressure': 1.0},
    }
    if model in AFM_FAMILY:
        # The AFM family stacks the hetero sheet on a substrate under a tip;
        # `layers = [1]` each keeps the replicated cell small.
        config['2D-1']['layers'] = [1]
        config['2D-2']['layers'] = [1]
        config['tip'] = {
            'mat': 'Si', 'pot_type': 'sw',
            'pot_path': str(POTENTIALS / 'Si.sw'),
            'cif_path': str(MATERIALS / 'Si.cif'), 'r': 10.0,
        }
        config['sub'] = {
            'mat': 'Si', 'pot_type': 'sw',
            'pot_path': str(POTENTIALS / 'Si.sw'),
            'cif_path': str(MATERIALS / 'Si.cif'),
            'amorph': 'a', 'thickness': 12.0,
        }
        config['general']['force'] = 10.0
        config['general']['temp'] = 1.0
    if model == 'pes_sheet':
        # A PES sheet scan is exactly a frozen bottom plus a scanned top.
        config['2D-1']['layers'] = [1]
        config['2D-2']['layers'] = [1]
        config['pes'] = {'grid_n': 4}
    if model == 'pes_tip':
        config['pes'] = {'grid_n': 3}
    return config


def _preview(model: str, output_root: Path):
    """Build and assemble a preview exactly as the GUI does.

    `preview_worker.run` is called rather than re-implemented: it is the
    build half of the real path (`PreviewController.start` spawns it in a
    subprocess), and the three lines after it are `_deliver`'s.

    Returns ``(system, init_script)``.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = write_config(_hetero_config(model), output_root / 'config.ini')
    result = run_preview_build({
        'config_file': str(config_path),
        'model': model,
        'output_root': str(output_root),
        'layer': None,
        'settings_file': None,
        'result_file': str(output_root / 'preview.json'),
    })
    assert result['ok'], result.get('error')

    init_script = Path(result['init_script'])
    assembly = parse_assembly(init_script, Path(result['simulation_root']))
    identity = (parse_identity(Path(result['settings_file']))
                if result.get('settings_file') else None)
    return assemble(assembly, identity), init_script


@pytest.fixture(scope='module', name='built')
def _built(tmp_path_factory):
    """Memoized real builds: each mode is assembled at most once per run."""
    root = tmp_path_factory.mktemp('hetero_preview')
    cache = {}

    def build(model: str):
        _require_lammps(model)
        if model not in cache:
            cache[model] = _preview(model, root / model)
        return cache[model]

    return build


class TestAHeterostructurePreviewActuallyBuilds:
    """Build one, for real, in every mode, and look at what came out."""

    @pytest.mark.parametrize('model', sorted(EXPECTED))
    def test_every_mode_previews_both_materials(self, model, built):
        """Both materials have to be visible, or it is a picture of one sheet.

        Checked through the legend, which `Identity.legend()` filters to the
        types that actually carry atoms -- so `{Mo, W, S}` cannot be satisfied
        by a stack that only got one material in, nor by the unassembled
        input.
        """
        system, init_script = built(model)
        script_name, components = EXPECTED[model]

        assert init_script.name == script_name, (
            f"{model} now previews {init_script.name}, not {script_name}: the "
            f"builder writes a different set of scripts than this file "
            f"assumes, so the shapes below may no longer hold")
        assert system.count > 0, 'the preview assembled no atoms at all'
        assert system.cell is not None, 'no simulation cell to frame the view on'
        assert [c.name for c in system.components] == components
        elements = set(system.legend.values())
        assert {'Mo', 'W', 'S'} <= elements, (
            f'both materials must be visible in the preview, got {elements}')
        assert not system.warnings, system.warnings

    def test_the_sheet_family_arrives_as_one_pre_assembled_file(self, built):
        """One component is the honest answer for the sheet family -- the
        script really does read a single `hetero.data` -- and every atom is
        in it."""
        system, _ = built('sheetonsheet')

        assert system.components[0].count == system.count

    def test_the_afm_family_keeps_its_components_apart(self, built):
        """Placed separately, so they stay separate: each owns its own block
        of LAMMPS types, which is what the `offset` on its `read_data` line
        buys and what keeps the substrate's silicon from being coloured as
        the tip's."""
        system, _ = built('afm')

        blocks = [set(c.type_ids) for c in system.components]
        for index, block in enumerate(blocks):
            for other in blocks[index + 1:]:
                assert not block & other, (
                    f'components share LAMMPS types {sorted(block & other)}; '
                    f'the read_data type offsets were lost')

    def test_the_afm_family_stacks_its_components_in_z(self, built):
        """The `shift` on each `read_data` line is what puts the tip above
        the sheet and the sheet above the substrate. Drop it and every
        component lands at its own native z -- the tip buried inside the
        substrate -- with the component list, the type blocks and the atom
        count all still perfectly correct.
        """
        system, _ = built('afm')
        zs = {c.name: system.positions[c.first:c.last, 2] for c in system.components}

        assert zs['2D'].min() > zs['sub'].max(), \
            'the sheet is not above the substrate; a read_data shift was lost'
        assert zs['tip'].min() > zs['2D'].max(), \
            'the tip is not above the sheet; a read_data shift was lost'


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(name='panel')
def _panel(qt_app, monkeypatch):
    """A real `ViewportPanel` with only the GL/pipeline layer stubbed.

    ovito itself is never stubbed -- the module is imported for real above --
    because a stub is what lets a broken viewport pass its own tests.
    """
    del qt_app
    from frictionsim2d.gui.preview import viewport as vp  # noqa: PLC0415

    monkeypatch.setattr(vp.ViewportPanel, '_update_pipeline', lambda self: None)
    monkeypatch.setattr(vp.ViewportPanel, '_ensure_widget', lambda self: None)
    return vp.ViewportPanel()


def _spanning(name: str, cell=(0.0, 12.0, 0.0, 11.0, -15.0, 30.0)) -> AssembledSystem:
    """One component filling the whole periodic cell, named `name`."""
    xlo, xhi, ylo, yhi = cell[:4]
    positions = np.array([
        [xlo, ylo, 0.0], [xhi, ylo, 0.0], [xlo, yhi, 0.0], [xhi, yhi, 0.0],
    ])
    return AssembledSystem(
        positions=positions,
        type_ids=np.array([1, 1, 2, 2]),
        group_ids=np.zeros(4, dtype=int),
        cell=cell,
        components=[ComponentSlice(name, 0, 0, 4, [1, 2])],
        legend={1: 'Mo', 2: 'W'},
    )


class TestTheViewportDoesNotInventAProblem:
    """A preview that succeeded must not be reported as if it had failed.

    The negative assertions here lean on `warning.isHidden()`, which is also
    True on a panel that was never shown -- so on their own they would pass
    even if `show_system` did nothing at all. The positives (`tip`, `flake`,
    and the real-geometry test below) are what make the class load-bearing:
    they can only pass if `show_system` really runs the check.
    """

    def test_a_full_cell_hetero_stack_is_not_called_self_interacting(self, panel):
        """The hetero sheet stack is one component, `all_atoms`, and it fills
        the coincidence cell because that cell *is* the interface's
        periodicity. Warning that it "reaches the periodic boundary" and
        advising the user to "enlarge the sheet footprint" is both false and
        impossible to act on."""
        panel.show_system(_spanning('all_atoms'))

        assert 'periodic boundary' not in panel.warning.text()
        assert panel.warning.isHidden(), \
            f'a clean hetero preview raised a warning: {panel.warning.text()!r}'

    def test_a_homogeneous_sheet_stack_is_not_called_self_interacting(self, panel):
        """The same false alarm reached the homogeneous sheet-on-sheet
        preview, whose single component is named `bot`."""
        panel.show_system(_spanning('bot'))

        assert panel.warning.isHidden(), panel.warning.text()

    def test_a_tip_that_reaches_the_wall_is_still_reported(self, panel):
        """The check exists for a reason and has to keep working: a tip wide
        enough to meet the wall interacts with its own periodic image, which
        is the failure the >=2x tip-diameter rule is there to prevent."""
        panel.show_system(_spanning('tip'))

        assert not panel.warning.isHidden(), \
            'a tip touching the periodic wall must still be reported'
        assert 'tip' in panel.warning.text()
        assert 'periodic boundary' in panel.warning.text()

    def test_a_flake_that_reaches_the_wall_is_still_reported(self, panel):
        """A flake is finite by definition, same as a tip."""
        panel.show_system(_spanning('flake'))

        assert not panel.warning.isHidden()
        assert 'flake' in panel.warning.text()

    def test_a_finite_sheet_can_be_declared_finite(self, panel):
        """`[general] finite_sheet` sizes the sheet as an island inside the
        substrate box, so there -- and only there -- a `2D` at the wall is the
        real >=2x failure. Nothing in an `AssembledSystem` records that, so
        `MainWindow` has to say it."""
        panel.set_finite_components(['tip', 'flake', '2D'])
        panel.show_system(_spanning('2D'))

        assert not panel.warning.isHidden()
        assert '2D' in panel.warning.text()

    def test_the_finite_list_can_be_put_back(self, panel):
        """Left standing after a finite_sheet config, a `2D` in the list would
        warn about every ordinary sheet from then on -- every one of which
        spans its cell by design.

        `None` has to restore the *default*, not merely empty the list: both
        silence `2D`, but only one of them leaves the check working at all.
        So the tip is checked too.
        """
        panel.set_finite_components(['tip', 'flake', '2D'])
        panel.set_finite_components(None)

        panel.show_system(_spanning('2D'))
        assert panel.warning.isHidden(), panel.warning.text()

        panel.show_system(_spanning('tip'))
        assert not panel.warning.isHidden(), \
            'the reset emptied the list instead of restoring the default'
        assert 'tip' in panel.warning.text()


class TestTheSuppressionOnRealGeometry:
    """The two halves of this file, meeting.

    Everything above tests the periodic-image check against four synthetic
    atoms. This feeds it a real assembled heterostructure, in both directions:
    the build as it comes out (nothing to report), and the same real atoms in
    a cell that genuinely does not fit the tip.

    A note on why the second one has to narrow the cell rather than configure
    a small one. `[2D] x = 12` is *not* the built footprint on the hetero AFM
    path: the builder replicates the coincidence cell until it satisfies the
    >=2x tip-diameter rule, so the 12 A config lands as a 50.96 x 44.13 box
    with the 20 A tip sitting well inside it (x 18.7-32.3). The config-level
    precheck in `MainWindow._validate` reads the config and would flag it; the
    viewport reads the built geometry and correctly does not. There is no
    config that produces an oversized tip here, so the failure is constructed
    from real atoms instead.
    """

    def test_a_real_hetero_afm_preview_reports_nothing(self, built, panel):
        """`sub` and `2D` really do reach the walls in this build -- they are
        meant to -- and the tip really does not. Nothing to say."""
        system, _ = built('afm')
        touching = system.touches_boundary()
        assert touching['sub'] and touching['2D'], \
            'this build no longer spans its cell; the test below means nothing'
        assert not touching['tip']

        panel.show_system(system)

        assert panel.warning.isHidden(), panel.warning.text()

    def test_a_real_tip_too_big_for_its_cell_is_reported(self, built, panel):
        """The same real atoms, in a box narrowed to the tip's own span --
        what an under-replicated build would produce. The tip must be named;
        `sub` and `2D`, which overflow that box far more dramatically, must
        not be."""
        system, _ = built('afm')
        tip = next(c for c in system.components if c.name == 'tip')
        block = system.positions[tip.first:tip.last]
        narrowed = AssembledSystem(
            positions=system.positions,
            type_ids=system.type_ids,
            group_ids=system.group_ids,
            cell=(float(block[:, 0].min()) - 0.5, float(block[:, 0].max()) + 0.5,
                  float(block[:, 1].min()) - 0.5, float(block[:, 1].max()) + 0.5,
                  system.cell[4], system.cell[5]),
            components=system.components,
            legend=system.legend,
        )

        panel.show_system(narrowed)
        message = panel.warning.text()

        assert not panel.warning.isHidden(), \
            'a real tip that does not fit its cell must still be reported'
        assert 'periodic boundary' in message
        named = message.split(' reaches')[0]
        assert named == 'tip', \
            f'only the finite component may be named, got {named!r}'
