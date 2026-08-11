"""Previewing a heterostructure.

Two questions, and the second one only exists because of what the first one
answered.

**Does it work at all?** Yes -- and that had to be established by building one,
not by reading the code. Every one of the five modes routes a two-material
config to a hetero builder, and each of those runs a real LAMMPS pass to
assemble the stack. So the first class here builds one for real (no ovito stub,
no fake data file: an ovito stub is exactly what would let a broken preview
pass) and walks it through the same steps ``src.gui.preview_worker`` and
``PreviewController._deliver`` do.

**Is what it then says about the result true?** Not quite, and that is the fix
this file carries. The two families assemble into different *shapes*:

* the AFM family (``afm``, ``indent``, ``pes_tip``) writes an explicit box and
  one ``read_data`` per component -- ``sub``, ``tip``, ``2D``;
* the sheet family (``sheetonsheet``, ``pes_sheet``) assembles the whole stack
  into a single ``hetero.data`` first, so its script reads exactly one file,
  under the LAMMPS group name ``all_atoms``.

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
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('ovito')
pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.core.config_io import write_config  # noqa: E402
from src.core.orchestration import discover_preview_target  # noqa: E402
from src.gui.preview.assemble import (  # noqa: E402
    AssembledSystem, ComponentSlice, assemble,
)
from src.gui.preview.init_script import parse_assembly  # noqa: E402
from src.gui.preview.settings_file import parse_identity  # noqa: E402
from src.gui.preview_worker import run as run_preview_build  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALS = REPO_ROOT / 'examples' / 'materials'
POTENTIALS = REPO_ROOT / 'examples' / 'potentials' / 'sw'

#: `src.builders.hetero._run_lammps` shells out to this by name.
LMP_SERIAL = shutil.which('lmp_serial')


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
        'general': {'temp': 300, 'scan_speed': 1, 'hetero_stacking': 'grouped'},
    }
    if model in ('afm', 'indent', 'pes_tip'):
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
    return config


def _preview(model: str, output_root: Path) -> AssembledSystem:
    """Build and assemble a preview exactly as the GUI does.

    `preview_worker.run` is called rather than re-implemented: it is the
    build half of the real path (`PreviewController.start` spawns it in a
    subprocess), and the three lines after it are `_deliver`'s.
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

    assembly = parse_assembly(Path(result['init_script']),
                              Path(result['simulation_root']))
    identity = (parse_identity(Path(result['settings_file']))
                if result.get('settings_file') else None)
    return assemble(assembly, identity)


@pytest.mark.skipif(LMP_SERIAL is None,
                    reason='hetero assembly shells out to lmp_serial')
class TestAHeterostructurePreviewActuallyBuilds:
    """Build one, for real, and look at what came out.

    Two modes, not five: `indent` and `pes_tip` emit the same assembly script
    as `afm` (the same tip/sheet/substrate stack -- only the *run* phase after
    it differs), and `pes_sheet` the same one as `sheetonsheet`. So these two
    cover both assembly shapes the preview has to read.
    """

    def test_sheet_family_preview_shows_both_materials(self, tmp_path):
        """The whole stack arrives as one pre-assembled `hetero.data`.

        One component, therefore -- the script really does read one file --
        but both materials are in it, which is what makes it a heterostructure
        preview rather than a picture of one sheet. Checked through the
        legend, because that is where the two materials are distinguishable
        once they share a data file.
        """
        system = _preview('sheetonsheet', tmp_path / 'sheet')

        assert system.count > 0, 'the preview assembled no atoms at all'
        assert system.cell is not None, 'no simulation cell to frame the view on'
        assert [c.name for c in system.components] == ['all_atoms']
        assert system.components[0].count == system.count
        elements = set(system.legend.values())
        assert {'Mo', 'W', 'S'} <= elements, (
            f'both materials must be visible in the preview, got {elements}')
        assert not system.warnings, system.warnings

    def test_afm_family_preview_keeps_its_components_apart(self, tmp_path):
        """The AFM family places components separately, so they stay separate:
        the substrate, the tip and the hetero sheet are three named
        components the user can toggle independently."""
        system = _preview('afm', tmp_path / 'afm')

        assert system.count > 0, 'the preview assembled no atoms at all'
        assert system.cell is not None
        assert [c.name for c in system.components] == ['sub', 'tip', '2D']

        # Each component owns its own block of LAMMPS types -- that is what
        # the `offset` on its `read_data` line buys, and what keeps the
        # substrate's silicon from being coloured as the tip's. Overlapping
        # ranges would mean the preview had lost the offsets and was showing
        # every component under the first one's element names.
        blocks = [set(c.type_ids) for c in system.components]
        for index, block in enumerate(blocks):
            for other in blocks[index + 1:]:
                assert not block & other, (
                    f'components share LAMMPS types {sorted(block & other)}; '
                    f'the read_data type offsets were lost')

        elements = set(system.legend.values())
        assert {'Mo', 'W', 'S'} <= elements, (
            f'both materials must be visible in the preview, got {elements}')
        assert not system.warnings, system.warnings


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
    from src.gui.preview import viewport as vp  # noqa: PLC0415

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
    """A preview that succeeded must not be reported as if it had failed."""

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
