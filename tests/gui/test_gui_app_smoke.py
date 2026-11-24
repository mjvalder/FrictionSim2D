"""Driving the real window offscreen.

These are the tests that would have caught a window that cannot open, a form
that cannot build, or a Render button that raises. They construct the actual
MainWindow under ``QT_QPA_PLATFORM=offscreen`` and exercise it.

Rendering itself is not driven here -- it spawns a subprocess and needs an
event loop -- but every step up to and including the request is.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from frictionsim2d.gui.controllers.document import ConfigDocument, ExpansionDirs  # noqa: E402
from frictionsim2d.gui.controllers.preview import cache_key, code_fingerprint  # noqa: E402
from frictionsim2d.gui.preview.assemble import AssembledSystem, ComponentSlice  # noqa: E402
from frictionsim2d.gui.workspace import RecentWorkspaces, Workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'examples'
ALL_MODELS = ['afm', 'indent', 'pes_tip', 'sheetonsheet', 'pes_sheet']


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(name='workspace')
def _workspace(tmp_path):
    return Workspace.open(tmp_path / 'ws')


@pytest.fixture
def two_component_system():
    """A minimal 4-atom assembled system -- enough to make the viewport treat
    a system as "shown" without needing a real render. Mirrors the fixture of
    the same name in tests/gui/test_gui_viewport.py (Task 4); redefined locally
    per the Task 12 brief rather than importing across test modules."""
    return AssembledSystem(
        positions=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]),
        type_ids=np.array([1, 1, 2, 2]),
        group_ids=np.array([0, 0, 1, 1]),
        cell=(0., 10., 0., 10., 0., 10.),
        components=[ComponentSlice('tip', 0, 0, 2, [1]),
                    ComponentSlice('sub', 1, 2, 2, [2])],
        legend={1: 'Si', 2: 'Mo'},
    )


class TestWorkspace:

    def test_opening_creates_the_layout(self, tmp_path):
        workspace = Workspace.open(tmp_path / 'ws')
        assert workspace.root.is_dir()
        assert workspace.configs_dir.is_dir()
        assert workspace.preview_dir.is_dir()

    def test_state_survives_reopening(self, tmp_path):
        workspace = Workspace.open(tmp_path / 'ws')
        workspace.remember('last_model', 'indent')
        assert Workspace.open(tmp_path / 'ws').state['last_model'] == 'indent'

    def test_default_config_path_does_not_collide(self, workspace):
        first = workspace.default_config_path('afm')
        first.write_text('', encoding='utf-8')
        assert workspace.default_config_path('afm') != first

    def test_previews_are_pruned_to_the_most_recent(self, workspace):
        for index in range(12):
            (workspace.preview_dir / f"key{index:02d}").mkdir()
        workspace.prune_previews(keep=3)
        assert len(list(workspace.preview_dir.iterdir())) == 3

    def test_pruning_never_deletes_the_database_cache(self, workspace):
        """`.preview/_db` sits beside the preview builds. Pruning to the
        newest few would otherwise throw the query cache away after eight
        renders, silently."""
        (workspace.preview_dir / '_db').mkdir()
        for index in range(12):
            (workspace.preview_dir / f"key{index:02d}").mkdir()

        workspace.prune_previews(keep=3)

        assert (workspace.preview_dir / '_db').is_dir()
        assert len(list(workspace.preview_dir.iterdir())) == 4

    def test_recent_workspaces_round_trip(self, tmp_path):
        recent = RecentWorkspaces(path=tmp_path / 'recent.json')
        recent.add(tmp_path)
        assert recent.most_recent() == tmp_path.resolve()

    def test_recent_skips_directories_that_vanished(self, tmp_path):
        recent = RecentWorkspaces(path=tmp_path / 'recent.json')
        gone = tmp_path / 'gone'
        gone.mkdir()
        recent.add(gone)
        gone.rmdir()
        assert recent.load() == []


class TestCacheKey:
    """A preview may only be reused when it would be rebuilt identically."""

    def test_same_input_same_key(self):
        config = {'2D': {'mat': 'MoS2'}}
        assert cache_key(config, 'afm', 1) == cache_key(config, 'afm', 1)

    def test_config_change_changes_the_key(self):
        assert cache_key({'2D': {'mat': 'MoS2'}}, 'afm', 1) != \
            cache_key({'2D': {'mat': 'WS2'}}, 'afm', 1)

    def test_model_change_changes_the_key(self):
        config = {'2D': {'mat': 'MoS2'}}
        assert cache_key(config, 'afm', 1) != cache_key(config, 'indent', 1)

    def test_layer_change_changes_the_key(self):
        config = {'2D': {'mat': 'MoS2'}}
        assert cache_key(config, 'afm', 1) != cache_key(config, 'afm', 2)

    def test_referenced_file_change_changes_the_key(self, tmp_path):
        potential = tmp_path / 'p.sw'
        potential.write_text('a', encoding='utf-8')
        config = {'2D': {'mat': 'MoS2'}}
        before = cache_key(config, 'afm', 1, [potential])
        potential.write_text('bb', encoding='utf-8')
        assert cache_key(config, 'afm', 1, [potential]) != before

    def test_the_code_is_part_of_the_key(self):
        """Otherwise editing a template and re-rendering would show a system
        built by the previous code."""
        config = {'2D': {'mat': 'MoS2'}}
        assert code_fingerprint() in ''.join([
            code_fingerprint(),
        ])
        assert cache_key(config, 'afm', 1)  # smoke: fingerprint is included


class TestRawIniTruthiness:
    """`bool('false')` is True, and a config read from a file holds exactly
    that until the pydantic model coerces it -- so a flag read straight off
    the document has to be interpreted, not just tested for truth."""

    @pytest.mark.parametrize('value', ['false', 'False', 'FALSE', 'no', '0',
                                       'off', '', '  false  ', None, False, 0])
    def test_falsey(self, value):
        from frictionsim2d.gui.app import _is_true
        assert _is_true(value) is False

    @pytest.mark.parametrize('value', ['true', 'True', 'yes', '1', 'on',
                                       ' True ', True, 1])
    def test_truthy(self, value):
        from frictionsim2d.gui.app import _is_true
        assert _is_true(value) is True


class TestMainWindow:

    @pytest.fixture(name='window')
    def _window(self, qt_app, workspace, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from frictionsim2d.gui.app import MainWindow
        window = MainWindow(workspace)
        yield window
        # Tests here exercise edits that mark the document/settings dirty;
        # closeEvent now prompts to save on a dirty close, which would block
        # forever on the real QMessageBox with no one to click it. Cleanup
        # is not the thing under test, so close it quietly.
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()

    def test_window_opens(self, window):
        assert window.windowTitle().startswith('FrictionSim2D')

    def test_it_starts_on_the_afm_family(self, window):
        assert window.document.model == 'afm'
        assert window.family_box.currentData() == 'afm'

    def test_mode_selector_lists_the_family_modes(self, window):
        modes = [window.mode_box.itemData(i) for i in range(window.mode_box.count())]
        assert modes == ['afm', 'indent', 'pes_tip']

    def test_switching_family_repopulates_the_modes(self, window):
        window.family_box.setCurrentIndex(1)
        modes = [window.mode_box.itemData(i) for i in range(window.mode_box.count())]
        assert modes == ['sheetonsheet', 'pes_sheet']

    def test_switching_family_switches_the_document(self, window):
        window.family_box.setCurrentIndex(1)
        assert window.document.model == 'sheetonsheet'
        assert 'tip' not in window.form.sections

    @pytest.mark.parametrize('index,expected', [(0, 'afm'), (1, 'indent'), (2, 'pes_tip')])
    def test_each_afm_mode_builds_a_form(self, window, index, expected):
        window.mode_box.setCurrentIndex(index)
        assert window.document.model == expected
        assert window.form.sections

    def test_the_form_has_a_section_per_config_section(self, window):
        assert '2D' in window.form.sections
        assert 'tip' in window.form.sections
        assert 'general' in window.form.sections

    def test_substrate_cannot_be_switched_off(self, window):
        """Every model but the PES tip scan aborts without one."""
        assert not window.form.sections['sub'].isCheckable()

    def test_pes_tip_may_be_freestanding(self, window):
        window.mode_box.setCurrentIndex(2)
        assert window.form.sections['sub'].isCheckable()

    def test_counts_are_shown(self, window):
        assert '→' in window.counts.text()

    def test_opening_an_example_populates_the_form(self, window):
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

        editor = window.form.sections['2D'].editors['mat']
        assert editor.value() == 'h-MoS2'

    def test_validation_reports_a_valid_config(self, window):
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()
        assert window._validate()
        # indent_config.ini's 60x60 sheet against a r=25 tip is under the
        # 2x-diameter sizing threshold (D12), so it now also reports the
        # non-blocking tip-sizing warning -- see
        # test_tip_sizing_warning_is_non_blocking, which pins that behaviour
        # down explicitly. It stays non-blocking, so no error lines appear.
        assert 'required' not in window.issues.toPlainText()
        assert 'Input should be' not in window.issues.toPlainText()

    def test_validation_reports_and_decorates_a_bad_field(self, window):
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()
        window.form.sections['tip'].editors['pot_path'].set_value('nope.sw')
        window.form.commit()

        assert not window._validate()
        assert 'pot_path' in window.issues.toPlainText()

    def test_counts_update_from_the_loaded_config(self, window):
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window._refresh_counts()
        assert '1 config → 2 builds → 10 jobs' in window.counts.text()

    def test_an_unsupported_list_is_explained(self, window):
        """Tip radius is not part of the output directory name, so sweeping it
        would make every run overwrite the last. The editor keeps the list so
        validation can say that, rather than reporting 'not a number'."""
        window.form.sections['tip'].editors['r'].set_value([10, 20])
        window._validate()
        assert 'overwrite each other' in window.issues.toPlainText()

    def test_angle_split_with_a_force_gate_is_reported(self, window):
        """The angle split cannot honour scan_angle_force, and the builders
        only say so from write_inputs -- after the geometry has been built,
        leaving a half-written tree behind. The panel has to show it here,
        before Generate is ever pressed. The mirror case (a model whose
        builder never splits, where the pair is inert and must NOT be
        reported) is pinned in tests/gui/test_gui_document.py."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'afm_flake_config.ini', 'afm')
        window.form.document = window.document
        window.form.rebuild()
        window.form.sections['general'].editors['scan_angle'].set_value([0.0, 30.0])
        window.form.sections['general'].editors['scan_angle_force'].set_value(30.0)
        window.form.sections['general'].editors['outer_loop'].set_value('scan_angle')

        assert not window._validate()
        text = window.issues.toPlainText()
        assert 'scan_angle_force' in text
        # Reported because *this* builder splits, and it names which one.
        assert 'AFMSimulation' in text

    def test_tip_sizing_warning(self, window):
        """A footprint under 2x the tip diameter lets the tip interact with
        its own periodic image (D12) -- non-blocking, but worth flagging.
        Set through the editors (as a user would), not the document directly:
        `_validate` calls `form.commit()` first, which would otherwise write
        the still-blank editor values straight back over a direct set()."""
        window.form.sections['2D'].editors['x'].set_value(40.0)
        window.form.sections['2D'].editors['y'].set_value(40.0)
        window.form.sections['tip'].editors['r'].set_value(25.0)
        window._validate()
        assert '2× tip diameter' in window.issues.toPlainText()

    def test_tip_sizing_warning_is_not_raised_for_a_heterostructure(self, window):
        """`[2D] x`/`y` are not the built footprint on the hetero path: the
        builder replicates the coincidence cell until it is >= 2x the tip
        diameter (`hetero_afm.compute_replication_factors`), and neither
        `hetero_afm.py` nor `hetero.py` reads those two fields at all. The
        same false alarm was removed from the 3D viewport; this is the other
        place that made the claim."""
        window.form.sections['2D'].editors['x'].set_value(40.0)
        window.form.sections['2D'].editors['y'].set_value(40.0)
        window.form.sections['tip'].editors['r'].set_value(25.0)
        window._validate()
        # The single-material config it starts from does warn -- without this
        # the assertion below would pass on a config that never qualified.
        assert '2× tip diameter' in window.issues.toPlainText()

        window.document.add_second_material()
        window.form.rebuild()
        window._validate()

        assert '2× tip diameter' not in window.issues.toPlainText()

    def test_tip_sizing_warning_is_non_blocking(self, window):
        """The precheck must not make an otherwise-valid config invalid."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()
        # The radius is set HERE rather than relied on from the example, and
        # it has to sit in the band BETWEEN the two thresholds:
        #
        #   blocking (model validator): footprint < 1.5 x tip diameter -> 3r
        #   warning  (this precheck):   footprint < 2   x tip diameter -> 4r
        #
        # On a 60x60 sheet that is 20 < r <= 15 inverted -- i.e. r in (15, 20].
        # r=18 gives 60 >= 3*18 (valid) and 60 < 4*18 (warned), which is the
        # only combination that can show a warning is NON-blocking.
        #
        # indent_config.ini used to ship r=25, which trips BOTH and so cannot
        # demonstrate it; d386926 shrank every example radius to 15, which
        # trips neither. The test has had no usable fixture since.
        window.form.sections['tip'].editors['r'].set_value(18.0)

        assert window._validate()
        assert '2× tip diameter' in window.issues.toPlainText()

    def test_a_setting_that_does_nothing_is_reported_not_blocked(self, window):
        """`hetero_stacking` orders the layers of a two-material stack. On a
        single-material config it builds and runs, it just does not do what it
        looks like it does -- so it is said, not refused."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()
        window.document.set('general', 'hetero_stacking', 'alternating')

        ok = window._validate()

        assert 'hetero_stacking' in window.issues.toPlainText()
        assert ok

    def test_a_refused_commit_reaches_the_issues_panel(self, window, monkeypatch):
        """REVIEW (task 3): a section the form could not commit used to be a
        modal, one per failing section, raised from the path Render, Generate
        and Save all take. Offscreen there is nobody to dismiss it, so a
        regression on that path hung instead of failing.

        It belongs in the panel that already holds every other reason the
        config will not build -- and it has to survive `_validate`, which
        rewrites that panel from scratch straight after the commit that
        produced the refusal.
        """
        from frictionsim2d.gui.controllers.document import DocumentError

        dialogs = []
        for name in ('warning', 'critical', 'information'):
            monkeypatch.setattr(QMessageBox, name,
                                lambda *a, _n=name, **k: (dialogs.append(_n),
                                                          QMessageBox.Ok)[1])

        def refuse():
            raise DocumentError('pair_mode says this config is not a pair')

        monkeypatch.setattr(window.form.sections['2D'], 'commit', refuse)

        window._validate()

        assert 'pair_mode' in window.issues.toPlainText()
        assert dialogs == [], f'a refused commit opened a dialog: {dialogs}'

    def test_a_refusal_stops_being_reported_once_it_is_fixed(self, window,
                                                             monkeypatch):
        """Refusals are per-commit, not cumulative. A list that is only ever
        appended to would leave the panel accusing a config of a problem the
        user had already fixed -- and there would be no way to clear it."""
        from frictionsim2d.gui.controllers.document import DocumentError

        section = window.form.sections['2D']
        original = section.commit

        def refuse():
            raise DocumentError('pair_mode says this config is not a pair')

        monkeypatch.setattr(section, 'commit', refuse)
        window._validate()
        assert 'pair_mode' in window.issues.toPlainText()

        monkeypatch.setattr(section, 'commit', original)      # the fixing edit
        window._validate()

        assert 'pair_mode' not in window.issues.toPlainText()

    @staticmethod
    def _refuse_a_section(window, monkeypatch, form=None, key='2D'):
        """Make one section of `form` refuse its commit, as the document does
        when the form and the document have diverged."""
        from frictionsim2d.gui.controllers.document import DocumentError

        def refuse():
            raise DocumentError('pair_mode says this config is not a pair')

        target = (form or window.form).sections[key]
        monkeypatch.setattr(target, 'commit', refuse)

    def test_a_refused_commit_blocks_save_and_leaves_a_message(
            self, window, monkeypatch):
        """REVIEW (round 1): routing the refusal to the status bar made it
        *invisible* on this path. `_save_config` does not call `_validate`, so
        the only report was a status-bar line -- overwritten by this method's
        own `Saved …` a few lines later, behind an intervening file dialog.
        The `.ini` on disk silently disagreed with the form. Trading a modal
        for an invisible message is worse than the modal.

        So: it blocks (the file is the artefact; writing a config the user
        never approved is a wrong-results path), and it is said in the issues
        panel, which survives, as well as in a dialog the user cannot miss.
        """
        dialogs = []
        monkeypatch.setattr(QMessageBox, 'warning',
                            lambda *a, **k: (dialogs.append(a[1:3]),
                                             QMessageBox.Ok)[1])
        monkeypatch.setattr(
            QFileDialog, 'getSaveFileName',
            lambda *a, **k: pytest.fail('a refused config reached the file dialog'))
        self._refuse_a_section(window, monkeypatch)

        window._save_config()

        assert 'pair_mode' in window.issues.toPlainText()
        assert dialogs, 'the refusal was not reported at all'
        assert 'pair_mode' in dialogs[0][1]

    @staticmethod
    def _valid_config(window):
        """A config that validates, so a later block can only be the refusal."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

    def test_a_refused_commit_blocks_generate(self, window, monkeypatch):
        """Generate launches work from `self.document`. After a refusal that
        document is not what is on screen, so building from it is a
        wrong-results path -- it must not even reach the confirmation.

        The same config reaches the confirmation without the refusal, so what
        stops it here is the refusal and not some unrelated invalidity."""
        self._valid_config(window)
        asked = []
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: (asked.append(1), QMessageBox.No)[1])

        window._generate()
        assert asked, 'this config does not reach the confirmation even when valid'

        self._refuse_a_section(window, monkeypatch)
        asked.clear()
        window._generate()

        assert not asked, 'a refused config reached the Generate confirmation'
        assert 'pair_mode' in window.issues.toPlainText()

    def test_a_refused_commit_does_not_block_render(self, window, monkeypatch):
        """Render persists nothing and launches nothing, and looking at the
        geometry is often how the disagreement gets diagnosed -- so there the
        refusal is reported and not gated on."""
        self._valid_config(window)
        started = []
        monkeypatch.setattr(window.preview, 'start', started.append)
        self._refuse_a_section(window, monkeypatch)

        window._render()

        assert started, 'a refusal must not stop a preview'
        assert 'pair_mode' in window.issues.toPlainText()

    def test_a_refused_settings_commit_blocks_the_build(self, window, monkeypatch):
        """The Settings tab has the same overwrite pattern: `_apply_settings`
        would have returned None and let the build run on settings that do not
        match the tab."""
        self._refuse_a_section(window, monkeypatch,
                               form=window.settings_form,
                               key=next(iter(window.settings_form.sections)))

        error = window._apply_settings()

        assert error and 'pair_mode' in error
        assert 'pair_mode' in window.issues.toPlainText()

    def test_a_refused_settings_commit_blocks_saving_settings(
            self, window, monkeypatch):
        """`_save_settings` has the same overwrite pattern as `_save_config`:
        its `Saved …` would have been the only thing left on screen."""
        dialogs = []
        monkeypatch.setattr(QMessageBox, 'warning',
                            lambda *a, **k: (dialogs.append(a[1:3]),
                                             QMessageBox.Ok)[1])
        monkeypatch.setattr(
            window.settings_document, 'save',
            lambda *a, **k: pytest.fail('refused settings were written anyway'))
        self._refuse_a_section(window, monkeypatch,
                               form=window.settings_form,
                               key=next(iter(window.settings_form.sections)))

        window._save_settings()

        assert 'pair_mode' in window.issues.toPlainText()
        assert dialogs and 'pair_mode' in dialogs[0][1]

    def test_a_finite_sheet_makes_the_sheet_worth_warning_about(
            self, window, monkeypatch):
        """`[general] finite_sheet` sizes `2D` as an island inside the
        substrate box, so it is genuinely finite -- and nothing in the
        assembled system records that, so the window has to say it."""
        declared = []
        monkeypatch.setattr(window.viewport, 'set_finite_components',
                            declared.append)
        monkeypatch.setattr(window.viewport, 'show_system', lambda _s: None)
        system = SimpleNamespace(count=0, warnings=[])

        window.document.set('general', 'finite_sheet', 'false')
        window._on_preview_ready(system)
        assert '2D' not in declared[-1]

        window.document.set('general', 'finite_sheet', 'true')
        window._on_preview_ready(system)
        assert '2D' in declared[-1]

    def test_a_refused_edit_reaches_the_status_bar(self, window):
        """The per-keystroke path (`SectionForm._on_field_changed`) has no
        validation pass behind it to write the panel, so it says so in the
        status bar -- and, again, without waiting for anyone to click."""
        window.form.refused.emit('[2D] pair_mode disagrees with the document')

        assert 'pair_mode' in window.statusBar().currentMessage()

    def test_a_refused_mode_switch_puts_the_selector_back(self, window, monkeypatch):
        """A refusal that leaves the combo showing a mode the document is not
        in is worse than the refusal: `_on_mode_changed` returns early when the
        combo already reads the document's model, so selecting that same mode
        again would then silently do nothing at all."""
        from frictionsim2d.gui.controllers.document import DocumentError

        monkeypatch.setattr(QMessageBox, 'warning',
                            lambda *args, **kwargs: QMessageBox.Ok)

        def refuse(_model, drop_second=False):
            raise DocumentError('no heterostructure build for that mode')

        monkeypatch.setattr(window.document, 'set_model', refuse)

        window.mode_box.setCurrentIndex(1)          # 'indent'

        assert window.document.model == 'afm'
        assert window.mode_box.currentData() == 'afm'

    def test_generate_confirmation_lists_the_output_dir(self, window, monkeypatch):
        """The Generate confirmation must preview the directory(ies) it is
        about to create -- and, with nothing colliding, no warning."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

        captured = {}

        def fake_question(*args, **_kwargs):
            captured['message'] = args[2]
            return QMessageBox.No  # abort before generator.start actually runs

        monkeypatch.setattr(QMessageBox, 'question', fake_question)
        window._generate()

        message = captured['message']
        assert 'indent/h-MoS2/60x_60y/sub_amorph-Si_tip_Si_r15/K300/pot_sw_MoS2_wen' in message
        assert 'WARNING' not in message

    def test_generate_reports_why_a_pair_config_cannot_expand(self, window, monkeypatch):
        """A pair config whose [2D-2] is templated on the wrong marker cannot
        expand at all. Both the counts and the directory preview come from that
        same expansion, so before this the click raised out of the slot and the
        user got nothing. It must come back as the expansion's own message, and
        must not reach the confirmation -- there is nothing to confirm."""
        monkeypatch.chdir(REPO_ROOT)
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'hetero_pair_sweep.ini', 'sheetonsheet')
        window.document.set('2D-2', 'cif_path', 'examples/materials/{mat1}.cif')
        window.form.document = window.document
        window.form.rebuild()

        warned = {}
        monkeypatch.setattr(QMessageBox, 'warning',
                            lambda *args, **_kw: warned.setdefault('text', args[2]))
        monkeypatch.setattr(
            QMessageBox, 'question',
            lambda *args, **_kw: pytest.fail('confirmation shown for a config '
                                             'that cannot expand'))

        window._generate()

        assert '[2D-2]' in warned['text']
        assert '{mat2}' in warned['text']
        assert '[2D-2]' in window.issues.toPlainText()

    def test_generate_confirmation_flags_duplicate_output_dirs(self, window, monkeypatch):
        """When two members of a sweep would build into the same directory --
        the exact hazard the loopability rule exists to prevent, e.g. a
        temperature sweep whose members both truncate to the same K{int(temp)}
        -- the confirmation must call it out. Exercised directly against
        `expansion_output_dirs` (already covered producing a real collision
        for a truncating temperature sweep in test_gui_document.py) so this
        test is about `_generate`'s handling of that list, not about
        reproducing the collision through form/pydantic validation, which a
        temp sweep does not pass (`temp` is a scalar field on the model)."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

        dup = Path('indent/h-MoS2/60x_60y/sub_amorph-Si_tip_Si_r15/K300/pot_sw_MoS2_wen')
        monkeypatch.setattr(window.document, 'expansion_output_dirs',
                            lambda: ExpansionDirs([dup, dup]))

        captured = {}

        def fake_question(*args, **_kwargs):
            captured['message'] = args[2]
            return QMessageBox.No

        monkeypatch.setattr(QMessageBox, 'question', fake_question)
        window._generate()

        message = captured['message']
        assert message.count(str(dup)) >= 2
        assert 'WARNING' in message
        assert str(dup) in message.split('WARNING', 1)[1]

    def test_saving_produces_a_config_the_cli_can_read(self, window, tmp_path):
        from frictionsim2d.core.config import parse_config

        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        out = window.document.save(tmp_path / 'saved.ini')
        assert parse_config(out) == parse_config(EXAMPLES / 'indent_config.ini')

    def test_viewport_starts_empty(self, window):
        assert window.viewport.system is None

    def test_settings_tab_names_its_source(self, window):
        assert 'Settings from:' in window.settings_origin.text()

    def test_settings_are_editable(self, window):
        assert window.settings_form.sections
        assert 'thermostat' in window.settings_form.sections

    def test_settings_survive_a_form_round_trip(self, window):
        """Committing the settings form unchanged must not invalidate them --
        empty strings and nested sub-models are both easy to destroy."""
        window.settings_form.commit()
        result = window.settings_document.validate()
        assert result.ok, result.render()

    def test_saving_settings_writes_a_complete_file(self, window):
        """load_settings replaces rather than merges, so a partial file would
        silently reset everything it omitted."""
        import yaml

        window.settings_form.commit()
        window._save_settings()
        path = window.workspace.settings_path
        assert path.exists()
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        assert set(data) == set(window.settings_document.values)

    def test_a_saved_settings_file_is_passed_to_builds(self, window):
        """Nothing loads settings.yaml from a directory on its own; the GUI has
        to pass it explicitly."""
        assert window.settings_file is None
        window.settings_form.commit()
        window._save_settings()
        assert window.settings_file == window.workspace.settings_path

    def test_render_and_generate_shortcuts(self, window):
        """Render and Generate buttons have keyboard shortcuts."""
        assert window.render_button.shortcut().toString() == 'Ctrl+R'
        assert window.generate_button.shortcut().toString() == 'Ctrl+G'

    def test_scroll_position_retained_when_mode_changes(self, window, qt_app):
        """Switching simulation mode preserves the form's scroll position.

        A real end-state assertion (scrollbar.value() == saved_value after the
        switch) was tried and rejected: verified directly against this exact
        window, offscreen Qt does not settle the rebuilt scrollbar's range
        synchronously, so `_on_mode_changed`'s restore call clamps to 0 against
        a transiently empty range and never recovers -- even though the code
        does the restore in the documented, correct order (after rebuild()).
        Perversely, moving the restore to *before* rebuild() (the exact bug
        this test exists to catch) reads back as 50 in that same offscreen
        run, because the old, still-large range is still in effect at the
        moment of the call. A bare "does the final value match" assertion is
        therefore backwards here, not just unstable -- so this pins down the
        actual causal claim instead: the restore call happens, and it happens
        strictly after rebuild() has run.
        """
        from unittest.mock import patch

        # Show the window so the scrollbar gets a real range
        window.show()
        qt_app.processEvents()

        scrollbar = window.config_scroll.verticalScrollBar()
        # Scroll to a meaningful position
        scrollbar.setValue(50)
        saved_value = scrollbar.value()

        order = []
        real_rebuild = window.form.rebuild
        real_set_value = scrollbar.setValue

        def rebuild_spy():
            order.append('rebuild')
            return real_rebuild()

        def set_value_spy(value):
            if value == saved_value:
                order.append('restore')
            return real_set_value(value)

        with patch.object(window.form, 'rebuild', side_effect=rebuild_spy), \
             patch.object(scrollbar, 'setValue', side_effect=set_value_spy):
            # Change mode: AFM (with tip section) → Sheet (no tip section) = height change
            # Switch to sheetonsheet family first for greater height difference
            window.family_box.setCurrentIndex(1)  # Switch family to sheet-on-sheet
            qt_app.processEvents()
            window.mode_box.setCurrentIndex(0)  # Switch to sheetonsheet mode

        assert 'restore' in order, (
            f"scrollbar.setValue({saved_value}) was never called during mode "
            f"switch. Calls: {order}")
        assert 'rebuild' in order
        assert order.index('restore') > order.index('rebuild'), (
            f"the restore ran before rebuild(), which then wipes it: {order}")


class TestSettingsTakeEffect:
    """The tab edits a document but a build reads a file, so an edit that is
    only on screen used to be silently ignored."""

    @pytest.fixture(name='window')
    def _window(self, qt_app, workspace, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from frictionsim2d.gui.app import MainWindow
        window = MainWindow(workspace)
        yield window
        # See TestMainWindow's fixture: avoid a real, blocking save prompt
        # from closeEvent during teardown of a dirty settings document.
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()

    def test_hpc_scripts_are_blocked_until_modules_are_set(self, window):
        from frictionsim2d.run.hpc_dispatch import hpc_generation_blocker

        assert hpc_generation_blocker(window.settings) is not None

    def test_typing_modules_applies_without_a_separate_save(self, window):
        from frictionsim2d.run.hpc_dispatch import hpc_generation_blocker

        window.settings_form.sections['hpc'].editors['modules'].set_value(
            ['lammps', 'openmpi'])
        assert window._apply_settings() is None
        assert hpc_generation_blocker(window.settings) is None

    def test_applying_writes_the_workspace_file_builds_are_given(self, window):
        assert window.settings_file is None
        window.settings_form.sections['hpc'].editors['modules'].set_value(['lammps'])
        window._apply_settings()
        assert window.settings_file == window.workspace.settings_path

    def test_invalid_settings_stop_the_build_and_are_not_saved(self, window):
        window.settings_form.sections['hpc'].editors['num_cpus'].set_value('lots')
        error = window._apply_settings()
        assert error is not None
        assert 'num_cpus' in error
        assert not window.workspace.settings_path.exists()

    def test_syncing_with_no_edits_is_a_no_op(self, window):
        assert window._apply_settings() is None
        assert not window.workspace.settings_path.exists()

    def test_the_hpc_group_explains_the_modules_requirement(self, window):
        section = window.settings_form.sections['hpc']
        assert section.title() == 'HPC'
        bubble = section._bubbles.get('modules')
        assert bubble is not None
        assert 'no job scripts' in bubble.toolTip()

    def test_applying_settings_moves_the_star_on_the_onscreen_config_form(
            self, window):
        """_apply_settings must fan the new settings out to window.form (the
        actual on-screen config form), not only to window.settings -- driving
        the check through SectionForm.set_settings directly (as
        TestDependentFields and friends do) cannot catch the app wiring that
        calls it being dropped."""
        window.family_box.setCurrentIndex(1)   # sheet-on-sheet: has driving_spring
        window.mode_box.setCurrentIndex(0)

        label, _cell = window.form.sections['general']._rows['driving_spring']
        assert '*' in label.text()   # sanity: virtual_atom is the default drive method

        window.settings_form.sections['simulation'].editors['drive_method'].set_value('smd')
        assert window._apply_settings() is None

        label, _cell = window.form.sections['general']._rows['driving_spring']
        assert '*' not in label.text()

    def test_saving_settings_moves_the_star_on_the_onscreen_config_form(
            self, window):
        """Same claim as above, through _save_settings (the Settings tab's
        'Save to workspace' button) instead of _apply_settings."""
        window.family_box.setCurrentIndex(1)   # sheet-on-sheet: has driving_spring
        window.mode_box.setCurrentIndex(0)

        label, _cell = window.form.sections['general']._rows['driving_spring']
        assert '*' in label.text()

        window.settings_form.sections['simulation'].editors['drive_method'].set_value('smd')
        window._save_settings()

        label, _cell = window.form.sections['general']._rows['driving_spring']
        assert '*' not in label.text()

    def test_reverting_settings_refreshes_self_settings_and_the_form(self, window):
        """_apply_settings and _save_settings both refresh self.settings (and
        fan it out to the config form) after changing the settings document;
        _revert_settings must too, or a star -- and _validate, which hands
        self.settings straight to document.validate -- keeps reading the
        pre-revert object even though the Settings tab itself already shows
        defaults. A fresh SettingsDocument.defaults() is never dirty, so
        _apply_settings's own early return can never repair this later."""
        window.settings_form.sections['simulation'].editors['drive_method'].set_value('smd')
        window.settings_form.commit()
        assert window._apply_settings() is None
        assert window.settings.simulation.drive_method == 'smd'
        assert window.form.settings.simulation.drive_method == 'smd'

        window._revert_settings()

        assert window.settings.simulation.drive_method == 'virtual_atom'
        assert window.form.settings.simulation.drive_method == 'virtual_atom'


class TestRenderIgnoresNonBlockingRequired:
    """`force`/`pressure` are starred and `required_blocks` for Generate (an
    absent load breaks the LAMMPS script), but they do not affect the
    rendered geometry -- a user clearing a load to try another value must
    still be able to preview."""

    @pytest.fixture(name='window')
    def _window(self, qt_app, workspace, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from frictionsim2d.gui.app import MainWindow
        window = MainWindow(workspace)
        yield window
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()

    def test_render_proceeds_but_generate_still_refuses(self, window, monkeypatch):
        # indent_config.ini validates cleanly out of the box (unlike
        # afm_config.ini, whose materials_list path is relative to a `run/`
        # directory this checkout does not carry) so the only thing standing
        # between _render and the preview request is the field cleared below.
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

        window.form.sections['general'].editors['force'].set_value('')
        window.form.commit()
        missing = {(sec, key): blocks for sec, key, _label, blocks
                  in window.form.missing_required()}
        assert missing.get(('general', 'force')) is True

        render_calls = []
        monkeypatch.setattr(window.preview, 'start',
                            lambda request: render_calls.append(request))
        window._render()
        assert render_calls, 'a blocked-but-cosmetic field must not stop Render'

        generate_calls = []
        monkeypatch.setattr(window.generator, 'start',
                            lambda request: generate_calls.append(request))
        window._generate()
        assert not generate_calls, 'Generate must still refuse the missing load'


class TestValidateEnforcesBlockingRequired:
    """`missing_required()` only reports which fields are empty and whether
    they are supposed to block; nothing before this pinned down that
    `_validate()` actually acts on that report -- `blocking` could be thrown
    away entirely and every other assertion in this file would still pass."""

    @pytest.fixture(name='window')
    def _window(self, qt_app, workspace, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from frictionsim2d.gui.app import MainWindow
        window = MainWindow(workspace)
        yield window
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()

    def test_an_empty_blocking_field_fails_validate(self, window):
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

        window.form.sections['general'].editors['force'].set_value('')
        window.form.commit()

        assert not window._validate()

    def test_enforce_required_false_lets_the_same_config_through(self, window):
        """_render's own use of enforce_required=False (see
        TestRenderIgnoresNonBlockingRequired) already covers this end to end;
        pinned down here directly against _validate() so the two do not only
        agree by accident."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

        window.form.sections['general'].editors['force'].set_value('')
        window.form.commit()

        assert window._validate(enforce_required=False)

    def test_the_issues_panel_names_the_empty_required_field(self, window):
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.form.document = window.document
        window.form.rebuild()

        window.form.sections['general'].editors['force'].set_value('')
        window.form.commit()

        window._validate()
        issues_text = window.issues.toPlainText()
        assert '[general] Normal load' in issues_text
        assert 'required, currently empty' in issues_text


class TestSubstrateSection:
    """Switching to a mode that requires a substrate must restore it -- and the
    form must not silently discard what is typed into a section that is absent."""

    def test_switching_from_pes_tip_restores_the_substrate(self):
        """Only the PES tip scan may be freestanding, so switching away from it
        has to bring the substrate back -- otherwise the build aborts with
        'substrate is not optional' however carefully the form was filled in."""
        document = ConfigDocument.blank('pes_tip')
        document.enable_section('sub', False)
        document.set_model('afm')
        assert document.has_section('sub')

    def test_a_filled_substrate_validates_after_the_switch(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.blank('pes_tip')
        document.enable_section('sub', False)
        document.set_model('afm')

        for section, values in (
            ('2D', {'mat': 'h-MoS2', 'pot_type': 'sw',
                    'pot_path': 'examples/potentials/sw/MoS2_wen.sw',
                    'cif_path': 'examples/materials/h-MoS2.cif',
                    # >= 2x the tip diameter (4 * r), or the footprint rule rejects it
                    'x': 100, 'y': 100, 'layers': [1]}),
            ('tip', {'mat': 'Si', 'pot_type': 'sw',
                     'pot_path': 'examples/potentials/sw/Si.sw',
                     'cif_path': 'examples/materials/Si.cif', 'r': 25}),
            ('sub', {'mat': 'Si', 'pot_type': 'sw',
                     'pot_path': 'examples/potentials/sw/Si.sw',
                     'cif_path': 'examples/materials/Si.cif',
                     'thickness': 12, 'amorph': 'a'}),
        ):
            for key, value in values.items():
                document.set(section, key, value)

        result = document.validate()
        assert result.ok, result.render()

    def test_a_required_section_accepts_edits_even_if_it_was_absent(self, qt_app,
                                                                   monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.blank('afm')
        del document.values['sub']          # as a mode switch used to leave it

        form = ConfigForm(document)
        form.sections['sub'].editors['mat'].set_value('Si')
        form.commit()

        assert document.has_section('sub')
        assert document.get('sub', 'mat') == 'Si'


class TestFamilyFieldFiltering:
    """A field the family's templates never read should not be offered."""

    @pytest.fixture(name='general_fields')
    def _general_fields(self, qt_app, monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)

        def fields_for(model):
            return set(ConfigForm(ConfigDocument.blank(model)).sections['general'].editors)
        return fields_for

    def test_afm_offers_load_but_not_pressure(self, general_fields):
        fields = general_fields('afm')
        assert 'force' in fields
        assert 'pressure' not in fields

    def test_afm_hides_the_sheet_only_springs(self, general_fields):
        fields = general_fields('afm')
        assert 'bond_spring' not in fields
        assert 'driving_spring' not in fields

    def test_sheet_on_sheet_offers_pressure_but_not_load(self, general_fields):
        fields = general_fields('sheetonsheet')
        assert 'pressure' in fields
        assert 'force' not in fields

    def test_sheet_on_sheet_keeps_the_springs(self, general_fields):
        assert 'bond_spring' in general_fields('sheetonsheet')

    @pytest.mark.parametrize('model,expected', [
        ('afm', {'force', 'scan_angle'}),
        ('indent', {'force'}),
        # Both PES scans override write_inputs entirely and never look at
        # outer_loop -- offering a split here would silently do nothing, and
        # the job count would predict a split the build never produces. See
        # test_expansion_count.py for the build-side pin. With nothing to
        # offer, the row is not rendered at all (see
        # TestEmptySplitControlIsHidden in test_gui_form.py).
        ('pes_tip', set()),
        ('sheetonsheet', {'pressure', 'scan_speed', 'scan_angle'}),
        ('pes_sheet', set()),
    ])
    def test_outer_loop_offers_only_what_the_builder_acts_on(self, qt_app, model,
                                                             expected, monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.blank(model))
        editors = form.sections['general'].editors
        if not expected:
            assert 'outer_loop' not in editors
            return
        box = editors['outer_loop'].box
        choices = {box.itemData(i) for i in range(box.count())} - {None}
        assert choices == expected


class TestDependentFields:
    """A greyed control says 'this does nothing yet'; a live one that is ignored
    says nothing at all."""

    @pytest.fixture(name='form')
    def _form(self, qt_app, monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        return ConfigForm(ConfigDocument.blank('afm'))

    def test_match_tolerance_is_disabled_until_lattice_match_is_ticked(self, form):
        section = form.sections['sub']
        assert not section.editors['match_strain_tol'].isEnabled()

        section.editors['match'].set_value(True)
        section._apply_dependencies()
        assert section.editors['match_strain_tol'].isEnabled()

    def test_twist_tolerances_are_disabled_without_a_twist_angle(self, form):
        section = form.sections['general']
        for key in ('twist_angle_tol', 'twist_strain_tol', 'twist_max_supercell'):
            assert not section.editors[key].isEnabled(), key

        section.editors['twist_angle'].set_value(5.0)
        section._apply_dependencies()
        assert section.editors['twist_angle_tol'].isEnabled()

    def test_scan_angle_selector_is_disabled_without_scan_angles(self, form):
        section = form.sections['general']
        section.editors['scan_angle'].set_value(None)
        section._apply_dependencies()
        assert not section.editors['scan_angle_force'].isEnabled()


class TestRequiredMarking:

    def test_required_fields_are_marked_and_optional_ones_are_not(self, qt_app,
                                                                 monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.blank('afm'))
        section = form.sections['2D']

        required_label, _ = section._rows['mat']
        optional_label, _ = section._rows['lat_c']
        assert '*' in required_label.text()
        assert '*' not in optional_label.text()


class TestCollapsibleSections:
    """Both tabs run to nine sections; folding is what makes them navigable."""

    @pytest.fixture(name='form')
    def _form(self, qt_app, monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.blank('afm'))
        yield form
        form.deleteLater()

    def test_required_sections_start_open(self, form):
        for key in ('2D', 'tip', 'sub', 'general'):
            assert not form.sections[key].is_collapsed(), key

    def test_switched_off_optional_sections_start_folded(self, form):
        """They have nothing to show, and they are most of the clutter."""
        for key in ('flake', 'lattice_match'):
            assert form.sections[key].is_collapsed(), key

    def test_toggling_folds_and_unfolds(self, form):
        section = form.sections['tip']
        section.toggle_collapsed()
        assert section.is_collapsed()
        section.toggle_collapsed()
        assert not section.is_collapsed()

    def test_collapse_all_and_expand_all(self, form):
        form.set_all_collapsed(True)
        assert all(f.is_collapsed() for f in form.sections.values())
        form.set_all_collapsed(False)
        assert not any(f.is_collapsed() for f in form.sections.values())

    def test_fold_state_survives_a_rebuild(self, form):
        """Switching mode should not reopen everything just tidied away."""
        form.sections['general'].set_collapsed(True)
        form.rebuild()
        assert form.sections['general'].is_collapsed()

    def test_switching_a_section_on_opens_it(self, form):
        """Ticking it is a request to work on it."""
        section = form.sections['flake']
        assert section.is_collapsed()
        section.setChecked(True)
        assert not section.is_collapsed()

    def test_state_is_tracked_not_read_from_visibility(self, form):
        """isVisible() is False for everything until the window is shown, so
        deriving fold state from it reports every section as folded."""
        assert not form.sections['tip'].is_collapsed()
        assert not form.sections['tip'].body.isVisible()


class TestFoldedSummary:
    """A folded section still has to say enough to decide whether to open it."""

    @pytest.fixture(name='form')
    def _form(self, qt_app, monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent'))
        yield form
        form.deleteLater()

    def test_a_filled_section_reports_how_much_is_set(self, form):
        section = form.sections['tip']
        section.set_collapsed(True)
        assert 'set' in section.summary.text()

    def test_missing_required_fields_are_called_out(self, qt_app, monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        blank = ConfigForm(ConfigDocument.blank('afm'))
        section = blank.sections['2D']
        section.set_collapsed(True)
        assert 'required still empty' in section.summary.text()

    def test_a_section_that_is_off_says_so(self, qt_app, monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        blank = ConfigForm(ConfigDocument.blank('afm'))
        assert blank.sections['flake'].summary.text() == 'not included'

    def test_the_summary_only_shows_while_folded(self, form):
        section = form.sections['tip']
        section.set_collapsed(True)
        section.set_collapsed(False)
        assert not section.summary.isVisibleTo(section)


class TestPairOverrides:
    """[lj_override] has user-chosen keys, so introspection renders nothing --
    it used to appear as an empty box."""

    @pytest.fixture(name='section')
    def _section(self, qt_app, monkeypatch):
        from frictionsim2d.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        document = ConfigDocument.blank('afm')
        document.enable_section('lj_override', True)
        # Held for the test's duration: letting the form be collected destroys
        # its Qt children while the Python wrappers survive.
        form = ConfigForm(document)
        yield form.sections['lj_override']
        form.deleteLater()

    def test_the_section_has_a_working_editor(self, section):
        assert section.pair_editor is not None

    def test_pairs_round_trip(self, section):
        section.pair_editor.set_value({'Mo-S': [0.41, 3.75]})
        assert section.pair_editor.value() == {'Mo-S': [0.41, 3.75]}

    def test_editing_writes_through_to_the_document(self, section):
        section.pair_editor.set_value({'Mo-S': [0.41, 3.75]})
        section.commit()
        assert section.document.values['lj_override'] == {'Mo-S': [0.41, 3.75]}

    def test_incomplete_rows_are_skipped(self, section):
        section.pair_editor.set_value({'Mo-S': [0.41, 3.75]})
        section.pair_editor._add_row('S-S', '', '')
        assert section.pair_editor.value() == {'Mo-S': [0.41, 3.75]}


class TestFormEditors:
    """Values must round-trip through the widgets unchanged."""

    @pytest.fixture(name='form')
    def _form(self, qt_app, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from frictionsim2d.gui.widgets.form import ConfigForm
        document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
        return ConfigForm(document)

    def test_string_round_trip(self, form):
        editor = form.sections['2D'].editors['mat']
        editor.set_value('WS2')
        assert editor.value() == 'WS2'

    def test_float_round_trip(self, form):
        editor = form.sections['tip'].editors['r']
        editor.set_value(25.0)
        assert editor.value() == 25.0

    def test_list_round_trip(self, form):
        editor = form.sections['2D'].editors['layers']
        editor.set_value([1, 2, 3])
        assert editor.value() == [1, 2, 3]

    def test_scalar_or_list_keeps_a_scalar_scalar(self, form):
        editor = form.sections['general'].editors['force']
        editor.set_value(5.0)
        assert editor.value() == 5.0

    def test_scalar_or_list_keeps_a_list_a_list(self, form):
        editor = form.sections['general'].editors['force']
        editor.set_value([5, 10, 20])
        assert editor.value() == [5.0, 10.0, 20.0]

    def test_empty_is_none_not_empty_string(self, form):
        """The writer emits None as an empty value; '' would round-trip wrong."""
        editor = form.sections['2D'].editors['lat_c']
        editor.set_value(None)
        assert editor.value() is None

    def test_enum_round_trip(self, form):
        editor = form.sections['tip'].editors['amorph']
        editor.set_value('a')
        assert editor.value() == 'a'

    def test_bool_round_trip(self, form):
        editor = form.sections['general'].editors['finite_sheet']
        editor.set_value(True)
        assert editor.value() is True

    def test_committing_writes_through_to_the_document(self, form):
        form.sections['2D'].editors['mat'].set_value('WSe2')
        form.commit()
        assert form.document.get('2D', 'mat') == 'WSe2'

    def test_full_form_round_trip_preserves_the_config(self, form, tmp_path):
        """Building the form and committing it unchanged must not alter values."""
        from frictionsim2d.core.config import parse_config

        before = form.document.to_dict()
        form.commit()
        out = form.document.save(tmp_path / 'c.ini')
        assert parse_config(out) == before


class TestCloseConfirmation:
    """Closing the window used to silently discard an edited config or
    settings document. Now it must ask first."""

    @pytest.fixture(name='window')
    def _window(self, qt_app, workspace, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from frictionsim2d.gui.app import MainWindow
        window = MainWindow(workspace)
        yield window
        # Several tests here deliberately leave the document/settings dirty
        # (and, within their own body, monkeypatch QMessageBox.question --
        # a monkeypatch that is undone by the time teardown runs). Without
        # resetting dirty first, this close() would hit the real, blocking
        # QMessageBox with nothing able to click it.
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()

    def test_close_prompts_when_dirty(self, window, monkeypatch):
        window.document.dirty = True
        asked = []
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: asked.append(1) or QMessageBox.Cancel)
        event = QCloseEvent()
        window.closeEvent(event)
        assert asked and not event.isAccepted()

    def test_close_prompts_when_only_settings_are_dirty(self, window, monkeypatch):
        window.document.dirty = False
        window.settings_document.dirty = True
        asked = []
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: asked.append(1) or QMessageBox.Cancel)
        event = QCloseEvent()
        window.closeEvent(event)
        assert asked and not event.isAccepted()

    def test_close_does_not_prompt_when_clean(self, window, monkeypatch):
        window.document.dirty = False
        window.settings_document.dirty = False
        asked = []
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: asked.append(1) or QMessageBox.Cancel)
        event = QCloseEvent()
        window.closeEvent(event)
        assert not asked
        assert event.isAccepted()

    def test_close_discard_accepts_without_saving(self, window, monkeypatch):
        window.document.dirty = True
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Discard)
        saved = []
        monkeypatch.setattr(window, '_save_config', lambda: saved.append(1))
        event = QCloseEvent()
        window.closeEvent(event)
        assert event.isAccepted()
        assert not saved

    def test_close_save_routes_through_save_config_and_accepts(self, window, monkeypatch):
        window.document.dirty = True
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Save)

        def fake_save():
            window.document.dirty = False
        saved = []
        monkeypatch.setattr(window, '_save_config',
                            lambda: (saved.append(1), fake_save())[-1])
        event = QCloseEvent()
        window.closeEvent(event)
        assert saved
        assert event.isAccepted()

    def test_cancelled_save_dialog_does_not_close_the_window(self, window, monkeypatch):
        """Choosing Save but then cancelling the file dialog leaves the
        document dirty -- the window must stay open, not close anyway."""
        window.document.dirty = True
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Save)
        # _save_config's own file dialog was cancelled: it returns without
        # touching dirty.
        monkeypatch.setattr(window, '_save_config', lambda: None)
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()

    def test_close_save_routes_through_settings_save_when_only_settings_dirty(
            self, window, monkeypatch):
        """A settings-only edit must save through the settings path, not the
        config one -- routing it to _save_config used to open the wrong
        dialog and then close the window with the settings edits lost."""
        window.document.dirty = False
        window.settings_document.dirty = True
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Save)

        def fake_save_settings():
            window.settings_document.dirty = False
        settings_saved = []
        config_saved = []
        monkeypatch.setattr(
            window, '_save_settings',
            lambda: (settings_saved.append(1), fake_save_settings())[-1])
        monkeypatch.setattr(window, '_save_config', lambda: config_saved.append(1))
        event = QCloseEvent()
        window.closeEvent(event)
        assert settings_saved
        assert not config_saved
        assert event.isAccepted()

    def test_settings_save_failure_does_not_close_the_window(self, window, monkeypatch):
        """An invalid settings document (or a save the user backs out of)
        leaves settings_document.dirty True -- the window must stay open,
        not close with the edits discarded."""
        window.document.dirty = False
        window.settings_document.dirty = True
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Save)
        # _save_settings warns and returns without clearing dirty -- the
        # real method does this when validation fails.
        monkeypatch.setattr(window, '_save_settings', lambda: None)
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()

    def test_close_save_message_names_what_is_unsaved(self, window, monkeypatch):
        """The settings-only case must not blame 'the configuration' -- that
        used to be the only wording, regardless of which document was dirty."""
        window.document.dirty = False
        window.settings_document.dirty = True
        messages = []

        def fake_question(*args, **kwargs):
            messages.append(args[2])
            return QMessageBox.Cancel
        monkeypatch.setattr(QMessageBox, 'question', fake_question)
        event = QCloseEvent()
        window.closeEvent(event)
        assert messages and 'settings' in messages[0].lower()
        assert 'configuration' not in messages[0].lower()


class TestStaleRenderBadge:
    """The viewport keeps showing the previous system after an edit unless
    told otherwise -- the badge is the only sign anything is out of date."""

    @pytest.fixture(name='window')
    def _window(self, qt_app, workspace, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        from frictionsim2d.gui.app import MainWindow
        window = MainWindow(workspace)
        yield window
        # See TestCloseConfirmation's fixture: avoid a real, blocking save
        # prompt from closeEvent when a test's edits left the document dirty.
        window.document.dirty = False
        window.settings_document.dirty = False
        window.close()

    def test_stale_badge_toggles(self, qt_app, two_component_system):
        from frictionsim2d.gui.preview.viewport import ViewportPanel

        panel = ViewportPanel()
        panel.set_stale(True)
        assert panel.stale_label.isVisibleTo(panel) is False   # no system yet
        panel.system = two_component_system
        panel.set_stale(True)
        assert panel.stale_label.isVisibleTo(panel)
        panel.set_stale(False)
        assert not panel.stale_label.isVisibleTo(panel)

    def test_editing_after_a_render_marks_the_badge_stale(self, window, monkeypatch,
                                                          two_component_system):
        monkeypatch.setattr(window.viewport.__class__, '_update_pipeline',
                            lambda self: None)
        monkeypatch.setattr(window.viewport.__class__, '_ensure_widget',
                            lambda self: None)

        window._on_preview_ready(two_component_system)
        assert window._render_snapshot == window.document.to_dict()
        assert window.viewport.stale_label.isVisibleTo(window.viewport) is False

        window.form.sections['2D'].editors['mat'].set_value('WS2')
        window.form.commit()
        window._on_config_changed()
        assert window.viewport.stale_label.isVisibleTo(window.viewport)

    def test_re_rendering_clears_the_badge(self, window, monkeypatch,
                                           two_component_system):
        monkeypatch.setattr(window.viewport.__class__, '_update_pipeline',
                            lambda self: None)
        monkeypatch.setattr(window.viewport.__class__, '_ensure_widget',
                            lambda self: None)

        window._on_preview_ready(two_component_system)
        window.form.sections['2D'].editors['mat'].set_value('WS2')
        window.form.commit()
        window._on_config_changed()
        assert window.viewport.stale_label.isVisibleTo(window.viewport)

        window._on_preview_ready(two_component_system)
        assert window.viewport.stale_label.isVisibleTo(window.viewport) is False

    def test_opening_a_config_marks_the_badge_stale(self, window, monkeypatch,
                                                     two_component_system):
        """Opening a different config swaps the whole document out from
        under a render -- the largest staleness event there is -- yet
        `_open_config` never emitted `changed` (`load()` runs under
        `_loading`), so the badge stayed hidden until the user separately
        edited a field by hand."""
        monkeypatch.setattr(window.viewport.__class__, '_update_pipeline',
                            lambda self: None)
        monkeypatch.setattr(window.viewport.__class__, '_ensure_widget',
                            lambda self: None)

        window._on_preview_ready(two_component_system)
        assert window.viewport.stale_label.isVisibleTo(window.viewport) is False

        monkeypatch.setattr(
            QFileDialog, 'getOpenFileName',
            lambda *a, **k: (str(EXAMPLES / 'afm_config.ini'), ''))
        window._open_config()
        assert window.viewport.stale_label.isVisibleTo(window.viewport)
