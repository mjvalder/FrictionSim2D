"""The Amorphous block (quench) family in the GUI.

Nothing here starts LAMMPS: the serial-launch boundary and `subprocess` are
patched to refuse wherever a test is not itself about a launch, and the one
test that does launch (Cancel) uses a stand-in `lmp` that only sleeps.
"""
from __future__ import annotations

import os
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from frictionsim2d.builders.construction import amorphous  # noqa: E402
from frictionsim2d.core.config import QuenchRunConfig, load_settings  # noqa: E402
from frictionsim2d.gui import generate_worker, preview_worker  # noqa: E402
from frictionsim2d.gui.app import MainWindow  # noqa: E402
from frictionsim2d.gui.controllers.document import ConfigDocument  # noqa: E402
from frictionsim2d.gui.controllers.quench import (  # noqa: E402
    CONDITION_KEYS, prefill_quench_values)
from frictionsim2d.gui.controllers.quench_run import QuenchRunController  # noqa: E402
from frictionsim2d.gui.controllers.staged_paths import absolutize_staged_paths  # noqa: E402
from frictionsim2d.gui.schema.overrides import CAPTION_BUDGET, overrides_for  # noqa: E402
from frictionsim2d.gui.schema.simulation_types import (  # noqa: E402
    FAMILIES, MODES, family_for_model, sections_for)
from frictionsim2d.gui.workspace import Workspace  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / 'examples' / 'quench_config.ini'


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(name='no_lammps')
def _no_lammps(monkeypatch):
    """Record any attempt to start LAMMPS; none is expected."""
    calls = []
    real_run, real_popen = subprocess.run, subprocess.Popen

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError(f'unexpected LAMMPS start: {args}')

    def guarded_run(*args, **kwargs):
        if 'lmp' in str(args[0] if args else kwargs.get('args')):
            return refuse(*args, **kwargs)
        return real_run(*args, **kwargs)

    class GuardedPopen(real_popen):
        def __init__(self, *args, **kwargs):
            if 'lmp' in str(args[0] if args else kwargs.get('args')):
                refuse(*args, **kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', guarded_run)
    monkeypatch.setattr(subprocess, 'Popen', GuardedPopen)
    monkeypatch.setattr(amorphous, 'run_lmp_serial', refuse)
    return calls


@pytest.fixture(name='window')
def _window(qt_app, tmp_path, monkeypatch):
    del qt_app
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg_data'))
    window = MainWindow(Workspace.open(tmp_path / 'ws'))
    # A fresh Settings tab fails its own validation in this tree (the Task 18
    # settings defect, present at the base commit); these tests are about the
    # quench flow, so the settings step is stepped over.
    monkeypatch.setattr(window, '_sync_settings', lambda: True)
    yield window
    window.document.dirty = False
    window.settings_document.dirty = False
    window.close()


def _quench_ini(tmp_path, **extra) -> Path:
    lines = {
        'material': 'Si',
        'potential': f'{REPO}/examples/potentials/sw/{{mat}}.sw',
        'pot_type': 'sw',
        'cif': f'{REPO}/examples/materials/{{mat}}.cif',
        'slab': '[12, 12, 12]',
        'melt_temp': '3000',
        **{k: str(v) for k, v in extra.items()},
    }
    path = tmp_path / 'quench.ini'
    path.write_text('[quench]\n' + ''.join(f'{k} = {v}\n' for k, v in lines.items()),
                    encoding='utf-8')
    return path


def _document(path: Path) -> ConfigDocument:
    return ConfigDocument.from_file(path, 'quench')


def _miss_payload(tmp_path, monkeypatch) -> dict:
    """What a friction Generate returns when its amorphous block is missing."""
    monkeypatch.chdir(REPO)
    settings = tmp_path / 'settings.yaml'
    settings.write_text(yaml.safe_dump(
        {'quench': {'melt_temp': 3000, 'quench_slab_dims': [12, 12, 12]}}), encoding='utf-8')
    result = generate_worker.run({
        'config_file': 'examples/afm_config.ini', 'model': 'afm',
        'output_root': str(tmp_path / 'out'), 'settings_file': str(settings)})
    assert result['ok'] is False and result['quench_members']
    return result


class TestFamilySchema:

    def test_the_family_is_registered_with_one_mode(self):
        family = family_for_model('quench')
        assert family.key == 'amorphous' and family.label == 'Amorphous block'
        assert [m.model for m in family.modes] == ['quench']
        assert FAMILIES[-1] is family
        mode = family.modes[0]
        assert mode.key == 'melt_quench'
        assert not mode.has_system_phase and not mode.supports_hetero
        assert not family.has_layer_levels
        assert family.outer_loop_choices_for(mode) == ()

    def test_the_only_section_is_quench(self):
        assert sections_for('quench') == ['quench']

    def test_every_field_is_labelled_within_the_caption_budget(self):
        override = overrides_for('quench')['quench']
        assert set(override.fields) == set(QuenchRunConfig.model_fields)
        for key, field in override.fields.items():
            assert field.label, key
            assert len(field.help or '') <= CAPTION_BUDGET, key

    def test_no_integrity_threshold_is_offered(self):
        """CFG-19: thresholds judge results; they are not a quench condition."""
        document = ConfigDocument.blank('quench')
        keys = {f.key for s in document.sections() for f in s.fields}
        assert keys == set(QuenchRunConfig.model_fields)
        assert not [k for k in keys if 'integrity' in k or 'threshold' in k]

    def test_the_units_are_the_quench_units(self):
        fields = overrides_for('quench')['quench'].fields
        assert (fields['melt_temp'].unit, fields['quench_rate'].unit,
                fields['timestep'].unit) == ('K', 'K/s', 'ps')


class TestQuenchDocument:

    def test_the_example_config_is_valid_and_already_satisfied(self):
        document = _document(EXAMPLE)
        document.workspace_root = REPO
        assert document.validate().ok
        # The example reproduces the shipped block: nothing to make.
        assert document.expansion().jobs == 0
        assert document.expansion().notes[0].startswith('[quench] Si')

    def test_a_changed_condition_is_one_job_in_the_quench_layout(self, tmp_path):
        document = _document(_quench_ini(tmp_path))
        summary = document.expansion()
        assert (summary.configs, summary.jobs) == (1, 1)
        (rel,) = document.expansion_output_dirs().dirs
        assert rel.parts[:2] == ('quench', 'Si') and len(rel.parts[2]) == 8

    def test_a_list_makes_one_job_per_value_and_is_never_unsupported(self, tmp_path):
        document = _document(_quench_ini(tmp_path, melt_temp='[2800, 3000]'))
        assert document.expansion().jobs == 2
        assert document.unsupported_lists() == []

    def test_a_missing_potential_is_reported_on_the_field(self, tmp_path):
        ini = _quench_ini(tmp_path)
        ini.write_text(ini.read_text().replace('{mat}.sw', 'nope.sw'), encoding='utf-8')
        issues = _document(ini).validate().issues
        assert [(i.section, i.key) for i in issues] == [('quench', 'potential')]

    def test_an_incomplete_section_is_reported_and_not_expanded(self):
        document = ConfigDocument.blank('quench')
        assert {i.key for i in document.validate().issues} >= {'material', 'potential'}
        assert document.expansion_output_dirs().error

    def test_save_and_reload_round_trip(self, tmp_path):
        document = _document(_quench_ini(tmp_path))
        target = tmp_path / 'again.ini'
        document.save(target)
        assert _document(target).values == document.values
        assert '[quench]' in target.read_text(encoding='utf-8')

    def test_a_second_material_is_refused(self):
        document = ConfigDocument.blank('afm')
        document.add_second_material()
        with pytest.raises(Exception, match='no heterostructure build'):
            document.set_model('quench')

    def test_staging_makes_the_potential_and_cif_absolute(self, tmp_path):
        (tmp_path / 'cfg').mkdir()
        (tmp_path / 'cfg' / 'Si.sw').write_text('x')
        staged = absolutize_staged_paths(
            {'quench': {'potential': 'Si.sw', 'cif': 'cif/{mat}.cif'}}, tmp_path / 'cfg')
        assert staged['quench']['potential'] == str(tmp_path / 'cfg' / 'Si.sw')


class TestPrefill:

    @staticmethod
    def member(material='Si', **over):
        base = {'material': material, 'potential': f'/lib/{material}.sw',
                'pot_type': 'sw', 'cif': f'/lib/{material}.cif', 'slab': [12, 12, 12],
                'melt_temp': 3000.0, 'quench_temp': 300.0, 'quench_rate': 1e12,
                'timestep': 0.002, 'melt_steps': 20000, 'equilibrate_steps': 10000}
        return {**base, **over}

    def test_one_member_keeps_its_literal_paths_and_conditions(self):
        values, left = prefill_quench_values([self.member()])
        assert left == []
        assert values['material'] == 'Si' and values['potential'] == '/lib/Si.sw'
        assert all(values[k] == self.member()[k] for k in CONDITION_KEYS)
        assert values['slab'] == [12.0, 12.0, 12.0]

    def test_shared_conditions_make_a_material_list_with_a_template(self):
        values, left = prefill_quench_values([
            self.member('Si'), self.member('Ge', slab=[20, 10, 12])])
        assert left == []
        assert values['material'] == ['Si', 'Ge']
        assert values['potential'] == '/lib/{mat}.sw'
        assert values['slab'] == [20.0, 12.0, 12.0]

    def test_other_conditions_are_left_out_not_multiplied(self):
        values, left = prefill_quench_values([
            self.member('Si'), self.member('Ge', melt_temp=2800.0)])
        assert values['material'] == 'Si'
        assert [m['material'] for m in left] == ['Ge']


class TestMissButton:

    def test_the_button_opens_a_prefilled_quench_document_and_runs_nothing(
            self, window, tmp_path, monkeypatch, no_lammps):
        payload = _miss_payload(tmp_path, monkeypatch)
        shown = []
        monkeypatch.setattr(window, '_offer_quench_job',
                            lambda message: shown.append(message) or True)
        window._on_generate_failed.__func__  # the slot exists
        window.generator.last_failure = payload
        window._on_generate_failed(payload['error'])

        assert 'No amorphous Si block' in shown[0]
        assert 'run quench --for' in shown[0]
        assert window.document.model == 'quench'
        assert window.family_box.currentData() == 'amorphous'
        values = window.document.values['quench']
        assert values['material'] == 'Si' and values['melt_temp'] == 3000.0
        assert window.document.validate(window.settings).ok
        # Opened, not generated: no worker, no LAMMPS, nothing ticked.
        assert not window.generator.running and not window.preview.running
        assert not window.quench_run.running
        assert not window.run_local_box.isChecked()
        assert no_lammps == []

    def test_close_leaves_the_friction_config_alone(self, window, tmp_path, monkeypatch):
        payload = _miss_payload(tmp_path, monkeypatch)
        before = window.document
        monkeypatch.setattr(window, '_offer_quench_job', lambda message: False)
        window.preview.last_failure = payload
        window._on_preview_failed(payload['error'])
        assert window.document is before and window.document.model == 'afm'

    def test_an_edited_friction_config_is_not_replaced_without_a_yes(
            self, window, tmp_path, monkeypatch):
        payload = _miss_payload(tmp_path, monkeypatch)
        window.document.dirty = True
        before = window.document
        monkeypatch.setattr(window, '_offer_quench_job', lambda message: True)
        monkeypatch.setattr(window, '_confirm_replace_config', lambda: False)
        window.preview.last_failure = payload
        window._on_preview_failed(payload['error'])
        assert window.document is before
        monkeypatch.setattr(window, '_confirm_replace_config', lambda: True)
        window._on_preview_failed(payload['error'])
        assert window.document.model == 'quench'

    def test_a_failure_without_members_is_the_plain_warning(self, window, monkeypatch):
        warned = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: warned.append(a[1]))
        window.preview.last_failure = {'ok': False, 'error': 'boom'}
        window._on_preview_failed('boom')
        assert warned == ['Preview failed']


class TestWindow:

    def test_the_family_box_offers_it_and_run_locally_is_quench_only(self, window):
        assert window.family_box.findData('amorphous') >= 0
        assert not window.run_local_box.isVisibleTo(window)
        window.family_box.setCurrentIndex(window.family_box.findData('amorphous'))
        assert window.document.model == 'quench'
        assert window.run_local_box.isVisibleTo(window)
        assert not window.run_local_box.isChecked()
        window.run_local_box.setChecked(True)
        window.family_box.setCurrentIndex(window.family_box.findData('afm'))
        assert not window.run_local_box.isVisibleTo(window)
        window.family_box.setCurrentIndex(window.family_box.findData('amorphous'))
        assert not window.run_local_box.isChecked()

    def test_opening_a_quench_config_selects_its_family(self, window, tmp_path, monkeypatch):
        ini = _quench_ini(tmp_path)
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(ini), ''))
        window._open_config()
        assert window.document.model == 'quench'
        assert window.family_box.currentData() == 'amorphous'
        assert window.counts.text() == '1 block → 1 job'

    def _ready(self, window, tmp_path, monkeypatch):
        ini = _quench_ini(tmp_path)
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(ini), ''))
        window._open_config()
        started = []
        monkeypatch.setattr(window.generator, 'start', started.append)
        monkeypatch.setattr(QMessageBox, 'question',
                            lambda *a, **k: QMessageBox.Yes)
        return started

    def test_generate_asks_for_the_quench_model_and_run_locally_stays_off(
            self, window, tmp_path, monkeypatch, no_lammps):
        started = self._ready(window, tmp_path, monkeypatch)
        window._generate()
        assert [r.model for r in started] == ['quench']
        assert window._run_local_requested is False
        ran = []
        monkeypatch.setattr(window.quench_run, 'start', ran.append)
        monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: None)
        window._on_generated({'created': ['a'], 'simulation_root': str(tmp_path)})
        assert ran == [] and no_lammps == []

    def test_run_locally_is_only_started_when_ticked(
            self, window, tmp_path, monkeypatch, no_lammps):
        started = self._ready(window, tmp_path, monkeypatch)
        window.run_local_box.setChecked(True)
        window._generate()
        assert window._run_local_requested is True and len(started) == 1
        ran = []
        monkeypatch.setattr(window.quench_run, 'start', ran.append)
        monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: None)
        window._on_generated({'created': ['a'], 'simulation_root': str(tmp_path)})
        assert ran == [tmp_path] and window._run_local_requested is False

    def test_generate_refuses_when_every_block_is_already_there(
            self, window, monkeypatch, no_lammps):
        window.family_box.setCurrentIndex(window.family_box.findData('amorphous'))
        document = _document(EXAMPLE)
        document.workspace_root = REPO
        window.document = window.form.document = document
        window.form.rebuild()
        told = []
        monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: told.append(a[1]))
        started = []
        monkeypatch.setattr(window.generator, 'start', started.append)
        window._generate()
        assert told == ['Nothing to generate'] and started == []

    def test_the_file_menu_imports_a_chosen_root(self, window, tmp_path, monkeypatch):
        seen = []
        report = amorphous.QuenchImportReport()
        monkeypatch.setattr(amorphous, 'import_quench_root',
                            lambda root, **kw: seen.append((root, kw)) or report)
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                            lambda *a, **k: str(tmp_path))
        monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: None)
        window.import_quench_action.trigger()
        assert seen == [(tmp_path, {'source': 'hpc'})]


class TestWorkers:
    """No worker melts anything: previewing and generating only build decks."""

    def test_preview_builds_the_crystal_slab_only(self, tmp_path, no_lammps):
        result = preview_worker.run({
            'config_file': str(_quench_ini(tmp_path)), 'model': 'quench',
            'output_root': str(tmp_path / 'p'), 'result_file': str(tmp_path / 'r.json')})
        assert result['ok'] and Path(result['crystal_block']).is_file()
        assert result['natoms'] > 0 and result['steps'] > 0
        assert 'Nothing is melted' in result['label']
        assert no_lammps == []

    def test_generate_writes_the_root_and_starts_nothing(self, tmp_path, no_lammps):
        settings = tmp_path / 'settings.yaml'
        settings.write_text(yaml.safe_dump({'hpc': {'modules': ['lammps']}}), encoding='utf-8')
        result = generate_worker.run({
            'config_file': str(_quench_ini(tmp_path)), 'model': 'quench',
            'output_root': str(tmp_path / 'out'), 'generate_hpc': True,
            'settings_file': str(settings), 'result_file': str(tmp_path / 'r.json')})
        assert result['ok'] and len(result['created']) == 1
        root = Path(result['simulation_root'])
        assert list(root.glob('quench/Si/*/lammps/quench.in'))
        assert result['members'][0]['material'] == 'Si'
        assert result['hpc_scripts'] > 0
        assert no_lammps == []

    def test_generate_says_so_when_the_library_has_everything(self, tmp_path, no_lammps):
        result = generate_worker.run({
            'config_file': str(EXAMPLE), 'model': 'quench',
            'output_root': str(tmp_path / 'out'), 'result_file': str(tmp_path / 'r.json')})
        assert result['ok'] is False and result['nothing_to_make']
        assert not (tmp_path / 'out').exists() or not any((tmp_path / 'out').iterdir())

    def test_the_preview_system_is_the_crystal_block(self, tmp_path, no_lammps):
        pytest.importorskip('ovito')
        from frictionsim2d.gui.preview.assemble import assemble_data_file
        result = preview_worker.run({
            'config_file': str(_quench_ini(tmp_path)), 'model': 'quench',
            'output_root': str(tmp_path / 'p'), 'result_file': str(tmp_path / 'r.json')})
        system = assemble_data_file(Path(result['crystal_block']), 'Si', result['elements'])
        assert system.count == result['natoms']
        assert system.legend == {1: 'Si'}
        assert system.cell[1] > system.cell[0]


def _pump(controller_signal, timeout_ms: int) -> list:
    loop = QEventLoop()
    got: list = []
    controller_signal.connect(lambda *a: (got.append(a), loop.quit()))
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    return got


class TestRunLocally:

    @staticmethod
    def _fake_lmp(tmp_path, monkeypatch) -> Path:
        """A stand-in `lmp` that records its pid and sleeps."""
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        pid_file = tmp_path / 'lmp.pid'
        script = bin_dir / 'lmp'
        script.write_text(f'#!/bin/sh\necho $$ > {pid_file}\nexec sleep 600\n')
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
        return pid_file

    def test_stop_active_lmp_ends_the_running_child(self, tmp_path, monkeypatch):
        self._fake_lmp(tmp_path, monkeypatch)
        errors = []

        def run():
            try:
                amorphous.run_lmp_serial('d.in', 'd.log', tmp_path)
            except subprocess.CalledProcessError as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.time() + 10
        while amorphous._ACTIVE_LMP is None and time.time() < deadline:  # pylint: disable=protected-access
            time.sleep(0.05)
        amorphous.stop_active_lmp()
        thread.join(10)
        assert not thread.is_alive() and len(errors) == 1

    def test_cancel_kills_lmp_and_imports_nothing(
            self, qt_app, tmp_path, monkeypatch):
        del qt_app
        monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg'))
        pid_file = self._fake_lmp(tmp_path, monkeypatch)
        members = amorphous.expand_quench_config(
            {'material': 'Si', 'potential': str(REPO / 'examples/potentials/sw/{mat}.sw'),
             'pot_type': 'sw', 'cif': str(REPO / 'examples/materials/{mat}.cif'),
             'slab': [12, 12, 12], 'melt_temp': 3000}, load_settings())
        root = amorphous.write_quench_root(members, tmp_path / 'out', load_settings())

        controller = QuenchRunController(Workspace.open(tmp_path / 'ws'))
        finished, cancelled = [], []
        controller.finished.connect(finished.append)
        controller.cancelled.connect(lambda: cancelled.append(1))
        controller.start(root)
        deadline = time.time() + 60
        while not pid_file.exists() and time.time() < deadline:
            _pump(controller.progress, 200)
        assert pid_file.exists(), 'the worker never started lmp'
        pid = int(pid_file.read_text())

        controller.cancel()
        assert cancelled == [1] and not controller.running and finished == []
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail('the lmp child outlived Cancel')
        assert not (tmp_path / 'xdg' / 'FrictionSim2D' / 'amorphous').exists() or \
            not list((tmp_path / 'xdg' / 'FrictionSim2D' / 'amorphous').glob('*.lmp'))
