"""HPC script generation must not fail silently.

It has a precondition that is easy to miss -- ``settings.hpc.modules`` defaults
to unset, and without it nothing is written. That was reported only to the log,
so asking for job scripts appeared to succeed while producing none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import GlobalSettings
from src.run.artifacts import generate_package
from src.run.hpc_dispatch import generate_hpc_scripts_for_root, hpc_generation_blocker

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'examples'


def _settings_with_modules() -> GlobalSettings:
    settings = GlobalSettings()
    settings.hpc.modules = ['lammps', 'openmpi']
    return settings


class TestBlocker:

    def test_unset_modules_are_reported_as_the_reason(self):
        reason = hpc_generation_blocker(GlobalSettings())
        assert reason is not None
        assert 'modules' in reason

    def test_the_reason_names_the_setting_to_change(self):
        assert "'Modules'" in hpc_generation_blocker(GlobalSettings())

    def test_the_reason_says_the_other_hpc_settings_do_not_substitute(self):
        """Filling in scheduler, nodes and walltime looks like enough."""
        reason = hpc_generation_blocker(GlobalSettings())
        assert 'substitute' in reason
        assert 'scheduler' in reason

    def test_configured_modules_are_not_blocked(self):
        assert hpc_generation_blocker(_settings_with_modules()) is None

    def test_an_empty_list_is_still_a_blocker(self):
        settings = GlobalSettings()
        settings.hpc.modules = []
        assert hpc_generation_blocker(settings) is not None


class TestGenerationReturnsItsOutcome:

    @pytest.fixture(name='simulation_root')
    def _simulation_root(self, tmp_path, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        result = generate_package(
            EXAMPLES / 'indent_config.ini', 'indent',
            output_root=tmp_path, generate_hpc=False,
        )
        assert result.ok, result.render()
        return result.simulation_root

    def test_it_reports_why_nothing_was_written(self, simulation_root):
        reason = generate_hpc_scripts_for_root(simulation_root, GlobalSettings())
        assert reason is not None
        assert not (simulation_root / 'hpc').exists()

    def test_it_returns_none_when_scripts_are_written(self, simulation_root):
        assert generate_hpc_scripts_for_root(
            simulation_root, _settings_with_modules()) is None
        assert list((simulation_root / 'hpc').iterdir())


class TestWorkerPayload:
    """What the window shows after generating comes from here."""

    def _run(self, tmp_path, settings_file=None):
        from src.gui.generate_worker import run

        return run({
            'config_file': str(EXAMPLES / 'indent_config.ini'),
            'model': 'indent',
            'output_root': str(tmp_path / 'out'),
            'generate_hpc': True,
            'settings_file': str(settings_file) if settings_file else None,
            'root_name': 'r',
            'result_file': str(tmp_path / 'result.json'),
        })

    def test_it_says_no_scripts_were_written_and_why(self, tmp_path, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        payload = self._run(tmp_path)
        assert payload['ok']
        assert payload['hpc_requested'] is True
        assert payload['hpc_scripts'] == 0
        assert 'modules' in payload['hpc_blocked']

    def test_it_counts_the_scripts_it_wrote(self, tmp_path, monkeypatch):
        import yaml

        monkeypatch.chdir(REPO_ROOT)
        settings_file = tmp_path / 'settings.yaml'
        settings_file.write_text(
            yaml.safe_dump(_settings_with_modules().model_dump(mode='json')),
            encoding='utf-8')

        payload = self._run(tmp_path, settings_file)
        assert payload['hpc_scripts'] > 0
        assert payload['hpc_blocked'] is None

    def test_hpc_keys_are_absent_when_not_requested(self, tmp_path, monkeypatch):
        from src.gui.generate_worker import run

        monkeypatch.chdir(REPO_ROOT)
        payload = run({
            'config_file': str(EXAMPLES / 'indent_config.ini'),
            'model': 'indent',
            'output_root': str(tmp_path / 'out'),
            'generate_hpc': False,
            'settings_file': None,
            'root_name': 'r',
            'result_file': str(tmp_path / 'result.json'),
        })
        assert 'hpc_requested' not in payload
