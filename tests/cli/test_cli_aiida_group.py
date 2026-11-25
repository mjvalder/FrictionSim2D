"""The aiida CLI surface (spec §4.2): import / export-results / rebuild.

The commands that need a real store are in `tests/aiida/test_cli_aiida.py`.

`aiida dump` is gone -- superseded by `export-results`, which writes a set
back out as an ordinary simulation tree instead of a `postprocess read
--export` blob.
"""
from pathlib import Path

import pytest
from click.testing import CliRunner

from frictionsim2d.cli import cli


class _Report:
    """Stands in for `ingest.ImportReport`: just what the CLI reads off it."""

    def __init__(self, ok=True, unparseable=(), dry_run=False):
        self._ok = ok
        self.unparseable = list(unparseable)
        self.dry_run = dry_run

    def ok(self):
        return self._ok

    def clean(self):
        # The real rule, over this fake's fields.
        from frictionsim2d.aiida.ingest import ImportReport
        return ImportReport.clean(self)

    def render(self):
        return 'set X\n  runs created:    2\n  results created: 4'


def test_import_passes_label_and_dry_run(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'frictionsim2d.aiida.ingest.import_root',
        lambda root, label=None, machine=None, profile=None, dry_run=False,
        settings_file=None, integrity=None:
            seen.update(root=Path(root), label=label, machine=machine, profile=profile,
                        dry_run=dry_run, settings_file=settings_file,
                        integrity=integrity) or _Report())

    result = CliRunner().invoke(cli, [
        'aiida', 'import', str(tmp_path), '--label', '260910-afm',
        '--profile', 'friction2d', '--dry-run', '--machine', 'archer2'])

    assert result.exit_code == 0, result.output
    assert seen == {'root': tmp_path, 'label': '260910-afm', 'machine': 'archer2',
                    'profile': 'friction2d', 'dry_run': True, 'settings_file': None,
                    'integrity': None}
    assert 'results created: 4' in result.output


def test_import_passes_integrity_overrides_through(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr('frictionsim2d.aiida.ingest.import_root',
                        lambda root, **kwargs: seen.update(kwargs) or _Report())

    result = CliRunner().invoke(cli, [
        'aiida', 'import', str(tmp_path), '--integrity', 'hole_bonds=2.5',
        '--integrity', 'margins.hole=0.1', '--integrity', 'ejected_atoms=none'])

    assert result.exit_code == 0, result.output
    assert seen['integrity'] == {'hole_bonds': 2.5, 'ejected_atoms': None,
                                 'margins': {'hole': 0.1}}


@pytest.mark.parametrize('command', ['import', 'recheck'])
@pytest.mark.parametrize('pair', ['nonsense=1', 'hole_bonds', 'isolated_atoms=many'])
def test_a_bad_integrity_override_is_a_usage_error(tmp_path, monkeypatch, command, pair):
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)
    target = str(tmp_path) if command == 'import' else 'camp'

    result = CliRunner().invoke(cli, ['aiida', command, target, '--integrity', pair])

    assert result.exit_code == 2 and '--integrity' in result.output


def test_recheck_passes_integrity_overrides_through(monkeypatch):
    seen = {}
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)

    class _Recheck:
        failures = []

        @staticmethod
        def render():
            return 'rechecked'

    monkeypatch.setattr(
        'frictionsim2d.aiida.steps.recheck_set',
        lambda label, **kwargs: seen.update(label=label, **kwargs) or _Recheck())

    result = CliRunner().invoke(cli, ['aiida', 'recheck', 'camp', '--integrity', 'hole_bonds=3'])

    assert result.exit_code == 0, result.output
    assert seen['label'] == 'camp' and seen['integrity'] == {'hole_bonds': 3}


def test_import_exits_non_zero_on_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr('frictionsim2d.aiida.ingest.import_root',
                        lambda *a, **k: _Report(ok=False))

    result = CliRunner().invoke(cli, ['aiida', 'import', str(tmp_path)])

    assert result.exit_code == 1


def test_import_exits_non_zero_on_unparseable_runs(tmp_path, monkeypatch):
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'frictionsim2d.aiida.ingest.import_root',
        lambda *a, **k: _Report(unparseable=['h_MoS2/broken']))

    result = CliRunner().invoke(cli, ['aiida', 'import', str(tmp_path)])

    assert result.exit_code == 1


def test_import_dry_run_ignores_unparseable(tmp_path, monkeypatch):
    """R8 fix round 1: --dry-run's unparseable paths must not fail it;
    only a real import treats them as failures."""
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'frictionsim2d.aiida.ingest.import_root',
        lambda *a, dry_run=False, **k: _Report(unparseable=['h_MoS2/broken'],
                                               dry_run=dry_run))

    result = CliRunner().invoke(cli, ['aiida', 'import', str(tmp_path), '--dry-run'])

    assert result.exit_code == 0, result.output


def test_import_reports_a_set_root_mismatch_cleanly(tmp_path, monkeypatch):
    from frictionsim2d.aiida.ingest import SetRootMismatch

    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)

    def _raise(*_a, **_k):
        raise SetRootMismatch("label 'x' already names a different tree")

    monkeypatch.setattr('frictionsim2d.aiida.ingest.import_root', _raise)

    result = CliRunner().invoke(cli, ['aiida', 'import', str(tmp_path)])

    assert result.exit_code != 0
    assert result.exc_info[0] is not SetRootMismatch
    assert 'already names a different tree' in result.output


def test_export_results_writes_a_tree(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'frictionsim2d.aiida.export.export_results',
        lambda label, dest, profile=None:
            seen.update(label=label, dest=Path(dest), profile=profile)
            or [Path(dest) / 'a.txt'])

    result = CliRunner().invoke(cli, [
        'aiida', 'export-results', '260910-afm', str(tmp_path / 'out')])

    assert result.exit_code == 0, result.output
    assert seen['label'] == '260910-afm'
    assert seen['dest'] == tmp_path / 'out'
    assert '1 file' in result.output


def test_rebuild_calls_rebuild_set(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'frictionsim2d.aiida.export.rebuild_set',
        lambda label, dest, profile=None, generate_hpc=False, failures=None:
            seen.update(label=label, dest=Path(dest), hpc=generate_hpc)
            or ([Path(dest) / 'run'], Path(dest)))

    result = CliRunner().invoke(cli, [
        'aiida', 'rebuild', '260910-afm', '-o', str(tmp_path), '--hpc-scripts'])

    assert result.exit_code == 0, result.output
    assert seen == {'label': '260910-afm', 'dest': tmp_path, 'hpc': True}


def test_rebuild_exits_nonzero_and_names_the_failed_runs(tmp_path, monkeypatch):
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)

    def _rebuild(_label, dest, *, failures=None, **_kwargs):
        failures.append('afm/Si/100x_100y/K300/L1: amorphous block missing')
        return [], Path(dest)

    monkeypatch.setattr('frictionsim2d.aiida.export.rebuild_set', _rebuild)

    result = CliRunner().invoke(cli, ['aiida', 'rebuild', '260910-afm', '-o', str(tmp_path)])

    assert result.exit_code == 1
    assert 'afm/Si/100x_100y/K300/L1: amorphous block missing' in result.output
    assert 'Rebuilt 0 run(s)' in result.output and '1 failed' in result.output


@pytest.mark.parametrize('target, argv', [
    ('frictionsim2d.aiida.export.export_results', ['export-results', 'nope', 'OUT']),
    ('frictionsim2d.aiida.export.rebuild_set', ['rebuild', 'nope', '-o', 'OUT']),
])
def test_unknown_label_is_reported_cleanly(tmp_path, monkeypatch, target, argv):
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)

    def _raise(*_a, **_k):
        raise LookupError("no simulation set for label or UUID 'nope'")

    monkeypatch.setattr(target, _raise)

    result = CliRunner().invoke(cli, [
        'aiida', *[str(tmp_path / 'out') if a == 'OUT' else a for a in argv]])

    assert result.exit_code != 0
    assert result.exc_info[0] is not LookupError
    assert 'nope' in result.output



def test_import_takes_the_profile_from_an_explicit_settings_file(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr('frictionsim2d.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr('frictionsim2d.aiida.ingest.import_root',
                        lambda root, **kwargs: seen.update(kwargs) or _Report())
    settings = tmp_path / 'settings.yaml'
    settings.write_text('aiida:\n  profile: from-file\n', encoding='utf-8')

    result = CliRunner().invoke(cli, ['aiida', 'import', str(tmp_path),
                                      '--settings-file', str(settings)])

    assert result.exit_code == 0, result.output
    assert seen['profile'] == 'from-file' and seen['settings_file'] == settings


@pytest.mark.parametrize('command', ['rule', 'reindex', 'verify', 'postprocess', 'dataset'])
def test_the_store_commands_exist(command):
    result = CliRunner().invoke(cli, ['aiida', command, '--help'])
    assert result.exit_code == 0, result.output
