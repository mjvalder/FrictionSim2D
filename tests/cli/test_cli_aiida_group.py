"""The aiida CLI surface (spec §4.2): import / export-results / rebuild.

`aiida dump` is gone -- superseded by `export-results`, which writes a set
back out as an ordinary simulation tree instead of a `postprocess read
--export` blob.
"""
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.cli import cli


class _Report:
    """Stands in for `ingest.IngestReport`: just what the CLI reads off it."""

    def __init__(self, ok=True, empty_runs=(), unparseable=()):
        self._ok = ok
        self.empty_runs = list(empty_runs)
        self.unparseable = list(unparseable)

    def ok(self):
        return self._ok

    def render(self):
        return 'set X\n  runs created:    2\n  results created: 4'


def test_import_passes_label_and_dry_run(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'src.aiida.ingest.ingest_root',
        lambda root, label=None, profile=None, dry_run=False:
            seen.update(root=Path(root), label=label, profile=profile,
                        dry_run=dry_run) or _Report())

    result = CliRunner().invoke(cli, [
        'aiida', 'import', str(tmp_path), '--label', '260910-afm',
        '--profile', 'friction2d', '--dry-run'])

    assert result.exit_code == 0, result.output
    assert seen == {'root': tmp_path, 'label': '260910-afm',
                    'profile': 'friction2d', 'dry_run': True}
    assert 'results created: 4' in result.output


def test_import_exits_non_zero_on_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr('src.aiida.ingest.ingest_root',
                        lambda *a, **k: _Report(ok=False))

    result = CliRunner().invoke(cli, ['aiida', 'import', str(tmp_path)])

    assert result.exit_code == 1


def test_import_exits_non_zero_on_empty_runs(tmp_path, monkeypatch):
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'src.aiida.ingest.ingest_root',
        lambda *a, **k: _Report(empty_runs=['h_MoS2/L1']))

    result = CliRunner().invoke(cli, ['aiida', 'import', str(tmp_path)])

    assert result.exit_code == 1


def test_import_exits_non_zero_on_unparseable_runs(tmp_path, monkeypatch):
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'src.aiida.ingest.ingest_root',
        lambda *a, **k: _Report(unparseable=['h_MoS2/broken']))

    result = CliRunner().invoke(cli, ['aiida', 'import', str(tmp_path)])

    assert result.exit_code == 1


def test_import_dry_run_ignores_empty_runs_and_unparseable(tmp_path, monkeypatch):
    """R8 fix round 1: --dry-run's empty_runs/unparseable must not fail it.

    A dry-run preview of a fresh, pre-HPC tree is legitimately all-empty; only
    a real import treats empty/unparseable runs as failures.
    """
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'src.aiida.ingest.ingest_root',
        lambda *a, **k: _Report(empty_runs=['h_MoS2/L1'],
                                unparseable=['h_MoS2/broken']))

    result = CliRunner().invoke(cli, ['aiida', 'import', str(tmp_path), '--dry-run'])

    assert result.exit_code == 0, result.output


def test_import_reports_a_set_root_mismatch_cleanly(tmp_path, monkeypatch):
    from src.aiida.ingest import SetRootMismatch

    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)

    def _raise(*_a, **_k):
        raise SetRootMismatch("label 'x' already names a different tree")

    monkeypatch.setattr('src.aiida.ingest.ingest_root', _raise)

    result = CliRunner().invoke(cli, ['aiida', 'import', str(tmp_path)])

    assert result.exit_code != 0
    assert result.exc_info[0] is not SetRootMismatch
    assert 'already names a different tree' in result.output


def test_export_results_writes_a_tree(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'src.aiida.export.export_results',
        lambda label, dest, profile=None:
            seen.update(label=label, dest=Path(dest), profile=profile)
            or [Path(dest) / 'a.txt'])

    result = CliRunner().invoke(cli, [
        'aiida', 'export-results', '260910-afm', str(tmp_path / 'out')])

    assert result.exit_code == 0, result.output
    assert seen['label'] == '260910-afm'
    assert seen['dest'] == tmp_path / 'out'
    assert '1 file' in result.output


def test_export_results_reports_an_unknown_label_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)

    def _raise(*_a, **_k):
        raise LookupError("no simulation set for label or UUID 'nope'")

    monkeypatch.setattr('src.aiida.export.export_results', _raise)

    result = CliRunner().invoke(cli, [
        'aiida', 'export-results', 'nope', str(tmp_path / 'out')])

    assert result.exit_code != 0
    assert result.exc_info[0] is not LookupError
    assert 'nope' in result.output


def test_rebuild_calls_rebuild_set(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)
    monkeypatch.setattr(
        'src.aiida.export.rebuild_set',
        lambda label, dest, profile=None, generate_hpc=False:
            seen.update(label=label, dest=Path(dest), hpc=generate_hpc)
            or ([Path(dest) / 'run'], Path(dest)))

    result = CliRunner().invoke(cli, [
        'aiida', 'rebuild', '260910-afm', '-o', str(tmp_path), '--hpc-scripts'])

    assert result.exit_code == 0, result.output
    assert seen == {'label': '260910-afm', 'dest': tmp_path, 'hpc': True}


def test_rebuild_reports_an_unknown_label_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', True)

    def _raise(*_a, **_k):
        raise LookupError("no simulation set for label or UUID 'nope'")

    monkeypatch.setattr('src.aiida.export.rebuild_set', _raise)

    result = CliRunner().invoke(cli, [
        'aiida', 'rebuild', 'nope', '-o', str(tmp_path)])

    assert result.exit_code != 0
    assert result.exc_info[0] is not LookupError
    assert 'nope' in result.output


def test_dump_is_gone():
    result = CliRunner().invoke(cli, ['aiida', 'dump', '--help'])

    assert result.exit_code != 0
    assert 'export-results' in CliRunner().invoke(cli, ['aiida', '--help']).output


# --- stamp-uuid (Task 5b): legacy runs get a run_uuid, with no aiida import ---

def _legacy(path):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema_version': '2.1.0', 'family': 'afm'}),
                    encoding='utf-8')
    return path


def test_stamp_uuid_stamps_and_dry_run_does_not(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr('src.aiida.AIIDA_AVAILABLE', False)   # needs no aiida
    prov = _legacy(tmp_path / 'afm' / 'L1' / 'provenance.json')

    dry = CliRunner().invoke(cli, ['aiida', 'stamp-uuid', str(tmp_path), '--dry-run'])
    assert dry.exit_code == 0, dry.output
    assert 'would stamp 1' in dry.output
    assert 'run_uuid' not in json.loads(prov.read_text(encoding='utf-8'))

    real = CliRunner().invoke(cli, ['aiida', 'stamp-uuid', str(tmp_path)])
    assert real.exit_code == 0, real.output
    assert 'stamped 1' in real.output
    assert json.loads(prov.read_text(encoding='utf-8'))['run_uuid']


def test_stamp_uuid_exits_1_on_an_unreadable_file(tmp_path):
    bad = tmp_path / 'x' / 'provenance.json'
    bad.parent.mkdir()
    bad.write_text('{', encoding='utf-8')
    result = CliRunner().invoke(cli, ['aiida', 'stamp-uuid', str(tmp_path)])
    assert result.exit_code == 1
    assert str(bad) in result.output


def test_stamp_uuid_imports_no_aiida(tmp_path):
    import subprocess
    import sys
    _legacy(tmp_path / 'provenance.json')
    code = ('import sys\n'
            'from src.cli import cli\n'
            f'try:\n    cli(["aiida", "stamp-uuid", {str(tmp_path)!r}])\n'
            'except SystemExit as exc:\n    assert not exc.code, exc.code\n'
            'bad = sorted(m for m in sys.modules if m == "aiida" or m.startswith("aiida."))\n'
            'assert not bad, bad\n')
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[2], check=False)
    assert done.returncode == 0, done.stderr
