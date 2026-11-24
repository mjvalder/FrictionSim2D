"""`run quench`, `quench import`, `quench list`, and a friction run's miss.

Nothing here runs LAMMPS: `run quench` only writes decks unless `--local`, and
`--local` is exercised with the serial-launch boundary
(`amorphous.run_lmp_serial`) replaced by a stand-in that writes what a finished
job leaves behind.
"""
from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from frictionsim2d import cli as cli_module
from frictionsim2d.builders.construction import amorphous
from frictionsim2d.builders.construction.amorphous import QuenchKey, find_block
from frictionsim2d.core.config import QuenchSettings

REPO = Path(__file__).resolve().parents[2]
SI_POT = REPO / 'examples' / 'potentials' / 'sw' / 'Si.sw'


@pytest.fixture(autouse=True)
def _data_home(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg_data'))


@pytest.fixture(name='no_lammps')
def _no_lammps(monkeypatch):
    """Record any attempt to start LAMMPS; none is expected (git still runs)."""
    calls = []
    real_run = subprocess.run

    def refuse(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(f'unexpected LAMMPS start: {args}')

    def guarded_run(*args, **kwargs):
        if 'lmp' in str(args[0] if args else kwargs.get('args')):
            return refuse(*args, **kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', guarded_run)
    monkeypatch.setattr(amorphous, 'run_lmp_serial', refuse)
    return calls


def _quench_ini(tmp_path, melt_temp=3000) -> Path:
    path = tmp_path / 'quench.ini'
    path.write_text(
        '[quench]\n'
        'material = Si\n'
        f'potential = {REPO}/examples/potentials/sw/{{mat}}.sw\n'
        'pot_type = sw\n'
        f'cif = {REPO}/examples/materials/{{mat}}.cif\n'
        'slab = [12, 12, 12]\n'
        f'melt_temp = {melt_temp}\n', encoding='utf-8')
    return path


def _settings_file(tmp_path, **quench) -> Path:
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump({'quench': quench}), encoding='utf-8')
    return path


def _invoke(*args):
    return CliRunner().invoke(cli_module.cli, [str(a) for a in args])


def test_run_quench_writes_the_root_and_starts_nothing(tmp_path, no_lammps):
    result = _invoke('run', 'quench', _quench_ini(tmp_path), '-o', tmp_path / 'out')
    assert result.exit_code == 0, result.output
    roots = list((tmp_path / 'out').glob('quench_*'))
    assert len(roots) == 1
    decks = list(roots[0].glob('quench/Si/*/lammps/quench.in'))
    assert len(decks) == 1
    assert no_lammps == []


def test_run_quench_list_fields_expand_into_members(tmp_path, no_lammps):
    ini = _quench_ini(tmp_path)
    ini.write_text(ini.read_text().replace('melt_temp = 3000', 'melt_temp = [2800, 3000]'))
    result = _invoke('run', 'quench', ini, '-o', tmp_path / 'out')
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / 'out').glob('quench_*/quench/Si/*'))) == 2


def test_run_quench_skips_a_satisfied_member_unless_force(tmp_path, no_lammps):
    ini = _quench_ini(tmp_path, melt_temp=2500)
    ini.write_text(ini.read_text().replace('slab = [12, 12, 12]', 'slab = [12, 12, 12]\n'
                                           'melt_steps = 20000'))
    result = _invoke('run', 'quench', ini, '-o', tmp_path / 'out')
    assert result.exit_code == 0, result.output
    assert 'all amorphous blocks present' in result.output
    assert not list((tmp_path / 'out').glob('quench_*'))
    forced = _invoke('run', 'quench', ini, '-o', tmp_path / 'out', '--force')
    assert forced.exit_code == 0, forced.output
    assert len(list((tmp_path / 'out').glob('quench_*/quench/Si/*'))) == 1


def test_run_quench_for_a_satisfied_friction_config_writes_nothing(tmp_path, no_lammps):
    result = _invoke('run', 'quench', '--for', REPO / 'examples' / 'afm_config.ini',
                     '-o', tmp_path / 'out')
    assert result.exit_code == 0, result.output
    assert 'all amorphous blocks present' in result.output
    assert not (tmp_path / 'out').exists() or not any((tmp_path / 'out').iterdir())


def test_run_quench_for_turns_each_miss_into_a_member(tmp_path, no_lammps, monkeypatch):
    monkeypatch.chdir(REPO)
    settings = _settings_file(tmp_path, melt_temp=3000, quench_slab_dims=[12, 12, 12])
    result = _invoke('run', 'quench', '--for', 'examples/afm_config.ini',
                     '--settings-file', settings, '-o', tmp_path / 'out')
    assert result.exit_code == 0, result.output
    members = list((tmp_path / 'out').glob('quench_*/quench/Si/*'))
    assert len(members) == 1
    prov = json.loads((members[0] / 'provenance.json').read_text(encoding='utf-8'))
    # slab = axis-wise max(quench_slab_dims, needed size): the 100x100 sheet, 12 A thick.
    assert prov['slab_requested'] == [100.0, 100.0, 12.0]
    assert prov['melt_temp'] == 3000


def test_run_never_calls_lammps_without_opt_in(tmp_path, no_lammps, monkeypatch):
    """A friction run with no matching block aborts with the command, before writing."""
    monkeypatch.chdir(REPO)
    settings = _settings_file(tmp_path, melt_temp=3000)
    result = _invoke('run', 'afm', 'examples/afm_config.ini', '--settings-file', settings,
                     '-o', tmp_path / 'out')
    assert result.exit_code == 1
    assert 'No amorphous Si block matches this run.' in result.output
    assert 'FrictionSim2D run quench --for examples/afm_config.ini' in result.output
    assert f'--settings-file {settings}' in result.output
    assert not (tmp_path / 'out').exists()
    assert no_lammps == []


def _fake_lmp(deck: str, log: str, cwd: Path) -> None:
    """What `lmp -in quench.in -log run_quench.log` leaves behind."""
    text = (Path(cwd) / deck).read_text(encoding='utf-8')
    out = next(line.split()[1] for line in text.splitlines() if line.startswith('write_data'))
    src = next(line.split()[1] for line in text.splitlines() if line.startswith('read_data'))
    (Path(cwd) / out).write_bytes((Path(cwd) / src).read_bytes())
    (Path(cwd) / log).write_text('LAMMPS\nTotal wall time: 0:00:01\n', encoding='utf-8')


def test_quench_local_opt_in_runs_and_registers(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(amorphous, 'run_lmp_serial',
                        lambda deck, log, cwd: (calls.append(deck), _fake_lmp(deck, log, cwd)))
    result = _invoke('run', 'quench', _quench_ini(tmp_path), '-o', tmp_path / 'out', '--local')
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0].endswith('lammps/quench.in')
    key = QuenchKey.from_settings('Si', SI_POT, QuenchSettings(melt_temp=3000))
    found = find_block(key, (10.0, 10.0, 10.0))
    assert found is not None
    meta = json.loads(Path(str(found) + '.json').read_text(encoding='utf-8'))
    assert meta['source'] == 'local'


def test_quench_import_cli_reports_incomplete_and_exits_1(tmp_path, no_lammps):
    _invoke('run', 'quench', _quench_ini(tmp_path), '-o', tmp_path / 'out')
    root = next((tmp_path / 'out').glob('quench_*'))
    result = _invoke('quench', 'import', root)
    assert result.exit_code == 1
    assert 'incomplete' in result.output


def test_quench_import_cli_imports_a_finished_root(tmp_path, no_lammps):
    _invoke('run', 'quench', _quench_ini(tmp_path), '-o', tmp_path / 'out')
    root = next((tmp_path / 'out').glob('quench_*'))
    member = next(root.glob('quench/Si/*'))
    rel = member.relative_to(root).as_posix()
    _fake_lmp(f'{rel}/lammps/quench.in', f'{rel}/lammps/run_quench.log', root)
    result = _invoke('quench', 'import', root)
    assert result.exit_code == 0, result.output
    assert (amorphous.amorphous_library_dir() / f'amor_Si__{member.name}.lmp').is_file()


def test_quench_list_shows_the_shipped_block(tmp_path):
    result = _invoke('quench', 'list', 'Si', '--json')
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert any(r['root'] == 'packaged' and r['key8'] for r in rows)


def test_removed_quench_keys_load_with_a_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        settings = QuenchSettings.model_validate(
            {'run_local': True, 'n_procs': 4, 'allow_generate': False})
    assert settings.melt_steps == 20000 and settings.equilibrate_steps == 10000
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert not hasattr(settings, 'run_local')
