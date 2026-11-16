"""`aiida submit` / `run_with_aiida` end to end, dispatch stubbed (spec §4.3).

The final review's C1: the local mode could not store results at all -- no
profile was loaded, runs were looked for only one level below the argument,
`root_dir` was the run directory (so no run node was ever found), and
`run_with_aiida` never registered the fresh tree. These tests drive the real
submission code down to `_dispatch`, which is replaced by a recorder so no
LAMMPS and no broker are needed.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('aiida')

from click.testing import CliRunner  # noqa: E402

import src.aiida as aiida_pkg  # noqa: E402
import src.aiida.submit as submit_mod  # noqa: E402
from src.cli import cli  # noqa: E402
from tests.helpers.synthetic_results import make_run  # noqa: E402

pytestmark = pytest.mark.aiida

REL_A = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1'
REL_B = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'


def _code(tmp_path):
    from aiida import orm

    computer = orm.Computer(label='localhost-test', hostname='localhost',
                            transport_type='core.local', scheduler_type='core.direct',
                            workdir=str(tmp_path / 'work'))
    computer.store()
    computer.configure()
    code = orm.InstalledCode(computer=computer, filepath_executable='/bin/true',
                             label='lammps',
                             default_calc_job_plugin='friction2d.lammps')
    code.store()
    return code


def _record_dispatch(monkeypatch, *, ok=True):
    seen = []

    def _fake(builder, daemon):
        seen.append(builder)
        return SimpleNamespace(pk=1000 + len(seen), is_finished_ok=ok,
                               exit_status=0 if ok else 310, exit_message='')

    monkeypatch.setattr(submit_mod, '_dispatch', _fake)
    return seen


def _record_profile(monkeypatch):
    loaded = []
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile',
                        lambda profile=None: loaded.append(profile))
    return loaded


def _ingested_tree(tmp_path):
    from src.aiida.ingest import ingest_root

    root = tmp_path / 'simulation_x'
    make_run(root, REL_A, 'afm')
    make_run(root, REL_B, 'afm')
    report = ingest_root(root, label='flow')
    assert report.ok(), report.render()
    return root


def _run_uuids(label='flow'):
    from aiida.orm import QueryBuilder

    from src.aiida.data import FrictionSimulationData

    qb = QueryBuilder()
    qb.append(FrictionSimulationData, filters={'attributes.set_label': label})
    return {n.rel_path: str(n.uuid) for n in qb.all(flat=True)}


@pytest.mark.skip(reason='removed in Task 11')
def test_cli_submit_on_a_root_tracks_every_run(aiida_profile_clean, tmp_path,
                                               monkeypatch):
    root = _ingested_tree(tmp_path)
    code = _code(tmp_path)
    seen = _record_dispatch(monkeypatch)
    loaded = _record_profile(monkeypatch)

    result = CliRunner().invoke(cli, [
        'aiida', 'submit', str(root), '--profile', 'myprof',
        '--code', code.full_label], input='y\n')

    assert result.exit_code == 0, result.output
    assert loaded == ['myprof']
    assert len(seen) == 2
    uuids = _run_uuids()
    assert {str(b.simulation_node.uuid) for b in seen} == set(uuids.values())
    assert 'Ran 2 job(s): 2 finished ok, 0 failed' in result.output
    assert 'Submitted' not in result.output


@pytest.mark.skip(reason='removed in Task 11')
def test_cli_submit_on_a_run_dir_finds_its_root(aiida_profile_clean, tmp_path,
                                               monkeypatch):
    root = _ingested_tree(tmp_path)
    code = _code(tmp_path)
    seen = _record_dispatch(monkeypatch)
    _record_profile(monkeypatch)

    result = CliRunner().invoke(cli, [
        'aiida', 'submit', str(root / REL_B), '--code', code.full_label],
        input='y\n')

    assert result.exit_code == 0, result.output
    assert len(seen) == 1
    assert str(seen[0].simulation_node.uuid) == _run_uuids()[REL_B]


@pytest.mark.skip(reason='removed in Task 11')
def test_cli_submit_reports_failed_in_process_runs(aiida_profile_clean, tmp_path,
                                                  monkeypatch):
    root = _ingested_tree(tmp_path)
    code = _code(tmp_path)
    _record_dispatch(monkeypatch, ok=False)
    _record_profile(monkeypatch)

    result = CliRunner().invoke(cli, [
        'aiida', 'submit', str(root), '--code', code.full_label], input='y\n')

    assert result.exit_code == 1
    assert '0 finished ok, 2 failed' in result.output
    assert 'exit status 310' in result.output


@pytest.mark.skip(reason='removed in Task 11')
def test_cli_submit_with_every_submission_failing_is_a_clean_error(
        aiida_profile_clean, tmp_path, monkeypatch):
    """`pks[0]` used to raise IndexError when submit_batch returned []."""
    root = _ingested_tree(tmp_path)
    code = _code(tmp_path)
    _record_profile(monkeypatch)

    def _boom(builder, daemon):
        raise OSError('synthetic launch failure')

    monkeypatch.setattr(submit_mod, '_dispatch', _boom)

    result = CliRunner().invoke(cli, [
        'aiida', 'submit', str(root), '--code', code.full_label], input='y\n')

    assert result.exit_code != 0
    assert 'None of the 2 run(s) could be submitted' in result.output
    assert not isinstance(result.exception, IndexError)


def test_cli_submit_profile_failure_is_a_clean_error(aiida_profile_clean,
                                                     tmp_path, monkeypatch):
    root = _ingested_tree(tmp_path)

    def _bad(profile=None):
        raise RuntimeError(f'no such profile {profile}')

    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', _bad)

    result = CliRunner().invoke(cli, ['aiida', 'submit', str(root),
                                      '--profile', 'ghost'])

    assert result.exit_code != 0
    assert 'Could not load AiiDA profile ghost' in result.output
    assert 'Traceback' not in result.output


def test_find_simulation_root_rejects_a_path_outside_any_tree(tmp_path):
    with pytest.raises(ValueError, match='not inside a simulation tree'):
        submit_mod.find_simulation_root(tmp_path)


@pytest.mark.skip(reason='removed in Task 11')
def test_run_with_aiida_registers_the_tree_before_submitting(
        aiida_profile_clean, tmp_path, monkeypatch):
    def _fake_run_simulations(config_file, model, **kwargs):
        root = Path(kwargs['output_root']) / 'simulation_fresh'
        run_dir = make_run(root, REL_B, model)
        for path in (run_dir / 'results').glob('*'):
            path.unlink()
        return ([run_dir.parent], root, [config_file], None)

    monkeypatch.setattr(submit_mod, 'run_simulations', _fake_run_simulations)
    code = _code(tmp_path)
    seen = _record_dispatch(monkeypatch)

    _dirs, root, processes = submit_mod.run_with_aiida(
        'examples/afm_config.ini', model='afm', output_root=tmp_path,
        code_label=code.full_label, label='fresh')

    assert len(processes) == 1
    assert root.name == 'simulation_fresh'
    uuids = _run_uuids('fresh')
    assert list(uuids) == [REL_B]
    assert str(seen[0].simulation_node.uuid) == uuids[REL_B]
