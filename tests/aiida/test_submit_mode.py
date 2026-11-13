"""Submission works with the broker down (spec §4.3)."""
import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

import src.aiida.submit as submit_mod  # noqa: E402


def test_default_is_engine_run_not_the_daemon(aiida_profile_clean, monkeypatch):
    calls = []
    monkeypatch.setattr(submit_mod, '_run_get_node',
                        lambda builder: calls.append('run') or ('res', 'node'))
    monkeypatch.setattr(submit_mod, 'aiida_submit',
                        lambda builder: calls.append('submit'))

    submit_mod._dispatch(builder=object(), daemon=False)

    assert calls == ['run']


def test_daemon_flag_uses_the_broker_path(aiida_profile_clean, monkeypatch):
    calls = []
    monkeypatch.setattr(submit_mod, '_run_get_node',
                        lambda builder: calls.append('run') or ('res', 'node'))
    monkeypatch.setattr(submit_mod, 'aiida_submit',
                        lambda builder: calls.append('submit'))

    submit_mod._dispatch(builder=object(), daemon=True)

    assert calls == ['submit']


def test_array_submission_is_removed():
    """AiiDA-managed array CalcJobs are removed (task-B brief): array jobs on
    an HPC go through `hpc generate` + qsub + `aiida import` instead."""
    assert not hasattr(submit_mod, 'submit_array')
    assert not hasattr(submit_mod, '_build_array_directive')

    from src.aiida import calcjob as calcjob_mod  # pylint: disable=import-outside-toplevel
    assert not hasattr(calcjob_mod, 'prepare_simulation_root')
    assert not hasattr(calcjob_mod, '_handle_array_mode')
    assert not hasattr(calcjob_mod, '_build_array_wrapper')


def test_get_code_discovers_a_lammps_code_without_a_label(
        aiida_profile_clean, tmp_path):
    """With no explicit label, the `%lammps%` discovery scan finds the code."""
    from aiida import orm

    computer = orm.Computer(label='localhost-test', hostname='localhost',
                            transport_type='core.local', scheduler_type='core.direct',
                            workdir=str(tmp_path))
    computer.store()
    computer.configure()
    code = orm.InstalledCode(computer=computer, filepath_executable='/bin/true',
                             label='lammps',
                             default_calc_job_plugin='core.arithmetic.add')
    code.store()

    result = submit_mod._get_code(None)

    assert result.pk == code.pk


def test_get_code_raises_on_an_explicit_missing_label(aiida_profile_clean):
    """Unlike a stale settings label, an explicitly passed one is
    authoritative: it must still raise, not silently fall through."""
    with pytest.raises(ValueError, match="not found"):
        submit_mod._get_code('lammps@nowhere')


def test_full_setup_registers_lammps_at_localhost(monkeypatch):
    import src.aiida.setup as setup_mod

    seen = {}
    monkeypatch.setattr(setup_mod, 'setup_profile', lambda name: True)
    monkeypatch.setattr(setup_mod, 'setup_localhost_computer',
                        lambda label: seen.setdefault('computer', label) or
                        type('C', (), {'label': label})())
    monkeypatch.setattr(setup_mod, 'setup_lammps_code',
                        lambda label, computer, exe:
                        seen.setdefault('code', f'{label}@{computer}'))
    monkeypatch.setattr(setup_mod, 'start_daemon',
                        lambda n: seen.setdefault('started', n) or True)

    out = setup_mod.full_setup()

    assert seen['computer'] == 'localhost'
    assert seen['code'] == 'lammps@localhost'
    assert 'started' not in seen          # no daemon started by default
    assert out['daemon_workers'] == 0
    assert out['daemon'] is False
    assert 'rabbitmq' not in out
