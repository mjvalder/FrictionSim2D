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


def test_get_code_falls_back_to_the_settings_label(aiida_profile_clean, monkeypatch,
                                                   tmp_path):
    settings_file = tmp_path / 'settings.yaml'
    settings_file.write_text('aiida:\n  lammps_code_label: lammps@localhost\n',
                             encoding='utf-8')
    seen = {}
    # Not `seen.setdefault('label', label) or 'CODE'`: that returns `label`
    # itself whenever `label` is truthy (which it always is here), so the
    # test would pass even if `_get_code` fed the wrong string into
    # `load_code` -- it only ever inspects what `seen` recorded, never what
    # `_get_code` actually returned.
    monkeypatch.setattr('aiida.orm.load_code',
                        lambda label: (seen.update(label=label), 'CODE')[1])

    code = submit_mod._get_code(None, settings_file=settings_file)

    assert seen['label'] == 'lammps@localhost'
    assert code == 'CODE'


def test_get_code_falls_through_to_discovery_when_settings_label_is_stale(
        aiida_profile_clean, tmp_path):
    """Fix round 1, finding 1: `AiidaSettings.lammps_code_label` always has a
    value (it defaults to `'lammps@my_hpc'`, not `None`) -- unlike an
    explicit `--code`, it is not authoritative. Treating it as authoritative
    made the `%lammps%` discovery scan below unreachable for anyone who had
    not hand-edited settings.yaml to match whatever `aiida setup` actually
    registered, so `aiida submit` (no `--code`) failed right after `aiida
    setup` with "Code 'lammps@my_hpc' not found" even though a real LAMMPS
    code was registered under a different label.
    """
    from aiida import orm

    settings_file = tmp_path / 'settings.yaml'
    settings_file.write_text('aiida:\n  lammps_code_label: lammps@nowhere\n',
                             encoding='utf-8')

    computer = orm.Computer(label='localhost-test', hostname='localhost',
                            transport_type='core.local', scheduler_type='core.direct',
                            workdir=str(tmp_path))
    computer.store()
    computer.configure()
    code = orm.InstalledCode(computer=computer, filepath_executable='/bin/true',
                             label='lammps',
                             default_calc_job_plugin='core.arithmetic.add')
    code.store()

    result = submit_mod._get_code(None, settings_file=settings_file)

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
