"""Focused tests for script selection helpers in src.aiida.calcjob."""

from types import SimpleNamespace

from src.aiida.calcjob import _collect_lammps_scripts


class _FakeRepository:
    def __init__(self, names):
        self._names = names

    def list_object_names(self, path='.'):
        if path == 'lammps':
            return list(self._names)
        return []


def _make_sim_folder(names):
    repo = _FakeRepository(names)
    return SimpleNamespace(base=SimpleNamespace(repository=repo))


def test_collect_lammps_scripts_string_override_kept_as_single_script():
    """A string override should be treated as one script, not split into chars."""
    sim_folder = _make_sim_folder(['system.in'])

    scripts = _collect_lammps_scripts(sim_folder, 'slide.in')

    assert scripts == ['slide.in']


def test_collect_lammps_scripts_orders_discovered_scripts():
    """Discovery should prioritize system, then slide*, then other .in files."""
    sim_folder = _make_sim_folder(['misc.in', 'slide_p2.in', 'system.in', 'slide.in'])

    scripts = _collect_lammps_scripts(sim_folder, None)

    assert scripts == ['system.in', 'slide.in', 'slide_p2.in', 'misc.in']
