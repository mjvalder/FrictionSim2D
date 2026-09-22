"""ADDED SCOPE (Task 23 controller ruling) + fix round 1 finding 2.

A per-layer AFM run's own ``lammps/*.in`` scripts read files that live one
level ABOVE the run's own directory -- ``build/`` and ``provenance/potentials/``
are shared across every layer count and sit at the run's parent (the
``K<temp>`` directory), while ``results/``, ``data/``, ``visuals/`` and
``provenance.json`` sit at the run's own directory (see
``AFMSimulation.build``/``write_inputs`` and ``run/tree.py``'s
``provenance_dir_for``). ``read_data``/``include``/``pair_coeff`` lines in the
rendered scripts spell every one of those paths relative to the simulation
ROOT (``relative_run_dir``, not ``relative_run_dir_layer`` --
``builders/simulation_base.py``'s ``_run_dir_path`` docstring), so the
CalcJob's staged remote workdir must reproduce all of them at that same
root-relative path, not just the run's own leaf.

Fix round 1 finding 2: this only worked when a ``simulation_node`` input was
given (``simulation_node.rel_path`` is the run's own full sim-root-relative
path, authoritative in ``_sim_prefix_for``). ``smart_submit``/``aiida submit``
(``submit_batch(config_path=None)``) never sets one, so every test below runs
twice -- with and without ``simulation_node`` -- over several real AFM-family
examples (plain AFM, a flake AFM, and indent, which additionally has more
than one layer count).

Uses REAL built examples (``tests.helpers.examples.build``), not a synthetic
tree: the bug is about the actual generated directory shape, and a
hand-rolled fixture could accidentally match whatever the staging code
happens to do.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('aiida')

pytestmark = [pytest.mark.integration, pytest.mark.aiida]

from tests.helpers.examples import build  # noqa: E402

_REF_LINE = re.compile(r'^\s*(read_data|include|read_restart)\s+(\S+)', re.IGNORECASE)

#: AFM-family examples that nest a shared K-level `build/`/`provenance/`
#: above one or more per-layer run directories -- the shape this whole file
#: is about. `afm_freestanding_pes_config.ini` (family `pes_tip`) has the
#: same nesting too but is left out here to keep the matrix focused; nothing
#: in the fix is family-specific (it keys off the directory SHAPE, not the
#: family name).
_CONFIGS = [
    ('afm_config.ini', 'afm'),
    ('afm_flake_config.ini', 'afm'),
    ('indent_config.ini', 'indent'),
]


def _referenced_paths(text: str) -> set:
    """Every sim-root-relative, statically-checkable path a LAMMPS script reads.

    ``read_data``/``include``/``read_restart`` always take the path as their
    first argument -- except when it is a LAMMPS variable expression (``
    load_$(v_find)N.data``, resolved and written by the slide loop itself at
    run time, not a static input this staging fix is about). ``pair_coeff``
    potential-file arguments have no fixed position (it depends on the pair
    style), so every token containing ``/`` is taken, except a bare pair-style
    name like ``lj/cut`` -- one slash and no directory nesting, unlike every
    real path this codebase writes (at least ``family/material/...``, several
    slashes deep).
    """
    paths = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        match = _REF_LINE.match(stripped)
        if match:
            candidate = match.group(2)
            if '$' not in candidate:
                paths.add(candidate)
            continue
        if stripped.lower().startswith('pair_coeff'):
            paths.update(tok for tok in stripped.split()
                        if tok.count('/') >= 2 and '$' not in tok)
    return paths


@pytest.mark.parametrize('config_name,model', _CONFIGS)
@pytest.mark.parametrize('with_simulation_node', [True, False],
                        ids=['with_simulation_node', 'without_simulation_node'])
def test_afm_calcjob_staging_includes_everything_its_scripts_read(
        aiida_profile_clean, tmp_path, config_name, model, with_simulation_node):
    from aiida.common.folders import SandboxFolder

    from src.aiida.calcjob import (  # pylint: disable=import-outside-toplevel
        _SHARED_SUBDIRS_ATTR,
        _collect_lammps_scripts,
        _sim_prefix_for,
        _stage_files_into_sandbox,
        prepare_simulation_folder,
    )
    from src.run.tree import discover_runs  # pylint: disable=import-outside-toplevel

    root = tmp_path / 'sim'
    result = build(config_name, model, root)
    assert result.ok, result.failures

    locators = list(discover_runs(result.simulation_root))
    assert locators

    for loc in locators:
        run_dir = loc.run_dir
        referenced = set()
        for script in sorted((run_dir / 'lammps').glob('*.in*')):
            referenced |= _referenced_paths(script.read_text(encoding='utf-8'))
        assert referenced, f"no read_data/include/pair_coeff lines found in {run_dir}"
        # Sanity: this run really does reference something outside its own
        # leaf -- otherwise the test would pass even with the bug.
        assert any('/build/' in p or '/provenance/' in p for p in referenced), referenced

        rel_path = str(run_dir.relative_to(result.simulation_root))

        folder_data = prepare_simulation_folder(run_dir)
        folder_data.store()
        repo = folder_data.base.repository

        inputs = {'simulation_dir': folder_data}
        if with_simulation_node:
            inputs['simulation_node'] = SimpleNamespace(rel_path=rel_path)
            lammps_scripts = None  # never consulted once simulation_node is given
        else:
            lammps_scripts = _collect_lammps_scripts(folder_data, None)

        sim_prefix = _sim_prefix_for(inputs, repo, lammps_scripts)
        assert sim_prefix == rel_path, (
            f"{config_name}/{run_dir.name} "
            f"({'with' if with_simulation_node else 'without'} simulation_node): "
            f"expected prefix {rel_path!r}, got {sim_prefix!r}")

        # `None`, not `()`, as the "missing" default: `NodeAttributes.get`'s
        # own sentinel for "no default given" IS the empty tuple (CPython
        # interns it), so passing `()` here would be silently indistinguishable
        # from omitting the default and raise instead of falling back.
        shared_subdirs = frozenset(
            folder_data.base.attributes.get(_SHARED_SUBDIRS_ATTR, None) or ())

        with SandboxFolder() as folder:
            _stage_files_into_sandbox(folder, repo, sim_prefix, shared_subdirs)
            workdir = Path(folder.abspath)
            missing = [p for p in referenced if not (workdir / p).is_file()]
            assert not missing, (
                f"{run_dir}: staged workdir is missing paths its own scripts "
                f"read: {missing}")


@pytest.mark.parametrize('with_simulation_node', [True, False],
                        ids=['with_simulation_node', 'without_simulation_node'])
def test_afm_calcjob_prepare_for_submission_dry_run(
        aiida_profile_clean, tmp_path, monkeypatch, with_simulation_node):
    """The real `prepare_for_submission` wiring, not just the private staging
    helpers exercised above: a genuine `aiida.engine.run` dry run of
    `LammpsFrictionCalcJob` over the real `afm_config.ini` example, with and
    without `simulation_node`. A dry run calls `presubmit` ->
    `prepare_for_submission` and then a real (local) `upload_calculation`,
    writing the exact remote-workdir tree to a local folder -- no scheduler
    or real remote host needed.
    """
    from aiida import orm
    from aiida.engine import run_get_node

    from src.aiida.calcjob import LammpsFrictionCalcJob, prepare_simulation_folder
    from src.aiida.data import FrictionSimulationData
    from src.run.tree import discover_runs

    # `_perform_dry_run` always writes to `./submit_test/<date>-NNNNN`
    # relative to the CURRENT directory (aiida-core's `SubmitTestFolder`,
    # hardcoded, never cleaned up) -- confine that to this test's tmp_path
    # rather than littering the repo checkout.
    monkeypatch.chdir(tmp_path)

    root = tmp_path / 'sim'
    result = build('afm_config.ini', 'afm', root)
    assert result.ok, result.failures
    loc = next(iter(discover_runs(result.simulation_root)))
    run_dir = loc.run_dir

    referenced = set()
    for script in sorted((run_dir / 'lammps').glob('*.in*')):
        referenced |= _referenced_paths(script.read_text(encoding='utf-8'))
    assert referenced

    computer = orm.Computer(label='dry-run-computer', hostname='localhost',
                            transport_type='core.local', scheduler_type='core.direct',
                            workdir=str(tmp_path / 'workdir'))
    computer.store()
    computer.configure()
    code = orm.InstalledCode(computer=computer, filepath_executable='/bin/echo',
                             default_calc_job_plugin='friction2d.lammps')
    code.store()

    folder_data = prepare_simulation_folder(run_dir)
    folder_data.store()

    inputs = dict(
        code=code, simulation_dir=folder_data,
        metadata={'dry_run': True,
                'options': {'resources': {'num_machines': 1,
                                            'num_mpiprocs_per_machine': 1}}},
    )
    if with_simulation_node:
        sim_node = FrictionSimulationData()
        sim_node.set_from_locator(loc, set_uuid='set-dryrun')
        sim_node.store()
        inputs['simulation_node'] = sim_node

    _result, node = run_get_node(LammpsFrictionCalcJob, **inputs)

    assert getattr(node, 'dry_run_info', None), (
        f"no dry_run_info on the node (process_state={node.process_state})")
    workdir = Path(node.dry_run_info['folder'])
    missing = [p for p in referenced if not (workdir / p).is_file()]
    assert not missing, (
        f"{'with' if with_simulation_node else 'without'} simulation_node: "
        f"dry-run workdir missing {missing}")
