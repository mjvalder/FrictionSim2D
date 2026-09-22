"""ADDED SCOPE (Task 23 controller ruling): a per-layer AFM run's own
``lammps/*.in`` scripts read files that live one level ABOVE the run's own
directory -- ``build/`` and ``provenance/potentials/`` are shared across every
layer count and sit at the run's parent (the ``K<temp>`` directory), while
``results/``, ``data/``, ``visuals/`` and ``provenance.json`` sit at the run's
own directory (see ``AFMSimulation.build``/``write_inputs`` and
``run/tree.py``'s ``provenance_dir_for``). ``read_data``/``include``/
``pair_coeff`` lines in the rendered scripts spell every one of those paths
relative to the simulation ROOT (``relative_run_dir``, not
``relative_run_dir_layer`` -- ``builders/simulation_base.py``'s
``_run_dir_path`` docstring), so the CalcJob's staged remote workdir must
reproduce all of them at that same root-relative path, not just the run's own
leaf.

Uses a REAL built AFM example (``tests.helpers.examples.build``), not a
synthetic tree: the bug is about the actual generated directory shape, and a
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


def test_afm_calcjob_staging_includes_everything_its_scripts_read(
        aiida_profile_clean, tmp_path):
    from aiida.common.folders import SandboxFolder

    from src.aiida.calcjob import (  # pylint: disable=import-outside-toplevel
        _SHARED_SUBDIRS_ATTR,
        _sim_prefix_for,
        _stage_files_into_sandbox,
        prepare_simulation_folder,
    )
    from src.run.tree import discover_runs  # pylint: disable=import-outside-toplevel

    root = tmp_path / 'sim'
    result = build('afm_config.ini', 'afm', root)
    assert result.ok, result.failures

    locators = list(discover_runs(result.simulation_root))
    assert len(locators) >= 2, "afm_config.ini should build more than one layer"

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
        sim_node = SimpleNamespace(rel_path=rel_path)

        folder_data = prepare_simulation_folder(run_dir)
        folder_data.store()
        repo = folder_data.base.repository
        sim_prefix = _sim_prefix_for({'simulation_node': sim_node}, repo, None)
        assert sim_prefix == rel_path

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
