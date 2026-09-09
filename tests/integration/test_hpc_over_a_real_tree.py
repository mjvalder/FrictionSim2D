"""HPC generation over a tree a builder actually wrote.

Spec Phase 3, item 2. ``tests/hpc/test_hpc_scripts.py`` drives the generator
directly with a synthetic ``["mat0/L1"]`` path list, and
``tests/run/test_hpc_generation_reporting.py`` checks the ``hpc.modules``
blocker and that *something* was written. Neither looks at what the scripts
point at, so a manifest that names a file the builder never wrote, or an array
range that does not match the manifest it indexes, passes both today.

Three real trees cover all three dispatch shapes ``dispatch_hpc_generation``
chooses between, and each runs for PBS and Slurm:

    afm_config.ini        3 system : 3 slide, 1:1  -> combined  (run.pbs/run.sh)
    indent_config.ini     2 system : 10 slide      -> two-phase (run_system, run_slide)
    pes_sheet_config.ini  slide only               -> array     (run.pbs/run.sh)

Generation writes into ``<simulation root>/hpc``, so each (config, scheduler)
pair works on its own copy of the built tree; the builds themselves stay shared
through the session-scoped ``build_example`` fixture.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from src.core.config import GlobalSettings
from src.run.hpc_dispatch import generate_hpc_scripts_for_root

pytestmark = pytest.mark.integration

#: (config, model, dispatch shape the tree's system:slide ratio selects)
TREES: List[Tuple[str, str, str]] = [
    ('afm_config.ini', 'afm', 'combined'),
    ('indent_config.ini', 'indent', 'two-phase'),
    ('pes_sheet_config.ini', 'pes_sheet', 'array'),
]
SCHEDULERS = ['pbs', 'slurm']

CASES = [(name, model, shape, scheduler)
         for name, model, shape in TREES for scheduler in SCHEDULERS]
IDS = [f"{name}-{scheduler}" for name, _m, _s, scheduler in CASES]

_ARRAY_RANGE_RE = re.compile(r'^#(?:PBS -J|SBATCH --array=)\s*1-(\d+)\s*$', re.M)
_MANIFEST_REF_RE = re.compile(r'\$(?:PBS_O_WORKDIR|SLURM_SUBMIT_DIR)/(hpc/\S+\.txt)')
_PHASE_RE = re.compile(r'-in\s+\$\{LAMMPS_SIM_PATH\}/(\S+\.in)')


@pytest.fixture(scope='module', name='hpc_tree')
def _hpc_tree(build_example, tmp_path_factory):
    """Return ``hpc_tree(name, model, scheduler) -> simulation root``.

    The tree is copied before generation so the two schedulers cannot overwrite
    each other's ``hpc/`` directory, and memoised so six cases cost three builds
    and six copies.
    """
    cache: Dict[Tuple[str, str], Path] = {}

    def hpc_tree(name: str, model: str, scheduler: str) -> Path:
        key = (name, scheduler)
        if key not in cache:
            source = Path(build_example(name, model).simulation_root)
            root = tmp_path_factory.mktemp(f"{model}_{scheduler}") / source.name
            shutil.copytree(source, root)
            settings = GlobalSettings()
            settings.hpc.modules = ['lammps']
            settings.hpc.scheduler_type = scheduler
            reason = generate_hpc_scripts_for_root(root, settings)
            assert reason is None, reason
            cache[key] = root
        return cache[key]

    return hpc_tree


def _manifests(root: Path) -> List[Path]:
    return sorted((root / 'hpc').glob('manifest*.txt'))


def _entries(manifest: Path) -> List[str]:
    return [line for line in manifest.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _job_scripts(root: Path) -> List[Path]:
    """Every scheduler script in ``hpc/``, both schedulers' extensions."""
    return sorted(p for p in (root / 'hpc').iterdir()
                  if p.suffix in ('.pbs', '.sh') and p.name != 'submit_jobs.sh')


def _built_inputs(root: Path) -> set:
    """Every LAMMPS input script the builder wrote, root-relative."""
    return {str(p.relative_to(root)) for p in root.rglob('lammps/*.in')}


def _manifest_entries_are_scripts(entries: List[str]) -> bool:
    """Whether a manifest lists scripts or simulation directories.

    The two-phase and array shapes list one *script* per line and the job runs
    exactly it. The combined shape lists one *simulation directory* per line and
    the job appends each phase itself -- so the two cannot be compared to the
    built tree the same way.
    """
    return all(entry.endswith('.in') for entry in entries)


def _scripts_a_job_will_run(root: Path) -> set:
    """Every script the generated jobs will actually run, root-relative.

    For a combined manifest the phase suffixes are read back out of the job
    script (``-in ${LAMMPS_SIM_PATH}/lammps/system.in``) rather than assumed, so
    this tracks the template instead of restating it.
    """
    will_run = set()
    for script in _job_scripts(root):
        text = script.read_text(encoding='utf-8')
        phases = _PHASE_RE.findall(text)
        for ref in _MANIFEST_REF_RE.findall(text):
            entries = _entries(root / ref)
            if _manifest_entries_are_scripts(entries):
                will_run.update(entries)
                continue
            assert phases, (
                f"{script.name}: {ref} lists simulation directories but the "
                f"script names no phase to run in them")
            will_run.update(f"{entry}/{phase}"
                            for entry in entries for phase in phases)
    return will_run


# --------------------------------------------------------------------------

@pytest.mark.parametrize('name,model,shape,scheduler', CASES, ids=IDS)
def test_the_manifest_names_only_files_the_builder_wrote(
        name, model, shape, scheduler, hpc_tree):
    root = hpc_tree(name, model, scheduler)
    manifests = _manifests(root)
    assert manifests, sorted(p.name for p in (root / 'hpc').iterdir())
    for manifest in manifests:
        entries = _entries(manifest)
        is_script = _manifest_entries_are_scripts(entries)
        for entry in entries:
            target = root / entry
            assert (target.is_file() if is_script else target.is_dir()), (
                f"{manifest.name} names '{entry}', which the builder did not "
                f"write under {root}")


@pytest.mark.parametrize('name,model,shape,scheduler', CASES, ids=IDS)
def test_the_manifests_between_them_name_every_script_the_builder_wrote(
        name, model, shape, scheduler, hpc_tree):
    """A dropped simulation directory is invisible otherwise: the job array
    still submits, it just never runs that member."""
    root = hpc_tree(name, model, scheduler)
    listed = _scripts_a_job_will_run(root)
    built = _built_inputs(root)
    assert listed == built, {
        'in the tree but no job runs it': sorted(built - listed),
        'a job runs it but the builder did not write it': sorted(listed - built),
    }


@pytest.mark.parametrize('name,model,shape,scheduler', CASES, ids=IDS)
def test_every_simulation_directory_appears_in_a_manifest(
        name, model, shape, scheduler, hpc_tree):
    root = hpc_tree(name, model, scheduler)
    listed = {str(Path(entry).parent.parent)
              for entry in _scripts_a_job_will_run(root)}
    built = {str(p.parent.relative_to(root))
             for p in root.rglob('lammps') if p.is_dir()}
    assert listed == built, {'manifested': sorted(listed),
                             'built': sorted(built)}


@pytest.mark.parametrize('name,model,shape,scheduler', CASES, ids=IDS)
def test_the_array_range_matches_the_manifest_it_indexes(
        name, model, shape, scheduler, hpc_tree):
    """The script picks its task with ``sed -n "${TASK_ID}p" <manifest>``, so a
    range longer than the manifest runs empty tasks and a shorter one silently
    drops the tail. Nothing else pins the two together."""
    root = hpc_tree(name, model, scheduler)
    scripts = _job_scripts(root)
    assert scripts, sorted(p.name for p in (root / 'hpc').iterdir())
    for script in scripts:
        text = script.read_text(encoding='utf-8')
        referenced = _MANIFEST_REF_RE.findall(text)
        assert referenced, f"{script.name}: indexes no manifest"
        manifest = root / referenced[0]
        assert manifest.is_file(), f"{script.name}: {referenced[0]} is missing"

        declared = _ARRAY_RANGE_RE.search(text)
        n_entries = len(_entries(manifest))
        if declared is None:
            assert n_entries == 1, (
                f"{script.name}: no array range, but {manifest.name} has "
                f"{n_entries} entries")
            continue
        assert int(declared.group(1)) == n_entries, (
            f"{script.name}: array range 1-{declared.group(1)} against "
            f"{manifest.name}'s {n_entries} entries")


@pytest.mark.parametrize('name,model,shape,scheduler', CASES, ids=IDS)
def test_the_dispatch_shape_matches_the_trees_system_to_slide_ratio(
        name, model, shape, scheduler, hpc_tree):
    """`dispatch_hpc_generation` picks combined / two-phase / array from the
    manifest's job counts. Which one it picked is only visible in the file names
    it wrote, and getting it wrong means a slide job that runs before the
    system.in that writes the data file it reads."""
    root = hpc_tree(name, model, scheduler)
    written = {p.name for p in (root / 'hpc').iterdir()}
    ext = '.pbs' if scheduler == 'pbs' else '.sh'
    if shape == 'two-phase':
        assert {f'run_system{ext}', f'run_slide{ext}'} <= written, written
    else:
        assert f'run{ext}' in written, written
        assert f'run_system{ext}' not in written, written
    combined = 'manifest_combined.txt' in written
    assert combined == (shape == 'combined'), written
