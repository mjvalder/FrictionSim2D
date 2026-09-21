"""The ONE retention rule: ingest and the CalcJob keep exactly the same files."""
from src.aiida.retention import (
    RETAINED_GLOBS, RETRIEVE_LIST, retained_paths, retrieve_patterns)


def _touch(root, *rels):
    for rel in rels:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('x', encoding='utf-8')


def test_keeps_inputs_results_and_state_files(tmp_path):
    _touch(tmp_path,
           'provenance.json',
           'provenance/config.json', 'provenance/potentials/MoS.sw',
           'lammps/system.in', 'lammps/slide.in', 'lammps/system.in.settings',
           'results/friction_f30_a0_s2_layer2.txt',
           'results/raw_f30_a0_s2_layer2.txt',
           'data/hold_f30.txt', 'data/final_f30.data', 'data/load_30N.data',
           'log_slide.lammps', 'job.out')

    kept = {p.relative_to(tmp_path).as_posix() for p in retained_paths(tmp_path)}

    assert kept == {
        'provenance.json', 'provenance/config.json',
        'provenance/potentials/MoS.sw', 'lammps/system.in', 'lammps/slide.in',
        'lammps/system.in.settings',
        'results/friction_f30_a0_s2_layer2.txt',
        'results/raw_f30_a0_s2_layer2.txt',
        'data/hold_f30.txt', 'data/final_f30.data', 'data/load_30N.data',
        'log_slide.lammps', 'job.out',
    }


def test_keeps_hpc_run_logs(tmp_path):
    """R30: HPC job scripts write per-deck logs to lammps/run_<deck>.log."""
    _touch(tmp_path, 'provenance.json', 'lammps/run_slide.log',
           'lammps/run_hold.log')

    kept = {p.relative_to(tmp_path).as_posix() for p in retained_paths(tmp_path)}

    assert kept == {'provenance.json', 'lammps/run_slide.log',
                     'lammps/run_hold.log'}


def test_drops_trajectories_and_build_scratch(tmp_path):
    _touch(tmp_path, 'provenance.json',
           'visuals/slide.lammpstrj', 'build/tip.data', 'data/dump.lammpstrj')

    kept = {p.relative_to(tmp_path).as_posix() for p in retained_paths(tmp_path)}

    assert kept == {'provenance.json'}


def test_result_is_sorted_and_absolute(tmp_path):
    _touch(tmp_path, 'results/b.txt', 'results/a.txt', 'provenance.json')

    kept = retained_paths(tmp_path)

    assert all(p.is_absolute() for p in kept)
    assert kept == sorted(kept)


def _category(pattern: str) -> str:
    """The retention category a glob belongs to: its first path segment."""
    head = pattern.split('/')[0]
    return '.' if '*' in head or head.endswith('.json') else head


def test_the_two_dialects_cover_the_same_categories():
    """NOT `retrieve_patterns() == list(RETAINED_GLOBS)`.

    That comparison is a tautology -- it compares a function to the constant it
    returns and cannot fail. The two lists are deliberately DIFFERENT: local
    globs are `Path.glob` (recursive `**`), remote ones are the transport's
    fnmatch (no recursion, and tuple entries to preserve the tree). What must
    not drift is the set of categories they retain.
    """
    assert {_category(g) for g in RETAINED_GLOBS} == \
           {_category(remote) for remote, _local, _depth in RETRIEVE_LIST}


def test_remote_patterns_are_tuples_that_preserve_the_tree():
    """A plain string is flattened to its basename by execmanager, which puts
    `friction_*.txt` at the top level and leaves no `results/` for the parser."""
    assert all(isinstance(entry, tuple) and len(entry) == 3
               for entry in RETRIEVE_LIST)
    by_remote = {remote: local for remote, local, _d in RETRIEVE_LIST}
    assert by_remote['results/*'] == 'results'
    assert by_remote['provenance.json'] == '.'
    assert by_remote['provenance/cif/*'] == 'provenance/cif'


def test_no_remote_pattern_relies_on_recursive_globbing():
    """The transport's glob is fnmatch-style; `**` matches nothing at depth 1."""
    assert all('**' not in remote for remote, _l, _d in RETRIEVE_LIST)
    remotes = {remote for remote, _l, _d in RETRIEVE_LIST}
    # every provenance level that add_run_files stores is named explicitly
    assert {'provenance/*', 'provenance/cif/*',
            'provenance/potentials/*'} <= remotes


def test_retrieve_patterns_prefixes_only_the_remote_side():
    prefixed = retrieve_patterns('afm/h-MoS2/K300/L2')

    assert prefixed[0] == ('afm/h-MoS2/K300/L2/provenance.json', '.', 0)
    assert ('afm/h-MoS2/K300/L2/results/*', 'results', 0) in prefixed
    assert retrieve_patterns() == list(RETRIEVE_LIST)


def test_retrieve_list_includes_hpc_run_logs():
    """R30: the CalcJob retrieve_list must pull back lammps/run_*.log too."""
    by_remote = {remote: local for remote, local, _d in RETRIEVE_LIST}
    assert by_remote['lammps/run_*.log'] == 'lammps'
