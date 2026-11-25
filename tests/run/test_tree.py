"""discover_runs walks a simulation root once and types every run (spec §3 F2)."""
import json

from frictionsim2d.run.tree import RunLocator, discover_runs, unparseable_runs


def _write_run(root, rel, family, prov_at_parent=False):
    run_dir = root / rel
    (run_dir / 'results').mkdir(parents=True, exist_ok=True)
    (run_dir / 'lammps').mkdir(parents=True, exist_ok=True)
    prov_parent = run_dir.parent if prov_at_parent else run_dir
    (prov_parent / 'provenance').mkdir(parents=True, exist_ok=True)
    (prov_parent / 'provenance' / 'config.json').write_text('{}', encoding='utf-8')
    (run_dir / 'provenance.json').write_text(
        json.dumps({'schema_version': '2.0.0', 'family': family,
                    'model': rel.split('/')[0]}), encoding='utf-8')
    return run_dir


def test_finds_every_run_sorted_by_rel_path(tmp_path):
    _write_run(tmp_path, 'afm/h-MoS2/100x_100y/sub_amorph-Si_tip_Si_r25/K300/L2',
               'afm', prov_at_parent=True)
    _write_run(tmp_path, 'sheetonsheet/h-MoS2+h-WS2/100x_100y/K300', 'hetero')

    found = discover_runs(tmp_path)

    assert [loc.tokens.rel_path for loc in found] == [
        'afm/h-MoS2/100x_100y/sub_amorph-Si_tip_Si_r25/K300/L2',
        'sheetonsheet/h-MoS2+h-WS2/100x_100y/K300',
    ]
    assert all(isinstance(loc, RunLocator) for loc in found)


def test_family_comes_from_provenance_not_the_path(tmp_path):
    _write_run(tmp_path, 'sheetonsheet/h-MoS2+h-WS2/100x_100y/K300', 'hetero')

    loc, = discover_runs(tmp_path)

    assert loc.tokens.model == 'sheetonsheet'
    assert loc.tokens.family == 'hetero'


def test_afm_provenance_dir_resolves_one_level_up(tmp_path):
    run_dir = _write_run(
        tmp_path, 'afm/h-MoS2/100x_100y/sub_amorph-Si_tip_Si_r25/K300/L2',
        'afm', prov_at_parent=True)

    loc, = discover_runs(tmp_path)

    assert loc.run_dir == run_dir
    assert loc.provenance_file == run_dir / 'provenance.json'
    assert loc.provenance_dir == run_dir.parent / 'provenance'
    assert loc.results_dir == run_dir / 'results'
    assert loc.data_dir == run_dir / 'data'
    assert loc.lammps_dir == run_dir / 'lammps'


def test_directories_outside_a_model_top_level_are_ignored(tmp_path):
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs' / 'provenance.json').write_text('{}', encoding='utf-8')

    assert discover_runs(tmp_path) == []


def test_an_unparseable_run_is_skipped_not_fatal(tmp_path):
    bad = tmp_path / 'afm' / 'h-MoS2' / 'nonsense'
    bad.mkdir(parents=True)
    (bad / 'provenance.json').write_text('{}', encoding='utf-8')
    _write_run(tmp_path, 'sheetonsheet/h-MoS2/100x_100y/K300', 'sheetonsheet')

    found = discover_runs(tmp_path)

    assert [loc.tokens.rel_path for loc in found] == [
        'sheetonsheet/h-MoS2/100x_100y/K300']


def test_unparseable_runs_reports_exactly_what_discover_runs_skipped(tmp_path):
    bad = tmp_path / 'afm' / 'h-MoS2' / 'nonsense'
    bad.mkdir(parents=True)
    (bad / 'provenance.json').write_text('{}', encoding='utf-8')
    _write_run(tmp_path, 'sheetonsheet/h-MoS2/100x_100y/K300', 'sheetonsheet')

    assert unparseable_runs(tmp_path) == ['afm/h-MoS2/nonsense']


def test_runs_are_sorted_by_path_not_by_model_walk_order(tmp_path):
    """MODELS walks sheetonsheet before pes_sheet; the result is still ordered by path."""
    _write_run(tmp_path, 'sheetonsheet/h-MoS2/100x_100y/K300', 'sheetonsheet')
    _write_run(tmp_path, 'pes_sheet/h-MoS2/100x_100y/K300', 'pes_sheet')

    assert [loc.tokens.rel_path for loc in discover_runs(tmp_path)] == [
        'pes_sheet/h-MoS2/100x_100y/K300', 'sheetonsheet/h-MoS2/100x_100y/K300']


def test_a_corrupt_provenance_json_keeps_the_family_from_the_path(tmp_path):
    run_dir = _write_run(tmp_path, 'sheetonsheet/h-MoS2/100x_100y/K300', 'hetero')
    (run_dir / 'provenance.json').write_text('{not json', encoding='utf-8')

    loc, = discover_runs(tmp_path)

    assert loc.tokens.family == 'sheetonsheet'


def test_a_run_with_its_own_provenance_dir_does_not_borrow_the_parents(tmp_path):
    run_dir = _write_run(tmp_path, 'sheetonsheet/h-MoS2/100x_100y/K300', 'sheetonsheet')

    loc, = discover_runs(tmp_path)

    assert loc.provenance_dir == run_dir / 'provenance'
