"""`import_root`: a campaign tree -> one CalcJob per deck (plan Task 6).

Runs are keyed on `run_uuid`, decks on `run_uuid|deck_name`; a deck's
fingerprint covers its files and the structures handed to it, so an unchanged
deck is skipped and a changed one supersedes the old CalcJob -- locally, as
extras. Replaces `test_ingest.py` and `test_aiida_round_trip.py`; the pure
`load_points` tests are ported at the bottom.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from frictionsim2d.aiida.ingest import (  # noqa: E402
    ImportReport, LoadPoint, SetRootMismatch, group_label, import_root,
    load_points, supersede,
)
from frictionsim2d.aiida.store import DECK_KEY_EXTRA, DECK_SHA_EXTRA  # noqa: E402
from frictionsim2d.data.provenance import config_hash  # noqa: E402
from frictionsim2d.run.tree import discover_runs  # noqa: E402
from tests.helpers.results_tree import (  # noqa: E402
    make_family_run, slide_config, write_final_data,
)
from tests.helpers.synthetic_results import PES_GRID_CSV, make_run, write_results  # noqa: E402

VERSION = '29 Aug 2024 - Update 1'
LOG = f"""LAMMPS ({VERSION})
Loop time of 12.5 on 4 procs for 1000 steps with 600 atoms
Total wall time: 0:01:05
"""
LOG_NO_VERSION = """Loop time of 12.5 on 4 procs for 1000 steps with 600 atoms
Total wall time: 0:01:05
"""


def _afm_run(root, material, *, loads=(10.0,), logs=True):
    """An AFM run with `system.in` + `slide.in`, `load_*.data` from the system
    deck, a build, results and end snapshots for every load."""
    run_dir = make_family_run(root, 'afm', material=material, loads=loads,
                              config=slide_config('afm', loads=loads))
    lammps = run_dir / 'lammps'
    (lammps / 'slide.in').write_text('# slide\n', encoding='utf-8')
    if logs:
        (lammps / 'run_system.log').write_text(LOG, encoding='utf-8')
        (lammps / 'run_slide.log').write_text(LOG, encoding='utf-8')
    data = run_dir / 'data'
    data.mkdir(exist_ok=True)
    for load in loads:
        (data / f'load_{load:g}N.data').write_text(f'LAMMPS data {load}\n',
                                                   encoding='utf-8')
        write_final_data(run_dir, 'afm', load=load)
    build = run_dir.parent / 'build'
    build.mkdir(exist_ok=True)
    (build / 'sheet.lmp').write_text(f'LAMMPS sheet {material}\n', encoding='utf-8')
    return run_dir


def _tree(root, materials=('h-MoS2', 'h-WS2'), **kwargs):
    for material in materials:
        _afm_run(root, material, **kwargs)
    return root


def _calcjobs():
    return orm.QueryBuilder().append(
        orm.CalcJobNode, filters={'process_type': 'aiida.calculations:friction2d.lammps'}
    ).all(flat=True)


def _live(nodes):
    return [n for n in nodes if not n.base.extras.get('superseded', False)]


def _node_count():
    return orm.QueryBuilder().append(orm.Node).count()


def _group_calcjobs(label):
    group = orm.Group.collection.get(label=group_label(label))
    return {n.pk for n in group.nodes if isinstance(n, orm.CalcJobNode)}


def _set_record(run_dir, **fields):
    path = run_dir / 'provenance.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    record.update(fields)
    path.write_text(json.dumps(record), encoding='utf-8')


def _edit_config(loc, path=None, **general):
    """Rewrite the run's config.json (or ``path``) with ``general`` changed;
    return the new config."""
    source = loc.provenance_dir / 'config.json'
    config = json.loads(source.read_text(encoding='utf-8'))
    config['general'].update(general)
    target = path or source
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config


def _stored_config(rel_material):
    (calc, *_rest) = _by_deck(rel_material).values()
    return json.loads(calc.inputs.provenance.base.repository.get_object_content(
        'provenance/config.json', mode='r'))


def _by_deck(rel_material):
    """The live CalcJobs of the run whose rel_path names `rel_material`."""
    return {n.inputs.parameters['deck_name']: n for n in _live(_calcjobs())
            if rel_material in n.inputs.parameters['rel_path']}


# --- first import, idempotence, identity ---------------------------------------

def test_first_import_creates_one_calcjob_per_deck(aiida_profile_clean, tmp_path):
    root = _tree(tmp_path / 'sim')
    report = import_root(root, label='camp')

    assert isinstance(report, ImportReport)
    assert report.ok(), report.render()
    assert len(report.created_decks) == 4
    assert len(_calcjobs()) == 4
    assert all(n.is_finished_ok for n in _calcjobs())
    assert _group_calcjobs('camp') == {n.pk for n in _calcjobs()}
    for node in _calcjobs():
        params = node.inputs.parameters.get_dict()
        assert node.base.extras.get(DECK_KEY_EXTRA) == \
            f"{params['run_uuid']}|{params['deck_name']}"
        assert len(node.base.extras.get(DECK_SHA_EXTRA)) == 64
    assert len(report.points) == 2
    assert not report.log_missing and not report.no_version
    group = orm.Group.collection.get(label='friction2d/camp')
    extras = group.base.extras.all
    for key in ('model', 'materials', 'source_root', 'source_host', 'git_commit',
                'git_dirty', 'created'):
        assert key in extras, key
    assert extras['source_root'] == str(root.resolve())
    assert extras['model'] == 'afm'
    assert extras['materials'] == ['h-MoS2', 'h-WS2']
    assert report.group_uuid == group.uuid


def test_reimport_skips_every_deck_and_adds_no_node(aiida_profile_clean, tmp_path):
    root = _tree(tmp_path / 'sim')
    import_root(root, label='camp')
    before = _node_count()

    again = import_root(root, label='camp')

    assert again.ok(), again.render()
    assert len(again.skipped_decks) == 4
    assert again.created_decks == [] and again.superseded_decks == []
    assert _node_count() == before


def test_same_root_under_a_second_label_is_already_imported(aiida_profile_clean, tmp_path):
    root = _tree(tmp_path / 'sim')
    import_root(root, label='first')

    second = import_root(root, label='second')

    assert second.ok(), second.render()
    assert len(second.already_imported) == 4
    assert second.created_decks == []
    assert len(_calcjobs()) == 4
    assert _group_calcjobs('first') == _group_calcjobs('second') == \
        {n.pk for n in _calcjobs()}


def test_a_moved_root_is_the_same_campaign(aiida_profile_clean, tmp_path):
    root = _tree(tmp_path / 'sim')
    import_root(root, label='camp')
    moved = tmp_path / 'elsewhere' / 'sim'
    shutil.copytree(root, moved)

    report = import_root(moved, label='camp')

    assert report.ok(), report.render()
    assert len(report.skipped_decks) == 4
    group = orm.Group.collection.get(label='friction2d/camp')
    assert group.base.extras.get('source_root') == str(moved.resolve())


def test_a_different_tree_under_a_used_label_is_refused(aiida_profile_clean, tmp_path):
    import_root(_tree(tmp_path / 'a'), label='camp')
    other = _tree(tmp_path / 'b', materials=('h-MoTe2',))

    with pytest.raises(SetRootMismatch, match='different campaign'):
        import_root(other, label='camp')
    with pytest.raises(SetRootMismatch):
        import_root(other, label='camp', dry_run=True)
    assert len(_calcjobs()) == 4


def test_a_dry_run_creates_nothing_and_classifies(aiida_profile_clean, tmp_path):
    root = _tree(tmp_path / 'sim')
    before = _node_count()
    plan = import_root(root, label='camp', dry_run=True)
    assert plan.dry_run and len(plan.created_decks) == 4
    assert _node_count() == before
    assert not orm.QueryBuilder().append(orm.Group).count()

    import_root(root, label='camp')
    assert len(import_root(root, label='camp', dry_run=True).skipped_decks) == 4


# --- Review Focus 1: fingerprints and supersession -----------------------------

def test_a_new_results_file_supersedes_only_that_deck(aiida_profile_clean, tmp_path):
    """Review Focus 1a."""
    root = _tree(tmp_path / 'sim')
    import_root(root, label='camp')
    old = _by_deck('h-MoS2')['slide']
    old_points = list(old.outputs.points.values())
    old_snapshots = list(old.outputs.snapshots.values())
    other = list(_by_deck('h-WS2').values())
    other_outputs = [n for c in other for n in c.base.links.get_outgoing().all_nodes()]
    (mos2_run,) = [loc.run_dir for loc in discover_runs(root) if 'h-MoS2' in loc.tokens.rel_path]
    write_results(mos2_run, 'afm', loads=(10.0,), tiers=('raw',), layers=2)
    before = {n.pk for n in _calcjobs()}

    report = import_root(root, label='camp')

    assert report.ok(), report.render()
    new = [n for n in _calcjobs() if n.pk not in before]
    assert len(new) == 1 and new[0].inputs.parameters['deck_name'] == 'slide'
    assert len(report.superseded_decks) == 1 and len(report.skipped_decks) == 3
    for node in [old, *old_points, *old_snapshots]:
        assert node.base.extras.get('superseded') is True, node
        assert node.base.extras.get('superseded_at')
    for node in [*other, *other_outputs]:
        assert not node.base.extras.get('superseded', False), node
    assert not _by_deck('h-MoS2')['system'].base.extras.get('superseded', False)
    incoming = [(t.link_label, t.node) for t in old.base.links.get_incoming().all()]
    assert any(label.startswith('blocks') for label, _n in incoming)
    for label, node in incoming:     # blocks, provenance, schema, derived, Code, ...
        assert not node.base.extras.get('superseded', False), label
    assert 'code' in old.inputs and not old.inputs.code.base.extras.get('superseded', False)
    build = _by_deck('h-MoS2')['system'].inputs.structures['build_sheet_lmp']
    assert not build.base.extras.get('superseded', False)
    names = new[0].outputs.points['force10_a0_s2'].base.repository.list_object_names('results')
    assert any(name.startswith('raw_') for name in names)


def test_a_changed_load_structure_reimports_system_and_its_slide(aiida_profile_clean,
                                                                 tmp_path):
    """Review Focus 1b."""
    root = _tree(tmp_path / 'sim')
    import_root(root, label='camp')
    old = _by_deck('h-MoS2')
    (mos2_run,) = [loc.run_dir for loc in discover_runs(root) if 'h-MoS2' in loc.tokens.rel_path]
    load = mos2_run / 'data' / 'load_10N.data'
    load.write_text(load.read_text(encoding='utf-8') + 'x', encoding='utf-8')

    report = import_root(root, label='camp')

    assert report.ok(), report.render()
    assert sorted(report.superseded_decks) == sorted(
        f'{old["system"].inputs.parameters["rel_path"]}:{d}' for d in ('system', 'slide'))
    new = _by_deck('h-MoS2')
    assert {d: n.pk for d, n in new.items()}.keys() == {'system', 'slide'}
    assert all(new[d].pk != old[d].pk for d in ('system', 'slide'))
    assert all(old[d].base.extras.get('superseded') for d in ('system', 'slide'))
    for calc in _live(_calcjobs()):
        if 'structures' in calc.inputs:
            for node in calc.inputs.structures.values():
                assert not node.base.extras.get('superseded', False), calc
    handed = {n.uuid for n in new['slide'].inputs.structures.values()}
    assert handed <= {n.uuid for n in new['system'].outputs.structures.values()}


def test_a_replaced_system_deck_cascades_to_every_slide_reading_it(aiida_profile_clean,
                                                                   tmp_path):
    """A system deck re-imported for its log alone: the slide decks' own files
    and structure bytes are unchanged, yet they consumed the old system deck's
    (now superseded) structures, so they re-import too (I2)."""
    root = _tree(tmp_path / 'sim', materials=('h-MoS2',))
    import_root(root, label='camp')
    (loc,) = discover_runs(root)
    (loc.lammps_dir / 'run_system.log').write_text(LOG + '# rerun\n', encoding='utf-8')

    report = import_root(root, label='camp')

    assert report.ok(), report.render()
    assert len(report.superseded_decks) == 2
    for calc in _live(_calcjobs()):
        for node in (calc.inputs.structures.values() if 'structures' in calc.inputs else []):
            assert not node.base.extras.get('superseded', False)
    plan = import_root(root, label='camp', dry_run=True)
    assert len(plan.skipped_decks) == 2


def test_a_dry_run_predicts_the_cascade(aiida_profile_clean, tmp_path):
    root = _tree(tmp_path / 'sim', materials=('h-MoS2',))
    import_root(root, label='camp')
    (loc,) = discover_runs(root)
    (loc.data_dir / 'load_10N.data').write_text('changed\n', encoding='utf-8')
    plan = import_root(root, label='camp', dry_run=True)
    assert len(plan.superseded_decks) == 2 and plan.skipped_decks == []


def test_supersede_follows_calcfunction_outputs_not_calcjob_consumers(
        aiida_profile_clean, tmp_path):
    from aiida.engine import calcfunction

    @calcfunction
    def verdict(point):
        return orm.Dict({'status': point.base.attributes.get('status')})

    @calcfunction
    def ruled(value):
        return orm.Dict({'ruled': value.get_dict()['status']})

    root = _tree(tmp_path / 'sim', materials=('h-MoS2',))
    import_root(root, label='camp')
    decks = _by_deck('h-MoS2')
    point = decks['slide'].outputs.points['force10_a0_s2']
    first = verdict(point)
    second = ruled(first)

    supersede(decks['system'])

    assert decks['system'].base.extras.get('superseded') is True
    assert all(n.base.extras.get('superseded') for n in decks['system'].outputs.structures.values())
    assert not decks['slide'].base.extras.get('superseded', False)     # INPUT_CALC consumer
    assert not point.base.extras.get('superseded', False)

    supersede(decks['slide'])
    assert point.base.extras.get('superseded') is True
    assert first.base.extras.get('superseded') is True
    assert second.base.extras.get('superseded') is True


def test_a_config_edit_reimports_every_deck_of_the_run(aiida_profile_clean, tmp_path):
    """The provenance folder is in every deck's fingerprint: a changed config
    changes the blocks, so the run's decks re-import (the dry run says so).
    A rebuild writes the config and its config_hash together."""
    root = _tree(tmp_path / 'sim')
    import_root(root, label='camp')
    (loc,) = [r for r in discover_runs(root) if 'h-MoS2' in r.tokens.rel_path]
    config = _edit_config(loc, temp=350)
    _set_record(loc.run_dir, config_hash=config_hash(config))

    plan = import_root(root, label='camp', dry_run=True)
    assert len(plan.superseded_decks) == 2 and len(plan.skipped_decks) == 2

    report = import_root(root, label='camp')
    assert report.ok(), report.render()
    assert sorted(d.rsplit(':', 1)[1] for d in report.superseded_decks) == ['slide', 'system']
    assert len(report.skipped_decks) == 2


def _mos2(root):
    (loc,) = [r for r in discover_runs(root) if 'h-MoS2' in r.tokens.rel_path]
    return loc


def test_a_config_that_does_not_match_its_hash_is_a_failure(aiida_profile_clean, tmp_path):
    """config.json overwritten after the build (a one-layer rerun of a shared
    pot-level provenance/) and no copy matches: the run fails alone."""
    root = _tree(tmp_path / 'sim')
    _edit_config(_mos2(root), temp=350)

    for dry_run in (True, False):
        report = import_root(root, label='camp', dry_run=dry_run)
        (failure,) = report.failures
        assert 'h-MoS2' in failure[0] and failure[1].startswith('ConfigMismatch:')
        assert 'config_hash' in failure[1]
    assert _by_deck('h-MoS2') == {}
    assert len(_by_deck('h-WS2')) == 2


@pytest.mark.parametrize('where', ['sync_conflict', 'v3_crashed'])
def test_the_config_matching_the_hash_is_the_one_imported(aiida_profile_clean, tmp_path,
                                                          where):
    """The overwritten config's original survives as a Syncthing copy or in
    the crashed build's `_v3_crashed/provenance/`: that one is imported."""
    root = _tree(tmp_path / 'sim')
    loc = _mos2(root)
    original = loc.provenance_dir / 'config.json'
    keep = (loc.provenance_dir / 'config.sync-conflict-20261015-101010-ABCDEFG.json'
            if where == 'sync_conflict'
            else loc.run_dir.parent / '_v3_crashed' / 'provenance' / 'config.json')
    keep.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(original, keep)
    built = json.loads(original.read_text(encoding='utf-8'))
    _edit_config(loc, temp=350)                      # the rerun's config, not this run's

    report = import_root(root, label='camp')

    assert report.ok(), report.render()
    assert any(str(keep) in w and 'config_hash' in w for w in report.warnings)
    assert _stored_config('h-MoS2') == built
    assert import_root(root, label='camp').created_decks == []   # same fingerprint


def test_a_run_with_no_decks_is_reported(aiida_profile_clean, tmp_path):
    """A results-only run (no lammps/*.in) imports nothing, and says so."""
    root = _tree(tmp_path / 'sim')
    loc = _mos2(root)
    shutil.rmtree(loc.lammps_dir)

    for dry_run in (True, False):
        report = import_root(root, label='camp', dry_run=dry_run)
        assert report.ok()
        assert report.no_decks == [loc.tokens.rel_path]
        text = report.render()
        assert 'runs with no decks:  1' in text
        assert f'no decks (nothing imported): {loc.tokens.rel_path}' in text
    assert _by_deck('h-MoS2') == {}
    assert len(_by_deck('h-WS2')) == 2

def test_a_truncated_slide_trace_is_a_truncated_point(aiida_profile_clean, tmp_path):
    root = tmp_path / 'sim'
    run_dir = make_family_run(root, 'afm', loads=(10.0,), n=20,
                              config=slide_config('afm', loads=(10.0,), steps=50_000,
                                                  freq=1000))
    (run_dir / 'lammps' / 'system.in').unlink()
    (run_dir / 'lammps' / 'slide.in').write_text('# slide\n', encoding='utf-8')
    report = import_root(root, label='camp')
    assert report.ok(), report.render()
    (calc,) = _calcjobs()
    point = calc.outputs.points['force10_a0_s2']
    assert point.base.attributes.get('status') == 'truncated'
    assert 'window 20/50 rows' in point.base.attributes.get('status_detail')


def test_files_over_max_stored_file_mb_are_recorded_not_stored(aiida_profile_clean,
                                                              tmp_path, monkeypatch):
    monkeypatch.setattr('frictionsim2d.aiida.calcjob.aiida_setting',   # every byte is over
                        lambda name, default=None, settings_file=None: 0)
    root = tmp_path / 'sim'
    run_dir = make_family_run(root, 'afm', loads=(10.0, 30.0), tiers=('window', 'peratom'),
                              config=slide_config('afm', loads=(10.0, 30.0)))
    (run_dir / 'lammps' / 'slide.in').write_text('# slide\n', encoding='utf-8')
    (run_dir / 'data').mkdir(exist_ok=True)
    (run_dir / 'data' / 'load_10N.data').write_text('LAMMPS data\n', encoding='utf-8')
    (run_dir / 'data' / 'hold_10N.txt').write_text('# hold\n', encoding='utf-8')

    report = import_root(root, label='camp')

    assert report.ok(), report.render()
    decks = _by_deck('h-MoS2')
    system, slide = decks['system'], decks['slide']
    assert 'aux' not in system.outputs
    (hold,) = system.outputs.run_info['not_stored']
    assert hold['name'] == 'data/hold_10N.txt' and hold['size_bytes'] > 0
    assert len(hold['sha256']) == 64
    assert 'structures' in system.outputs          # structures are never capped
    assert 'peratom' not in slide.outputs          # nothing per-atom was kept
    notes = []
    for label, load in (('force10_a0_s2', '10'), ('force30_a0_s2', '30')):
        point = slide.outputs.points[label]
        attrs = point.base.attributes
        assert attrs.get('has_peratom') is False
        (note,) = attrs.get('not_stored')          # its own file only
        assert note['point'] == label
        assert note['name'] == f'results/peratom_f{load}_a0_s2_layer2.txt'
        assert point.base.repository.list_object_names('results')   # tiers never capped
        notes.append(note)
    assert sorted(slide.outputs.run_info['not_stored'], key=lambda n: n['point']) == notes


def test_files_under_the_cap_are_stored(aiida_profile_clean, tmp_path):
    root = tmp_path / 'sim'
    run_dir = _afm_run(root, 'h-MoS2')
    (run_dir / 'data' / 'hold_10N.txt').write_text('# hold\n', encoding='utf-8')
    import_root(root, label='camp')
    system = _by_deck('h-MoS2')['system']
    assert 'hold_10N_txt' in system.outputs.aux
    assert 'not_stored' not in system.outputs.run_info.get_dict()
    assert system.inputs.parameters['max_stored_file_mb'] > 0


def test_superseding_a_deck_another_set_holds_is_reported(aiida_profile_clean, tmp_path):
    root = _tree(tmp_path / 'sim', materials=('h-MoS2',))
    import_root(root, label='first')
    import_root(root, label='second')
    (loc,) = discover_runs(root)
    write_results(loc.run_dir, 'afm', loads=(10.0,), tiers=('raw',), layers=2)

    plan = import_root(root, label='second', dry_run=True)
    assert plan.superseded_elsewhere == [(f'{loc.tokens.rel_path}:slide', 'first')]
    report = import_root(root, label='second')

    assert report.superseded_elsewhere == [(f'{loc.tokens.rel_path}:slide', 'first')]
    assert "also held by set 'first'" in report.render()


# --- per-run failures, logs, versions, machines --------------------------------

def test_a_run_without_run_uuid_fails_alone(aiida_profile_clean, tmp_path):
    root = _tree(tmp_path / 'sim')
    (bad,) = [loc for loc in discover_runs(root) if 'h-MoS2' in loc.tokens.rel_path]
    record = json.loads(bad.provenance_file.read_text(encoding='utf-8'))
    del record['run_uuid']
    bad.provenance_file.write_text(json.dumps(record), encoding='utf-8')

    report = import_root(root, label='camp')

    assert not report.ok()
    assert report.failures[0][0] == bad.tokens.rel_path
    assert 'MissingRunUUID' in report.failures[0][1]
    assert 'FrictionSim2D aiida stamp-uuid' in report.failures[0][1]
    assert len(report.created_decks) == 2
    assert len(_calcjobs()) == 2


def test_a_spring_in_newtons_per_metre_is_a_unit_mismatch(aiida_profile_clean, tmp_path):
    root = _tree(tmp_path / 'sim')
    (bad,) = [loc for loc in discover_runs(root) if 'h-WS2' in loc.tokens.rel_path]
    config = json.loads((bad.provenance_dir / 'config.json').read_text(encoding='utf-8'))
    spring = (config.get('general') or {}).get('driving_spring') or 50.0
    _set_record(bad.run_dir, schema_version='2.2.0', drive_k=float(spring))

    report = import_root(root, label='camp')

    assert [where for where, _m in report.failures] == [bad.tokens.rel_path]
    assert 'UnitMismatch' in report.failures[0][1]
    assert len(report.created_decks) == 2
    plan = import_root(root, label='camp', dry_run=True)
    assert 'UnitMismatch' in plan.failures[0][1]


def test_log_missing_and_no_version_are_reported(aiida_profile_clean, tmp_path):
    root = tmp_path / 'sim'
    no_log = _afm_run(root, 'h-MoS2', logs=False)
    no_version = _afm_run(root, 'h-WS2')
    (no_version / 'lammps' / 'run_slide.log').write_text(LOG_NO_VERSION, encoding='utf-8')
    codes = orm.QueryBuilder().append(orm.InstalledCode).count()

    report = import_root(root, label='camp')

    rel_no_log = no_log.relative_to(root).as_posix()
    rel_no_version = no_version.relative_to(root).as_posix()
    assert sorted(report.log_missing) == [f'{rel_no_log}:slide', f'{rel_no_log}:system']
    assert f'{rel_no_version}:slide' in report.no_version
    assert f'{rel_no_version}:system' not in report.no_version
    assert orm.QueryBuilder().append(orm.InstalledCode).count() == codes + 1
    assert 'no LAMMPS version (no Code)' in report.render()


def test_a_per_pressure_log_in_results_is_found(aiida_profile_clean, tmp_path):
    """MLIP sheet runs: `slide_p<P>gpa.in` logs to `results/log_p<P>gpa.lammps`."""
    root = tmp_path / 'sim'
    run_dir = make_family_run(root, 'sheetonsheet', loads=(0.5, 1.0))
    (run_dir / 'lammps' / 'system.in').unlink()
    for p in ('0p5', '1'):
        (run_dir / 'lammps' / f'slide_p{p}gpa.in').write_text('# slide\n', encoding='utf-8')
        (run_dir / 'results' / f'log_p{p}gpa.lammps').write_text(LOG, encoding='utf-8')

    report = import_root(root, label='camp')

    assert report.ok(), report.render()
    assert len(report.created_decks) == 2
    assert report.log_missing == [] and report.no_version == []
    assert {n.inputs.parameters['log_name'] for n in _calcjobs()} == {
        'results/log_p0p5gpa.lammps', 'results/log_p1gpa.lammps'}

def test_a_log_that_arrives_later_reimports_and_gets_its_code(aiida_profile_clean,
                                                              tmp_path):
    root = tmp_path / 'sim'
    run_dir = _afm_run(root, 'h-MoS2', logs=False)
    first = import_root(root, label='camp')
    assert len(first.log_missing) == 2
    assert all('code' not in n.inputs for n in _calcjobs())

    (run_dir / 'lammps' / 'run_slide.log').write_text(LOG, encoding='utf-8')
    second = import_root(root, label='camp')

    assert second.ok(), second.render()
    assert [d.rsplit(':', 1)[1] for d in second.superseded_decks] == ['slide']
    slide = _by_deck('h-MoS2')['slide']
    assert slide.inputs.code.label == 'lammps-29Aug2024_Update1@localhost'
    assert second.log_missing == []


def test_a_machine_label_with_another_host_is_reused_with_a_warning(aiida_profile_clean,
                                                                    tmp_path):
    import yaml

    from frictionsim2d.aiida.store import machine_computer

    existing = machine_computer('archer2', {'hpc_host': 'login.archer2.ac.uk',
                                            'scheduler_type': 'slurm'})
    root = tmp_path / 'sim'
    run_dir = _afm_run(root, 'h-MoS2')
    (run_dir.parent / 'provenance' / 'settings.yaml').write_text(
        yaml.safe_dump({'hpc': {'hpc_host': 'other.host.org', 'scheduler_type': 'pbs'}}),
        encoding='utf-8')

    report = import_root(root, label='camp', machine='archer2')

    assert report.ok(), report.render()
    assert len(report.warnings) == 1
    assert 'archer2' in report.warnings[0] and 'other.host.org' in report.warnings[0]
    assert 'WARNING' in report.render()
    assert all(n.computer.pk == existing.pk for n in _calcjobs())


def _with_settings_machine(monkeypatch, machine):
    """settings.aiida.machine = `machine`, as `aiida_setting` reads it."""
    monkeypatch.setattr('frictionsim2d.aiida.ingest.aiida_setting',
                        lambda name, default=None, settings_file=None: machine)


def test_settings_aiida_machine_names_the_computer(aiida_profile_clean, tmp_path,
                                                   monkeypatch):
    """`machine=` > settings.aiida.machine > the run's hpc_host > 'localhost'."""
    import yaml

    _with_settings_machine(monkeypatch, 'cluster9')
    root = tmp_path / 'sim'
    run_dir = _afm_run(root, 'h-MoS2')
    (run_dir.parent / 'provenance' / 'settings.yaml').write_text(
        yaml.safe_dump({'hpc': {'hpc_host': 'login.cluster.org'}}), encoding='utf-8')

    assert import_root(root, label='camp').ok()
    assert {n.computer.label for n in _calcjobs()} == {'cluster9'}

    _afm_run(root, 'h-WS2')
    import_root(root, label='camp', machine='explicit')
    assert {n.computer.label for n in _by_deck('h-WS2').values()} == {'explicit'}


def test_the_machine_defaults_to_the_runs_hpc_host(aiida_profile_clean, tmp_path):
    import yaml

    root = tmp_path / 'sim'
    run_dir = _afm_run(root, 'h-MoS2')
    (run_dir.parent / 'provenance' / 'settings.yaml').write_text(
        yaml.safe_dump({'hpc': {'hpc_host': 'login.cluster.org', 'scheduler_type': 'pbs'}}),
        encoding='utf-8')
    report = import_root(root, label='camp')
    assert report.ok() and not report.warnings
    assert {n.computer.label for n in _calcjobs()} == {'login.cluster.org'}
    assert {n.computer.scheduler_type for n in _calcjobs()} == {'core.pbspro'}


def test_without_settings_yaml_the_machine_is_the_configs_hpc_host(aiida_profile_clean,
                                                                   tmp_path):
    """No provenance/settings.yaml: `config.settings.hpc` names the Computer,
    as it does for the blocks (`split_blocks`)."""
    root = tmp_path / 'sim'
    loc_dir = _afm_run(root, 'h-MoS2')
    prov = loc_dir.parent / 'provenance'
    assert not (prov / 'settings.yaml').exists()
    config = json.loads((prov / 'config.json').read_text(encoding='utf-8'))
    config['settings']['hpc'] = {'hpc_host': 'login.cluster.org', 'scheduler_type': 'pbs'}
    (prov / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    _set_record(loc_dir, config_hash=config_hash(config))

    report = import_root(root, label='camp')

    assert report.ok(), report.render()
    assert {n.computer.label for n in _calcjobs()} == {'login.cluster.org'}
    assert {n.computer.scheduler_type for n in _calcjobs()} == {'core.pbspro'}

def test_a_parse_failure_keeps_run_info_and_is_retried(aiida_profile_clean, tmp_path,
                                                      monkeypatch):
    from frictionsim2d.aiida import calcjob

    def boom(self, *args, **kwargs):
        raise RuntimeError('boom')

    root = _tree(tmp_path / 'sim', materials=('h-MoS2',))
    with monkeypatch.context() as patch:
        patch.setattr(calcjob.LammpsFrictionParser, '_points', boom)
        report = import_root(root, label='camp')
    assert [m for _w, m in report.failures] and 'boom' in report.failures[0][1]
    slide = _by_deck('h-MoS2')['slide']
    assert slide.exit_status == 330
    assert slide.outputs.run_info['log_has_version'] is True
    assert 'log' in slide.outputs
    assert DECK_SHA_EXTRA not in slide.base.extras.all

    again = import_root(root, label='camp')
    assert again.ok(), again.render()
    assert [d.rsplit(':', 1)[1] for d in again.created_decks] == ['slide']
    assert slide.base.extras.get('superseded') is True
    assert _by_deck('h-MoS2')['slide'].is_finished_ok


def test_reindex_restores_the_deck_key(aiida_profile_clean, tmp_path):
    from frictionsim2d.aiida.store import reindex
    import_root(_tree(tmp_path / 'sim', materials=('h-MoS2',)), label='camp')
    keys = {n.pk: n.base.extras.get(DECK_KEY_EXTRA) for n in _calcjobs()}
    for node in _calcjobs():
        node.base.extras.delete(DECK_KEY_EXTRA)
    reindex()
    assert {n.pk: n.base.extras.get(DECK_KEY_EXTRA) for n in _calcjobs()} == keys


def test_an_empty_root_is_a_failure(aiida_profile_clean, tmp_path):
    (tmp_path / 'empty').mkdir()
    report = import_root(tmp_path / 'empty')
    assert not report.ok()


# --- the aiida-free helpers (ported from test_ingest.py) ------------------------

REL = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'


def test_slide_points_group_the_three_tiers(tmp_path):
    root = tmp_path / 'sim'
    make_run(root, REL, 'afm', loads=(10.0, 30.0), angles=(0.0, 90.0),
             speeds=(2.0,), tiers=('window', 'raw'))
    loc, = discover_runs(root)

    points = load_points(loc)

    assert len(points) == 4
    assert all(isinstance(p, LoadPoint) for p in points)
    assert sorted((p.load, p.angle) for p in points) == [
        (10.0, 0.0), (10.0, 90.0), (30.0, 0.0), (30.0, 90.0)]
    one = points[0]
    assert (one.family, one.load_kind, one.speed, one.layers) == ('afm', 'force', 2.0, 2)
    assert set(one.tiers) == {'window', 'raw'}


def test_sheet_points_are_pressures(tmp_path):
    root = tmp_path / 'sim'
    make_run(root, 'sheetonsheet/h-MoS2/100x_100y/K300', 'sheetonsheet', loads=(1.0,))
    loc, = discover_runs(root)
    point, = load_points(loc)
    assert (point.load_kind, point.load, point.layers) == ('pressure', 1.0, None)



def test_a_sync_conflict_copy_is_not_a_tier(tmp_path):
    """A Syncthing copy sorts before its original (`.sync` < `.txt`) and also
    matches the point pattern: it must never stand in for the original."""
    root = tmp_path / 'sim'
    make_run(root, REL, 'afm', loads=(10.0,), tiers=('window', 'raw'))
    loc, = discover_runs(root)
    for path in sorted(loc.results_dir.iterdir()):
        stem, suffix = path.name.split('.', 1)
        shutil.copy(path, loc.results_dir / f'{stem}.sync-conflict-20261001-120000-ABCDEFG.{suffix}')

    point, = load_points(loc)

    assert set(point.tiers) == {'window', 'raw'}
    assert not any('.sync-conflict-' in p.name for p in point.tiers.values())

def test_indent_pairs_its_retract_curve_with_the_hold(tmp_path):
    root = tmp_path / 'sim'
    run_dir = make_run(root, 'indent/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1',
                       'indent', loads=(30.0,))
    (run_dir / 'results' / 'indent_f30N_l1.csv').write_text(
        'step,z_tip,fz_tip\n0,0,5.0\n', encoding='utf-8')
    loc, = discover_runs(root)
    point, = load_points(loc)
    assert point.tiers['retract'].name == 'indent_f30N_l1.csv'


def test_pes_refine_grid_is_not_a_second_point(tmp_path):
    root = tmp_path / 'sim'
    run_dir = make_run(root, 'pes_sheet/h-MoS2/100x_100y/K300', 'pes_sheet', tiers=())
    (run_dir / 'results' / 'pes_scan_refine.csv').write_text(
        PES_GRID_CSV['pes_sheet'], encoding='utf-8')
    loc, = discover_runs(root)
    point, = load_points(loc)
    assert point.load_kind == 'none'
    assert point.tiers['window'].name == 'pes_scan.csv'


FAMILY_CASES = [
    ('afm', 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2', 'force'),
    ('indent', 'indent/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1', 'force'),
    ('pes_tip', 'pes_tip/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L1', 'none'),
    ('sheetonsheet', 'sheetonsheet/h-MoS2/100x_100y/K300', 'pressure'),
    ('hetero', 'sheetonsheet/h-MoS2+h-WS2/100x_100y/K300', 'pressure'),
    ('flake', 'sheetonsheet/h-MoS2+flake-h-WS2/100x_100y/K300', 'pressure'),
    ('pes_sheet', 'pes_sheet/h-MoS2/100x_100y/K300', 'none'),
]


@pytest.mark.parametrize('family, rel, load_kind', FAMILY_CASES)
def test_every_family_yields_a_load_point(tmp_path, family, rel, load_kind):
    root = tmp_path / 'sim'
    make_run(root, rel, family, tiers=() if family in PES_GRID_CSV else ('window',))
    loc, = discover_runs(root)
    point, = load_points(loc)
    assert (point.family, point.load_kind) == (family, load_kind)
    assert 'window' in point.tiers


def test_families_tuple_is_fully_covered():
    from frictionsim2d.core.observables import FAMILIES
    assert {f for f, _rel, _kind in FAMILY_CASES} == set(FAMILIES)


def test_a_tier_collision_is_warned_not_silently_overwritten(tmp_path, caplog):
    root = tmp_path / 'sim'
    run_dir = make_run(root, REL, 'afm', loads=(30.0,))
    original = run_dir / 'results' / 'friction_f30_a0_s2_layer2.txt'
    dup = run_dir / 'results' / 'friction_f30.0_a0_s2_layer2.txt.bak'
    dup.write_text(original.read_text(encoding='utf-8'), encoding='utf-8')
    loc, = discover_runs(root)
    with caplog.at_level('WARNING'):
        load_points(loc)
    assert original.name in caplog.text and dup.name in caplog.text


def test_importing_ingest_does_not_require_aiida():
    code = (
        "import sys\n"
        "sys.modules['aiida'] = None\n"
        "import frictionsim2d.aiida.ingest as ingest\n"
        "assert sys.modules.get('aiida') is None\n"
        "assert hasattr(ingest, 'load_points') and hasattr(ingest, 'import_root')\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


def test_an_explicit_settings_file_names_the_machine(aiida_profile_clean, tmp_path):
    settings_file = tmp_path / 'settings.yaml'
    settings_file.write_text('aiida:\n  machine: from-file\n', encoding='utf-8')
    root = tmp_path / 'sim'
    _afm_run(root, 'h-MoS2')

    assert import_root(root, label='camp', settings_file=settings_file).ok()
    assert {n.computer.label for n in _calcjobs()} == {'from-file'}
