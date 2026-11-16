"""The query API filters on RunTokens and returns RunSummary (spec §4.4)."""
import pytest

pytest.importorskip('aiida')
# The old set/run/result ingest is gone (plan Task 6, `import_root`);
# Task 8 rewrites the query layer over the CalcJob graph.
pytest.skip('old ingest graph; Task 8 rewrites the query layer over the CalcJob graph', allow_module_level=True)
pytestmark = pytest.mark.aiida

from src.aiida.ingest import ingest_root  # noqa: E402
from src.aiida.query import Friction2DDB, QueryResult  # noqa: E402
from tests.helpers.synthetic_results import make_run  # noqa: E402

AFM = 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/L2'
SHEET = 'sheetonsheet/h-MoS2+h-WS2/100x_100y/K300'


def _populate(tmp_path):
    root = tmp_path / 'simulation_q'
    make_run(root, AFM, 'afm', loads=(10.0, 30.0), tiers=('window', 'raw'))
    make_run(root, SHEET, 'hetero', loads=(1.0,))
    ingest_root(root, label='q1')
    return root


def test_query_filters_on_family_and_model(aiida_profile_clean, tmp_path):
    _populate(tmp_path)
    db = Friction2DDB()

    hetero = db.query(family='hetero')
    afm = db.query(model='afm')

    assert isinstance(hetero, QueryResult)
    assert [r.rel_path for r in hetero.runs] == [SHEET]
    assert [r.rel_path for r in afm.runs] == [AFM]


def test_query_filters_on_set_label_and_load_range(aiida_profile_clean, tmp_path):
    _populate(tmp_path)
    db = Friction2DDB()

    out = db.query(set_label='q1', force_range=(20.0, 40.0))

    assert out.total_count == 1
    assert out.results[0].load == 30.0
    assert out.results[0].load_kind == 'force'


def test_run_filters_scope_the_result_rows(aiida_profile_clean, tmp_path):
    """A cross product would return all three results for a model='afm' query."""
    _populate(tmp_path)

    out = Friction2DDB().query(model='afm')

    assert out.total_count == 2                     # not 3
    assert {r.load for r in out.results} == {10.0, 30.0}
    assert [run.rel_path for run in out.runs] == [AFM]


def test_every_run_token_axis_is_filterable(aiida_profile_clean, tmp_path):
    _populate(tmp_path)
    db = Friction2DDB()

    assert db.query(size_x=100.0).total_count == 3
    assert db.query(size_y=999.0).total_count == 0
    assert db.query(substrate_amorphous=True).total_count == 2
    assert db.query(tip_radius=25.0).total_count == 2
    assert db.query(flake_material=None, rel_path=SHEET).total_count == 1
    assert db.query(rel_path=AFM, angle_range=(0.0, 0.0)).total_count == 2
    assert db.query(temperature_range=(299.0, 301.0)).total_count == 3


def test_dataframe_is_one_row_per_result_with_summary_columns(aiida_profile_clean,
                                                              tmp_path):
    _populate(tmp_path)

    df = Friction2DDB().query().to_dataframe()

    assert len(df) == 3
    for col in ('model', 'family', 'materials', 'rel_path', 'set_label',
                'run_uuid', 'result_uuid', 'load_kind', 'load', 'angle', 'speed',
                'mean_cof', 'std_cof', 'time_series_hash', 'ntimesteps'):
        assert col in df.columns, col
    assert df['mean_cof'].notna().all()


def test_available_conditions_and_materials_still_serve_the_gui(aiida_profile_clean,
                                                                tmp_path):
    _populate(tmp_path)
    db = Friction2DDB()

    conditions = db.get_available_conditions()

    assert set(conditions) == {'forces', 'pressures', 'angles', 'speeds',
                               'temperatures', 'layers', 'families', 'models',
                               'potentials', 'set_labels'}
    assert conditions['forces'] == [10.0, 30.0]
    assert conditions['pressures'] == [1.0]
    assert conditions['families'] == ['afm', 'hetero']
    assert conditions['models'] == ['afm', 'sheetonsheet']
    assert conditions['set_labels'] == ['q1']
    assert db.get_available_materials() == ['h-MoS2', 'h-WS2']


def test_potentials_and_set_labels_are_filterable_and_listed(aiida_profile_clean,
                                                             tmp_path):
    """Every value get_available_conditions offers must be usable as a filter."""
    root = tmp_path / 'simulation_p'
    make_run(root, 'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/pot_sw_MoS/L2',
             'afm', loads=(30.0,))
    ingest_root(root, label='p1')
    db = Friction2DDB()

    conditions = db.get_available_conditions()

    assert conditions['potentials'] == ['pot_sw_MoS']
    assert conditions['set_labels'] == ['p1']
    assert db.query(potential='pot_sw_MoS').total_count == 1
    assert db.query(set_label='p1').total_count == 1


def test_orphaned_results_are_excluded_from_query(aiida_profile_clean, tmp_path):
    """A results node whose run_uuid names no live run must never come back --
    not from a bare query(), and not from a result-level-only filter query."""
    from src.aiida.data import FrictionResultsData

    _populate(tmp_path)
    orphan = FrictionResultsData()
    orphan.family = 'afm'
    orphan.load_kind = 'force'
    orphan.load = 999.0
    orphan.angle = 0.0
    orphan.speed = 0.0
    orphan.tiers = []
    orphan.run_uuid = 'not-a-real-run-uuid'
    orphan.store()

    bare = Friction2DDB().query()
    assert bare.total_count == 3
    assert 999.0 not in {r.load for r in bare.results}

    ranged = Friction2DDB().query(force_range=(0.0, 2000.0))
    assert 999.0 not in {r.load for r in ranged.results}


def test_run_uuid_filter_is_chunked(aiida_profile_clean, tmp_path, monkeypatch):
    """Chunking the `run_uuid IN (...)` filter must not drop or duplicate
    results across batches, however small the batch size."""
    monkeypatch.setattr('src.aiida.query._RUN_UUID_CHUNK_SIZE', 1)
    _populate(tmp_path)

    out = Friction2DDB().query()

    assert out.total_count == 3
    assert {r.load for r in out.results} == {10.0, 30.0, 1.0}


def test_materials_filter_uses_all_semantics(aiida_profile_clean, tmp_path):
    """A run matches only if it contains EVERY given material, not just one."""
    _populate(tmp_path)
    db = Friction2DDB()

    assert db.query(materials=['h-WS2']).total_count == 1          # SHEET only
    assert db.query(materials=['h-MoS2']).total_count == 3         # both runs
    assert db.query(materials=['h-MoS2', 'h-WS2']).total_count == 1     # SHEET only
    assert db.query(materials=['h-WS2', 'h-MoS2']).total_count == 1     # order-independent


def test_layers_substrate_and_tip_material_are_filterable(aiida_profile_clean, tmp_path):
    _populate(tmp_path)
    db = Friction2DDB()

    assert db.query(layers=2).total_count == 2
    assert db.query(substrate='Si').total_count == 2
    assert db.query(tip_material='Si').total_count == 2


def test_pressure_and_speed_ranges_filter_results(aiida_profile_clean, tmp_path):
    _populate(tmp_path)
    db = Friction2DDB()

    out = db.query(pressure_range=(0.5, 2.0))
    assert out.total_count == 1
    assert out.results[0].load_kind == 'pressure'
    assert out.results[0].load == 1.0

    assert db.query(speed_range=(1.0, 3.0)).total_count == 3
    assert db.query(speed_range=(2.5, 3.0)).total_count == 0


def test_order_by_sorts_results_ascending_and_descending(aiida_profile_clean, tmp_path):
    _populate(tmp_path)
    db = Friction2DDB()

    ascending = db.query(model='afm', order_by='load')
    descending = db.query(model='afm', order_by='-load')

    assert [r.load for r in ascending.results] == [10.0, 30.0]
    assert [r.load for r in descending.results] == [30.0, 10.0]


def test_default_order_is_deterministic_by_rel_path_then_load(aiida_profile_clean,
                                                              tmp_path):
    """No order_by given -- still the same order every time, not backend luck."""
    _populate(tmp_path)

    out = Friction2DDB().query()

    assert [r.load for r in out.results] == [10.0, 30.0, 1.0]


def test_limit_caps_the_result_list(aiida_profile_clean, tmp_path):
    _populate(tmp_path)

    out = Friction2DDB().query(limit=1)

    assert out.total_count == 1
    assert out.results[0].load == 10.0


def test_get_statistics_counts_runs_by_model_family_and_material(aiida_profile_clean,
                                                                  tmp_path):
    _populate(tmp_path)

    stats = Friction2DDB().get_statistics()

    assert stats['total_simulations'] == 2
    assert stats['by_model'] == {'afm': 1, 'sheetonsheet': 1}
    assert stats['by_family'] == {'afm': 1, 'hetero': 1}
    assert stats['by_material'] == {'h-MoS2': 2, 'h-WS2': 1}
    assert stats['n_materials'] == 2


def test_export_for_reproduction_delegates_to_export_results(aiida_profile_clean,
                                                             tmp_path, monkeypatch):
    _populate(tmp_path)
    seen = {}
    monkeypatch.setattr('src.aiida.export.export_results',
                        lambda label, dest, profile=None:
                        seen.update(label=label, dest=dest) or [])
    run = Friction2DDB().query(model='afm').runs[0]

    Friction2DDB().export_for_reproduction(run, tmp_path / 'repro')

    assert seen['label'] == 'q1'
    assert seen['dest'] == tmp_path / 'repro'


def test_provenance_bundle_is_five_categories_only(aiida_profile_clean, tmp_path):
    """The GUI's "download provenance" must not hand back the results too."""
    from src.aiida.data import FrictionSimulationData
    from src.aiida.export import export_provenance_bundle
    from src.run.tree import discover_runs

    root = tmp_path / 'simulation_b'
    run_dir = make_run(root, AFM, 'afm', loads=(30.0,))
    prov = run_dir.parent / 'provenance'
    (prov / 'cif').mkdir(parents=True, exist_ok=True)
    (prov / 'cif' / 'h-MoS2.cif').write_text('data_MoS2\n', encoding='utf-8')
    (prov / 'potentials').mkdir(parents=True, exist_ok=True)
    (prov / 'potentials' / 'MoS.sw').write_text('# sw\n', encoding='utf-8')
    locator, = discover_runs(root)

    # Build the run node by hand so a results file can be staged on it while it
    # is still unstored -- a stored node's repository is immutable, and
    # `add_run_files` never stores results, which is exactly what this asserts.
    run = FrictionSimulationData()
    run.set_from_locator(locator, set_uuid='s', set_label='b1')
    run.add_run_files(locator)
    run.base.repository.put_object_from_file(
        str(run_dir / 'results' / 'friction_f30_a0_s2_layer2.txt'),
        'results/friction_f30_a0_s2_layer2.txt')
    run.store()

    dest = tmp_path / 'bundle'
    written = export_provenance_bundle(run, dest)

    relative = {path.relative_to(dest).as_posix() for path in written}
    assert relative == {
        'provenance.json',
        'provenance/config.json',
        'provenance/settings.yaml',
        'provenance/cif/h-MoS2.cif',
        'provenance/potentials/MoS.sw',
    }
    assert not (dest / 'results').exists()
    assert not (dest / 'lammps').exists()
    assert written == sorted(written)
    assert (dest / 'provenance' / 'potentials' / 'MoS.sw').read_text() == '# sw\n'


def test_provenance_bundle_creates_its_destination(aiida_profile_clean, tmp_path):
    """And it works on a run node that came out of a normal ingest."""
    from src.aiida.export import export_provenance_bundle

    root = tmp_path / 'simulation_b2'
    make_run(root, AFM, 'afm', loads=(30.0,))
    ingest_root(root, label='b2')
    run = Friction2DDB().query(set_label='b2').runs[0]

    dest = tmp_path / 'nested' / 'bundle'
    written = export_provenance_bundle(run, dest)

    assert dest.is_dir()
    assert (dest / 'provenance.json').is_file()
    assert written


def test_export_for_reproduction_still_exports_the_whole_set(aiida_profile_clean,
                                                             tmp_path, monkeypatch):
    root = tmp_path / 'simulation_b3'
    make_run(root, AFM, 'afm', loads=(30.0,))
    make_run(root, SHEET, 'hetero', loads=(1.0,))
    ingest_root(root, label='b3')
    seen = {}
    monkeypatch.setattr('src.aiida.export.export_results',
                        lambda label, dest, profile=None:
                        seen.update(label=label, dest=dest) or [])
    run = Friction2DDB().query(model='afm').runs[0]

    Friction2DDB().export_for_reproduction(run, tmp_path / 'whole')

    assert seen['label'] == 'b3'          # the SET, not just this run


def test_removed_helpers_are_gone():
    for name in ('query_by_conditions', 'query_by_tip', 'query_by_substrate',
                 'query_by_material', 'compare_materials', 'find_similar',
                 'get_provenance', 'get_set_results'):
        assert not hasattr(Friction2DDB, name), name


def test_a_pes_grid_adds_no_ghost_conditions(aiida_profile_clean, tmp_path):
    """m4: a PES point is load_kind='none' with placeholder zeros; it used to
    land in `pressures` as 0.0 (and as a 0 deg / 0 speed option)."""
    root = tmp_path / 'simulation_pes'
    make_run(root, AFM, 'afm', loads=(10.0,), speeds=(2.0,), angles=(30.0,))
    make_run(root, 'pes_sheet/h-MoS2/100x_100y/K300', 'pes_sheet')
    ingest_root(root, label='pes-cond')

    conditions = Friction2DDB().get_available_conditions()

    assert conditions['forces'] == [10.0]
    assert conditions['pressures'] == []
    assert conditions['angles'] == [30.0]
    assert conditions['speeds'] == [2.0]
    assert 'pes_sheet' in conditions['families']
