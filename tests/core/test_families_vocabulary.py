"""One family vocabulary and one model vocabulary (AiiDA spec §3 F1)."""
from src.core.observables import FAMILIES, REGISTRY
from src.hpc.manifest import JobManifest
from src.postprocessing.read_data import DataReader
from src.run.naming import AFM_LIKE_MODELS, MODELS, SHEET_LIKE_MODELS


def test_families_is_the_registry_vocabulary():
    assert FAMILIES == ('afm', 'indent', 'sheetonsheet', 'hetero', 'flake',
                        'pes_tip', 'pes_sheet')
    declared = {st for ob in REGISTRY.values() for st in ob.sim_types}
    assert declared <= set(FAMILIES), f"registry declares {declared - set(FAMILIES)}"


def test_models_are_the_top_directory_axis():
    assert MODELS == ('afm', 'indent', 'pes_tip', 'sheetonsheet', 'pes_sheet')
    assert set(MODELS) == set(AFM_LIKE_MODELS) | set(SHEET_LIKE_MODELS)
    assert set(MODELS) <= set(FAMILIES)


def test_consumers_share_the_one_tuple():
    assert DataReader._KNOWN_FAMILIES is FAMILIES
    assert JobManifest._TOP_DIRS is FAMILIES


def test_manifest_accepts_an_indent_top_directory(tmp_path):
    lammps = tmp_path / 'indent' / 'h-MoS2' / '100x_100y' / 'K300' / 'L1' / 'lammps'
    lammps.mkdir(parents=True)
    (lammps / 'system.in').write_text('# system\n', encoding='utf-8')
    (lammps / 'slide.in').write_text('# slide\n', encoding='utf-8')

    manifest = JobManifest.from_simulation_directory(tmp_path)

    assert {j.material for j in manifest.jobs} == {'h-MoS2'}
