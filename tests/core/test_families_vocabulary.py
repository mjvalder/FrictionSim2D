"""One family vocabulary and one model vocabulary (AiiDA spec §3 F1)."""
import pytest

from frictionsim2d.core.observables import FAMILIES, PES_FAMILIES, REGISTRY
from frictionsim2d.hpc.manifest import JobManifest
from frictionsim2d.postprocessing.read_data import DataReader
from frictionsim2d.run.naming import AFM_LIKE_MODELS, MODELS, SHEET_LIKE_MODELS


def test_families_is_the_registry_vocabulary():
    assert FAMILIES == ('afm', 'indent', 'sheetonsheet', 'hetero', 'flake',
                        'pes_tip', 'pes_sheet')
    declared = {st for ob in REGISTRY.values() for st in ob.sim_types}
    assert declared <= set(FAMILIES), f"registry declares {declared - set(FAMILIES)}"


def test_models_are_the_top_directory_axis():
    assert MODELS == ('afm', 'indent', 'pes_tip', 'sheetonsheet', 'pes_sheet')
    assert set(MODELS) == set(AFM_LIKE_MODELS) | set(SHEET_LIKE_MODELS)
    assert set(MODELS) <= set(FAMILIES)


def test_data_reader_uses_the_family_tuple():
    for family in FAMILIES:
        assert DataReader._registry_sim_type('tip', f'x family={family}') == family
    with pytest.raises(ValueError, match='unknown family'):
        DataReader._registry_sim_type('tip', 'x family=bogus')


def test_pes_families_is_the_one_shared_tuple():
    import frictionsim2d.aiida.calcjob as calcjob
    import frictionsim2d.aiida.ingest as ingest
    import frictionsim2d.data.models as models
    import frictionsim2d.postprocessing.run_status as run_status
    assert PES_FAMILIES == ('pes_tip', 'pes_sheet')
    assert set(PES_FAMILIES) <= set(FAMILIES)
    for module in (calcjob, ingest, models, run_status):
        assert module.PES_FAMILIES is PES_FAMILIES
        assert not hasattr(module, '_PES_FAMILIES')


def test_manifest_accepts_an_indent_top_directory(tmp_path):
    lammps = tmp_path / 'indent' / 'h-MoS2' / '100x_100y' / 'K300' / 'L1' / 'lammps'
    lammps.mkdir(parents=True)
    (lammps / 'system.in').write_text('# system\n', encoding='utf-8')
    (lammps / 'slide.in').write_text('# slide\n', encoding='utf-8')

    manifest = JobManifest.from_simulation_directory(tmp_path)

    assert {j.material for j in manifest.jobs} == {'h-MoS2'}
