"""Every builder stamps its family, its model and every material (spec §3 F3)."""
import json
from types import SimpleNamespace

import pytest
import yaml

from src.builders.afm.base import AFMSimulation
from src.builders.afm.hetero import FiniteHeteroAFMSimulation, HeteroAFMSimulation
from src.builders.afm.indent import IndentSimulation
from src.builders.afm.pes_tip import PESTipSimulation
from src.builders.sheetonsheet.base import SheetOnSheetSimulation
from src.builders.sheetonsheet.flake_slide import FlakeSlideSimulation
from src.builders.sheetonsheet.hetero import HeteroSheetOnSheetSimulation
from src.builders.sheetonsheet.pes_sheet import PESSheetSimulation
from src.builders.simulation_base import SimulationBase
from src.core.observables import FAMILIES
from src.run.tree import discover_runs
from tests.helpers.examples import build

FAMILY_BY_CLASS = [
    (AFMSimulation, 'afm'),
    (IndentSimulation, 'indent'),
    (PESTipSimulation, 'pes_tip'),
    (SheetOnSheetSimulation, 'sheetonsheet'),
    (HeteroSheetOnSheetSimulation, 'hetero'),
    (FlakeSlideSimulation, 'flake'),
    (PESSheetSimulation, 'pes_sheet'),
    # R3: the AFM hetero builders' results files still follow the AFM schema
    # (v_fz, v_xfrict, ...), not the sheet family's 'hetero' stack schema
    # (v_comx_top, react_*, ...) -- 'family' names the schema, not the geometry.
    (HeteroAFMSimulation, 'afm'),
    (FiniteHeteroAFMSimulation, 'afm'),
]


@pytest.mark.parametrize('cls, family', FAMILY_BY_CLASS)
def test_every_builder_declares_a_known_family(cls, family):
    assert cls.FAMILY == family
    assert cls.FAMILY in FAMILIES


def test_obs_ctx_is_built_from_the_class_family(monkeypatch):
    """An indent run must stamp its OWN family ('indent'), not AFM's literal.

    Regression guard for the pre-Task-6 code, which hardcoded
    `sim_type='afm'` in `AFMSimulation._build_obs_ctx` regardless of which
    subclass called it -- so `IndentSimulation` (inherits `_build_obs_ctx`
    unchanged) silently stamped `family=afm` into its own results header
    instead of `'indent'`. `HeteroAFMSimulation` cannot catch this: under R3
    its FAMILY is `'afm'` too, the same as the old hardcoded literal, so a
    test built on it would pass against the un-fixed code just as well as the
    fixed one (confirmed: it did, in the RED run for this task).
    """
    import src.builders.utils.obs_ctx as obs_mod

    seen = {}
    monkeypatch.setattr(obs_mod, 'build_obs_ctx',
                        lambda **kw: seen.update(kw) or {'title1': ''})

    builder = IndentSimulation.__new__(IndentSimulation)
    monkeypatch.setattr(type(builder), '_contact_cutoff', lambda self: 4.8)
    monkeypatch.setattr(
        type(builder), 'config',
        property(lambda self: SimpleNamespace(settings=SimpleNamespace(
            output=SimpleNamespace(channels={'window': True}, optional_channels=(),
                                   results_frequency=1000)))),
        raising=False)

    builder._build_obs_ctx(2, 'friction_x.txt', use_langevin=True)

    assert seen['sim_type'] == 'indent'


def test_model_defaults_to_family_until_the_runner_sets_it():
    class _Stub(SimulationBase):
        FAMILY = 'flake'

    stub = _Stub.__new__(_Stub)
    assert stub.model == 'flake'
    stub.set_model('sheetonsheet')
    assert stub.model == 'sheetonsheet'


@pytest.mark.integration
def test_afm_run_records_sheet_tip_and_substrate(tmp_path):
    # examples/afm_config.ini sweeps layers=[1,2,3] -> 3 discovered runs
    # (R15); every one of them must carry the same family/model/materials
    # shape, so check them all rather than unpacking a single result.
    result = build('afm_config.ini', 'afm', tmp_path)
    assert not result.failures

    runs = list(discover_runs(result.simulation_root))
    assert runs
    for loc in runs:
        record = json.loads(loc.provenance_file.read_text(encoding='utf-8'))

        assert record['schema_version'] == '2.1.0'
        assert record['family'] == 'afm'
        assert record['model'] == 'afm'
        roles = [m['role'] for m in record['materials']]
        assert roles.count('sheet') == 1
        assert 'tip' in roles and 'substrate' in roles
        sheet = next(m for m in record['materials'] if m['role'] == 'sheet')
        assert sheet['pot_type'] and sheet['pot_file_sha256']


@pytest.mark.integration
def test_settings_yaml_is_written_beside_config_json(tmp_path):
    result = build('sheet_config.ini', 'sheetonsheet', tmp_path)
    assert not result.failures
    loc, = discover_runs(result.simulation_root)

    settings_file = loc.provenance_dir / 'settings.yaml'

    assert settings_file.is_file()
    loaded = yaml.safe_load(settings_file.read_text(encoding='utf-8'))
    assert loaded['thermostat']['type']
    assert (loc.provenance_dir / 'config.json').is_file()
