"""Validation speaks the v3 record's vocabulary (AiiDA spec section 5)."""
import pytest

from src.core.observables import FAMILIES
from src.data.models import ResultRecord
from src.data.validation import (
    validate_completeness, validate_consistency, validate_ranges, validate_record,
)


def _record(**over):
    base = dict(material='h-MoS2', family='afm', model='afm',
                materials=['h-MoS2'], mean_cof=0.012, force_nN=30.0,
                temperature=300.0, layers=2)
    base.update(over)
    return ResultRecord(**base)


def test_a_plain_afm_record_validates():
    assert validate_record(_record()).is_valid


@pytest.mark.parametrize('family', FAMILIES)
def test_every_family_is_accepted(family):
    model = 'sheetonsheet' if family in ('hetero', 'flake') else family
    if family in ('pes_tip', 'pes_sheet'):
        record = _record(family=family, model=model, mean_cof=None,
                         force_nN=None, extras={'pes_barrier_U': 0.4})
    else:
        record = _record(family=family, model=model)

    result = validate_record(record)

    assert result.is_valid, result.errors


def test_simulation_type_is_no_longer_required_or_known():
    import src.data.validation as validation

    assert 'simulation_type' not in validation._REQUIRED_FIELDS
    assert not hasattr(validation, '_VALID_SIMULATION_TYPES')


def test_an_unknown_family_or_model_is_refused():
    assert not validate_ranges(_record(family='poke')).is_valid
    assert not validate_ranges(_record(model='hetero')).is_valid   # a family, not a model


def test_a_pes_row_is_complete_without_a_cof():
    """A grid has no COF; requiring one would reject every PES row."""
    record = _record(family='pes_sheet', model='pes_sheet', mean_cof=None,
                     force_nN=None, extras={'pes_barrier_U': 0.4})

    assert validate_completeness(record).is_valid


def test_a_row_with_neither_cof_nor_extras_is_incomplete():
    record = _record(mean_cof=None, extras={})

    result = validate_completeness(record)

    assert not result.is_valid
    assert 'mean_cof' in ' '.join(result.errors)


def test_an_incomplete_row_is_refused():
    """is_complete=False means the COF could not be computed -- do not publish."""
    result = validate_completeness(_record(mean_cof=None, is_complete=False,
                                           extras={'slip_n_events': 3.0}))

    assert not result.is_valid
    assert 'is_complete' in ' '.join(result.errors)


def test_zero_cof_is_flagged_not_silently_published():
    """0.0 used to validate as in-range; it is now a warning at least."""
    result = validate_ranges(_record(mean_cof=0.0))

    assert result.warnings
    assert 'mean_cof' in ' '.join(result.warnings)


def test_consistency_uses_family_not_simulation_type():
    afm = validate_consistency(_record(force_nN=None, pressure_gpa=1.0))
    sheet = validate_consistency(_record(family='sheetonsheet',
                                         model='sheetonsheet',
                                         force_nN=30.0, pressure_gpa=None))

    assert afm.warnings and 'force_nN' in ' '.join(afm.warnings)
    assert sheet.warnings and 'pressure_gpa' in ' '.join(sheet.warnings)
