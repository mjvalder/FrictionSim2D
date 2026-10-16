"""Descriptor -> measured range (src.core.scaling)."""
import pytest

from src.core.scaling import REGIMES, ScaleContext, ScaledRange, scale


AFM = ScaleContext(model='afm', tip_radius_A=25, driving_spring_N_m=8, temperature_K=300)


@pytest.mark.parametrize('descriptor, low, high', [
    ('low', 10, 20), ('mid', 20, 50), ('high', 50, 100)])
@pytest.mark.parametrize('model', ['afm', 'indent', 'pes_tip'])
def test_afm_family_load_regimes(model, descriptor, low, high):
    result = scale('load', descriptor, ScaleContext(model=model))
    assert isinstance(result, ScaledRange)
    assert (result.quantity, result.unit, result.low, result.high) == ('load', 'nN', low, high)
    assert result.warnings == ()


@pytest.mark.parametrize('descriptor, expected', [
    ('low load', 10), ('small', 10), ('light', 10), ('LOW', 10),
    ('moderate', 20), ('medium', 20), ('mid', 20),
    ('large', 50), ('heavy', 50), ('high loads', 50)])
def test_descriptor_synonyms(descriptor, expected):
    assert scale('load', descriptor, AFM).low == expected


@pytest.mark.parametrize('quantity', ['load', 'loads', 'force', 'normal load'])
def test_quantity_synonyms(quantity):
    assert scale(quantity, 'low', AFM).high == 20


def test_working_names_the_measured_source():
    result = scale('load', 'low', AFM)
    assert 'ML_taxonomy' in result.working and 'BASELINE_LOADS' in result.working
    assert '\n' not in result.working
    assert 'MID_LOAD_CORE' in scale('load', 'mid', AFM).working


def test_every_regime_has_a_source():
    assert REGIMES
    for regime in REGIMES:
        assert regime.source and regime.low < regime.high


def test_inside_validity_no_warning():
    assert scale('load', 'low', AFM).warnings == ()
    edge = ScaleContext('afm', tip_radius_A=20, driving_spring_N_m=50, temperature_K=350)
    assert scale('load', 'low', edge).warnings == ()


def test_tip_radius_outside_validity_warns_same_numbers():
    result = scale('load', 'low', ScaleContext('afm', tip_radius_A=60))
    assert (result.low, result.high) == (10, 20)
    assert len(result.warnings) == 1
    assert 'R = 25' in result.warnings[0] and 'R = 60' in result.warnings[0]


def test_spring_and_temperature_outside_validity_warn():
    result = scale('load', 'mid', ScaleContext('afm', driving_spring_N_m=2, temperature_K=500))
    assert len(result.warnings) == 2
    assert any('2 N/m' in w for w in result.warnings)
    assert any('500 K' in w for w in result.warnings)


def test_swept_temperature_checks_every_value():
    result = scale('load', 'low', ScaleContext('afm', temperature_K=[300, 400]))
    assert len(result.warnings) == 1 and '400' in result.warnings[0]


@pytest.mark.parametrize('quantity, model', [
    ('pressure', 'sheetonsheet'), ('load', 'sheetonsheet'), ('speed', 'afm'),
    ('temperature', 'afm'), ('load', 'pes_sheet')])
def test_no_measured_regime_returns_none(quantity, model):
    assert scale(quantity, 'low', ScaleContext(model=model)) is None


def test_unknown_descriptor_returns_none():
    assert scale('load', 'gigantic', AFM) is None
