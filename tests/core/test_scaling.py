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


def test_working_says_what_each_band_is():
    low = scale('load', 'low', AFM).working
    assert 'baseline loads' in low and 'BASELINE_LOADS' in low and 'poor contact' in low
    assert 'lower part' in low
    mid = scale('load', 'mid', AFM).working
    assert 'MID_LOAD_CORE' in mid and 'reference trends' in mid
    high = scale('load', 'high', AFM).working
    assert '100 nN' in high and 'no dedicated constant' in high
    for text in (low, mid, high):
        assert 'measured' not in text.lower() and '\n' not in text


def test_every_regime_has_a_source_and_none_claims_measurement():
    assert REGIMES
    for regime in REGIMES:
        assert regime.source and regime.low < regime.high
        assert 'measured' not in regime.source.lower()


def test_module_docstring_disclaims_measurement():
    import src.core.scaling as module
    assert 'not measured regimes' in module.__doc__
    assert 'not a contact model' in module.__doc__


@pytest.mark.parametrize('context', [
    AFM,
    ScaleContext('afm', tip_radius_A=25.4, driving_spring_N_m=50.3, temperature_K=300.8),
    ScaleContext('afm'),
    ScaleContext('afm', driving_spring_N_m=[8, 50]),
])
def test_reference_setup_no_warning(context):
    assert scale('load', 'low', context).warnings == ()


def test_tip_radius_differs_warns_same_numbers():
    result = scale('load', 'low', ScaleContext('afm', tip_radius_A=22))
    assert (result.low, result.high) == (10, 20)
    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert 'reference setup is' in warning and 'R = 25' in warning
    assert 'tip radius' in warning and '22' in warning and 'starting point' in warning


def test_spring_and_temperature_differ_warn():
    result = scale('load', 'mid', ScaleContext('afm', driving_spring_N_m=20, temperature_K=310))
    assert len(result.warnings) == 2
    assert any('driving spring' in w and '20 N/m' in w for w in result.warnings)
    assert any('temperature' in w and '310 K' in w for w in result.warnings)


def test_swept_temperature_checks_every_value():
    result = scale('load', 'low', ScaleContext('afm', temperature_K=[300, 400]))
    assert len(result.warnings) == 1 and '400' in result.warnings[0]
    assert '300,' not in result.warnings[0].split('differs')[0].split('your')[-1]


@pytest.mark.parametrize('quantity, model', [
    ('pressure', 'sheetonsheet'), ('load', 'sheetonsheet'), ('speed', 'afm'),
    ('temperature', 'afm'), ('load', 'pes_sheet')])
def test_no_measured_regime_returns_none(quantity, model):
    assert scale(quantity, 'low', ScaleContext(model=model)) is None


def test_unknown_descriptor_returns_none():
    assert scale('load', 'gigantic', AFM) is None
