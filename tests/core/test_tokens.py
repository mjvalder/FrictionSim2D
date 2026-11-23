import pytest

from frictionsim2d.core.utils.tokens import (_is_float, _is_int_token, format_dimension_token,
                                   format_numeric_token)


@pytest.mark.parametrize('value, token', [
    (1.5, '1p5'), (-2.3, 'm2p3'), (100, '100'), (1e-5, '1em05'), (0.0, '0'),
])
def test_format_numeric_token_is_filename_safe(value, token):
    assert format_numeric_token(value) == token


@pytest.mark.parametrize('value, token', [
    (100.0, '100'), ('100', '100'), (1.5, '1.5'), ('12.50', '12.5'),
    (1000000.0, '1000000'),
])
def test_format_dimension_token_drops_integral_decimals(value, token):
    assert format_dimension_token(value) == token


@pytest.mark.parametrize('value', ['abc', None, '', '1,5'])
def test_format_dimension_token_returns_non_numeric_input_unchanged(value):
    assert format_dimension_token(value) == str(value)


@pytest.mark.parametrize('s, ok', [('1', True), ('-3.2e+4', True), ('', False), ('nan', True), ('1,5', False)])
def test_is_float(s, ok):
    assert _is_float(s) is ok


@pytest.mark.parametrize('s, ok', [('7', True), ('+7', True), ('-0', True), ('7.0', False), ('', False)])
def test_is_int_token(s, ok):
    assert _is_int_token(s) is ok
