"""Field-level comparison of parameter blocks (aiida-free)."""
from src.core.compare import VARIES, compare_blocks, merge_varying

UNITS = {'general.temp': 'K', 'derived.geometry.lat_c': 'angstrom',
         'sheets.x': 'angstrom'}


def _blocks(temp=300.0, sha='aa'):
    return {'thermostat': {'kind': 'thermostat', 'block_schema': '1.0.0',
                           'fields': {'general.temp': temp}},
            'potential__sheet_0': {'pot_file_sha256': sha}}


def test_differing_fields_carry_units():
    report = compare_blocks(_blocks(300.0, 'aa'), _blocks(400.0, 'bb'), units=UNITS)
    assert report.differs['thermostat'] == [('general.temp', 300.0, 400.0, 'K')]
    assert report.differs['potential__sheet_0'] == [('pot_file_sha256', 'aa', 'bb', '')]
    text = report.render()
    assert 'thermostat' in text and 'temp' in text and 'K' in text


def test_identical_sets():
    report = compare_blocks(_blocks(), _blocks(), units=UNITS)
    assert report.differs == {} and report.same
    assert sorted(report.identical) == ['potential__sheet_0', 'thermostat']


def test_derived_tolerance_and_unit():
    close = compare_blocks({}, {}, units=UNITS, derived_a={'geometry.lat_c': 3.1905},
                           derived_b={'geometry.lat_c': 3.1905000001})
    assert close.derived_differs == []
    far = compare_blocks({}, {}, units=UNITS, derived_a={'geometry.lat_c': 3.19},
                         derived_b={'geometry.lat_c': 3.21})
    assert far.derived_differs == [('geometry.lat_c', 3.19, 3.21, 'angstrom')]


def test_elided_field_equals_default_including_indexed_paths():
    a = {'thermostat': {'kind': 'thermostat', 'fields': {}}}
    b = {'thermostat': {'kind': 'thermostat', 'fields': {'general.temp': 300.0}}}
    assert compare_blocks(a, b, units=UNITS, defaults={'general.temp': 300.0}).same
    c = {'geometry': {'sheets.0.x': 100.0}}
    d = {'geometry': {}}
    assert compare_blocks(c, d, units=UNITS, defaults={'sheets.x': 100.0}).same
    report = compare_blocks(c, d, units=UNITS, defaults={'sheets.x': 90.0})
    assert report.differs['geometry'] == [('sheets.0.x', 100.0, 90.0, 'angstrom')]


def test_blocks_on_one_side_only():
    report = compare_blocks({'a': {}, 'c': {}}, {'b': {}, 'c': {}}, units={})
    assert (report.only_a, report.only_b, report.identical) == (['a'], ['b'], ['c'])
    assert 'only in A: a' in report.render()


def test_merge_varying_marks_fields_that_differ():
    one = ({'thermostat': {'general.temp': 300.0, 'x': 1}}, {'geometry.lat_c': 3.19})
    two = ({'thermostat': {'general.temp': 400.0}}, {'geometry.lat_c': 3.19})
    blocks, derived = merge_varying([one, two], defaults={'x': 1})
    assert blocks['thermostat'] == {'general.temp': VARIES, 'x': 1}
    assert derived == {'geometry.lat_c': 3.19}
    assert compare_blocks(blocks, blocks, units={}).same
