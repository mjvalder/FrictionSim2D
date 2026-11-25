"""Field-level comparison of parameter blocks (aiida-free)."""
from frictionsim2d.core.compare import VARIES, CompareReport, compare_blocks, merge_varying

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


def test_render_lists_blocks_in_name_order_and_names_the_identical_ones():
    """Two differing blocks given out of order, one identical, one one-sided."""
    a = {'zeta': {'x': 1}, 'alpha': {'x': 1}, 'same': {'x': 5}, 'only_a': {}}
    b = {'zeta': {'x': 2}, 'alpha': {'x': 2}, 'same': {'x': 5}}

    lines = compare_blocks(a, b, units={}).render().splitlines()

    assert lines == ['alpha:', "  x: 1 -> 2", 'zeta:', "  x: 1 -> 2",
                     'only in A: only_a', 'identical blocks: same']


def test_render_sorts_a_report_built_out_of_order():
    """`render` orders blocks by name whatever order the report was filled in."""
    report = CompareReport(differs={'zeta': [('x', 1, 2, '')], 'alpha': [('x', 1, 2, '')]})

    assert report.render().splitlines() == ['alpha:', '  x: 1 -> 2', 'zeta:', '  x: 1 -> 2']


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
    one = ({'thermostat': {'general.temp': 300.0, 'x': 1}}, {'geometry.lat_c': 3.19}, {'x': 1})
    two = ({'thermostat': {'general.temp': 400.0}}, {'geometry.lat_c': 3.19}, {'x': 1})
    blocks, derived = merge_varying([one, two])
    assert blocks['thermostat'] == {'general.temp': VARIES, 'x': 1}
    assert derived == {'geometry.lat_c': 3.19}
    assert compare_blocks(blocks, blocks, units={}).same


def test_units_and_defaults_are_per_side():
    # A stores lengths in angstrom, B in nm: never compared as raw numbers.
    a = {'geometry': {'sheets.0.x': 100.0}}
    b = {'geometry': {'sheets.0.x': 10.0}}
    report = compare_blocks(a, b, units={'sheets.x': 'angstrom'},
                            units_b={'sheets.x': 'nm'})
    assert report.differs == {} and report.identical == []
    assert report.unit_differs == {'geometry': [('sheets.0.x', 'angstrom', 'nm')]}
    assert 'unit differs: A=angstrom B=nm, not compared' in report.render()
    assert not report.same


def test_default_changed_between_versions():
    kinds = {'general.temp': 'thermostat'}
    elided = {'thermostat': {}}
    # same real value 300 K, elided at different recorded defaults: both written out
    a = {'thermostat': {}}
    b = {'thermostat': {'general.temp': 300.0}}
    ok = compare_blocks(a, b, units=UNITS, defaults={'general.temp': 300.0},
                        defaults_b={'general.temp': 250.0}, kinds=kinds)
    assert ok.same
    # A elided at 300, B elided at 250: really different, though neither writes it
    hidden = compare_blocks(elided, elided, units=UNITS, defaults={'general.temp': 300.0},
                            defaults_b={'general.temp': 250.0}, kinds=kinds)
    assert hidden.differs == {'thermostat': [('general.temp', 300.0, 250.0, 'K')]}
    # one shared default: no false difference
    assert compare_blocks(elided, elided, units=UNITS, kinds=kinds,
                          defaults={'general.temp': 300.0}).same


def test_derived_atol_floor():
    base = dict(units=UNITS)
    assert compare_blocks({}, {}, derived_a={'k': 0.0}, derived_b={'k': 1e-12},
                          **base).derived_differs == []
    real = compare_blocks({}, {}, derived_a={'k': 0.0}, derived_b={'k': 1e-6}, **base)
    assert real.derived_differs == [('k', 0.0, 1e-6, '')]
    assert compare_blocks({}, {}, derived_a={'k': 0.0}, derived_b={'k': 1e-6},
                          atol=1e-3, **base).derived_differs == []


def test_derived_unit_mismatch_is_not_compared():
    report = compare_blocks({}, {}, units={'derived.k': 'eV'}, units_b={'derived.k': 'J'},
                            derived_a={'k': 1.0}, derived_b={'k': 9.0})
    assert report.derived_differs == []
    assert report.unit_differs == {'derived': [('k', 'eV', 'J')]}


def test_merge_and_pairwise_agree_on_derived():
    members = [({}, {'k': 0.0}, {}), ({}, {'k': 1e-12}, {})]
    _, derived = merge_varying(members)
    assert derived == {'k': 0.0}
    far = [({}, {'k': 0.0}, {}), ({}, {'k': 1e-6}, {})]
    assert merge_varying(far)[1] == {'k': VARIES}
    assert merge_varying(far, atol=1e-3)[1] == {'k': 0.0}


def test_merge_varying_derived_values():
    members = [({}, {'a': 1.0, 'b': 2.0}, {}), ({}, {'a': 1.0, 'b': 3.0}, {})]
    assert merge_varying(members)[1] == {'a': 1.0, 'b': VARIES}


def test_merge_refills_each_member_from_its_own_defaults():
    kinds = {'general.temp': 'thermostat'}
    members = [({'thermostat': {}}, {}, {'general.temp': 300.0}),
               ({'thermostat': {}}, {}, {'general.temp': 250.0})]
    blocks, _ = merge_varying(members, kinds=kinds)
    assert blocks['thermostat'] == {'general.temp': VARIES}


def test_real_key_matches_compare_tolerance():
    from frictionsim2d.core.compare import real_key, same_value
    assert real_key(0.1 + 0.2) == real_key(0.3)
    assert real_key(1.0) != real_key(1.001)
    assert same_value([0.1 + 0.2, 'x'], [0.3, 'x']) and not same_value(1.0, 1.001)
    report = compare_blocks({'integration': {'settings.simulation.timestep': 0.1 + 0.2}},
                            {'integration': {'settings.simulation.timestep': 0.3}},
                            units={})
    assert report.identical == ['integration'] and report.differs == {}


def test_integers_compare_exactly_floats_to_tolerance():
    from frictionsim2d.core.compare import canonical, canonical_text, same_value
    assert not same_value(12345678, 12345679)
    assert same_value(12345678.0, 12345679.0)          # floats, even integral
    assert same_value(10000000, 1e7)                   # the float keys as that int
    assert canonical_text(10000000) == canonical_text(1e7)
    assert not same_value(12345678, 12345678.0)        # 12345678.0 keys as 12345700.0
    assert canonical(True) is True and canonical([3, 2.5]) == [3, 2.5]
    report = compare_blocks({'realization': {'indent.seed': 12345678}},
                            {'realization': {'indent.seed': 12345679}}, units={})
    assert report.differs['realization'] == [('indent.seed', 12345678, 12345679, '')]
    merged, _ = merge_varying([({'b': {'n': 12345678}}, {}, {}),
                               ({'b': {'n': 12345679}}, {}, {})])
    assert merged['b']['n'] == VARIES


def test_refill_writes_back_elided_fields_but_not_indexed_ones():
    from frictionsim2d.core.compare import refill
    defaults = {'general.temp': 300.0, 'general.speed': 2.0, 'sheets.x': 50.0}
    kinds = {'general.temp': 'general', 'general.speed': 'general', 'sheets.x': 'general'}
    bare = {'general.temp': 400.0, 'sheets.0.x': 60.0}
    out = refill('general', bare, defaults, kinds)
    assert out == {'general.temp': 400.0, 'sheets.0.x': 60.0, 'general.speed': 2.0}
    assert refill('general', {'kind': 'general', 'fields': bare}, defaults, kinds) == out
    assert refill('general', {}, defaults, kinds) == {'general.temp': 300.0, 'general.speed': 2.0}
