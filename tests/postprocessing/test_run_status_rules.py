"""Direct tests of the run-status rules the DataReader and the store share."""
from types import SimpleNamespace

from frictionsim2d.postprocessing.run_status import (HALT_RECORD, UNWINDOWED_FAMILIES,
                                                     legacy_truncated, read_halt_registry)


def _point(rows, family='afm', size='100x100y', expected=False, status='complete'):
    return (family, size, rows, expected, status)


def test_legacy_truncated_needs_more_than_three_samples_short():
    points = [_point(1000), _point(997), _point(996)]

    assert legacy_truncated(points) == [2]


def test_legacy_truncated_judges_each_family_and_size_alone():
    points = [_point(1000), _point(10, size='50x50y'), _point(10, family='sheetonsheet')]

    assert legacy_truncated(points) == []


def test_legacy_truncated_skips_expected_rows_non_complete_and_unwindowed():
    unwindowed = next(iter(UNWINDOWED_FAMILIES))
    points = [_point(1000), _point(1, expected=True), _point(1, status='failed_integrity'),
              _point(1, family=unwindowed)]

    assert legacy_truncated(points) == []


def test_read_halt_registry_skips_lines_that_are_not_halts_and_tokens_without_equals(tmp_path):
    (tmp_path / HALT_RECORD).write_text(
        '# HALT stage=slide load=5 angle=0 stray\nnot a halt line\n# HALT stage=system\n',
        encoding='utf-8')
    locator = SimpleNamespace(results_dir=tmp_path)

    assert read_halt_registry(locator) == [
        {'stage': 'slide', 'load': '5', 'angle': '0'}, {'stage': 'system'}]


def test_read_halt_registry_without_a_file_is_empty(tmp_path):
    assert read_halt_registry(SimpleNamespace(results_dir=tmp_path)) == []
