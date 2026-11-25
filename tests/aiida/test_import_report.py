"""`ImportReport.clean()` is the rule behind `aiida import`'s exit status."""
import pytest

from frictionsim2d.aiida.ingest import ImportReport

pytestmark = pytest.mark.aiida


@pytest.mark.parametrize('failures, unparseable, dry_run, clean', [
    ([], [], False, True),
    ([('a', 'x')], [], False, False),
    ([], ['afm/d'], False, False),
    ([], ['afm/d'], True, True),
])
def test_import_report_clean_is_the_cli_exit_rule(failures, unparseable, dry_run, clean):
    report = ImportReport(label='x', failures=failures, unparseable=unparseable,
                          dry_run=dry_run)
    assert report.clean() is clean
