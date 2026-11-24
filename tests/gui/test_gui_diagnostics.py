"""The environment check requires lmp; atomsk is optional and never fails it."""
from frictionsim2d.gui import diagnostics


def test_missing_atomsk_does_not_fail_the_check(monkeypatch):
    monkeypatch.setattr(
        diagnostics.shutil, "which", lambda name: None if name == "atomsk" else f"/bin/{name}"
    )
    report = diagnostics.Report()
    diagnostics._external_tools(report)
    assert report.ok
    assert any("atomsk" in c.name for c in report.checks)


def test_missing_lmp_fails_the_check_with_a_note(monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: None)
    report = diagnostics.Report()
    diagnostics._external_tools(report)
    assert not report.ok
    assert any("lmp not found" in n for n in report.notes)
