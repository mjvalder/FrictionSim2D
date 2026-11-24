"""The environment check reports the external binary the GUI actually needs."""
from frictionsim2d.gui import diagnostics


def test_external_tools_check_only_lmp(monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: f"/bin/{name}")
    report = diagnostics.Report()
    diagnostics._external_tools(report)
    assert [c.name for c in report.checks] == ["lmp on PATH"]
    assert report.ok


def test_missing_lmp_fails_the_check_with_a_note(monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: None)
    report = diagnostics.Report()
    diagnostics._external_tools(report)
    assert not report.ok
    assert any("lmp not found" in n for n in report.notes)
