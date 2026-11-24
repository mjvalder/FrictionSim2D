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


def test_run_checks_fails_when_any_probe_fails(monkeypatch):
    """`gui --check` reads `report.ok`: one failed probe makes it not ready."""
    def passing(name):
        return lambda report: report.add(name, True)

    for probe in ('_python_env', '_display', '_external_tools', '_ovito', '_qt'):
        monkeypatch.setattr(diagnostics, probe, passing(probe))
    monkeypatch.setattr(diagnostics, '_opengl_viewport',
                        lambda report: report.add('viewport widget', False, 'no GL'))
    report = diagnostics.run_checks()
    assert not report.ok
    assert 'not ready' in report.render()

    monkeypatch.setattr(diagnostics, '_opengl_viewport', passing('viewport widget'))
    assert diagnostics.run_checks().ok
