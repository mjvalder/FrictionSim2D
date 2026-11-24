"""Environment diagnostics for the FrictionSim2D GUI.

Answers the questions that decide whether the GUI can actually run here, before
any of it is built: is there a usable Qt platform, does ovito's interactive
viewport widget get a real OpenGL context, and are the external build tools on
PATH.

Imported only by ``FrictionSim2D gui --check`` -- it pulls in Qt and ovito.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import List, Optional


#: Qt platform plugins that cannot present a rendered frame to a user.
NON_INTERACTIVE_PLATFORMS = frozenset({"offscreen", "minimal", "minimalegl", "vnc"})


@dataclass
class Check:
    """One diagnostic line."""
    name: str
    ok: bool
    detail: str = ""

    def render(self) -> str:
        mark = "OK  " if self.ok else "FAIL"
        line = f"  [{mark}] {self.name}"
        if self.detail:
            line += f": {self.detail}"
        return line


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def render(self) -> str:
        lines = ["FrictionSim2D GUI environment check", ""]
        lines += [c.render() for c in self.checks]
        if self.notes:
            lines += ["", "Notes:"]
            lines += [f"  - {n}" for n in self.notes]
        lines += ["", "Result: " + ("ready" if self.ok else "not ready")]
        return "\n".join(lines)


def _python_env(report: Report) -> None:
    report.add("python", True, sys.version.split()[0])
    report.add("conda env", True, os.environ.get("CONDA_DEFAULT_ENV", "(none)"))


def _display(report: Report) -> None:
    display = os.environ.get("DISPLAY")
    wayland = os.environ.get("WAYLAND_DISPLAY")
    platform_hint = os.environ.get("QT_QPA_PLATFORM", "(auto)")
    has_display = bool(display or wayland)
    detail = f"DISPLAY={display or '-'} WAYLAND_DISPLAY={wayland or '-'} QT_QPA_PLATFORM={platform_hint}"
    report.add("display", has_display, detail)
    if not has_display:
        report.notes.append(
            "No display detected. The GUI needs a graphical session; "
            "set QT_QPA_PLATFORM=offscreen only for tests."
        )


def _external_tools(report: Report) -> None:
    for tool in ("lmp", "atomsk"):
        path = shutil.which(tool)
        report.add(f"{tool} on PATH", path is not None, path or "not found")
    if shutil.which("lmp") is None:
        report.notes.append(
            "lmp not found -- the active conda environment is probably not 'aiida'."
        )


def _ovito(report: Report) -> Optional[object]:
    try:
        import ovito  # noqa: PLC0415
    except Exception as exc:  # pylint: disable=broad-except
        report.add("ovito import", False, f"{type(exc).__name__}: {exc}")
        return None

    version = getattr(ovito, "version_string", "unknown")
    report.add("ovito import", True, version)

    # ovito installed from PyPI into a conda environment ships its own Qt build.
    # That works, but a second PySide6 (e.g. from conda) will conflict.
    module_path = getattr(ovito, "__file__", "") or ""
    if "site-packages" in module_path and "conda" in module_path:
        report.notes.append(
            "ovito is installed into a conda environment. This is the supported "
            "configuration for this project as long as PySide6 comes from ovito's "
            "own wheel. Do not 'conda install pyside6' alongside it."
        )
    return ovito


def _qt(report: Report) -> Optional[object]:
    try:
        from PySide6 import QtCore  # noqa: PLC0415
    except Exception as exc:  # pylint: disable=broad-except
        report.add("PySide6 import", False, f"{type(exc).__name__}: {exc}")
        return None
    report.add("PySide6 import", True, QtCore.__version__)
    return QtCore


def _opengl_viewport(report: Report) -> None:
    """The load-bearing check: does ovito's Qt viewport get a real GL context."""
    try:
        from PySide6.QtWidgets import QApplication  # noqa: PLC0415
        import ovito.vis  # noqa: PLC0415
    except Exception as exc:  # pylint: disable=broad-except
        report.add("viewport widget", False, f"{type(exc).__name__}: {exc}")
        return

    app = QApplication.instance() or QApplication([])
    platform = app.platformName()
    interactive_platform = platform not in NON_INTERACTIVE_PLATFORMS
    report.add(
        "qt platform",
        interactive_platform,
        platform if interactive_platform else f"{platform} (cannot render interactively)",
    )

    try:
        viewport = ovito.vis.Viewport(type=ovito.vis.Viewport.Type.Perspective)
        widget = viewport.create_qt_widget()
    except Exception as exc:  # pylint: disable=broad-except
        report.add("viewport widget", False, f"{type(exc).__name__}: {exc}")
        return

    # PySide reports the widget's Python type as plain QWidget, because ovito's
    # C++ viewport class has no registered Python wrapper.  The Qt meta-object
    # carries the real class.
    meta = widget.metaObject()
    chain = [meta.className()]
    parent = meta.superClass()
    while parent:
        chain.append(parent.className())
        parent = parent.superClass()
    report.add("viewport widget", "QOpenGLWidget" in chain, " <- ".join(chain))

    # A GL context alone is not enough: under the offscreen platform Qt still
    # hands out a valid context but then reports "QOpenGLWidget: No fbo, cannot
    # render".  Interactive rendering needs both a context and a real platform.
    context_ok = False
    try:
        from PySide6.QtGui import QOpenGLContext  # noqa: PLC0415
        ctx = QOpenGLContext()
        context_ok = bool(ctx.create() and ctx.isValid())
    except Exception as exc:  # pylint: disable=broad-except
        report.add("opengl context", False, f"{type(exc).__name__}: {exc}")
        return

    is_gl = context_ok and interactive_platform
    if is_gl:
        detail = ""
    elif not context_ok:
        detail = "no GL context available"
    else:
        detail = f"context ok but platform '{platform}' cannot present it"
    report.add("opengl context", is_gl, detail)

    if not is_gl:
        report.notes.append(
            "Interactive rotation is unavailable here. Under "
            "QT_QPA_PLATFORM=offscreen this is expected and only affects tests; "
            "in a real session it means the viewport would fall back to static "
            "rendered images."
        )
    del app  # keep the reference alive until here


def run_checks() -> Report:
    """Run every diagnostic and return the report."""
    report = Report()
    _python_env(report)
    _display(report)
    _external_tools(report)
    _ovito(report)
    _qt(report)
    _opengl_viewport(report)
    return report
