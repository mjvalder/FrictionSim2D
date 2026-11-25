"""The core install (no optional extras) imports and serves --help.

Runs in a subprocess with every optional top-level package blocked, so a
module-level import of an extra fails here and not on a user's bare install.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_BLOCKED = ["aiida", "PySide6", "ovito", "pydantic_ai", "matplotlib", "huggingface_hub",
            "seaborn", "openai", "anthropic", "google"]

_DRIVER = textwrap.dedent("""
    import importlib.abc, sys
    BLOCKED = {blocked!r}

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError("blocked optional dependency: " + name)
            return None

    sys.meta_path.insert(0, _Block())
    for _m in [m for m in sys.modules if m.split(".")[0] in BLOCKED]:
        del sys.modules[_m]
    args = {args!r}
    if args is None:
        import frictionsim2d
    else:
        from frictionsim2d.cli import main
        sys.argv = ["FrictionSim2D"] + args
        main()
""")


@pytest.mark.parametrize("args", [
    None,
    ["--help"],
    ["run", "afm", "--help"],
    ["run", "quench", "--help"],
    ["fetch", "--help"],
], ids=["import", "help", "run-afm", "run-quench", "fetch"])
def test_core_install_without_extras(args):
    code = _DRIVER.format(blocked=_BLOCKED, args=args)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, check=False, timeout=120)
    # click exits 0 via SystemExit(0) after --help
    assert result.returncode == 0, result.stderr[-2000:]


def test_core_install_builds_a_deck(tmp_path):
    """A real build, not just --help: the builder path must not reach an extra."""
    out = tmp_path / "out"
    code = _DRIVER.format(blocked=_BLOCKED, args=[
        "run", "sheetonsheet", "examples/sheetonsheet/basic.ini", "--output-dir", str(out)])
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=False, timeout=600, cwd=REPO)
    assert result.returncode == 0, result.stderr[-2000:]
    assert list(out.rglob("*.in")), "no LAMMPS input rendered"
