"""load_settings() must never read the developer's real ~/.config file.

Twenty test files call load_settings() with no argument. Without isolation
the suite's result depends on whether ~/.config/FrictionSim2D/settings.yaml
exists on the machine running it.
"""
import os
from pathlib import Path

from frictionsim2d.core.config import _global_settings_path, load_settings


def test_xdg_config_home_points_into_a_temp_dir():
    xdg = os.environ.get("XDG_CONFIG_HOME")
    assert xdg, "conftest must set XDG_CONFIG_HOME for the whole session"
    assert Path(xdg) != Path.home() / ".config"
    assert _global_settings_path() == Path(xdg) / "FrictionSim2D" / "settings.yaml"


def test_settings_file_in_isolated_dir_is_honoured():
    target = _global_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text("simulation:\n  timestep: 0.0042\n", encoding="utf-8")
        assert load_settings().simulation.timestep == 0.0042
    finally:
        target.unlink()
    assert load_settings().simulation.timestep != 0.0042
