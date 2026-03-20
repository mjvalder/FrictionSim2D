"""Tests for SheetOnSheetSimulation builder behavior."""

from pathlib import Path

import pytest

from src.builders.sheetonsheet import SheetOnSheetSimulation
from src.core.config import SheetOnSheetSimulationConfig, load_settings


def _make_sheetonsheet_config(tmp_path: Path, layers: list[int]) -> SheetOnSheetSimulationConfig:
    """Create a minimal valid sheet-on-sheet config for builder tests."""
    pot_path = tmp_path / "dummy.sw"
    cif_path = tmp_path / "dummy.cif"
    pot_path.write_text("# dummy potential", encoding="utf-8")
    cif_path.write_text("# dummy cif", encoding="utf-8")

    settings = load_settings().model_dump()

    return SheetOnSheetSimulationConfig(
        **{
            "general": {
                "temp": 300.0,
            },
            "2D": {
                "mat": "h-MoS2",
                "pot_type": "sw",
                "pot_path": str(pot_path),
                "cif_path": str(cif_path),
                "x": 20.0,
                "y": 20.0,
                "layers": layers,
            },
            "lj_override": {},
            "settings": settings,
        }
    )


def test_n_layers_uses_single_explicit_value(tmp_path: Path) -> None:
    """Builder should use the single configured layer count."""
    config = _make_sheetonsheet_config(tmp_path, layers=[4])

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))

    assert builder.n_layers == 4


def test_n_layers_raises_for_multiple_layer_values(tmp_path: Path) -> None:
    """Builder should fail fast instead of silently collapsing layer lists."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3, 5, 4])
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))

    with pytest.raises(ValueError, match="requires exactly one value in 2D.layers"):
        _ = builder.n_layers


def test_build_raises_when_less_than_three_layers(tmp_path: Path) -> None:
    """Sheet-on-sheet builder intentionally rejects <3-layer configurations."""
    config = _make_sheetonsheet_config(tmp_path, layers=[2])
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))

    with pytest.raises(ValueError, match="requires at least 3 layers"):
        builder.build()


def test_sheetonsheet_build_orchestrates_single_layer_flow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Build should execute the expected high-level flow for one layer value."""
    config = _make_sheetonsheet_config(tmp_path, layers=[4])
    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))

    called = {
        "init": 0,
        "hpc": 0,
        "write_inputs": 0,
        "generate_potentials": 0,
        "build_sheet_kwargs": None,
    }

    monkeypatch.setattr("src.builders.sheetonsheet.SheetOnSheetSimulation._create_directories", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "src.builders.sheetonsheet.SheetOnSheetSimulation._init_provenance",
        lambda *_a, **_k: called.__setitem__("init", called["init"] + 1),
    )
    monkeypatch.setattr(
        "src.builders.sheetonsheet.SheetOnSheetSimulation._generate_hpc_scripts",
        lambda *_a, **_k: called.__setitem__("hpc", called["hpc"] + 1),
    )
    monkeypatch.setattr(
        "src.builders.sheetonsheet.SheetOnSheetSimulation.write_inputs",
        lambda *_a, **_k: called.__setitem__("write_inputs", called["write_inputs"] + 1),
    )
    monkeypatch.setattr(
        "src.builders.sheetonsheet.SheetOnSheetSimulation._generate_potentials",
        lambda *_a, **_k: called.__setitem__("generate_potentials", called["generate_potentials"] + 1) or object(),
    )

    def mock_build_sheet(*_args, **kwargs):
        called["build_sheet_kwargs"] = kwargs
        return (
            tmp_path / "sheet.lmp",
            {"xlo": 0, "xhi": 100, "ylo": 0, "yhi": 100, "zlo": 0, "zhi": 20},
            3.0,
        )

    monkeypatch.setattr("src.builders.components.build_sheet", mock_build_sheet)

    builder.build()

    assert called["init"] == 1
    assert called["generate_potentials"] == 1
    assert called["write_inputs"] == 1
    assert called["hpc"] == 1
    assert called["build_sheet_kwargs"]["n_layers_override"] == 4
    assert builder.structure_paths["sheet"].name == "sheet.lmp"
    assert builder.lat_c == 3.0
    assert builder.z_positions["layer_1"] == 0.0
    assert builder.z_positions["layer_4"] == 9.0


def test_sheetonsheet_rejects_internal_lj_with_constraints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Internal-LJ potentials should be rejected for constrained modes."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    config.sheet.pot_type = "reaxff"
    config.settings.simulation.constraint_mode = "atom_bonds"

    builder = SheetOnSheetSimulation(config, output_dir=str(tmp_path / "out"))
    monkeypatch.setattr("src.builders.sheetonsheet.SheetOnSheetSimulation._create_directories", lambda *_a, **_k: None)
    monkeypatch.setattr("src.builders.sheetonsheet.SheetOnSheetSimulation._init_provenance", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="does not support external LJ"):
        builder.build()


def test_sheetonsheet_collect_simulation_paths(tmp_path: Path) -> None:
    """Collect path should report a single simulation when lammps dir exists."""
    config = _make_sheetonsheet_config(tmp_path, layers=[3])
    out_dir = tmp_path / "out"
    builder = SheetOnSheetSimulation(config, output_dir=str(out_dir))

    assert builder._collect_simulation_paths() == []

    (out_dir / "lammps").mkdir(parents=True, exist_ok=True)
    assert builder._collect_simulation_paths() == ["."]
