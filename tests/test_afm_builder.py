"""Tests for AFMSimulation builder layer sweep behavior."""

from pathlib import Path

import pytest

from src.builders.afm import AFMSimulation
from src.core.config import AFMSimulationConfig, load_settings


def _make_afm_config(tmp_path: Path, layers: list[int]) -> AFMSimulationConfig:
    """Create a minimal valid AFM config for builder tests."""
    pot_path = tmp_path / "dummy.sw"
    cif_path = tmp_path / "dummy.cif"
    pot_path.write_text("# dummy potential", encoding="utf-8")
    cif_path.write_text("# dummy cif", encoding="utf-8")

    settings = load_settings().model_dump()

    return AFMSimulationConfig(
        **{
            "general": {
                "temp": 300.0,
                "force": 10.0,
                "scan_speed": 2.0,
            },
            "tip": {
                "mat": "Si",
                "pot_type": "sw",
                "pot_path": str(pot_path),
                "cif_path": str(cif_path),
                "r": 20.0,
                "dspring": 0.1,
                "s": 1.0,
            },
            "sub": {
                "mat": "Si",
                "pot_type": "sw",
                "pot_path": str(pot_path),
                "cif_path": str(cif_path),
                "thickness": 10.0,
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


def test_afm_builder_supports_layer_sweep(tmp_path: Path) -> None:
    """AFM builder should accept multiple layer values (unlike sheet-on-sheet)."""
    config = _make_afm_config(tmp_path, layers=[1, 2, 3])

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    # Should not raise; multiple layers are intentional for AFM
    assert config.sheet.layers == [1, 2, 3]


def test_afm_builder_layer_specific_paths(tmp_path: Path, monkeypatch) -> None:
    """AFM should create L{n} directories for each layer count."""
    # Mock out actual file operations to avoid needing real files
    monkeypatch.setattr(
        "src.builders.components.build_sheet",
        lambda *args, **kwargs: (tmp_path / "sheet.lmp", {"xlo": 0, "xhi": 100, "ylo": 0, "yhi": 100, "zhi": 15}, 3.0)
    )
    monkeypatch.setattr(
        "src.builders.components.build_tip",
        lambda *args, **kwargs: (tmp_path / "tip.lmp", 20.0)
    )
    monkeypatch.setattr(
        "src.builders.components.build_substrate",
        lambda *args, **kwargs: tmp_path / "sub.lmp"
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._create_directories",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._init_provenance",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation.write_inputs",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._generate_hpc_scripts",
        lambda *args, **kwargs: None
    )

    config = _make_afm_config(tmp_path, layers=[2, 3])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))

    # Verify internal layer storage before build
    assert builder.config.sheet.layers == [2, 3]


def test_afm_normalized_pot_type_consistency() -> None:
    """AFM should normalize pot_type consistently for checks."""
    test_cases = [
        ("SW", "sw"),
        ("Tersoff", "tersoff"),
        ("ReaxFF", "reaxff"),
        ("REAX/C", "reax/c"),
        ("  sw  ", "sw"),
    ]

    for input_val, expected in test_cases:
        assert AFMSimulation._normalized_pot_type(input_val) == expected


def test_afm_build_orchestrates_per_layer_steps(tmp_path: Path, monkeypatch) -> None:
    """Build should execute per-layer orchestration for all requested layers."""
    config = _make_afm_config(tmp_path, layers=[2, 4])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))

    called = {
        "build_sheet_layers": [],
        "create_dirs": [],
        "build_components": 0,
        "generate_potentials": [],
        "z_positions": [],
        "write_inputs": [],
        "hpc": 0,
    }

    def mock_build_sheet(*_args, **kwargs):
        called["build_sheet_layers"].append(kwargs["n_layers_override"])
        return (
            tmp_path / f"sheet_L{kwargs['n_layers_override']}.lmp",
            {"xlo": 0, "xhi": 100, "ylo": 0, "yhi": 100, "zlo": 0, "zhi": 20},
            3.0,
        )

    monkeypatch.setattr("src.builders.components.build_sheet", mock_build_sheet)
    monkeypatch.setattr("src.builders.afm.AFMSimulation._init_provenance", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._create_directories",
        lambda _self, out_dir=None: called["create_dirs"].append(out_dir),
    )

    def mock_build_components(_self, _build_dir):
        called["build_components"] += 1
        return tmp_path / "tip.lmp", 20.0, tmp_path / "sub.lmp"

    monkeypatch.setattr("src.builders.afm.AFMSimulation._build_components", mock_build_components)
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._generate_potentials",
        lambda _self, n: called["generate_potentials"].append(n) or object(),
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._calculate_z_positions",
        lambda _self, n, _r: called["z_positions"].append(n),
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation.write_inputs",
        lambda _self, n: called["write_inputs"].append(n),
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._generate_hpc_scripts",
        lambda *_a, **_k: called.__setitem__("hpc", called["hpc"] + 1),
    )

    builder.build()

    assert called["build_sheet_layers"] == [2, 4]
    assert called["build_components"] == 1
    assert called["generate_potentials"] == [2, 4]
    assert called["z_positions"] == [2, 4]
    assert called["write_inputs"] == [2, 4]
    assert called["hpc"] == 1
    assert sorted(builder.output_dir_layer.keys()) == [2, 4]


def test_afm_collect_simulation_paths_lists_layer_dirs(tmp_path: Path) -> None:
    """Collect only layer directories that contain a lammps folder."""
    config = _make_afm_config(tmp_path, layers=[1])
    out_dir = tmp_path / "out"
    builder = AFMSimulation(config, output_dir=str(out_dir))

    (out_dir / "L1" / "lammps").mkdir(parents=True, exist_ok=True)
    (out_dir / "L3" / "lammps").mkdir(parents=True, exist_ok=True)
    (out_dir / "misc").mkdir(parents=True, exist_ok=True)

    assert builder._collect_simulation_paths() == ["L1", "L3"]
