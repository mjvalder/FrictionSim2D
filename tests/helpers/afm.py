"""AFM builder fixtures shared across builder, template and provenance tests."""
from pathlib import Path
from types import SimpleNamespace

from src.builders.afm import AFMSimulation
from src.core.config import AFMSimulationConfig, load_settings

# --- bodies moved verbatim from tests/builders/test_afm_builder.py ---


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
                "amorph": "a",
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


def _seed_builder_state(builder: AFMSimulation, tmp_path: Path) -> Path:
    """Seed builder state so write_inputs can render slide.in in isolation.

    Does not call write_inputs -- callers that need to assert on a raised
    error wrap that call themselves; callers that just want the rendered
    slide.in should use :func:`_prepare_builder_for_slide_write` instead.
    """
    output_dir = tmp_path / "out"
    layer_dir = output_dir / "L1"

    builder.output_dir = output_dir
    builder.relative_run_dir = Path("out")
    builder.output_dir_layer[1] = layer_dir
    builder.relative_run_dir_layer[1] = Path("out") / "L1"

    builder.sheet_paths[1] = tmp_path / "sheet.lmp"
    builder.tip_path = tmp_path / "tip.lmp"
    builder.sub_path = tmp_path / "sub.lmp"

    builder.z_positions[1] = {"sub": 0.0, "sheet": 16.0, "tip": 40.0, "tip_contact_gap": 3.5}
    builder.groups[1] = {
        "sub_types": "1",
        "tip_types": "2",
        "sheet_types": "3",
    }
    class _StubTypes(list):
        """Minimal stand-in for PotentialManager.types: len() + the two
        queries _layer_groups_available needs (sheet type 3, one layer)."""

        def get_layer_group_string(self, component, layer):
            return '3' if (component == 'sheet' and layer == 0) else ''

        def ids_by_component(self, component):
            return [3] if component == 'sheet' else []

    builder.pm[1] = SimpleNamespace(types=_StubTypes([1, 2, 3]))
    builder.lat_c = 6.0
    sheet_dims = {
        "xlo": 0.0,
        "xhi": 100.0,
        "ylo": 0.0,
        "yhi": 100.0,
        "zlo": 0.0,
        "zhi": 12.0,
    }
    builder.sheet_dims = dict(sheet_dims)
    builder.box_dims = dict(sheet_dims)
    builder.sheet_offset = {"x": 0.0, "y": 0.0}

    for path in (builder.sheet_paths[1], builder.tip_path, builder.sub_path):
        path.write_text("# dummy\n", encoding="utf-8")

    return layer_dir


def _prepare_builder_for_slide_write(builder: AFMSimulation, tmp_path: Path) -> Path:
    """Seed builder state and render slide.in."""
    layer_dir = _seed_builder_state(builder, tmp_path)
    builder.write_inputs(1)
    return layer_dir / "lammps" / "slide.in"
