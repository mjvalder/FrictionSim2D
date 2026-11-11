"""AFM builder fixtures shared across builder, template and provenance tests."""
import re
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
                # 90x90 clears validate_tip_sheet_footprint's 2x-tip-diameter
                # floor (4 * r == 80.0 for the r=20.0 tip below) with margin.
                "x": 90.0,
                "y": 90.0,
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


# --- interlayer-potential-choice Task 5: real builds of the example configs ---

REPO_ROOT = Path(__file__).resolve().parents[2]

_SEED_PATTERNS = (
    (re.compile(r'(langevin [\d.]+ [\d.]+ \$\(100\.0\*dt\) )\d+( zero yes)'), r'\1SEED\2'),
    (re.compile(r'(velocity        system create [\d.]+ )\d+'), r'\1SEED'),
)


def build_example_afm(tmp_path: Path, monkeypatch, ini: str, builder_cls=AFMSimulation,
                      **sections) -> Path:
    """Build ``examples/<ini>`` with per-section overrides; return the run dir.

    Default ``GlobalSettings()`` (no user settings file), so the output depends
    only on the code. ``sections`` maps an .ini section (``general``, ``2D``,
    ``tip``, ...) to the keys it overrides.
    """
    from src.core.config import GlobalSettings, read_config  # pylint: disable=import-outside-toplevel
    monkeypatch.chdir(REPO_ROOT)
    raw = read_config(REPO_ROOT / "examples" / ini)
    for section, values in sections.items():
        raw.setdefault(section, {}).update(values)
    settings = GlobalSettings()
    settings.hpc.modules = ["lammps"]        # gates HPC generation, which build() runs
    config = AFMSimulationConfig(**raw, settings=settings)
    out = tmp_path / "out"
    builder_cls(config, output_dir=str(out)).build()
    return out


def normalised_lammps_files(out: Path) -> dict:
    """``{L<n>/lammps/<name>: text}``; run dir -> <OUT>, per-build seeds -> SEED."""
    files = {}
    for p in sorted(out.glob("L*/lammps/*")):
        text = p.read_text(encoding="utf-8").replace(str(out), "<OUT>")
        for pattern, repl in _SEED_PATTERNS:
            text = pattern.sub(repl, text)
        files[str(p.relative_to(out))] = text
    return files
