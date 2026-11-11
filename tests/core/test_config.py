import os
import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError
from src.core.config import (
    AFMSimulationConfig,
    FlakeConfig,
    PESConfig,
    SheetOnSheetSimulationConfig,
    TipConfig,
    SheetConfig,
    SubstrateConfig,
    GeneralConfig,
    GlobalSettings,
    parse_config,
    load_settings,
    settings_origin,
)
from tests.helpers.afm import _make_afm_config

# Sample data
VALID_TIP = {
    'mat': 'Si', 'pot_type': 'tersoff', 'pot_path': 'potentials/Si.tersoff',
    'cif_path': 'Si.cif', 'r': 10.0, 'amorph': 'c', 'cspring': 10.0, 's': 10.0
}
VALID_SHEET = {
    'mat': 'MoS2', 'pot_type': 'sw', 'pot_path': 'potentials/MoS2.sw',
    'cif_path': 'MoS2.cif', 'x': 50.0, 'y': 50.0, 'layers': [1]
}
VALID_SUB = {
    'mat': 'Si', 'pot_type': 'tersoff', 'pot_path': 'potentials/Si.tersoff',
    'cif_path': 'Si.cif', 'thickness': 10.0, 'amorph': 'a'
}
VALID_GENERAL = {'temp': 300.0}

def test_load_default_settings():
    """Ensure default settings load correctly from package resources."""
    settings = load_settings()
    assert isinstance(settings, GlobalSettings)
    assert settings.geometry.rigid_tip is not None
    assert settings.thermostat.type in ['langevin', 'nose-hoover']

def test_valid_config_creation(tmp_path):
    """Test creating a full configuration object."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings()

    config = AFMSimulationConfig(
        general=VALID_GENERAL,
        tip={**VALID_TIP, **real_paths},
        sub={**VALID_SUB, **real_paths},
        **{'2D': {**VALID_SHEET, **real_paths}},
        settings=settings
    )
    assert config.tip.r == 10.0
    assert config.sheet.mat == 'MoS2'

def test_validation_error(tmp_path):
    """Ensure invalid types raise errors."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings()
    invalid_tip = VALID_TIP.copy()
    invalid_tip.update(real_paths)
    invalid_tip['r'] = "not_a_number" # Should fail

    with pytest.raises(ValidationError):
        AFMSimulationConfig(
            general=VALID_GENERAL,
            tip=invalid_tip,
            sub={**VALID_SUB, **real_paths},
            **{'2D': {**VALID_SHEET, **real_paths}},
            settings=settings
        )

def test_parse_config_ini(temp_dir):
    """Test parsing an INI file."""
    ini_content = """
[general]
temp = 300
[tip]
r = 10.0
"""
    ini_file = temp_dir / "test.ini"
    ini_file.write_text(ini_content)
    
    data = parse_config(ini_file)
    assert data['general']['temp'] == 300 # int conversion check
    assert data['tip']['r'] == 10.0 # float conversion check

def test_parse_config_yaml(temp_dir):
    """Test parsing a YAML file."""
    yaml_content = {'general': {'temp': 300}}
    yaml_file = temp_dir / "test.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)
        
    data = parse_config(yaml_file)
    assert data['general']['temp'] == 300


def test_sheetonsheet_config_accepts_explicit_scan_angle_list(tmp_path):
    """Sheet-on-sheet config should accept explicit numeric angle lists."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")

    settings = load_settings()

    config = SheetOnSheetSimulationConfig(
        general={
            "temp": 300.0,
            "pressure": [0.1, 1.0],
            "scan_angle": [0, 45, 90],
            "scan_speed": 1.0,
            "scan_angle_force": [0.1, 1.0],
        },
        **{
            '2D': {
                'mat': 'MoS2',
                'pot_type': 'sw',
                'pot_path': str(dummy_pot),
                'cif_path': str(dummy_cif),
                'x': 50.0,
                'y': 50.0,
                'layers': [3],
            }
        },
        settings=settings,
    )

    assert config.general.scan_angle == [0.0, 45.0, 90.0]
    assert config.general.scan_angle_force == [0.1, 1.0]


def test_general_config_rejects_invalid_scan_angle_string() -> None:
    """scan_angle now only accepts numeric values."""
    with pytest.raises(ValidationError):
        GeneralConfig(scan_angle=[0, 90, 'all'])


def test_tip_config_allows_explicit_tip_coordinates_with_offsets(tmp_path) -> None:
    """Tip placement fields should belong to [tip] and allow explicit+offset inputs."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings()
    tip = dict(VALID_TIP)
    tip.update(real_paths)
    tip.update({"tip_x": 10.0, "tip_y": 5.0, "tip_x_offset": 1.0, "tip_y_offset": -0.5})

    cfg = AFMSimulationConfig(
        general=VALID_GENERAL,
        tip=tip,
        sub={**VALID_SUB, **real_paths},
        **{'2D': {**VALID_SHEET, **real_paths}},
        settings=settings,
    )
    assert cfg.tip.tip_x == 10.0
    assert cfg.tip.tip_y == 5.0
    assert cfg.tip.tip_x_offset == 1.0
    assert cfg.tip.tip_y_offset == -0.5


def test_finite_sheet_requires_substrate_xy(tmp_path):
    """Finite-sheet AFM should require explicit substrate x/y in [sub]."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings()
    with pytest.raises(ValidationError, match="requires substrate x and y"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": True},
            tip={**VALID_TIP, **real_paths},
            sub={**VALID_SUB, **real_paths, "amorph": "c"},
            **{'2D': {**VALID_SHEET, **real_paths}},
            settings=settings,
        )


def test_finite_sheet_rejects_substrate_smaller_than_sheet(tmp_path):
    """Finite-sheet AFM should reject substrate sizes smaller than the sheet."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings()
    bad_sub = dict(VALID_SUB)
    bad_sub.update(real_paths)
    bad_sub.update({"x": 40.0, "y": 60.0, "amorph": "c"})
    with pytest.raises(ValidationError, match="cannot be smaller than sheet x"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": True},
            tip={**VALID_TIP, **real_paths},
            sub=bad_sub,
            **{'2D': {**VALID_SHEET, **real_paths}},
            settings=settings,
        )


def test_finite_sheet_rejects_margin_not_above_lj_cutoff(tmp_path):
    """Finite-sheet AFM should enforce strict substrate-sheet delta > LJ cutoff."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings().model_copy(deep=True)
    settings.potential.lj_cutoff = 11.0
    bad_sub = dict(VALID_SUB)
    bad_sub.update(real_paths)
    bad_sub.update({"x": 61.0, "y": 70.0, "amorph": "c"})
    # sheet x is 50, so x-delta is exactly 11.0 -> invalid (must be > cutoff)
    with pytest.raises(ValidationError, match="greater than LJ cutoff"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": True},
            tip={**VALID_TIP, **real_paths},
            sub=bad_sub,
            **{'2D': {**VALID_SHEET, **real_paths}},
            settings=settings,
        )


def test_non_finite_sheet_rejects_crystalline_substrate(tmp_path):
    """Non-finite-sheet AFM should allow only amorphous substrates."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings()
    with pytest.raises(ValidationError, match="requires an amorphous substrate"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": False},
            tip={**VALID_TIP, **real_paths},
            sub={**VALID_SUB, **real_paths, "amorph": "c"},
            **{'2D': {**VALID_SHEET, **real_paths}},
            settings=settings,
        )


# --- D-1: AFM sheet footprint vs. tip diameter (afm-tip-box-sizing rule) ---

def test_tip_sheet_footprint_rejects_undersized_sheet(tmp_path):
    """A periodic AFM sheet smaller than 2x the tip diameter must be refused,
    naming both the footprint and the required minimum in the message."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings()
    small_sheet = {**VALID_SHEET, **real_paths, "x": 30.0, "y": 30.0}
    with pytest.raises(ValidationError, match=r"30\.0 x 30\.0.*40\.0"):
        AFMSimulationConfig(
            general=VALID_GENERAL,
            tip={**VALID_TIP, **real_paths},  # r = 10.0 -> needs footprint >= 40.0
            sub={**VALID_SUB, **real_paths},
            **{'2D': small_sheet},
            settings=settings,
        )


def test_tip_sheet_footprint_accepts_exact_two_diameter_boundary(tmp_path):
    """A sheet sized to exactly 2x the tip diameter (the documented floor) passes."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings()
    boundary_sheet = {**VALID_SHEET, **real_paths, "x": 40.0, "y": 40.0}
    cfg = AFMSimulationConfig(
        general=VALID_GENERAL,
        tip={**VALID_TIP, **real_paths},  # r = 10.0 -> floor is exactly 40.0
        sub={**VALID_SUB, **real_paths},
        **{'2D': boundary_sheet},
        settings=settings,
    )
    assert cfg.sheet.x == 40.0 and cfg.sheet.y == 40.0


def test_tip_sheet_footprint_hetero_path_is_unaffected(tmp_path):
    """A hetero (>=2 [2D-N]) config is exempt: it replicates to clear the
    tip-diameter floor instead of being checked against its native size."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings()
    # Each material's own cell (12x12) is far smaller than the r=10.0 tip's
    # 40.0 floor; only HeteroAFMSimulation's in-plane replication clears it,
    # so the config itself must not raise.
    small_sheet = {**VALID_SHEET, **real_paths, "x": 12.0, "y": 12.0}
    cfg = AFMSimulationConfig(
        general=VALID_GENERAL,
        tip={**VALID_TIP, **real_paths},
        sub={**VALID_SUB, **real_paths},
        **{'2D-1': small_sheet, '2D-2': {**small_sheet, "mat": "WS2"}},
        settings=settings,
    )
    assert len(cfg.sheets) == 2


# ---------------------------------------------------------------------------
# PES eval-mode and optional substrate tests
# ---------------------------------------------------------------------------

def _afm_dict(tmp_path, include_sub=True, sub_amorph="a", finite_sheet=False, pes=None):
    pot = tmp_path / "d.sw"; pot.write_text("# pot", encoding="utf-8")
    cif = tmp_path / "d.cif"; cif.write_text("# cif", encoding="utf-8")
    d = {
        "general": {"temp": 300.0, "force": [2.0], "scan_speed": 2.0, "finite_sheet": finite_sheet},
        # 100x100 sits exactly at validate_tip_sheet_footprint's 2x-tip-diameter
        # floor for the r=25.0 tip below (4 * r == 100.0).
        "2D": {"mat": "h-MoS2", "pot_type": "sw", "pot_path": str(pot),
               "cif_path": str(cif), "x": 100.0, "y": 100.0, "layers": [4]},
        "tip": {"mat": "Si", "pot_type": "sw", "pot_path": str(pot),
                "cif_path": str(cif), "r": 25.0, "amorph": "c"},
        "settings": load_settings().model_dump(),
    }
    if include_sub:
        d["sub"] = {"mat": "Si", "pot_type": "sw", "pot_path": str(pot),
                    "cif_path": str(cif), "thickness": 12.0, "amorph": sub_amorph}
    if pes is not None:
        d["pes"] = pes
    return d


def test_pes_eval_mode_defaults_minimize():
    assert PESConfig().eval_mode == "minimize"
    assert PESConfig().md_steps == 2000


def test_pes_eval_mode_md_parses():
    cfg = PESConfig(eval_mode="md", md_steps=500)
    assert cfg.eval_mode == "md"
    assert cfg.md_steps == 500


def test_afm_config_sub_optional(tmp_path):
    cfg = AFMSimulationConfig(**_afm_dict(tmp_path, include_sub=False))
    assert cfg.sub is None


def test_afm_config_finite_sheet_requires_substrate(tmp_path):
    # Finite-sheet mode inherently needs a substrate to rest on; omitting [sub]
    # must fail loudly and clearly, not with a downstream substrate-sizing error.
    with pytest.raises(ValueError, match="finite sheet must rest on a substrate"):
        AFMSimulationConfig(**_afm_dict(tmp_path, include_sub=False, finite_sheet=True))


def test_afm_config_freestanding_skips_amorph_requirement(tmp_path):
    # The amorphous-substrate requirement is skipped specifically because the
    # substrate is ABSENT (sub is None), not blanket-removed: a no-[sub] config
    # is accepted, while the same config WITH a crystalline substrate still raises.
    free = AFMSimulationConfig(**_afm_dict(tmp_path, include_sub=False, finite_sheet=False))
    assert free.sub is None
    with pytest.raises(ValueError, match="amorphous substrate"):
        AFMSimulationConfig(**_afm_dict(tmp_path, include_sub=True, sub_amorph="c", finite_sheet=False))


def test_afm_config_nonamorph_sub_still_rejected(tmp_path):
    with pytest.raises(ValueError, match="amorphous substrate"):
        AFMSimulationConfig(**_afm_dict(tmp_path, include_sub=True, sub_amorph="c", finite_sheet=False))


# ---------------------------------------------------------------------------
# Settings precedence tests
# ---------------------------------------------------------------------------

def _write_settings(path: Path, timestep: float) -> None:
    """Write a minimal settings.yaml with a distinctive timestep value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump({'simulation': {'timestep': timestep}}), encoding='utf-8')


def test_defaults_used_when_no_file_exists(tmp_path, monkeypatch):
    """load_settings returns hardcoded defaults when no settings file exists."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    settings = load_settings()
    assert isinstance(settings, GlobalSettings)
    # Default timestep is 0.001 as defined in SimulationSettings
    assert settings.simulation.timestep == 0.001


def test_global_settings_used_when_no_explicit_file(tmp_path, monkeypatch):
    """load_settings picks up the global settings file when no explicit file is given."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    global_path = tmp_path / 'FrictionSim2D' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)

    settings = load_settings()
    assert settings.simulation.timestep == 0.005


def test_explicit_file_takes_precedence_over_global(tmp_path, monkeypatch):
    """Explicit settings_file argument overrides the global settings file."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    global_path = tmp_path / 'FrictionSim2D' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)

    explicit_path = tmp_path / 'explicit' / 'settings.yaml'
    _write_settings(explicit_path, timestep=0.002)

    settings = load_settings(settings_file=explicit_path)
    assert settings.simulation.timestep == 0.002


def test_env_var_is_not_honoured(tmp_path, monkeypatch):
    """FRICTIONSIM2D_SETTINGS_FILE env var is no longer part of the precedence chain."""
    env_path = tmp_path / 'env' / 'settings.yaml'
    _write_settings(env_path, timestep=0.007)

    # Set the env var but point the global path (via XDG_CONFIG_HOME) at an
    # isolated, empty directory with no settings file in it.
    monkeypatch.setenv('FRICTIONSIM2D_SETTINGS_FILE', str(env_path))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg'))

    # Should fall back to hardcoded defaults, NOT pick up the env var file
    settings = load_settings()
    assert settings.simulation.timestep == 0.001


def test_settings_origin_none_when_no_file(tmp_path, monkeypatch):
    """settings_origin returns None when no settings file exists."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    assert settings_origin() is None


def test_settings_origin_returns_global_path(tmp_path, monkeypatch):
    """settings_origin returns the global path when it exists and no explicit file given."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    global_path = tmp_path / 'FrictionSim2D' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)

    assert settings_origin() == global_path


def test_settings_origin_returns_explicit_path(tmp_path, monkeypatch):
    """settings_origin returns the explicit path when one is provided."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    global_path = tmp_path / 'FrictionSim2D' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)

    explicit_path = tmp_path / 'explicit' / 'settings.yaml'
    _write_settings(explicit_path, timestep=0.002)

    assert settings_origin(settings_file=explicit_path) == explicit_path


def test_lattice_match_section_is_parsed(tmp_path):
    from src.core.config import LatticeMatchConfig
    lm = LatticeMatchConfig()
    assert lm.method == "zur_mcgill"
    assert lm.strain_tol == 0.02
    assert lm.max_supercell == 10
    assert lm.area_tol == 0.10
    # out-of-range values are rejected
    import pytest
    with pytest.raises(Exception):
        LatticeMatchConfig(strain_tol=2.0)   # > 1.0
    with pytest.raises(Exception):
        LatticeMatchConfig(max_supercell=0)  # < 1


def test_lattice_match_area_tol_is_bounded():
    """area_tol must be in [0, 1]; out-of-range values are rejected."""
    from src.core.config import LatticeMatchConfig
    with pytest.raises(ValidationError):
        LatticeMatchConfig(area_tol=2.0)


def test_two_material_sections_fold_into_sheets_list(tmp_path):
    """[2D-1]/[2D-2] sections fold into cfg.sheets in numeric order."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")

    settings = load_settings()
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "alternating"},
        "2D-1": {"mat": "MoS2", "cif_path": str(dummy_cif), "pot_path": str(dummy_pot),
                 "pot_type": "sw", "x": 30, "y": 30, "layers": [1], "stack_type": "AB"},
        "2D-2": {"mat": "WS2", "cif_path": str(dummy_cif), "pot_path": str(dummy_pot),
                 "pot_type": "sw", "x": 30, "y": 30, "layers": [1], "stack_type": "AB"},
    }
    cfg = SheetOnSheetSimulationConfig(**raw, settings=settings)
    assert [s.mat for s in cfg.sheets] == ["MoS2", "WS2"]
    assert cfg.general.hetero_stacking == "alternating"
    # Backward-compatible single-sheet accessor still resolves to the first material.
    assert cfg.sheet.mat == "MoS2"


def test_single_2d_section_still_builds_one_element_sheets_list(tmp_path):
    """A lone [2D] section (no [2D-N]) still builds and yields len(sheets) == 1."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    real_paths = {'pot_path': str(dummy_pot), 'cif_path': str(dummy_cif)}

    settings = load_settings()
    cfg = SheetOnSheetSimulationConfig(
        general=VALID_GENERAL,
        **{'2D': {**VALID_SHEET, **real_paths}},
        settings=settings,
    )
    assert len(cfg.sheets) == 1
    assert cfg.sheets[0].mat == "MoS2"
    assert cfg.sheet.mat == "MoS2"

# --- D2b: [sub] match flag + relaxed amorphous-required validator ---

def test_crystalline_substrate_with_match_is_allowed():
    from src.core.config import AFMSimulationConfig, load_settings
    from tests.helpers.hetero import _afm_hetero_raw
    raw = _afm_hetero_raw()                       # non-finite-sheet hetero AFM
    raw["sub"]["amorph"] = "c"
    raw["sub"]["match"] = True
    cfg = AFMSimulationConfig(**raw, settings=load_settings())   # must not raise
    assert cfg.sub.amorph == "c" and cfg.sub.match is True


def test_crystalline_substrate_without_match_still_rejected():
    from src.core.config import AFMSimulationConfig, load_settings
    from tests.helpers.hetero import _afm_hetero_raw
    raw = _afm_hetero_raw()
    raw["sub"]["amorph"] = "c"                    # crystalline, no match flag
    with pytest.raises(ValueError, match="amorphous"):
        AFMSimulationConfig(**raw, settings=load_settings())


# --- B2: [Flake-1]/[Flake-2] fold into flake_sheets ---

def test_flake_sections_fold_into_flake_sheets():
    from src.core.config import SheetOnSheetSimulationConfig, load_settings
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D": {"mat": "h-MoS2", "cif_path": "examples/materials/h-MoS2.cif",
               "pot_path": "examples/potentials/sw/MoS2_wen.sw", "pot_type": "sw",
               "x": 30.0, "y": 30.0, "layers": [1]},
        "Flake": {"shape": "hexagon", "edge_length": 12.0, "corner_radius": 3.0},
        "Flake-1": {"mat": "h-MoS2", "cif_path": "examples/materials/h-MoS2.cif",
                    "pot_path": "examples/potentials/sw/MoS2_wen.sw", "pot_type": "sw",
                    "x": 12.0, "y": 12.0, "layers": [1]},
        "Flake-2": {"mat": "h-WS2", "cif_path": "examples/materials/h-WS2.cif",
                    "pot_path": "examples/potentials/sw/sw_lammps/t-WS2.sw", "pot_type": "sw",
                    "x": 12.0, "y": 12.0, "layers": [1]},
    }
    cfg = SheetOnSheetSimulationConfig(**raw, settings=load_settings())
    assert len(cfg.flake_sheets) == 2
    assert cfg.flake_sheets[0].mat == "h-MoS2" and cfg.flake_sheets[1].mat == "h-WS2"
    assert cfg.flake is not None and cfg.flake.shape == "hexagon"


def test_no_flake_sections_leaves_flake_sheets_empty():
    from src.core.config import SheetOnSheetSimulationConfig, load_settings
    raw = {"general": {"temp": 300, "scan_speed": 1},
           "2D": {"mat": "h-MoS2", "cif_path": "examples/materials/h-MoS2.cif",
                  "pot_path": "examples/potentials/sw/MoS2_wen.sw", "pot_type": "sw",
                  "x": 12.0, "y": 12.0, "layers": [1]}}
    cfg = SheetOnSheetSimulationConfig(**raw, settings=load_settings())
    assert cfg.flake_sheets == []


# --- AFM [flake] must carry its own material (no [2D] fallback) ---

def test_afm_flake_missing_material_fields_is_rejected():
    """An AFM [flake] missing its material fields must fail at config load.

    build_flake_component takes mat/cif/pot straight off [flake], with no [2D]
    fallback (that is a sheet-on-sheet flake-slide feature). Such a config used to
    validate and then die inside build_monolayer on a None cif_path.
    """
    from src.core.config import AFMSimulationConfig, load_settings
    from tests.helpers.hetero import _afm_flake_raw
    raw = _afm_flake_raw()
    del raw["flake"]["cif_path"]
    del raw["flake"]["pot_path"]
    with pytest.raises(ValueError, match=r"cif_path.*pot_path|pot_path.*cif_path"):
        AFMSimulationConfig(**raw, settings=load_settings())


def test_afm_flake_with_material_fields_still_validates():
    """The fully-specified AFM flake (examples/afm_flake_config.ini) is untouched."""
    from src.core.config import AFMSimulationConfig, load_settings
    from tests.helpers.hetero import _afm_flake_raw
    cfg = AFMSimulationConfig(**_afm_flake_raw(), settings=load_settings())
    assert cfg.flake is not None and cfg.flake.mat == "h-MoS2"


def test_sheetonsheet_flake_may_still_omit_its_material():
    """The AFM-only rule must not leak onto the sheet-on-sheet family.

    Its [Flake] legitimately reuses [2D] via
    FlakeSlideSimulation._filled_flake_config.
    """
    from src.core.config import SheetOnSheetSimulationConfig, load_settings
    raw = {"general": {"temp": 300, "scan_speed": 1},
           "2D": {"mat": "h-MoS2", "cif_path": "examples/materials/h-MoS2.cif",
                  "pot_path": "examples/potentials/sw/MoS2_wen.sw", "pot_type": "sw",
                  "x": 30.0, "y": 30.0, "layers": [1]},
           "Flake": {"shape": "hexagon", "edge_length": 5.0}}
    cfg = SheetOnSheetSimulationConfig(**raw, settings=load_settings())
    assert cfg.flake is not None and cfg.flake.mat is None


# --- D3: general.twist_angle + twist tolerances ---

def test_twist_angle_config_defaults_and_range():
    from src.core.config import GeneralConfig
    g = GeneralConfig(temp=300)
    assert g.twist_angle is None                    # inert by default
    # the twist tolerance knobs have documented defaults
    assert g.twist_angle_tol == 2.0
    assert g.twist_strain_tol == 0.02
    assert g.twist_max_supercell == 12
    g2 = GeneralConfig(temp=300, twist_angle=7.34)
    assert abs(g2.twist_angle - 7.34) < 1e-9
    with pytest.raises(ValueError):
        GeneralConfig(temp=300, twist_angle=95.0)   # out of [0,60)
    with pytest.raises(ValueError):
        GeneralConfig(temp=300, twist_angle=-1.0)   # below range


def test_outer_loop_accepts_scan_angle_and_rejects_unknown_values():
    """scan_angle is a fourth valid outer_loop choice; nonsense values still fail."""
    g = GeneralConfig(temp=300, outer_loop='scan_angle')
    assert g.outer_loop == 'scan_angle'
    with pytest.raises(ValidationError):
        GeneralConfig(temp=300, outer_loop='not_a_real_choice')


def test_the_model_does_not_judge_the_angle_split():
    """outer_loop='scan_angle' + scan_angle_force is a conflict only for the
    builders that really split by angle, which this model cannot see. The
    check is routing-aware and lives in src.run.expand.outer_loop_conflict (pinned
    by tests/run/test_run.py::TestAngleSplitWithAForceGate); GeneralConfig must
    keep accepting the combination so the builders that honour the gate
    in-script (indent, hetero, flake) still build."""
    g = GeneralConfig(temp=300, outer_loop='scan_angle',
                      scan_angle=[0.0, 30.0], scan_angle_force=30.0)
    assert g.scan_angle_force == 30.0


def test_damping_ratio_defaults_to_zero():
    from src.core.config import TipConfig
    assert TipConfig.model_fields['damping_ratio'].default == 0.0


def test_optional_channels_defaults_empty():
    from src.core.config import OutputSettings
    assert OutputSettings().optional_channels == []


def test_optional_channels_accepts_comma_separated_string():
    from src.core.config import OutputSettings
    assert OutputSettings(optional_channels='a, b').optional_channels == ['a', 'b']


def test_optional_channels_none_becomes_empty_list():
    from src.core.config import OutputSettings
    assert OutputSettings(optional_channels=None).optional_channels == []


def test_optional_channels_empty_string_becomes_empty_list():
    from src.core.config import OutputSettings
    assert OutputSettings(optional_channels='').optional_channels == []


def test_optional_channels_list_passes_through():
    from src.core.config import OutputSettings
    assert OutputSettings(optional_channels=['raw_pe', 'raw_fz']).optional_channels == \
        ['raw_pe', 'raw_fz']


def _finite(tmp_path, sub_x, sub_y, sheet=(100.0, 100.0)):
    cfg = _make_afm_config(tmp_path, layers=[1])
    raw = cfg.model_dump()
    raw['general']['finite_sheet'] = True
    raw['sub']['x'], raw['sub']['y'] = sub_x, sub_y
    # model_dump() emits the field name 'sheet', but AFMSimulationConfig has no
    # populate_by_name and only accepts its alias '2D' at construction time.
    sheet_dict = raw.pop('sheet')
    sheet_dict['x'], sheet_dict['y'] = sheet
    raw['2D'] = sheet_dict
    return raw


def test_finite_sheet_substrate_smaller_than_sheet_in_y_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match='substrate y'):
        AFMSimulationConfig(**_finite(tmp_path, 160.0, 90.0))


def test_finite_sheet_margin_must_exceed_lj_cutoff_in_x(tmp_path):
    with pytest.raises(ValidationError, match=r'substrate_x - sheet_x'):
        AFMSimulationConfig(**_finite(tmp_path, 105.0, 160.0))


def test_finite_sheet_margin_must_exceed_lj_cutoff_in_y(tmp_path):
    with pytest.raises(ValidationError, match=r'substrate_y - sheet_y'):
        AFMSimulationConfig(**_finite(tmp_path, 160.0, 105.0))


def test_flake_pot_type_left_at_default_dumps_and_reloads():
    """`ComponentConfig.check_pot_type` must accept `None`, matching its
    sibling `resolve_path` validator: `FlakeConfig.pot_type` (unlike
    `ComponentConfig`'s own `pot_type: str`, always required) is
    `Optional[str] = None` for a flake with no material of its own, carved
    from the sheet's -- the common single-material `[Flake]` case.
    `model_dump_json()` serializes an unset `pot_type` as an explicit
    `"pot_type": null`, not an omitted key, and reloading that dump (what
    `src.aiida.export.rebuild_set`/spec §4.2 does with a stored
    `config.json`) must pass an explicit `None` through the validator, not
    crash on it.
    """
    original = FlakeConfig(edge_length=20.0)
    assert original.pot_type is None

    dumped = original.model_dump_json()
    assert '"pot_type":null' in dumped.replace(' ', '')

    reloaded = FlakeConfig.model_validate_json(dumped)
    assert reloaded.pot_type is None
    assert reloaded.edge_length == 20.0


def test_flake_corner_hold_defaults_off():
    """`corner_hold` (follow-up E, 2026-09-22) defaults to `False`: a flake
    slide's flake is deformable and driven by its COM spring unless a config
    explicitly turns the corner tether on. AFM indentation does not read this
    field at all -- it always holds its corners, unconditionally, via a
    separate template block -- so this default only changes flake-on-sheet
    slide behaviour.
    """
    assert FlakeConfig(edge_length=20.0).corner_hold is False
    assert FlakeConfig(edge_length=20.0, corner_hold=True).corner_hold is True


def test_flake_pot_type_invalid_still_raises(monkeypatch):
    """The `None` guard only skips the check for `None` -- an actual bad
    pair style must still be refused."""
    monkeypatch.setattr('src.interfaces.lammps.lammps_pair_styles',
                        lambda: frozenset({'sw'}))
    with pytest.raises(ValidationError, match='not a known LAMMPS pair style'):
        FlakeConfig(edge_length=20.0, pot_type='not-a-real-potential')


def test_aiida_settings_is_five_fields():
    from src.core.config import AiidaSettings

    assert set(AiidaSettings.model_fields) == {
        'enabled', 'profile', 'lammps_code_label', 'computer_label',
        'max_stored_file_mb'}
    defaults = AiidaSettings()
    assert defaults.enabled is False
    assert defaults.profile is None
    assert defaults.lammps_code_label == 'lammps@localhost'
    assert defaults.computer_label == 'localhost'
    assert defaults.max_stored_file_mb == 20


def test_database_settings_is_central_only():
    from src.core.config import DatabaseSettings

    assert set(DatabaseSettings.model_fields) == {
        'host', 'port', 'dbname', 'user', 'password', 'api_key', 'api_url',
        'skip_fraction'}
    import src.core.config as config_mod
    assert not hasattr(config_mod, 'DatabaseProfileSettings')


def test_old_settings_file_with_removed_aiida_and_database_keys_still_loads(
        tmp_path, caplog):
    """Regression (T25 carry-over): a settings.yaml written before both the
    AiidaSettings reduction (Task 25) and the DatabaseSettings flattening
    (Task 31) must still load without error -- the removed keys are unknown
    extras, silently dropped by pydantic. The old nested `database.central`/
    `database.local` shape held real credentials, though, so losing them
    silently is a trap: `load_settings` must warn about that one specifically.
    """
    settings_file = tmp_path / 'settings.yaml'
    settings_file.write_text(
        'aiida:\n'
        '  enabled: true\n'
        '  transport: ssh\n'
        '  hostname: my_hpc\n'
        '  workdir: /scratch/me\n'
        '  username: me\n'
        '  ssh_port: 2222\n'
        '  key_filename: ~/.ssh/id_rsa\n'
        '  hpc_mode: remote\n'
        '  postprocess_code_label: python@my_hpc\n'
        '  postprocess_script_path: /scripts/post.py\n'
        '  create_provenance: true\n'
        '  auto_import_results: false\n'
        'database:\n'
        '  active_profile: central\n'
        '  local:\n'
        '    host: localhost\n'
        '  central:\n'
        '    host: db.example.org\n'
        '    port: 6543\n'
        '    user: alice\n'
        '    password: s3cr3t\n',
        encoding='utf-8')

    with caplog.at_level('WARNING'):
        settings = load_settings(settings_file=settings_file)

    # Loads without error; the removed aiida.* keys are dropped, the five
    # reachable fields keep their defaults since none of them were spelled here.
    assert settings.aiida.enabled is True
    assert not hasattr(settings.aiida, 'transport')

    # The nested database.central block is now silently ignored --
    # DatabaseSettings.host stays at its default, NOT 'db.example.org'.
    assert settings.database.host == ''
    assert any(
        'database.active_profile' in rec.message or 'central' in rec.message
        for rec in caplog.records
    ), "load_settings must warn when it silently drops an old nested database block"
