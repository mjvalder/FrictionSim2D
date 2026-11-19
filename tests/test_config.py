import os
import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError
from src.core.config import (
    AFMSimulationConfig,
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

def test_valid_config_creation(tmp_path, monkeypatch):
    """Test creating a full configuration object."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    
    config = AFMSimulationConfig(
        general=VALID_GENERAL,
        tip=VALID_TIP,
        sub=VALID_SUB,
        **{'2D': VALID_SHEET},
        settings=settings
    )
    assert config.tip.r == 10.0
    assert config.sheet.mat == 'MoS2'

def test_validation_error(tmp_path, monkeypatch):
    """Ensure invalid types raise errors."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    invalid_tip = VALID_TIP.copy()
    invalid_tip['r'] = "not_a_number" # Should fail
    
    with pytest.raises(ValidationError):
        AFMSimulationConfig(
            general=VALID_GENERAL,
            tip=invalid_tip,
            sub=VALID_SUB,
            **{'2D': VALID_SHEET},
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


def test_sheetonsheet_config_accepts_explicit_scan_angle_list(tmp_path, monkeypatch):
    """Sheet-on-sheet config should accept explicit numeric angle lists."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

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


def test_tip_config_allows_explicit_tip_coordinates_with_offsets(tmp_path, monkeypatch) -> None:
    """Tip placement fields should belong to [tip] and allow explicit+offset inputs."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    tip = dict(VALID_TIP)
    tip.update({"tip_x": 10.0, "tip_y": 5.0, "tip_x_offset": 1.0, "tip_y_offset": -0.5})

    cfg = AFMSimulationConfig(
        general=VALID_GENERAL,
        tip=tip,
        sub=VALID_SUB,
        **{'2D': VALID_SHEET},
        settings=settings,
    )
    assert cfg.tip.tip_x == 10.0
    assert cfg.tip.tip_y == 5.0
    assert cfg.tip.tip_x_offset == 1.0
    assert cfg.tip.tip_y_offset == -0.5


def test_finite_sheet_requires_substrate_xy(tmp_path, monkeypatch):
    """Finite-sheet AFM should require explicit substrate x/y in [sub]."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    with pytest.raises(ValidationError, match="requires substrate x and y"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": True},
            tip=VALID_TIP,
            sub={**VALID_SUB, "amorph": "c"},
            **{'2D': VALID_SHEET},
            settings=settings,
        )


def test_finite_sheet_rejects_substrate_smaller_than_sheet(tmp_path, monkeypatch):
    """Finite-sheet AFM should reject substrate sizes smaller than the sheet."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    bad_sub = dict(VALID_SUB)
    bad_sub.update({"x": 40.0, "y": 60.0, "amorph": "c"})
    with pytest.raises(ValidationError, match="cannot be smaller than sheet x"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": True},
            tip=VALID_TIP,
            sub=bad_sub,
            **{'2D': VALID_SHEET},
            settings=settings,
        )


def test_finite_sheet_rejects_margin_not_above_lj_cutoff(tmp_path, monkeypatch):
    """Finite-sheet AFM should enforce strict substrate-sheet delta > LJ cutoff."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings().model_copy(deep=True)
    settings.potential.lj_cutoff = 11.0
    bad_sub = dict(VALID_SUB)
    bad_sub.update({"x": 61.0, "y": 70.0, "amorph": "c"})
    # sheet x is 50, so x-delta is exactly 11.0 -> invalid (must be > cutoff)
    with pytest.raises(ValidationError, match="greater than LJ cutoff"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": True},
            tip=VALID_TIP,
            sub=bad_sub,
            **{'2D': VALID_SHEET},
            settings=settings,
        )


def test_non_finite_sheet_rejects_crystalline_substrate(tmp_path, monkeypatch):
    """Non-finite-sheet AFM should allow only amorphous substrates."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    with pytest.raises(ValidationError, match="requires an amorphous substrate"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": False},
            tip=VALID_TIP,
            sub={**VALID_SUB, "amorph": "c"},
            **{'2D': VALID_SHEET},
            settings=settings,
        )


# ---------------------------------------------------------------------------
# PES eval-mode and optional substrate tests
# ---------------------------------------------------------------------------

def _afm_dict(tmp_path, include_sub=True, sub_amorph="a", finite_sheet=False, pes=None):
    pot = tmp_path / "d.sw"; pot.write_text("# pot", encoding="utf-8")
    cif = tmp_path / "d.cif"; cif.write_text("# cif", encoding="utf-8")
    d = {
        "general": {"temp": 300.0, "force": [2.0], "scan_speed": 2.0, "finite_sheet": finite_sheet},
        "2D": {"mat": "h-MoS2", "pot_type": "sw", "pot_path": str(pot),
               "cif_path": str(cif), "x": 60.0, "y": 60.0, "layers": [4]},
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
    monkeypatch.setattr('src.core.config._global_settings_path', lambda: tmp_path / 'nonexistent.yaml')
    settings = load_settings()
    assert isinstance(settings, GlobalSettings)
    # Default timestep is 0.001 as defined in SimulationSettings
    assert settings.simulation.timestep == 0.001


def test_global_settings_used_when_no_explicit_file(tmp_path, monkeypatch):
    """load_settings picks up the global settings file when no explicit file is given."""
    global_path = tmp_path / 'global' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)
    monkeypatch.setattr('src.core.config._global_settings_path', lambda: global_path)

    settings = load_settings()
    assert settings.simulation.timestep == 0.005


def test_explicit_file_takes_precedence_over_global(tmp_path, monkeypatch):
    """Explicit settings_file argument overrides the global settings file."""
    global_path = tmp_path / 'global' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)

    explicit_path = tmp_path / 'explicit' / 'settings.yaml'
    _write_settings(explicit_path, timestep=0.002)

    monkeypatch.setattr('src.core.config._global_settings_path', lambda: global_path)

    settings = load_settings(settings_file=explicit_path)
    assert settings.simulation.timestep == 0.002


def test_env_var_is_not_honoured(tmp_path, monkeypatch):
    """FRICTIONSIM2D_SETTINGS_FILE env var is no longer part of the precedence chain."""
    env_path = tmp_path / 'env' / 'settings.yaml'
    _write_settings(env_path, timestep=0.007)

    # Set the env var but point the global path to a non-existent file
    monkeypatch.setenv('FRICTIONSIM2D_SETTINGS_FILE', str(env_path))
    monkeypatch.setattr('src.core.config._global_settings_path', lambda: tmp_path / 'nonexistent.yaml')

    # Should fall back to hardcoded defaults, NOT pick up the env var file
    settings = load_settings()
    assert settings.simulation.timestep == 0.001


def test_settings_origin_none_when_no_file(tmp_path, monkeypatch):
    """settings_origin returns None when no settings file exists."""
    monkeypatch.setattr('src.core.config._global_settings_path', lambda: tmp_path / 'nonexistent.yaml')
    assert settings_origin() is None


def test_settings_origin_returns_global_path(tmp_path, monkeypatch):
    """settings_origin returns the global path when it exists and no explicit file given."""
    global_path = tmp_path / 'global' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)
    monkeypatch.setattr('src.core.config._global_settings_path', lambda: global_path)

    assert settings_origin() == global_path


def test_settings_origin_returns_explicit_path(tmp_path, monkeypatch):
    """settings_origin returns the explicit path when one is provided."""
    global_path = tmp_path / 'global' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)

    explicit_path = tmp_path / 'explicit' / 'settings.yaml'
    _write_settings(explicit_path, timestep=0.002)

    monkeypatch.setattr('src.core.config._global_settings_path', lambda: global_path)

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


def test_two_material_sections_fold_into_sheets_list(tmp_path, monkeypatch):
    """[2D-1]/[2D-2] sections fold into cfg.sheets in numeric order."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

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


def test_single_2d_section_still_builds_one_element_sheets_list(tmp_path, monkeypatch):
    """A lone [2D] section (no [2D-N]) still builds and yields len(sheets) == 1."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    cfg = SheetOnSheetSimulationConfig(
        general=VALID_GENERAL,
        **{'2D': VALID_SHEET},
        settings=settings,
    )
    assert len(cfg.sheets) == 1
    assert cfg.sheets[0].mat == "MoS2"
    assert cfg.sheet.mat == "MoS2"

# --- D2b: [sub] match flag + relaxed amorphous-required validator ---

def test_crystalline_substrate_with_match_is_allowed():
    from src.core.config import AFMSimulationConfig, load_settings
    from tests.test_hetero_afm import _afm_hetero_raw
    raw = _afm_hetero_raw()                       # non-finite-sheet hetero AFM
    raw["sub"]["amorph"] = "c"
    raw["sub"]["match"] = True
    cfg = AFMSimulationConfig(**raw, settings=load_settings())   # must not raise
    assert cfg.sub.amorph == "c" and cfg.sub.match is True


def test_crystalline_substrate_without_match_still_rejected():
    from src.core.config import AFMSimulationConfig, load_settings
    from tests.test_hetero_afm import _afm_hetero_raw
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
    check is routing-aware and lives in core.run.outer_loop_conflict (pinned
    by tests/test_run.py::TestAngleSplitWithAForceGate); GeneralConfig must
    keep accepting the combination so the builders that honour the gate
    in-script (indent, hetero, flake) still build."""
    g = GeneralConfig(temp=300, outer_loop='scan_angle',
                      scan_angle=[0.0, 30.0], scan_angle_force=30.0)
    assert g.scan_angle_force == 30.0


def test_damping_ratio_defaults_to_zero():
    from src.core.config import TipConfig
    assert TipConfig.model_fields['damping_ratio'].default == 0.0


def test_d3_dispersion_settings_defaults():
    """The D3 knobs (spec §1.2) live on PotentialSettings with these defaults."""
    potential = GlobalSettings().potential
    assert potential.d3_functional == 'pbe'
    assert potential.d3_damping == 'bj'
    assert potential.d3_cutoff == 16.0
    assert potential.d3_cn_cutoff == 8.0


def test_d3_damping_rejects_non_canonical_token():
    """d3_damping is the canonical name, enum-validated: 'd3bj' is not one."""
    from src.core.config import PotentialSettings
    PotentialSettings(d3_damping='zerom')  # a valid member
    with pytest.raises(ValidationError):
        PotentialSettings(d3_damping='d3bj')


# ---------------------------------------------------------------------------
# MLIP config surface (v4 spec §5)
# ---------------------------------------------------------------------------

@pytest.fixture
def sheet_kwargs(tmp_path):
    """Baseline SheetConfig kwargs whose paths exist (so no resolver runs)."""
    cif = tmp_path / "h-MoS2.cif"
    cif.write_text("# dummy cif\n", encoding="utf-8")
    return {
        'mat': 'h-MoS2',
        'cif_path': str(cif),
        'x': 50.0,
        'y': 50.0,
        'layers': [3],
    }


def _model(tmp_path, name):
    """Write a fake (binary) MLIP model file and return its path string."""
    p = tmp_path / name
    p.write_bytes(b"\x80\x02not-utf8\xff")
    return str(p)


def test_mlip_keys_default_to_classical_behaviour(sheet_kwargs, tmp_path):
    """An untouched (SW) sheet carries the inert defaults."""
    pot = tmp_path / "MoS2.sw"
    pot.write_text("# sw\n", encoding="utf-8")
    sheet = SheetConfig(**{**sheet_kwargs, 'pot_type': 'sw', 'pot_path': str(pot)})
    assert sheet.dispersion == 'none'
    assert not hasattr(sheet, 'mlip_interlayer')


@pytest.mark.parametrize('value', ['lj', 'native', ''])
def test_mlip_interlayer_is_rejected_loudly(sheet_kwargs, tmp_path, value):
    """The key is GONE (v4 §5) and an old config must not be silently ignored.

    ``mlip_interlayer`` only ever chose between a partition ('lj') and full
    coverage ('native'); the partition is unavailable for both frameworks
    (v4 §1 E2/E3), so the key carries no information. Pydantic ignores unknown
    keys by default, which for this one would silently change the physics of an
    existing config -- hence the explicit rejection.
    """
    with pytest.raises(ValidationError, match=r"2026-08-11-mlip-post-m0-design"):
        SheetConfig(**{**sheet_kwargs, 'pot_type': 'deepmd',
                       'pot_path': _model(tmp_path, 'm.pth'), 'lat_c': 6.15,
                       'mlip_interlayer': value})


def test_mlip_interlayer_is_rejected_on_a_classical_sheet_too(sheet_kwargs, tmp_path):
    pot = tmp_path / "MoS2.sw"
    pot.write_text("# sw\n", encoding="utf-8")
    with pytest.raises(ValidationError, match=r"2026-08-11-mlip-post-m0-design"):
        SheetConfig(**{**sheet_kwargs, 'pot_type': 'sw', 'pot_path': str(pot),
                       'mlip_interlayer': 'native'})


def test_dispersion_requires_an_mlip_pot_type(sheet_kwargs, tmp_path):
    pot = tmp_path / "MoS2.sw"
    pot.write_text("# sw\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="dispersion"):
        SheetConfig(**{**sheet_kwargs, 'pot_type': 'sw', 'pot_path': str(pot),
                       'dispersion': 'd3'})


def test_mlip_requires_lat_c(sheet_kwargs, tmp_path):
    with pytest.raises(ValidationError, match="lat_c"):
        SheetConfig(**{**sheet_kwargs, 'pot_type': 'deepmd',
                       'pot_path': _model(tmp_path, 'm.pth')})


def test_mlip_requires_a_framework_matching_model_extension(sheet_kwargs, tmp_path):
    """A MACE model handed to deepmd (or vice versa) is a hard error."""
    with pytest.raises(ValidationError, match="extension"):
        SheetConfig(**{**sheet_kwargs, 'pot_type': 'deepmd',
                       'pot_path': _model(tmp_path, 'm.json'),
                       'lat_c': 6.15})
    with pytest.raises(ValidationError, match="extension"):
        SheetConfig(**{**sheet_kwargs, 'pot_type': 'mace',
                       'pot_path': _model(tmp_path, 'm.pb'),
                       'lat_c': 6.15})


def test_valid_mlip_sheets_are_accepted(sheet_kwargs, tmp_path):
    """Full-cell base and full-cell + D3, for both frameworks (v4 §3).

    ``base`` is now pinned to a SINGLE layer. A multilayer MLIP stack with
    ``dispersion = none`` is rejected since the bilayer benchmark measured the
    bare model as having no interlayer binding at all -- see
    ``TestMultilayerMLIPRequiresD3``. One layer has no interlayer gap, so
    base-only remains legitimate there.
    """
    base = SheetConfig(**{**sheet_kwargs, 'pot_type': 'deepmd',
                          'pot_path': _model(tmp_path, 'm.pth'),
                          'layers': [1], 'lat_c': 6.15})
    assert base.dispersion == 'none'

    full_d3 = SheetConfig(**{**sheet_kwargs, 'pot_type': 'deepmd',
                             'pot_path': _model(tmp_path, 'm.pth'),
                             'dispersion': 'd3', 'lat_c': 6.15})
    assert full_d3.dispersion == 'd3'

    mace = SheetConfig(**{**sheet_kwargs, 'pot_type': 'mace',
                          'pot_path': _model(tmp_path, 'm.json'),
                          'layers': [1], 'lat_c': 6.15})
    assert mace.dispersion == 'none'

    mace_d3 = SheetConfig(**{**sheet_kwargs, 'pot_type': 'mace',
                             'pot_path': _model(tmp_path, 'm.json'),
                             'dispersion': 'd3', 'lat_c': 6.15})
    assert mace_d3.dispersion == 'd3'


def test_ini_round_trips_the_mlip_keys(tmp_path):
    """``dispersion``/``lat_c`` typed into a [2D] section reach SheetConfig."""
    model = _model(tmp_path, 'dpa.pth')
    cif = tmp_path / "h-MoS2.cif"
    cif.write_text("# dummy cif\n", encoding="utf-8")
    ini = tmp_path / "sheet_mlip.ini"
    ini.write_text(
        "[general]\n"
        "temp = 300\n"
        "pressure = 1.0\n"
        "\n"
        "[2D]\n"
        "mat = h-MoS2\n"
        "pot_type = deepmd\n"
        f"pot_path = {model}\n"
        f"cif_path = {cif}\n"
        "x = 50\n"
        "y = 50\n"
        "layers = [3]\n"
        "lat_c = 6.15\n"
        "dispersion = d3\n",
        encoding="utf-8",
    )

    parsed = parse_config(ini)
    assert parsed['2D']['dispersion'] == 'd3'

    config = SheetOnSheetSimulationConfig(**{**parsed, 'settings': load_settings()})
    assert config.sheet.dispersion == 'd3'
    assert config.sheet.lat_c == 6.15


def test_ini_carrying_the_removed_key_fails_loudly(tmp_path):
    """A pre-v4 .ini must not build silently with different physics."""
    model = _model(tmp_path, 'dpa.pth')
    cif = tmp_path / "h-MoS2.cif"
    cif.write_text("# dummy cif\n", encoding="utf-8")
    ini = tmp_path / "old_sheet_mlip.ini"
    ini.write_text(
        "[general]\n"
        "temp = 300\n"
        "pressure = 1.0\n"
        "\n"
        "[2D]\n"
        "mat = h-MoS2\n"
        "pot_type = deepmd\n"
        f"pot_path = {model}\n"
        f"cif_path = {cif}\n"
        "x = 50\n"
        "y = 50\n"
        "layers = [3]\n"
        "lat_c = 6.15\n"
        "mlip_interlayer = lj\n",
        encoding="utf-8",
    )
    parsed = parse_config(ini)
    with pytest.raises(ValidationError, match=r"2026-08-11-mlip-post-m0-design"):
        SheetOnSheetSimulationConfig(**{**parsed, 'settings': load_settings()})


class TestMultilayerMLIPRequiresD3:
    """A multilayer MLIP stack without D3 is unbound -- measured, not argued.

    The bilayer benchmark (2026-08-11, DPA-3.1-3M/Alex2D on the backported
    binary) found the bare foundation model has **no interlayer binding at
    all**: its binding curve decreases monotonically to the edge of the scan,
    binding energy 5.7e-7 J/m^2. Adding D3 produces a real minimum at 6.112 A
    (experiment 6.15) with 0.265 J/m^2 (literature 0.25-0.30).

    So `dispersion = none` on a multilayer MLIP sheet yields layers that drift
    apart, and every interlayer friction/adhesion number from such a run is
    meaningless. It has to fail at config time, not silently produce a campaign
    that looks fine.
    """

    @staticmethod
    def _kwargs(tmp_path, **over):
        cif = tmp_path / 'MoS2.cif'
        cif.write_text('#', encoding='utf-8')
        model = tmp_path / 'dpa3_alex2d.pth'
        model.write_bytes(b'\x00')
        base = dict(mat='h-MoS2', pot_type='deepmd', pot_path=str(model),
                    cif_path=str(cif), x=30.0, y=30.0, layers=[3], lat_c=6.112)
        base.update(over)
        return base

    def test_multilayer_mlip_without_d3_is_rejected(self, tmp_path):
        from src.core.config import SheetConfig
        with pytest.raises(ValidationError, match='(?i)unbound|dispersion'):
            SheetConfig(**self._kwargs(tmp_path, dispersion='none'))

    def test_multilayer_mlip_with_d3_is_accepted(self, tmp_path):
        from src.core.config import SheetConfig
        cfg = SheetConfig(**self._kwargs(tmp_path, dispersion='d3'))
        assert cfg.dispersion == 'd3'

    def test_mace_without_d3_is_allowed_because_it_binds(self, tmp_path):
        """The guard is framework-aware, not a blanket MLIP rule.

        MACE-MPA-0 was measured (bench 2026-08-24) to bind a bare bilayer:
        a real minimum at 6.717 A, 0.068 J/m^2. It under-binds versus the
        0.25-0.30 literature range, and D3 then over-binds it (0.474 J/m^2),
        so the no-D3 arm is a legitimate control rather than a broken run.
        Blocking it would have generalised DPA-3's failure to a model that
        does not share it.
        """
        from src.core.config import SheetConfig
        model = tmp_path / 'mace.json'
        model.write_bytes(b'\x00')
        cfg = SheetConfig(**self._kwargs(
            tmp_path, pot_type='mace', pot_path=str(model), dispersion='none'))
        assert cfg.dispersion == 'none'

    def test_allow_unbound_permits_a_deliberate_deepmd_control(self, tmp_path):
        """`allow_unbound` exists so the no-D3 DeePMD control arm is runnable.

        Demonstrating that the stack comes apart IS the scientific point of
        that arm, so the guard must be opt-out-able -- while still firing by
        default so an accidental config cannot waste a campaign.
        """
        from src.core.config import SheetConfig
        cfg = SheetConfig(**self._kwargs(
            tmp_path, dispersion='none', allow_unbound=True))
        assert cfg.allow_unbound is True

    def test_allow_unbound_defaults_to_false(self, tmp_path):
        from src.core.config import SheetConfig
        cfg = SheetConfig(**self._kwargs(tmp_path, dispersion='d3'))
        assert cfg.allow_unbound is False

    def test_single_layer_mlip_without_d3_is_fine(self, tmp_path):
        """One layer has no interlayer gap to get wrong."""
        from src.core.config import SheetConfig
        cfg = SheetConfig(**self._kwargs(tmp_path, layers=[1], dispersion='none'))
        assert cfg.dispersion == 'none'

    def test_classical_multilayer_is_untouched(self, tmp_path):
        """SW sheets are not affected -- their LJ interlayer term is real."""
        from src.core.config import SheetConfig
        sw = tmp_path / 'MoS2.sw'
        sw.write_text('# sw\n', encoding='utf-8')
        cfg = SheetConfig(**self._kwargs(tmp_path, pot_type='sw',
                                         pot_path=str(sw), lat_c=None))
        assert cfg.pot_type == 'sw'
