import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError
from src.core.config import (
    AFMSimulationConfig, 
    SheetOnSheetSimulationConfig,
    TipConfig, 
    SheetConfig, 
    SubstrateConfig,
    GeneralConfig,
    GlobalSettings,
    parse_config,
    load_settings
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
    'cif_path': 'Si.cif', 'thickness': 10.0
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