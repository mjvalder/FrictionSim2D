import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError
from src.core.config import (
    AFMSimulationConfig, 
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

def test_valid_config_creation():
    """Test creating a full configuration object."""
    settings = load_settings()
    
    config = AFMSimulationConfig(
        general=VALID_GENERAL,
        tip=VALID_TIP,
        sub=VALID_SUB,
        sheet=VALID_SHEET, # Maps to '2D' alias
        settings=settings
    )
    assert config.tip.r == 10.0
    assert config.sheet.mat == 'MoS2'

def test_validation_error():
    """Ensure invalid types raise errors."""
    settings = load_settings()
    invalid_tip = VALID_TIP.copy()
    invalid_tip['r'] = "not_a_number" # Should fail
    
    with pytest.raises(ValidationError):
        AFMSimulationConfig(
            general=VALID_GENERAL,
            tip=invalid_tip,
            sub=VALID_SUB,
            sheet=VALID_SHEET,
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